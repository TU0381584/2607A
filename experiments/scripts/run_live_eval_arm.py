#!/usr/bin/env python3
"""Live-evaluation orchestrator for ONE arm, ONE seed ("rep"): wraps the
frozen qoe_oran_framework/xapp/saclb_xapp.py (learned arms) or this
campaign's experiments/scripts/run_baseline_static.py (baseline arm) in
small episode BATCHES, health-checking the rig before every batch and
doing a full stack restart if unhealthy -- per the finding in
CAMPAIGN_LOG.md that this rig cannot be assumed to sustain a live
gNB+3-UE session for many consecutive hours unattended (UE1/embb hit an
RLC max-RETX failure 3 times within ~1 hour of cumulative uptime during
Phase 0/1 work), and per the user's explicit decision (short episodes +
periodic health-checked restarts) on how to handle that.

Does NOT modify any frozen script -- calls them as subprocesses, exactly
as a human operator would from the shell, with --episodes set to the
batch size and --omega-jsonl set to the SAME path every batch (OmegaLogger
opens in append mode, see mc_runner.py's own comment on this) so one
continuous JSONL trace accumulates across batches.

IMPORTANT, honestly documented rather than glossed over: each batch is a
FRESH process invocation, so each batch's `--seed` reseeds
RANEnv/np.random/random from scratch (see mc_runner.set_seeds) -- the
synthetic-arrival RNG stream does NOT continue smoothly across a restart
the way it would within one continuous run_single() call. To keep this
honest and reproducible rather than silently repeating the same sequence
every batch, EACH BATCH gets its own DETERMINISTIC seed
(base_seed * 1000 + batch_index), logged in batch_manifest.jsonl. The raw
omega log's per-batch `run_id` and each row's wall-clock `timestamp_s`
(global_step, which itself resets to small values each batch) are
therefore NOT a globally-continuous episode/step counter across the whole
rep -- Phase 4 analysis scripts must re-derive global episode ordering
from batch_manifest.jsonl (batch_index, wall-clock start time) rather than
trusting the raw `episode`/`step` fields to already be rep-global. This is
a direct, load-bearing consequence of the short-episodes-plus-restarts
mitigation and is recorded here so it is never silently assumed away.

Usage:
    python3 experiments/scripts/run_live_eval_arm.py \
        --arm dqn_sla --algorithm dqn --reward-mode sla \
        --config experiments/configs/saclb_campaign.yaml \
        --checkpoint <path/to/checkpoint.pt> \
        --episodes-total 50 --batch-size 2 --seed 256 \
        --out-dir experiments/results/live
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

FRAMEWORK_DIR = "/home/kmanojp/oranslice_rig/framework"
SCRIPTS_DIR = "/home/kmanojp/oranslice_rig/experiments/scripts"
HEALTH_CHECK = f"{SCRIPTS_DIR}/health_check.sh"
RESTART_STACK = f"{SCRIPTS_DIR}/restart_ran_stack.sh"
BASELINE_SCRIPT = f"{SCRIPTS_DIR}/run_baseline_static.py"
XAPP_SCRIPT = f"{FRAMEWORK_DIR}/qoe_oran_framework/xapp/saclb_xapp.py"


def run(cmd, **kwargs):
    print(f"[orchestrator] $ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, **kwargs)


def ensure_healthy(max_restarts: int = 2) -> bool:
    for attempt in range(max_restarts + 1):
        result = run(["bash", HEALTH_CHECK])
        if result.returncode == 0:
            return True
        if attempt >= max_restarts:
            return False
        print(f"[orchestrator] health check failed, restarting stack (attempt {attempt + 1})", file=sys.stderr)
        restart = run(["bash", RESTART_STACK])
        if restart.returncode != 0:
            print("[orchestrator] restart_ran_stack.sh itself failed", file=sys.stderr)
            return False
        time.sleep(5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="e.g. baseline, dqn_sla, a2c_sla, dqn_qoe, a2c_qoe")
    ap.add_argument("--algorithm", required=True, choices=["dqn", "a2c", "rainbow", "lb_only", "baseline_static"])
    ap.add_argument("--reward-mode", choices=["sla", "qoe"], default="sla")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="", help="required unless --algorithm baseline_static")
    ap.add_argument("--gnb-id", default="gnb-0")
    ap.add_argument("--episodes-total", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=2, help="episodes per health-checked batch")
    ap.add_argument("--seed", type=int, required=True, help="this rep's base seed")
    ap.add_argument("--out-dir", default="/home/kmanojp/oranslice_rig/experiments/results/live")
    args = ap.parse_args()

    if args.algorithm != "baseline_static" and not args.checkpoint:
        ap.error("--checkpoint required unless --algorithm baseline_static")

    rep_dir = Path(args.out_dir) / args.arm / args.reward_mode / f"rep_seed{args.seed}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    omega_path = rep_dir / "omega_log.jsonl"
    manifest_path = rep_dir / "batch_manifest.jsonl"

    episodes_done = 0
    batch_idx = 0
    while episodes_done < args.episodes_total:
        batch_episodes = min(args.batch_size, args.episodes_total - episodes_done)
        batch_seed = args.seed * 1000 + batch_idx

        print(f"[orchestrator] === {args.arm} rep_seed{args.seed} batch {batch_idx} "
              f"({batch_episodes} episodes, batch_seed={batch_seed}, "
              f"{episodes_done}/{args.episodes_total} done so far) ===", file=sys.stderr)

        if not ensure_healthy():
            print("[orchestrator] FATAL: rig unhealthy after max restart attempts, aborting rep", file=sys.stderr)
            return 1

        run_id = f"{args.arm}_{args.reward_mode}_seed{args.seed}_batch{batch_idx}"
        t0 = time.time()

        if args.algorithm == "baseline_static":
            cmd = [
                "python3", BASELINE_SCRIPT,
                "--config", args.config, "--gnb-id", args.gnb_id,
                "--episodes", str(batch_episodes), "--seed", str(batch_seed),
                "--run-id", run_id, "--omega-jsonl", str(omega_path),
                "--reward-mode", args.reward_mode,
            ]
        else:
            cmd = [
                "python3", XAPP_SCRIPT,
                "--config", args.config, "--algorithm", args.algorithm,
                "--checkpoint", args.checkpoint, "--gnb-id", args.gnb_id,
                "--episodes", str(batch_episodes), "--seed", str(batch_seed),
                "--run-id", run_id, "--omega-jsonl", str(omega_path),
                "--reward-mode", args.reward_mode,
            ]

        result = run(cmd, cwd=FRAMEWORK_DIR)
        elapsed = time.time() - t0

        manifest_row = {
            "batch_index": batch_idx, "run_id": run_id, "batch_seed": batch_seed,
            "episodes_requested": batch_episodes, "wall_clock_start": t0, "elapsed_s": elapsed,
            "returncode": result.returncode,
        }
        with manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(manifest_row) + "\n")

        if result.returncode != 0:
            print(f"[orchestrator] batch {batch_idx} FAILED (returncode={result.returncode}) -- "
                  "treating as an unhealthy rig, restarting and retrying this batch", file=sys.stderr)
            if not ensure_healthy():
                print("[orchestrator] FATAL: could not recover after batch failure, aborting rep", file=sys.stderr)
                return 1
            continue  # retry same batch_idx, same episodes_done (not advanced)

        episodes_done += batch_episodes
        batch_idx += 1

    print(f"[orchestrator] DONE: {episodes_done}/{args.episodes_total} episodes completed for "
          f"{args.arm} rep_seed{args.seed}. omega log: {omega_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
