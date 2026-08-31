#!/usr/bin/env python3
"""M36 offline half: log the seed-900 single-agent-DQN policy's own
congestion_level state feature at every admission decision, offline,
across both demand levers this project's own prior recalibration work
identified (docs/PAPER5_M8_live_anchor.md, "Recalibrating the Simulator
to Match Live Congestion"): ArrivalConfig.synthetic_arrivals_per_step
(shown there to have NO effect on decisions -- re-confirmed here, not
assumed) and ClosedLoopKpmSource.mean_offered_ratio (the lever that
actually drives simulated congestion).

Reuses, does not reimplement: mc_runner.build_policy/run_single (the
same frozen harness every offline campaign in this project uses),
state_vector_probe.wrap_policy_for_state_logging (the same non-invasive
wrapper docs/PAPER5_M8_live_anchor.md's own state-vector capture used),
ClosedLoopKpmSource (frozen, unmodified).

congestion_level lives at indices 1, 4, 7 of the wrapper's logged 13-dim
state (encode_full_request_state's own layout: [urllc,embb,mmtc] x
[prb_used_ratio, congestion_level, queue_len_norm], then
slice_onehot(3), gnb_onehot(1)) -- one value per slice per decision;
this script reports the pooled distribution across slices, matching how
docs/PAPER5_M8_live_anchor.md's own live/offline ranges were reported.

Usage:
    python3 experiments/scripts/m36_offline_congestion_sweep.py \
        --checkpoint experiments/results/m2_campaign/single_agent_dqn/seed900/train/dqn/offline_train/rep_0/checkpoint.pt \
        --out experiments/results/m36_congestion_ranges.csv
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_single  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402
from state_vector_probe import wrap_policy_for_state_logging  # noqa: E402

CONGESTION_IDX = [1, 4, 7]  # urllc, embb, mmtc congestion_level within the 13-dim state
DEFAULT_CONFIG = "/home/kmanojp/oranslice_rig/framework/qoe_oran_framework/configs/saclb_live.yaml"


def run_one_condition(cfg, checkpoint, mean_offered_ratio, synthetic_arrivals_per_step, n_episodes, seed):
    cfg.arrivals.synthetic_arrivals_per_step = synthetic_arrivals_per_step
    sd_for_slice = {slice_id: spec.sd for slice_id, spec in cfg.slice_by_id.items()}
    kpm = ClosedLoopKpmSource(
        seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id), B=cfg.B,
        mean_offered_ratio={s: mean_offered_ratio for s in cfg.slice_by_id},
        sd_for_slice=sd_for_slice,
    )
    env = RANEnv(cfg, kpm, seed=seed, reward_mode="qoe")
    policy = build_policy("dqn", cfg)
    policy.load_checkpoint(checkpoint)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as state_tmp, \
         tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as omega_tmp:
        state_path, omega_path = state_tmp.name, omega_tmp.name

    state_fh = wrap_policy_for_state_logging(policy, state_path)
    try:
        with OmegaLogger(omega_path) as omega:
            run_single(
                env, policy, "dqn", omega, n_episodes, seed,
                run_id=f"m36_offline_ratio{mean_offered_ratio}_arr{synthetic_arrivals_per_step}",
                mode="offline_congestion_probe", training=False, cfg=cfg,
            )
    finally:
        state_fh.close()
        env.close()

    congestion_vals = []
    total_blocks, mmtc_blocks = 0, 0
    with open(state_path) as fh:
        for line in fh:
            row = json.loads(line)
            state = row["state"]
            for idx in CONGESTION_IDX:
                congestion_vals.append(state[idx])
    Path(state_path).unlink(missing_ok=True)
    Path(omega_path).unlink(missing_ok=True)
    return congestion_vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--out", default="experiments/results/m36_congestion_ranges.csv")
    args = ap.parse_args()

    mean_offered_ratios = [round(x, 2) for x in np.arange(0.1, 1.01, 0.1)]
    arrival_levels = [2, 4, 8]  # 2 = config default; 4/8 = the "doubled/quadrupled" levers M8 already tried

    rows = []
    print(f"[m36-offline] sweeping {len(mean_offered_ratios)} mean_offered_ratio levels "
          f"x {len(arrival_levels)} synthetic_arrivals_per_step levels, {args.episodes} episodes each")
    for arrivals in arrival_levels:
        for ratio in mean_offered_ratios:
            cfg = load_saclb_config(args.config)
            vals = run_one_condition(cfg, args.checkpoint, ratio, arrivals, args.episodes, args.seed)
            arr = np.asarray(vals)
            rows.append({
                "source": "offline", "lever_arrivals_per_step": arrivals, "lever_mean_offered_ratio": ratio,
                "n_decisions": len(arr), "congestion_mean": float(arr.mean()) if len(arr) else float("nan"),
                "congestion_min": float(arr.min()) if len(arr) else float("nan"),
                "congestion_p50": float(np.percentile(arr, 50)) if len(arr) else float("nan"),
                "congestion_p90": float(np.percentile(arr, 90)) if len(arr) else float("nan"),
                "congestion_p99": float(np.percentile(arr, 99)) if len(arr) else float("nan"),
                "congestion_max": float(arr.max()) if len(arr) else float("nan"),
            })
            print(f"  arrivals={arrivals} ratio={ratio:.1f}: congestion mean={rows[-1]['congestion_mean']:.4f} "
                  f"range=[{rows[-1]['congestion_min']:.4f}, {rows[-1]['congestion_max']:.4f}] n={len(arr)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[m36-offline] wrote {out_path} ({len(rows)} rows)")

    all_congestion = [r["congestion_mean"] for r in rows]
    arrivals_effect = {a: [r["congestion_mean"] for r in rows if r["lever_arrivals_per_step"] == a] for a in arrival_levels}
    print(f"\n[m36-offline] overall offline congestion range across every demand lever: "
          f"[{min(r['congestion_min'] for r in rows):.4f}, {max(r['congestion_max'] for r in rows):.4f}]")
    print("[m36-offline] synthetic_arrivals_per_step effect on congestion (mean across ratio levels, per arrivals level):")
    for a, vals in arrivals_effect.items():
        print(f"  arrivals={a}: mean={np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
