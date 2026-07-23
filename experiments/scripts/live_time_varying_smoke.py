#!/usr/bin/env python3
"""Minimal LIVE smoke test of within-episode demand variation: 1 seed x 2
episodes x 2 arms (dqn_sla, a2c_qoe), phase-timed real iperf3 traffic
(phase_traffic_control.py) synced by live step index to the same 3-phase
schedule as the offline probe (phase_varying_kpm_source.py) -- this is
the live-rig companion data point to that offline-only probe, explicitly
small in scope (per-user agreement: minimal smoke test, not a full
campaign). Uses saclb_campaign.yaml (the live-calibrated config, cap=12/4/3),
same checkpoints already evaluated in the main live campaign
(experiments/results/live_campaign) -- frozen weights, no training.

Usage (run from repo root, rig already up and healthy):
    source venv/bin/activate
    cd framework
    python3 ../experiments/scripts/live_time_varying_smoke.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from phase_traffic_control import phase_for_step, reset_phase_clock, set_phase_for_step  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.live_kpm_source import LiveKpmSource  # noqa: E402
from qoe_oran_framework.mc_runner import _select_actions, build_policy  # noqa: E402

CONFIG_PATH = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign.yaml"
CKPT_ROOT = "/home/kmanojp/oranslice_rig/experiments/results/offline"
SEED = 950
EPISODES = 2
ARMS = {"dqn_sla": ("dqn", "sla"), "a2c_qoe": ("a2c", "qoe")}
OUT_DIR = Path("/home/kmanojp/oranslice_rig/experiments/results/live_time_varying_smoke")


def run_arm(arm: str, algo: str, reward_mode: str) -> dict:
    cfg = load_saclb_config(CONFIG_PATH)
    by_phase = {name: {s: [] for s in cfg.slice_by_id} for name in ("low", "high", "medium")}

    kpm = LiveKpmSource(gnb_id=cfg.gnb_ids[0])
    env = RANEnv(cfg, kpm, seed=SEED, reward_mode=reward_mode)
    policy = build_policy(algo, cfg)
    ckpt = Path(CKPT_ROOT) / reward_mode / f"seed256" / algo / "offline_closed_loop" / "rep_0" / "checkpoint.pt"
    policy.load_checkpoint(str(ckpt))

    step_log = []
    try:
        for ep in range(1, EPISODES + 1):
            reset_phase_clock()
            set_phase_for_step(1)  # apply "low" phase BEFORE reset()'s first poll
            obs = env.reset()
            done = False
            step_idx = 0
            while not done:
                step_idx += 1
                step_start = time.monotonic()
                if step_idx > 1:
                    set_phase_for_step(step_idx)
                pending = env.pending_requests()
                cluster_state = env.last_cluster_state
                actions, _ = _select_actions(policy, algo, pending, obs, cluster_state, cfg, training=False)
                result = env.step(actions)
                obs = result.obs
                done = result.done
                phase_name = phase_for_step(step_idx)
                compliant = result.info["reward_breakdown"].get("per_slice_compliant", {})
                for s, c in compliant.items():
                    by_phase[phase_name][s].append(bool(c))
                step_log.append({"episode": ep, "step": step_idx, "phase": phase_name, "compliant": compliant})
                elapsed = time.monotonic() - step_start
                remaining = cfg.episode.step_seconds - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            print(f"[{arm}] episode {ep}/{EPISODES} done", file=sys.stderr)
    finally:
        env.close()

    summary = {
        phase: {s: (100.0 * sum(v) / len(v) if v else float("nan")) for s, v in slices.items()}
        for phase, slices in by_phase.items()
    }
    return {"summary": summary, "step_log": step_log}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm, (algo, reward_mode) in ARMS.items():
        print(f"=== running {arm} ({algo}/{reward_mode}) live, {EPISODES} episodes, seed {SEED} ===", file=sys.stderr)
        results[arm] = run_arm(arm, algo, reward_mode)
        print(f"[{arm}] summary: {json.dumps(results[arm]['summary'], indent=2)}")

    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
