#!/usr/bin/env python3
"""Preliminary, OFFLINE-ONLY probe: do the four frozen checkpoints trained
under saclb_offline_campaign.yaml's CONSTANT demand generalize to demand
that varies within an episode (PhaseVaryingClosedLoopKpmSource's low/high/
medium phases, see phase_varying_kpm_source.py)? No live rig time used --
this is meant as small, cheap, preliminary evidence pointing at the
journal-length follow-up's full live time-varying campaign (R1/R2 of
experiments/REWORK_PLAN.md), not a replacement for it.

Deliberately small scope (3 seeds x 5 episodes/arm, matching the live
campaign's own per-arm episode count for direct comparability) -- reuses
mc_runner._select_actions (the exact same action-selection call run_mc's
own run_single makes) and each arm's already-trained, frozen checkpoint;
trains nothing.

Usage:
    python3 experiments/scripts/probe_time_varying_demand_offline.py \
        --out experiments/results/time_varying_demand_probe
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from phase_varying_kpm_source import PHASE_BOUNDARIES, PhaseVaryingClosedLoopKpmSource, phase_for_step  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.mc_runner import _select_actions, build_policy  # noqa: E402

CONFIG_PATH = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_offline_campaign.yaml"
CKPT_ROOT = "/home/kmanojp/oranslice_rig/experiments/results/offline"
ARMS = {
    "dqn_sla": ("dqn", "sla"), "a2c_sla": ("a2c", "sla"),
    "dqn_qoe": ("dqn", "qoe"), "a2c_qoe": ("a2c", "qoe"),
}
OVERSUBSCRIPTION_FACTOR = 1.25  # matches train_offline.py exactly


def run_arm(arm: str, algo: str, reward_mode: str, seeds: list, episodes_per_seed: int) -> dict:
    cfg = load_saclb_config(CONFIG_PATH)
    base_mean_offered_ratio = {
        slice_id: min(0.98, OVERSUBSCRIPTION_FACTOR * spec.nominal_ratio / 100.0)
        for slice_id, spec in cfg.slice_by_id.items()
    }
    sd_for_slice = {slice_id: spec.sd for slice_id, spec in cfg.slice_by_id.items()}

    # phase -> slice -> list of per-step compliance bools, pooled across seeds/episodes
    by_phase = {name: {s: [] for s in cfg.slice_by_id} for _, _, name, _ in PHASE_BOUNDARIES}

    for seed in seeds:
        kpm = PhaseVaryingClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, base_mean_offered_ratio=base_mean_offered_ratio, sd_for_slice=sd_for_slice,
        )
        env = RANEnv(cfg, kpm, seed=seed, reward_mode=reward_mode)
        policy = build_policy(algo, cfg)
        ckpt = Path(CKPT_ROOT) / reward_mode / f"seed{seed}" / algo / "offline_closed_loop" / "rep_0" / "checkpoint.pt"
        policy.load_checkpoint(str(ckpt))

        for _ep in range(episodes_per_seed):
            kpm.reset_episode_clock()
            obs = env.reset()
            done = False
            while not done:
                pending = env.pending_requests()
                cluster_state = env.last_cluster_state
                actions, _ = _select_actions(policy, algo, pending, obs, cluster_state, cfg, training=False)
                result = env.step(actions)
                obs = result.obs
                done = result.done
                step_idx = result.info["step"]
                phase_name, _mult = phase_for_step(step_idx)
                compliant = result.info["reward_breakdown"].get("per_slice_compliant", {})
                for s, c in compliant.items():
                    by_phase[phase_name][s].append(bool(c))
        env.close()

    return {
        phase: {s: (float(np.mean(v)) * 100 if v else float("nan")) for s, v in slices.items()}
        for phase, slices in by_phase.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--episodes-per-seed", type=int, default=5)
    ap.add_argument("--out", default="experiments/results/time_varying_demand_probe")
    args = ap.parse_args()

    results = {}
    for arm, (algo, reward_mode) in ARMS.items():
        print(f"[probe] running {arm} ({algo}/{reward_mode}) ...", file=sys.stderr)
        results[arm] = run_arm(arm, algo, reward_mode, args.seeds, args.episodes_per_seed)
        for phase in results[arm]:
            print(f"  {phase:8s}: {results[arm][phase]}")

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[probe] wrote {out_path / 'results.json'}")


if __name__ == "__main__":
    main()
