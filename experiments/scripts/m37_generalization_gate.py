#!/usr/bin/env python3
"""M37 gate check: recalibrated-sim generalisation to an unseen live load.

Reviewer's exact spec (renumbered from their "M30"): take the recalibrated
seed-900 checkpoint (experiments/results/m34_realistic_retrain_v2/seed900/
train/dqn/offline_train/rep_0/checkpoint.pt -- RealisticServedKpmSource,
fit only to the 3-UE and 6-UE anchors, see realistic_served_kpm_source.py)
live at a held-out UE count NOT used as a fit anchor (candidates: 4 or 5),
>=15 episodes, real traffic. GATE M37 (pass = precision >= 0.9 AND zero
collapsed episodes): report precision + per-episode block counts + reward
CI. If FAIL, stop and report -- it is a real negative result, not a bug
to iterate away.

"Collapsed episode" is not defined anywhere upstream (M6's own "collapse
rate" is a per-seed/per-run concept, not per-episode). This script adopts
the definition already implicit in this project's own fig8 script
(paper5_fig_live_recalibrated_fix.py's `n_zero` count) and the
manuscript's own language describing the fixed 6-UE run as "every episode
blocked" / the broken one as "complete silence": an episode with zero
total blocks. That is the natural reading in this single-gNB,
load-high-enough-to-expect-shedding context -- flagged here explicitly
since it's a judgment call, not something the reviewer's spec pins down.

Usage (once the held-out-load live run exists):
    python3 experiments/scripts/m37_generalization_gate.py \
        --omega-jsonl experiments/results/m37_live/ue4/omega_log.jsonl

Self-check against already-committed data (no new run needed) --
cross-validates this script's numbers against the manuscript's own
already-published claims before trusting it on new data:
    python3 experiments/scripts/m37_generalization_gate.py \
        --omega-jsonl experiments/results/live/m34_realistic_retrain_check/6ue_20ep_omega_log.jsonl \
        --self-check "precision=1.0 collapsed=0"
    python3 experiments/scripts/m37_generalization_gate.py \
        --omega-jsonl experiments/results/live/m31_highconf/6ue_20ep_omega_log.jsonl \
        --self-check "precision=undefined collapsed=20"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_correctness_metrics import bootstrap_ci  # noqa: E402


def load_episode_rollups(path: str):
    rollups = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            ev = rec.get("evidence", rec)
            if isinstance(ev, dict) and ev.get("rollup"):
                rollups.append(ev)
    return rollups


def gate_check(episodes: list) -> dict:
    n = len(episodes)
    blocks_per_ep = [sum(e.get("episode_block_by_slice", {}).values()) for e in episodes]
    total_blocks = sum(blocks_per_ep)
    mmtc_blocks = sum(e.get("episode_block_by_slice", {}).get("mmtc", 0) for e in episodes)
    precision = (mmtc_blocks / total_blocks) if total_blocks > 0 else None
    collapsed_episodes = sum(1 for b in blocks_per_ep if b == 0)
    rewards = [e["episode_mean_reward"] for e in episodes if "episode_mean_reward" in e]
    reward_ci = bootstrap_ci(rewards) if rewards else (float("nan"), float("nan"))
    passed = (precision is not None) and (precision >= 0.9) and (collapsed_episodes == 0)
    return {
        "n_episodes": n,
        "precision": precision,
        "total_blocks": total_blocks,
        "mmtc_blocks": mmtc_blocks,
        "collapsed_episodes": collapsed_episodes,
        "blocks_per_episode": blocks_per_ep,
        "reward_mean": (sum(rewards) / len(rewards)) if rewards else float("nan"),
        "reward_ci": reward_ci,
        "gate_m37_pass": passed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omega-jsonl", required=True)
    ap.add_argument("--self-check", default=None,
                     help="Optional 'precision=X collapsed=Y' string to assert against, for validating "
                          "this script on already-published data before trusting it on new data.")
    args = ap.parse_args()

    episodes = load_episode_rollups(args.omega_jsonl)
    if not episodes:
        print(f"[m37-gate] ERROR: no rollup episodes found in {args.omega_jsonl}")
        sys.exit(1)

    result = gate_check(episodes)
    prec_str = f"{result['precision']:.3f}" if result["precision"] is not None else "undefined (0 blocks)"
    print(f"[m37-gate] n_episodes={result['n_episodes']} precision={prec_str} "
          f"collapsed_episodes={result['collapsed_episodes']}/{result['n_episodes']} "
          f"blocks_per_episode={result['blocks_per_episode']}")
    print(f"[m37-gate] reward_mean={result['reward_mean']:.3f} "
          f"CI=[{result['reward_ci'][0]:.3f},{result['reward_ci'][1]:.3f}]")
    verdict = "PASS" if result["gate_m37_pass"] else "FAIL"
    print(f"[m37-gate] GATE M37: {verdict} (criterion: precision>=0.9 AND zero collapsed episodes)")

    if args.self_check:
        expected_prec = args.self_check.split("precision=")[1].split()[0]
        expected_collapsed = int(args.self_check.split("collapsed=")[1])
        prec_ok = (expected_prec == "undefined" and result["precision"] is None) or \
                  (expected_prec != "undefined" and result["precision"] is not None
                   and abs(result["precision"] - float(expected_prec)) < 1e-6)
        collapsed_ok = result["collapsed_episodes"] == expected_collapsed
        if prec_ok and collapsed_ok:
            print(f"[m37-gate] self-check OK: matches expected '{args.self_check}'")
        else:
            print(f"[m37-gate] self-check MISMATCH: expected '{args.self_check}', "
                  f"got precision={prec_str} collapsed={result['collapsed_episodes']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
