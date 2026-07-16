#!/usr/bin/env python3
"""Offline training runner: DQN / A2C / Rainbow against a synthetic KPM
feed, for the full 300-episode Table I schedule (or a shorter smoke run).
Paper #1's SAC-only comparator is just algorithm=dqn run against a
paper_variant: paper1 config -- see comparators/sac_only.py.

Defaults to ClosedLoopKpmSource, where admission ceilings actually
constrain served PRBs and unmet demand accumulates as backlog -- the
version whose numbers are meaningful. --source open_loop switches to
SyntheticKpmSource (demand independent of admission decisions) for a fast
wiring-only smoke check; do not treat its numbers as evidence of learned
SLA-tradeoff behaviour -- see replay_kpm_source.py docstrings.

Usage:
    python3 qoe_oran_framework/scripts/train_offline.py \
        --algorithm dqn --config qoe_oran_framework/configs/saclb_offline_dqn.yaml \
        --episodes 300 --seed 256
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.mc_runner import run_mc  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource, SyntheticKpmSource  # noqa: E402

# Oversubscription factor applied to each slice's nominal_ratio to derive
# ClosedLoopKpmSource's mean offered demand: without this, offered demand
# would sit at exactly the nominal quota and an "always accept" policy
# could trivially satisfy it forever, reproducing the open-loop source's
# no-tradeoff problem by a different route. >1.0 guarantees genuine,
# persistent contention the agent has to learn to manage.
OVERSUBSCRIPTION_FACTOR = 1.25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=["dqn", "a2c", "rainbow"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--results-dir", default="qoe_oran_framework/results/offline")
    parser.add_argument("--source", choices=["closed_loop", "open_loop"], default="closed_loop")
    parser.add_argument(
        "--reward-mode", choices=["sla", "qoe"], default="sla",
        help="sla=Stage Zero's frozen eq.2 baseline (default); qoe=Stage One's eq.9 "
             "QoE-mapper-driven reward -- requires the config to have a 'qoe:' section.",
    )
    args = parser.parse_args()

    cfg = load_saclb_config(args.config)

    def kpm_source_factory(seed: int):
        if args.source == "open_loop":
            return SyntheticKpmSource(seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
        mean_offered_ratio = {
            slice_id: min(0.98, OVERSUBSCRIPTION_FACTOR * spec.nominal_ratio / 100.0)
            for slice_id, spec in cfg.slice_by_id.items()
        }
        return ClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=mean_offered_ratio,
        )

    summaries = run_mc(
        cfg,
        args.algorithm,
        kpm_source_factory,
        n_reps=1,
        episodes_per_rep=args.episodes,
        base_seed=args.seed,
        mode=f"offline_{args.source}",
        training=True,
        results_dir=args.results_dir,
        reward_mode=args.reward_mode,
    )

    out_dir = Path(args.results_dir) / args.algorithm / f"offline_{args.source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    payload = [asdict(s) for s in summaries]
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"\nSummary written to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
