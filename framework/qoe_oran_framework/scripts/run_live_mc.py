#!/usr/bin/env python3
"""Live Monte-Carlo runner: N independent reps x M episodes against the
real testbed, for one algorithm at a time (run sequentially per algorithm
against the same up stack, to avoid E2 control-plane contention between
concurrent processes -- see Stage Zero plan).

Defaults to a REDUCED-SCOPE campaign, not the full N=5 reps x 50 episodes
x 60 steps x 5s/step protocol (~4.2h/algorithm, ~21h for all five) --
this default (3 reps x 4 episodes x 10 steps x 2s/step = ~4min/algorithm)
is a directional smoke campaign only, explicitly logged as a deviation via
extra_limitations on every Omega-tuple record so it can never be mistaken
for the acceptance-bar protocol after the fact. Scale up toward the full
protocol with --n-reps/--episodes/--steps-per-episode/--step-seconds.

Usage:
    python3 qoe_oran_framework/scripts/run_live_mc.py \
        --config qoe_oran_framework/configs/saclb_live.yaml \
        --algorithm dqn --gnb-id gnb-0 \
        --checkpoint qoe_oran_framework/results/offline/dqn/offline_closed_loop/rep_0/checkpoint.pt \
        --run-tag quick
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.live_kpm_source import LiveKpmSource  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_mc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", required=True, choices=["dqn", "a2c", "rainbow", "lb_only"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--gnb-id", required=True)
    parser.add_argument("--n-reps", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps-per-episode", type=int, default=10)
    parser.add_argument("--step-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--run-tag", default="live_mc")
    parser.add_argument("--results-dir", default="qoe_oran_framework/results/live")
    parser.add_argument("--xapp-listen-port", type=int, default=6600)
    parser.add_argument("--gnb-listen-port", type=int, default=6655)
    parser.add_argument("--recv-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--reward-mode", choices=["sla", "qoe"], default="sla",
        help="sla=Stage Zero's frozen eq.2 baseline (default); qoe=Stage One's eq.9 "
             "QoE-mapper-driven reward -- requires the config to have a 'qoe:' section.",
    )
    args = parser.parse_args()

    if args.algorithm != "lb_only" and not args.checkpoint:
        parser.error("--checkpoint is required unless --algorithm lb_only")

    cfg = load_saclb_config(args.config)
    if len(cfg.gnbs) != 1 or cfg.gnbs[0].gnb_id != args.gnb_id:
        parser.error(
            f"live MC requires a single-gNB config matching --gnb-id; "
            f"{args.config} has gnbs={cfg.gnb_ids}"
        )

    is_reduced = (
        args.steps_per_episode != 60 or args.step_seconds != 5.0
        or args.n_reps != 5 or args.episodes != 50
    )
    cfg.episode.steps_per_episode = args.steps_per_episode
    cfg.episode.step_seconds = args.step_seconds

    extra_limitations = []
    if is_reduced:
        extra_limitations.append(
            f"REDUCED-SCOPE live MC campaign, not the acceptance-bar protocol: "
            f"steps_per_episode={args.steps_per_episode} (protocol: 60), "
            f"step_seconds={args.step_seconds} (protocol: 5.0), "
            f"n_reps={args.n_reps} (protocol: 5), episodes={args.episodes} (protocol: 50) "
            "-- directional signal only."
        )

    def kpm_source_factory(seed: int) -> LiveKpmSource:
        return LiveKpmSource(
            gnb_id=args.gnb_id, xapp_listen_port=args.xapp_listen_port,
            gnb_listen_port=args.gnb_listen_port, recv_timeout_s=args.recv_timeout_s,
        )

    def policy_factory(seed: int):
        policy = build_policy(args.algorithm, cfg)
        if args.checkpoint:
            policy.load_checkpoint(args.checkpoint)
        return policy

    # run_mc's per-rep omega_log.jsonl path is results_dir/algorithm/mode/rep_N --
    # it does NOT depend on --run-tag. Two campaigns at different scales (e.g. a
    # quick smoke run and this full-protocol run) sharing the same --results-dir,
    # --algorithm, and rep count would silently APPEND into the same file
    # (OmegaLogger opens in append mode) -- caught this empirically when a
    # 1x50x60x5s run started writing into a stale 3-rep quick_mc log. Folding
    # run_tag into the results_dir makes distinct campaigns collision-proof by
    # construction, not by remembering to pass a different --results-dir.
    scoped_results_dir = str(Path(args.results_dir) / args.run_tag)

    summaries = run_mc(
        cfg, args.algorithm, kpm_source_factory, n_reps=args.n_reps,
        episodes_per_rep=args.episodes, base_seed=args.seed, mode="live_testbed",
        training=False, results_dir=scoped_results_dir, policy_factory=policy_factory,
        extra_limitations=extra_limitations, reward_mode=args.reward_mode,
    )

    out_dir = Path(args.results_dir) / args.algorithm / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    payload = [asdict(s) for s in summaries]
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"\nSummary written to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
