#!/usr/bin/env python3
"""Live xApp main loop: KPM in / control out, against a real OAI gNB via
its built-in RIC-free E2_AGENT UDP loop (LiveKpmSource).

Evaluation-only by design: loads a frozen checkpoint from offline
training and does not train live (see Stage Zero plan -- live runs
evaluate frozen weights, both for wall-clock reasons and to avoid
conflating live-environment noise with training instability).

Single physical gNB only: this rig has one real OAI gNB, so any config
passed here must have exactly one gNB entry. That means the paper #2 LB
term is inherently untestable live in Stage Zero (fairness_ratio is
trivially 1.0 with one gNB) -- only the admission-control side (closer to
paper #1's scope, but keeping paper #2's 3-slice roster since that's what
this testbed's traffic-profile provisioning already sets up) can be
live-validated until a multi-gNB rig exists. This is logged as a
limitation on every record, not silently glossed over.

Usage:
    python3 qoe_oran_framework/xapp/saclb_xapp.py \
        --config qoe_oran_framework/configs/saclb_live.yaml \
        --algorithm dqn --checkpoint results/offline/dqn/offline_closed_loop/rep_0/checkpoint.pt \
        --gnb-id gnb-0 --episodes 2 --run-id live-smoke-dqn-001 \
        --omega-jsonl results/live/dqn/rep_0/omega_log.jsonl
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.live_kpm_source import LiveKpmSource  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_single  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402

SINGLE_GNB_LIVE_LIMITATION = (
    "live Stage Zero runs against exactly one physical OAI gNB, so the "
    "paper #2 LB term (fairness_ratio across a gNB cluster) is trivially "
    "1.0 here and cannot be meaningfully validated live yet -- only the "
    "admission-control side is being tested against the real testbed."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", required=True, choices=["dqn", "a2c", "rainbow", "lb_only"])
    parser.add_argument(
        "--checkpoint", default="",
        help="frozen weights from offline training; required unless --algorithm lb_only",
    )
    parser.add_argument("--gnb-id", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--omega-jsonl", required=True)
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
        parser.error("--checkpoint is required for a learned algorithm (live runs evaluate frozen weights only)")

    cfg = load_saclb_config(args.config)
    if len(cfg.gnbs) != 1:
        parser.error(
            f"live xApp requires a single-gNB config; {args.config} lists {len(cfg.gnbs)} gNBs"
        )
    if cfg.gnbs[0].gnb_id != args.gnb_id:
        parser.error(
            f"--gnb-id {args.gnb_id!r} does not match the config's gNB id {cfg.gnbs[0].gnb_id!r}"
        )

    policy = build_policy(args.algorithm, cfg)
    if args.checkpoint:
        policy.load_checkpoint(args.checkpoint)

    kpm_source = LiveKpmSource(
        gnb_id=args.gnb_id, xapp_listen_port=args.xapp_listen_port, gnb_listen_port=args.gnb_listen_port,
        recv_timeout_s=args.recv_timeout_s,
    )
    env = RANEnv(cfg, kpm_source, seed=args.seed, reward_mode=args.reward_mode)

    print(f"[{args.run_id}] talking to gNB E2 agent (listen={args.xapp_listen_port}, gNB={args.gnb_listen_port})...", file=sys.stderr)
    try:
        with OmegaLogger(args.omega_jsonl) as omega:
            summary = run_single(
                env, policy, args.algorithm, omega, args.episodes, args.seed, args.run_id,
                mode="live_testbed", training=False, cfg=cfg,
                extra_limitations=[SINGLE_GNB_LIVE_LIMITATION],
            )
    finally:
        kpm_source.close()

    print(json.dumps(asdict(summary), indent=2))
    print(f"\nOmega log: {args.omega_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()
