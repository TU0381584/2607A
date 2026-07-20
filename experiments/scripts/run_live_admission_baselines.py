#!/usr/bin/env python3
"""Live evaluation of the admission-efficiency objective's scripted
baselines (accept_all, reject_all, static_threshold) against the REAL
rig, using experiments/configs/saclb_campaign.yaml -- S1's own
already-validated live config (cap=12/4/3, proven against real measured
demand ~15/5/5 PRB on this exact rig, re-confirmed today at 16.73/5.00/5.02
via probe_e2_preconditions.py). Distinct from
saclb_admission_efficiency_v1.yaml (the offline-only config, whose
cap=65/30/20 were found NOT to bind against real demand -- see
CAMPAIGN_LOG.md's 2026-07-20 "Live testing" entry).

Unlike run_baseline_static.py (arrivals.ceiling_step_ratio=0, ceiling
frozen), THIS uses saclb_campaign.yaml's real ceiling_step_ratio=1 so
accept/reject decisions genuinely move the ceiling -- these are admission
heuristics, not a frozen-ratio arm.

Does not modify any frozen qoe_oran_framework/ source -- reuses
RANEnv/run_single/OmegaLogger/LiveKpmSource exactly as
run_baseline_static.py does. static_threshold uses the framework's own
LbOnlyHeuristic (algorithm="lb_only", the frozen comparator) with the
parameters tune_static_threshold.py already found
(utilization_threshold=0.7, capacity_margin=0.7).

Usage:
    python3 experiments/scripts/run_live_admission_baselines.py \
        --arm accept_all --episodes 2 --seed 950 \
        --gnb-id gnb-0 --run-id accept_all_live_seed950 \
        --omega-jsonl experiments/results/admission_efficiency_live/accept_all/seed950/omega_log.jsonl
"""
import argparse
import json
import sys
from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from qoe_oran_framework.comparators.lb_only_baseline import LbOnlyHeuristic  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.live_kpm_source import LiveKpmSource  # noqa: E402
from qoe_oran_framework.mc_runner import run_single  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402

LIVE_ADMISSION_BASELINE_LIMITATION = (
    "live admission-efficiency baseline: uses saclb_campaign.yaml's real "
    "ceiling_step_ratio=1 (not baseline_static's frozen ceiling), so "
    "accept/reject decisions genuinely move the ceiling -- distinct from "
    "the S1 campaign's own 'baseline' arm, which held the ceiling static. "
    "See experiments/scripts/run_live_admission_baselines.py module docstring."
)

TUNED_STATIC_THRESHOLD = {"utilization_threshold": 0.7, "capacity_margin": 0.7}


class AlwaysAcceptPolicy:
    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[int, Optional[dict]]:
        return 1, None


class AlwaysRejectPolicy:
    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[int, Optional[dict]]:
        return 0, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=["accept_all", "reject_all", "static_threshold"])
    parser.add_argument("--config", default="/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign.yaml")
    parser.add_argument("--gnb-id", default="gnb-0")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=950)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--omega-jsonl", required=True)
    parser.add_argument("--xapp-listen-port", type=int, default=6600)
    parser.add_argument("--gnb-listen-port", type=int, default=6655)
    parser.add_argument("--recv-timeout-s", type=float, default=30.0)
    parser.add_argument("--reward-mode", choices=["sla", "qoe"], default="qoe")
    args = parser.parse_args()

    cfg = load_saclb_config(args.config)
    if len(cfg.gnbs) != 1:
        parser.error(f"requires a single-gNB config; {args.config} lists {len(cfg.gnbs)} gNBs")

    if args.arm == "accept_all":
        policy, algorithm = AlwaysAcceptPolicy(), "accept_all"
    elif args.arm == "reject_all":
        policy, algorithm = AlwaysRejectPolicy(), "reject_all"
    else:
        policy, algorithm = LbOnlyHeuristic(cfg, **TUNED_STATIC_THRESHOLD), "lb_only"

    kpm_source = LiveKpmSource(
        gnb_id=args.gnb_id, xapp_listen_port=args.xapp_listen_port, gnb_listen_port=args.gnb_listen_port,
        recv_timeout_s=args.recv_timeout_s,
    )
    env = RANEnv(cfg, kpm_source, seed=args.seed, reward_mode=args.reward_mode)

    print(f"[{args.run_id}] {args.arm}: talking to gNB E2 agent "
          f"(listen={args.xapp_listen_port}, gNB={args.gnb_listen_port})...", file=sys.stderr)
    try:
        with OmegaLogger(args.omega_jsonl) as omega:
            summary = run_single(
                env, policy, algorithm, omega, args.episodes, args.seed, args.run_id,
                mode="live_admission_efficiency", training=False, cfg=cfg,
                extra_limitations=[LIVE_ADMISSION_BASELINE_LIMITATION],
            )
    finally:
        kpm_source.close()

    print(json.dumps(asdict(summary), indent=2))
    print(f"\nOmega log: {args.omega_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()
