#!/usr/bin/env python3
"""Live xApp entrypoint for the campaign's `baseline` arm: a static,
non-learning, non-per-step-adaptive per-slice PRB ratio -- the "heuristic,
built-in O-RAN, non-xApp" baseline the campaign handover requires, distinct
from the framework's own `lb_only` comparator (which IS an active,
context-reading heuristic that moves the ceiling every step based on
saturation/quota -- see qoe_oran_framework/comparators/lb_only_baseline.py;
that is a valid research comparator but not what "static RRM policy ratio,
no learning, no per-step control" means).

Realization chosen (see CAMPAIGN_LOG.md for why): the framework's built-in
`rrmPolicy.json` periodic-reload path (`nr_update_slice_policy()`) is
confirmed non-functional on this checkout (BRINGUP_LOG.md Stage 7 P5 --
commented out unless `doc/rrmPolicyJson.patch` is applied, which it is
not). Falls back to the handover's documented alternative: each slice's
ceiling is set to its calibrated NOMINAL ratio (not the cap -- the cap is
the upper bound a LEARNED policy may push toward; the nominal ratio is the
static operating point a non-learning RRM policy would actually run at)
and never adapted per step.

Mechanism (does not modify any frozen qoe_oran_framework/ source): reuses
RANEnv/run_single exactly as every other arm does, for byte-identical
logging/diagnostics/reward-computation code paths (critical for an
apples-to-apples comparison across all 5 arms), with two campaign-local,
non-invasive choices:
  1. `arrivals.ceiling_step_ratio: 0` in the campaign config (see
     experiments/configs/saclb_campaign_baseline.yaml) -- AdmissionGate.apply()
     moves each slice's ceiling by +/- step_ratio per accept/reject decision;
     with step_ratio=0 the ceiling can never move away from
     reset_ceilings()'s initial value (min_ratio_floor / nominal_ratio) for
     the entire episode, regardless of any decision.
  2. `AlwaysAcceptPolicy` (defined below, local to this script): every
     pending request is decided "accept" so the baseline never manufactures
     an admission-control signal of its own -- the static ratio itself is
     the only thing governing service, matching "no learning, no per-step
     control." Passed with algorithm="baseline_static" (NOT "lb_only") so
     the omega log's `method` field records accurate provenance and
     mc_runner._make_omega_tuple does not attach the (inapplicable)
     LB_ONLY_ROUTING_LIMITATION.

Note on the resulting wire behavior: because AdmissionGate.apply() still
calls kpm_source.send_control() once per (gnb_id, slice_id) touched by any
pending request each step (see env.py:180-184), the gNB DOES receive a
`slicing_control_m` most steps -- but every one carries the SAME
unchanging min/max ratio (step_ratio=0 keeps the value pinned), so this is
periodic reassertion of a static value, not adaptive per-step control. This
is realization "A" from the two the handover permits; documented here and
in CAMPAIGN_LOG.md as required ("record which realization was used").
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.live_kpm_source import LiveKpmSource  # noqa: E402
from qoe_oran_framework.mc_runner import run_single  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402

BASELINE_STATIC_LIMITATION = (
    "baseline_static is the campaign's non-learning, non-xApp RRM baseline: "
    "each slice's ceiling is fixed at its calibrated nominal_ratio for the "
    "whole run (arrivals.ceiling_step_ratio=0 in this arm's config prevents "
    "any accept/reject decision from moving it), realized via always-accept "
    "decisions rather than the framework's rrmPolicy.json periodic-reload "
    "path (confirmed non-functional on this checkout -- see "
    "MIGRATION_PRECONDITION_REPORT.md / BRINGUP_LOG.md Stage 7 P5). Distinct "
    "from the framework's own `lb_only` comparator, which actively adjusts "
    "the ceiling based on observed saturation/quota every step -- see "
    "experiments/scripts/run_baseline_static.py's module docstring."
)


class AlwaysAcceptPolicy:
    """Not an RLPolicy -- no learning, no checkpoint. Every request is
    decided 'accept' so the static ceiling (held fixed by this arm's
    ceiling_step_ratio=0 config) is the only thing governing service."""

    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[int, Optional[dict]]:
        return 1, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
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
        help="baseline is evaluated under BOTH: sla-mode reward is discarded "
             "(no learning happens either way) but QoE diagnostics are always "
             "computed passively when cfg.qoe is set -- see env.py. qoe-mode "
             "is also runnable for completeness, though the campaign's 5 arms "
             "list only one 'baseline' entry (sla-mode diagnostics already "
             "include the passive MOS/cost/sla_viol fields every other arm's "
             "sla-mode runs share).",
    )
    args = parser.parse_args()

    cfg = load_saclb_config(args.config)
    if len(cfg.gnbs) != 1:
        parser.error(f"requires a single-gNB config; {args.config} lists {len(cfg.gnbs)} gNBs")
    if cfg.gnbs[0].gnb_id != args.gnb_id:
        parser.error(f"--gnb-id {args.gnb_id!r} does not match the config's gNB id {cfg.gnbs[0].gnb_id!r}")
    if cfg.arrivals.ceiling_step_ratio != 0:
        parser.error(
            f"baseline_static requires arrivals.ceiling_step_ratio: 0 in {args.config} "
            f"(got {cfg.arrivals.ceiling_step_ratio}) -- otherwise accept/reject decisions "
            "would move the ceiling away from its static nominal_ratio, defeating the "
            "whole point of this arm. Use experiments/configs/saclb_campaign_baseline.yaml."
        )

    policy = AlwaysAcceptPolicy()
    kpm_source = LiveKpmSource(
        gnb_id=args.gnb_id, xapp_listen_port=args.xapp_listen_port, gnb_listen_port=args.gnb_listen_port,
        recv_timeout_s=args.recv_timeout_s,
    )
    env = RANEnv(cfg, kpm_source, seed=args.seed, reward_mode=args.reward_mode)

    print(f"[{args.run_id}] baseline_static: talking to gNB E2 agent "
          f"(listen={args.xapp_listen_port}, gNB={args.gnb_listen_port})...", file=sys.stderr)
    try:
        with OmegaLogger(args.omega_jsonl) as omega:
            summary = run_single(
                env, policy, "baseline_static", omega, args.episodes, args.seed, args.run_id,
                mode="live_testbed", training=False, cfg=cfg,
                extra_limitations=[BASELINE_STATIC_LIMITATION],
            )
    finally:
        kpm_source.close()

    print(json.dumps(asdict(summary), indent=2))
    print(f"\nOmega log: {args.omega_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()
