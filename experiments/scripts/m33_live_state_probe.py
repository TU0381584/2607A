#!/usr/bin/env python3
"""Live xApp run, IDENTICAL to qoe_oran_framework/xapp/saclb_xapp.py's own
main() (same functions, same order, same arguments -- build_policy,
load_checkpoint, LiveKpmSource, RANEnv, OmegaLogger, run_single), except
for one added line: the loaded policy's select_action is wrapped (a plain
instance-attribute override, not a class/file edit) to log the real
13-dim state vector at every admission decision. Reproduced rather than
importing saclb_xapp.main() directly because that frozen script gives no
hook to insert the wrap between load_checkpoint() and run_single() --
this is the minimal faithful copy needed to add exactly that one line.

Usage: same flags as saclb_xapp.py, plus --state-log.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")

from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.live_kpm_source import LiveKpmSource  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_single  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402
from qoe_oran_framework.xapp.saclb_xapp import SINGLE_GNB_LIVE_LIMITATION  # noqa: E402
from state_vector_probe import wrap_policy_for_state_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", required=True, choices=["dqn", "a2c", "rainbow", "lb_only"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--gnb-id", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--omega-jsonl", required=True)
    parser.add_argument("--xapp-listen-port", type=int, default=6600)
    parser.add_argument("--gnb-listen-port", type=int, default=6655)
    parser.add_argument("--recv-timeout-s", type=float, default=30.0)
    parser.add_argument("--reward-mode", choices=["sla", "qoe"], default="sla")
    parser.add_argument("--state-log", required=True)
    args = parser.parse_args()

    if args.algorithm != "lb_only" and not args.checkpoint:
        parser.error("--checkpoint is required for a learned algorithm")

    cfg = load_saclb_config(args.config)
    if len(cfg.gnbs) != 1:
        parser.error(f"live xApp requires a single-gNB config; {args.config} lists {len(cfg.gnbs)} gNBs")
    if cfg.gnbs[0].gnb_id != args.gnb_id:
        parser.error(f"--gnb-id {args.gnb_id!r} does not match the config's gNB id {cfg.gnbs[0].gnb_id!r}")

    policy = build_policy(args.algorithm, cfg)
    if args.checkpoint:
        policy.load_checkpoint(args.checkpoint)

    Path(args.state_log).parent.mkdir(parents=True, exist_ok=True)
    state_fh = wrap_policy_for_state_logging(policy, args.state_log)

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
        state_fh.close()

    print(json.dumps(asdict(summary), indent=2))
    print(f"\nOmega log: {args.omega_jsonl}", file=sys.stderr)
    print(f"State log: {args.state_log}", file=sys.stderr)


if __name__ == "__main__":
    main()
