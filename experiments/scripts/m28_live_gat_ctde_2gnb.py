#!/usr/bin/env python3
"""M28 prep: a live xApp orchestration script for a genuine 2-gNB
GAT-CTDE demo -- NOT a modification of qoe_oran_framework/xapp/
saclb_xapp.py (frozen, and hard-requires len(cfg.gnbs)==1 by design,
`parser.error` if not), and not usable via that script's --algorithm
choices anyway (dqn/a2c/rainbow/lb_only only -- GAT-CTDE is a different,
multi-agent runner: marl_training.run_episodes_marl +
marl.ctde_policy.GatCtdeMarlPolicy, not mc_runner.run_single). This is a
new, minimal, faithful mirror of saclb_xapp.py's own structure
(same functions in the same order: build policy, load checkpoint,
construct KpmSource, construct RANEnv, run, log, print summary) with
exactly the two swaps a genuine 2-gNB run needs: GatCtdeMarlPolicy +
run_episodes_marl instead of build_policy/run_single, and
MultiGnbLiveKpmSource instead of LiveKpmSource.

Evaluation-only (training=False), same as saclb_xapp.py -- loads a
frozen checkpoint trained via m27_scaling_reframe.py or
m6_run_experiment.py against a 2-gNB config, never trains live.

NOT YET RUN LIVE. See docs/PAPER5_M27_M28_scope.md: this rig has a
documented history of severe instability the moment multi-gNB
concurrency is pushed, gNB2's exact E2 ports are unconfirmed (see
saclb_live2gnb.yaml's header), and the user was unreachable for the
duration of the session that wrote this file -- the actual live 2-gNB
run is deliberately held for when they are back to watch it. Everything
short of that (this script's own control flow, checkpoint compatibility,
config dimensions) is offline-verified.

Usage (once gNB2's ports are confirmed and both gNBs are up):
    python3 experiments/scripts/m28_live_gat_ctde_2gnb.py \
        --config qoe_oran_framework/configs/saclb_live2gnb.yaml \
        --checkpoint experiments/results/m28_live_checkpoint/seed900/train/checkpoint.pt \
        --topology fully_connected \
        --gnb0-listen-port 6655 --gnb0-xapp-port 6600 \
        --gnb1-listen-port 6656 --gnb1-xapp-port 6601 \
        --episodes 5 --seed 900 --run-id m28_2gnb_pilot \
        --omega-jsonl experiments/results/live/m28_2gnb/pilot_omega_log.jsonl
"""
import argparse
import json
import sys

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")

from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402
from qoe_oran_framework.marl.ctde_policy import GatCtdeMarlPolicy  # noqa: E402
from qoe_oran_framework.marl.marl_env import node_feature_dim, request_context_dim  # noqa: E402
from qoe_oran_framework.marl.marl_training import run_episodes_marl  # noqa: E402
from qoe_oran_framework.marl.topology import build_adjacency, hex_grid_edges, ring_edges  # noqa: E402
from multi_gnb_live_kpm_source import MultiGnbLiveKpmSource  # noqa: E402

ACTION_DIM = 2

TWO_GNB_LIVE_LIMITATION = (
    "live 2-gNB GAT-CTDE run: gNB2's E2 ports are a best-effort guess "
    "(see saclb_live2gnb.yaml header), not independently verified against "
    "a confirmed source patch -- flagged on every record, not silently "
    "assumed correct."
)


def adjacency_for(topology: str, n_agents: int):
    if topology == "fully_connected":
        return build_adjacency(n_agents)
    if topology == "ring":
        return build_adjacency(n_agents, ring_edges(n_agents))
    if topology == "hex":
        return build_adjacency(n_agents, hex_grid_edges(n_agents))
    raise ValueError(f"unknown --topology {topology!r} (fully_connected|ring|hex)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topology", default="fully_connected", choices=["fully_connected", "ring", "hex"])
    parser.add_argument("--gnb0-id", default="gnb-0")
    parser.add_argument("--gnb1-id", default="gnb-1")
    parser.add_argument("--gnb0-listen-port", type=int, required=True, help="gNB0's E2AGENT_IN_PORT")
    parser.add_argument("--gnb0-xapp-port", type=int, required=True, help="this xApp's own recv port for gNB0")
    parser.add_argument("--gnb1-listen-port", type=int, required=True, help="gNB1's E2AGENT_IN_PORT")
    parser.add_argument("--gnb1-xapp-port", type=int, required=True, help="this xApp's own recv port for gNB1")
    parser.add_argument("--recv-timeout-s", type=float, default=30.0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--omega-jsonl", required=True)
    args = parser.parse_args()

    cfg = load_saclb_config(args.config)
    if len(cfg.gnbs) != 2:
        parser.error(f"this script is for exactly 2 gNBs; {args.config} lists {len(cfg.gnbs)}")

    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    adj = adjacency_for(args.topology, n_agents)

    policy = GatCtdeMarlPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM, adj)
    policy.load_checkpoint(args.checkpoint)

    kpm_source = MultiGnbLiveKpmSource(
        gnb_specs={
            args.gnb0_id: {"xapp_listen_port": args.gnb0_xapp_port, "gnb_listen_port": args.gnb0_listen_port},
            args.gnb1_id: {"xapp_listen_port": args.gnb1_xapp_port, "gnb_listen_port": args.gnb1_listen_port},
        },
        recv_timeout_s=args.recv_timeout_s,
    )
    env = RANEnv(cfg, kpm_source, seed=args.seed, reward_mode="sla")

    print(f"[{args.run_id}] talking to 2 gNB E2 agents "
          f"({args.gnb0_id}: listen={args.gnb0_xapp_port}/gNB={args.gnb0_listen_port}, "
          f"{args.gnb1_id}: listen={args.gnb1_xapp_port}/gNB={args.gnb1_listen_port})...", file=sys.stderr)
    try:
        with OmegaLogger(args.omega_jsonl) as omega:
            summary = run_episodes_marl(
                env, policy, "gat_ctde", omega, args.episodes, args.seed, args.run_id,
                mode="live_testbed", training=False, cfg=cfg,
                extra_limitations=[TWO_GNB_LIVE_LIMITATION],
            )
    finally:
        kpm_source.close()

    print(json.dumps(summary, indent=2))
    print(f"\nOmega log: {args.omega_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()
