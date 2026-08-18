#!/usr/bin/env python3
"""M6: topology-scaling campaign -- does GAT-CTDE's paired edge over
single-agent DQN grow with cluster size N and with topology sparsity, or
is the M2 3-gNB, fully-connected result a special case? Same three arms
as M2 (gat_ctde, independent_dqn, single_agent_dqn), same reward/action
space/per-request TD scheme, same training discipline (torch-seeded
before policy construction, seed dir cleared before a fresh run, resume
verified against the ON-DISK checkpoint's own architecture, not file
existence) -- ONLY the config path (N=3/7/19 gNBs, arrivals scaled
proportionally so per-gNB contention stays ~constant as N grows -- see
configs/saclb_offline_dqn_n{7,19}.yaml's own diff-minimality check) and
the gat_ctde arm's adjacency (fully-connected / ring / hex-grid, from
qoe_oran_framework.marl.topology) vary, so any measured difference is
attributable to topology, not a confounded config change.

independent_dqn and single_agent_dqn never consume an adjacency matrix
at all (that is the whole point of both ablations -- no shared graph
representation) so --topology only affects the gat_ctde arm; the other
two arms' topology sensitivity, if any, comes entirely through N and the
per-gNB contention level, unconfounded by any graph-structure change.

Usage (from repo root, cwd=framework/ required, matching m2's own note):
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m6_run_experiment.py \
        --config-path qoe_oran_framework/configs/saclb_offline_dqn_n7.yaml \
        --topology ring \
        --seeds 900 901 902 --train-episodes 100 --eval-episodes 20 \
        --out-dir ../experiments/results/m6_campaign/n7_ring
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_mc  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402
from qoe_oran_framework.marl.ctde_policy import GatCtdeMarlPolicy  # noqa: E402
from qoe_oran_framework.marl.independent_dqn_ablation import IndependentPerGnbDqnPolicy  # noqa: E402
from qoe_oran_framework.marl.marl_env import node_feature_dim, request_context_dim  # noqa: E402
from qoe_oran_framework.marl.marl_training import run_episodes_marl  # noqa: E402
from qoe_oran_framework.marl.topology import build_adjacency, hex_grid_edges, ring_edges  # noqa: E402

REPO_ROOT = "/home/kmanojp/oranslice_rig"
BACKLOG_CAPACITY = 2000.0  # this project's established offline default (train_offline_live_scale.py)
ACTION_DIM = 2  # accept/reject, mapped to a PRB-ceiling nudge (action_mapping.AdmissionGate) -- unchanged
EVAL_SEED_OFFSET = 5000  # disjoint from any train seed, matching M1/M2's convention


def adjacency_for(topology: str, n_agents: int):
    """fully_connected reproduces M2 exactly (edges=None); ring/hex-grid
    are M6's new sparser topologies -- see topology.py's module docstring
    for the exact construction and why N=7/19 are the only hex-grid sizes
    defined (they are the standard 1-ring/2-ring cellular cluster sizes)."""
    if topology == "fully_connected":
        return build_adjacency(n_agents)
    if topology == "ring":
        return build_adjacency(n_agents, ring_edges(n_agents))
    if topology == "hex":
        return build_adjacency(n_agents, hex_grid_edges(n_agents))
    raise ValueError(f"unknown --topology {topology!r} (fully_connected|ring|hex)")


def make_kpm_source_factory(cfg, sd_for_slice, gnb_load_multiplier_mode="default"):
    """gnb_load_multiplier_mode:
      "default"     -- omit the argument entirely, same as m2_run_experiment.py:
                        ClosedLoopKpmSource auto-generates a seeded per-gNB
                        multiplier in [0.6, 1.4] (see its own docstring) --
                        NOT homogeneous, despite never having been called that
                        anywhere before M6 (docs/PAPER5_M6_topology.md).
      "homogeneous" -- explicit override, every gNB's multiplier forced to
                        1.0, for a genuine homogeneous-load comparison point
                        that does not otherwise exist.
    """
    def factory(seed):
        kwargs = dict(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
            backlog_capacity=BACKLOG_CAPACITY, sd_for_slice=sd_for_slice,
        )
        if gnb_load_multiplier_mode == "homogeneous":
            kwargs["gnb_load_multiplier"] = {g: 1.0 for g in cfg.gnb_ids}
        elif gnb_load_multiplier_mode != "default":
            raise ValueError(f"unknown gnb_load_multiplier_mode {gnb_load_multiplier_mode!r}")
        return ClosedLoopKpmSource(**kwargs)
    return factory


def _reload_eval_compliance(eval_omega_path: str):
    values = []
    with open(eval_omega_path) as fh:
        for line in fh:
            rec = json.loads(line)
            ev = rec.get("evidence", rec)
            if isinstance(ev, dict) and ev.get("rollup"):
                values.append(ev["episode_sla_compliance_all_slices"])
    if not values:
        return None
    return {"sla_compliance_all_slices": sum(values) / len(values), "n_episodes": len(values)}


def _checkpoint_matches_current_architecture(ckpt_path: str, probe_policy: "GatCtdeMarlPolicy") -> bool:
    """See m2_run_experiment.py's identical function for the real bug this
    guards against (a stale, architecturally-different checkpoint silently
    reused as if it were the current architecture's) -- same fix, reused
    verbatim since the failure mode is identical regardless of topology."""
    try:
        ckpt = torch.load(ckpt_path, map_location=probe_policy.device, weights_only=True)
        probe_policy.online.load_state_dict(ckpt["online"], strict=True)
        return True
    except (RuntimeError, KeyError):
        return False


def _clear_seed_dir(out_dir: str, tag: str, seed: int) -> None:
    """See m2_run_experiment.py's identical function for the eval-log
    append-contamination bug this guards against."""
    shutil.rmtree(f"{out_dir}/{tag}/seed{seed}", ignore_errors=True)


def run_gat_ctde_arm(cfg, sd_for_slice, seeds, train_episodes, eval_episodes, out_dir, tag,
                      topology, resume_seeds=False, gnb_load_multiplier_mode="default"):
    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    adj = adjacency_for(topology, n_agents)
    kpm_factory = make_kpm_source_factory(cfg, sd_for_slice, gnb_load_multiplier_mode)

    results = {}
    for seed in seeds:
        ckpt_path = f"{out_dir}/{tag}/seed{seed}/train/checkpoint.pt"
        eval_omega_path = f"{out_dir}/{tag}/seed{seed}/eval/omega_log.jsonl"
        if resume_seeds and Path(ckpt_path).exists() and Path(eval_omega_path).exists():
            probe = GatCtdeMarlPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM, adj)
            if _checkpoint_matches_current_architecture(ckpt_path, probe):
                reloaded = _reload_eval_compliance(eval_omega_path)
                if reloaded is not None:
                    results[seed] = reloaded
                    print(f"[m6:{tag}] seed={seed}: RESUMED, "
                          f"eval sla_compliance_all_slices={reloaded['sla_compliance_all_slices']:.3f}")
                    continue
            else:
                print(f"[m6:{tag}] seed={seed}: on-disk checkpoint does NOT match current architecture "
                      "-- retraining, not resuming")

        _clear_seed_dir(out_dir, tag, seed)
        torch.manual_seed(seed)  # BEFORE construction -- see m2_run_experiment.py's identical comment
        policy = GatCtdeMarlPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM, adj)
        env = RANEnv(cfg, kpm_factory(seed), seed=seed, reward_mode="sla")
        train_dir = f"{out_dir}/{tag}/seed{seed}/train"
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        with OmegaLogger(f"{train_dir}/omega_log.jsonl") as omega:
            run_episodes_marl(env, policy, tag, omega, train_episodes, seed,
                               f"{tag}_seed{seed}_train", "offline_train", True, cfg)
        env.close()
        ckpt_path = f"{train_dir}/checkpoint.pt"
        policy.save_checkpoint(ckpt_path)

        eval_seed = EVAL_SEED_OFFSET + seed
        eval_env = RANEnv(cfg, kpm_factory(eval_seed), seed=eval_seed, reward_mode="sla")
        eval_dir = f"{out_dir}/{tag}/seed{seed}/eval"
        Path(eval_dir).mkdir(parents=True, exist_ok=True)
        with OmegaLogger(f"{eval_dir}/omega_log.jsonl") as omega:
            summary = run_episodes_marl(eval_env, policy, tag, omega, eval_episodes, eval_seed,
                                         f"{tag}_seed{seed}_eval", "offline_eval", False, cfg)
        eval_env.close()
        results[seed] = summary
        print(f"[m6:{tag}] seed={seed}: eval sla_compliance_all_slices={summary['sla_compliance_all_slices']:.3f}")
    return results


def run_independent_dqn_arm(cfg, sd_for_slice, seeds, train_episodes, eval_episodes, out_dir, tag,
                             gnb_load_multiplier_mode="default"):
    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    kpm_factory = make_kpm_source_factory(cfg, sd_for_slice, gnb_load_multiplier_mode)

    results = {}
    for seed in seeds:
        _clear_seed_dir(out_dir, tag, seed)
        torch.manual_seed(seed)
        policy = IndependentPerGnbDqnPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM)
        env = RANEnv(cfg, kpm_factory(seed), seed=seed, reward_mode="sla")
        train_dir = f"{out_dir}/{tag}/seed{seed}/train"
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        with OmegaLogger(f"{train_dir}/omega_log.jsonl") as omega:
            run_episodes_marl(env, policy, tag, omega, train_episodes, seed,
                               f"{tag}_seed{seed}_train", "offline_train", True, cfg)
        env.close()
        policy.save_checkpoint(f"{train_dir}/checkpoint.pt")

        eval_seed = EVAL_SEED_OFFSET + seed
        eval_env = RANEnv(cfg, kpm_factory(eval_seed), seed=eval_seed, reward_mode="sla")
        eval_dir = f"{out_dir}/{tag}/seed{seed}/eval"
        Path(eval_dir).mkdir(parents=True, exist_ok=True)
        with OmegaLogger(f"{eval_dir}/omega_log.jsonl") as omega:
            summary = run_episodes_marl(eval_env, policy, tag, omega, eval_episodes, eval_seed,
                                         f"{tag}_seed{seed}_eval", "offline_eval", False, cfg)
        eval_env.close()
        results[seed] = summary
        print(f"[m6:{tag}] seed={seed}: eval sla_compliance_all_slices={summary['sla_compliance_all_slices']:.3f}")
    return results


def run_single_agent_dqn_arm(cfg, sd_for_slice, seeds, train_episodes, eval_episodes, out_dir, tag,
                              gnb_load_multiplier_mode="default"):
    kpm_factory = make_kpm_source_factory(cfg, sd_for_slice, gnb_load_multiplier_mode)
    results = {}
    for seed in seeds:
        _clear_seed_dir(out_dir, tag, seed)
        train_dir = f"{out_dir}/{tag}/seed{seed}/train"
        run_mc(cfg, "dqn", kpm_factory, n_reps=1, episodes_per_rep=train_episodes, base_seed=seed,
               mode="offline_train", training=True, results_dir=train_dir, reward_mode="sla")
        ckpt = f"{train_dir}/dqn/offline_train/rep_0/checkpoint.pt"

        def policy_factory(_s, ckpt=ckpt):
            p = build_policy("dqn", cfg)
            p.load_checkpoint(ckpt)
            return p

        eval_seed = EVAL_SEED_OFFSET + seed
        eval_dir = f"{out_dir}/{tag}/seed{seed}/eval"
        summaries = run_mc(cfg, "dqn", kpm_factory, n_reps=1, episodes_per_rep=eval_episodes,
                           base_seed=eval_seed, mode="offline_eval", training=False,
                           results_dir=eval_dir, policy_factory=policy_factory, reward_mode="sla")
        compliance = summaries[0].sla_compliance_all_slices if summaries else float("nan")
        results[seed] = {"sla_compliance_all_slices": compliance, "n_episodes": eval_episodes}
        print(f"[m6:{tag}] seed={seed}: eval sla_compliance_all_slices={compliance:.3f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-path", required=True,
                     help="e.g. qoe_oran_framework/configs/saclb_offline_dqn_n7.yaml")
    ap.add_argument("--topology", required=True, choices=["fully_connected", "ring", "hex"],
                     help="only affects the gat_ctde arm's adjacency; independent_dqn/single_agent_dqn "
                          "never consume one")
    ap.add_argument("--seeds", type=int, nargs="+", default=[900, 901, 902])
    ap.add_argument("--train-episodes", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", nargs="+", default=["gat_ctde", "independent_dqn", "single_agent_dqn"])
    ap.add_argument("--resume-seeds", action="store_true")
    ap.add_argument("--gnb-load-multiplier-mode", default="default", choices=["default", "homogeneous"],
                     help="'default' matches M2/M3/M4 exactly (seeded-random [0.6,1.4] per gNB, NOT "
                          "homogeneous despite never having been called that before -- see "
                          "docs/PAPER5_M6_topology.md). 'homogeneous' forces every gNB's multiplier to "
                          "1.0, the genuine homogeneous-load comparison point M7 needs.")
    args = ap.parse_args()

    cfg = load_saclb_config(args.config_path)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    print(f"[m6] {len(cfg.gnb_ids)}-gNB config loaded: {cfg.gnb_ids}, topology={args.topology}, "
          f"arrivals/step={cfg.arrivals.synthetic_arrivals_per_step}, "
          f"gnb_load_multiplier_mode={args.gnb_load_multiplier_mode}")

    all_results = {}
    t0 = time.time()
    if "gat_ctde" in args.arms:
        all_results["gat_ctde"] = run_gat_ctde_arm(
            cfg, sd_for_slice, args.seeds, args.train_episodes, args.eval_episodes, args.out_dir,
            "gat_ctde", args.topology, resume_seeds=args.resume_seeds,
            gnb_load_multiplier_mode=args.gnb_load_multiplier_mode)
    if "independent_dqn" in args.arms:
        all_results["independent_dqn"] = run_independent_dqn_arm(
            cfg, sd_for_slice, args.seeds, args.train_episodes, args.eval_episodes, args.out_dir,
            "independent_dqn", gnb_load_multiplier_mode=args.gnb_load_multiplier_mode)
    if "single_agent_dqn" in args.arms:
        all_results["single_agent_dqn"] = run_single_agent_dqn_arm(
            cfg, sd_for_slice, args.seeds, args.train_episodes, args.eval_episodes, args.out_dir,
            "single_agent_dqn", gnb_load_multiplier_mode=args.gnb_load_multiplier_mode)

    out_path = Path(args.out_dir) / "m6_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"[m6] wrote {out_path}, total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
