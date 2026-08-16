#!/usr/bin/env python3
"""M2: three-arm comparison isolating the GAT+CTDE contribution specifically.

Arms:
  gat_ctde         -- new GatCtdeMarlPolicy (framework/qoe_oran_framework/marl/):
                       shared GAT encoder over the 3-gNB topology graph +
                       per-agent Q-heads + centralized QMIX mixer (training only).
  independent_dqn  -- new IndependentPerGnbDqnPolicy: N=3 separate DQNPolicy
                       instances (paper #4's own DQNPolicy class, reused
                       unchanged), NO topology sharing, NO parameter sharing --
                       same per-agent local input as gat_ctde's AgentQHead, so
                       the only difference from gat_ctde is the GAT+mixer.
  single_agent_dqn -- paper #4's existing single-agent DQNPolicy, UNMODIFIED,
                       run via the existing qoe_oran_framework.mc_runner.run_mc
                       against the same 3-gNB config (already supported --
                       env.encode_state() flattens every gNB into one joint
                       vector, one shared policy decides for all gNBs, exactly
                       paper #2's original multi-gNB pattern). No new code for
                       this arm at all.

Environment: framework/qoe_oran_framework/configs/saclb_offline_dqn.yaml
(paper #2's own 3-gNB, deliberately oversubscribed 110%-of-PRB-budget
config -- genuine offline contention, unmodified, read-only). Per
docs/PAPER5_M1_recalibration.md's conclusion, this offline environment is
used here as a live-anchored STRESS environment for the contention regime
paper #4's live rig never reaches -- not as a live-rank prediction claim.
Nothing here is evaluated live; the single live gNB paper #4 validated
remains the only real-hardware anchor. MEAN_OFFERED_RATIO (live-probe
anchored) and backlog_capacity=2000 (this project's established offline
default, per experiments/scripts/train_offline_live_scale.py) are reused
unchanged, matching M1's own env-parameter provenance.

Does not modify frozen qoe_oran_framework/ source, and old-rig artifacts
are irrelevant here since nothing this script produces is claimed as
current-rig-validated evidence -- it's exploratory paper #5 infrastructure,
clearly not paper #4 material.

Usage (from repo root, cwd=framework/ required -- see docs/STAGE10's
relative-path note re: RANEnv's qoe-mapper checkpoint path):
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m2_run_experiment.py \
        --seeds 900 901 902 --train-episodes 100 --eval-episodes 20 \
        --out-dir ../experiments/results/m2_gat_ctde
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
from qoe_oran_framework.marl.topology import build_adjacency  # noqa: E402

REPO_ROOT = "/home/kmanojp/oranslice_rig"
CONFIG_PATH = f"{REPO_ROOT}/framework/qoe_oran_framework/configs/saclb_offline_dqn.yaml"
BACKLOG_CAPACITY = 2000.0  # this project's established offline default (train_offline_live_scale.py)
ACTION_DIM = 2  # accept/reject, mapped to a PRB-ceiling nudge (action_mapping.AdmissionGate) -- unchanged
EVAL_SEED_OFFSET = 5000  # disjoint from any train seed, matching M1's fresh-seed convention


def make_kpm_source_factory(cfg, sd_for_slice):
    def factory(seed):
        return ClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
            backlog_capacity=BACKLOG_CAPACITY, sd_for_slice=sd_for_slice,
        )
    return factory


def _reload_eval_compliance(eval_omega_path: str):
    """Recompute sla_compliance_all_slices from an already-written eval
    omega_log.jsonl's rollup records -- the same quantity
    run_episodes_marl returns, reconstructed without re-running eval."""
    import json
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
    """True iff ckpt_path's state_dict keys/shapes match probe_policy's
    CURRENT network exactly (torch's own strict-load check, not a
    timestamp or file-existence proxy). Real bug this guards against: a
    resumed run's --out-dir can contain checkpoints from an earlier,
    architecturally-different campaign at the identical seed path (e.g.
    the pre-per-slice-heads AgentQHead's shared q_head.net vs. the
    current q_head.heads.N per-slice ModuleList) -- file-existence alone
    silently reused one such stale checkpoint's eval results as if they
    were the current architecture's, discovered when a 30-seed resume
    showed ALL 30 seeds as "already done" despite only 16 having actually
    been retrained. This function is the fix: load_state_dict(strict=True)
    IS the ground truth for "does this checkpoint fit this model," not a
    proxy for it."""
    try:
        ckpt = torch.load(ckpt_path, map_location=probe_policy.device, weights_only=True)
        probe_policy.online.load_state_dict(ckpt["online"], strict=True)
        return True
    except (RuntimeError, KeyError):
        return False


def _clear_seed_dir(out_dir: str, tag: str, seed: int) -> None:
    """Deletes seed<N>/ entirely before a fresh (non-resumed) train+eval
    run. Real bug this guards against, discovered during M4's build:
    OmegaLogger opens its output file in append ('a') mode and never
    truncates -- gat_ctde was retrained in place three times across this
    project's collapse-fix history (original, LayerNorm-only, per-slice-
    heads), and because nothing ever cleared seed<N>/eval/omega_log.jsonl
    between runs, every seed's eval log silently accumulated all three
    runs' episodes stacked together (150 records instead of 50, confirmed
    via the episode index resetting to 1 twice per file) -- every
    downstream metric computed from that file was a hidden 3-way average
    across two stale, architecturally-different runs and the true final
    one. See docs/PAPER5_M2_gat_ctde.md's correction section for the full
    diagnostic chain and its effect on the paper's headline numbers.
    Independent_dqn/single_agent_dqn never triggered this (never retrained
    in place) but get the same guard for future-proofing."""
    shutil.rmtree(f"{out_dir}/{tag}/seed{seed}", ignore_errors=True)


def run_gat_ctde_arm(cfg, sd_for_slice, seeds, train_episodes, eval_episodes, out_dir, tag,
                      resume_seeds=False):
    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    adj = build_adjacency(n_agents)
    kpm_factory = make_kpm_source_factory(cfg, sd_for_slice)

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
                    print(f"[m2:{tag}] seed={seed}: RESUMED (already trained, architecture verified), "
                          f"eval sla_compliance_all_slices={reloaded['sla_compliance_all_slices']:.3f}")
                    continue
            else:
                print(f"[m2:{tag}] seed={seed}: on-disk checkpoint does NOT match current architecture "
                      f"(stale from a different run) -- retraining, not resuming")

        _clear_seed_dir(out_dir, tag, seed)
        # torch.manual_seed BEFORE constructing the policy, not after --
        # weight init (nn.init.xavier_uniform_ in GATLayer, nn.Linear's own
        # default reset_parameters in AgentQHead) happens synchronously at
        # construction time, so seeding any later than this line has no
        # effect on it. mc_runner.py's own set_seeds() achieves the same
        # thing implicitly for single_agent_dqn (that path is unaffected by
        # this fix); this run_episodes_marl-based path had no torch seeding
        # at all before this fix -- see docs/PAPER5_REPRODUCIBILITY.md's
        # torch-seeding-gap section for how this was found and confirmed.
        torch.manual_seed(seed)
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
        print(f"[m2:{tag}] seed={seed}: eval sla_compliance_all_slices={summary['sla_compliance_all_slices']:.3f}")
    return results


def run_independent_dqn_arm(cfg, sd_for_slice, seeds, train_episodes, eval_episodes, out_dir, tag):
    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    kpm_factory = make_kpm_source_factory(cfg, sd_for_slice)

    results = {}
    for seed in seeds:
        _clear_seed_dir(out_dir, tag, seed)
        torch.manual_seed(seed)  # see run_gat_ctde_arm's comment -- must precede policy construction
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
        print(f"[m2:{tag}] seed={seed}: eval sla_compliance_all_slices={summary['sla_compliance_all_slices']:.3f}")
    return results


def run_single_agent_dqn_arm(cfg, sd_for_slice, seeds, train_episodes, eval_episodes, out_dir, tag):
    """Paper #4's own single-agent DQNPolicy, unmodified, via the existing
    mc_runner.run_mc -- no new code, exactly today's multi-gNB pattern."""
    kpm_factory = make_kpm_source_factory(cfg, sd_for_slice)
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
        print(f"[m2:{tag}] seed={seed}: eval sla_compliance_all_slices={compliance:.3f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[900, 901, 902])
    ap.add_argument("--train-episodes", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--out-dir", default=f"{REPO_ROOT}/experiments/results/m2_gat_ctde")
    ap.add_argument("--arms", nargs="+", default=["gat_ctde", "independent_dqn", "single_agent_dqn"])
    args = ap.parse_args()

    cfg = load_saclb_config(CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    print(f"[m2] {len(cfg.gnb_ids)}-gNB config loaded: {cfg.gnb_ids}")

    all_results = {}
    t0 = time.time()
    if "gat_ctde" in args.arms:
        all_results["gat_ctde"] = run_gat_ctde_arm(
            cfg, sd_for_slice, args.seeds, args.train_episodes, args.eval_episodes, args.out_dir, "gat_ctde")
    if "independent_dqn" in args.arms:
        all_results["independent_dqn"] = run_independent_dqn_arm(
            cfg, sd_for_slice, args.seeds, args.train_episodes, args.eval_episodes, args.out_dir, "independent_dqn")
    if "single_agent_dqn" in args.arms:
        all_results["single_agent_dqn"] = run_single_agent_dqn_arm(
            cfg, sd_for_slice, args.seeds, args.train_episodes, args.eval_episodes, args.out_dir, "single_agent_dqn")

    out_path = Path(args.out_dir) / "m2_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"[m2] wrote {out_path}, total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
