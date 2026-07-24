#!/usr/bin/env python3
"""Stage 2 metrics layer: recomputes from EXISTING logs only (no
retraining). Adds three things the raw per-slice compliance tables
cannot show on their own:

  1. Priority-weighted SLA utility U = sum_s w_s*compliance_s / sum_s w_s,
     using the per-slice priority_weight actually in each experiment's
     config (Table I) -- the objective the policy is trained against,
     not an unweighted mean across slices of unequal declared importance.
  2. Per-slice violation-severity distributions (median, IQR, P90) from
     the continuous per-step SLA margin already computed by the reward
     function (reward.py's per_slice_sla_margin / ViolationCheck.margin),
     not just the binary compliance rate.
  3. Fisher's exact test (2x2: fully-SLA-compliant episodes vs not,
     arm vs baseline) in place of a bootstrap CI, since DQN arms have
     zero variance and a CI/Cohen's d over zero-variance groups is not
     informative.

Two experiments, two data sources:
  - LIVE CAMPAIGN: read directly from the existing omega logs
    (experiments/results/live_campaign/**/omega_log.jsonl) -- no new run.
  - CONGESTED (offline, held-out eval): the existing
    eval_congested_vs_baseline.py only ever persisted the MEAN margin,
    not the raw per-step distribution, so median/IQR/P90 cannot be
    recomputed from what's on disk. This script re-runs that SAME
    frozen-checkpoint evaluation (identical seeds, identical config,
    identical arms) ONCE to additionally persist the raw per-step margin
    arrays -- this is a deterministic re-evaluation of already-frozen
    weights, not a new training run.

Usage:
    python3 experiments/scripts/metrics_stage2.py --out docs/stage2_metrics_raw.json
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")


def _read_omega(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


SLICE_ORDER = ["embb", "urllc", "mmtc"]

LIVE_ARMS = {
    "baseline": ("sla", [950, 951, 952]),
    "dqn_sla": ("sla", [950, 951, 952]),
    "dqn_qoe": ("qoe", [950, 951, 952]),
}
LIVE_PRIORITY_WEIGHT = {"embb": 3.5, "urllc": 5.0, "mmtc": 0.3}
LIVE_VIOLATION_PENALTY = {"embb": 5.0, "urllc": 8.0, "mmtc": 2.5}

CONGESTED_PRIORITY_WEIGHT = {"embb": 3.5, "urllc": 5.0, "mmtc": 0.3}
CONGESTED_VIOLATION_PENALTY = {"embb": 5.0, "urllc": 20.0, "mmtc": 2.5}


def weighted_u(compliance_pct: dict, weights: dict) -> float:
    num = sum(weights[s] * compliance_pct[s] for s in SLICE_ORDER)
    den = sum(weights[s] for s in SLICE_ORDER)
    return num / den


def percentile_stats(values):
    if not values:
        return {"median": float("nan"), "iqr": float("nan"), "p90": float("nan"), "n": 0}
    arr = np.array(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "p90": float(np.percentile(arr, 90)),
        "n": len(arr),
    }


def fisher_exact_vs_baseline(fully_compliant_arm: int, total_arm: int,
                              fully_compliant_base: int, total_base: int):
    from scipy.stats import fisher_exact
    table = [
        [fully_compliant_arm, total_arm - fully_compliant_arm],
        [fully_compliant_base, total_base - fully_compliant_base],
    ]
    odds_ratio, p = fisher_exact(table)
    return {"table": table, "odds_ratio": odds_ratio, "p_value": p}


def chi2_step_proportion_vs_baseline(compliant_arm: int, total_arm: int,
                                      compliant_base: int, total_base: int):
    """Substitute for the episode-level Fisher test when EVERY arm shows
    0 fully-SLA-compliant episodes (the congested scenario: compliance is
    a within-episode step FRACTION, not an across-episode binary, so no
    episode is ever fully compliant for any arm -- the episode-level test
    is degenerate there, not just underpowered). Uses the same per-step
    compliant/total counts already collected, n=2700 steps/arm/slice, via
    a 2x2 chi-square test of independence (large-sample; safe here given
    n >> 5 in every cell)."""
    from scipy.stats import chi2_contingency
    table = [
        [compliant_arm, total_arm - compliant_arm],
        [compliant_base, total_base - compliant_base],
    ]
    chi2, p, dof, expected = chi2_contingency(table)
    return {"table": table, "chi2": chi2, "p_value": p}


def live_campaign_metrics(live_root: Path) -> dict:
    out = {}
    for arm, (mode, seeds) in LIVE_ARMS.items():
        margins = {s: [] for s in SLICE_ORDER}
        compliant_steps = {s: 0 for s in SLICE_ORDER}
        total_steps = {s: 0 for s in SLICE_ORDER}
        episode_fully_compliant = 0
        episode_total = 0
        for seed in seeds:
            path = live_root / arm / mode / f"rep_seed{seed}" / "omega_log.jsonl"
            if not path.exists():
                print(f"[WARN] missing {path}", file=sys.stderr)
                continue
            for row in _read_omega(path):
                ev = row.get("evidence", {})
                if row.get("step", -1) == -1:
                    by_slice = ev.get("episode_sla_compliance_by_slice")
                    if by_slice:
                        episode_total += 1
                        if all(by_slice.get(s, 0.0) >= 0.99995 for s in SLICE_ORDER):
                            episode_fully_compliant += 1
                    continue
                m = ev.get("per_slice_sla_margin", {})
                c = ev.get("per_slice_compliant", {})
                for s in SLICE_ORDER:
                    if s in m:
                        margins[s].append(m[s])
                    if s in c:
                        total_steps[s] += 1
                        if c[s]:
                            compliant_steps[s] += 1
        compliance_pct = {s: 100.0 * compliant_steps[s] / max(1, total_steps[s]) for s in SLICE_ORDER}
        out[arm] = {
            "compliance_pct": compliance_pct,
            "u_priority_weight": weighted_u(compliance_pct, LIVE_PRIORITY_WEIGHT),
            "u_violation_penalty": weighted_u(compliance_pct, LIVE_VIOLATION_PENALTY),
            "severity": {s: percentile_stats(margins[s]) for s in SLICE_ORDER},
            "episodes_fully_compliant": episode_fully_compliant,
            "episodes_total": episode_total,
        }
    base = out["baseline"]
    for arm in ("dqn_sla", "dqn_qoe"):
        out[arm]["fisher_vs_baseline"] = fisher_exact_vs_baseline(
            out[arm]["episodes_fully_compliant"], out[arm]["episodes_total"],
            base["episodes_fully_compliant"], base["episodes_total"],
        )
    return out


def congested_metrics(ckpt_root: str, seeds, episodes_per_seed: int,
                       congestion_range, backlog_capacity: float, shared_pool_prb: float) -> dict:
    """Re-evaluates the SAME frozen checkpoints eval_congested_vs_baseline.py
    already evaluated, ONLY adding raw per-step margin persistence (that
    script discarded per-step data after computing the mean). Deterministic
    given the same seeds/checkpoints/config -- not a new training run."""
    from shared_pool_kpm_source import SharedPoolCongestedKpmSource  # noqa: E402
    from qoe_oran_framework.config import load_saclb_config  # noqa: E402
    from qoe_oran_framework.env import RANEnv  # noqa: E402
    from qoe_oran_framework.mc_runner import _select_actions, build_policy  # noqa: E402

    CONFIG_PATH = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_offline_congested_v1.yaml"
    ARMS = {"baseline": None, "dqn_sla": ("dqn", "sla"), "dqn_qoe": ("dqn", "qoe")}
    out = {}
    for arm, spec in ARMS.items():
        algo, reward_mode = spec if spec else (None, "sla")
        margins = {s: [] for s in SLICE_ORDER}
        compliant_steps = {s: 0 for s in SLICE_ORDER}
        total_steps = {s: 0 for s in SLICE_ORDER}
        episode_fully_compliant = 0
        episode_total = 0
        episode_slice_compliant = {s: 0 for s in SLICE_ORDER}

        for seed in seeds:
            cfg = load_saclb_config(CONFIG_PATH)
            if arm == "baseline":
                cfg.arrivals = dataclasses.replace(cfg.arrivals, ceiling_step_ratio=0)
                reward_mode = "sla"
            nominal_ratio = {s: spec_.nominal_ratio for s, spec_ in cfg.slice_by_id.items()}
            sd_for_slice = {s: spec_.sd for s, spec_ in cfg.slice_by_id.items()}
            episode_rng = np.random.RandomState(seed + 5000)
            kpm = SharedPoolCongestedKpmSource(
                seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id), B=cfg.B,
                nominal_ratio=nominal_ratio, congestion_range=congestion_range,
                episode_rng=episode_rng, sd_for_slice=sd_for_slice, backlog_capacity=backlog_capacity,
                shared_pool_prb=shared_pool_prb,
            )
            env = RANEnv(cfg, kpm, seed=seed, reward_mode=reward_mode)
            policy = None
            if arm != "baseline":
                policy = build_policy(algo, cfg)
                ckpt = Path(ckpt_root) / reward_mode / "seed256" / algo / "checkpoint.pt"
                policy.load_checkpoint(str(ckpt))
            try:
                for _ep in range(episodes_per_seed):
                    kpm.new_episode_congestion()
                    obs = env.reset()
                    done = False
                    ep_compliant = {s: True for s in SLICE_ORDER}
                    while not done:
                        pending = env.pending_requests()
                        cluster_state = env.last_cluster_state
                        if arm == "baseline":
                            actions = [1] * len(pending)
                        else:
                            actions, _ = _select_actions(policy, algo, pending, obs, cluster_state, cfg, training=False)
                        result = env.step(actions)
                        obs = result.obs
                        done = result.done
                        rb = result.info["reward_breakdown"]
                        for s, m in rb.get("per_slice_sla_margin", {}).items():
                            margins[s].append(m)
                        for s, c in rb.get("per_slice_compliant", {}).items():
                            total_steps[s] += 1
                            if c:
                                compliant_steps[s] += 1
                            else:
                                ep_compliant[s] = False
                    episode_total += 1
                    if all(ep_compliant[s] for s in SLICE_ORDER):
                        episode_fully_compliant += 1
                    for s in SLICE_ORDER:
                        if ep_compliant[s]:
                            episode_slice_compliant[s] += 1
            finally:
                env.close()

        compliance_pct = {s: 100.0 * compliant_steps[s] / max(1, total_steps[s]) for s in SLICE_ORDER}
        out[arm] = {
            "compliance_pct": compliance_pct,
            "u_priority_weight": weighted_u(compliance_pct, CONGESTED_PRIORITY_WEIGHT),
            "u_violation_penalty": weighted_u(compliance_pct, CONGESTED_VIOLATION_PENALTY),
            "severity": {s: percentile_stats(margins[s]) for s in SLICE_ORDER},
            "episodes_fully_compliant": episode_fully_compliant,
            "episodes_total": episode_total,
            "episodes_slice_compliant": dict(episode_slice_compliant),
            "compliant_steps": dict(compliant_steps),
            "total_steps": dict(total_steps),
        }
        print(f"[congested] {arm}: compliance={compliance_pct} "
              f"U(priority_weight)={out[arm]['u_priority_weight']:.2f} "
              f"U(violation_penalty)={out[arm]['u_violation_penalty']:.2f} "
              f"episodes_slice_compliant={episode_slice_compliant}/{episode_total}", file=sys.stderr)

    base = out["baseline"]
    for arm in ("dqn_sla", "dqn_qoe"):
        # Episode-level Fisher test is degenerate here (0/45 fully-compliant
        # for every arm, every slice, including baseline -- congested
        # compliance is a within-episode step FRACTION, not an
        # across-episode binary the way the live campaign's is). Recorded
        # anyway for transparency, then superseded by the step-level
        # chi-square test below, which uses the real per-arm n=2700-step
        # sample and is not degenerate.
        out[arm]["fisher_vs_baseline_degenerate"] = fisher_exact_vs_baseline(
            out[arm]["episodes_fully_compliant"], out[arm]["episodes_total"],
            base["episodes_fully_compliant"], base["episodes_total"],
        )
        out[arm]["chi2_step_level_vs_baseline"] = {
            s: chi2_step_proportion_vs_baseline(
                out[arm]["compliant_steps"][s], out[arm]["total_steps"][s],
                base["compliant_steps"][s], base["total_steps"][s],
            )
            for s in SLICE_ORDER
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-root", default="/home/kmanojp/oranslice_rig/experiments/results/live_campaign")
    ap.add_argument("--congested-ckpt-root", default="/home/kmanojp/oranslice_rig/experiments/results/offline_congested")
    ap.add_argument("--out", default="docs/stage2_metrics_raw.json")
    args = ap.parse_args()

    result = {
        "live_campaign": live_campaign_metrics(Path(args.live_root)),
        "congested": congested_metrics(
            args.congested_ckpt_root, [950, 951, 952], 15, (0.4, 1.3), 15.0, 8.0,
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
