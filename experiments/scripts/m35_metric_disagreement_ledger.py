#!/usr/bin/env python3
"""M35 (external-review milestone, renumbered from the reviewer's own "M28"
to avoid colliding with this project's existing M28 -- the aborted live
2-gNB demo, docs/PAPER5_M27_M28_scope.md): a unified metric-disagreement
ledger across every paired arm/condition comparison already reported in
M2, M3, M4, and M6, re-analysis only -- no retraining, no checkpoint
access, no new campaigns.

For each comparison already established in this paper's own docs
(docs/PAPER5_M2_gat_ctde.md, PAPER5_M3_fl_dp.md, PAPER5_M4_disruption.md,
PAPER5_M6_topology.md), recomputes the paired verdict under three metrics,
all from data every run already logs, reusing this project's own existing
metric primitives rather than reimplementing them:
  (a) sla_compliance_all_slices -- recomputed per seed by averaging each
      eval log's own `episode_sla_compliance_all_slices` rollup field
      (m2/m3/m6's cached campaign_results.json / m6_results.json are NOT
      used directly, since M6's cache turned out incomplete for two of
      three arms -- see the M35 gate report -- so every campaign is
      computed the same way, directly from the logs, for consistency).
  (b) the campaign's own correctness-aware reward metric: plain
      mean_reward_per_step for M2/M3 (no volume/N confound between the
      arms being compared), mean_reward_per_pending_request for M4 (the
      metric M4's own docstring establishes as necessary once demand
      volume changes under disruption), mean_reward_per_gnb for M6 (same
      reasoning, cluster size confound). Reusing m2/m4/m6_correctness_
      metrics.py's own functions, not reimplementing them.
  (c) block_precision (mmtc-fraction of blocks), undefined for seeds with
      zero blocks, paired only over seeds where BOTH arms in a comparison
      have a defined value.

A comparison is flagged "disagree" if (a)'s direction (sign of the mean
paired difference) or significance verdict (paired Wilcoxon p<0.05) does
not match (b)'s and/or (c)'s, wherever (c) is defined for enough seeds to
test (n>=2 paired non-undefined precision values).

M4 data-availability gap (real, not a bug in this script): the official
900-909 raw per-seed omega_log.jsonl files only survive on disk for
gat_ctde/independent_dqn x dropout/churn (experiments/results/
m4_paired_test/, git-tracked, same seeds/values as the aggregate JSON in
m4_campaign/). Spike, single_agent_dqn, and fl_gat_ctde_sigma0.0 at the
official seeds have compliance only (baked into m4_campaign/
campaign_results.json) -- reward/precision are not re-derivable from
logs that no longer exist locally. A different, independently-drawn seed
batch (1000-1009) exists with full arm/kind coverage under
experiments/results/fresh_seed_retrain/m4_campaign/, but per this
milestone's own instruction ("if a campaign's raw per-comparison data is
missing, list it as a gap -- do not recompute from scratch"), that batch
is NOT silently substituted here; it is a real, separate, un-git-tracked
source a follow-up could use, flagged in the gate report instead.

Usage:
    python3 experiments/scripts/m35_metric_disagreement_ledger.py \
        --out results/m35_metric_disagreement_ledger.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_correctness_metrics import bootstrap_ci, per_seed_metrics  # noqa: E402
from m4_correctness_metrics import per_seed_metrics_normalized  # noqa: E402
from m6_correctness_metrics import per_seed_metrics_per_gnb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
M2_DIR = REPO_ROOT / "experiments/results/m2_campaign"
M3_DIR = REPO_ROOT / "experiments/results/m3_campaign"
M4_PAIRED_DIR = REPO_ROOT / "experiments/results/m4_paired_test"
M4_CAMPAIGN_RESULTS = REPO_ROOT / "experiments/results/m4_campaign/campaign_results.json"
M6_DIR = REPO_ROOT / "experiments/results/m6_pilot"


def sla_compliance_per_seed(eval_omega_path: Path) -> float:
    """Mean of episode_sla_compliance_all_slices across every rollup
    record in one seed's eval log -- the same quantity every campaign's
    own cached campaign_results.json / m6_results.json stores per seed,
    recomputed directly from the log so every campaign in this ledger is
    computed the identical way (M6's own cache turned out to only cover
    gat_ctde for some combos, not independent_dqn/single_agent_dqn)."""
    vals = []
    with open(eval_omega_path) as fh:
        for line in fh:
            rec = json.loads(line)
            ev = rec.get("evidence", rec)
            if isinstance(ev, dict) and ev.get("rollup") and "episode_sla_compliance_all_slices" in ev:
                vals.append(ev["episode_sla_compliance_all_slices"])
    return float(np.mean(vals)) if vals else float("nan")


def eval_path_flat_or_nested(base_dir: Path, arm: str, seed: int) -> Path:
    """single_agent_dqn nests its eval log under dqn/offline_eval/rep_0/
    (mc_runner.run_mc's own convention); gat_ctde/independent_dqn/
    fl_gat_ctde_sigma* write a flat eval/omega_log.jsonl -- the same
    resolution every m2/m3/m4/m6_correctness_metrics.py script already
    does independently; centralised here so this ledger does it once."""
    base = base_dir / arm / f"seed{seed}" / "eval"
    flat = base / "omega_log.jsonl"
    return flat if flat.exists() else base / "dqn" / "offline_eval" / "rep_0" / "omega_log.jsonl"


def paired_verdict(a_vals: list, b_vals: list) -> dict:
    """a - b paired diff, direction, and Wilcoxon significance verdict.
    Returns None fields if fewer than 2 usable pairs (matching this
    project's own established convention that a CI/test on <2 points is
    not meaningful, e.g. m2_correctness_metrics.bootstrap_ci's own
    precondition)."""
    a, b = np.asarray(a_vals, dtype=float), np.asarray(b_vals, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 2:
        return {"n": n, "mean_diff": float("nan"), "direction": "n/a", "p_value": float("nan"), "significant": None}
    diff = a - b
    mean_diff = float(diff.mean())
    direction = "positive" if mean_diff > 0 else ("negative" if mean_diff < 0 else "zero")
    if np.all(diff == 0):
        p = float("nan")
    else:
        try:
            _stat, p = stats.wilcoxon(a, b)
        except ValueError:
            p = float("nan")
    significant = (p < 0.05) if not np.isnan(p) else None
    return {"n": n, "mean_diff": mean_diff, "direction": direction, "p_value": float(p), "significant": significant}


def collect_arm(seeds, compliance_path_fn, reward_fn, path_fn):
    """For one arm across a seed list: per-seed compliance (nan on
    missing file), per-seed reward via the given campaign-appropriate
    reward_fn, and per-seed precision (nan if the log has zero blocks --
    undefined, not silently zero, matching every existing correctness-
    metrics script's own convention)."""
    compliance, reward, precision = [], [], []
    for seed in seeds:
        p = path_fn(seed)
        cp = compliance_path_fn(seed)
        if not Path(p).exists():
            compliance.append(float("nan"))
            reward.append(float("nan"))
            precision.append(float("nan"))
            continue
        compliance.append(sla_compliance_per_seed(cp))
        r, mmtc_b, total_b = reward_fn(str(p))
        reward.append(r)
        precision.append(mmtc_b / total_b if total_b > 0 else float("nan"))
    return compliance, reward, precision


def run_comparison(row: dict) -> dict:
    seeds = row["seeds"]
    a_compliance, a_reward, a_precision = collect_arm(
        seeds, row["path_a"], row["reward_fn"], row["path_a"])
    b_compliance, b_reward, b_precision = collect_arm(
        seeds, row["path_b"], row["reward_fn"], row["path_b"])

    v_compliance = paired_verdict(a_compliance, b_compliance)
    v_reward = paired_verdict(a_reward, b_reward)
    v_precision = paired_verdict(a_precision, b_precision)

    def verdict_disagrees(other):
        """Returns (disagree, kind). kind distinguishes a significance
        FLIP (one metric significant, the other not -- strong evidence of
        disagreement) from a direction-only split between two already-
        non-significant results (both metrics agree nothing is
        significant, they just point different ways -- expected sampling
        noise at small n, not the same strength of finding). Both are
        real per this milestone's own definition ("direction OR
        significance verdict"), but conflating them into one flag would
        hide which comparisons actually matter."""
        if other["n"] < 2 or v_compliance["n"] < 2:
            return None, ""
        sig_disagree = (v_compliance["significant"] is not None and other["significant"] is not None
                         and v_compliance["significant"] != other["significant"])
        dir_disagree = v_compliance["direction"] != other["direction"] and other["direction"] != "n/a"
        if sig_disagree:
            return True, "significance_flip"
        if dir_disagree:
            return True, "direction_only"
        return False, ""

    disagree_reward, kind_reward = verdict_disagrees(v_reward)
    disagree_precision, kind_precision = verdict_disagrees(v_precision)
    disagree_flag = bool(disagree_reward) or bool(disagree_precision)
    worst_kind = "significance_flip" if "significance_flip" in (kind_reward, kind_precision) else (
        "direction_only" if disagree_flag else "")

    return {
        "campaign": row["campaign"],
        "comparison": row["label"],
        "n_seeds": row["n_seeds_label"],
        "compliance_n": v_compliance["n"], "compliance_direction": v_compliance["direction"],
        "compliance_p": v_compliance["p_value"], "compliance_significant": v_compliance["significant"],
        "reward_n": v_reward["n"], "reward_direction": v_reward["direction"],
        "reward_p": v_reward["p_value"], "reward_significant": v_reward["significant"],
        "precision_n": v_precision["n"], "precision_direction": v_precision["direction"],
        "precision_p": v_precision["p_value"], "precision_significant": v_precision["significant"],
        "disagree_with_reward": disagree_reward, "disagree_with_reward_kind": kind_reward,
        "disagree_with_precision": disagree_precision, "disagree_with_precision_kind": kind_precision,
        "disagree": disagree_flag, "disagree_kind": worst_kind,
        "gap": row.get("gap", ""),
    }


def build_comparisons() -> list:
    comparisons = []

    # ---- M2: centralised (seeds 900-929) ----
    m2_seeds = list(range(900, 930))
    for other in ["single_agent_dqn", "independent_dqn"]:
        comparisons.append({
            "campaign": "M2", "label": f"gat_ctde vs {other} (centralised)",
            "seeds": m2_seeds, "n_seeds_label": len(m2_seeds),
            "path_a": lambda s: eval_path_flat_or_nested(M2_DIR, "gat_ctde", s),
            "path_b": lambda s, o=other: eval_path_flat_or_nested(M2_DIR, o, s),
            "reward_fn": per_seed_metrics,
        })

    # ---- M3: federation cost + privacy sweep (seeds 900-909, the only
    # seeds M3 has; M2's gat_ctde is read at this same seed subset) ----
    m3_seeds = list(range(900, 910))
    comparisons.append({
        "campaign": "M3", "label": "gat_ctde (M2, centralised) vs fl_gat_ctde_sigma0.0 (federation cost)",
        "seeds": m3_seeds, "n_seeds_label": len(m3_seeds),
        "path_a": lambda s: eval_path_flat_or_nested(M2_DIR, "gat_ctde", s),
        "path_b": lambda s: eval_path_flat_or_nested(M3_DIR, "fl_gat_ctde_sigma0.0", s),
        "reward_fn": per_seed_metrics,
    })
    for sigma in ["0.5", "1.0", "2.0", "4.0"]:
        comparisons.append({
            "campaign": "M3", "label": f"fl_gat_ctde_sigma0.0 vs sigma{sigma} (privacy cost)",
            "seeds": m3_seeds, "n_seeds_label": len(m3_seeds),
            "path_a": lambda s: eval_path_flat_or_nested(M3_DIR, "fl_gat_ctde_sigma0.0", s),
            "path_b": lambda s, sg=sigma: eval_path_flat_or_nested(M3_DIR, f"fl_gat_ctde_sigma{sg}", s),
            "reward_fn": per_seed_metrics,
        })

    # ---- M4: disruption cost vs each arm's own undisrupted M2/M3
    # baseline. Full data (gat_ctde, independent_dqn x dropout/churn) from
    # m4_paired_test/, seeds 900-909, matching the official campaign's own
    # seeds and values (cross-checked against m4_campaign/campaign_
    # results.json in the M35 gate report). ----
    m4_seeds = list(range(900, 910))

    def m2_baseline_path(arm):
        return lambda s: eval_path_flat_or_nested(M2_DIR, arm, s)

    for arm in ["gat_ctde", "independent_dqn"]:
        for kind in ["dropout", "churn"]:
            for sev in [1, 2, 3]:
                sev_label = f"{kind}_sev{sev}"
                comparisons.append({
                    "campaign": "M4", "label": f"{arm} {sev_label} vs own undisrupted baseline",
                    "seeds": m4_seeds, "n_seeds_label": len(m4_seeds),
                    "path_a": m2_baseline_path(arm),  # "a" = baseline (undisrupted)
                    "path_b": lambda s, a=arm, sl=sev_label: M4_PAIRED_DIR / a / sl / f"seed{s}" / "eval" / "omega_log.jsonl",
                    "reward_fn": per_seed_metrics_normalized,
                })

    # M4 gap rows: spike (all 4 arms) and single_agent_dqn/fl_gat_ctde_sigma0.0
    # (dropout+churn) at the official 900-909 seeds -- compliance-only,
    # raw logs genuinely gone. Listed as gaps per this milestone's own
    # instruction, not recomputed from a different seed batch.
    gap_conditions = []
    for arm in ["gat_ctde", "independent_dqn", "single_agent_dqn", "fl_gat_ctde_sigma0.0"]:
        gap_conditions.append((arm, "spike"))
    for arm in ["single_agent_dqn", "fl_gat_ctde_sigma0.0"]:
        gap_conditions.append((arm, "dropout"))
        gap_conditions.append((arm, "churn"))
    for arm, kind in gap_conditions:
        for sev in [1, 2, 3]:
            comparisons.append({
                "campaign": "M4", "label": f"{arm} {kind}_sev{sev} vs own undisrupted baseline",
                "seeds": [], "n_seeds_label": 0,
                "path_a": lambda s: Path("/nonexistent"), "path_b": lambda s: Path("/nonexistent"),
                "reward_fn": per_seed_metrics_normalized,
                "gap": ("raw per-seed omega_log.jsonl missing at official seeds 900-909; "
                        "compliance-only available in m4_campaign/campaign_results.json; "
                        "reward/precision exist only under the independently-seeded "
                        "(1000-1009), not-git-tracked fresh_seed_retrain/m4_campaign/"),
            })

    # ---- M6: gat_ctde vs single_agent_dqn, per topology, at N=19
    # (capfix, seeds 900-911, the bug-fixed primary 12-seed sample) and
    # N=7 (seeds 900-902, never affected by the N=19 arrival-cap bug). ----
    n19_seeds = list(range(900, 912))
    n7_seeds = [900, 901, 902]
    for topo in ["hex", "ring", "fully_connected"]:
        combo19 = M6_DIR / f"n19_{topo}_capfix"
        comparisons.append({
            "campaign": "M6", "label": f"gat_ctde vs single_agent_dqn, N=19 {topo}",
            "seeds": n19_seeds, "n_seeds_label": len(n19_seeds),
            "path_a": lambda s, c=combo19: eval_path_flat_or_nested(c, "gat_ctde", s),
            "path_b": lambda s, c=combo19: eval_path_flat_or_nested(c, "single_agent_dqn", s),
            "reward_fn": lambda p, n=19: per_seed_metrics_per_gnb(p, n),
        })
        combo7 = M6_DIR / f"n7_{topo}"
        comparisons.append({
            "campaign": "M6", "label": f"gat_ctde vs single_agent_dqn, N=7 {topo}",
            "seeds": n7_seeds, "n_seeds_label": len(n7_seeds),
            "path_a": lambda s, c=combo7: eval_path_flat_or_nested(c, "gat_ctde", s),
            "path_b": lambda s, c=combo7: eval_path_flat_or_nested(c, "single_agent_dqn", s),
            "reward_fn": lambda p, n=7: per_seed_metrics_per_gnb(p, n),
        })

    return comparisons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments/results/m35_metric_disagreement_ledger.csv"))
    args = ap.parse_args()

    comparisons = build_comparisons()
    rows = []
    for row in comparisons:
        if row.get("gap"):
            rows.append({
                "campaign": row["campaign"], "comparison": row["label"], "n_seeds": 0,
                "compliance_n": "", "compliance_direction": "", "compliance_p": "", "compliance_significant": "",
                "reward_n": "", "reward_direction": "", "reward_p": "", "reward_significant": "",
                "precision_n": "", "precision_direction": "", "precision_p": "", "precision_significant": "",
                "disagree_with_reward": "", "disagree_with_reward_kind": "",
                "disagree_with_precision": "", "disagree_with_precision_kind": "",
                "disagree": "", "disagree_kind": "",
                "gap": row["gap"],
            })
            continue
        rows.append(run_comparison(row))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    scored = [r for r in rows if r["disagree"] != ""]
    gaps = [r for r in rows if r["gap"]]
    n_disagree = sum(1 for r in scored if r["disagree"])
    n_flip = sum(1 for r in scored if r["disagree_kind"] == "significance_flip")
    n_dir_only = sum(1 for r in scored if r["disagree_kind"] == "direction_only")
    n_total = len(scored)
    pct = 100.0 * n_disagree / n_total if n_total else float("nan")

    print(f"[m35] wrote {out_path} ({len(rows)} rows: {n_total} scored, {len(gaps)} gaps)")
    print(f"compliance disagreed with the correctness-aware pair in {n_disagree} of {n_total} comparisons ({pct:.1f}%)")
    print(f"  of which {n_flip} are a significance FLIP (one metric significant, the other not -- the strong form)")
    print(f"  and {n_dir_only} are direction-only splits between two already-non-significant results (the weak form)")
    if gaps:
        print(f"\n{len(gaps)} comparisons listed as gaps (no full re-analysis, per this milestone's own instruction):")
        for r in gaps:
            print(f"  - [{r['campaign']}] {r['comparison']}: {r['gap']}")


if __name__ == "__main__":
    main()
