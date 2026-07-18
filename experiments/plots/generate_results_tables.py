#!/usr/bin/env python3
"""Phase B analysis: computes every number RESULTS_REPORT.md and paper
Table II report, from omega logs ONLY (no hand-typed numbers anywhere
downstream -- per the campaign handover's explicit requirement). Emits:
  - a Markdown results table (for RESULTS_REPORT.md)
  - a LaTeX results table (paper_conf/tables/table2_results.tex)
  - paired-seed win/loss counts + simple effect sizes (Cohen's d) between
    baseline and each learned arm, per metric
  - the raw per-arm-per-seed numbers as JSON (for the anomaly-adjudication
    write-up, which needs the exact a2c_qoe blocking distribution etc.)

Usage:
    python3 experiments/plots/generate_results_tables.py \
        --live-root experiments/results/live_campaign --seeds 950 951 952 \
        --out-md experiments/results/live_campaign/results_tables.md \
        --out-tex paper_conf/tables/table2_results.tex \
        --out-json experiments/results/live_campaign/results_raw.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE, ARMS, SLICE_ORDER, arm_run_dir, read_omega_log  # noqa: E402

ARM_REWARD_MODE = {
    "baseline": "sla", "dqn_sla": "sla", "a2c_sla": "sla",
    "dqn_qoe": "qoe", "a2c_qoe": "qoe",
}


def per_rep_metrics(omega_path: Path) -> dict:
    """Returns per-(arm,seed) aggregate metrics from one rep's omega log."""
    compliance = {s: [] for s in SLICE_ORDER}   # per-episode compliance fraction
    blocks = {s: [] for s in SLICE_ORDER}       # per-episode block count
    margins = {s: [] for s in SLICE_ORDER}      # per-step margin
    mos = {s: [] for s in SLICE_ORDER}          # per-step mos
    episode_rewards = []                        # per-episode mean reward

    for row in read_omega_log(omega_path):
        if row.step == -1:
            by_slice_c = row.evidence.get("episode_sla_compliance_by_slice") or {}
            for s in SLICE_ORDER:
                if s in by_slice_c:
                    compliance[s].append(by_slice_c[s])
            by_slice_b = row.evidence.get("episode_block_by_slice") or {}
            for s in SLICE_ORDER:
                blocks[s].append(by_slice_b.get(s, 0))
            if "episode_mean_reward" in row.evidence:
                episode_rewards.append(row.evidence["episode_mean_reward"])
        else:
            m = row.evidence.get("per_slice_sla_margin") or {}
            for s in SLICE_ORDER:
                if s in m:
                    margins[s].append(m[s])
            mm = row.evidence.get("mos_by_slice") or {}
            for s in SLICE_ORDER:
                if s in mm:
                    mos[s].append(mm[s])

    return {
        "compliance_pct": {s: (float(np.mean(v)) * 100 if v else float("nan")) for s, v in compliance.items()},
        "blocks_per_episode": {s: (float(np.mean(v)) if v else float("nan")) for s, v in blocks.items()},
        "mean_margin": {s: (float(np.mean(v)) if v else float("nan")) for s, v in margins.items()},
        "mean_mos": {s: (float(np.mean(v)) if v else float("nan")) for s, v in mos.items()},
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else float("nan"),
        "n_episodes": len(episode_rewards),
        "raw_blocks_per_episode_by_slice": {s: v for s, v in blocks.items()},  # for anomaly adjudication
    }


def cohens_d(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-root", default="experiments/results/live_campaign")
    ap.add_argument("--seeds", type=int, nargs="+", default=[950, 951, 952])
    ap.add_argument("--out-md", default="experiments/results/live_campaign/results_tables.md")
    ap.add_argument("--out-tex", default="paper_conf/tables/table2_results.tex")
    ap.add_argument("--out-json", default="experiments/results/live_campaign/results_raw.json")
    args = ap.parse_args()

    # per_rep[arm][seed] = metrics dict
    per_rep: dict = {arm: {} for arm in ARMS}
    for arm in ARMS:
        mode = ARM_REWARD_MODE[arm]
        for seed in args.seeds:
            omega_path = arm_run_dir(args.live_root, arm, mode, seed) / "omega_log.jsonl"
            if omega_path.exists():
                per_rep[arm][seed] = per_rep_metrics(omega_path)
            else:
                print(f"[generate_results_tables] WARNING: missing {omega_path}", file=sys.stderr)

    # aggregate mean +/- std across seeds
    agg: dict = {}
    for arm in ARMS:
        reps = per_rep[arm]
        seeds_present = sorted(reps.keys())
        agg[arm] = {"n_seeds": len(seeds_present), "seeds": seeds_present}
        for s in SLICE_ORDER:
            comp_vals = [reps[sd]["compliance_pct"][s] for sd in seeds_present if not np.isnan(reps[sd]["compliance_pct"][s])]
            block_vals = [reps[sd]["blocks_per_episode"][s] for sd in seeds_present if not np.isnan(reps[sd]["blocks_per_episode"][s])]
            margin_vals = [reps[sd]["mean_margin"][s] for sd in seeds_present if not np.isnan(reps[sd]["mean_margin"][s])]
            mos_vals = [reps[sd]["mean_mos"][s] for sd in seeds_present if not np.isnan(reps[sd]["mean_mos"][s])]
            agg[arm][s] = {
                "compliance_pct_mean": float(np.mean(comp_vals)) if comp_vals else float("nan"),
                "compliance_pct_std": float(np.std(comp_vals)) if comp_vals else float("nan"),
                "blocks_per_episode_mean": float(np.mean(block_vals)) if block_vals else float("nan"),
                "blocks_per_episode_std": float(np.std(block_vals)) if block_vals else float("nan"),
                "mean_margin_mean": float(np.mean(margin_vals)) if margin_vals else float("nan"),
                "mean_margin_std": float(np.std(margin_vals)) if margin_vals else float("nan"),
                "mean_mos_mean": float(np.mean(mos_vals)) if mos_vals else float("nan"),
                "mean_mos_std": float(np.std(mos_vals)) if mos_vals else float("nan"),
                "compliance_pct_per_seed": dict(zip(seeds_present, comp_vals)),
            }
        reward_vals = [reps[sd]["mean_episode_reward"] for sd in seeds_present if not np.isnan(reps[sd]["mean_episode_reward"])]
        agg[arm]["episode_reward_mean"] = float(np.mean(reward_vals)) if reward_vals else float("nan")
        agg[arm]["episode_reward_std"] = float(np.std(reward_vals)) if reward_vals else float("nan")

    # paired-seed win/loss counts + Cohen's d: baseline vs each learned arm, per slice, on compliance
    comparisons = {}
    for arm in ARMS:
        if arm == "baseline":
            continue
        wins = losses = ties = 0
        per_slice_d = {}
        for s in SLICE_ORDER:
            base_per_seed = agg["baseline"][s]["compliance_pct_per_seed"]
            arm_per_seed = agg[arm][s]["compliance_pct_per_seed"]
            common_seeds = sorted(set(base_per_seed) & set(arm_per_seed))
            for sd in common_seeds:
                if arm_per_seed[sd] > base_per_seed[sd]:
                    wins += 1
                elif arm_per_seed[sd] < base_per_seed[sd]:
                    losses += 1
                else:
                    ties += 1
            base_vals = [base_per_seed[sd] for sd in common_seeds]
            arm_vals = [arm_per_seed[sd] for sd in common_seeds]
            per_slice_d[s] = cohens_d(arm_vals, base_vals)
        comparisons[arm] = {"wins": wins, "losses": losses, "ties": ties, "cohens_d_compliance": per_slice_d}

    # ---- write JSON (raw, for anomaly adjudication prose) ----
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump({"per_rep": {a: {str(s): v for s, v in per_rep[a].items()} for a in ARMS},
                   "aggregate": agg, "comparisons_vs_baseline": comparisons}, fh, indent=2)
    print(f"[generate_results_tables] wrote {args.out_json}")

    # ---- Markdown table ----
    md_lines = ["| Arm | Slice | SLA compliance % (mean±std) | Blocks/episode (mean±std) | "
                "Mean backlog margin (mean±std) | Mean inferred MOS (mean±std) | n seeds |",
                "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for s in SLICE_ORDER:
            d = agg[arm][s]
            md_lines.append(
                f"| {arm} | {s} | {d['compliance_pct_mean']:.1f}±{d['compliance_pct_std']:.1f} | "
                f"{d['blocks_per_episode_mean']:.1f}±{d['blocks_per_episode_std']:.1f} | "
                f"{d['mean_margin_mean']:.3f}±{d['mean_margin_std']:.3f} | "
                f"{d['mean_mos_mean']:.3f}±{d['mean_mos_std']:.3f} | {agg[arm]['n_seeds']} |"
            )
    md_lines.append("")
    md_lines.append("| Arm | Mean episodic reward (mean±std) | n seeds |")
    md_lines.append("|---|---|---|")
    for arm in ARMS:
        md_lines.append(f"| {arm} | {agg[arm]['episode_reward_mean']:.4f}±{agg[arm]['episode_reward_std']:.4f} | {agg[arm]['n_seeds']} |")
    md_lines.append("")
    md_lines.append("### Paired-seed win/loss (SLA compliance, vs. baseline, summed across 3 slices x n seeds)")
    md_lines.append("| Arm | Wins | Losses | Ties | Cohen's d (compliance, per slice) |")
    md_lines.append("|---|---|---|---|---|")
    for arm, c in comparisons.items():
        d_str = ", ".join(f"{s}={c['cohens_d_compliance'][s]:.2f}" for s in SLICE_ORDER)
        md_lines.append(f"| {arm} | {c['wins']} | {c['losses']} | {c['ties']} | {d_str} |")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md_lines) + "\n")
    print(f"[generate_results_tables] wrote {args.out_md}")

    # ---- LaTeX table (T2) ----
    tex = []
    tex.append("% T2: headline results table -- AUTO-GENERATED by")
    tex.append("% experiments/plots/generate_results_tables.py. Do not hand-edit; re-run the script.")
    tex.append("\\begin{table*}[t]")
    tex.append("\\centering")
    tex.append("\\caption{Per-arm, per-slice results (mean $\\pm$ std across " + str(len(args.seeds)) + " seeds)}")
    tex.append("\\label{tab:results}")
    tex.append("\\begin{tabular}{@{}llrrrr@{}}")
    tex.append("\\toprule")
    tex.append("Arm & Slice & SLA compliance (\\%) & Blocks/episode & Backlog margin & Inferred MOS \\\\")
    tex.append("\\midrule")
    for arm in ARMS:
        label = ARM_STYLE[arm]["label"]
        for i, s in enumerate(SLICE_ORDER):
            d = agg[arm][s]
            arm_cell = label if i == 0 else ""
            tex.append(
                f"{arm_cell} & {s} & {d['compliance_pct_mean']:.1f}$\\pm${d['compliance_pct_std']:.1f} & "
                f"{d['blocks_per_episode_mean']:.1f}$\\pm${d['blocks_per_episode_std']:.1f} & "
                f"{d['mean_margin_mean']:.2f}$\\pm${d['mean_margin_std']:.2f} & "
                f"{d['mean_mos_mean']:.2f}$\\pm${d['mean_mos_std']:.2f} \\\\"
            )
        if arm != ARMS[-1]:
            tex.append("\\addlinespace")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table*}")

    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(tex) + "\n")
    print(f"[generate_results_tables] wrote {args.out_tex}")


if __name__ == "__main__":
    main()
