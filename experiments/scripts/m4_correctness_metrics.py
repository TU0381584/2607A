#!/usr/bin/env python3
"""M4 correctness-aware metrics, mirroring m2/m3_correctness_metrics.py's
mean_reward_per_step / block_precision definitions -- per
docs/PAPER5_M3_fl_dp.md's explicit instruction for M4: use correctness-
aware metrics from the start, not sla_compliance_all_slices alone.

M4-specific wrinkle, found via a smoke test before trusting any real
campaign result (docs/PAPER5_M4_disruption.md's verification section):
raw mean_reward_per_step is NOT comparable between a disrupted run and
its undisrupted baseline when the disruption itself changes request
VOLUME. reward.compute_step_reward's service term is
`priority_weight * accept_reward * n_accepted` -- linear in how many
requests were accepted that step -- so a demand spike mechanically
inflates raw per-step reward by processing more requests per step, with
no bearing on whether those requests were handled any better or worse.
Confirmed directly, not assumed: a spike_sev1 (2x arrivals multiplier)
smoke run measured mean n_pending/step at 1.26x baseline, while dropout
(which never touches arrivals) measured exactly 1.00x -- so this
distortion is real and specific to "spike", not a general M4 artifact.

Fix: `mean_reward_per_pending_request` below divides each step's reward
by that step's own already-logged `n_pending` (skipping n_pending=0
steps, whose reward reflects pre-existing backlog/violation state, not
this step's request-handling decisions) -- built from data every M4 run
already logs, no new instrumentation, the same discipline
mean_reward_per_step/block_precision were built under in the first
place. This is the PRIMARY metric for M4's disruption-cost comparisons;
raw mean_reward_per_step is still reported alongside for continuity, but
should not be read as a fair comparison for "spike" specifically.

For each (arm, kind, severity), reports the disrupted-run stats AND a
paired comparison against that arm's own UNDISRUPTED M2/M3 eval log (the
already-existing "control" -- reused, not re-derived, matching this
project's established preference for reusing an existing reference point
over deriving a second possibly-divergent one).

Usage:
    python3 experiments/scripts/m4_correctness_metrics.py \
        --m4-results experiments/results/m4_campaign/campaign_results.json \
        --m4-campaign-dir experiments/results/m4_campaign
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_correctness_metrics import bootstrap_ci, per_seed_metrics  # noqa: E402


def per_seed_metrics_normalized(eval_omega_path: str):
    """Returns (mean_reward_per_pending_request, mmtc_blocks, total_blocks).
    Same block-counting as per_seed_metrics; reward is normalized per
    already-logged n_pending instead of per step -- see module docstring."""
    episode_step_ratios = {}
    mmtc_blocks = 0
    total_blocks = 0
    with open(eval_omega_path) as fh:
        for line in fh:
            rec = json.loads(line)
            ev = rec.get("evidence", rec)
            if not isinstance(ev, dict):
                continue
            if ev.get("rollup"):
                for slice_id, n in ev.get("episode_block_by_slice", {}).items():
                    total_blocks += n
                    if slice_id == "mmtc":
                        mmtc_blocks += n
            elif "reward" in ev and "n_pending" in ev:
                n_pending = ev["n_pending"]
                if n_pending <= 0:
                    continue
                ep = rec["episode"]
                episode_step_ratios.setdefault(ep, []).append(ev["reward"] / n_pending)

    per_episode_means = [float(np.mean(v)) for v in episode_step_ratios.values() if v]
    mean_reward_per_pending_request = float(np.mean(per_episode_means)) if per_episode_means else float("nan")
    return mean_reward_per_pending_request, mmtc_blocks, total_blocks


M2_CAMPAIGN_DIR = "/home/kmanojp/oranslice_rig/experiments/results/m2_campaign"
M3_CAMPAIGN_DIR = "/home/kmanojp/oranslice_rig/experiments/results/m3_campaign"


def baseline_eval_path(arm: str, seed: int) -> str:
    """The arm's own already-existing, UNDISRUPTED eval log -- same path
    convention m2/m3_correctness_metrics.py already use, not a new one."""
    if arm == "fl_gat_ctde_sigma0.0":
        return f"{M3_CAMPAIGN_DIR}/fl_gat_ctde_sigma0.0/seed{seed}/eval/omega_log.jsonl"
    if arm == "single_agent_dqn":
        return f"{M2_CAMPAIGN_DIR}/single_agent_dqn/seed{seed}/eval/dqn/offline_eval/rep_0/omega_log.jsonl"
    return f"{M2_CAMPAIGN_DIR}/{arm}/seed{seed}/eval/omega_log.jsonl"


def disrupted_eval_path(m4_campaign_dir: str, arm: str, severity_label: str, seed: int) -> str:
    return f"{m4_campaign_dir}/{arm}/{severity_label}/seed{seed}/eval/omega_log.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m4-results", default="/home/kmanojp/oranslice_rig/experiments/results/m4_campaign/campaign_results.json")
    ap.add_argument("--m4-campaign-dir", default="/home/kmanojp/oranslice_rig/experiments/results/m4_campaign")
    args = ap.parse_args()

    with open(args.m4_results) as fh:
        data = json.load(fh)
    cells = data["results"]

    # Group by (arm, kind, severity) -> list of seeds
    conditions = {}
    for key, cell in cells.items():
        cond_key = (cell["arm"], cell["kind"], cell["severity"])
        conditions.setdefault(cond_key, []).append(cell["seed"])

    for (arm, kind, severity), seeds in sorted(conditions.items()):
        seeds = sorted(seeds)
        severity_label = f"{kind}_sev{severity}"
        disrupted_norm, baseline_norm = [], []
        disrupted_raw, baseline_raw = [], []
        disrupted_precisions = []
        for seed in seeds:
            d_path = disrupted_eval_path(args.m4_campaign_dir, arm, severity_label, seed)
            b_path = baseline_eval_path(arm, seed)
            if not Path(d_path).exists() or not Path(b_path).exists():
                continue
            d_norm, d_mmtc, d_total = per_seed_metrics_normalized(d_path)
            b_norm, _, _ = per_seed_metrics_normalized(b_path)
            d_raw, _, _ = per_seed_metrics(d_path)
            b_raw, _, _ = per_seed_metrics(b_path)
            disrupted_norm.append(d_norm)
            baseline_norm.append(b_norm)
            disrupted_raw.append(d_raw)
            baseline_raw.append(b_raw)
            if d_total > 0:
                disrupted_precisions.append(d_mmtc / d_total)

        if not disrupted_norm:
            continue
        d_arr, b_arr = np.array(disrupted_norm), np.array(baseline_norm)
        d_raw_arr, b_raw_arr = np.array(disrupted_raw), np.array(baseline_raw)
        diff = b_arr - d_arr  # positive = disruption cost reward, normalized metric
        lo, hi = bootstrap_ci(d_arr)
        print(f"=== {arm} / {severity_label} (n={len(d_arr)}) ===")
        print(f"  disrupted mean_reward_per_pending_request (PRIMARY): mean={d_arr.mean():.4f}, "
              f"95% CI=[{lo:.4f}, {hi:.4f}]")
        print(f"  [reference only, not volume-comparable for spike] raw mean_reward_per_step: "
              f"disrupted={d_raw_arr.mean():.3f}, baseline={b_raw_arr.mean():.3f}")
        if disrupted_precisions:
            p = np.array(disrupted_precisions)
            plo, phi = bootstrap_ci(p)
            print(f"  disrupted block_precision: mean={p.mean():.3f}, 95% CI=[{plo:.3f}, {phi:.3f}], "
                  f"n_seeds_with_any_blocks={len(disrupted_precisions)}/{len(d_arr)}")
        else:
            print(f"  disrupted block_precision: UNDEFINED (0/{len(d_arr)} ever blocked anything)")

        if len(d_arr) >= 2 and np.any(diff != 0):
            dlo, dhi = bootstrap_ci(diff)
            try:
                _w, w_p = stats.wilcoxon(b_arr, d_arr)
            except ValueError:
                w_p = float("nan")
            print(f"  disruption cost (baseline - disrupted, normalized): mean={diff.mean():.4f}, "
                  f"95% CI=[{dlo:.4f}, {dhi:.4f}], Wilcoxon p={w_p:.4f}, "
                  f"baseline wins {int((diff>0).sum())}, ties {int((diff==0).sum())}, "
                  f"disruption wins {int((diff<0).sum())}")
        print()


if __name__ == "__main__":
    main()
