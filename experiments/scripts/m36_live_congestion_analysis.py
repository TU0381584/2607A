#!/usr/bin/env python3
"""M36 aggregation: combine the offline demand-lever sweep
(m36_congestion_ranges_offline.csv) with each live UE-count's own
state_log.jsonl (captured via state_vector_probe.wrap_policy_for_state_
logging during the live m33_live_state_probe.py runs) into one ledger,
and mark the collapse-onset UE count -- the first UE count where this
policy's live block precision (mmtc-fraction of blocks, from that
UE-count's own omega_log.jsonl) hits zero.

Usage:
    python3 experiments/scripts/m36_live_congestion_analysis.py \
        --offline experiments/results/m36_congestion_ranges_offline.csv \
        --live-root experiments/results/m36_live \
        --ue-counts 1 2 3 4 5 6 \
        --out-csv experiments/results/m36_congestion_ranges.csv \
        --out-fig experiments/results/m36_congestion_ranges.png
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

CONGESTION_IDX = [1, 4, 7]


def congestion_stats(state_log_path: Path) -> dict:
    vals = []
    with open(state_log_path) as fh:
        for line in fh:
            row = json.loads(line)
            for idx in CONGESTION_IDX:
                vals.append(row["state"][idx])
    arr = np.asarray(vals)
    return {
        "n_decisions": len(arr),
        "congestion_mean": float(arr.mean()) if len(arr) else float("nan"),
        "congestion_min": float(arr.min()) if len(arr) else float("nan"),
        "congestion_p50": float(np.percentile(arr, 50)) if len(arr) else float("nan"),
        "congestion_p90": float(np.percentile(arr, 90)) if len(arr) else float("nan"),
        "congestion_p99": float(np.percentile(arr, 99)) if len(arr) else float("nan"),
        "congestion_max": float(arr.max()) if len(arr) else float("nan"),
    }


def block_precision(omega_log_path: Path):
    """Returns (precision or None-if-undefined, total_blocks) from the
    live run's own rollup records -- same convention as every other
    correctness-metrics script in this project (m2/m4/m6)."""
    mmtc_blocks, total_blocks = 0, 0
    with open(omega_log_path) as fh:
        for line in fh:
            rec = json.loads(line)
            ev = rec.get("evidence", rec)
            if isinstance(ev, dict) and ev.get("rollup"):
                for slice_id, n in ev.get("episode_block_by_slice", {}).items():
                    total_blocks += n
                    if slice_id == "mmtc":
                        mmtc_blocks += n
    if total_blocks == 0:
        return None, 0
    return mmtc_blocks / total_blocks, total_blocks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", default="experiments/results/m36_congestion_ranges_offline.csv")
    ap.add_argument("--live-root", default="experiments/results/m36_live")
    ap.add_argument("--ue-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--out-csv", default="experiments/results/m36_congestion_ranges.csv")
    ap.add_argument("--out-fig", default="experiments/results/m36_congestion_ranges.png")
    args = ap.parse_args()

    rows = []
    with open(args.offline) as fh:
        for r in csv.DictReader(fh):
            rows.append({"source": "offline", "ue_count": "", **{
                k: v for k, v in r.items() if k not in ("source",)
            }})

    live_root = Path(args.live_root)
    live_rows = []
    for n_ue in args.ue_counts:
        state_path = live_root / f"ue{n_ue}" / "state_log.jsonl"
        omega_path = live_root / f"ue{n_ue}" / "omega_log.jsonl"
        if not state_path.exists():
            print(f"[m36-analysis] SKIP ue{n_ue}: {state_path} not found yet")
            continue
        stats = congestion_stats(state_path)
        prec, total_b = block_precision(omega_path) if omega_path.exists() else (None, 0)
        row = {"source": "live", "ue_count": n_ue, "lever_arrivals_per_step": "", "lever_mean_offered_ratio": "",
               **stats, "block_precision": prec if prec is not None else "undefined", "total_blocks": total_b}
        live_rows.append(row)
        rows.append(row)
        prec_str = f"{prec:.3f}" if prec is not None else "undefined (0 blocks)"
        print(f"[m36-analysis] ue{n_ue}: congestion mean={stats['congestion_mean']:.4f} "
              f"p50={stats['congestion_p50']:.4f} p90={stats['congestion_p90']:.4f} "
              f"max={stats['congestion_max']:.4f} n={stats['n_decisions']} | block_precision={prec_str}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_fields = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[m36-analysis] wrote {out_path} ({len(rows)} rows)")

    collapse_onset = None
    for row in sorted(live_rows, key=lambda r: r["ue_count"]):
        if row["block_precision"] == "undefined" or row["block_precision"] == 0.0:
            collapse_onset = row["ue_count"]
            break
    if collapse_onset is not None:
        print(f"[m36-analysis] collapse onset: {collapse_onset} UEs (first UE count with zero block precision)")
    elif live_rows:
        print("[m36-analysis] no collapse observed across the UE counts run so far")

    if live_rows:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            off_ratio = [float(r["lever_mean_offered_ratio"]) for r in rows if r["source"] == "offline" and r.get("lever_arrivals_per_step") == "2"]
            off_mean = [float(r["congestion_mean"]) for r in rows if r["source"] == "offline" and r.get("lever_arrivals_per_step") == "2"]
            ax.plot(off_ratio, off_mean, "o-", color="#898781", label="offline (arrivals=2, by mean_offered_ratio)")
            ue_counts = [r["ue_count"] for r in live_rows]
            live_means = [r["congestion_mean"] for r in live_rows]
            ax2 = ax.twiny()
            ax2.plot(ue_counts, live_means, "s-", color="#c1502e", label="live (by UE count)")
            ax.set_xlabel("offline mean_offered_ratio")
            ax2.set_xlabel("live UE count")
            ax.set_ylabel("congestion_level (mean)")
            if collapse_onset is not None:
                ax2.axvline(collapse_onset, color="#c1502e", linestyle="--", alpha=0.5)
                ax2.annotate(f"collapse onset\n({collapse_onset} UEs)", (collapse_onset, max(live_means)),
                             fontsize=8, color="#c1502e")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
            ax.set_title("M36: offline vs. live congestion_level distribution")
            fig.tight_layout()
            fig.savefig(args.out_fig, dpi=150)
            print(f"[m36-analysis] wrote {args.out_fig}")
        except ImportError:
            print("[m36-analysis] matplotlib not available, skipping figure")


if __name__ == "__main__":
    main()
