#!/usr/bin/env python3
"""M3 privacy-utility campaign (Block F deliverable): FederatedGatPolicy
(aggregator=fedavg) swept over a noise-multiplier grid, each level run
across the SAME seed subset Block E's centralized-CTDE campaign used
(experiments/results/m2_campaign/campaign_results.json, seeds 900-909 --
the first 10 of that campaign's 30-seed list), so m3_campaign_analysis.py
can pair each M3 seed directly against that already-completed centralized
result instead of re-running gat_ctde.

noise_multiplier=0.0 is included as the FL-only/no-DP control arm: it
isolates the federation-vs-centralization training-regime effect from the
DP privacy cost, exactly the two-step comparison the writeup needs
(centralized gat_ctde -> FL/no-DP -> FL/DP at increasing noise).

Same training budget as Block E's campaign (300 train / 50 eval episodes)
for direct comparability -- not the smaller smoke-test budget
m3_run_experiment.py defaults to.

Usage (from repo root, cwd=framework/ required):
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m3_privacy_sweep.py \
        --out-dir ../experiments/results/m3_campaign
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
import m3_run_experiment as m3  # noqa: E402

SEEDS = list(range(900, 910))  # first 10 seeds of Block E's 30-seed campaign list
NOISE_MULTIPLIERS = [0.0, 0.5, 1.0, 2.0, 4.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/home/kmanojp/oranslice_rig/experiments/results/m3_campaign")
    ap.add_argument("--train-episodes", type=int, default=300)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--local-steps-per-round", type=int, default=50)
    ap.add_argument("--noise-multipliers", type=float, nargs="+", default=NOISE_MULTIPLIERS)
    args = ap.parse_args()

    cfg = m3.load_saclb_config(m3.CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    print(f"[privacy_sweep] {len(SEEDS)} seeds x {len(args.noise_multipliers)} noise levels: "
          f"{SEEDS} x {args.noise_multipliers}")

    t0 = time.time()
    all_results = {}
    for sigma in args.noise_multipliers:
        tag = f"fl_gat_ctde_sigma{sigma}"
        print(f"[privacy_sweep] === noise_multiplier={sigma} ===")
        res = m3.run_fl_arm(
            cfg, sd_for_slice, SEEDS, args.train_episodes, args.eval_episodes, args.out_dir, tag,
            aggregator="fedavg", fedprox_mu=0.0, dp_noise_multiplier=sigma,
            dp_clip_norm=1.0, local_steps_per_round=args.local_steps_per_round,
        )
        all_results[str(sigma)] = res
        elapsed = time.time() - t0
        print(f"[privacy_sweep] sigma={sigma} done, cumulative elapsed {elapsed:.0f}s")
        out_path = Path(args.out_dir) / "campaign_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump({"seeds": SEEDS, "noise_multipliers": args.noise_multipliers,
                       "train_episodes": args.train_episodes, "eval_episodes": args.eval_episodes,
                       "local_steps_per_round": args.local_steps_per_round, "results": all_results}, fh, indent=2)

    print(f"[privacy_sweep] wrote {args.out_dir}/campaign_results.json, total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
