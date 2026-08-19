#!/usr/bin/env python3
"""M7: the heterogeneity-dividend question docs/PAPER5_M6_topology.md
Part 2 set up but did not answer -- Section VI of the paper originally
assumed "our topology treats every gNB as an interchangeable peer with
no client heterogeneity for FedProx's correction to address," a belief
M6 corrected: every M2/M3/M4 seed already had implicit, uncontrolled
per-gNB load heterogeneity (ClosedLoopKpmSource's auto-generated
[0.6, 1.4] multiplier), and m3_run_experiment.py now has a
gnb_load_multiplier_mode override (added this milestone) that makes
"homogeneous" (every gNB forced to 1.0) a genuine, deliberate
comparison point that did not exist before.

This sweeps {aggregator=fedavg, aggregator=fedprox} x
{gnb_load_multiplier_mode=homogeneous, =default(heterogeneous)}, DP
noise fixed at sigma=0 throughout (isolates the heterogeneity/FedProx
question from the already-characterised privacy cost,
Section~\\ref{sec:results-m3}'s own discipline), same seeds and episode
budget as the M2/M3 campaigns for direct comparability.

Hypothesis under test: if FedProx's proximal term is actually
correcting for client drift, its benefit over FedAvg (if any) should be
LARGER under "default" (genuine heterogeneity) than under
"homogeneous" (no heterogeneity for it to correct) -- a heterogeneity
DIVIDEND specific to FedProx. If FedProx shows the same effect (or no
effect) in both load modes, that is a null result, reported as such,
matching this project's "do not invent an effect" discipline.

Resumable (same pattern as m3_privacy_sweep.py): existing
campaign_results.json cells are kept, not re-run.

Usage (from repo root, cwd=framework/ required):
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m7_fedprox_heterogeneity.py \
        --out-dir ../experiments/results/m7_campaign \
        --seeds 900 901 902 --fedprox-mu 0.01
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
import m3_run_experiment as m3  # noqa: E402

CELLS = [
    ("fedavg_homogeneous", "fedavg", 0.0, "homogeneous"),
    ("fedavg_heterogeneous", "fedavg", 0.0, "default"),
    ("fedprox_homogeneous", "fedprox", None, "homogeneous"),   # mu filled from --fedprox-mu
    ("fedprox_heterogeneous", "fedprox", None, "default"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/home/kmanojp/oranslice_rig/experiments/results/m7_campaign")
    ap.add_argument("--seeds", type=int, nargs="+", default=[900, 901, 902])
    ap.add_argument("--train-episodes", type=int, default=300)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--local-steps-per-round", type=int, default=50)
    ap.add_argument("--fedprox-mu", type=float, default=0.01)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    results_path = out_dir / "campaign_results.json"
    campaign = {"seeds": args.seeds, "fedprox_mu": args.fedprox_mu, "results": {}}
    if results_path.exists():
        with open(results_path) as fh:
            campaign["results"] = json.load(fh).get("results", {})
        print(f"[m7] resuming, {len(campaign['results'])} cell(s) already present")

    cfg = m3.load_saclb_config(m3.CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    print(f"[m7] {len(cfg.gnb_ids)}-gNB config loaded: {cfg.gnb_ids}, "
          f"fedprox_mu={args.fedprox_mu}, seeds={args.seeds}")

    t0 = time.time()
    for tag, aggregator, mu, load_mode in CELLS:
        if tag in campaign["results"]:
            print(f"[m7] {tag}: already present, skipping")
            continue
        actual_mu = args.fedprox_mu if mu is None else mu
        results = m3.run_fl_arm(
            cfg, sd_for_slice, args.seeds, args.train_episodes, args.eval_episodes,
            str(out_dir), tag,
            aggregator=aggregator, fedprox_mu=actual_mu, dp_noise_multiplier=0.0,
            local_steps_per_round=args.local_steps_per_round,
            gnb_load_multiplier_mode=load_mode,
        )
        campaign["results"][tag] = results
        with open(results_path, "w") as fh:
            json.dump(campaign, fh, indent=2)
        print(f"[m7] {tag} done, elapsed so far: {time.time()-t0:.0f}s")

    print(f"[m7] ALL DONE, total elapsed {time.time()-t0:.0f}s, wrote {results_path}")


if __name__ == "__main__":
    main()
