#!/usr/bin/env python3
"""Stage 4 (engineering instrumentation), offline half: per-decision
inference latency and policy footprint. Neither needs the live rig --
inference latency is a property of the network's compute graph running
on CPU (the real xApp's deployment device: DQNPolicy/A2CAdmissionPolicy
default to device="cpu", no GPU on the near-RT control path -- see
framework/drl_slicing/oranslice_drl/drl_policy.py:104), and footprint is
static (checkpoint file size + parameter count). Uses the SAME frozen
checkpoint already evaluated live (seed256/dqn/offline_closed_loop),
loaded read-only via the framework's own build_policy/load_checkpoint --
no framework source touched, no training.

The state vector used for timing is a correctly-shaped, randomly
initialized float32 array (via request_state_dim(cfg) +
encode_full_request_state's actual dimension) -- NOT a real state pulled
from a log, since per-request encoded states aren't persisted anywhere.
This is standard/adequate for latency benchmarking: a fixed feedforward
network's forward-pass cost does not depend on the input's specific
values, only its shape and the network's architecture, both of which are
real and unmodified here.

Usage:
    python3 experiments/scripts/metrics_stage4_offline.py --out docs/stage4_metrics_offline_raw.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import request_state_dim  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy  # noqa: E402

CAMPAIGN_CFG = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign.yaml"
CHECKPOINTS = {
    "dqn_sla": "/home/kmanojp/oranslice_rig/experiments/results/offline/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt",
    "dqn_qoe": "/home/kmanojp/oranslice_rig/experiments/results/offline/qoe/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt",
}
N_WARMUP = 50
N_TIMED = 2000


def count_params(policy) -> int:
    return sum(p.numel() for p in policy.q_network.parameters())


def measure_inference_latency(policy, state_dim: int, rng: np.random.RandomState) -> dict:
    states = [rng.randn(state_dim).astype(np.float32) for _ in range(N_WARMUP + N_TIMED)]
    for s in states[:N_WARMUP]:
        policy.select_action(s, training=False)

    latencies_us = []
    for s in states[N_WARMUP:]:
        t0 = time.perf_counter()
        policy.select_action(s, training=False)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)

    arr = np.array(latencies_us)
    return {
        "n": len(arr),
        "mean_us": float(arr.mean()),
        "median_us": float(np.median(arr)),
        "p90_us": float(np.percentile(arr, 90)),
        "p99_us": float(np.percentile(arr, 99)),
        "min_us": float(arr.min()),
        "max_us": float(arr.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/stage4_metrics_offline_raw.json")
    args = ap.parse_args()

    cfg = load_saclb_config(CAMPAIGN_CFG)
    dim = request_state_dim(cfg)
    rng = np.random.RandomState(4444)

    out = {"state_dim": dim, "device": "cpu", "n_warmup": N_WARMUP, "n_timed": N_TIMED, "arms": {}}
    for arm, ckpt_path in CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            print(f"[WARN] missing checkpoint for {arm}: {ckpt_path}", file=sys.stderr)
            continue
        policy = build_policy("dqn", cfg)
        policy.load_checkpoint(ckpt_path)
        n_params = count_params(policy)
        ckpt_bytes = Path(ckpt_path).stat().st_size
        latency = measure_inference_latency(policy, dim, rng)
        out["arms"][arm] = {
            "checkpoint_path": ckpt_path,
            "checkpoint_bytes": ckpt_bytes,
            "n_params": n_params,
            "inference_latency_us": latency,
        }
        print(f"[{arm}] n_params={n_params} ckpt_bytes={ckpt_bytes} "
              f"inference_median_us={latency['median_us']:.1f} p99_us={latency['p99_us']:.1f}",
              file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
