#!/usr/bin/env python3
"""Stage 4 follow-up: attribute the ~85-90ms/step gap between the
configured cadence (5.0s) and the measured live wall-clock (~5.08-5.09s)
found in docs/STAGE4_instrumentation.md. measure_e2_latency.py already
ruled out the E2 wire protocol itself (poll ~0.57ms median, send_control
~0.17ms) and metrics_stage4_offline.py already ruled out DQN inference
(~68us) -- neither is large enough to explain an ~85-90ms residual.

This profiles RANEnv.reset()/.step() directly, live, against the real
gNB, using cProfile (external, stdlib instrumentation -- does not modify
any qoe_oran_framework/ source) to see where the REST of a step's
wall-clock time actually goes. Uses AlwaysAcceptPolicy (no checkpoint
needed) so the profile isolates the environment's own per-step cost from
a policy's forward pass, which is already separately measured and known
to be negligible.

Usage:
    python3 experiments/scripts/profile_step_overhead.py --steps 30
"""
import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.live_kpm_source import LiveKpmSource  # noqa: E402

CAMPAIGN_CFG = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign.yaml"


class AlwaysAccept:
    def select_action(self, state, training=False):
        return 1, None


def run_steps(env, policy, n_steps):
    obs = env.reset()
    for _ in range(n_steps):
        pending = env.pending_requests()
        actions = [1 for _ in pending]
        result = env.step(actions)
        obs = result.obs
        if result.done:
            obs = env.reset()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--gnb-id", default="gnb-0")
    args = ap.parse_args()

    cfg = load_saclb_config(CAMPAIGN_CFG)
    kpm = LiveKpmSource(gnb_id=args.gnb_id)
    env = RANEnv(cfg, kpm, seed=4444, reward_mode="sla")
    policy = AlwaysAccept()

    # untimed warmup (first reset/step pays one-time import/lazy-init costs)
    run_steps(env, policy, 3)

    wall_times = []
    profiler = cProfile.Profile()
    obs = env.reset()
    for i in range(args.steps):
        t0 = time.perf_counter()
        profiler.enable()
        pending = env.pending_requests()
        actions = [1 for _ in pending]
        result = env.step(actions)
        profiler.disable()
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)
        obs = result.obs
        if result.done:
            obs = env.reset()

    env.close()

    import numpy as np
    arr = np.array(wall_times)
    print(f"\nmeasured RANEnv.step() wall time over {len(arr)} steps (ms):")
    print(f"  mean={arr.mean():.2f} median={np.median(arr):.2f} p90={np.percentile(arr,90):.2f} max={arr.max():.2f}")

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats("cumulative")
    stats.print_stats(25)
    print("\n=== top 25 by cumulative time (this process's step() calls only, E2 poll wait time is real wall-clock but shows as socket recv) ===")
    print(buf.getvalue())


if __name__ == "__main__":
    main()
