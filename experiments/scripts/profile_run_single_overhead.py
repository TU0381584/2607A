#!/usr/bin/env python3
"""Stage 4 follow-up, part 2: profile_step_overhead.py showed RANEnv.step()
itself (E2 poll + QoE-mapper LSTM + everything) costs only ~13ms/step --
nowhere near the ~85-90ms/step gap between configured (5.0s) and measured
(~5.08-5.09s) live cadence. That test called env.step() directly, bypassing
mc_runner.run_single()'s own per-step bookkeeping (reward/compliance
tallying) and OmegaLogger.log() (file write + flush per step), which run
AFTER the sleep-to-cadence block in run_single's loop -- i.e. anything
there is NOT compensated for by the sleep and becomes pure additive
overhead on top of the 5.0s target. This runs the REAL run_single() (same
function every live arm uses), live, for a short synthetic episode
(steps_per_episode temporarily reduced -- diagnostic only, not a reported
result) to see if going through the full orchestration reproduces the
gap.

Usage:
    python3 experiments/scripts/profile_run_single_overhead.py --steps 20
"""
import argparse
import dataclasses
import sys
import time

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.live_kpm_source import LiveKpmSource  # noqa: E402
from qoe_oran_framework.mc_runner import run_single  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402

CAMPAIGN_CFG = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign.yaml"


class AlwaysAccept:
    def select_action(self, state, training=False):
        return 1, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--gnb-id", default="gnb-0")
    ap.add_argument("--omega-out", default="/tmp/profile_run_single_omega.jsonl")
    args = ap.parse_args()

    cfg = load_saclb_config(CAMPAIGN_CFG)
    cfg = dataclasses.replace(cfg, episode=dataclasses.replace(cfg.episode, steps_per_episode=args.steps))

    kpm = LiveKpmSource(gnb_id=args.gnb_id)
    env = RANEnv(cfg, kpm, seed=4444, reward_mode="sla")
    policy = AlwaysAccept()

    t0 = time.perf_counter()
    with OmegaLogger(args.omega_out) as omega:
        run_single(
            env, policy, "always_accept_diagnostic", omega, n_episodes=1, seed=4444,
            run_id="profile_run_single", mode="live_testbed", training=False, cfg=cfg,
        )
    t1 = time.perf_counter()
    env.close()

    total_s = t1 - t0
    per_step_ms = (total_s / args.steps) * 1000.0
    print(f"\nrun_single(): {args.steps} steps, total={total_s:.2f}s, "
          f"per-step={per_step_ms:.1f}ms (configured target=5000.0ms)")
    print(f"gap vs configured: {per_step_ms - 5000.0:.1f}ms/step")


if __name__ == "__main__":
    main()
