#!/usr/bin/env python3
"""Zero-training validity sweep for saclb_admission_efficiency_live_v1.yaml:
does accept_all/reject_all/static_threshold show real, non-saturated
differentiation at Lmax=10 / cap=12/4/3 (S1's own live-proven values),
before spending any training compute? Same methodology as the original
admission-efficiency design work (CAMPAIGN_LOG, 2026-07-20), reapplied to
the live-transferable config's smaller cap/Lmax scale.
"""
import sys
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

import numpy as np
from qoe_oran_framework.config import load_saclb_config
from qoe_oran_framework.env import RANEnv
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource

CONFIG = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_admission_efficiency_live_v1.yaml"


def make_env(seed, backlog_capacity, oversub_of_cap, reward_mode="qoe"):
    cfg = load_saclb_config(CONFIG)
    sd_for_slice = {slice_id: spec.sd for slice_id, spec in cfg.slice_by_id.items()}
    mean_offered_ratio = {
        slice_id: min(0.98, oversub_of_cap * cfg.slice_by_id[slice_id].max_ratio_cap / 100.0)
        for slice_id in cfg.slice_by_id
    }
    kpm = ClosedLoopKpmSource(seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
                               B=cfg.B, mean_offered_ratio=mean_offered_ratio,
                               backlog_capacity=backlog_capacity, sd_for_slice=sd_for_slice)
    return RANEnv(cfg, kpm, seed=seed, reward_mode=reward_mode)


def accept_all(req, env): return 1
def reject_all(req, env): return 0
def threshold_like(req, env):
    key = None
    for k in env.kpm_source._backlog:
        if k[1] == req.slice_id: key = k; break
    if key is None: return 1
    spec = env.cfg.slice_by_id[req.slice_id]
    return 0 if env.kpm_source._backlog[key] > spec.max_ratio_cap * 2 else 1


def run(policy_fn, seed, backlog_capacity, oversub, n_episodes=10):
    env = make_env(seed, backlog_capacity, oversub)
    compl = {"embb": [], "urllc": [], "mmtc": []}
    for ep in range(n_episodes):
        env.reset()
        for step in range(60):
            pending = env.pending_requests()
            actions = [policy_fn(req, env) for req in pending]
            result = env.step(actions)
            rb = result.info.get("reward_breakdown", {})
            for s, c in rb.get("per_slice_compliant", {}).items():
                compl[s].append(bool(c))
    return compl


if __name__ == "__main__":
    for backlog_capacity in [30.0, 60.0, 100.0]:
        for oversub in [1.2, 1.5]:
            print(f"--- backlog_capacity={backlog_capacity} oversub_of_cap={oversub} ---")
            for name, fn in [("accept_all", accept_all), ("reject_all", reject_all), ("threshold_like", threshold_like)]:
                agg = {"embb": [], "urllc": [], "mmtc": []}
                for seed in [256, 257, 258]:
                    c = run(fn, seed, backlog_capacity, oversub)
                    for s in agg: agg[s].extend(c[s])
                line = f"{name:16s}"
                for s in ["embb", "urllc", "mmtc"]:
                    arr = np.array(agg[s])
                    line += f"  {s}={np.mean(arr)*100:5.1f}%"
                print(line)
