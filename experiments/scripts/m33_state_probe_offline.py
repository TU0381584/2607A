import sys
sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

import m6_run_experiment as m6  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_mc  # noqa: E402
from state_vector_probe import wrap_policy_for_state_logging  # noqa: E402

ORIG_CKPT = "/home/kmanojp/oranslice_rig/experiments/results/m8_live_anchor/offline_train/single_agent_dqn/seed900/train/dqn/offline_train/rep_0/checkpoint.pt"
CONFIG_PATH = "qoe_oran_framework/configs/saclb_offline_live1gnb.yaml"

cfg = m6.load_saclb_config(CONFIG_PATH)
sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}

def run_check(label, mean_offered_ratio, out_dir, state_log_path):
    m6.MEAN_OFFERED_RATIO = mean_offered_ratio
    kpm_factory = m6.make_kpm_source_factory(cfg, sd_for_slice)
    eval_seed = m6.EVAL_SEED_OFFSET + 900

    fh_holder = {}
    def policy_factory(_s, ckpt=ORIG_CKPT):
        p = build_policy("dqn", cfg)
        p.load_checkpoint(ckpt)
        fh_holder["fh"] = wrap_policy_for_state_logging(p, state_log_path)
        return p

    summaries = run_mc(cfg, "dqn", kpm_factory, n_reps=1, episodes_per_rep=2,
                        base_seed=eval_seed, mode="offline_eval", training=False,
                        results_dir=out_dir, policy_factory=policy_factory, reward_mode="sla")
    fh_holder["fh"].close()
    print(f"[{label}] wrote state log to {state_log_path}")

BASE = "/home/kmanojp/oranslice_rig/experiments/results/m33_realistic_demand_check"
run_check("1x", {"embb": 0.15, "urllc": 0.05, "mmtc": 0.05}, f"{BASE}/state_probe_1x/eval", f"{BASE}/state_probe_1x/states.jsonl")
run_check("2x", {"embb": 0.30, "urllc": 0.10, "mmtc": 0.10}, f"{BASE}/state_probe_2x/eval", f"{BASE}/state_probe_2x/states.jsonl")
