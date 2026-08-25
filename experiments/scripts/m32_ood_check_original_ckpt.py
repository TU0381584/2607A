import sys
sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from qoe_oran_framework.config import load_saclb_config
from qoe_oran_framework.mc_runner import build_policy, run_mc
from m6_run_experiment import make_kpm_source_factory
from m2_run_experiment import EVAL_SEED_OFFSET

cfg = load_saclb_config("qoe_oran_framework/configs/saclb_offline_live1gnb_2xload.yaml")
sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
kpm_factory = make_kpm_source_factory(cfg, sd_for_slice)

orig_ckpt = "/home/kmanojp/oranslice_rig/experiments/results/m8_live_anchor/offline_train/single_agent_dqn/seed900/train/dqn/offline_train/rep_0/checkpoint.pt"

def policy_factory(_s, ckpt=orig_ckpt):
    p = build_policy("dqn", cfg)
    p.load_checkpoint(ckpt)
    return p

seed = 900
eval_seed = EVAL_SEED_OFFSET + seed
eval_dir = "/home/kmanojp/oranslice_rig/experiments/results/m32_heavy_load_retrain/ood_check_original_ckpt/eval"
summaries = run_mc(cfg, "dqn", kpm_factory, n_reps=1, episodes_per_rep=50,
                    base_seed=eval_seed, mode="offline_eval", training=False,
                    results_dir=eval_dir, policy_factory=policy_factory, reward_mode="sla")
compliance = summaries[0].sla_compliance_all_slices if summaries else float("nan")
print(f"[ood-check] ORIGINAL 1x-load checkpoint evaluated under 2x-load config: sla_compliance_all_slices={compliance:.4f}")
print(f"[ood-check] eval log written to {eval_dir}")
