"""Control test: does doubling ClosedLoopKpmSource's mean_offered_ratio
(the REAL lever for offline demand -- see live_scale_offline_env.py,
calibrated from real single-UE probe measurements: embb=0.15,
urllc=0.05, mmtc=0.05) actually change the ORIGINAL (untouched, already
collapsing live at 6 UEs) checkpoint's offline behavior, unlike the
earlier arrivals.synthetic_arrivals_per_step attempt which touched a
completely different, unrelated knob (request-decision cadence, not the
KPM source's own demand model at all -- confirmed by reading
make_kpm_source_factory: it hardcodes MEAN_OFFERED_RATIO from
live_scale_offline_env, never reads cfg.arrivals at all).

mean_offered_ratio was calibrated for ONE UE per slice (matching the
3-UE/1-per-slice live anchor). Doubling it models 2 UEs per slice (6
UEs total), same linear scaling logic used everywhere else in this
project's own load-heterogeneity work.

No frozen file touched, no existing script edited -- monkeypatches
m6_run_experiment's own MEAN_OFFERED_RATIO module global (a plain
reassignment resolved at CALL time via Python's normal name lookup,
confirmed by reading make_kpm_source_factory's closure) from this new,
standalone script before calling its own already-established
make_kpm_source_factory/run_mc pattern -- same technique
m2_single_agent_eval_only_from_checkpoint.py already uses to reuse
m2_run_experiment's pieces without editing it.
"""
import sys
sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

import m6_run_experiment as m6  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_mc  # noqa: E402

ORIG_CKPT = "/home/kmanojp/oranslice_rig/experiments/results/m8_live_anchor/offline_train/single_agent_dqn/seed900/train/dqn/offline_train/rep_0/checkpoint.pt"
CONFIG_PATH = "qoe_oran_framework/configs/saclb_offline_live1gnb.yaml"

cfg = m6.load_saclb_config(CONFIG_PATH)
sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}

def policy_factory(_s, ckpt=ORIG_CKPT):
    p = build_policy("dqn", cfg)
    p.load_checkpoint(ckpt)
    return p

def run_check(label, mean_offered_ratio, out_dir):
    m6.MEAN_OFFERED_RATIO = mean_offered_ratio
    kpm_factory = m6.make_kpm_source_factory(cfg, sd_for_slice)
    eval_seed = m6.EVAL_SEED_OFFSET + 900
    summaries = run_mc(cfg, "dqn", kpm_factory, n_reps=1, episodes_per_rep=50,
                        base_seed=eval_seed, mode="offline_eval", training=False,
                        results_dir=out_dir, policy_factory=policy_factory, reward_mode="sla")
    compliance = summaries[0].sla_compliance_all_slices if summaries else float("nan")
    print(f"[{label}] mean_offered_ratio={mean_offered_ratio} sla_compliance_all_slices={compliance:.4f} eval_dir={out_dir}")

BASE_DIR = "/home/kmanojp/oranslice_rig/experiments/results/m33_realistic_demand_check"
run_check("1x (original calibration)", {"embb": 0.15, "urllc": 0.05, "mmtc": 0.05}, f"{BASE_DIR}/orig_ckpt_1x/eval")
run_check("2x (models 2 UEs/slice)", {"embb": 0.30, "urllc": 0.10, "mmtc": 0.10}, f"{BASE_DIR}/orig_ckpt_2x/eval")
