# Admission-efficiency offline training: 3-seed results summary

**Generated:** 2026-07-20. Config: `experiments/configs/saclb_admission_efficiency_v1.yaml`.
Training driver: `experiments/scripts/train_offline_admission_efficiency.py`.
300 episodes x 3 seeds (256, 257, 258) x 4 arms (DQN/A2C x SLA/QoE reward).
All 12 runs completed with zero errors (`experiments/results/admission_efficiency_offline/log_*.log`).

Compliance is computed from `reward_breakdown["per_slice_compliant"]` (the framework's
own definition, `queue_len_norm <= 1.0`, non-strict) -- see CAMPAIGN_LOG.md's
2026-07-20 correction entry for why an earlier version of this session's own
scripts got this wrong with a stricter check.

## Final performance (last 30 of 300 episodes, mean across 3 seeds)

| Arm | Mean reward | Inter-seed CV | eMBB compliant | URLLC compliant | mMTC compliant | Mean cost |
|---|---|---|---|---|---|---|
| dqn_sla | 3.7694 ± 0.0613 | 1.63% | 100.0% | 100.0% | 100.0% | 1.3453 |
| a2c_sla | 3.0415 ± 0.0403 | 1.32% | 100.0% | 100.0% | 100.0% | 2.0927 |
| dqn_qoe | 0.0143 ± 0.0004 | 2.92% | 100.0% | 100.0% | 100.0% | 0.0223 |
| a2c_qoe | 0.0186 ± 0.0001 | 0.64% | 100.0% | 100.0% | 100.0% | 0.0009 |

(SLA-reward and QoE-reward arms are on different reward scales -- eq.2 vs eq.9 --
not directly comparable to each other, same convention as the original S1 campaign.)

## Reading

- **Compliance is saturated at 100% for every arm**, matching what the corrected
  scripted-baseline validity check already showed (accept_all/reject_all/
  static_threshold all reach 100% too) -- compliance is not the axis that
  differentiates policies in this environment at this load level.
- **Reward/cost is the real differentiator, and it shows real, sensible structure:**
  - `dqn_sla` beats `a2c_sla` on reward (3.77 vs 3.04) while paying LESS cost
    (1.35 vs 2.09) -- it learned genuine selectivity, not just "accept more."
  - Under QoE reward, both arms converge toward heavy rejection (mean cost
    0.02 and 0.001 respectively, near the reject-all extreme) -- consistent
    with the earlier beta-sensitivity finding that cost dominates reward at
    the current beta=0.2 placeholder. `a2c_qoe` edges out `dqn_qoe` on reward
    (0.0186 vs 0.0143) by rejecting slightly more aggressively.
- **Convergence quality**: inter-seed CV is 0.64-2.92% across the 4 arms.
  `a2c_qoe` (0.64%) and `a2c_sla` (1.32%) meet the original campaign's
  ~1.5% benchmark; `dqn_sla` (1.63%) and especially `dqn_qoe` (2.92%) run
  somewhat over it -- stated plainly, not smoothed over. Training curves
  themselves (`experiments/plots/out/admission_efficiency_fig1_training_convergence.png`)
  show all 4 arms visibly plateaued by episode ~150-200, so the higher CV
  looks like seed-to-seed noise around a shared plateau rather than
  non-convergence, but this has not been further investigated (e.g. whether
  more episodes or a different batch size would tighten it).

## What this is, and is not

This is single-config, 3-seed OFFLINE training evidence only. It is NOT yet:
- Compared against the tuned static-threshold/accept-all/reject-all baselines
  on the SAME held-out evaluation protocol (the baseline validity check used
  10 episodes/seed with scripted, non-learning policies; a proper comparison
  needs the trained checkpoints evaluated the same way).
- Confirmed live (no rig time used in this entire admission-efficiency
  workstream to date).
- Using a swept beta (still the 0.2 placeholder -- see
  `experiments/scripts/beta_sensitivity_probe.py`'s finding that beta's exact
  value in a 0.05-1.0 range doesn't change which extreme (accept vs reject)
  wins, suggesting the interesting beta range is below what was tested here).
