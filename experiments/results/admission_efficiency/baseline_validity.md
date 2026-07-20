# Admission-efficiency baseline validity check

Config: `experiments/configs/saclb_admission_efficiency_v1.yaml` (backlog_capacity=1000.0, oversub_of_cap=1.2 -- see admission_efficiency_env.py)
Seeds: [256, 257, 258], episodes/seed: 10

Compliance = `reward_breakdown['per_slice_compliant']` (queue_len_norm <= 1.0, non-strict) -- the SAME field mc_runner.py's episode_sla_compliance_by_slice uses, matching every other figure/table in this project. (Corrected 2026-07-20: an earlier version of this script used a strict per_slice_sla_margin>0 check, which undercounts -- margins in this environment sit almost exactly at the 0.0 boundary most of the time.)

| Policy | Slice | Frac compliant | Block rate | n samples |
|---|---|---|---|---|
| accept_all | embb | 1.000 | 0.000 | 1800 |
| accept_all | urllc | 1.000 | 0.000 | 1800 |
| accept_all | mmtc | 1.000 | 0.000 | 1800 |
| reject_all | embb | 1.000 | 1.000 | 1800 |
| reject_all | urllc | 1.000 | 1.000 | 1800 |
| reject_all | mmtc | 1.000 | 1.000 | 1800 |
| static_threshold | embb | 1.000 | 0.503 | 1800 |
| static_threshold | urllc | 1.000 | 0.503 | 1800 |
| static_threshold | mmtc | 1.000 | 0.508 | 1800 |

| Policy | Mean per-step reward |
|---|---|
| accept_all | -0.3820 |
| reject_all | 0.0251 |
| static_threshold | -0.1465 |

## Validity verdict
**FAIL** -- all policies saturated at the same extreme on every slice; no differentiation. Design needs another iteration.
