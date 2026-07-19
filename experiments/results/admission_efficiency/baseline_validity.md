# Admission-efficiency baseline validity check

Config: `experiments/configs/saclb_admission_efficiency_v1.yaml` (backlog_capacity=1000.0, oversub_of_cap=1.2 -- see admission_efficiency_env.py)
Seeds: [256, 257, 258], episodes/seed: 10

| Policy | Slice | Mean margin | Frac compliant | Block rate | n samples |
|---|---|---|---|---|---|
| accept_all | embb | 0.063 | 0.161 | 0.000 | 1800 |
| accept_all | urllc | 0.136 | 0.512 | 0.000 | 1800 |
| accept_all | mmtc | 0.209 | 0.703 | 0.000 | 1800 |
| reject_all | embb | 0.040 | 0.095 | 1.000 | 1800 |
| reject_all | urllc | 0.048 | 0.098 | 1.000 | 1800 |
| reject_all | mmtc | 0.063 | 0.180 | 1.000 | 1800 |
| static_threshold | embb | 0.048 | 0.157 | 0.503 | 1800 |
| static_threshold | urllc | 0.071 | 0.214 | 0.503 | 1800 |
| static_threshold | mmtc | 0.114 | 0.444 | 0.508 | 1800 |

| Policy | Mean per-step reward |
|---|---|
| accept_all | -0.3820 |
| reject_all | 0.0251 |
| static_threshold | -0.1465 |

## Validity verdict
**PASS** -- policies show real, non-saturated, per-slice differentiation in SLA compliance (see table above).
