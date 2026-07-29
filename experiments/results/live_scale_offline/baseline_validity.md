# live_scale_offline_env baseline validity check (Stage 5)

Config: `experiments/configs/saclb_campaign_v2.yaml` (REAL cap=12/4/3, nominal=3/2/2 -- not rescaled) + mean_offered_ratio={'embb':0.15,'urllc':0.05,'mmtc':0.05} (real live-observed demand) + backlog_capacity=2000.0
Seeds: [256, 257, 258], episodes/seed: 10

| Policy | Slice | Frac compliant | Block rate | n samples |
|---|---|---|---|---|
| accept_all | embb | 0.263 | 0.000 | 1800 |
| accept_all | urllc | 0.432 | 0.000 | 1800 |
| accept_all | mmtc | 0.319 | 0.000 | 1800 |
| reject_all | embb | 0.139 | 1.000 | 1800 |
| reject_all | urllc | 0.282 | 1.000 | 1800 |
| reject_all | mmtc | 0.296 | 1.000 | 1800 |
| static_threshold | embb | 0.144 | 0.503 | 1800 |
| static_threshold | urllc | 0.282 | 0.549 | 1800 |
| static_threshold | mmtc | 0.292 | 0.543 | 1800 |

| Policy | Mean per-step reward |
|---|---|
| accept_all | -0.1466 |
| reject_all | -0.2866 |
| static_threshold | -0.2748 |

## Validity verdict
**PASS** -- policies show real, non-saturated, per-slice differentiation in SLA compliance (see table above).
