# Static-threshold (LbOnlyHeuristic) honest tuning sweep

Tuned on held-out seed 950 (disjoint from eval seeds [256, 257, 258]); 10 episodes/seed, 4x4=16 grid points.

## Sweep (on tuning seed only)

| utilization_threshold | capacity_margin | mean reward | mean compliance |
|---|---|---|---|
| 0.7 | 0.7 | -0.0780 | 0.185 | **<- winner**
| 0.8 | 0.7 | -0.0783 | 0.187 |
| 0.9 | 0.7 | -0.0783 | 0.188 |
| 0.97 | 0.7 | -0.0783 | 0.188 |
| 0.7 | 1.15 | -0.1043 | 0.215 |
| 0.7 | 1.0 | -0.1047 | 0.204 |
| 0.7 | 0.85 | -0.1054 | 0.218 |
| 0.8 | 0.85 | -0.1141 | 0.229 |
| 0.9 | 0.85 | -0.1141 | 0.230 |
| 0.97 | 0.85 | -0.1141 | 0.230 |
| 0.8 | 1.15 | -0.1373 | 0.298 |
| 0.8 | 1.0 | -0.1387 | 0.273 |
| 0.9 | 1.0 | -0.1525 | 0.310 |
| 0.97 | 1.0 | -0.1525 | 0.310 |
| 0.9 | 1.15 | -0.1717 | 0.366 |
| 0.97 | 1.15 | -0.1929 | 0.448 |

## Winner: utilization_threshold=0.7, capacity_margin=0.7

Honest, held-out performance on eval seeds [256, 257, 258] (NOT the seed used to pick these parameters):

| Slice | Mean margin | Frac compliant |
|---|---|---|
| embb | 0.040 | 0.099 |
| urllc | 0.055 | 0.125 |
| mmtc | 0.077 | 0.258 |

Mean reward on eval seeds: -0.0737

(For comparison, default params utilization_threshold=0.97, capacity_margin=1.0 scored -0.1525 mean reward on the TUNING seed -- the honest sweep is not just picking defaults.)
