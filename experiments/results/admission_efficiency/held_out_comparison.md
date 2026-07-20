# Held-out admission-efficiency comparison

Seeds: [960, 961, 962] (fresh -- distinct from training seeds 256/257/258 and the static-threshold tuning seed 950). 20 episodes/seed.
Learned arms use frozen weights from their seed256 checkpoint, select_action(training=False) (greedy).

## SLA-reward group

| Arm | eMBB compliant | URLLC compliant | mMTC compliant | Mean reward |
|---|---|---|---|---|
| accept_all | 100.0% | 100.0% | 100.0% | 3.0008 |
| reject_all | 100.0% | 100.0% | 100.0% | 0.0000 |
| static_threshold | 100.0% | 100.0% | 100.0% | 1.5995 |
| dqn_sla | 100.0% | 100.0% | 100.0% | 3.8999 |
| a2c_sla | 100.0% | 100.0% | 100.0% | 3.0008 |

## QOE-reward group

| Arm | eMBB compliant | URLLC compliant | mMTC compliant | Mean reward |
|---|---|---|---|---|
| accept_all | 100.0% | 100.0% | 100.0% | -0.3894 |
| reject_all | 100.0% | 100.0% | 100.0% | 0.0215 |
| static_threshold | 100.0% | 100.0% | 100.0% | -0.0770 |
| dqn_qoe | 100.0% | 100.0% | 100.0% | 0.0215 |
| a2c_qoe | 100.0% | 100.0% | 100.0% | 0.0227 |

