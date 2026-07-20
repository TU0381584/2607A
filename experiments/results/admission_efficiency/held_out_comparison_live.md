# Held-out admission-efficiency comparison

Seeds: [960, 961, 962] (fresh -- distinct from training seeds 256/257/258 and the static-threshold tuning seed 950). 20 episodes/seed.
Learned arms use frozen weights from their seed256 checkpoint, select_action(training=False) (greedy).

## SLA-reward group

| Arm | eMBB compliant | URLLC compliant | mMTC compliant | Mean reward |
|---|---|---|---|---|
| accept_all | 28.3% | 17.9% | 12.2% | -6.7264 |
| reject_all | 0.7% | 0.2% | 0.7% | -15.4308 |
| static_threshold | 1.4% | 1.8% | 1.6% | -12.4524 |
| dqn_sla | 28.3% | 17.9% | 10.8% | -6.7641 |
| a2c_sla | 28.3% | 17.9% | 12.2% | -6.7264 |

## QOE-reward group

| Arm | eMBB compliant | URLLC compliant | mMTC compliant | Mean reward |
|---|---|---|---|---|
| accept_all | 28.3% | 17.9% | 12.2% | -0.4986 |
| reject_all | 0.7% | 0.2% | 0.7% | -0.4788 |
| static_threshold | 1.4% | 1.8% | 1.6% | -0.4882 |
| dqn_qoe | 25.8% | 12.9% | 10.9% | -0.5008 |
| a2c_qoe | 0.7% | 0.2% | 0.7% | -0.4788 |

