"""Paper #5, M2: net-net GAT + CTDE-MARL extension of paper #4's
single-agent, single-gNB admission control.

New subpackage, additive to qoe_oran_framework/ -- does not modify any
existing file in the parent package. Confirmed absent before this (see
../../../PAPER5_STATUS.md and experiments/REWORK_PLAN.md section 2): no
graph-neural-network code, no centralized-training/decentralized-execution
structure, no federated-learning aggregation existed anywhere in this
repository before this subpackage.

Per docs/PAPER5_M1_recalibration.md's conclusion, all training/evaluation
here runs in the OFFLINE environment as a live-anchored STRESS environment
for the contention regime -- not as a claim that offline compliance
predicts live compliance (M1 found it does not, for the single-agent case,
and there is no reason to assume a different architecture changes that).
The single live gNB paper #4 validated remains the only real-hardware
anchor; nothing here is evaluated live.
"""
