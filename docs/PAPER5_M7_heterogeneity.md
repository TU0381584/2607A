# Paper #5 M7: FedProx-under-heterogeneity investigation

Status: **complete. FedProx-under-heterogeneity: null result with a
verified mechanism (not a bug and not an unexplained absence of
effect). Collapse-reliability characterization (M7's other half, per
the original roadmap): complete, GAT-CTDE's N=19 collapse rate now
resolved to 46.0% [28.6%, 65.1%] across 21 independent seeds.**

## What M7 asks

Section VI of the paper originally assumed "our topology treats every
gNB as an interchangeable peer with no client heterogeneity for
FedProx's correction to address," so no FedProx campaign was run in
M3. M6 corrected that belief: every M2/M3/M4 seed already has implicit,
uncontrolled per-gNB load heterogeneity (`ClosedLoopKpmSource`'s
auto-generated `[0.6, 1.4]` multiplier), and M6 added a
`gnb_load_multiplier_mode` override that makes a genuine
homogeneous-vs-heterogeneous comparison possible for the first time.
M7 asks the question that opened up: does FedProx's proximal term earn
a measurable benefit specifically under genuine heterogeneity (a
**heterogeneity dividend**) that it does not show under homogeneous
load?

## Infrastructure

`m3_run_experiment.py` gained the same `gnb_load_multiplier_mode`
parameter M6 added to `m2_run_experiment.py`/`m6_run_experiment.py`
(`make_kpm_source_factory`, `run_fl_arm`, CLI) so the federated training
path can run under a controlled load mode. Regression-checked (seed
8801, 5 train/3 eval episodes, `gnb_load_multiplier_mode="default"`):
train and eval omega logs byte-identical to the pre-change script.
Verified the override reaches the environment: `"homogeneous"` vs.
`"default"`, otherwise identical arguments, 612/612 compared train-log
lines differ.

New `experiments/scripts/m7_fedprox_heterogeneity.py`: sweeps
`{fedavg, fedprox at each --fedprox-mu value} x {homogeneous,
heterogeneous}`, DP noise fixed at $\sigma=0$ throughout (isolates the
heterogeneity/FedProx question from the already-characterised privacy
cost, matching M3's own discipline), same 300-train/50-eval episode
budget as M2/M3 for direct comparability, resumable the same way
`m3_privacy_sweep.py` is.

## Pilot 1: mu=0.01 -- bit-identical to FedAvg

3 seeds (900-902), both load modes, `fedprox_mu=0.01`. Result: FedAvg
and FedProx produce **bit-identical eval-time behaviour in every
cell** -- `mean_reward_per_step` matching to full float precision
(e.g. `13.572239174145839` exactly), identical `mmtc_blocks`/
`total_blocks` counts, in all 6 (load mode, seed) combinations.

Checked whether this meant `fedprox_mu` was silently not being applied
(a wiring bug) before trusting it as a real null: `fedprox_mu=0.01`
was correctly recorded in the checkpoint metadata, and the checkpoint
**weights themselves differ meaningfully** between the FedAvg and
FedProx runs on the same seed (up to 0.09-0.27 max-abs difference
across different weight tensors, checked directly with `torch.equal`/
max-abs-diff, not assumed). So `mu` is genuinely being applied and
genuinely perturbing training -- the perturbation just never flips a
single eval-time argmax accept/reject decision, consistent with this
project's own repeated finding (M2, M6) that this problem's Q-value
gaps are typically large.

## Pilot 2: mu in {0.01, 0.1, 1.0} -- still bit-identical at every level

Extended `--fedprox-mu` to accept a list (swept the same way
`m3_privacy_sweep.py` sweeps `noise_multiplier`), re-ran with
`mu in {0.01, 0.1, 1.0}` (0.01 resumed from Pilot 1, 0.1/1.0 fresh),
both load modes, 3 seeds, full budget. A smoke test at 5/3 episodes had
shown mu=0.1/1.0 diverging from the FedAvg baseline, suggesting the
extended sweep would be informative -- but at the full 300/50-episode
budget, **every mu level reproduced the identical bit-for-bit result**:
`sla_compliance_all_slices` matching FedAvg to full float precision at
every seed, every load mode, every mu up to 1.0. The smoke-test
divergence at 5 episodes did not survive to a fully-trained (300
episode) checkpoint -- consistent with the mechanism found below
(short-budget/early-training divergence is real, but by convergence the
proximal term's contribution becomes too small to matter).

## Chasing the mechanism: two hypotheses tested, one confirmed

**Hypothesis 1 (grad-norm clipping): tested directly, falsified.**
`fl_ctde_policy.py`'s `train_step` calls
`nn.utils.clip_grad_norm_(..., max_norm=self.dp_clip_norm)`
unconditionally every step, even with DP noise off (`dp_clip_norm=1.0`
default, applied "to isolate the privacy cost" per
`m3_run_experiment.py`'s own design) -- a plausible mechanism for why a
large `mu`'s pull might get renormalized away. Exposed `--dp-clip-norm`
on the CLI (already an `m3.run_fl_arm` parameter, not previously
threaded through) and re-ran `mu=1.0`, both load modes, 3 seeds, with
`dp_clip_norm=100` (effectively unclipped for this problem's gradient
scale). **Result: still bit-identical to FedAvg in every cell.**
Clipping was not the mechanism.

**Hypothesis 2 (per-round local drift is simply too small): tested
directly, confirmed.** Rather than keep guessing with more full
65-70-minute campaign runs, wrote two small diagnostic probes
(`m7_gradnorm_probe.py`, `m7_gradnorm_probe2.py`) that instrument the
proximal term's actual gradient contribution directly:

- First probe artificially injected a 0.05-std-per-parameter
  perturbation (to make the effect measurable) and found the proximal
  gradient's norm comparable to or larger than the TD-loss gradient's
  at `mu>=1.0` (ratio 1.07 at mu=1.0, 10.7 at mu=10) -- proving the
  mechanism CAN matter at that magnitude of drift.
- Second probe measured the REAL drift with no artificial injection:
  monkeypatched `_local_loss` to log both terms on every real
  `train_step` call across 10 real training episodes (1,707 calls, 11
  real rounds, `mu=1.0`). **The proximal term's real, scaled
  contribution averaged 0.03% of the TD loss (mean ratio 0.000299,
  max 0.0058 even at the end of a round, the point of maximum
  accumulated drift).**

`fl_ctde_policy.py`'s `_aggregate_round` resets `global_snapshot` to
the fresh cross-client average every `local_steps_per_round` (default
50) `train_step` calls. Fifty local optimiser steps simply does not
give a client enough room to drift far from the broadcast point before
the next reset, regardless of `mu` or whether the underlying data is
heterogeneous -- **the proximal term has essentially nothing to correct
for at this round length**, which is the real, verified reason for the
null result, not a bug and not clipping.

## Conclusion

FedProx shows no measurable heterogeneity dividend in this setup,
**verified mechanistically rather than reported as an unexplained
absence of effect**: at `local_steps_per_round=50`, real per-round
client drift is roughly 300x too small (0.03% vs. a scale where it
would plausibly matter, based on Probe 1's artificial-injection
comparison) for the proximal term to meaningfully alter training,
independent of `mu` (tested 0.01-1.0, two orders of magnitude) and
independent of load heterogeneity (tested both). This is a genuine,
well-supported null, consistent with this paper's own standing
discipline of reporting what actually ran rather than what an
intuitive story would predict (the same discipline that produced the
M6 reward-margin null and the M2/M3 architecture-margin corrections).

**Not yet tested**: whether a much longer round (`local_steps_per_round`
well above 50, letting real drift accumulate further before the next
reset) would let a heterogeneity dividend emerge. This is a plausible,
cheap next step if the question is worth reopening, but it changes a
different variable (round length) than M7 set out to test
(heterogeneity mode), so it is flagged here rather than run without
being asked.

## Collapse-reliability characterization: resolved (Part 10 of docs/PAPER5_M6_topology.md)

The other half of M7 per the original roadmap: 6 more seeds
(2000-2005, disjoint from every prior sample), all three N=19
topologies, gat_ctde only. Combined across all three samples now
available (primary 900-911, replication 1000-1002, this extension):
**29/63 cells collapsed, 46.0%, seed-level bootstrap 95% CI
[28.6%, 65.1%]** (21 independent seeds; bootstrapped at the seed level
rather than the pooled-cell level since collapse status is correlated
across a given seed's three topologies). Neither of the two earlier
small samples (31%, 78%) was wrong -- both were honest small-sample
reads of a genuinely wide distribution, and the combined estimate
supersedes rather than contradicts them. Full detail in
`docs/PAPER5_M6_topology.md` Part 10.
