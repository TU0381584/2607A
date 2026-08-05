# Paper #5, M1 — offline/live SLA-margin recalibration attempt

**Question:** can refitting the offline environment's `backlog_capacity`
and temporal-dynamics parameters (drift coefficient, noise volatility, and
a new autocorrelation mechanism) against the live per-slice SLA-margin
*distribution* (not just its mean) re-establish offline/live compliance
rank-correlation on the six single-gNB `dqn_sla` checkpoints (training
seeds 256-261), the limitation paper #4 flags in Section III-C?

This directly follows on from `docs/STAGE12_offline_online_gap.md` (which
root-caused the gap to the offline demand process's temporal *structure*,
not just its mean) and `docs/STAGE13_recalibration_attempt.md` (which swept
`backlog_capacity` alone via an accept-all/reject-all differentiability
criterion and found no fix, rho=0.10->0.23, both indistinguishable from
zero at n=6). Full context: `PAPER5_STATUS.md` section 3.

## Method

1. **Extracted live per-step traces** (`experiments/scripts/
   m1_extract_live_traces.py`) from all 6 checkpoints' live omega logs:
   checkpoint 256 from the full `live_campaign_v2/dqn_sla` arm (28 seeds,
   the same data paper #4's Table I uses), checkpoints 257-261 from
   `live_checkpoint_sensitivity/` (Stage 11's 21-episode protocol).
   41,940 per-slice `per_slice_sla_margin` readings pooled; 4.17% dropped
   as the known hardware-failure artifact Stage 13 documented (its own
   reported range was 3.5-8.5%, so this is consistent, not a new
   discovery). Live target: mean/std margin per slice = eMBB
   0.714/1.239, URLLC 0.745/0.643, mMTC 0.734/0.813. No live "backlog
   occupancy" field exists in the omega log schema (confirmed by direct
   inspection) -- that is purely an offline-simulator-internal state
   variable, so the fit target is the SLA-margin distribution (which the
   M1 brief itself specifies as the primary target), not a literal
   backlog-occupancy trace comparison.

2. **New, additive `RecalibratedClosedLoopKpmSource`**
   (`experiments/scripts/m1_recalibrated_kpm_source.py`), subclassing --
   not modifying -- the frozen `qoe_oran_framework.replay_kpm_source.
   ClosedLoopKpmSource`. Exposes `backlog_capacity` (already a parent
   constructor arg), `drift_coef` (hardcoded at 0.1 in the parent, not
   previously exposed at all), `offered_volatility` (already a parent
   arg), and a new `ar1_coef` giving the demand noise term AR(1)
   persistence at the same marginal variance -- a mechanism the parent
   does not have at all, aimed directly at Stage 12's "temporal
   structure, not mean" diagnosis. Verified bit-identical to the parent
   under default kwargs (20-step check, exact sample equality). Old
   behaviour stays reproducible by construction: this file does not
   touch `qoe_oran_framework/`.

3. **Grid search** (`experiments/scripts/m1_fit_recalibration.py`): 162
   combinations of backlog_capacity (6 values, 200-3200) x drift_coef (3)
   x offered_volatility (3) x ar1_coef (3), each scored on 3 representative
   checkpoints (256, 257, 258) x 6 short episodes, by standardized squared
   error between offline and live (mean, std) of per-slice margin, summed
   over slices. ~5 minutes wall-clock (cheap: this evaluates FROZEN
   checkpoints only, no training).

4. **Held-out evaluation** (`experiments/scripts/m1_run_held_out_eval.py`):
   the exact Stage 12/13 protocol reused unmodified -- 10 fresh seeds
   (5001-5010) never used for training or live eval, 10 episodes/seed =
   100 held-out episodes/checkpoint, greedy, all 6 frozen checkpoints, run
   under both the original defaults (bc=2000, drift=0.1, vol=0.04, ar1=0,
   i.e. bit-identical to the frozen class) and the grid search's best-fit
   config.

## Result

**Grid search: the loss surface is flat and the live target is
unreachable within this parameter family.** All top-5 configs cluster in
loss 1.542-1.562 (a ~1% spread) -- changing any of the 4 parameters barely
moves the fit. The best config found (`backlog_capacity=3200,
drift_coef=0.1, offered_volatility=0.04, ar1_coef=0.0` -- notably,
autocorrelated noise never wins; every top-5 entry has ar1 in {0.0, 0.5}
never dominating) still leaves eMBB's offline margin mean at -0.09,
nowhere near live's +0.71. **Adding temporal autocorrelation, the
mechanism Stage 12 specifically flagged as the likely deeper issue, did
not help.**

**Held-out evaluation: baseline reproduces Stage 12 exactly (9%, 9%, 14%,
13%, 20%, 9% for seeds 256/257/258/259/260/261 respectively) -- confirms
the harness is correct before trusting the new number.** Under the
recalibrated env (bc=3200), compliance came back **bit-for-bit identical
per checkpoint: 9%, 9%, 14%, 13%, 20%, 9%.** This is not a bug (verified:
the underlying per-step margin distribution for seed 258 genuinely shifted
between the two runs, mean -0.52 -> -0.87, std 0.58 -> 0.93 -- the
environment is behaving differently) -- it means the recalibration moved
the margin distribution without moving any episode across the
compliant/non-compliant boundary. The episodes that fail, fail by enough
margin already that this class of parameter change doesn't flip them.

**Spearman rank correlation: rho=0.097 (p=0.855), identical before and
after recalibration to three decimal places.** Both indistinguishable
from zero at n=6 -- not a small improvement like Stage 13's 0.10->0.23,
literally no movement at all. Figure:
`paper_conf/figures/m1_offline_live_correlation.pdf` (two-panel scatter,
offline vs. live compliance, ARM_STYLE's dqn_sla color/marker, per-checkpoint
seed labels; both panels are visually identical, which is itself the
finding).

## Honest conclusion

**M1 does not close the offline/live gap, and does so more starkly than
Stage 13: not merely "no significant improvement" but zero measurable
movement in either the fit loss or the resulting rank correlation.**
This rules out, with more precision than Stage 13 had, that the gap is a
`backlog_capacity`/demand-temporal-structure calibration problem within
this model family -- extending the grid or adding one more knob to this
same mechanism (mean-reverting-walk demand -> capped backlog -> margin) is
unlikely to succeed either, since the loss surface is already flat across
a wide range and even the best-fit point produces literally unchanged
compliance decisions.

Stage 13's two named unexplored hypotheses remain unexplored, and this
investigation adds a plausible reason to prioritize the first over
continuing on this axis:

1. **The offline loss/`bler` channel is entirely derived from backlog
   fraction** (`bler = 0.02 + 0.3*backlog_frac`) -- since `queue_margin`
   and `loss_margin` are both downstream of the same single backlog state
   variable, no combination of demand-side parameters (which is all M1
   touched) can decouple them. This is now a stronger candidate: fixing
   only the demand generator cannot fix a margin formula whose two
   components are not independent to begin with.
2. **`mean_offered_ratio` may not represent effective demand under a
   commanded ceiling** (point-probe vs. full-episode measurement) --
   untouched by M1, which only reweighted noise around the same fixed
   point-probe means.

## What this means for paper #5

Nothing in `paper_conf/main.tex` changes -- this is exploratory work for a
future paper, not paper #4. For M2-M4 (not yet scoped), this result argues
for treating "offline pre-screening of multi-agent/GAT-CTDE checkpoints
before live evaluation" as **not yet trustworthy** -- any new arms trained
under this same offline environment family should expect the same live
unpredictability paper #4 already reports for the single-agent case, not
an implicit assumption that a new architecture fixes a sim-to-real gap
that traces to the demand/loss model, not to single-agent-ness.

## Acceptance status

- [x] Reused the established Stage 11 live-compliance numbers rather than
      re-deriving them, avoiding a second, possibly-divergent number for
      an already-audited quantity.
- [x] Verified the harness against Stage 12's exact baseline numbers
      before trusting the new recalibrated numbers.
- [x] No frozen `qoe_oran_framework/` source modified; old behaviour
      reproducible (verified bit-identical under default kwargs).
- [x] No checkpoint retrained -- only the evaluation environment changed,
      isolating the question M1 asked from Stage 13's train-time question.
- [x] Reported a starker negative result plainly (identical rho, not a
      small non-significant change) rather than searching the grid further
      until something looked better by chance.
- [x] Named a sharper, motivated next hypothesis (bler/backlog coupling)
      rather than leaving "investigate further" vague.
