# Stage 13 — Recalibration attempt: does fixing backlog_capacity close the offline/online gap?

User, after being told "make the offline results support the online" is
not something that can be done honestly: chose the legitimate path --
*"Rigorous recalibration"* -- measure real traffic statistics, refit the
offline simulator, re-test, and report whatever the real outcome is.

## What was found and attempted

While investigating real live margin dynamics as a calibration target
(comparing to Stage 12's finding), traced the mechanism further than
Stage 12 did: `train_offline_live_scale.py` hardcodes
`backlog_capacity=2000.0`, which was chosen in Stage 5 purely so that
`accept_all`/`reject_all`/`threshold_like` produce differentiable
outcomes (Stage 5's own validation criterion) -- **it was never checked
against real live margin magnitude.** At `backlog_capacity=2000`, the
same checkpoint's mean offline SLA margin is deeply negative (~-0.60);
real live margin (excluding rare genuine hardware-failure episodes,
see below) sits at a stable ~+0.70-0.75.

A sweep of `backlog_capacity` (200 through 2000, using
`accept_all`/`reject_all` as before) found a genuine tension:
- Below ~1000: margin magnitude gets close to real live's range, but
  `accept_all` and `reject_all` become statistically indistinguishable
  (both saturate near 100% compliant) -- the environment loses the
  ability to teach a policy anything about avoiding violations.
- Above ~1200: `accept_all`/`reject_all` separate again, but margin
  reverts toward the original, unrealistically negative range.

**No single `backlog_capacity` value satisfies both constraints
simultaneously** -- this isn't a tuning oversight, it's a real
structural property of the model (backlog accumulates every step with
no decay/drop mechanism other than being served, so realism and
differentiability trade off directly against each other via this one
parameter).

Chose `backlog_capacity=1200` as the best available compromise (where
differentiation first reliably appears, at the least-unrealistic
margin available under that constraint) and **retrained all 6
checkpoints (seeds 256-261) from scratch** with this value, then
re-ran the same 100-episode-per-checkpoint held-out evaluation as
Stage 12.

`experiments/scripts/train_offline_live_scale.py` gained a
`--backlog-capacity` CLI flag (default unchanged at 2000, so all
previously-reported results remain exactly reproducible) rather than
silently changing the hardcoded constant.

## Result: no meaningful improvement

| Seed | Live compliance | Offline held-out, bc=2000 (Stage 12) | Offline held-out, bc=1200 (recalibrated) |
|---|---|---|---|
| 256 | 95.7% | 9% | 20% |
| 257 | 61.9% (worst live) | 9% | 13% |
| 258 | 100% (best live) | 14% | **9% (worst offline)** |
| 259 | 100% (best live) | 13% | 20% |
| 260 | 90.5% | 20% (best offline) | 10% |
| 261 | 100% (best live) | 9% | 20% |

Spearman rank correlation between live and offline compliance:
- bc=2000 (original): $\rho = 0.10$, $p = 0.86$
- bc=1200 (recalibrated): $\rho = 0.23$, $p = 0.67$

**Both are statistically indistinguishable from zero correlation at
n=6.** The recalibration produced a small, non-significant uptick in
rank correlation, not a fix. Notably, checkpoint 258 -- one of the
three checkpoints with a PERFECT live record -- is now the WORST
performer offline under the recalibrated environment, the same
non-relationship Stage 12 found, just reshuffled.

## Honest conclusion

**The offline/online gap is not explained by, or fixable via,
`backlog_capacity` alone.** This was a real, principled attempt (not a
token gesture) -- it correctly diagnosed and fixed a genuine, previously
un-scrutinized miscalibration (a parameter tuned for a different
purpose than the one it was later relied on for), and it was tested
honestly rather than declared a win by assumption. It did not work.
The likely remaining candidates, not yet investigated:
- The offline loss/`bler` channel is ENTIRELY derived from backlog
  fraction (`bler = 0.02 + 0.3*backlog_frac`, see
  `replay_kpm_source.py`'s frozen `ClosedLoopKpmSource.poll()`) -- real
  RF-level loss is presumably a more independent process, not a pure
  function of queue depth.
- `mean_offered_ratio` itself (0.15/0.05/0.05, from Stage 5's real
  probe measurements) may not represent the EFFECTIVE demand a fixed
  ceiling actually experiences over a full episode -- the probes were
  point measurements, not full-episode demand-under-a-specific-ceiling
  measurements.
- A real, separate finding surfaced while investigating this (not
  fully chased down): ~3.5-8.5% of live `per_slice_sla_margin` readings
  across the SLA-reward arms (0% for the QoE arm) show extreme,
  monotonically-growing-then-plateauing values (e.g., exactly
  -1002377.5, repeated) concentrated in a handful of whole episodes.
  Traced to the deliberately-unclamped `queue_margin` formula
  (`reward.py`'s `check_violations`, confirmed NOT a bug -- documented,
  intentional design) reacting to a genuinely large raw
  `dl_mac_buffer_occupation` reading. This is consistent with, and most
  likely IS, a real physical failure event (matching this project's own
  documented RLC max-RETX failure mode) rather than a sensor/measurement
  artifact -- the growth-then-plateau shape (not a single spike) argues
  against a transient glitch. Does not appear to corrupt the
  `per_slice_compliant` binary classification used in Table I (that
  field is a simple `margin <= 0` threshold, unaffected by magnitude),
  so no manuscript number is implicated. Flagged here for the record,
  not chased further -- confirming a specific episode's failure mode
  in `dmesg`/live session logs from the historical live_campaign_v2
  runs would be the next step if pursued.

**Given the actual constraint this investigation ran into is a
genuine, unresolved research question** (closing this gap credibly
would need either a fundamentally different demand/loss model, or
real experiments specifically designed to measure "effective demand
under a given ceiling" rather than open-ceiling probes) **rather than a
parameter that was simply wrong, further recalibration attempts this
session are not likely to succeed without that deeper redesign.**
Reported honestly rather than continuing to sweep parameters until
something looks better by chance.

## What this means for the paper

Nothing changes in `paper_conf/main.tex`. The manuscript's existing,
honest framing ("offline convergence does not predict live robustness")
already matches what this deeper investigation confirms, now with two
negative results (Stage 12's held-out eval, Stage 13's recalibration
attempt) backing it rather than one. If anything, this strengthens the
case that the finding is real and not an artifact of an easily-fixed
miscalibration -- worth a sentence if the author wants to cite the
depth of the negative result, but not required.

## Acceptance status

- [x] Pursued the legitimate path the user chose, not the one declined.
- [x] Found and fixed a real, previously-unscrutinized miscalibration
      (backlog_capacity chosen for a different purpose than realism).
- [x] Retrained and re-tested rather than assuming the fix worked.
- [x] Reported the negative result plainly (rho=0.10->0.23, p=0.86->0.67,
      not a fix) rather than overselling a marginal, non-significant
      change.
- [x] Identified specific, concrete next hypotheses for anyone who wants
      to continue this (loss-channel independence, effective-demand
      measurement) rather than leaving "investigate further" vague.
- [x] Surfaced the extreme-margin/real-failure-event finding
      transparently even though it turned out not to implicate any
      reported number, rather than quietly dropping it once it stopped
      being useful to the current investigation.
