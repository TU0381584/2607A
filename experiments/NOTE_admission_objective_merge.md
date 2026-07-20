# Note: Objectives A and B collapse into one — proposal to merge

**Date:** 2026-07-20
**Status:** ADOPTED. Design formalized and validity-checked (see "Result" section
below) — `experiments/configs/saclb_admission_efficiency_v1.yaml` is now a real,
frozen, script-validated config. **Retraining and live-rig time are still NOT
authorized** — nothing below produces a trained policy or touches the rig.
**Full evidence trail:** `CAMPAIGN_LOG.md`, 2026-07-20 entries ("Objectives v3 (Design-only)" onward).

## The finding in one paragraph

The v3 handover's Objective A ("stochastic admission control under overload — does a
smart policy selectively reject low-value requests?") and Objective B ("resource
efficiency — does a smart policy avoid over-allocating?") were scoped as two
independent design tracks. Design-only investigation into A1 surfaced a real bug
(eMBB's admission ceiling was a permanent no-op in offline training — see CAMPAIGN_LOG,
now fixed in `c523e02`). Once fixed and the environment recalibrated to a genuine
overload regime, the diagnostic evidence shows **accept-all is the strongest naive
policy for raw SLA compliance**, not the weakest — riding the ceiling to cap simply
serves the most demand. Naive rejection (reject-all, or a crude threshold) *loses* on
compliance. So Objective A's premise — "a smart policy should selectively reject to
protect compliance" — doesn't hold in the direction the plan assumed.

What *does* differentiate accept-all from a smarter policy is the reward's
`beta*cost` term (eq. 9): accepting unconditionally racks up a congestion-proportional
cost penalty regardless of whether the accepted request was worth it. That is
Objective B's efficiency question. In other words: **the interesting policy tradeoff
in this environment is entirely about *when accepting is worth its cost*, not about
protecting compliance via rejection** — a single tradeoff, not two.

## Why this matters for scope

If A and B stay split, the natural next steps (A2/A3: baseline family + retraining
under a frozen overload regime; B2/B3: separate beta sweep + retraining under a frozen
efficiency regime) would very likely retrain against two environments that differ only
in which axis of the *same* tradeoff gets emphasized — real compute and rig time spent
twice for one underlying result.

## Proposed merged objective

**"Admission efficiency under overload"**: a single evaluation objective where the
metric of interest is SLA-weighted utility *net of allocation cost*, not raw
compliance and not raw allocated-PRB efficiency separately. Concretely:

- Retire the separate A1 "heterogeneous request classes with distinct resource
  demand/lifetime" ask — already a stop condition (frozen `AdmissionRequest` has no
  such fields; see CAMPAIGN_LOG). Keep the achievable version: per-slice value
  asymmetry via `SliceSpec.priority_weight`/`violation_penalty` (already
  config-driven, already differentiated urllc > embb > mmtc in the existing config).
- Use the validated post-fix overload calibration (`backlog_capacity=1000,
  oversub_of_cap=1.2, Lmax=1000`, nominal/cap/floor at tens-of-units matching the
  papers' own "% of B=100" convention) as the frozen environment for both what were
  previously "A" and "B."
- Single beta sweep (Objective B's original ask), evaluated on THIS corrected
  environment — the earlier retroactive beta-sweep is void (computed against both the
  scale bug and the sd bug; see CAMPAIGN_LOG).
- Baseline family: accept-all, reject-all, static-threshold (tuned honestly),
  static_oracle if desired — same as originally planned for both A2 and B2, now one
  shared set instead of two.
- Headline exhibit: SLA-weighted utility (or net reward) vs. allocated capacity,
  per arm — showing where each baseline sits on the compliance/cost frontier and
  whether a learned policy dominates all of them, not two separate scoreboards.

## What this does NOT change

- Still requires retraining (offline, 3 seeds x 300 episodes per arm) and a live
  confirmation subset to produce real numbers — out of scope for a design-only
  session, same as before.
- Objective C (QoE-vs-QoS dissociation via the ABR client) is unaffected by this
  finding and remains a separate, later question.
- S1's already-published live-campaign results are unaffected (this bug only lived
  in the offline synthetic training path).

## Result (2026-07-20, executed)

Formalized as real, reusable, script-generated artifacts (not scratch/in-memory
diagnostics):
- `experiments/configs/saclb_admission_efficiency_v1.yaml` — the frozen offline
  config (nominal/floor/cap at tens-of-units, Lmax=1000, existing per-slice value
  asymmetry unchanged).
- `experiments/scripts/admission_efficiency_env.py` — the non-frozen factory
  (`backlog_capacity=1000.0`, `oversub_of_cap=1.2`, real `sd_for_slice` wiring),
  the single source of truth for building this environment going forward.
- `experiments/scripts/run_admission_efficiency_baselines.py` — runs accept_all,
  reject_all, and the framework's own `LbOnlyHeuristic` (static-threshold, unmodified)
  against the frozen config and writes a validity report.

**Validity check: CORRECTED 2026-07-20 (see CAMPAIGN_LOG for the full trace).**
The original run below used a strict `per_slice_sla_margin > 0` compliance check,
which undercounts -- margins in this environment sit almost exactly at the 0.0
boundary the vast majority of the time. Using the framework's own correct
definition (`per_slice_compliant`, non-strict `queue_len_norm <= 1.0` -- the same
field every other figure/table in this project reads), **all three baselines
show 100.0% compliance on every slice.** The table below is retained for the
record with its error stated plainly, not silently edited away:

| Policy | eMBB compliant (WRONG, strict check) | URLLC (WRONG) | mMTC (WRONG) | Mean reward (unaffected, correct) |
|---|---|---|---|---|
| accept_all | 16.1% | 51.2% | 70.3% | -0.382 |
| static_threshold | 15.7% | 21.4% | 44.4% | -0.147 |
| reject_all | 9.5% | 9.8% | 18.0% | +0.025 |

**Corrected reading: compliance does NOT differentiate any policy here -- everyone
achieves it. Reward is the ONLY axis that honestly differentiates.** This
strengthens rather than weakens this note's core thesis: the interesting tradeoff
in this environment was never "does the policy protect compliance," it's purely
"how much reward/cost efficiency does the policy achieve while compliance takes
care of itself." The objective's headline metric should be reward (or a
cost-normalized efficiency metric), not raw compliance.

## Remaining ask

Retraining (offline, 3 seeds x 300 episodes/arm) against this frozen config, and
eventually a live confirmation subset, are the natural next steps to produce real
learned-arm evidence -- both still require separate authorization, not granted by
this note.
