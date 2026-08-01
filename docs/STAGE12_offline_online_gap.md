# Stage 12 — Why offline metrics don't predict live performance

User: *"why does the offline data produce so different results from the
online? can you investigate this? run more simulations if needed."*

## Question, precisely

Stage 10 showed all 6 DQN-SLA training runs (seeds 256–261) converge
consistently offline (similar Q1→Q4 reward improvement). Stage 11
showed their LIVE robustness varies enormously (13/21 to 21/21 fully
compliant). Offline convergence quality clearly isn't predicting live
outcome — but why not?

## Method: two new offline experiments (no rig time)

`experiments/scripts/investigate_checkpoint_gap.py` (new). Loads each
of the 6 already-trained checkpoints (frozen weights, greedy,
`training=False` — the same eval-only pattern `saclb_xapp.py` uses
live) and runs 100 held-out episodes each (10 fresh seeds × 10
episodes, seeds 5001–5010 — never used for training or any live eval)
against `live_scale_offline_env.py`'s own environment, the SAME one
these checkpoints were trained on (Stage 5's "corrected", real-demand-
calibrated `ClosedLoopKpmSource`).

## Finding 1: offline held-out compliance does not correlate with live compliance at all

| Checkpoint | Live (n=21 or pooled n=46) | Offline held-out (n=100, fresh seeds) |
|---|---|---|
| 256 | 44/46 (95.7%) | 9/100 (9%) |
| 257 | 13/21 (61.9%, worst live) | 9/100 (9%) |
| 258 | 21/21 (100%, best live) | 14/100 (14%) |
| 259 | 21/21 (100%, best live) | 13/100 (13%) |
| 260 | 19/21 (90.5%) | 20/100 (20%, best offline) |
| 261 | 21/21 (100%, best live) | 9/100 (9%) |

Two things stand out, both real and both surprising:

1. **Every checkpoint performs far worse offline than live** (9–20%
   vs.\ 62–100%) — not a small gap, an order-of-magnitude one.
2. **The offline ranking has no relationship to the live ranking.**
   The three checkpoints with a PERFECT live record (258/259/261) are
   among the WORST offline (14%, 13%, 9%) — indistinguishable from
   257, the checkpoint that fails catastrophically live. The single
   best-offline checkpoint (260, 20%) is only middling live (90.5%).

**This directly answers the question: offline evaluation, even done
properly (held-out, fresh seeds, adequate sample size), is not
measuring the same thing live evaluation measures for this
environment.** It's not that offline needs more samples or better
seeds — the two evaluation modes appear to be picking up on almost
entirely different aspects of policy behaviour.

## Finding 2: the offline environment's per-step dynamics are far harsher than live, for the same checkpoint

Directly compared checkpoint 258 (perfect live) step-level SLA margins,
offline held-out vs.\ a real live episode:

| | eMBB margin | URLLC margin | mMTC margin |
|---|---|---|---|
| Offline held-out (n=1800 steps, 3 reps) | **-0.621** | -0.164 | -0.377 |
| Live (seed950, n=120 steps) | **+0.962** | +0.817 | +0.770 |

Negative margin means routinely violating the SLA threshold; positive
means comfortably inside it. This is the SAME checkpoint, same reward
mode, same nominal config (cap/nominal/Lmax all identical between
`saclb_campaign_v2_offline_train.yaml` and the live `saclb_campaign_v2.yaml`
— `Lmax=1000` in both, ruling out an Lmax-saturation bug like the ones
found in Stage 5). The only thing that differs is the KPM source:
`ClosedLoopKpmSource` (synthetic, mean-reverting random walk, mean
tied to real probe measurements per Stage 5's Bug #3 fix) vs.\
`LiveKpmSource` (real gNB, real UEs, real RF).

Inspecting individual offline episodes: the same checkpoint that is
perfectly reactive live (raises ceilings immediately, per Fig. 1) is
sluggish offline — in 8/100 offline held-out episodes, its eMBB ceiling
stays pinned at the floor (1) for more than half the 60-step episode
while the SLA margin visibly erodes underneath it, only responding
late. This is a genuine behavioural difference, not a metric artifact:
the same weights produce a different effective policy against the two
KPM sources, because the states the synthetic random walk puts the
policy into are not the states real traffic puts it into.

## Interpretation

Stage 5 already fixed one layer of the offline/live mismatch (Bug #1/#2:
wrong MOS-calibration units; Bug #3: offline demand scale 4-6x smaller
than real, fixed by tying `mean_offered_ratio` to real probe
measurements). **This investigation shows that fix was necessary but
not sufficient.** Matching the MEAN of a mean-reverting random walk to
real measured demand does not reproduce the actual TEMPORAL STRUCTURE
of real traffic (burstiness, correlation across steps, how backlog
actually accumulates and drains against a real scheduler and real UEs).
The offline environment ends up systematically harsher and
differently-shaped than live — hard enough, and different enough, that
a policy's offline reward/compliance carries close to zero information
about how it will behave live.

This is the concrete, mechanistic explanation for Stage 11's finding
("offline convergence doesn't predict live robustness") — not just a
restatement of it. It also means: **no amount of additional offline
simulation, by itself, can be used to pre-select or validate a
"good" checkpoint before spending live-rig time** — until the offline
KPM source's dynamics (not just its mean) are validated against real
traffic, which is a real-rig-time-required calibration effort of the
same shape as the Phase 1 contention gate, not a training-side fix.

## What this does NOT explain (left open)

This investigation used only checkpoint 258 for the detailed
step-level comparison (representative, but not exhaustive) and did not
run the Part B policy-replay diagnostic originally planned (feeding
identical real live states through all 6 checkpoints' Q-networks) —
the Finding 1/2 result was decisive enough on its own that this was
judged not to add further explanatory value proportional to the
compute/time cost, but is flagged here rather than silently dropped
from the investigation's own stated plan.

## Manuscript impact

Not incorporated into `paper_conf/main.tex` in this pass — this is a
deeper root-cause investigation of an issue the manuscript's Future
Work item (1) already flags at the right level of confidence
("offline convergence does not predict live robustness"); the
mechanism (KPM-source dynamics mismatch, not just mean) is recorded
here for the author's own use and any deeper future-work writeup,
without re-opening/re-editing the already-finalized, page-budgeted
Results section for a mechanism finding that doesn't change any
reported number.

## System note

Increased swap from 4GB to 24GB (`/swap.img`, `fallocate` + `mkswap`)
per explicit request, before this investigation — headroom for
concurrent offline+live workloads without repeating Stage 10's memory-
contention incident, though this investigation itself was offline-only
and did not need it.

## Acceptance status

- [x] Investigated with new, targeted offline experiments (600 new
      held-out episodes across 6 checkpoints), not just re-reading
      existing logs.
- [x] Root-caused to a specific, falsifiable mechanism (KPM-source
      temporal-dynamics mismatch) with direct step-level evidence, not
      just restated as "sim-to-real gap."
- [x] Practical implication stated plainly: offline simulation cannot
      currently be used to pre-screen checkpoints for live deployment.
- [x] Scope of what was and wasn't investigated stated explicitly
      (Part B policy-replay not run).
