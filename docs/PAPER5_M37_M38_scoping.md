# M37/M38 scoping (both fully blocked on live rig -- no live work done here)

Written 2026-09-01 as offline prep only, per explicit user instruction to
hold all new live-hardware risk (the M36 2-UE RLC max-RETX instability,
twice recurring, `docs/PAPER5_M36_congestion_characterization.md`) until
they can be present. Nothing below touches the rig. This is the same
"make the concrete design decisions the spec never made, grounded in
what the code and checkpoints support" approach already used for M4's
own plan.

## Exact reviewer spec (recovered verbatim from the pre-compaction
transcript -- the milestone-status memory only carried a lossy one-line
paraphrase of each, and re-deriving from that risked inventing detail
the reviewer never actually specified)

```
=== M30 (-> M37): recalibrated-sim generalisation to an UNSEEN live load (rig) ===
Precondition: pick a live load from M29(->M36) NOT used as a recalibration
anchor (anchors were 3 and 6 UE; target intermediate/asymmetric, e.g. 4 or
5 UE). Confirm RealisticServedKpmSource was NOT fit to it.
1. Take the recalibrated seed-900 checkpoint live at the held-out load,
   >=15 episodes, real traffic, block precision primary.
GATE M37 (pass = precision >= 0.9 AND zero collapsed episodes): report
precision + per-episode block counts + reward CI. If FAIL, stop and
report -- do not proceed; it is a real negative result to write up.

=== M31 (-> M38): properly-powered live correctness campaign (rig) ===
1. Original vs recalibrated checkpoint, >=20 live episodes each, at
   {3 UE, 6 UE, M37 held-out load}, real traffic, block precision primary
   + mean reward/step secondary, 95% bootstrap CIs (10,000 resamples).
2. Emit results/m38_live_correctness.csv + a 3-condition figure mirroring
   current Fig. 8.
GATE M38: stop, report the full CI table, lock collapse/non-collapse
status per condition. Await go before M39 (manuscript restructure).
```

## What "recalibrated seed-900 checkpoint" actually is (not obvious --
had to be traced)

This is **not** M1's recalibration (`docs/PAPER5_M1_recalibration.md`,
which fits `ClosedLoopKpmSource`'s `backlog_capacity`/temporal params
against seeds 256-261's `dqn_sla` checkpoints -- an entirely different,
earlier checkpoint lineage from Paper #4, and a recalibration attempt
that itself failed ("the loss surface is flat and the live target is
unreachable"). Silently reusing that lineage's checkpoint here would
have been exactly the kind of quiet substitution this project's own
methodology already treats as a serious mistake (the M4 disruption
churn retraction).

It is instead **M34's** work, already written into the manuscript
(`Papers_4-5/Paper_5/WPC/main.tex`, the recalibration subsection around
line 573-601, Fig. 8 = `fig8_live_recalibrated_fix`): `RealisticServedKpmSource`
(`experiments/scripts/realistic_served_kpm_source.py`) replaces
`ClosedLoopKpmSource`'s ratio-derived served-PRB with empirically
measured served-PRB, interpolated between two live-measured anchors:

```python
SERVED_PRB_3UE = {"urllc": 5.0, "embb": 13.0, "mmtc": 5.0}
SERVED_PRB_6UE = {"urllc": 10.0, "embb": 45.0, "mmtc": 10.0}
```

seed900 was retrained under this simulator
(`experiments/scripts/m34_realistic_train.py`). Two versions exist on
disk: `experiments/results/m34_realistic_retrain/seed900/...` (v1, the
manuscript's own documented "real implementation bug" version -- offered
demand stayed fixed while only served capacity scaled, still collapsed)
and `experiments/results/m34_realistic_retrain_v2/seed900/train/dqn/
offline_train/rep_0/checkpoint.pt` (v2, the corrected version -- this is
**the** recalibrated seed-900 checkpoint, confirmed by cross-checking
its live 6-UE eval against the manuscript's own published numbers, see
below).

**Original checkpoint** (for comparison): the same one every M35/M36
work already uses, `experiments/results/m8_live_anchor/offline_train/
single_agent_dqn/seed900/train/dqn/offline_train/rep_0/checkpoint.pt`
(confirmed via `experiments/scripts/m32_ood_check_original_ckpt.py`,
which references this exact path as "original checkpoint").

## What already exists (do not rerun -- reuse)

The manuscript's current Fig. 8 already covers 3 of the up-to-6 cells
{original, recalibrated} x {3 UE, 6 UE, held-out}, all real live data,
20 episodes each, on this rig:

| Condition | Data | Reward mean [95% CI] |
|---|---|---|
| Original, 3 UE | `experiments/results/live/m31_highconf/3ue_20ep_omega_log.jsonl` | -7.091 [-7.484, -6.703] |
| Original, 6 UE | `experiments/results/live/m31_highconf/6ue_20ep_omega_log.jsonl` | -8.363 [-8.757, -7.975] |
| Recalibrated, 6 UE | `experiments/results/live/m34_realistic_retrain_check/6ue_20ep_omega_log.jsonl` | -7.816 [-8.195, -7.442] |

Verified by running the new `m37_generalization_gate.py` (below) against
all three: numbers reproduce the manuscript's own published figures
exactly, and precision/collapse status matches the manuscript's prose
("every episode blocked... zero collapsed episodes" for both healthy
conditions; "complete silence," i.e. 20/20 collapsed episodes, for
original-6UE).

## The actual gap -- what M37/M38 need that does NOT exist yet

1. **M37**: recalibrated checkpoint, live, at a held-out UE count. Never
   run. Candidates per the reviewer's own suggestion: **4 or 5 UE**
   (confirmed via direct inspection of `realistic_served_kpm_source.py`
   that only 3 and 6 are baked in as fit anchors -- 4/5 are genuinely
   interpolated, unseen operating points, not just unlabeled data the
   simulator already saw).
2. **M38** needs the full 2x3 factorial; only 3 of 6 cells exist.
   Missing: **recalibrated checkpoint @ 3 UE** (never run -- does fixing
   the 6-UE collapse cost anything at the load where the original
   checkpoint was already healthy?), and **both checkpoints @ the M37
   held-out load**.

Both milestones are blocked on the identical live-rig requirement M36
already hit and stopped on: even reaching a stable 3-UE campaign (let
alone 4/5 UE) needs the 2-UE instability resolved first, since the
UE-count sequence in this project has always been built up one at a
time with a health check after each addition (`docs/
PAPER5_M36_congestion_characterization.md`'s own recommended next
steps). No amount of offline prep changes that.

## Prepared, tested, not yet run

`experiments/scripts/m37_generalization_gate.py` -- computes precision,
per-episode block counts, collapsed-episode count, and reward CI from
any eval `omega_log.jsonl`, and prints the GATE M37 PASS/FAIL verdict.
Validated against all three existing conditions above (exact match to
the manuscript's own published reward means and CIs; correctly reports
PASS for both healthy conditions and FAIL for the original-6UE
collapse). Ready to point at the real held-out-load run's log the
moment it exists.

**One definitional judgment call, flagged rather than silently decided**:
"collapsed episode" is not defined anywhere upstream -- M6's "collapse
rate" is a per-seed/per-run concept, not per-episode. This script defines
a collapsed episode as one with zero total blocks, matching the
convention already implicit in `paper5_fig_live_recalibrated_fix.py`'s
own `n_zero` count and the manuscript's own descriptive language
("every episode blocked" / "complete silence"). This reading held up
exactly against all three existing conditions; worth confirming it still
feels right once real held-out-load data exists, since a genuinely new
load level could in principle produce a different failure shape (e.g.
some-but-insufficient blocking) this binary framing wouldn't distinguish
from a healthy low-block episode.

## Not yet decided -- for the user, not a call to make unsupervised

- **4 or 5 UE** for the held-out load. Slight lean toward 5: it sits
  closer to 6 (where the original checkpoint is known to fail hardest),
  making a recalibrated-checkpoint PASS there a slightly stronger claim
  than at 4. But this is a judgment call with no strong technical reason
  to prefer one over the other -- both are equally "unseen" to the fit.
- Whether to **rerun** original/recalibrated @ 3 UE and 6 UE fresh
  (consistency with M36's more careful, freshly-collected 1-UE
  methodology, and this rig's current state) or **reuse** the existing
  `m31_highconf`/`m34_realistic_retrain_check` logs as-is (faster, avoids
  spending more live-rig time restating what's already solid evidence).
  Current lean: reuse -- nothing about the rig or methodology has been
  called into question for those specific runs, and M38's own novel
  contribution is the missing cells, not re-litigating settled ones.

## Once the rig is stable enough to resume (per the M36 doc's own
recommended sequence: 2, 3 UE with health checks, then decide on 4/5/6)

1. Confirm 3-UE stability first (needed for M38's missing recal@3UE
   cell) using the exact `m36_add_ue2.sh`/`ue3.sh` pattern.
2. Run recalibrated checkpoint @ 3 UE, >=20 episodes (closes M38's first
   gap). Reuse `m36_run_probe.sh`'s launch pattern, pointed at the
   `m34_realistic_retrain_v2` checkpoint instead of `m8_live_anchor`'s.
3. Bring up the held-out UE count (4 or 5, per user's call above), run
   the recalibrated checkpoint there, >=15 episodes -- this closes M37.
   Check with `m37_generalization_gate.py` immediately.
4. If M37 passes: run the **original** checkpoint at the same held-out
   load too (closes M38's second gap), >=20 episodes.
5. Extend `paper5_fig_live_recalibrated_fix.py` from its current
   hardcoded 3-condition dict to however many of the up-to-6 cells exist
   -- straightforward (the `COND_STYLE`/`runs` dict is already the only
   thing that needs to grow), not written yet since there's nothing real
   to test it against until step 4 completes.
6. Write `docs/PAPER5_M38_live_correctness.md` with the full CI table
   once all cells exist, mirroring every prior milestone doc's structure.
