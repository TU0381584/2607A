# M37/M38 scoping (blocked on a newly-characterized, deeper rig instability)

Written 2026-09-01, originally as offline prep only, per explicit user
instruction to hold all new live-hardware risk (the M36 2-UE RLC
max-RETX instability, twice recurring,
`docs/PAPER5_M36_congestion_characterization.md`) until they could be
present. The offline scoping below (checkpoint identities, gap analysis,
gate-check tooling) was written first with the rig untouched.

**Update, same day, user present and directing the live attempt:** user
authorized a live 3-UE pilot for the recalibrated checkpoint (M38's
missing recal@3UE cell), explicitly instructing "reuse the 3UE/6UE"
existing data (already cross-validated against the manuscript's own
numbers, see below) rather than rerun it, and "investigate deep,
diagnose and rectify whatever you can" once the pilot failed. Full
account of that investigation is in its own section near the bottom of
this doc. Headline result: **the failure is not simple CPU/RAM
contention or "tired rig" -- it tracks the live probe's own control
activity, not traffic level, and two host-level fixes (CPU governor,
swappiness) that would have helped a pure-contention story did not
change the outcome.** M37/M38 remain blocked, now on a better-understood
but unresolved problem.

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

## Live 3-UE pilot attempts, 2026-09-01: two failures, deep investigation, root cause narrowed but not fixed

User authorized a live pilot with the recalibrated (M34) checkpoint at 3
UEs, present and directing the attempt, explicitly asking to reuse the
existing 3UE/6UE data rather than rerun it. That existing data was
re-validated first: `m37_generalization_gate.py` reproduces the
manuscript's own published reward means/CIs for all three existing
conditions exactly (see table above), so "reuse, don't rerun" stands --
nothing about that data is in question.

### Attempt 1: clean bring-up, failure during the probe

Docker core (17 containers) up clean, subscriber DB intact (9 IMSIs
survived). gNB + UE1 + UE2 + UE3 all attached cleanly via
`restart_ran_stack.sh` (after fixing a real bug in it, see below) plus
manual continuation. All 3 UEs reachable, 60s of real combined traffic
(embb 4M/1200B sustained, urllc 300K/100B sustained, mmtc 50K/80B
bursty, exactly `traffic_profiles.yaml`'s spec) with zero packet loss
and zero RLC errors before the probe was launched -- this is already
further than either 2-UE attempt in M36 got.

Shortly after the pilot probe (recalibrated seed900 checkpoint, 6
episodes) started making live decisions, **all three UEs** hit `[RLC]
max RETX reached on DRB 1` simultaneously and went to 100% packet loss
-- worse than M36's 2-UE failures, where only one UE failed at a time.
The probe process died silently around episode 1/step 29 of 60 (no
Python traceback, no OOM-killer signature in dmesg). The last few
logged decisions showed `per_slice_sla_margin` around -10^5 to -10^6 --
the exact catastrophic-backlog-explosion artifact the manuscript's own
recalibration section already documents ("$\approx$-1,002,377.5") as
what happens once a real backlog-failure regime is entered. That data
was discarded, not kept, same reasoning as M36's 2-UE discard.

**Found and fixed a real, unrelated bug while diagnosing this**:
`restart_ran_stack.sh` sets `LOG_DIR="$ORANSLICE_HOME/logs"`, a
directory that doesn't exist (every other script in this project uses
`experiments/logs`). The gNB was actually healthy and heartbeating the
whole time (confirmed via `tmux capture-pane`); the script's own health
check just couldn't find the log file it had written, reported a false
FATAL, and aborted before launching any UEs. Fixed in the script
(commit pending); the actual bring-up was continued by hand using the
corrected path.

### Investigation: CPU governor and memory, both real findings, neither the cause

1. **`ulimit -r` / RTPRIO check**: initially looked like the RAN
   processes weren't getting real-time scheduling at all (checked the
   wrong PIDs -- the sudo wrapper and main control thread, both
   legitimately SCHED_OTHER). Checking the actual worker threads
   (`Tpool0-7`, `UEthread_0`) showed **RTPRIO=97 correctly granted** --
   real-time scheduling is working as OAI intends. Not the cause.
2. **CPU governor was `powersave`**, cores running ~2.7GHz against a
   4.2GHz max (64%) even under moderate load -- not thermal throttling
   (dmesg clean, temps 69-71C, well under this CPU's throttle point).
   Switched all 8 logical CPUs to `performance` (safe, reversible,
   standard tuning; left in place afterward -- revert manually if
   battery life matters more for daily use).
3. **Memory was genuinely tight**: the gNB process alone uses ~2.17GB
   RSS and runs at a sustained ~159% CPU even at idle (real PHY-layer
   rfsim compute cost, not obviously a leak, but worth someone
   double-checking against an earlier session's gNB footprint if this
   recurs). This laptop was also running a full desktop session (Chrome,
   Firefox, VS Code, gnome-shell) concurrently with the entire live 5G
   stack -- genuine resource competition on a 7.4GB/8-core machine.
   Lowered `vm.swappiness` 60->10 to reduce unnecessary swap activity.
   Did **not** close the user's other applications -- out of scope to
   touch without asking.
4. **This is an i5-1135G7, 4 physical cores + HT = 8 logical**, 106-PRB
   band78 rfsim gNB + 3 native UE processes is genuinely demanding for
   this hardware class.

### Attempt 2: full clean restart with both fixes applied -- failed again, faster

Full teardown (by exact PID, not `pkill -f` pattern -- see footgun note
below), fresh relaunch of gNB+UE1+UE2+UE3, governor confirmed
`performance` throughout. **100+ seconds of combined 3-UE traffic ran
completely clean** (0 RLC errors, 0% packet loss) before the probe was
launched -- a materially longer clean window than attempt 1 had. The
same pilot probe was launched again. RLC max-RETX began within the
first ~30s of the probe running and reached into the thousands per UE
within 90s -- **faster than attempt 1**, despite the governor/swappiness
fixes and a longer proven-clean pre-probe window.

**This is the key diagnostic result**: the failure's onset tracks the
*probe starting*, not cumulative traffic duration, not CPU governor,
not "rig fatigue" from a long prior session (this was a fresh restart).
The exact same traffic, on the exact same freshly-restarted stack, ran
fine for 100+s with no probe attached. The moment the probe began
issuing live ceiling-reconfiguration decisions (~1/s cadence, PRB
min/max_ratio via `slicing_control_m`, one shared decision affecting
all 3 slices' currently-active bearers simultaneously), failures began
almost immediately.

Direct evidence at the point of first failure: UE1's log shows a single
anomalous `RSRP = -93 dBm` reading (every other reading throughout the
whole session, before and after, reads -42dBm) immediately preceding
the first `max RETX` line -- consistent with a transient PHY-layer
timing/sample discontinuity in the rfsim channel simulation at that
exact instant, not a gradual signal degradation.

**Working hypothesis, not confirmed**: with only 1 UE (M36's own
successful 10-episode campaign, same probe, same ~1/s decision cadence),
a ceiling-reconfiguration event only ever touches one active bearer.
With 3 UEs, the identical event touches three simultaneously-active
bearers at once. If applying `slicing_control_m` on the gNB side
introduces even a brief processing hiccup, one UE's momentary miss might
be recoverable; three at once, sharing the same finite radio-frame
timing budget, may not be. This points at the gNB-side handling of
concurrent ceiling application under live traffic, not at probe-side
CPU scheduling -- which is why CPU-pinning the probe process was
considered and **not attempted**: the mechanism increasingly looks like
it lives on the gNB/OAI-source side, and pinning the probe alone
wouldn't test that.

**Not attempted, and why**: modifying the decision cadence (would need a
sleep/pacing knob) doesn't exist in `m33_live_state_probe.py` or the
live config, and the cadence is likely emergent from
`qoe_oran_framework/`'s own live-stepping logic -- frozen, not to be
touched (see [[feedback-never-invent-honest-reporting]]'s constraint 1).
CPU-affinity isolation of all 4 native RAN processes was considered but
not attempted given the mechanism now looks gNB-side rather than
host-scheduling-side, and reconfiguring affinity on an already-live
multi-process stack carries real risk of a *new*, harder-to-diagnose
failure mode for uncertain benefit.

**A process-management footgun worth recording**: `pkill -f "<pattern>"`
and `pgrep -f "<pattern>"` match against the full command line of every
process -- including the shell wrapper currently running the very
`pkill`/`pgrep` invocation, if that wrapper's own argv happens to
contain the same substring (it does, here, since the pattern text is
literally an argument on that line). This silently SIGKILLs the
invoking shell mid-script with no further output. Killing by exact PID
avoids it entirely; this bit twice in one session before being
diagnosed.

### State after attempt 2: fully torn down

Native RAN processes and traffic generators killed (by PID, not
pattern). Docker core (17 containers) and `iperf3-target` left running
-- never implicated in either failure, no reason to cycle them. Memory
recovered to 4.4Gi available. CPU governor left at `performance`
(user's call to revert for battery life). `ue2ns`/`ue3ns` namespaces
left in place per established convention.

### Recommended next steps

1. This is now a **twice-confirmed-in-one-session, mechanism-narrowed**
   finding, not a vague instability -- it may be worth writing up as a
   real result (multi-UE live control-loop ceiling, analogous to M28's
   2-gNB hardware ceiling) rather than continuing to treat it as a
   blocker to route around.
2. If further live debugging is wanted, the next genuinely new lever is
   CPU-affinity isolation of the 4 native processes (gNB, UE1, UE2, UE3)
   across the 4 physical cores, explicitly reserving none of them for
   the probe -- testing whether reducing host-scheduling noise around
   the control-write moment helps at all, which would argue for a
   host-side rather than gNB-source-side cause after all. This has not
   been tried.
3. Alternatively, someone with OAI/ORANSlice source familiarity could
   look at what `slicing_control_m` application actually does on the
   gNB side when multiple UEs are RRC-connected -- whether it touches
   shared MAC-scheduler state in a way that could stall or corrupt an
   in-flight transmission on an unrelated UE.
4. M37/M38 stay blocked either way until this resolves -- both need
   stable multi-UE live campaigns this rig has not yet produced.
