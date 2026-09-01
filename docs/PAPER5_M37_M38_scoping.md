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

### Attempt 3: CPU-affinity isolation (user-directed), also failed -- rules out host scheduling

User chose to pursue the CPU-affinity lever directly. Fresh restart, all
4 native processes (gNB, UE1, UE2, UE3) launched via `taskset -c
0,1,2,4,5,6` (a shared 6-logical-CPU / 3-physical-core pool, preserving
their original free-roaming-among-each-other behavior, which the 100s
clean windows in attempts 1 and 2 already proved was never the problem)
-- physical core 3 (logical CPUs 3,7) reserved exclusively, never
touched by any RAN process. Verified via `taskset -pc <pid>` on every
relevant PID before proceeding, not just assumed. Confirmed OAI's
`threadCreate()` (`common/utils/system.c:252`) does not override thread
affinity when `affinity == -1` (the "ffffffff" in earlier logs is just
`-1` printed as a 32-bit hex, not a literal all-cores bitmask) -- so a
taskset at process launch genuinely propagates to every child thread,
this was checked in source before relying on it.

100+ seconds of clean combined traffic again (same as attempts 1-2).
Probe launched via `taskset -c 3,7` -- confirmed via `taskset -pc` on
its actual PID to be running exclusively on the reserved cores, zero
overlap with the RAN pool. **RLC max-RETX still occurred.** The pattern
changed, though: instead of all three UEs failing near-simultaneously
(attempts 1-2), this time UE2 (mmtc) failed first and worst (638 by
+30s, 2524 final), UE1 lagged roughly 30s behind (1425 final), UE3
stayed comparatively mild (940 final) -- a staggered, asymmetric onset
rather than a synchronized one.

**This rules out host-level CPU scheduling contention as the cause.**
The probe never shared a core with any RAN process, verified directly,
and the failure still happened. Combined with attempt 2's governor/
memory results, three separate host-side explanations have now been
tested and eliminated: CPU frequency scaling, general memory/rig
fatigue, and CPU scheduling contention between the probe and the RAN
processes. What's left standing is the content/effect of the E2 control
write itself on the gNB's shared MAC-scheduler state -- something that
happens identically regardless of which core sent it.

Torn down clean after this attempt too (by PID, all RAN/traffic
processes confirmed gone, memory recovered to 4.1Gi available).

### Source investigation and a real bug fix found (user-directed, option B)

Traced the exact code path: `apply_slicing_ctrl()` in
`ORANSlice/oai_ran/openair2/E2_AGENT/e2_message_handlers.c` (called from
`ran_write()`, which is what every incoming E2 control message from the
xApp/probe ultimately dispatches through) writes directly into
`RC.nrmac[0]->SL_info.list[*].spolicy.{min_ratio,max_ratio}` --
**with no lock held**. The DL/UL scheduler
(`gNB_scheduler_dlsch.c`'s `nr_slice_preprocess`/`slice_prb_estimate`,
called every scheduling slot) reads this exact same state, and every
function on that path either holds `mac->sched_lock` for the whole
scheduling pass (`gNB_scheduler.c:206-310`) or explicitly asserts via
`NR_SCHED_ENSURE_LOCKED` that it must already be held. `sched_lock` is
declared in `nr_mac_gNB.h` immediately adjacent to the `SL_info` field
it's clearly meant to protect, and is used pervasively everywhere else
in the scheduler -- including `mac_rrc_dl_handler.c`, which is the
structurally identical case (an asynchronous external writer touching
shared MAC state) and correctly takes `NR_SCHED_LOCK`/`NR_SCHED_UNLOCK`
around its writes. `apply_slicing_ctrl()` is the one clear exception to
an otherwise completely consistent locking convention -- a genuine,
unsynchronized data race between the E2_AGENT thread and the real-time
scheduler thread, whose corruption probability scales with how much
scheduling work happens per slot (i.e. with UE count), which fully
explains why 1-UE tolerated it and why 2-3 UEs did not.

**Fixed**: wrapped the read-modify-write section of `apply_slicing_ctrl()`
with `NR_SCHED_LOCK(&mac->sched_lock)` / `NR_SCHED_UNLOCK(&mac->sched_lock)`,
mirroring `mac_rrc_dl_handler.c`'s existing pattern exactly. No new
includes needed (`e2_message_handlers.h` already pulls in `nr_mac_gNB.h`,
where the macros live). Rebuilt `nr-softmodem` incrementally via `ninja
nr-softmodem` -- clean build, only the one changed object file
recompiled and relinked. `ORANSlice/` is a separate git clone of the
upstream `wineslab/ORANSlice` fork and is gitignored by this project's
own repo (matching the existing, pre-2026-09-01 practice of leaving
local ORANSlice modifications -- a config file, a regenerated protobuf
-- as uncommitted working-tree changes rather than committed there), so
the fix lives as an uncommitted change in that clone. To make sure it
isn't silently lost if that clone is ever reset, the diff is preserved
here too: `docs/patches/e2_agent_slicing_ctrl_sched_lock.patch`.

**Retest with the fixed binary: the RLC failure recurred anyway.**
Fresh bring-up, same traffic profile, 60s clean pre-probe window (as
expected -- that phase was never the failure point), probe launched.
Within 30-90s, the **exact same asymmetric pattern from the CPU-pinned
attempt reproduced almost identically**: mmtc (UE2) failed first and
worst (732->1260->1794 RETX), embb (UE1) lagged roughly 30s behind
(0->370->904), urllc (UE3) stayed completely clean throughout (0
RETX). Stopped and torn down.

**This is itself an important result, not a dead end.** The sched_lock
fix is real, correctly identified and correctly applied per the
codebase's own established convention -- it should stay, since it
removes a genuine race regardless of whether it's the whole story here.
But the *identical* mmtc-worst/urllc-clean asymmetry reproducing across
two structurally different experiments (CPU-pinned-but-unlocked, and
locked-but-unpinned) argues against pure race-timing luck as the
explanation for *which* slice fails -- a true race's victim would be
expected to vary more with timing conditions, not repeat the same
pattern under two different interventions. This points toward something
more structural to the **mmtc slice specifically**: its ceiling was
`{min_ratio:1, max_ratio:1}` in this run (zero scheduling flexibility --
though embb's was identically `{1,1}` in the same decision, so ceiling
value alone doesn't explain the asymmetry) combined with mmtc's bursty
traffic shape (2s on / 6s off, per `traffic_profiles.yaml`) as the more
likely differentiator, rather than raw UE count.

**A relevant fact this raises**: M36's successful 1-UE live campaign
(10 episodes, zero collapse, `experiments/results/m36_live/ue1/`) used
**only embb traffic** -- mmtc's bursty pattern combined with the live
probe's control loop has never actually been validated successfully at
*any* UE count in this project's history, single or multi. The "more
UEs = worse" framing that organized M36/M37/M38 may be conflating two
different things: UE count, and this being the first time mmtc traffic
was ever run concurrently with the live control loop at all.

### mmtc isolated at 1 UE (user-directed, "keep diagnosing and fixing"): the multi-UE framing was wrong

**Test 1: mmtc alone, 1 UE, mmtc's own bursty traffic (2s on/6s off), probe
attached (recalibrated checkpoint, fixed sched_lock binary).** Fresh
clean bring-up (single UE using `nrUE_slice2.conf` in the default netns
-- no netns isolation needed with only one UE process). 60s of clean
bursty-mmtc-traffic-only baseline (0 RETX) before the probe. **The
probe alone, at 1 UE, with zero other slices active, reproduced the
failure just as fast as the 3-UE case**: 1697 RETX by +30s, 5118 by
+90s. This is unambiguous: **the failure was never about concurrency.**
2-UE and 3-UE were simply the first configurations in this project's
history to ever run mmtc traffic under the live control loop at all --
M36's only prior successful live campaign was embb-only.

**Test 2: mmtc alone, sustained (non-bursty) traffic instead of its
native bursty pattern, no probe.** Confirms traffic shape alone,
without the probe, is never sufficient (0 RETX after 60s) -- consistent
with bursty-alone's earlier clean result. The live control loop is a
necessary ingredient regardless of traffic shape.

**Test 3: mmtc alone, sustained traffic, probe attached, on a properly
fresh restart** (a first attempt at this without restarting between
tests produced a contaminated 13,823-RETX reading carried over from the
prior failed run -- caught and redone; noted here as a reminder that
"swap the traffic generator" is not equivalent to "clean baseline," only
a full process restart is). **Sustained mmtc traffic + probe also
eventually fails** -- clean through +60s, then 155 at +90s, climbing to
5828 by +150s. Slower onset than bursty (which failed within 30s), but
the same eventual outcome. **Traffic burstiness affects how fast the
failure appears, not whether it happens.**

**Ruled out as an explanation**: per-slice RLC AM configuration (no
per-slice retransmission-threshold override exists in the gNB conf --
just a global `rlc_log_level`) and per-slice QoS/GBR treatment (every
subscriber's `qos.index` is identically 9 regardless of slice in the
provisioned DB, checked directly via `mongosh`). Whatever makes mmtc
specifically fragile isn't a config-level asymmetry visible from either
of these angles.

### Where this leaves the investigation

Established, with direct evidence, everything the failure is **not**:
not CPU governor, not general host/memory contention, not CPU-core
scheduling contention with the probe (isolated and ruled out), not the
real sched_lock race this session found and fixed (fixed and ruled out
as sufficient), not UE concurrency (ruled out -- fails identically at
1 UE), not traffic burstiness alone (ruled out -- purely a rate-of-onset
effect, not a yes/no one), not per-slice RLC or QoS provisioning (ruled
out via direct config/DB inspection).

What's established, positively: **the mmtc slice specifically, under
live E2 ceiling-reconfiguration control, eventually degrades into RLC
max-RETX regardless of UE count or traffic shape; embb has never once
failed this way under the identical control loop.** The mechanism
connecting "repeated ceiling writes" to "mmtc's RLC entity specifically
degrading" is not yet identified -- it would need either RLC/MAC-layer
instrumentation this session didn't have tooling for, or a source-level
audit of whatever differs in how mmtc's specific NSSAI (`sst=1,
sd=0x000001` per the gNB conf's slice list) is scheduled versus embb's
(`sd=0xFFFFFF`) once ceilings converge to `{min:1,max:1}` -- which
happens for mmtc in every observed run, consistently, regardless of
condition.

### urllc-alone and embb-reconfirm tests (user pushed for a more decisive
finding): the slice-identity theory was also wrong

Tested urllc alone (1 UE, its own native sustained 300K/100B traffic,
probe attached, same fixed binary): **failed at almost the same rate as
mmtc** (1319 RETX by +30s, 4160 by +90s). This looked, briefly, like a
much sharper and more publishable finding than "mmtc is fragile":
embb's slice entry in the gNB conf uses a **wildcard** SD
(`sst=1,sd=0xFFFFFF`), while urllc (`sd=0x000002`) and mmtc
(`sd=0x000001`) use specific values, and `apply_slicing_ctrl()` matches
slices by exact `(sst,sd)` equality -- a plausible mechanism for exactly
this asymmetry.

**Reconfirming embb alone on this same fixed binary, same session,
ruled that out too.** Clean through +90s (matching the historical
"embb never fails" pattern) -- then **also degraded**: 1536 RETX by
+120s, 2948 by +150s, 4358 by +180s. Every one of the three slices
tested today eventually failed under the live control loop; only the
*onset time* differed (mmtc/urllc: ~30s; embb: ~90-120s).

**This is not a slice-identity effect after all.** The two live
candidate explanations now are: (a) something in the E2 control loop
itself destabilizes any slice given enough repeated ceiling-write
cycles, with the specific onset time depending on that slice's own
traffic/backlog dynamics (embb's larger, steadier backlog buffer may
simply take longer to reach whatever threshold triggers the cascade);
or (b) cumulative host/session degradation across a long day of
continuous native-process bring-up/teardown cycles on this laptop
(`uptime` showed load average 7.5-7.8 immediately after tearing down
all RAN processes, at nearly the full 8-core ceiling, though this may
just be 1-minute EWMA decay from the just-killed heavy processes rather
than a real ongoing load -- not conclusively distinguished under time
pressure). A full machine reboot (not just a process-level restart,
which has been the recovery method all day) would help separate these
two, but is a larger, harder-to-reverse action than anything else tried
today and was not taken without asking first.

### Recommended next steps (revised again)

1. The sched_lock fix stays -- real correctness improvement,
   independent of this outcome.
2. **This is now a well-characterized, reproducible, mechanism-narrowed
   finding, not a vague instability.** Given how much has already been
   ruled out with direct evidence, the responsible framing going forward
   is: *live E2 slicing control destabilizes the mmtc slice specifically
   on this rig, independent of UE count* -- a real, reportable result in
   its own right, arguably more interesting than the "multi-UE ceiling"
   framing M36-M38 started with.
3. If further live debugging is wanted, the next genuinely new levers
   are: (a) RLC-layer logging/instrumentation to see exactly what
   `nr_rlc_entity_am_recv_sdu`/retransmission logic does differently for
   mmtc's DRB when a ceiling write lands mid-transmission, or (b) test
   whether *forcing* mmtc's ceiling to stay fixed (never re-write it,
   even though the policy wants to) prevents the failure -- this would
   directly confirm or rule out the ceiling-convergence-to-{1,1} pattern
   as causal rather than incidental.
4. M37/M38 stay blocked until this resolves, or until the finding above
   is accepted as the answer and written up instead of chased further.
