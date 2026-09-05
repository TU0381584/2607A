# M41: live (offered-load x write-cadence) envelope sweep

External-review-proposed follow-up to the M38 live-instability finding
(`docs/PAPER5_M37_M38_scoping.md`). Goal: find an operating region where
the live E2 control loop survives a full validation campaign without RLC
max-RETX; if one exists, it becomes the operating point for a positive
live result instead of a pure limitation finding.

Harness: `experiments/scripts/m41_envelope_sweep.py`. Gated, supervised,
per the reviewer's own plan -- stop at each GATE, report, await go.

## Harness bugs found and fixed before any result could be trusted

Three real problems surfaced across the first several live attempts,
each found, fixed, and verified before proceeding -- none silently
patched over:

1. **Missing `XAPP_OAI_PROTO_DIR` for the contention-gate subprocess.**
   The env var was only set inside the probe role's own process; the
   gate runs as a separate subprocess and doesn't inherit it. Fixed by
   setting it at module load time so every subprocess this script spawns
   inherits it.
2. **Gate/bring-up ordering was backwards.** The contention gate
   (`phase1_contention_gate.py`) needs an already-attached UE producing
   real traffic-driven backlog to have anything to measure -- run before
   bring-up, it just polls a nonexistent gNB until every one of its ~120
   polls times out. Fixed the ordering (bring-up, traffic, then gate),
   increased the subprocess timeout 180s->240s for margin, and closed a
   related robustness gap: an uncaught `subprocess.TimeoutExpired` from
   the gate would have crashed straight past teardown once the gate ran
   after bring-up, leaving the stack live. Wrapped the whole bring-up-
   through-probe-launch stretch in a try/except/finally keyed on an
   explicit `reached_probe_launch` flag.
3. **A `pkill -f` self-match footgun, reintroduced from earlier in this
   same project's history.** `sh()`'s `subprocess.run(cmd, shell=True)`
   spawns `/bin/sh -c "<cmd>"`, whose own argv then literally contains
   the pkill pattern text -- `pkill -f` matches every process's full
   command line, so that wrapper shell can self-match and die instead of
   (or in addition to) the real target, leaving stragglers running past
   teardown (observed directly: a stray mmtc iperf3 client survived one
   teardown). Fixed with a `pkill_pattern()` helper using
   `subprocess.run([...], shell=False)` -- no wrapper shell exists to
   self-match. Verified directly with a marker-tagged throwaway process
   (via a script file, after an initial verification attempt was itself
   fooled by the identical footgun one layer up, in the *test*
   invocation's own shell wrapping).

## A confound investigated and ruled out

C1 and C2's first clean-ish attempts both failed at an identical,
suspiciously fast t=10s onset -- far faster than the 60-100+s of clean
combined-traffic windows this project's own M38 investigation repeatedly
recorded with no probe attached at all. Hypothesis: the contention
gate's own PIN phase drives real traffic-relayed backlog on embb to
10M+ units (confirmed directly in every gate trace: `recovery mean ==
pinned max`, i.e. it never drains within the gate's own 30-poll restore
window, matching this project's own already-documented M8 finding) --
and since the harness only restarted the *native* stack (gNB/UEs) after
the gate, not the Docker core (AMF/SMF/UPF) those UPF-relayed backlog
units actually passed through, every condition might have been
inheriting residual core-network state from the gate, not measuring a
clean baseline.

Tested directly: extended the fresh-restart step to a full Docker core
cycle (`compose down` then `up`, not just `up`) before the post-gate
native restart. **Re-ran C1 under this fix: onset was still exactly
10.0s.** The core-state hypothesis is therefore ruled out -- the
identical onset was not a core-network artifact. The full-core-cycle
fix is kept anyway (strictly more rigorous, and now confirms the result
is robust to this specific concern) and both C1 and C2 were re-run under
it for a clean, apples-to-apples pair.

## GATE S0 result

Both conditions, native load (mult=1.0), recalibrated checkpoint, run
under the final (bug-fixed, core-cycle-included) harness:

| Condition | write_mode | interval | onset | outcome |
|---|---|---|---|---|
| C1 | normal | 1.0s | 10.0s | FAIL -- all 3 slices, 100.0% loss |
| C2 | static (1 write at t=0, then none) | -- | 10.0s | FAIL -- all 3 slices, 100.0% loss |

**Decision per the plan's own GATE S0 tree: C1 fails, C2 ALSO fails ->
load itself is implicated; cadence alone won't save it.** A single
initial ceiling write is exactly as fatal as continuous ~1/s rewriting
-- write repetition is not the causal ingredient at native load. Per the
plan, next is S1 (coarse 1D axis probes on both cadence and load), not
a cadence-only exploration.

Full manifest: `experiments/results/m41_envelope/manifest.csv` (includes
the earlier bugged/confounded attempts too, left in place rather than
deleted, each row's own `gate_pass`/`onset_reason` making clear which
ones predate which fix).

## GATE S1 result: the envelope is empty at every level tested

Coarse 1D screens (300s each, native recalibrated checkpoint), reusing
C1 as the interval=1s/mult=1.0 anchor point on both axes:

| Axis | Value | Onset | Trigger |
|---|---|---|---|
| cadence (mult=1.0) | interval=1s (=C1) | 10.0s | 100% loss, all 3 slices |
| cadence (mult=1.0) | interval=5s | 10.0s | 100% loss, all 3 slices |
| cadence (mult=1.0) | interval=30s | 10.0s | 100% loss, all 3 slices |
| load (interval=1s) | mult=1.0 (=C1) | 10.0s | 100% loss, all 3 slices |
| load (interval=1s) | mult=0.5 | 10.0s | max_retx (embb, n=1) |
| load (interval=1s) | mult=0.25 | 10.0s | max_retx (urllc, n=8) |

**Every point on both axes fails within the same ~10-second window.**
There is no survivor anywhere in this screen -- slowing the write
cadence 30x (1s -> 30s) makes no difference, and cutting native load to
a quarter makes no difference either (the trigger shifts from a clean
100%-loss reading to a smaller, earlier RLC max-RETX count, but the
onset timing and ultimate outcome are identical). Per the plan's own
GATE S1 instruction: **"If NOTHING survives 300s on either axis -> STOP
and report (envelope empty at these levels; next lever is finer/lower
steps or write-magnitude, await decision -- do not auto-expand)."**
Stopping here; not proceeding to S2 (which the plan gates behind a
"promising region" from S1 -- none exists to refine).

One operational note from this stage: a single condition
(`S1_load0.25`, first attempt) exited with code 1 and an apparently
truncated console log (stopped right after the Docker-core-cycle step,
no explicit error printed). The run's own `manifest.csv` row, written
via direct file I/O rather than the buffered console log, correctly
recorded `onset_reason=post_gate_bringup_failed` -- i.e. the script
itself handled the failure cleanly (a native-stack bring-up call simply
returned `False` that one time, plausibly just transient timing after
many consecutive live cycles in one session), and only the *console*
log was incomplete, almost certainly due to Python's default block
buffering when stdout is redirected to a file rather than a terminal.
Retried with `python3 -u` (unbuffered) and got a clean, fully-logged
result. Worth keeping `-u` on all future invocations of this script.

## Write-magnitude test: implemented, run, but the cap never engaged

Added `--write-magnitude-cap N` to the harness: wraps `send_control`
(same non-invasive technique, `env.py` untouched) to clamp each write's
ratio to move at most N units from *this process's own last-sent value*
for that slice key. Verified offline first (unit tests for the clamp
logic, including a simulated PIN-like jump correctly smoothed into
gradual steps) before running live. Also fixed a real harness bug
found on the way: `restart_native_stack()`'s final 3-way ping
connectivity check had no diagnostic output at all on failure (unlike
every other check in the same function) -- a run died with exit 1 and
an apparently-silent log even under `python3 -u`, and the cause turned
out to be this one unlogged branch, not buffering. Fixed with per-UE
diagnostics and one retry after a 5s grace period.

**Result at cap=1** (native load, 1s cadence): failed at t=10.0s,
identical to every uncapped condition -- **and zero
`magnitude-capped` log lines appeared at all.** The cap never had
anything to clamp. On reflection this makes sense and is itself
informative: the policy's own per-decision step size is already ±1
ratio unit (`ceiling_step_ratio: 1` in `saclb_live.yaml`), and the
gate's own PIN-phase damage does **not** carry over into the probe's
own ceiling tracking -- each probe launch constructs a fresh
policy/environment/`AdmissionGate` instance via its own
`reset_ceilings()` call, with no memory of what a separate, already-
killed process (the gate) wrote to a since-restarted gNB. There is no
large, discontinuous jump anywhere in this pipeline for a magnitude cap
of 1 to ever need to intervene on.

**This means "magnitude," as a policy-driven per-step quantity in this
architecture, was never actually large to begin with** -- it is not
that capping it failed to help; it is that there was nothing here for
capping to do. A genuinely different test of this axis would need
`--write-magnitude-cap 0` (freeze the ceiling at its reset default
forever, never letting even the first decision move it) -- but this is
very close to what C2's static mode already tested (one real write,
then permanent silence) and already failed identically at t=10s, so a
different outcome here seems unlikely and was not run, to avoid
spending more live-rig time on a low-probability repeat of an existing
result without being asked to confirm that specific redundancy first.

## A previously-untested axis: true UE-count (concurrency), not just load intensity

Every condition above -- including S1's own "load" axis (mult=1.0/0.5/
0.25) -- kept all 3 UEs simultaneously RRC-attached throughout; `--load-
mult` only scales each already-attached UE's traffic bitrate, never the
number of UEs actually connected. This project's own earlier M38
investigation found a genuinely single UE (embb alone, nothing else
attached) survives much longer before eventually failing (~90-180s at
native load) than any multi-UE combined condition -- but that was never
re-tested under M41's own more rigorous methodology (gate, full core+
native fresh restart, write-cadence control).

Extended the harness with `--slices` (comma-separated subset of embb/
urllc/mmtc controlling which UEs actually get attached and traffic-
loaded, independent of `--load-mult`) and ran a genuine single-UE
condition.

**At native (1x) load, embb alone: the contention gate itself could not
validate the run.** Backlog stayed at exactly 0.000 throughout baseline,
pinned, AND recovery phases -- pinning the ceiling to 1 created no
measurable contention at all with only one UE's traffic present. This
is a real, informative signal in its own right (a completely different
signature from every 3-UE gate trace, which all showed 8-10 million
backlog units under the identical pin), independently corroborating
that concurrency changes the contention picture -- but it also means
the actual live-control-loop probe never ran, since the gate blocked
the condition per the standing "gate must pass before every live run"
rule.

**Retried at 5x native load (20 Mbps) so the gate could pass legitimately
(it did) -- and the actual probe then failed at the identical t=10.0s
onset, clean 100% loss, single UE.** This is a real result, but a
**confounded** one: reaching a passing gate required jumping load all
the way to 5x, a level never tested at 3 UEs (S1's own load axis only
went up to native 1.0x). It is not possible to tell from this run alone
whether the ~10s failure here is because UE-count doesn't actually
matter after all (matching the S0/S1 pattern), or because 5x load is
simply high enough to fail fast regardless of UE-count, independent of
concurrency.

**Open question, not resolved**: does single-UE, *native* load survive
meaningfully longer than 10s (matching M38's own ~90-180s prior
finding), or does it also fail fast once the current sched_lock-fixed,
gate-checked, freshly-restarted methodology is applied? Answering this
cleanly needs either (a) deliberately running the probe at native load
with the gate check skipped for this one condition (a real, flagged
deviation from the standing "gate must pass" rule, since the gate's own
detection method demonstrably doesn't apply at this specific low-
contention operating point -- not evidence the rig itself is unhealthy),
or (b) finding some load level between 1x (gate can't detect anything)
and 5x (fails immediately, confounded) that both passes the gate and
stays close enough to native to isolate the concurrency question
cleanly. Neither has been run; flagging rather than deciding
unilaterally, since (a) specifically means deviating from an explicit
standing instruction from the reviewer's own plan.

## Status

All three originally-proposed axes (write cadence, offered load, write magnitude)
have now been explored and none of them changes the outcome: every
condition across all three fails within roughly the same 10-second
window, at native load down to 1/4, at cadences from 1s to 30s, and
under a magnitude cap that (it turns out) never needed to bind. GATE
S0 and S1 are both closed; the magnitude axis produced a null-
engagement result rather than a clean comparison, for the structural
reason above.

A fourth axis (true UE-count/concurrency, distinct from load intensity)
was added afterward and produced a genuinely new but confounded data
point: single-UE at native load couldn't even clear the contention gate
(zero measurable backlog pressure with only one UE), and single-UE at
5x load to force a gate pass failed at the same ~10s onset as
everything else -- but 5x is high enough that this doesn't cleanly rule
concurrency in or out.

**Recommendation**: treat the empty envelope (across cadence, load, and
magnitude, all at 3-UE concurrency) as the finding itself for the
manuscript -- this rig's live per-slice E2 ceiling control destabilizes
RLC within about 10 seconds across a wide, deliberately-explored
parameter range. Two genuinely open threads remain, both requiring a
decision rather than a unilateral next step: (1) the UE-count question
above -- does native-load single-UE survive meaningfully longer, which
would need either skipping the gate for that one condition (a real,
flagged deviation from the plan's own standing rule) or finding an
intermediate load level; (2) reading the actual RLC/MAC source path
directly to find the real mechanism instead of continuing to screen
around it black-box (this project's own earlier, non-M41 investigation
already proposed this and did not pursue it). Awaiting direction on
either, or to close this out and write the empty-envelope result into
the manuscript as-is.

## Source-reading investigation (Op B) and the `get_ue_list()` fix retest

Pursued thread (2) above via a multi-agent source-reading investigation
(4 parallel surveys of the RLC/MAC scheduler and E2 write paths, 1
synthesis, 3 independent adversarial skeptics) rather than continuing
to screen the failure black-box.

**Leading hypothesis produced (not confirmed):** an uninitialized-stack
read of `SL_sched[0]` in `dl_sched_unit()` (`gNB_scheduler_dlsch.c:1506`
inits from `i=1` while the real read/subtract loop at `:1514` starts at
`i=0`, on a non-zeroed VLA post-qsort), combined with a floor-less
cross-slice `min_prbs` subtraction in `pf_dl_slice()` and the hard
5-PRB scheduling cliff, chained through RLC's own 32-retry/45ms budget.

**Adversarial verification: refuted.** 2 of 3 skeptics traced that this
bug is unconditional on `dl_num_slice >= 2` (always true) and would
fire every slot regardless of whether any E2 write ever happened --
directly contradicting the established fact that write-free
traffic-only baselines run clean for 60-300+ seconds. The 3rd found
that in the exact single-UE case, `SL_sched[0]`'s garbage value gets
overwritten with real usage data before the real slice ever reads it --
the mechanism doesn't survive tracing even for the case it targeted.

**A second, distinct finding did survive:** `get_ue_list()`
(`e2_message_handlers.c:324-418`) took no lock at all on live MAC
scheduler state (`UE_info.mutex`/`sched_lock`), unlike its sibling
`set_gbr_ue()`. It runs on the E2 agent's own real-time thread and
fires on every single `poll()` -- the only candidate mechanism whose
trigger condition actually matches "as long as the control loop
exists." Flagged explicitly as a **hypothesis**, not proven: torn UE
telemetry returned to the RL policy could drive a bad-but-plausible
ceiling write via the now-correctly-locked `apply_slicing_ctrl()`.

**Fix applied and retested live.** Locked `get_ue_list()`'s full
critical section (list-count through field population) under
`sched_lock` then `UE_info.mutex`, matching the double-lock convention
already established by `dump_mac_stats()`/`mac_remove_nr_ue()`
elsewhere in this codebase. `nr-softmodem` rebuilt clean via `ninja`.
Retested with S0_C1's exact parameters (3-UE, native load, 1s write
cadence, 300s) for a clean before/after comparison against the
documented pre-fix baseline.

(Two of my own operational mistakes contaminated the first two retest
attempts before a clean run: a botched shell backgrounding launched a
second, orphaned orchestrator that raced the deliberate one over the
same tmux sessions -- caught and killed; then a launch using the
system `python3` instead of the project's venv crashed the probe
subprocess on `import numpy` in ~2s. Neither reached the actual
control loop; both are noted here for the record, not folded into the
result.)

**Result: unchanged.** `postfix_S0_C1_retest_v3` failed at the
identical t=10.0s onset, identical 100%-loss-across-all-three-slices
signature, and the gNB-side stats show `dlsch_errors`/`ulsch_errors`
near zero throughout (PHY healthy) while all three UE-side logs show
the same `[RLC] max RETX reached on DRB 1` — matching every prior
condition in this sweep exactly. **The `get_ue_list()` lock fix is a
real, independently-justified correctness improvement (kept), but it
does not explain or resolve this failure.** This directly rules out
the lock-race hypothesis via a live test, joining CPU governor,
scheduling contention, Docker-core residual state, write cadence,
offered load, and write magnitude as ruled-out explanations.

**Status after this thread:** every mechanism proposed so far --
black-box (cadence/load/magnitude/UE-count) and now white-box
(scheduler PRB-accounting bugs, E2 lock gaps) -- has either been
directly ruled out by a live test or failed adversarial verification
against the code. The empty envelope stands. The synthesis's own
recommended next step, not yet pursued, is runtime instrumentation:
log `SL_sched[i].{min_prbs,max_prbs}` per slot for ~15s around a
single-write test and confirm UL vs DL direction on the failing DRB,
to observe the mechanism directly rather than continue reasoning about
it from static code or ruling out candidates one at a time.

## Root cause found: config ratio scale vs. the scheduler's hard PRB floor

Pursued the runtime instrumentation directly. Added wall-clock-timestamped
`M41DBG` logging (temporary, not yet reverted) at four points: the E2
write itself (`apply_slicing_ctrl()`), `dl_sched_unit()`'s per-slot
`SL_sched[i].{min_prbs,max_prbs}` both pre- and post-`pf_dl_slice()`
(rate-limited to every 20th call), and RLC's `retx_count` ramp-up at
both increment sites plus the terminal `max_retx_reached()` call.
Rebuilt `nr-softmodem`+`nr-uesoftmodem` clean. Ran one minimal capture
(`experiments/scripts/m41_diag_single_write.py`, new, diagnostic-only,
not an envelope-sweep condition -- no gate, no manifest row, since
neither is relevant to a single mechanism-diagnosis capture): 3-UE,
native load, `write_mode=static` (one write per slice, ever), 40s
window, gNB and UE processes on the same host clock so wall-clock
timestamps line up directly across both logs.

**Confirmed mechanism, three independent ways:**

1. **Live instrumentation.** The single write landed at
   `t=1788605959.41` with `(min_ratio,max_ratio)` = `(1,1)` for mmtc,
   `(1,2)` for embb, `(1,2)` for urllc. Immediately after, `SL_sched`'s
   computed ceiling for every one of the three real slices
   (`n_rb_sched_init=106`) was `max_prbs` = 1 or 2 raw PRBs -- and
   stayed there, sampled every ~15ms for the full 40s window. The
   *actual usage* (`pf_dl_slice`'s post-call `min_prbs`) was 0 in
   **15,104/15,150** samples for mmtc and **14,536/15,150** for urllc
   post-write (the ~600-normal count non-zero samples were all exactly
   5-6 PRBs, matching HARQ retransmission grants, which structurally
   bypass the slice ceiling -- not new-transmission scheduling).
   Pre-write, the same slices scheduled freely (up to 70+ PRBs/slot).
2. **Source-level confirmation.** `pf_dl_slice()`'s scheduling loop
   (`gNB_scheduler_dlsch.c:1061`) guards on
   `n_rb_remain_s >= min_rbSize` where `min_rbSize=5` (line 1058) and
   `n_rb_remain_s` starts from the slice's own `max_prbs`. With
   `max_prbs` in {1,2}, this guard is false on function entry --  the
   loop body (the only new-transmission scheduling path) never
   executes, not even once. This isn't a race or an uninitialized
   read; it's a plain, deterministic guard-clause miss.
3. **Config-level confirmation.** `saclb_live.yaml`'s per-slice ratios
   (as % of the ~106-PRB carrier): urllc `nominal=3/floor=1/cap=3`,
   embb `nominal=3/floor=1/cap=4`, mmtc `nominal=2/floor=1/cap=3`.
   Every single value in every slice's *entire configured range* --
   floor through cap -- converts to 1-4 raw PRBs, which is *always*
   below the scheduler's 5-PRB floor. This isn't a bad policy decision
   landing on a bad value; there is no good value available anywhere
   in this config's range. The file's own header comments show this
   was a deliberate, documented calibration against measured live
   traffic (avg ~5.17 PRB/UE demand) -- but never cross-checked
   against the gNB's own hard-coded minimum grant size, which lives in
   C source three layers away from the YAML and was never read
   against these numbers until this session's investigation.

**Why this explains everything observed across the entire M41/M36-M40
history:** any live control write -- the very first one, regardless of
what triggered it (policy reject-decision, static single-send, or a
plain reset-time default) -- applies a ratio from this range. The
instant it lands, the affected slice's new-transmission DL scheduling
stops completely (confirmed: ~100% zero-PRB usage post-write). Without
DL grants, the gNB can't deliver the STATUS-PDU/ACK traffic the UE's
own RLC AM sender is waiting on, so `retx_count` climbs every
~90-100ms until it clears `max_retx_threshold=32` -- 6.9s for mmtc, 20.9s
for embb, and urllc still climbing at capture end (+29s), in this
capture; timing differences reflect each slice's pre-existing buffered
margin, not different starvation severity (all three were equally at
~0 PRBs). This is independent of write cadence (a single static write
is sufficient -- matches every C2/S1/magnitude-cap test that failed
identically), independent of magnitude (there is no "small enough"
write in this config's range to avoid it), and independent of the
`get_ue_list()`/`apply_slicing_ctrl()` lock fixes (neither touches
what ratio gets chosen or what the floor is). Write-free baselines run
clean indefinitely because no control write ever applies any ratio
from this range at all -- ceilings stay at their generous compiled-in
defaults (`min=0,max=100`).

**The uninitialized `SL_sched[0]` bug from the Op B investigation is
independently re-confirmed live** (`sid=0`'s first sampled ceiling
line this run: `min_prbs=1465999376`, clearly garbage stack memory) --
consistent with the skeptics' verdict that it's real but unconditional
and unrelated to this failure (`sid=0` has no attached UE; its garbage
value is inert here).

## Fix applied and confirmed live: full 300s survival, zero RLC failures

User authorized applying the fix and retesting. Scaled every slice's
`nominal_ratio`/`min_ratio_floor`/`max_ratio_cap` in `saclb_live.yaml`
by 6x (proportions preserved from the 2026-07-14 measured-demand
calibration, not re-derived): urllc 3/1/3 -> 18/6/18, embb 3/1/4 ->
18/6/24, mmtc 2/1/3 -> 12/6/18. The floor alone now converts to
106*6% = 6.36 -> 6 PRBs, a full PRB above `pf_dl_slice`'s hard
`min_rbSize=5` guard. `ceiling_step_ratio` (1) left unchanged.
Rebuilt nothing (config-only change, no recompile needed).

**Retest 1** (`m41_diag_single_write.py`, same single-static-write
methodology as the root-cause capture, 3-UE native load, 40s): the
write landed as `(min,max)` = mmtc `(6,10)`, urllc `(6,17)`, embb
`(6,17)` -- real, moderate reject-decisions this time, not an
instant floor-collapse. **Zero `retx_inc`/`max RETX reached` events on
any of the three UEs for the full 40s window** (vs. 123-413 RETX
events per UE, failing within 7-30s, on the identical capture before
the fix). Slices showed a genuine mix of idle (0 PRB, nothing queued)
and real grants (5-18 PRB) rather than permanent zero -- the scheduler
is working normally again.

**Retest 2** (`m41_envelope_sweep.py`, full standard S0_C1 parameters:
3-UE, native load, normal 1s write cadence, 300s, through the full
gate+fresh-restart ritual): survived to t=192s with 0% loss on all
three slices, then one anomalous-looking reading on embb -- investigated
before concluding anything, since `max RETX reached` count was **zero**
for embb (ue1) despite the apparent "100% loss." Root cause: a second,
independent, real bug in the harness's own `check_loss()` -- its regex
`([+-]?\d+)% packet loss` has no `.` in its digit class, so on any
non-round ping loss reading (e.g. ping's own "66.6667% packet loss"
whenever exactly 1 or 2 of 3 packets are lost) it matches only the
post-decimal digits ("6667"), which then fails the `>100` sanity check
and gets clamped to a false 100%. This is the same "anomalous 6667%"
symptom from the write-magnitude-cap testing earlier in this
investigation, now root-caused rather than just clamped-and-moved-on:
it was never a duplicate-reply artifact, it was a decimal-parsing bug,
and at least this once it silently converted a real, mild, transient
reading (1 of 3 received, high latency, other two slices fully healthy)
into an indistinguishable-from-catastrophic "100% loss" record. Fixed
the regex to `([+-]?\d+(?:\.\d+)?)% packet loss` (captures the full
decimal value). Also independently confirmed via the ceiling log that
by t~190s every slice's ceiling had walked down to exactly the new
floor `(6,6)` through sustained normal-cadence reject-decisions --  a
real, expected dynamic at this floor value, not a bug, and worth
knowing since it means the floor is the operating point that actually
matters under sustained load, not just the nominal/cap values.

**Retest 3** (identical to retest 2, with the `check_loss()` fix):
**`survived=True` -- 0% loss on all three slices at every checkpoint
from t=10s through t=290s, zero `retx_inc`/`max RETX reached` events
on any UE across the full 300s.** This is the first live condition in
the entire M36-M41 history to survive the standard 300s test. Manifest
row: `postfix_configfix_S0_C1_retest_v2,...,survived=True,...`.

**Status: root cause fixed and live-confirmed.** Both bugs found this
session are real and independently useful: the config recalibration
(the actual fix for the catastrophic universal failure) and the
`check_loss()` decimal-parsing bug (a harness measurement bug, caught
in the act of almost producing a false "the fix didn't work" reading).
The M41DBG instrumentation is left in place in ORANSlice/ (gitignored,
preserved at `docs/patches/m41_diagnostic_instrumentation.patch`),
harmless at its current sampling rate, not yet reverted. Two items
flagged, not yet acted on: `max_ratio_cap` for urllc/mmtc is still the
old-era, never-independently-verified-against-live-demand value
(proportion-preserved from the 2026-07-14 numbers, not re-measured),
and embb's floor (6 PRB) is well below its own measured peak demand
(12 PRB) -- both survived this specific test but haven't been stress-
tested against sustained high-reject-rate conditions the way this
retest happened to explore only somewhat. `Lmax=10`'s own calibration
(tuned against the pre-fix 1-2 PRB/step deficit scale) has not been
revisited against the new, 6x-larger PRB quotas either.
