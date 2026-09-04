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

## Status

All three proposed axes (write cadence, offered load, write magnitude)
have now been explored and none of them changes the outcome: every
condition across all three fails within roughly the same 10-second
window, at native load down to 1/4, at cadences from 1s to 30s, and
under a magnitude cap that (it turns out) never needed to bind. GATE
S0 and S1 are both closed; the magnitude axis produced a null-
engagement result rather than a clean comparison, for the structural
reason above.

**Recommendation**: treat the empty envelope as the finding itself --
this rig's live per-slice E2 ceiling control destabilizes RLC within
about 10 seconds regardless of cadence, load (within the tested 4x
range), or the (already-small) natural magnitude of each write. The one
lever that hasn't been tried is `--write-magnitude-cap 0` for full
confirmation, or reading the actual RLC/MAC source path directly (this
project's own earlier, non-M41 investigation already proposed this and
did not pursue it) to find the real mechanism rather than continuing to
screen around it black-box. Awaiting direction on which, if either, to
pursue, or to close this out and write the empty-envelope result into
the manuscript.
