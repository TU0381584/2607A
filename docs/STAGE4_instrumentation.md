# Stage 4 — Engineering instrumentation

Four numbers a reviewer would ask about the real-hardware realization,
none of which were measured or logged anywhere before this stage. Two
are purely offline (no rig), one is derived from wall-clock data already
collected across the whole live campaign (no new rig time), and one
needed a short, dedicated live measurement (~10 minutes total rig time —
no training, no traffic generators, no episodes; just the gNB/E2 agent
and 3 attached UEs).

---

## 1. E2 control-loop round-trip latency (live, directly measured)

`experiments/scripts/measure_e2_latency.py` imports `LiveKpmSource`
directly (the exact class every live arm uses) and times 500 real
`poll()` calls and 500 real `send_control()` calls against the actual
gNB E2 agent, after a 20-call warmup. Raw: `docs/stage4_e2_latency_raw.json`.

| Operation | n | mean | median | p90 | p99 | min | max |
|---|---|---|---|---|---|---|---|
| `poll()` — true round trip (INDICATION_REQUEST → blocks for INDICATION_RESPONSE) | 500 | 0.656 ms | 0.566 ms | 1.099 ms | 1.773 ms | 0.286 ms | 2.864 ms |
| `send_control()` — call/send overhead only | 500 | 0.225 ms | 0.168 ms | 0.399 ms | 0.867 ms | 0.096 ms | 1.610 ms |

`send_control()` is **not** a round trip — confirmed directly from
`live_kpm_source.py`'s own docstring: CONTROL messages get no response
at all on this OAI E2 agent (fire-and-forget, applied directly to
`gNB_MAC_INST`). What's reported for it is send-call overhead, not
latency to effect — reported as such rather than mislabeled.

All 3 UEs were seen in every poll (`mean_ues_per_poll = 3.0`), so this
is a realistic, non-empty payload, not an idle/degenerate measurement.

## 2. Per-decision inference latency (offline, directly measured)

`experiments/scripts/metrics_stage4_offline.py` loads the same frozen
seed-256 checkpoints already evaluated live (`dqn_sla`, `dqn_qoe`) via
the framework's own `build_policy`/`load_checkpoint`, on CPU — the real
deployment device (`DQNPolicy` defaults to `device="cpu"`; there is no
GPU on the near-RT control path). 2000 timed `select_action()` calls per
arm after 50 warmup calls, on correctly-shaped (`request_state_dim`),
randomly-initialized state vectors — adequate for latency benchmarking
since a fixed feedforward network's forward-pass cost depends on input
shape and architecture, not input values, and both are real/unmodified
here. Raw: `docs/stage4_metrics_offline_raw.json`.

| Arm | n | mean | median | p90 | p99 |
|---|---|---|---|---|---|
| dqn_sla | 2000 | 71.3 μs | 67.9 μs | 84.6 μs | 115.0 μs |
| dqn_qoe | 2000 | 71.8 μs | 68.3 μs | 85.1 μs | 108.9 μs |

Both reward modes share the identical network architecture (same
`state_dim`, `hidden_dim`), so near-identical timing is expected, not a
coincidence.

## 3. Policy footprint (offline, directly measured)

| Arm | Parameters | Checkpoint file size |
|---|---|---|
| dqn_sla | 18,562 | 306,741 bytes (~300 KB) |
| dqn_qoe | 18,562 | 306,741 bytes (~300 KB) |

Both DQN checkpoints have identical parameter counts (same architecture,
different learned weights). This is a small enough footprint that model
size is not a deployment constraint on any realistic near-RT xApp host.

## 4. Control-loop period: configured vs. measured (derived from existing logs, no new rig time)

Configured: `episode.step_seconds: 5.0` (`saclb_campaign.yaml`).
Empirically measured from **every already-completed live batch across
this entire campaign** (60 batches, all 5 original arms + static_at_cap
+ dqn_sla_reverify, `batch_manifest.jsonl`'s wall-clock `elapsed_s` ÷
episodes ÷ 60 steps/episode):

| | seconds/step |
|---|---|
| Configured (target sleep) | 5.000 |
| Measured, mean (naive: batch elapsed_s / episodes / 60) | 5.094 |
| Measured, median (same naive method) | 5.078 |

**Follow-up (user asked directly whether this gap was a real problem):
it is not, and it is now fully attributed, not left open.** The naive
figure above divides each *subprocess's* total wall-clock time
(`run_live_eval_arm.py`'s `elapsed_s`, which wraps the entire batch
subprocess) by episode count — and that subprocess also pays a
one-time Python/PyTorch startup cost (interpreter start, `torch`/
`protobuf` imports, `RANEnv.__init__` loading three QoE-mapper LSTM
checkpoints via `torch.load`) every time a batch launches. Two
follow-up measurements, live, isolate this cleanly:

1. **`RANEnv.step()` in isolation** (`profile_step_overhead.py`, 30
   steps, cProfile): mean **12.85 ms**/step — nowhere near the ~90 ms
   gap on its own, and it already includes the E2 poll, the per-slice
   QoE-mapper LSTM inference (`_compute_mos_by_slice`, ≈7.3 ms/step —
   the actual dominant cost inside a step, an order of magnitude more
   than the DQN admission network's ~0.07 ms), and `send_control`.
2. **The real `run_single()` loop** (`profile_run_single_overhead.py`,
   same function every live arm uses, live, 20 steps): **5000.7 ms/step
   — a 0.7 ms gap from the 5000 ms target, i.e. essentially exact.**
3. **Splitting the naive batch_manifest.jsonl figure by episode count
   per subprocess** confirms the mechanism directly: 1-episode batches
   average 307.85 s/episode (+7.81 s vs. the 300.04 s pure-cadence
   figure); 2-episode batches average 304.51 s/episode (+4.47 s) —
   almost exactly half the 1-episode overhead, the signature of a
   roughly constant ~8–9 s **per-subprocess-launch** fixed cost being
   amortized over however many episodes that particular subprocess
   happened to run, not a per-step cost at all.

**Conclusion: the live control loop hits its configured 5.0 s cadence
almost exactly (5000.7 ms measured directly). The ~85–90 ms/step figure
in the naive aggregate was a measurement artifact of this stage's own
first-pass method (dividing subprocess wall-clock, which includes a
one-time startup cost, by episode count) — not a property of the
control loop, and not something requiring a fix.** The batched,
health-checked restart architecture (`run_live_eval_arm.py`, chosen for
rig-reliability reasons documented in `CAMPAIGN_LOG.md`) is what
introduces the one-time-per-batch cost; it is invisible to any
per-episode compliance/SLA number already reported, since those are
computed from `run_single()`'s own per-step data, not from subprocess
wall-clock.

---

## Bottom line for the paper

The learned control loop itself is cheap and not a deployment concern:
sub-millisecond E2 round trip, sub-100-microsecond inference, a
~300 KB model, and the loop hits its configured 5.0 s cadence to within
0.7 ms when measured directly through the real control loop. The
dominant per-step cost inside a step is the QoE-mapper LSTM
(≈7.3 ms), not the RL decision (≈0.07 ms) or the E2 wire protocol
(≈0.6 ms) — all three combined are still under 2% of the 5 s budget.
The naive ~85–90 ms/step figure that first appeared in this stage was
traced to its exact source (a fixed ~8–9 s per-batch subprocess-startup
cost divided by episode count, not a per-step cost) and is not present
in any reported compliance/SLA/utility number, all of which come from
`run_single()`'s own per-step data, not from subprocess wall-clock.

## Acceptance status

- [x] E2 round-trip latency measured live and directly (not assumed).
- [x] Per-decision inference latency measured offline on the real
      deployment device (CPU) with the real checkpoints.
- [x] Policy footprint (parameter count + file size) measured directly.
- [x] Control-loop period reported as configured AND empirically
      measured, with the initial gap investigated to a root cause
      (subprocess startup cost, not a per-step effect) rather than left
      as an unexplained residual.
- [x] No framework source modified; no training; total live rig time for
      this stage was ~25 minutes across two short sessions (E2 latency
      measurement, ~10 min; the follow-up root-cause profiling of the
      cadence gap, ~15 min, dominated by one deliberate 100s real-cadence
      test run to confirm the fix empirically).
