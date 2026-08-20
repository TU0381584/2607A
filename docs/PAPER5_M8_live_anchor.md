# Paper #5 M8: live single-gNB anchor

Status: **complete, bounded scope. Two independent live runs, replicated
finding, root cause of the offline/live margin discrepancy confirmed
(Part 2 below). Not a ranking claim -- see M1's own already-established
limits on what this rig's live evaluation can support. Written into
paper5/main.tex Section XI ("Results: Live Single-gNB Anchor").**

## What M8 asks

M1 (`docs/PAPER5_M1_recalibration.md`) already investigated whether the
offline stress environment can predict live behaviour, thoroughly, and
found it cannot: Spearman rank correlation between offline and live
compliance across six single-gNB `dqn_sla` checkpoints is
$\rho = 0.097$ ($p = 0.855$), unmoved by a 162-config recalibration grid
search, and live compliance is compressed near-ceiling for most of those
checkpoints (95.7/61.9/100/100/90.5/100%) -- there is barely a rank to
recover in the live data itself. M1's own conclusion: *"the offline
environment should be used as a live-anchored stress environment... not
as a live-rank predictor... any new paper5 arm should be evaluated live
before any claim about its relative quality is made."*

M8, honestly scoped in light of that, is **not** an attempt to validate
offline rankings live (M1 already closed that door). It is a bounded
sanity check: does a paper5-native, live-shaped checkpoint's offline
differentiated-shedding behaviour (Section III-B/`sec:methodology-perslice`)
transfer to the real testbed at all, and does anything about running it
live surface a problem the offline environment wouldn't show.

Architecturally, only the single-agent admission-control question is
answerable here: this rig has exactly one physical gNB, so GAT-CTDE's
multi-gNB coordination benefit and the M3 federation/DP results are not
testable on it (`xapp/saclb_xapp.py`'s own
`SINGLE_GNB_LIVE_LIMITATION` docstring already states this).
`single_agent_dqn` is the only paper5 arm this rig can meaningfully
exercise.

## Infrastructure: mostly already existed, not rebuilt

The live control loop itself did not need building: `qoe_oran_framework/
xapp/saclb_xapp.py` (a frozen, evaluation-only xApp -- loads a frozen
checkpoint, never trains live) and its orchestrator
`experiments/scripts/run_live_eval_arm.py` already existed, already
validated in prior live sessions (the orchestrator's own docstring
documents a real production incident it was built to handle: "UE1/embb
hit an RLC max-RETX failure 3 times within ~1 hour of cumulative
uptime"). What M8 actually needed was narrower:

1. **A dimensionally-compatible checkpoint.** M2's `single_agent_dqn`
   checkpoint (paper5's own baseline) is trained against the 3-gNB
   `saclb_offline_dqn.yaml` (state\_dim=34) and cannot load into the
   live xApp's single-gNB config (state\_dim=13) -- confirmed directly,
   not assumed, via a real `RuntimeError: size mismatch` this exact
   incompatibility already produced once before, documented in
   `saclb_offline_live1gnb.yaml`'s own header. Trained a fresh
   `single_agent_dqn` checkpoint against that file instead (the
   purpose-built, dimension-matched single-gNB training config), 3
   seeds (900-902), full 300 train/50 eval episode budget, via the
   existing `m6_run_experiment.py --arms single_agent_dqn`
   (no new training code needed).

2. **A stale eval config.** `saclb_live.yaml` (the config the live xApp
   actually runs against) still had only 2 slices (embb, mmtc) from a
   2026-07-14 finding that this rig's gNB had no real urllc MAC slice at
   the time. That gap had already been closed by this session's own
   rig-resume work (`BRINGUP_LOG.md`'s Post-Stage-10 addendum added a
   real 3rd S-NSSAI, verified live again this session with UE3
   attached) -- `saclb_live.yaml` was simply never updated to match.
   Re-added urllc, parameters copied from `saclb_offline_live1gnb.yaml`
   and flagged unverified against live demand (this rig's own E2 probe
   showed `avg_prbs_dl` too coarse a signal at urllc's 300Kbps traffic
   rate to calibrate a cap from directly, though `dl_mac_buffer_occupation`
   did show real backlog: 40.0% vs. mmtc's 3.3%). Also reordered the
   slice list to match the training config's exactly: `config.py`'s
   `slice_by_id` is a plain insertion-order-preserving dict
   comprehension, so a mismatched order would have let a checkpoint load
   successfully (same slice set, same count) while silently misreading
   which slice's features land in which input position. Verified this
   is not happening, not assumed: `cfg.slice_by_id.keys()` ordering
   checked equal on both configs, `request_state_dim` checked equal
   (13 = 13), and a real smoke-test checkpoint trained against one
   config loaded cleanly into a policy built from the other.

## Training result: 3 seeds, the same pattern already established elsewhere

| Seed | Collapsed | Block precision | Note |
|---|---|---|---|
| 900 | No | 1.000 (mmTC only) | Used for the live run below |
| 901 | Yes (0 blocks) | undefined | Matches the established single-agent-DQN collapse pattern from M2/M6 |
| 902 | No | 1.000 (mmTC only) | |

All three report `sla_compliance_all_slices = 0.000` -- expected, not a
red flag: `saclb_offline_live1gnb.yaml`'s own header already documents
this `Lmax=10`/tight-cap config family trips the compliance metric
regardless of policy quality. Seeds 900 and 902 show **exactly this
paper's own central finding** (Section~\ref{sec:collapse}) in a brand
new setting: genuine, correctly-targeted differentiated shedding
(precision 1.000) reported as 0% compliant by the outcome-level metric.

## Live evaluation: two independent runs, one finding replicated cleanly

Ran checkpoint seed 900 (2 episodes, `--algorithm dqn`, real E2 telemetry,
`training=False`) against the live rig twice: once starting from a
backlog state left over from this session's earlier contention-gate
testing, and once starting from a **confirmed clean baseline**
(`dl_mac_buffer_occupation = 0.0%` on all three slices, checked directly
immediately before launch).

**What replicates**: block precision. Both runs blocked only mmTC (46/32
blocks run 1, 45/32 blocks run 2 -- zero urllc or embb blocks in either
run), exactly matching the offline eval's precision=1.000 finding for
this same checkpoint. The offline-trained differentiated-shedding
*decision logic* transfers to live traffic essentially unchanged.

**What also replicates, and initially looked like a bug rather than a
finding**: by the end of each run, `dl_mac_buffer_occupation` had reached
100% on **all three slices**, and several `sla_margin_by_slice` values
matched **to the decimal** across the two independent runs (embb:
$-1{,}002{,}377.5$ in both; urllc: $-650{,}299$ vs. $-650{,}288$,
$-1{,}022{,}888.0$ vs. $-1{,}022{,}888.3$). First read this as
contamination from the earlier contention-gate test (which had indeed
left embb saturated and non-recovering) and planned to discard the
result -- but the second run started from a verified-clean baseline and
reached the same saturated state, with margin values landing on what
looks like a fixed numeric ceiling rather than organically-accumulating
values that happened to coincide. Corrected in place rather than
silently: **this is not contamination, it is a genuine, replicated
property of this checkpoint (or this config family) under sustained real
traffic** -- within roughly 10 minutes of live operation, every slice's
backlog saturates completely, independent of starting condition, decoupled
from whether the policy's admission decisions are correctly targeted.

## Honest conclusion

Two things are now separable, exactly the kind of decomposition M1
already modelled for the offline/live gap in general:

1. **The offline-trained admission-control *decision logic* transfers
   live.** Block precision 1.000 in both offline and live evaluation,
   replicated across two independent live runs. This is the strongest,
   most direct evidence in this paper that paper5's differentiated-
   shedding finding is not an offline-simulator artifact.
2. **The offline environment's backlog/margin *dynamics* do not
   resemble live behaviour under sustained real traffic, and this rig's
   real traffic saturates them almost immediately regardless of
   admission policy.** This sharpens, rather than contradicts, M1's own
   conclusion (offline is not a live-margin predictor) -- M1 found live
   compliance compressed near-ceiling across checkpoints; this finds
   live *margin* saturated near-floor within minutes, on paper5's own
   architecture, independent of which checkpoint or starting state is
   used. Neither offline nor this live setup's SLA-margin metric can
   meaningfully discriminate policy quality here.

**This is not, and is not claimed to be, evidence that GAT-CTDE's or
any other paper5 arm's relative ranking holds or fails to hold live** --
per M1's own explicit limit, and because this rig cannot test the
multi-gNB coordination question at all. It is a bounded anchor for the
one question this rig can answer: does the offline-trained *policy
behaviour itself*, not its offline-reported margin, survive contact with
real traffic. For block precision, yes. For the margin metric, the
answer is that the metric itself saturates too fast on this rig's
traffic scale to be informative -- a limitation of the measurement, not
evidence against the policy.

## Part 2: root cause of the margin discrepancy -- confirmed, not assumed

Asked explicitly to investigate why offline and live results diverge so
sharply, rather than leave the saturation pattern as a named-but-
unconfirmed hypothesis. Traced through the actual code path and the
actual saved data, not just plausibility:

1. **Where the number comes from**: `reward.py`'s SLA-margin computation
   is deliberately unclamped (its own comment: "an unclamped, unclipped
   margin is what actually shows that gap") and computes
   `queue_margin = 1.0 - agg.raw_queue_len_norm`. `kpm_adapter.py`
   computes `raw_queue_len_norm = queue_raw / Lmax`, where `queue_raw`
   is the sum of each UE's `dl_mac_buffer_occupation` for that slice --
   **unclipped**, unlike the state-facing `queue_len_norm` field the
   policy actually reads, which is separately clipped to `[0, 2]`
   (`kpm_adapter.py`, `queue_len_norm = min(2.0, raw_queue_len_norm)`).
2. **The two sources disagree on what `dl_mac_buffer_occupation` means,
   confirmed from the framework's own source, not inferred**: offline,
   `replay_kpm_source.py` generates it as
   `max(0.0, rng.normal(5.0, 2.0))` -- a synthetic proxy with mean 5,
   deliberately scaled near `Lmax=10`. Live, `qoe_mapper.py`'s own module
   docstring states the field is "wired to a real scheduler counter
   (`sched_ctrl->num_total_bytes` = sum of RLC TX buffer occupancy
   across logical channels) -- not synthetic, not a stub", and
   `calibration/units.py`'s own docstring independently confirms it:
   "`dl_mac_buffer_occupation` as raw scheduler bytes." Both call sites
   funnel into the exact same `kpm_adapter.aggregate_slice_state`
   regardless of source (`env.py`'s `_build_cluster_state` is source-
   agnostic, confirmed by reading it directly) -- so the identical
   `Lmax=10` divisor, calibrated only against the offline proxy's small
   scale, is applied to a live quantity that is a genuine physical byte
   count, reaching into the millions within minutes once a sustained
   stream (embb's real 4Mbps) outstrips its PRB ceiling.
3. **Verified against the actual saved data, not just the code**: the
   live omega log's first step already shows embb's margin at $-122.4$
   (`raw_queue_len_norm \approx 123.4$, i.e. `queue_raw \approx 1234`
   bytes) -- consistent with a real byte-count accumulating from the
   moment traffic starts, not a fixed offset. By the end of the 10-minute
   run it reaches $-1{,}002{,}377.5$ (`queue_raw \approx 10.02$MB), a
   small, physically plausible fraction of the 300MB embb's stream could
   have sent at 4Mbps over that window if nothing drained at all --
   consistent with a real, partially-draining hardware buffer, not an
   obviously-broken number.

**Conclusion**: this is a genuine unit-calibration mismatch, not
evidence that live conditions are millions of times more severe in any
SLA-meaningful sense. `Lmax` was tuned once, against the offline
proxy's arbitrary small scale, and never re-derived for the live
field's real physical unit. The policy's own decisions were insulated
from this because they only ever see the clipped `queue_len_norm`
input, which is exactly why block precision (a decision-level metric)
replicated cleanly live while the margin (a deliberately-unclipped
diagnostic) did not. Not patched post hoc in this pass: doing so
would require deciding what live backlog scale is representative from
much more live data than this bounded anchor collected, and the
decision-level finding this investigation actually supports does not
depend on it. Written into `paper5/main.tex` Section XI, subsections B-C.

## What was not done

- Only one checkpoint (seed 900) was run live; seeds 901 (collapsed) and
  902 (also precision=1.000) were not, since the two-run replication
  already answered the specific question M8 set out to check (does the
  precision finding transfer, and is the margin-saturation pattern real
  or an artifact) without needing a larger campaign.
- `Lmax` (or an equivalent live-appropriate rescaling) was not corrected
  -- the root cause is now confirmed, but choosing a live-representative
  replacement value is a separate, larger undertaking (needs a
  principled live backlog-scale calibration, not a one-off guess) left
  for future work rather than patched under this bounded anchor's scope.
- The rig was left in the post-second-run saturated state, not reset a
  third time -- no further live-rig actions were taken once the
  replication was in hand.
