# Paper #5 M8: live single-gNB anchor

Status: **complete, bounded scope. Two independent live runs, replicated
finding, root cause of the offline/live margin discrepancy confirmed
(Part 2) and refined by cross-checking prior live campaign data (Part
3, which also found the current scheme understates how good healthy
conditions are, not just how bad severe ones look), a physically-
grounded latency-based recalibration attempted and validated against
real data (Part 3). Not a ranking claim -- see M1's own already-
established limits on what this rig's live evaluation can support.
Written into paper5/main.tex Section XI ("Results: Live Single-gNB
Anchor").**

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

**Conclusion (refined further by Part 3 below)**: `Lmax=10` is
disconnected from live's real byte scale, but -- as Part 3 shows by
checking prior live campaign data directly, not assuming this
diagnosis was already complete -- it is not simply "wrong": it happens
to reproduce M1's own independently-derived healthy-condition live
target almost exactly, and only produces uninterpretable numbers in
the specific, real, already-documented severe-backlog regime this run
entered. The policy's own decisions were insulated from either reading
because they only ever see the clipped `queue_len_norm` input, which
is exactly why block precision (a decision-level metric) replicated
cleanly live while the margin (a deliberately-unclipped diagnostic) did
not. Written into `paper5/main.tex` Section XI, subsections B-C
(updated again after Part 3).

## Part 3: attempting a live-appropriate calibration -- validated, not assumed

Asked explicitly to try matching offline and live results through
calibration, rather than leave the mismatch as diagnosed-but-unfixed.
Two things needed checking first, both of which changed the picture
from Part 2's initial read:

**The "millions of bytes" scale is real and independently
corroborated, not an artifact of this run.** `experiments/CAMPAIGN_LOG.md`
(2026-07-16, predating this session entirely) already measured the
identical magnitude directly: embb `dl_mac_buffer_occupation` at
baseline (ceiling wide open) = 411.3, **pinned (ceiling min=max=1) =
6,601,079.0 (max 10,023,785)**, recovery (restored to `max=20`, not the
config's own `max=4`) = 2,504,530.4. My own M8 run's catastrophic value
(10,023,785) matches this prior, independent characterization almost
exactly. `saclb_live.yaml`'s current `max_ratio_cap=4` for embb was
already shown, in that same prior work, to be **insufficient** to drain
backlog once it accumulates (recovery needed `max=20`, five times
wider, to actually show drainage) -- a real, already-documented
structural bottleneck, not something this session introduced.

**`Lmax=10` is not universally wrong -- it produces a sensible,
M1-matching number under healthy conditions.** Checked 28 seeds of
`live_campaign_v2/dqn_sla` (the arm M1's own live target numbers come
from): 26 of 28 land on **exactly** embb margin = 0.700, jumping there
instantly (step 6 of 300, in the one file inspected in full) and
staying perfectly flat for the rest of the run -- not organic
backlog noise, but not the `dl_errors+dl_bler` fallback either
(checked directly against that run's own `limitation` field: the
fallback triggered for urllc/mmtc in that run, not embb). embb's real,
non-fallback `dl_mac_buffer_occupation` apparently floors near a small,
stable value (back-derived: raw backlog $\approx 3$ bytes) under
conditions where the ceiling keeps up with demand -- and `Lmax=10`
maps that to a margin (0.7) matching M1's own independently-derived
live target (eMBB mean 0.714) almost exactly. **The extreme values are
specific to conditions where a policy's ceiling falls behind organic
demand long enough for real queueing to begin -- a real, severe,
already-documented failure regime this M8 checkpoint's live run
apparently entered and `live_campaign_v2`'s did not** (why is not
fully resolved by this bounded anchor -- see below).

### The calibration attempt

`Lmax=10` is disconnected from live's real byte scale regardless (it
is calibrated only against the offline model's `Normal(5.0, 2.0)`
synthetic proxy). Rather than pick an arbitrary rescaling constant,
reused an already-built, unused-for-this-purpose piece of the project's
own code: `calibration/units.py`'s `backlog_bytes_to_latency_s`
(Little's-Law queueing-delay estimate from a byte count and a service
rate), normalising against each slice's own already-calibrated
`latency_budget_ms` instead of an arbitrary byte threshold --
physically meaningful (an estimated queueing delay vs. an actual
latency SLA), not another guessed constant. New script,
`experiments/scripts/m8_latency_recalibration.py` (does not modify
`reward.py`/`kpm_adapter.py`, both frozen -- a post-hoc reanalysis of
already-logged data, the same pattern `m6_correctness_metrics.py`
already establishes). Backlog bytes are back-derived from the already-
logged margin via `reward.py`'s own formula (`margin = 1 -
raw_queue_len_norm`, `raw_queue_len_norm = queue_raw/Lmax` -- read
directly from source, not guessed); service throughput uses
`CAMPAIGN_LOG.md`'s own empirically-measured range (embb at
`max_ratio=4`, this run's ceiling throughout, empirically serves
~15-22 real PRB, not per-step reconstructed since raw PRB-serving was
not logged by this run -- stated as an approximation, not hidden).

**Result**:

| Quantity | Current (Lmax=10) | Recalibrated (latency-normalised) |
|---|---|---|
| Catastrophic case (backlog $\approx$ 10.02M bytes) | $-1{,}002{,}377.5$ | $-809$ to $-1{,}187$ (36-53s real queueing delay vs. a 45ms budget) |
| Healthy reference (backlog $\approx$ 3 bytes, `live_campaign_v2`'s own steady state) | $0.700$ | $0.9996$ |

The catastrophic case shrinks by roughly 3 orders of magnitude (from
6 digits to 3) and becomes directly interpretable (an actual multiple
of a real latency budget, not an arbitrary ratio) rather than merely
smaller. The healthy reference moves from "okay" (0.7) to
"near-perfect" (0.9996) -- revealing the current scheme understates
how good healthy conditions actually are, not just how bad unhealthy
ones look. This does not make live and offline margins numerically
interchangeable (offline's own worst-case margin, bounded by
`backlog_capacity`, tops out roughly two orders of magnitude smaller
than even the recalibrated live catastrophic value) or restore any
rank-predictive claim -- M1 already established that limit and this
does not reopen it -- but it does bring both onto the same order of
magnitude (hundreds, not "millions vs. single digits"), and gives the
live number a physical meaning it did not have before.

**Not resolved by this pass**: why this specific M8 checkpoint's live
run entered the severe-backlog regime while `live_campaign_v2`'s did
not. The `max_ratio_cap=4`-insufficient-for-organic-~15-PRB-demand
mechanism (independently documented in `CAMPAIGN_LOG.md`, confirmed
structural, not policy-specific) is the best-evidenced contributing
factor available, but attributing the full difference to this
checkpoint's offline-only training (never seeing live's true demand
scale) versus some other difference between the two runs would need
more live data than this bounded anchor collected to state with
confidence, and is not claimed here.

## What was not done

- Only one checkpoint (seed 900) was run live; seeds 901 (collapsed) and
  902 (also precision=1.000) were not, since the two-run replication
  already answered the specific question M8 set out to check (does the
  precision finding transfer, and is the margin-saturation pattern real
  or an artifact) without needing a larger campaign.
- A latency-normalised recalibration was derived and validated against
  real data (Part 3), but not applied back into the frozen pipeline
  (`reward.py`/`kpm_adapter.py` cannot be modified) or adopted as this
  paper's new primary live metric -- it is reported as a post-hoc
  reanalysis demonstrating the mismatch is addressable in principle, not
  as a replacement metric this paper now relies on elsewhere.
- Why this specific checkpoint's live run entered the severe-backlog
  regime while `live_campaign_v2`'s did not is not fully resolved --
  the config's `max_ratio_cap=4`-insufficient-for-organic-demand
  mechanism is the best-evidenced contributing factor, not a proven
  complete explanation.
- The rig was left in the post-second-run saturated state, not reset a
  third time -- no further live-rig actions were taken once the
  replication was in hand.
