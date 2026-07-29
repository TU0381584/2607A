# Stage 5 — Recalibration, retraining, and a directional live re-run

Three independent, real, well-evidenced problems were found and fixed in
this stage, in the order they surfaced. A full statistically-powered
live re-run (3 seeds × 5 episodes × 4 arms, ~5-6h) remains **deferred by
explicit user request** ("do the full run, but in 1 hour. we'll do the
6 hour one later") — this document's live numbers are a 1-hour,
1-seed/2-episode directional trial only, not a replacement for Stage 2/3's
statistically powered tables.

---

## 1. Bug #1: eMBB's MOS calibration used the wrong throughput scale

**Symptom:** eMBB was 100% SLA-compliant in every DQN run (Stage 2/3),
yet its own inferred MOS sat pinned near the floor of the 1–5 scale
(~1.19–1.23) regardless of policy quality.

**Root cause:** the frozen `env.py` feeds `agg.prb_used_ratio`
(`prb_sum / B`, B=100 — a slice's PRB usage as a fraction of the gNB's
*total* capacity) into `iqx_mos`'s `throughput` parameter
([env.py:313](../framework/qoe_oran_framework/env.py#L313)). The frozen
calibration script (`calibration/fit_iqx.py`) fits and samples that same
parameter as an **absolute PRB/UE count** in [0.05, 5.0] — confirmed by
`units.py`'s own docstring ("avg_prbs_dl as a PRB count") and its
`prb_to_kbps` conversion. eMBB's real demand (~15 PRB, measured
repeatedly this session) divided by B=100 gives ratio≈0.15 — a number
the old calibration interpreted as "an almost-starved 0.15 PRB/UE
connection," not "well-served, near its 12-PRB ceiling." eMBB's fitted
`epsilon` (199.9999) sat exactly at `fit_iqx.py`'s own box constraint —
a second, independent sign of a degenerate fit.

**Fix:** `experiments/scripts/recalibrate_iqx.py` (new, non-frozen) —
reuses `fit_iqx.py`'s importable building blocks (`iqx_mos`, `IqxCoeffs`,
`p1203_mos_from_throughput_trace`) but samples "throughput" as the RATIO
`prb_used_ratio` actually is, converting ratio→real-PRB→kbps correctly
(this rig runs exactly 1 UE/slice, so `ratio × B` recovers the true
per-UE PRB count) before generating P.1203 labels.

**Verified, before/after, at real observed ratios:**

| Ratio (real, observed) | Old MOS | New MOS |
|---|---|---|
| 0.05 | 1.000 | 2.294 |
| 0.15 | 1.000 | 3.624 |
| 0.24 | 1.000 | 4.111 |

Fit quality: pearson_r(test)=0.987, MAE(test)=0.122 — as good as or
better than the original (0.926).

## 2. Bug #2: URLLC/mMTC's calibration used the wrong SLA deadlines

**Root cause:** `fit_iqx.py`'s `generate_urllc_dataset`/
`generate_mmtc_dataset` call `acr_score_urllc(latency_s, packet_loss)` /
`acr_score_mmtc(packet_loss, latency_s)` with **no** `deadline_s`/
`loss_budget` override — so objective labels were scored against
`acr_scoring.py`'s generic defaults (URLLC 5ms/0.1% loss; mMTC 1000ms/5%
loss), not this campaign's real configured per-slice SLA thresholds
(`saclb_campaign.yaml`: URLLC 20ms/0.5%; mMTC 65ms/3.5%).

**Verified, before/after, at each slice's own REAL deadline:**

| Slice | Test point | Old MOS (wrong deadline) | New MOS (real deadline) |
|---|---|---|---|
| URLLC | latency = 20ms (its real deadline) | 1.000 (total failure) | 3.525 (sensible borderline) |
| mMTC | latency = 65/130ms | 5.000 / 5.000 | 5.000 / 5.000 (see note) |

mMTC shows no practical difference — its own generator
(`acr_score_mmtc`) only applies a small (≤15%) linear latency discount by
design ("delay-tolerant applications barely notice it"), so the wrong
deadline was real but low-impact for this specific slice; URLLC's steep
sigmoid gate makes its deadline error highly consequential, and that one
resolves cleanly.

Both bugs fixed in one script; four corrected config siblings created
(`saclb_campaign_v2.yaml`, `saclb_campaign_baseline_v2.yaml`,
`saclb_campaign_static_at_cap_v2.yaml`, `saclb_offline_campaign_v2.yaml`)
— originals untouched for reproducibility of every previously-reported
number. Raw: `docs/stage5_recalibration_raw.json`.

## 3. Bug #3: the offline training environment's demand scale doesn't match live

Retraining `dqn_qoe` under the Bug #1/#2 fix ALONE left its reward curve
flat (Q1→Q4 delta: +0.004, indistinguishable from the original +0.007).
Root cause, matching a concern this session's own Stage 0 audit already
flagged: `qoe_oran_framework/scripts/train_offline.py`'s frozen
`OVERSUBSCRIPTION_FACTOR × nominal_ratio` formula produces mean offered
demand of ~3.75% of gNB capacity for eMBB — 4× smaller than what real
live traffic actually produces (~15%, measured repeatedly). The
corrected MOS calibration had nothing to differentiate within the range
the offline simulator ever visited.

**Fix, validated zero-training before any retraining** (same discipline
as the earlier admission-efficiency workstream):
`experiments/scripts/live_scale_offline_env.py` (new, non-frozen) builds
`ClosedLoopKpmSource` with `mean_offered_ratio` set directly from real,
repeated live probe measurements (embb=0.15, urllc=0.05, mmtc=0.05),
while keeping `saclb_campaign_v2.yaml`'s REAL cap/nominal structure
(12/4/3, 3/2/2 — deliberately NOT rescaled to "tens of units" the way
the unrelated admission-efficiency environment was, so a policy trained
here is about the same MDP the live rig actually presents).

Lmax=10 (live-appropriate) saturated the offline synthetic backlog
immediately regardless of policy at this demand scale (accept_all vs.
reject_all statistically indistinguishable, <5% compliance for both) —
matching the exact failure mode the admission-efficiency workstream
diagnosed before. Swept Lmax/backlog_capacity (offline-training-only;
the live config's own Lmax=10 is untouched) and validated
Lmax=1000/backlog_capacity=2000:

| Policy | eMBB | URLLC | mMTC | Mean reward |
|---|---|---|---|---|
| accept_all | 26.3% | 43.2% | 31.9% | -0.147 |
| static_threshold | 14.4% | 28.2% | 29.2% | -0.275 |
| reject_all | 13.9% | 28.2% | 29.6% | -0.287 |

Real, non-saturated, sensibly-ordered differentiation on both
compliance and reward (embb/urllc clearly differentiate; mmtc's spread
is narrower but not saturated). `experiments/results/live_scale_offline/baseline_validity.md`.

## 4. Retraining, under all three fixes together

`experiments/scripts/train_offline_live_scale.py` (new, mirrors
`train_offline.py`'s CLI), 300 episodes, seed 256, both reward modes:

- **dqn_sla**: blocks-per-episode fell from ~67 (episode 1) to a ~10-30
  range by episode 300 — a clear, healthy convergence signature.
  (Epsilon-decay separately verified correct for this training lineage:
  `mc_runner.run_single`'s `on_episode_end()` call — the mechanism the
  2026-07-24 congested/diurnal bugfix added to those *other* scripts —
  was already present when this checkpoint lineage was first trained,
  confirmed via commit history and by the ORIGINAL v1 dqn_sla checkpoint's
  own healthy Q1→Q4 reward improvement, +1.698.)
- **dqn_qoe**: mean_mos Q1→Q4 delta +0.103 (1.524→1.627) — modest but
  real, a genuine change from the original's floor-pinned, non-responsive
  signal. sla_viol rose slightly (+0.057) alongside it — a real,
  reportable tradeoff (the corrected reward appears to let the agent
  trade a little compliance for MOS), not force-fit into a cleaner story.

New checkpoints: `experiments/results/offline_v2/{sla,qoe}/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt`.

## 5. The 1-hour live trial (directional only — n=2 episodes/arm, seed 950)

Two unrelated operational problems surfaced and were fixed mid-session,
neither caused by the calibration/environment work above:

- **`health_check.sh`'s segfault check was a false-positive trap**: it
  grepped the last 200 `dmesg` lines for ANY "segfault", with no
  time-window or process filter. An unrelated `iperf3` traffic-client
  crash (`libiperf.so`, a host-side generator process, not a RAN
  process) poisoned every subsequent health check, causing 3 of 4 arms
  to fail with "rig unhealthy after max restart attempts" even though
  the actual RAN stack was fine throughout. **Fixed** (restricted the
  match to `nr-softmodem`/`nr-uesoftmodem` specifically — the only
  processes this check should ever act on).
- **`iperf3-target`'s port 5201 server wedged again** — the same known,
  previously-documented failure mode (`CAMPAIGN_LOG.md`), same fix
  (recreate the container).

After both fixes, all 4 arms completed (2 episodes each, seed 950):

| Arm | eMBB compl. | URLLC compl. | mMTC compl. | eMBB MOS | URLLC MOS | mMTC MOS | Note |
|---|---|---|---|---|---|---|---|
| baseline_v2 | 0.8% | 0.0% | 0.8% | 1.37 | 1.00 | 4.35 | Catastrophic collapse, both episodes |
| dqn_sla_v2 (retrained) | 35.8% | 0.0% | 1.7% | 2.64 | 1.00 | 4.30 | URLLC collapsed both episodes; eMBB partial |
| dqn_qoe_v2 (retrained) | 100% | 100% | 100% | 2.17 | 3.16 | 4.76 | **drift-flagged**: URLLC block rate 33.5/ep, >>historical norm |
| static_at_cap_v2 | 100% | 100% | 100% | 2.17 | 4.53 | 4.89 | Landed in the "good" regime this draw |

**Read this table as directional evidence only, not a result.** n=2
episodes/arm on one seed is exactly the sample size that already fooled
an earlier stage this session (`static_at_cap`'s first smoke trial
sampled the unlucky batch and reported a misleadingly bad number that
the full 3-seed run corrected). Two things ARE worth taking away even at
this sample size, because they're about calibration honesty, not
statistical power:

1. **MOS numbers are no longer floor-pinned.** eMBB's MOS now ranges
   2.17–2.64 across arms instead of a uniform, uninformative ~1.2 — the
   calibration fix is doing its job regardless of which arm "wins."
2. **dqn_qoe's behavior changed materially** under the corrected,
   harder (real-demand-scale) environment: it now rejects aggressively
   enough to trip the orchestrator's own drift-flag (33.5 URLLC
   blocks/episode vs. a 1.0/episode historical bar) while still holding
   100% compliance — a genuinely different operating point than the v1
   "ride every ceiling to 100%, reject almost nothing" behavior. Whether
   this holds up, and whether dqn_sla's URLLC collapse in this draw is
   real or an unlucky sample (same live-hardware non-determinism already
   documented — the same seed does not force identical outcomes on real
   RF, unlike a pure simulator), is exactly what the deferred, properly
   powered 3-seed × 5-episode × 4-arm campaign will settle.

Raw logs: `experiments/results/live_campaign_v2_trial/{baseline,dqn_sla,dqn_qoe,static_at_cap}/*/rep_seed950/omega_log.jsonl`.

---

## 5b. The 3-hour campaign (n=6 episodes/arm, 3 real seeds — properly cross-seeded)

User: *"proceed with the longer run at 3 hours, no need for 6 hours."*
Scope cut from the full 3-seed × 5-episode protocol (episodes per seed
reduced 5→2, not seed count, so real seed-to-seed hardware variance is
still captured) — 4 arms × 3 seeds × 2 episodes = 24 episodes. All 12
(arm, seed) blocks completed cleanly, no failures, no restarts needed,
total wall-clock ~2h14m (under the 3h budget). Metrics:
`experiments/scripts/metrics_stage5_v2.py` →
`docs/stage5_v2_campaign_metrics_raw.json`. Figures:
`experiments/plots/out/stage5_v2_campaign_fig2_sla_compliance.{png,pdf}`,
`experiments/plots/out/stage5_v2_campaign_fig_mos_and_compliance.{png,pdf}`.

| Arm | eMBB | URLLC | mMTC | U | Episodes fully compliant | eMBB MOS | URLLC MOS | mMTC MOS |
|---|---|---|---|---|---|---|---|---|
| baseline | 66.9% | 66.9% | 67.2% | 67.0 | 4/6 | 1.91 | 3.38 | 4.73 |
| DQN (SLA) | 100.0% | 100.0% | 100.0% | 100.0 | 6/6 | 2.17 | 4.05 | 4.87 |
| DQN (QoE) | 100.0% | 100.0% | 100.0% | 100.0 | 6/6 | 2.17 | 4.72 | 4.90 |
| static-at-cap | 100.0% | 100.0% | 100.0% | 100.0 | 6/6 | 2.17 | 4.83 | 4.90 |

**This directly corrects the 1-hour trial's misleading dqn_sla result.**
The n=2/1-seed trial showed DQN-SLA catastrophically failing on URLLC
(0% compliance) — with 3 real seeds, DQN-SLA is actually 100% compliant
on every slice, 6/6 episodes. That single-seed draw was exactly the kind
of unlucky sample this project has repeatedly had to catch and correct
(the same pattern as `static_at_cap`'s first smoke trial in Stage 3).

**One finding is NOT resolved by this campaign, and is flagged rather
than glossed over: `static_at_cap` also shows 100% compliance here (6/6),
identical to both DQN arms.** Stage 3 established a real, statistically
significant (p=0.0149) collapse-rate difference between DQN (0/25
episodes ever collapsed) and `static_at_cap` (4/15 collapsed, ~27%) under
the OLD calibration. This campaign's n=6 sample not showing a single
`static_at_cap` collapse is fully consistent with that same ~27% true
rate simply not being sampled at n=6 (binomial: ~15% chance of zero
collapses in 6 draws even if the true rate is unchanged) — it is NOT
evidence the collapse-proneness went away, and should not be reported as
such. Whether Stage 3's collapse-avoidance finding still holds under the
v2 calibration/retrained checkpoints is an open question this campaign's
sample size cannot settle; the original full 3-seed × 5-episode protocol
(or more) applied to `static_at_cap` specifically would be the direct way
to check.

MOS numbers continue to look healthy and non-degenerate across the
board — this is the same qualitative confirmation as the 1-hour trial,
now on a real cross-seed sample rather than one draw.

## 5c. Validation round: does DQN's collapse-avoidance edge hold under v2?

User: *"proceed to a bigger test to validate."* Extended the two arms
needed to answer this directly, at the FULL 5-episode protocol (not the
3h campaign's 2-episode compression): `static_at_cap_v2` on 3 new seeds
(953/954/955, n=6→21) and `dqn_sla_v2` on 2 new seeds (953/954, n=6→16).
All 5 new (arm, seed) blocks completed cleanly, ~2h29m.
`experiments/plots/out/stage5_v2_validated_fig2_sla_compliance.{png,pdf}`.

| Arm | n episodes | Episodes fully compliant |
|---|---|---|
| dqn_sla | 16 | **16/16 (100%)** |
| static_at_cap | 21 | **19/21 (90.5%)** — 2 real collapses |

**Both halves of Stage 3's original finding reproduce directionally under
v2:** static_at_cap is NOT collapse-immune (2 real collapsed episodes
showed up once the sample got big enough to catch them — visible as a
distinct low cluster in the compliance figure, the same bimodal signature
Stage 2/3 already documented); DQN-SLA has a clean 16/16 record, zero
collapses.

**Statistical significance, reported precisely rather than rounded up:**
Fisher exact test, 16/16 vs 19/21 → **p = 0.495 — NOT significant** at
this sample size. This is lower confidence than Stage 3's original
p=0.0149 (which used n=15 static_at_cap / n=25 DQN under the old
calibration) — mainly because static_at_cap's observed collapse rate
here (2/21 ≈ 9.5%) came in lower than Stage 3's v1 rate (4/15 ≈ 27%), so
the gap this round needed to detect was smaller. **Honest conclusion:**
the DIRECTION of Stage 3's finding replicates (DQN: 0 collapses; static_at_cap:
some collapses, not 0) but this validation round's sample is not large
enough to call the gap statistically proven under v2 — a real, reportable
result, not a null one, but weaker than Stage 3's original claim.
Closing this fully would need either more static_at_cap episodes (to
pin down its true v2 collapse rate more precisely) or accepting the
directional replication as sufficient corroboration.

## 6. What's deferred, explicitly

**The full live re-run (4 arms × 3 seeds × 5 episodes, ~5-6h) — user's
own words: "we'll do the 6 hour one later."** Until that runs:
- Stage 2/3's existing tables and findings (baseline vs. DQN, static_at_cap's
  collapse-rate comparison) remain the paper's citable numbers — they were
  computed under the OLD calibration, but Bug #1/#2 only affect *reported
  MOS diagnostics*, not the SLA-compliance/margin numbers those findings
  are actually built on (compliance is a separate code path,
  `per_slice_compliant`/`per_slice_sla_margin`, untouched by iqx_coeffs).
- The v2 trial's compliance numbers above are NOT a replacement for
  those tables and should not be cited as one.

## Acceptance status

- [x] MOS/SLA-threshold calibration bugs diagnosed with direct code
      evidence (not assumed) and fixed via non-frozen config overrides.
- [x] Fix verified before/after at real observed operating points for
      all 3 slices.
- [x] Offline training environment's demand-scale mismatch diagnosed,
      fixed, and validated zero-training before any retraining spent.
- [x] Both DQN arms retrained under the corrected environment; dqn_sla's
      epsilon-decay separately confirmed correct (not assumed) via
      commit history and reward-trend evidence.
- [x] A 1-hour directional live check run across all 4 arms, honestly
      labeled as directional, with two unrelated operational bugs found
      and fixed along the way.
- [x] A 3-hour, properly cross-seeded campaign (4 arms × 3 seeds ×
      2 episodes, n=6/arm) — corrected the 1-hour trial's misleading
      single-seed dqn_sla result (0%→100% URLLC compliance once 3 real
      seeds are sampled instead of 1).
- [x] Whether Stage 3's DQN-vs-static_at_cap collapse-avoidance finding
      still holds under v2 — validated with a targeted follow-up
      (static_at_cap n=21, dqn_sla n=16). Direction replicates (DQN 0
      collapses, static_at_cap 2 real collapses); statistical
      significance does NOT replicate at this sample size (p=0.495 vs.
      Stage 3's p=0.0149) — reported precisely, not rounded up to
      "confirmed."
- [ ] The full 3-seed × 5-episode (n=15/arm) live re-run for baseline and
      dqn_qoe specifically (still at n=6 each) — dqn_sla and
      static_at_cap are now at n=16/n=21 respectively via the validation
      round above, ahead of the other two arms.
