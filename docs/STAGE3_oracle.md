# Stage 3 — the static-at-cap oracle arm

**Question this stage exists to answer:** does DQN differ from a trivial
"park every slice's ceiling at its calibrated cap and never adapt"
provisioning oracle under constant live load, on any metric (compliance,
U, violation severity, inferred MOS)? Or does DQN's converged live-campaign
behaviour ("ride the ceiling to cap") just rediscover this fixed policy?

**Answer, now backed by a fully-powered, reverified comparison: yes, DQN
differs, and the difference is now statistically significant (p=0.0149),
not just directional.** static_at_cap collapses into a catastrophic
backlog-blowup regime in 4 of 15 episodes (27%) — statistically
indistinguishable from `baseline`'s own collapse rate (11/15, exactly
tied, p=1.0). DQN, re-evaluated across **5 distinct seeds and 25
episodes total (950–954)**, never collapsed once. Two brand-new seeds
(953, 954) that DQN had never been evaluated on before were run
specifically to rule out "the original 3 seeds got lucky" — the result
replicated cleanly (10/10 fully compliant, zero collapses).

---

## 1. What was run, honestly scoped, in two rounds

**Round 1** (2 seeds, ~1h budget): static_at_cap on seeds 950, 951.
**Round 2, reverification** (user: *"truly verify whether DQN is
superior to the baseline by preventing collapsing. run more tests to
confirm before proceeding."*): two additions run back-to-back on one
further rig session (~1.5h) —
1. static_at_cap's originally-planned 3rd seed (952), completing it to
   **n=15 episodes**, matching every other arm's sample size exactly.
2. DQN(SLA)'s existing seed-256 checkpoint re-evaluated on **2 brand-new
   live seeds (953, 954)** it had never been run on before — the
   original campaign only ever used 950/951/952. Written to a separate
   arm name (`dqn_sla_reverify`) so the original `dqn_sla` data
   (950/951/952) was never touched, overwritten, or mixed with this new
   data.

**Mechanism** (unchanged from the initial report): `static_at_cap` is a
config-only addition
(`experiments/configs/saclb_campaign_static_at_cap.yaml` — every slice's
`nominal_ratio` set equal to its `max_ratio_cap`, `ceiling_step_ratio: 0`
unchanged) run through the exact same, unmodified entrypoint as the
`baseline` arm (`run_baseline_static.py`'s `AlwaysAcceptPolicy` — no
admission-control decision of its own). No framework or orchestrator
source touched in either round.

**Operational notes:** pre-flight probe passed cleanly before both
rounds (no drift from calibration: eMBB ~14–15 PRB mean/17–21 max vs.
calibrated ~15 mean/5–23 range; URLLC/mMTC both at the ~5 PRB floor
every time). `iperf3-target`'s port 5201 server wedged once (same known
failure mode already documented in `CAMPAIGN_LOG.md`, same fix: recreate
the container). Routine health-check-triggered RAN-stack restarts
happened between most batches (matches the pattern of the original
15-arm campaign, not a new instability). Rig fully torn down after each
round: 0 containers, 0 stray RAN processes, confirmed directly both
times.

Raw logs:
`experiments/results/live_campaign/static_at_cap/sla/rep_seed{950,951,952}/omega_log.jsonl`,
`experiments/results/live_campaign/dqn_sla_reverify/sla/rep_seed{953,954}/omega_log.jsonl`.
Recomputed metrics: `experiments/scripts/metrics_stage3.py` →
`docs/stage3_metrics_raw.json` (`baseline`/`dqn_sla`/`dqn_qoe`'s original
950–952 numbers reused as-is from `docs/stage2_metrics_raw.json`, never
recomputed or altered).

---

## 2. The headline result: episode-level bimodality, and it replicates across seeds

Per-episode compliance (all 3 slices), static_at_cap, in run order:

| Seed | Episodes 1–2 | Episodes 3–5 |
|---|---|---|
| 950 | **collapsed** (33–43% / 0–2% / 0–5%) | 100% / 100% / 100% |
| 951 | 100% / 100% / 100% | 100% / 100% / 100% |
| 952 | **collapsed** (33–37% / 0% / 0–3%) | 100% / 100% / 100% |

**11 of 15 episodes are perfectly compliant on all 3 slices. 4 collapse
almost completely** — and in every one of the 2 seeds where a collapse
happened, it happened in exactly the same place: the first batch of
that seed's rotation, immediately after a fresh `drain_backlog.sh`.
Seed 951 never collapsed at all. This is not randomly scattered noise —
it is a reproducible signature (URLLC's SLA margin during collapse:
consistently in the −650,000 to −1,022,888 range across both seeds 950
and 952 independently) matching the same catastrophic backlog-blowup
regime Stage 2 already found in `baseline`'s per-step data.

**static_at_cap's collapse rate (4/15 = 27%) is statistically identical
to `baseline`'s (4/15 = 27%, exactly tied — Fisher exact p=1.0, odds
ratio=1.0).** Parking the ceiling at the calibrated cap instead of the
nominal ratio provides **zero measurable collapse-avoidance benefit** on
its own. The collapse is a property of `AlwaysAcceptPolicy` (never
rejecting, regardless of backlog), not of where the (immovable) ceiling
sits.

## 3. DQN's zero-collapse record, reverified on seeds it had never seen

| DQN(SLA) sample | Seeds | Episodes | Fully compliant |
|---|---|---|---|
| Original campaign | 950, 951, 952 | 15 | 15/15 |
| **Reverification (new)** | **953, 954 (never used before)** | **10** | **10/10** |
| **Combined** | **950–954 (5 distinct seeds)** | **25** | **25/25** |

Zero collapses were observed on the 2 fresh seeds — the same clean
100%/100%/100% result on every single episode, matching the original
campaign's texture exactly (severity distributions are statistically
identical too — median 1.0/0.7/0.7, near-zero IQR, same as the original
15-episode record). This directly answers the concern behind "run more
tests to confirm": DQN's collapse-avoidance is not an artifact of the 3
seeds used in the original campaign.

## 4. Significance: now clears p<0.05

| Comparison | Table (compliant, failed) | p-value |
|---|---|---|
| static_at_cap (11/15) vs baseline (11/15) | [[11,4],[11,4]] | **1.0** — no evidence of any difference |
| static_at_cap (11/15) vs DQN, original only (15/15) | [[11,4],[15,0]] | 0.0996 — not significant (same power problem the paper already discloses for its other live comparisons) |
| **static_at_cap (11/15) vs DQN, combined (25/25)** | **[[11,4],[25,0]]** | **0.0149 — significant at p<0.05** |
| DQN reverify alone (10/10) vs baseline (11/15) | [[10,0],[11,4]] | 0.125 — directionally consistent, underpowered alone (small n by design; the value of this sample is the *replication*, not a standalone test) |

**The reverification round is what moved this from "suggestive" to
"significant."** The original static_at_cap-vs-DQN comparison (8/10 vs
15/15, then 11/15 vs 15/15) never cleared p<0.05 on its own — it needed
the additional, independently-collected DQN evidence (25 episodes
across 5 seeds, not 15 across 3) to do so. This is a case where "run
more tests to confirm" was the right call methodologically, not just
due diligence: the claim was true but underpowered before this round.

## 5. Severity in the stable regime: still statistically indistinguishable from DQN

Restricting to episodes where static_at_cap is NOT collapsing, its
per-step SLA margin texture (median 1.0/0.7/0.7, IQR≈0) continues to
match DQN's (median 1.0/0.7/0.7, IQR≈0) almost exactly — confirmed again
in the reverification round's DQN data. **DQN's advantage is not a
smoother steady state — it is that it has no unstable state at all.**

---

## 6. Direct answer to the stage's question

**Does DQN differ from static-at-cap under constant load, on any
metric?** Yes, and this is now a statistically significant, reverified
finding: DQN has never collapsed once across 25 episodes and 5 distinct
seeds (three from the original campaign, two run specifically to rule
out seed-selection luck); static_at_cap collapses at the same rate as a
non-learning `baseline` arm (27%, statistically tied, p=1.0) despite
starting every episode with its ceiling already at the calibrated cap.
**The mechanism is the admission decision, not the ceiling value:**
`static_at_cap` inherits `AlwaysAcceptPolicy` from `baseline` — neither
arm ever rejects, and neither arm avoids the collapse. DQN's real
contribution is learning *when to reject* under backlog pressure, which
a fixed ceiling — however generously set — cannot do.

## 7. Left for Stage 7 (not decided here, out of this stage's scope)

- Whether to add a `static_at_cap` row to Table II, now that it has a
  full n=15 record directly comparable to the other arms.
- How this interacts with the Stage 2 finding that `baseline` beats DQN
  on U in the *congested* (offline) scenario — a different experiment;
  should not be conflated when Section IV-C is rewritten.
- Whether the collapse's exact trigger (something specific to the first
  batch immediately after a fresh drain, on 2 of 3 seeds) is worth its
  own root-cause note, or is adequately explained by
  "AlwaysAcceptPolicy + no admission control" as already established.

## Acceptance test — actual status

- [x] Fourth arm added through configuration only, no framework or
      orchestrator source touched.
- [x] Run under the full protocol: 3 seeds × 5 episodes for
      static_at_cap (n=15, matching every other arm), plus a targeted
      2-seed reverification of DQN(SLA) on previously-unused seeds.
- [x] `docs/STAGE3_oracle.md` answers the question directly: DQN differs
      from the trivial oracle on collapse-avoidance, a statistically
      significant result (p=0.0149) once the reverification round's data
      is included — not on steady-state per-step severity, where the two
      are indistinguishable.
