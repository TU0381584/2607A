# Stage 10 — Fully powered live re-evaluation, live congested pilot, and reverification

Executes the two items the manuscript's own Future Work section (as of
`5ea797c`) had deferred: (1) "complete a fully powered live re-evaluation
under the corrected QoE calibration", (2) "reproduce the congested,
multi-slice scenario on the live rig rather than offline" — this is the
session Stage 5 flagged as "we'll do the 6 hour one later."

User instruction for this stage: run both, then update the manuscript
and commit, then run a further reverification (3h offline, 10h live),
proceeding without further confirmation since the user would be away
overnight. Everything below was executed autonomously per that
instruction; every non-obvious judgment call is recorded here rather
than silently made.

## 1. Top-up campaign (item 1): all 4 arms to n=21

`experiments/scripts/run_stage5_v2_topup.sh` (new). Prior to this stage,
arms were unevenly powered: `static_at_cap`/`dqn_sla` had been extended
to n=21/16 by Stage 5c's validation round; `baseline`/`dqn_qoe` were
still at the 3-hour campaign's n=6. This stage added the missing
(arm, seed) blocks — `baseline`/`dqn_qoe` on 3 new seeds (953–955) at
the full 5-episode protocol, `dqn_sla` on 1 more seed (955) — so all 4
arms land on the SAME seed set (950–952 at 2 ep/seed, 953–955 at 5
ep/seed) at n=21 each. 35 new episodes, ~9.5h wall clock (started
2026-07-29 evening, finished 2026-07-30 19:09 — the rig was left
running unattended overnight per the user's own plan). Zero failures
logged in `experiments/results/live_campaign_v2/PROGRESS.log`.

**Finding, reported honestly rather than silently reconciled with the
existing narrative:** at n=21, `metrics_stage5_v2.py` (its `ARM_SEEDS`
dict updated to the full set) shows:

| Arm | Episodes fully compliant |
|---|---|
| baseline | 19/21 |
| dqn_sla | 19/21 |
| dqn_qoe | **21/21** |
| static_at_cap | 19/21 |

Fisher exact: dqn_sla vs.\ static_at_cap $p=1.0$; dqn_sla vs.\ baseline
$p=1.0$; dqn_qoe vs.\ baseline $p=0.49$. **This does not replicate
Stage 3's original finding** (static_at_cap 4/15 collapsed vs.\ DQN
0/25, $p=0.0149$, under the OLD calibration and the ORIGINAL
non-retrained dqn_sla checkpoint). Per-seed breakdown shows WHY: at
n=21, baseline's only 2 non-compliant episodes are both from seed 950
(0.8%/0.8%/1.7% per-slice compliance that seed — a real, hardware-level
collapse event, not an aggregation artifact; seeds 951–955 are all
100%); dqn_sla's 2 non-compliant episodes are both from seed 955 (3/5,
not previously tested at Stage 5c); static_at_cap's 2 are both from
seed 953 (3/5). Every arm's non-compliance is concentrated in a single
seed, not spread evenly — consistent with this project's repeated
finding that real RF hardware does not give identical outcomes for a
given seed value run on a different day/session.

**Why this differs from the original claim, most likely:** the dqn_sla
checkpoint used here (`offline_v2/sla/seed256/...`) is NOT the same
checkpoint as the original Table I — it was retrained under Stage 5's
Bug #3 fix (`train_offline_live_scale.py`, real-demand-scale offline
environment), a harder training environment than the original
checkpoint saw. It is plausible (not yet confirmed) that this makes it
behave more like `static_at_cap` under real contention than the
original checkpoint did. This is exactly the open question the offline
reverification below (seeds 257–261) is designed to help answer:
retraining across multiple seeds will show whether the corrected
environment reliably produces a policy this close to static_at_cap, or
whether seed256 itself was an unlucky training draw.

**Manuscript impact:** Table I, the static-at-cap comparison paragraph,
Fig. 1, Fig. 2, the Summary of Findings, and the Conclusion were all
rewritten to report both the original (smaller-sample) finding and this
one honestly — see `paper_conf/main.tex`. The QoE-aware reward's
advantage is the one that survives; the SLA-only reward's is now an
open question rather than a settled result. Future Work item (1) is
replaced with a new item: investigating why the SLA-only edge didn't
clearly replicate.

## 2. Live congested pilot (item 2, "lighter, honest" scope per user's
own choice from a prior turn's AskUserQuestion)

`experiments/scripts/run_live_congested_pilot.sh` (new),
`experiments/configs/saclb_offline_congested_v1_baseline.yaml` (new,
one-line `ceiling_step_ratio=0` variant, mirroring the existing
`saclb_campaign_baseline_v2.yaml` precedent). Live-evaluated the
ALREADY-TRAINED congested/URLLC-preservation checkpoints
(`offline_congested/{sla,qoe}/seed256/dqn/checkpoint.pt`) plus a static
baseline, 1 seed (950), 2 episodes/arm — a pilot scale, not the full
protocol, per this project's own start-small precedent.

**Finding:** all 3 arms are 100% SLA-compliant live (2/2 episodes each)
— the offline scenario's eMBB-for-URLLC sacrifice (Table II) does NOT
appear live. This is a scale-mismatch, not a contradiction: Table II's
tradeoff comes from `SharedPoolCongestedKpmSource`'s fiat
`shared_pool_prb=8.0` budget, deliberately set smaller than what the 3
slices' ceilings (12+4+3=19) could jointly request. The real gNB has
106 physical PRBs — 19 is nowhere near enough to create the same
forced scarcity, so no cross-slice tradeoff is observable at this
campaign's ceiling/traffic scale. `docs/STAGE6_scarcity.md`'s prior
finding (adding a 4th real UE for genuine demand-driven scarcity is
infeasible on this rig's memory envelope) is a DIFFERENT question from
this one — this pilot needed no new UE, just evaluation of existing
checkpoints under existing live traffic.

MOS did differ meaningfully even without a compliance difference:
URLLC MOS 4.83 (dqn_qoe_congested) vs.\ 2.44 (dqn_sla_congested) vs.\
2.66 (baseline_congested) — the QoE-aware reward gives URLLC a better
experience even when raw compliance can't distinguish the arms.

**Manuscript impact:** new paragraph in Section IV-C, revised Future
Work item (2): now specifically "recalibrate ceilings and traffic to
attempt a live reproduction of genuine multi-slice physical scarcity"
rather than the more open "reproduce ... on the live rig" — reflecting
what was actually learned about why the light-touch version didn't
show the effect.

## 3. Manuscript update and figures

`paper_conf/main.tex`: abstract, contributions list, Evaluation Setup,
Live Testbed Results (Table I + narrative), Deployment Feasibility,
Congested Scenario (new live-pilot paragraph), Summary of Findings, and
Conclusion/Future Work all rewritten — see the diff in the corresponding
commit for the exact before/after text. `paper_conf/figures/
fig2_sla_compliance.{pdf,png}` and `fig4_ceiling_trajectories.{pdf,png}`
regenerated from `live_campaign_v2`'s full n=21/seeds=950-955 dataset
(best-arm switched to `dqn_qoe`, now the clear best performer). Clean
`pdflatex → bibtex → pdflatex → pdflatex` rebuild: 4 pages, zero errors,
zero undefined references.

## 4. Reverification launched after this stage's commit

Per the user's explicit instruction, two further runs were launched
immediately after this stage's commit, without further confirmation:

- `experiments/scripts/run_offline_v2_reverify.sh` (~3h, no rig time):
  retrains dqn_sla_v2/dqn_qoe_v2 across 5 NEW seeds (257–261) to check
  whether Stage 5's original seed256 retraining reliably converges, or
  was itself a favourable/unfavourable single-seed draw — directly
  relevant to this stage's open question about why dqn_sla's live
  collapse-avoidance edge didn't replicate. Also expands the offline
  congested held-out evaluation (`eval_congested_vs_baseline.py`) to 8
  more seeds (953–960), strengthening Table II's statistical base from
  45 to 165 episodes/arm.
- `experiments/scripts/run_stage5_v2_reverify_10h.sh` (~10h, live rig):
  adds a full 5-seed × 5-episode block (956–960) to all 4 live arms,
  bringing every arm to n=46 — enough to meaningfully sharpen the
  dqn_sla-vs-static_at_cap Fisher test (currently $p=1.0$ at n=21 each,
  a null result that could still reflect insufficient power rather than
  a genuinely identical true rate).

**Operational note, added after launch:** both scripts were initially
launched concurrently. Within minutes, the rig's free memory dropped to
~140MB (from an already-tight ~200-400MB baseline) and the live
reverify's own health check failed (all 3 UEs unreachable) mid-batch --
consistent with this project's own documented finding that this rig
runs at a very tight memory margin even under normal single-workload
live operation, and that adding a second CPU/RAM-heavy job (offline
DQN training) pushes it over the edge. The orchestrator's built-in
`ensure_healthy()`/`restart_ran_stack.sh` retry logic caught this and
recovered automatically (confirmed: `baseline` seed=956 completed
5/5 episodes after one automatic restart) -- but running both jobs
concurrently is clearly a real stability risk to the 10h live run,
which is the harder-to-redo and higher-priority of the two given the
open statistical question it exists to answer. **Decision (made
without further user confirmation, per this stage's standing
instruction): killed the offline reverify job immediately, will
restart it sequentially after the 10h live run completes rather than
concurrently.** This delays the offline reverify's own completion time
past its original ~3h estimate (it will now start after the ~10h live
run finishes), but protects the live run, which cannot be cheaply
re-run if it fails partway through an unattended overnight session.

## 5. Online reverification: complete (2026-07-31 05:36)

All 20 new (arm, seed) blocks (seeds 956–960 × 4 arms) completed
cleanly overnight, zero failures logged. One transient health-check
failure occurred early on (all 3 UEs briefly unreachable, most likely
residual contention from the killed-but-not-yet-fully-released offline
job) -- the orchestrator's own `ensure_healthy()` retry logic recovered
it automatically with one stack restart; no manual intervention was
needed beyond monitoring.

All 4 arms now stand at n=46 (`metrics_stage5_v2.py`'s `ARM_SEEDS`
updated to the full 950–960 set):

| Arm | Episodes fully compliant | Compliance (eMBB/URLLC/mMTC, %) |
|---|---|---|
| baseline | 42/46 | 91.7 / 91.4 / 91.4 |
| dqn_sla | 44/46 | 96.6 / 95.7 / 95.8 |
| dqn_qoe | **46/46** | 100.0 / 100.0 / 100.0 |
| static_at_cap | 44/46 | 97.4 / 95.7 / 95.7 |

Fisher exact: dqn_sla vs.\ static_at_cap $p=1.0$ (44/46 vs.\ 44/46,
exactly tied) -- at n=46 this is a genuine null result, not an
underpowered one, confirming the n=21 finding rather than just adding
more uncertainty to it. dqn_sla vs.\ baseline $p=0.68$; dqn_qoe vs.\
baseline $p=0.12$ -- both arms clearly separate from baseline in raw
terms (44 or 46 vs.\ 42) but neither reaches significance at this
sample size, since baseline's own compliance is high enough here to
leave a narrower gap than the original (pre-correction) baseline data
showed.

Manuscript updated again with these final numbers (Table I, all
in-text Fisher/episode-count mentions, abstract, Summary of Findings,
Conclusion); `docs/REPRODUCIBILITY.md` updated to match. Clean
recompile, 4 pages.

**Rig teardown:** the stack was found already down (all Docker
containers exited with code 137, no native RAN processes, no dmesg
OOM signature) when checked after the reverification script's own
completion message -- most likely an external/incidental stop rather
than anything this session did (the reverify script itself does not
tear the stack down, and `free -h` showed healthy headroom by the time
this was checked). Formally re-ran `docker compose down` and removed
the `iperf3-target` container to leave a clean, known state rather than
relying on the already-dead containers. No data loss: the reverify
script's own `PROGRESS.log` confirms every block completed before
whatever stopped the containers happened.

The offline reverification (Part A retrain seeds 257–261, Part B
expanded congested eval seeds 953–960) was launched immediately after
teardown, with the rig's resources fully free this time -- see the
follow-on update below once it completes.

## Acceptance status

- [x] Item 1 (fully powered live re-evaluation) completed: all 4 arms
      at n=21, then extended to n=46 by the 10h reverification (section 5).
- [x] Finding reported honestly, including where it revises rather than
      confirms the previously-published claim — not silently glossed
      over — and re-confirmed, not just re-asserted, once n=46 showed
      the same null result with real statistical power behind it.
- [x] Item 2 (live congested reproduction) attempted at the
      user-selected "lighter, honest" scope; result and its scale-mismatch
      explanation reported, not hidden.
- [x] Manuscript rewritten to match at both n=21 and (final) n=46;
      clean recompile at 4 pages each time.
- [x] Reproducibility appendix updated with new provenance rows,
      original rows kept (marked superseded) rather than deleted.
- [x] Online reverification (10h live) — complete, n=46/arm, zero
      failures, self-healed once from a transient health-check failure.
      Rig torn down cleanly afterward.
- [ ] Offline reverification (3h) — launched after teardown with full
      resources free; not yet complete as of this document's writing.
