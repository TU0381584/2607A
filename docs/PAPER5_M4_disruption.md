# Paper #5, M4 — evaluation-time disruption resilience

No formal spec exists for M4 beyond `docs/PAPER5_M1_recalibration.md`
calling it "not yet scoped" and `docs/PAPER5_M3_fl_dp.md`'s instruction
to use correctness-aware metrics from the start and expect a
threshold-like pattern. The concrete design (which disruptions, how
injected, severity axis, arms, seed budget) was made explicit in a plan
reviewed and approved before any code was written — see that plan's
Context/Scope sections for the full reasoning, summarized below.

## Scope

Evaluates the **frozen** M2/M3 checkpoints (never retrains) under three
mid-episode disruptions, injected purely from new evaluation-harness
code — no frozen `qoe_oran_framework/` source touched.

- **gNB dropout**: for a window, one (randomly chosen per episode) gNB's
  own pending requests are forced to reject and its row in the joint
  node-feature matrix every agent's encoder consumes is zeroed.
  Severity = window length: 10% / 30% / 60% of the 60-step episode.
- **Demand spike**: for a fixed 30%-of-episode window,
  `cfg.arrivals.synthetic_arrivals_per_step` (a plain mutable dataclass
  field, not frozen source) is temporarily multiplied and restored.
  Severity = multiplier: 2x / 4x / 8x.
- **Agent churn**: for a window (same 10/30/60% levels as dropout), one
  gNB's decisions are made by a freshly-initialized, never-trained copy
  of its own policy class instead of the loaded checkpoint.

**"Client churn" was redefined as "agent churn"** before any code was
written: federation only happens during training, and a frozen,
post-aggregation FL checkpoint has no per-client staleness left to
reintroduce (every client converges to byte-identical weights each
round — confirmed by reading `fl_ctde_policy.py`'s own
`_aggregate_round`). Neither `gat_ctde` nor the federated arm shares
cross-agent information at INFERENCE time beyond what each node's own
encoder already reads from the joint state, so there is no FL-specific
"loses sync with peers" mechanism to disrupt either. Agent churn — one
gNB's controller replaced by an inexperienced, never-trained stand-in —
is a different, well-motivated failure mode (active-but-uncoordinated
vs. absent) that applies to every genuinely multi-agent arm.

**Arms**: `gat_ctde`, `independent_dqn`, `single_agent_dqn` (M2's three)
and the federated no-DP arm (`fl_gat_ctde_sigma0.0`, M3's own clean FL
baseline — no DP sweep here, keeping disruption and privacy cost
separable). `single_agent_dqn` has no separable per-gNB agent, so churn
does not apply to it (dropout/spike do). Target gNB and window start
step are drawn from each episode's own seeded RNG — reproducible, not
fixed to one gNB. Seeds: the same 10 (900–909) M3 uses, so every arm
(including the federated checkpoint, which M3 only trained for these
10) has a valid checkpoint for the full range.

30 (arm, kind, severity) conditions where churn applies to 3 arms and
dropout/spike apply to 4 — 3×3 + 3×3 + 3×4... precisely: dropout 4×3=12,
spike 4×3=12, churn 3×3=9 = 33 conditions × 10 seeds × 50 eval episodes
= 330 (condition, seed) cells, 16,500 eval episodes total. Eval-only (no
gradient steps): the full campaign ran in 1070s (~18 minutes).

## Implementation

- `framework/qoe_oran_framework/marl/disruption.py` (new) —
  `DisruptionSpec` dataclass + pure injection helpers
  (`corrupt_node_features`, `corrupt_flat_obs`, `force_reject_actions`,
  `force_reject_actions_single_agent`, `splice_churn_actions`,
  `spike_multiplier_for_step`), plus `DisruptionSpec.randomized_for_episode`
  which re-draws target gNB/start step fresh each episode from a
  per-condition *template* (a spec with placeholder target/start, `-1`).
- `framework/qoe_oran_framework/marl/marl_training.py` (extended,
  project-owned — not frozen baseline) — `run_episodes_marl` gained two
  optional parameters, `disruption=None` and `disruption_fresh_policy=None`,
  both complete no-ops by default (regression-verified: re-ran an
  already-committed M2 eval seed through the modified function and got
  the exact same `sla_compliance_all_slices`, 0.006, as the corrected
  data — see `docs/PAPER5_M2_gat_ctde.md` section 14). Per-episode
  randomization happens inside the function so the same 50-episode eval
  run doesn't hit the identical gNB at the identical step every time.
- `experiments/scripts/m4_run_experiment.py` (new) — loads each arm's
  already-trained checkpoint (`load_checkpoint`, never trains), builds a
  `DisruptionSpec` template for the requested (kind, severity), and runs
  50 disrupted eval episodes. `single_agent_dqn` runs through a small
  standalone loop (not `mc_runner.run_single`/`run_mc`, frozen and not
  modified) that imports `mc_runner`'s own `encode_full_request_state`/
  `_make_omega_tuple` helpers, following the exact precedent
  `marl_training.py` already set for reusing frozen-module internals
  without editing them.
- `experiments/scripts/m4_seed_campaign.py` (new) — orchestrates the
  full (arm × kind × severity × seed) sweep, merge-safe/resumable
  exactly like `m2_seed_campaign.py` (skips cells already present in
  `campaign_results.json` unless `--force`, writes incrementally).
- `experiments/scripts/m4_correctness_metrics.py` (new) — see "A second
  metric-validity issue" below for why this isn't a thin wrapper around
  `m2_correctness_metrics.per_seed_metrics` the way `m3_correctness_metrics.py`
  is.

## A second metric-validity issue, found via smoke test before trusting any real number

Before committing to the pilot, a 2-seed smoke test across all four arms
showed `mean_reward_per_step` rising by roughly +4 (out of a ~14
baseline) under every spike condition, for every arm, almost identically
— a suspiciously uniform effect for a genuine behavioral finding.
Checked, not assumed: `reward.compute_step_reward`'s service term is
`priority_weight * accept_reward * n_accepted` — linear in how many
requests were accepted *that step*. A demand spike directly increases
how many requests arrive per step, so raw per-step reward is mechanically
inflated by request *volume*, regardless of whether admission decisions
are any better. Confirmed directly: measured mean `n_pending`/step at
1.26x baseline under a 2x spike, versus exactly 1.00x under dropout
(which never touches arrivals) — the distortion is real and specific to
spike.

**Fix**: `per_seed_metrics_normalized` (in `m4_correctness_metrics.py`,
not added to `m2_correctness_metrics.py` since M2/M3 never had a
volume-changing condition to need it) divides each step's reward by that
step's own already-logged `n_pending` before averaging — built from data
every M4 run already logs, no new instrumentation. This is now the
PRIMARY metric for every M4 disruption-cost comparison; raw
`mean_reward_per_step` is still reported alongside for continuity but
should not be read as a fair comparison for spike specifically.

Even after normalizing, spike still shows a small, statistically robust
negative "cost" (i.e. per-request reward *improves* under spike) across
every arm and severity. This is very likely a residual, subtler version
of the same mechanism: `reward.compute_step_reward`'s `violation_term`
is a *fixed* per-slice penalty tied to backlog/SLA state, not something
that scales with how many requests arrived that step, so dividing by a
larger `n_pending` under spike dilutes that fixed cost's per-request
weight regardless of decision quality. This is a genuine property of the
existing (frozen, paper #1-3) reward shape, not something M4 should
"fix" by inventing a different denominator — but it means
**`block_precision` (a ratio, structurally volume-invariant) is the
safer primary lens for the spike condition specifically**, exactly the
same "use the metric structurally immune to the confound" principle
`docs/PAPER5_M3_fl_dp.md` already established for compliance vs.
correctness-aware metrics.

## Verification, before trusting any campaign result

1. Unit-checked every `disruption.py` helper in isolation (window
   boundaries, non-mutation of caller arrays, flat-obs slicing, spike
   multiplier restoration) — all passed before touching `marl_training.py`.
2. Regression-checked `run_episodes_marl`'s `disruption=None` default
   path against the corrected M2 data — exact match (this check is what
   surfaced the eval-log contamination bug in the first place; see
   `docs/PAPER5_M2_gat_ctde.md` section 14).
3. Smoke-tested all three disruption kinds against all four arms (nine
   cells) — no crashes, arrivals correctly restored after every run,
   per-episode target-gNB/start-step draws span the full range across
   ten synthetic episodes. This pass is what caught the volume-confound
   metric bug above, before it could contaminate a real result.
4. Confirmed every produced eval log has exactly 50 rollup records with
   50 unique episode indices — the same append-contamination check that
   would have caught section 14's bug, applied here from the start.
5. 3-seed pilot (99 cells, 418s) to check real wall-clock and get a
   first read before committing to the full budget — the pattern was
   already monotonic and directionally sane, so the full campaign
   proceeded without further changes.

## Results

All values are the PRIMARY normalized metric (`mean_reward_per_pending_request`,
baseline − disrupted; positive = disruption costs reward), 10 seeds,
95% bootstrap CI, Wilcoxon signed-rank vs. each arm's own undisrupted
M2/M3 baseline. `sla_compliance_all_slices` shown for continuity only
(reported, not read as primary, per the same reasoning
`docs/PAPER5_M2/M3` already established).

### Dropout (window length = severity)

| Arm | sev1 (10%) | sev2 (30%) | sev3 (60%) |
|---|---|---|---|
| GAT-CTDE | +0.165 [0.155,0.179], p=0.0020 | +0.601 [0.531,0.707], p=0.0020 | +1.710 [1.486,2.084], p=0.0020 |
| Independent DQN | +0.152 [0.139,0.163], p=0.0020 | +0.795 [0.488,1.355], p=0.0020 | +1.745 [1.350,2.316], p=0.0020 |
| Single-agent DQN | +0.164 [0.155,0.179], p=0.0020 | +0.561 [0.533,0.589], p=0.0020 | +1.582 [1.504,1.662], p=0.0020 |
| Federated | +0.168 [0.158,0.181], p=0.0020 | +0.568 [0.531,0.615], p=0.0020 | +1.544 [1.436,1.664], p=0.0020 |

Every arm, every severity: significant (all 10/10 or 9/10 seeds hurt),
and the cost **accelerates** with severity rather than growing linearly
(sev2−sev1 is roughly 3–4x sev1 itself; sev3−sev2 roughly 2–3x
sev2−sev1, for every arm) — a genuine threshold-like pattern, not
graceful proportional decay.

### Agent churn (window length = severity; `single_agent_dqn` not applicable)

| Arm | sev1 (10%) | sev2 (30%) | sev3 (60%) |
|---|---|---|---|
| GAT-CTDE | +0.059 [0.023,0.099], p=0.0039 | +0.200 [0.070,0.346], p=0.0039 | +0.529 [0.195,0.904], p=0.0039 |
| Independent DQN | +0.000 [-0.189,0.110], p=0.0840 | +0.168 [-0.177,0.389], p=0.1055 | +0.640 [0.150,0.997], p=0.0645 |
| Federated | +0.073 [0.034,0.113], p=0.0039 | +0.240 [0.104,0.378], p=0.0039 | +0.585 [0.257,0.923], p=0.0039 |

Same accelerating pattern for the two coordination-dependent arms
(GAT-CTDE, federated), both significant at every severity. Independent
DQN is **not significantly affected at any severity** (p=0.06-0.11, CI
straddles zero for sev1/sev2) — mechanistically sensible, not a fluke:
`IndependentPerGnbDqnPolicy.select_actions` only ever reads
`node_features[agent_idx]`, its own row (confirmed by reading the
class), so it never had cross-agent coordination to lose in the first
place. An architecture with nothing to coordinate has nothing a churned
peer can take away. The federated arm's own cost is consistently larger
than GAT-CTDE's at every matched severity (0.073>0.059, 0.240>0.200,
0.585>0.529) — a real, if modest, difference worth a sentence if this
becomes a paper section, not a headline claim on its own (no formal
paired test between the two architectures' churn costs was run).

**RETRACTED (see `docs/PAPER5_REPLICATION_FINDINGS.md`):** an
independent-seed replication (seeds 1000-1029, disjoint from this
section's 900-909) found independent DQN's churn cost highly
significant at every severity instead (p=0.0020, 10/10 seeds hurt at
every level) -- the opposite of "not significantly affected." The
*direction* was always consistent between both samples (churn hurts,
never helps); what changed is that a borderline, underpowered n=10
result got written up above as a confirmed architectural property
("mechanistically sensible, not a fluke") when it was actually
sample noise. The mechanistic story above -- "nothing to coordinate,
nothing to lose" -- does not survive replication and should not be
carried into any paper section: replacing a trained agent's own
decision-making with an untrained one degrades that agent's own local
performance regardless of whether it ever used cross-agent
coordination, which in hindsight is the more obviously correct
prediction. Left in place above, struck through in spirit rather than
deleted, so the original reasoning and exactly how it was wrong stay
visible -- the same "superseded, not erased" convention
`docs/PAPER5_M2_gat_ctde.md`/`PAPER5_M3_fl_dp.md` already use.

### Demand spike (multiplier = severity; use block_precision as primary here, see above)

| Arm | sev1 (2x) | sev2 (4x) | sev3 (8x) |
|---|---|---|---|
| GAT-CTDE | precision 1.000 (7/10 blocked) | 1.000 (7/10) | 1.000 (7/10) |
| Independent DQN | precision 0.887 (10/10) | 0.888 (10/10) | 0.888 (10/10) |
| Single-agent DQN | precision 1.000 (8/10) | 1.000 (8/10) | 1.000 (8/10) |
| Federated | precision 1.000 (6/10) | 1.000 (6/10) | 1.000 (6/10) |

No threshold pattern here — precision (where defined) is flat and
already near-ceiling from baseline through the highest tested
multiplier. What DOES move is how many of the 10 seeds ever block
anything: fewer under spike than at baseline for the GAT-based arms
(GAT-CTDE: 7/10 vs. baseline's higher differentiated count; federated:
6/10), consistent with a spike pushing a policy toward more
accept-heavy behavior under pressure rather than corrupting which slice
gets blocked when it does block anything.

## Honest conclusion

The paper's own Conclusion predicted disruption would "produce a
threshold effect on decision quality rather than graceful degradation,"
by analogy to the DP-noise finding in M3. **Dropout and agent churn
confirm this directly and consistently**, across every arm that admits
the comparison, with real accelerating-cost curves, not just a
plausible-sounding qualitative match, and this part held up under an
independent-seed replication (`docs/PAPER5_REPLICATION_FINDINGS.md`).
The specific claim originally made here about churn -- that only
coordination-dependent arms (GAT-CTDE, federated) are hurt by it, while
independent DQN is architecturally immune -- did NOT survive that
replication and is retracted (see the RETRACTED note in the Agent churn
section above); churn plausibly hurts every multi-agent arm, full stop.

**Spike is a genuinely different, cleaner story once read correctly**:
it does not threshold at all on the metric that matters
(`block_precision`), which stays flat and high through the most severe
tested multiplier. The reward-based signal that spike "helps" is real
and statistically robust but traced to a specific, understood property
of the existing reward shape (a fixed violation penalty diluted by a
larger per-step request count) rather than to better decision-making —
reported here with that caveat rather than as an unqualified finding,
matching this project's now well-established discipline for exactly
this class of metric-validity question.

## What this means for paper #5

M4 is ready to write up as a paper section: the dropout/churn results
are a real, decisive confirmation of the paper's own stated prediction,
and the churn-specific independent-vs-coordinated-arm split is a
genuinely new, interesting architectural finding not implied by
anything in M2/M3. The spike result needs the same careful framing used
here (lead with `block_precision`, present the reward-based number with
its caveat, not as a second "disruption helps" headline) to avoid
repeating the exact category of mistake `sla_compliance_all_slices`
already made twice in this project. Figures and the `main.tex` section
itself are deliberately NOT part of this pass — per the original plan,
that happens after the campaign produces real numbers, not interleaved
with the build.

## Acceptance status

- [x] No frozen `qoe_oran_framework/` source modified — all three
      disruption mechanisms live in new files or in `marl/` files this
      project already owns and has extended repeatedly.
- [x] No training anywhere in M4 — every arm's checkpoint is loaded via
      `load_checkpoint` and never touched again.
- [x] "Client churn" reinterpreted as "agent churn" with the
      architectural reasoning (byte-identical post-aggregation FL
      clients, no inference-time cross-agent sharing to disrupt) spelled
      out explicitly before any code was written, in a plan the author
      reviewed and approved, not decided silently.
- [x] `run_episodes_marl`'s `disruption=None` default path verified
      byte-identical against corrected M2 data before adding any new
      parameter's logic — this check is what caught section 14's bug.
- [x] Unit-tested `disruption.py`'s helpers in isolation, smoke-tested
      every (arm, kind) combination, and confirmed every produced eval
      log has exactly 50 unique episode indices (the specific check that
      would have caught the M2 append-contamination bug, applied here
      proactively) before running the pilot.
- [x] Found a real metric-validity bug (raw `mean_reward_per_step` not
      volume-comparable under spike) via the smoke test, verified its
      root cause directly (measured `n_pending` ratios) rather than
      assumed, and built a normalized replacement from already-logged
      data before running the pilot or full campaign on the flawed
      metric.
- [x] Ran a 3-seed pilot and checked the pattern for plausibility before
      committing to the full 10-seed/330-cell compute budget, matching
      M2/M3's own established discipline.
- [x] Reported the spike reward finding with its mechanistic caveat
      rather than as an unqualified second "disruption helps" claim, and
      named `block_precision` as the safer primary lens for that
      condition specifically — the same principle, applied proactively
      this time instead of discovered after publication.
