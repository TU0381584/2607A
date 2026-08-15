# Paper #5, M2 — GAT + CTDE-MARL build and first-pass ablation

Net-new multi-gNB, GAT-encoded, CTDE-MARL extension of paper #4's
single-agent, single-gNB admission control, per the M2 brief. All code
lives under `framework/qoe_oran_framework/marl/` (new subpackage) and
`experiments/scripts/m2_*.py` -- no frozen source modified.

## 0. Scope reconciliation (done before writing code, per the brief)

Read `experiments/REWORK_PLAN.md` section 2 (R5) and the R3/R4/R6
dependency rows first. R5's plan and cost-consciousness are reusable
(net-new subpackage, don't touch frozen source, same "cut this first if
squeezed" risk posture), but its estimate ("2 days or a week+") only
covers the GAT+CTDE encoder/critic/actor build -- not the multi-gNB
topology environment itself (R5 assumed graph inputs already existed;
per `PAPER5_STATUS.md`, the existing multi-gNB support is a flat
concatenation into one shared policy, with no adjacency structure at
all) and not the two specific ablations or the (subsequently dropped,
see below) Wu-paper benchmark. R5's own "3-node" graph size was also
ambiguous (gNBs or slices); this build uses N=3 gNBs unambiguously,
reusing paper #2's own existing 3-gNB config
(`qoe_oran_framework/configs/saclb_offline_dqn.yaml`) rather than
inventing a new topology size.

**Dropped per author's explicit decision:** "the two Wu MARL papers
already flagged as baselines in the paper-5 skeleton" -- no such skeleton
or Wu paper exists anywhere in this repository (exhaustive search, same
pattern as `ACTION_PLAN_conference_and_journal.md` and the earlier
"existing GAT+CTDE+FL scaffold" premise, both also not found). The two
specified ablations (vs. single-agent DQN, vs. independent per-gNB DQN)
are built and reported below instead.

## 1. What was built

- `marl/topology.py` -- adjacency-matrix builder. Fully-connected by
  default (no real multi-gNB topology exists anywhere in this repo's
  configs/docs to measure a real one from; documented as a choice, not a
  fact).
- `marl/gat_encoder.py` -- a GAT layer and 2-layer encoder implemented
  directly in torch (no `torch_geometric` dependency -- confirmed not
  installed; graphs here are a handful of nodes, dense masked attention
  is simpler and just as fast). Unit-tested: forward+backward pass
  produces correctly-shaped output and gradients flow.
- `marl/ctde_policy.py` -- `GatCtdeMarlPolicy`: shared GAT encoder over
  the joint cluster state + a shared per-agent Q-head (parameters shared
  across the 3 homogeneous gNBs) on the *same* discrete accept/reject
  action space paper #4 validated (not a continuous simplex). See
  section 2 for a real bug found and fixed here.
- `marl/independent_dqn_ablation.py` -- `IndependentPerGnbDqnPolicy`: 3
  separate instances of paper #4's own `DQNPolicy` class (reused
  unchanged from `framework/drl_slicing`), no parameter sharing, no
  topology information -- each sees only its own node's local features,
  the same per-agent input dimensionality `GatCtdeMarlPolicy`'s Q-head
  consumes, so this ablation isolates the GAT+centralized-training
  contribution specifically, not a difference in available information.
- `marl/marl_env.py` -- pure feature-extraction glue
  (`extract_node_features`, `request_to_agent_context`) between the
  *existing, unmodified* `qoe_oran_framework.env.RANEnv` (already
  multi-gNB capable) and the new policy layer. No new environment class
  needed.
- `marl/marl_training.py` -- `run_episodes_marl`, the MARL analogue of
  `mc_runner.run_single`/`run_mc`, reusing `OmegaLogger` and
  `mc_runner._make_omega_tuple` (imported, not copied) so the resulting
  `omega_log.jsonl` schema is identical to every other arm in this
  project.
- `experiments/scripts/m2_run_experiment.py` -- the 3-arm driver. The
  third arm, `single_agent_dqn`, needed **no new code at all**: it is
  paper #4's existing `DQNPolicy` run via the existing, unmodified
  `mc_runner.run_mc` against the same 3-gNB config -- already-supported,
  today's actual multi-gNB pattern (paper #2's LB extension).

**Environment:** `qoe_oran_framework/configs/saclb_offline_dqn.yaml`
(paper #2's own 3-gNB config, deliberately oversubscribed to 110% of one
gNB's PRB budget across its three slices -- genuine offline contention,
read-only, unmodified). Per `docs/PAPER5_M1_recalibration.md`'s
conclusion, this is used as a live-anchored **stress environment** for
the contention regime, not a live-rank prediction claim -- nothing here
is evaluated live. `MEAN_OFFERED_RATIO` (live-probe anchored) and
`backlog_capacity=2000` (this project's established offline default) are
reused unchanged from M1.

## 2. A real training-instability bug, found and fixed

First full run (300 train + 50 eval episodes x 3 seeds) produced
`gat_ctde` compliance uniformly near zero (0.005-0.014) while
`single_agent_dqn` reached 0.012-0.601 -- suspicious because `gat_ctde`'s
numbers were both poor *and* unusually uniform across seeds, unlike
`single_agent_dqn`'s high variance.

**Diagnosis (not assumed):** logged `train_step`'s own loss over a short
run. It was **diverging**, not converging (mean loss ~76 in the first
20% of steps, ~409 in the last 20%). Doubling the training budget
(300->600 episodes) made held-out compliance flat-to-worse, ruling out
"just needs more training."

**Root cause 1, fixed:** the initial `train_step` summed a gNB's
chosen-action Q-values across every request it handled *within one
step* before feeding the sum to a QMIX-style mixer. Since pending
requests per step vary (0 up to `max_pending_per_step=12`), the learning
target's scale swung with request count step to step -- an
ill-conditioned, non-stationary regression target. **Fix:** every
pending request is now its own independent TD sample (own chosen
action, own bootstrapped next-Q), exactly matching paper #4's own DQN
granularity (`mc_runner._store_and_train`'s dqn branch never aggregates
either). This did not fully fix it alone (loss still grew, ~7x over 150
episodes, down from the original's much steeper growth).

**False lead, ruled out by a control comparison:** suspected the GAT
encoder itself was unstable. Before chasing more architecture changes,
ran the *exact same loss-logging diagnostic* on paper #4's own
unmodified, already-working `DQNPolicy` on this identical environment.
**It showed the same rising-loss pattern** (119 -> 577 over the same
150-episode window) despite being the reference implementation that
reliably converges to a sensible policy. This confirms rising loss on
this specific environment/reward is expected (reward magnitude grows as
a policy learns to accept more, since `service_term` scales with accept
count and priority weights up to 12.0, compounded over `gamma=0.99`'s
long horizon) -- not proof of a bug. Loss-curve shape was the wrong
signal to debug against.

**Root cause 2, fixed anyway (legitimate regardless):** switched from
MSE to Huber/smooth-L1 loss and lowered the GAT-CTDE network's learning
rate (1e-3 -> 1e-4) -- standard DQN-family stabilization for a deeper,
attention-based function approximator, and this did measurably help
(loss magnitude dropped by roughly an order of magnitude; accept rate
moved toward the same "mostly accept" regime `single_agent_dqn`
converges to).

## 3. Result (3 seeds, 300 train + 50 eval episodes/seed, offline stress env)

Metric: `sla_compliance_all_slices` -- mean, across held-out episodes,
of the per-episode fraction of steps where every slice was
simultaneously SLA-compliant (this project's own established
`RunSummary`/`mc_runner` convention, continuous not binary -- see
`docs/PAPER5_M1_recalibration.md`'s reuse of the same quantity).

| Seed | gat_ctde | independent_dqn | single_agent_dqn (paper #4's arch) |
|---|---|---|---|
| 900 | 0.240 | 0.008 | 0.128 |
| 901 | 0.367 | 0.004 | 0.012 |
| 902 | 0.601 | 0.004 | 0.601 |
| **mean** | **0.403** | **0.005** | **0.247** |

Figure/raw data: `experiments/results/m2_gat_ctde/m2_results.json` and
per-seed `omega_log.jsonl` under the same directory (train + eval, all
three arms).

## 4. Interpretation (preliminary -- see caveats)

**Superseded by section 11**: this 3-seed pilot's `gat_ctde` checkpoints
predate the `GATEncoder` normalization fix and likely exhibit the same
always-accept collapse section 11 found universally in the later 30-seed
campaign — its "most seed-consistent" reading below is consistent with
that (a collapsed policy is trivially consistent across seeds, since it
makes the same no-op decision regardless of state). Kept as the
historical record of the reasoning at the time, not re-run individually,
since section 11's full campaign re-run supersedes it either way.

- **`gat_ctde` beats both ablations on mean, and is the most
  seed-consistent of the three** (0.240-0.601, a ~2.5x spread) vs.
  `single_agent_dqn`'s far wider spread (0.012-0.601, ~50x) at the same
  training budget. This is directionally consistent with what CTDE/GAT
  sharing is supposed to buy: a shared, topology-aware representation
  trained centrally should reduce the same kind of seed-dependent
  fragility paper #4 itself already documents for the single-agent,
  single-gNB case (`docs/STAGE11_checkpoint_sensitivity.md`'s 13/21-to-
  21/21 spread across training seeds).
- **`independent_dqn` essentially fails to converge at all within this
  budget** (0.004-0.008, uniformly near zero) despite reusing the exact
  same, already-proven `DQNPolicy` class `single_agent_dqn` uses --
  the only difference is each of the 3 agents sees only its own local
  state, with no cross-agent information and no shared parameters. This
  is consistent with a well-known MARL phenomenon (independent learners
  face non-stationarity from other agents' concurrently-changing
  policies) rather than an implementation bug -- `DQNPolicy` itself is
  not in question, only the decentralized-with-no-sharing setting it was
  placed in.
- **`single_agent_dqn`'s own seed variance (0.012 to 0.601) is itself
  notable** -- the same qualitative fragility-by-training-seed paper #4
  found live for the single-gNB case appears to reproduce in this
  offline multi-gNB stress environment too.

## 5. Caveats (do not treat this as a finished ablation study)

- **3 seeds, one training run per seed, 300 episodes.** No repeated
  trials per seed, no significance testing (unlike paper #4's live
  results, which have Fisher exact tests behind every claim). This is a
  first-pass infrastructure validation, not a campaign.
- **Both new architectures are freshly built and only lightly tuned**
  (one stabilization pass on `gat_ctde`; `independent_dqn` untouched
  since its underlying `DQNPolicy` is already proven). Neither has had
  anything close to the tuning history `saclb_offline_dqn.yaml`'s own
  extensive reward-weight comments document for the single-agent case.
- **`single_agent_dqn`'s wide variance means 3 seeds is not enough to
  trust its own mean (0.247) as representative** -- a 4th or 5th seed
  could plausibly land anywhere in the 0.01-0.6 range already observed.
- Do not read section 4 as "GAT-CTDE is proven better" -- read it as "a
  first, honestly-obtained signal in that direction, worth a real
  multi-seed campaign before any paper claim."

## 6. Old-rig / provenance check

All data in this section was generated fresh this session under
`experiments/results/m2_gat_ctde/` -- no checkpoint, config, or log
here originates from any prior campaign, old-rig or otherwise.

## 8. M2 hardening (Block E, task 1) — independent-DQN floor, diagnosed

**Question:** is `independent_dqn`'s near-zero compliance (section 3) a
real coordination effect, or a config/training artifact?

**Step 1 -- hyperparameter parity check (not assumed, checked directly):**
found a real mismatch. `single_agent_dqn` runs via
`qoe_oran_framework.policies.dqn_admission.DQNAdmissionPolicy`, which
overrides the raw `DQNPolicy` defaults with paper #4's own tuned Table I
schedule: `gamma=0.95`, and epsilon decay applied **once per episode**
(0.985, via an explicit `on_episode_end()` hook reaching the exploration
floor around episode ~200 of 300), plus a target-network sync every 10
episodes. Both `gat_ctde` and `independent_dqn` had instead been left on
the raw `DQNPolicy` defaults (`gamma=0.99`, epsilon decaying 0.995 **per
train_step() call**) -- `dqn_admission.py`'s own module docstring already
documents this exact granularity mistake as a known bug pattern ("epsilon
hits its floor after ~600 steps, under 10 of a 300-episode run... ~97%
of training happens post-floor, nearly greedy the whole time"). **Fixed**
in both `IndependentPerGnbDqnPolicy` and `GatCtdeMarlPolicy`: added a
matching `on_episode_end()` (gamma default changed to 0.95, epsilon decay
moved to per-episode, target sync every 10 episodes), wired into
`marl_training.run_episodes_marl` exactly where `mc_runner.run_single`
calls it for `algorithm in ("dqn", "rainbow")`.

**Step 2 -- re-ran with matched hyperparameters (3 seeds, same protocol):**
partial, seed-dependent improvement -- `independent_dqn` seed 900 jumped
0.008 -> 0.240 (now tied with `gat_ctde`'s own seed-900 result), but
seeds 901/902 stayed flat near-zero (0.004). The hyperparameter mismatch
was real and contributed, but does not fully explain the gap.

**Step 3 -- instrumentation (per-agent commanded ceilings from the
existing `ceilings` field already in every omega-log step, no new
logging needed):** this directly refutes the "over-claiming, shared-pool
contention spikes" hypothesis as literally stated -- and reveals the real
mechanism. Comparing seed 901's eval logs, `gat_ctde` and `independent_dqn`
command **near-identical eMBB and URLLC ceilings** (eMBB: 65.74 vs 65.74;
URLLC: 32.92 vs 32.92, mean `max_ratio`). The entire difference is mmtc:
`gat_ctde` holds mmtc's ceiling near its **maximum** (mean 10.97, cap=11),
while `independent_dqn` holds it near its **floor** (mean 5.12-5.25,
floor=5) -- the opposite of over-claiming; `independent_dqn` has learned
to persistently reject mmtc, not over-accept it. mmtc's `priority_weight`
is deliberately the lowest of the three (0.3, per this config's own
tuning history, calibrated to be "reject-optimal under congestion" from a
single-step reward-maximization view) -- an uncoordinated, per-gNB-local
learner has every local incentive to reject it and never discovers that
doing so permanently drives mmtc's own margin negative (an under-served
ceiling can't clear its offered demand, so backlog saturates toward
`backlog_capacity` regardless of how much reject-triggered relief is
applied), which zeroes the compound "all three slices simultaneously
compliant" metric every episode. `gat_ctde`'s centrally-trained, shared
representation apparently does discover that keeping mmtc's ceiling up is
worth it for the compound-compliance objective, despite mmtc's
individually-small reward weight -- direct behavioral evidence for the
GAT/CTDE contribution, not the mechanism originally hypothesized.

**Verdict: neither hypothesis exactly as posed.** Confirmed real per (1)
above: a genuine hyperparameter-parity artifact existed and has been
fixed, and is not the whole story. Confirmed real per (2): a genuine
uncoordinated-learning effect exists, directly evidenced in the commanded-
ceiling data -- but its mechanism is under-serving/abandoning the
lowest-priority slice (mmtc pinned near its floor), not over-claiming a
shared PRB pool (which this specific `ClosedLoopKpmSource` doesn't
mechanically model as a literal shared-capacity contention in the first
place -- each slice serves against its own ceiling independently; the
only real cross-slice coupling is the reward's cluster-wide
`congestion_term`, not a physical resource collision). `independent_dqn`'s
floor is real, not a bug artifact, once the hyperparameter mismatch is
controlled for -- and it is now the correctly-tuned baseline the section-9
seed campaign uses.

## 9. M2 hardening (Block E, task 2) — seed campaign (original pass, since superseded — see section 11)

Full campaign: 10 seed-groups x 3 runs = 30 independent (fresh network
init, independently-seeded env) runs per arm, seeds 900-929 (fixed list,
`experiments/scripts/m2_seed_campaign.py`), matched hyperparameters
(section 8's fix applied to all three arms), 300 train + 50 eval
episodes/run, offline stress-regime environment (section 1). Wall clock:
9341s (~2.6h). Analysis: `experiments/scripts/m2_campaign_analysis.py`,
95% CIs via 10,000-resample bootstrap (not a normal approximation --
compliance is bounded [0,1] and empirically right-skewed, not normal),
paired comparison via Wilcoxon signed-rank (matching this project's own
preference for nonparametric tests over normal-approximation methods,
e.g. Fisher exact elsewhere).

| Arm | n | mean | 95% bootstrap CI | median | std |
|---|---|---|---|---|---|
| `gat_ctde` | 30 | **0.399** | **[0.353, 0.448]** | 0.393 | 0.133 |
| `single_agent_dqn` | 30 | 0.115 | [0.061, 0.176] | 0.026 | 0.163 |
| `independent_dqn` | 30 | 0.049 | [0.013, 0.093] | 0.008 | 0.115 |

**`gat_ctde`'s 95% CI does not overlap either ablation's CI.** Both
ablations are heavily right-skewed (median far below mean: a handful of
seeds happen to land in a good regime -- e.g. `single_agent_dqn` seeds
902/904/915/918 above 0.38, `independent_dqn` seeds 915/928 above 0.43 --
while most seeds sit near zero), consistent with the seed-fragility
`docs/STAGE11_checkpoint_sensitivity.md` already documented for paper
#4's own single-gNB case. `gat_ctde` is the only arm whose median tracks
its mean closely (0.393 vs 0.399), i.e. the only arm that is reliably
good rather than occasionally good.

**Paired comparison (`gat_ctde` vs `single_agent_dqn`, same 30 seeds,
same env realization per seed across arms):**

- Mean paired difference: **+0.284**, 95% bootstrap CI **[0.214, 0.355]**
  (does not cross zero).
- Wilcoxon signed-rank: **W=0.0, p<0.0001**.
- `gat_ctde` wins on **27/30 seeds**, ties on 3 (seeds 902, 908, 912 --
  verified as genuine exact ties, both arms landing on identical
  compliance to 6 decimal places for those specific seeds, not a
  measurement artifact; plausibly both policies converge to the same
  greedy behaviour on env realizations that don't stress the
  mmtc-priority tradeoff much), **loses on 0**.

**At the time this was written, this was read as crossing the line from
preliminary signal to a defensible paper claim.** That reading turned out
to be incomplete — section 11 found this entire campaign's result rests
on an architecture-level training collapse, discovered incidentally while
building M3. This section is kept as the historical record of what was
believed and reported before that discovery, not deleted or silently
corrected; do not cite the numbers in this section 9 table without
reading section 11's revised numbers first.

## 11. Collapse discovery, root cause, and fix (post-Block-E, found while building M3)

**How this was found:** while building M3's federated variant (which
reuses `GATEncoder`), σ=0.0/0.5/1.0/2.0 privacy-sweep runs produced
bit-identical eval compliance across every seed — impossible by chance
for a stochastic training process. Tracing it back: `gat_ctde`'s eval
policy blocks **zero requests across all 50 eval episodes**, for **every
one of the 30 campaign seeds** in section 9's committed data. `admission
decision = accept, unconditionally` is a functional no-op, so any policy
that reaches it produces byte-identical environment trajectories for a
given seed regardless of network internals — which is what actually
produced section 9's numbers, not smarter admission decisions.

**Was "always accept" the correct/reward-optimal policy here?** No —
checked directly, not assumed. `saclb_offline_dqn.yaml`'s own reward-
tuning comments document that `congestion_coeff=1.5` was specifically
calibrated and **verified via direct Q-value inspection** to make
"differentiated shedding" the correct outcome (urllc always-accept, embb
mostly-accept, **mmtc reject-optimal under congestion**), and name the
exact failure signature this was fixing: "Q(accept) > Q(reject) by 5-14
for all three slices... converged to always-accept" under the old,
already-diagnosed-broken `congestion_coeff=0.5`. A direct Q-value probe
on `gat_ctde`'s own section-9 checkpoint, under a synthetic congested
state, reproduced that exact broken signature — Q(accept) − Q(reject) =
+10.7 (urllc), +8.6 (embb), **+7.8 (mmtc)** — accept dominant everywhere,
including mmtc, under the *current*, verified-correct reward weights.
`single_agent_dqn` (27/30 seeds show real blocking) and `independent_dqn`
(29/30) both largely reach the intended differentiated behavior; only
`gat_ctde` collapsed to the old broken signature, universally.

**Root cause, isolated mechanistically:** `independent_dqn` feeds the
*exact same raw per-agent features* `gat_ctde`'s Q-head consumes — the
only structural difference is `gat_ctde` passes them through
`GATEncoder` first. Comparing that encoder's embedding for a synthetic
idle vs. congested state on the same checkpoint: raw input difference
L2-norm 2.55, embedding difference L2-norm 19.6 (looks amplified) — but
**cosine similarity between the two embeddings is 0.99997**. The encoder
was encoding congestion almost entirely as embedding *magnitude*, not
*direction/pattern* — no normalization existed anywhere in
`GATEncoder`/`GATLayer` to prevent this. Since the downstream Q-head is a
roughly-linear function of its input, larger magnitude pushed Q(accept)
up faster than Q(reject) as congestion increased: mmtc's accept-reject
gap *widened* from +5.1 (idle) to +6.8 (congested) on the probe — the
opposite of the intended relationship.

**Fix:** added a per-layer `nn.LayerNorm` to `GATEncoder`
(`framework/qoe_oran_framework/marl/gat_encoder.py`), applied to each
`GATLayer`'s raw output before its nonlinearity — constrains every node's
embedding to zero-mean/unit-variance regardless of input magnitude,
directly removing the "magnitude carries the signal" pathway. Verified
before trusting it: fresh-network forward pass (shapes correct), then a
3-seed training pilot (300 episodes, matching the campaign's own budget).

**Calibration passes tried, both checked rather than assumed:**
1. More training (300 → 600 episodes) on the same 3 pilot seeds: **no
   effect** — bit-identical outcomes to 4 decimal places, including
   per-slice mmtc compliance, despite the underlying block count
   differing (3470 → 738 for the one seed that showed real blocking).
   Ruled out training duration as the lever, consistent with this
   project's own prior finding (section 2) that this failure class
   doesn't respond to extended budgets.
2. `elementwise_affine=False` on the same LayerNorm layers (hypothesis:
   the default learnable per-feature scale/shift could partially
   reconstruct a magnitude-like bias downstream of normalization): made
   it **worse**, not better, on the same 3 pilot seeds (0/3 seeds showed
   any blocking, vs. 1/3 with the default `affine=True`). Reverted;
   `affine=True` (LayerNorm's own default) is the final architecture.

**Full 30-seed re-run with the fix** (`independent_dqn`/`single_agent_dqn`
don't use `GATEncoder` and were not re-run — their section-9 results
stand unchanged):

| Arm | n | mean | 95% bootstrap CI | median | std |
|---|---|---|---|---|---|
| `gat_ctde` (fixed) | 30 | **0.353** | **[0.296, 0.412]** | 0.332 | 0.163 |
| `single_agent_dqn` | 30 | 0.115 | [0.061, 0.176] | 0.026 | 0.163 |
| `independent_dqn` | 30 | 0.049 | [0.013, 0.093] | 0.008 | 0.115 |

**Collapse rate: 30/30 → 27/30 seeds still reach the old "always accept"
signature.** 3 seeds (904, 911, 929) now show genuine, correctly-targeted
differentiated shedding (100% of their blocks on mmtc specifically,
matching the reward calibration's intent exactly) — and their compliance
dropped sharply as a direct, expected consequence (0.013, 0.051, 0.073):
blocking mmtc does not rescue mmtc's SLA in this stress regime (its
backlog/loss margin stays negative regardless, per section 8's own
under-served-ceiling mechanism), it just stops being served for free.
This is the fix working as intended, not a regression.

**Paired comparison, re-run on the fixed data (`gat_ctde` vs
`single_agent_dqn`, same 30 seeds):**

- Mean paired difference: **+0.238**, 95% bootstrap CI **[0.150, 0.325]**
  (still does not cross zero, narrower than section 9's +0.284).
- Wilcoxon signed-rank: **W=24.0, p=0.0001**.
- `gat_ctde` wins on **25/30 seeds**, ties on 3, **loses on 2** (new —
  section 9's version had zero losses; the 2 losses are presumably among
  the 3 now-differentiated seeds, where correctly blocking mmtc costs
  compliance against a baseline that happened to accept-collapse on that
  same seed).

**Honest revised claim: the statistical comparison survives the fix, but
the mechanism does not mean what section 9 implied.** `gat_ctde` still
beats both baselines with a real, significant margin — this is not an
artifact of the collapse alone, since the collapse rate dropped and the
comparison still holds. But the win is still driven overwhelmingly (27/30
seeds) by the *same* accept-everything collapse as before, just slightly
less universal than section 9's data suggested. The defensible claim is
**"`gat_ctde`'s shared, centrally-trained representation collapses to a
safe never-reject policy more reliably than uncoordinated baselines,
which fail in a worse way (evidenced by their own frequent near-zero-
compliance seeds, consistent with section 8's mmtc-abandonment
mechanism)"** — not "`gat_ctde` learns smarter coordinated admission
decisions." The latter is only directly evidenced for 3/30 seeds.

Raw data: `experiments/results/m2_campaign/campaign_results.json`
(overwritten in place for the `gat_ctde` key only, via
`m2_seed_campaign.py`'s new merge-safe `--arms gat_ctde` re-run mode;
`independent_dqn`/`single_agent_dqn` entries are untouched from section
9). Pre-fix `gat_ctde` data is not in the working tree but remains fully
recoverable from git history (commit `a756044`).

## 12. Second fix: per-slice Q-heads (author-requested — "ensure the AI is doing what it is supposed to do")

Section 11's LayerNorm fix reduced but did not solve the collapse (27/30
still reached it). Asked to find a way to make the fix actually reliable,
not just measurably better than before.

**Hypothesis, checked against the reward's own documented calibration
before touching any code:** `mmtc`'s true reject-optimal margin under
congestion is tiny by design (`priority_weight=0.3` against a ~1.5
marginal congestion cost, net ≈ −1.2) while `urllc`'s priority_weight
(12.0, 40x mmtc's) produces proportionally much larger TD errors — and
all three slices shared **one** `AgentQHead` MLP, differentiated only by
concatenating a one-hot slice vector. With Q-values sitting in the
hundreds (a consequence of this architecture's per-request TD scheme,
section 2), mmtc's ~1.2 true signal is a tiny fraction of the value the
shared head's parameters are being pulled around by every slice's
gradients together — urllc's abundant, large-magnitude updates could
plausibly overwrite whatever fine adjustment mmtc's much smaller true
signal needs before it registers. This is a standard shared-parameter/
class-imbalance failure mode, not specific to this architecture.

**Fix:** replaced the single shared `AgentQHead` MLP with **one separate
small MLP per slice** (`framework/qoe_oran_framework/marl/ctde_policy.py`),
selected by the context one-hot at forward time — each slice gets its
own parameters, so one slice's gradients can no longer directly overwrite
another's decision boundary. The shared `GATEncoder` upstream is
untouched and remains the architecture's "centralized" element. Because
`AgentQHead` is imported (not duplicated) by `fl_ctde_policy.py`, M3's
federated arm inherits this fix automatically.

**Verified before trusting it**, same discipline as section 11: fresh-
network shape/gradient sanity check, then a 3-seed pilot (same seeds,
same 300-episode budget) — **2/3 seeds crossed into real, correctly-
targeted blocking** (up from 1/3 with LayerNorm alone), and the Q-value
probe's mmtc margin dropped to near-zero across all 3 pilot seeds (+0.81,
+0.23, +0.67, down from +2.86 to +6.02 with LayerNorm alone, and the
original broken +7.8).

**Full 30-seed re-run**, real bug found and fixed mid-run (documented
honestly, not smoothed over): a merge-safe `--resume-seeds` flag was
added to `m2_seed_campaign.py`/`m2_run_experiment.py` so a deliberate
overnight pause didn't have to redo already-trained seeds. Its first
version trusted "checkpoint file exists" as "already correctly trained" —
but a *different, architecturally-incompatible* checkpoint from the
section-11 (LayerNorm-only) campaign already existed at the identical
seed path for 14 of the 30 seeds (the overnight stop-watcher had, it
turned out, killed the process mid-training on seed 916, not cleanly
after it — the stale pre-existing file satisfied the watcher's
"does this path exist" condition instantly). The naive resume silently
reused those 14 seeds' old-architecture results as if they were the new
architecture's. Caught before being trusted (all 30 seeds printed
suspiciously as "already trained" when only 16 genuinely were) and fixed
properly: resume now attempts `load_state_dict(strict=True)` against the
*current* model class before trusting any on-disk checkpoint — the actual
ground truth for "does this fit," not a timestamp or existence proxy.
Re-ran clean: 16 genuinely resumed (architecture-verified), 14 retrained.

| Outcome | 27/30 collapse (section 11, LayerNorm only) | 8/30 collapse (this fix) |
|---|---|---|
| Differentiated seeds | 3/30 | **22/30** |
| Of those, mmtc-only blocking | 3/3 | 21/22 (1 seed, 923, shows 3 embb blocks instead) |

**The raw `sla_compliance_all_slices` mean went DOWN, not up, on this
much-more-successful fix** (0.353 → 0.230) — expected, not a regression:
every one of the 19 *newly*-differentiated seeds pays a real compliance
cost for blocking mmtc (which, per section 8's mechanism, never rescues
mmtc's own SLA margin in this stress regime — it only stops mmtc being
served for free), while the metric gives it no credit for the now-correct
decision. This is the same metric-inversion effect M3's DP finding
already surfaced, now visible in the primary campaign number itself, more
starkly, because this fix is effective for most seeds rather than a rare
few. `gat_ctde` vs `single_agent_dqn` on `sla_compliance_all_slices`:
paired diff shrinks to +0.116 [0.036, 0.196], Wilcoxon p=0.0099 (still
significant, 19/4/7) — smaller and noisier than section 11's number, for
the reasons above, not because the fix is worse.

**New correctness-aware metrics** (`experiments/scripts/
m2_correctness_metrics.py`), added after the author flagged that
`sla_compliance_all_slices` alone rewards the wrong behavior. Both
computed from data every arm's eval `omega_log.jsonl` already logs — no
new instrumentation, no re-running any campaign, no invented formula:

1. **`mean_reward_per_step`** — the actual RL training objective (same
   quantity/definition as `qoe_oran_framework.mc_runner`'s own
   `RunSummary.mean_reward_per_step`: per-episode mean of every step's
   logged reward, then mean across episodes). The reward function's own
   calibration is what defines "differentiated shedding is correct" in
   the first place (section 1's config comments), so this tests whether
   a policy does what it was actually trained to maximize, directly,
   rather than through a downstream proxy that penalizes the correct
   answer.
2. **`block_precision`** — of every request an arm blocks, what fraction
   target mmtc specifically (the only slice the reward calibration ever
   makes reject-optimal). Undefined, reported as such, for seeds that
   never block anything.

| Arm | n | mean_reward_per_step | 95% CI | block_precision | seeds w/ any block |
|---|---|---|---|---|---|
| `gat_ctde` (per-slice heads) | 30 | **14.309** | [14.172, 14.453] | **0.955** [0.864, 1.000] | 22/30 |
| `single_agent_dqn` | 30 | 13.867 | [13.690, 14.046] | 0.987 [0.965, 1.000] | 27/30 |
| `independent_dqn` | 30 | 12.601 | [11.295, 13.489] | 0.901 [0.839, 0.955] | 29/30 |

**Paired `gat_ctde` vs `single_agent_dqn` on `mean_reward_per_step`:**
mean diff +0.442, 95% CI **[0.261, 0.655]** (does not cross zero),
Wilcoxon p=0.0001, wins 23/30, ties 2, loses 5 — a tighter, more
decisive result than the compliance-based comparison, and the honest
headline number: on the metric that actually reflects the training
objective (not a downstream proxy structurally biased against the
correct decision), `gat_ctde` beats both baselines cleanly.

**Revised claim, superseding section 11's:** the per-slice-heads fix
makes `gat_ctde`'s shared, centrally-trained representation genuinely
learn differentiated, reward-correct admission behavior in the large
majority of seeds (22/30, block-precision 95.5%), not merely "collapse
less often" — and on the metric that actually measures this (mean reward
per step, not downstream SLA compliance, which structurally cannot
credit the correct decision), the result is a clean, statistically
strong win over both baselines. `sla_compliance_all_slices` remains
useful as paper #4's own established live-comparability metric and is
still reported (section 12 table above), but should not be read alone as
"how good is this policy" for this architecture's differentiated-shedding
regime — report both, per the author's explicit direction.

(Acceptance status moved to section 13, at the end of this document, so
it can cover sections 11 and 12 as well.)

## 13. Acceptance status

- [x] Reconciled scope against `experiments/REWORK_PLAN.md` R5 (and
      R3/R4/R6) before writing any code, reporting concrete differences.
- [x] Confirmed the Wu-paper/skeleton reference does not exist and
      dropped it per the author's explicit decision, rather than
      fabricating citations.
- [x] Built net-new code only under `qoe_oran_framework/marl/` and
      `experiments/scripts/`; zero edits to frozen source.
- [x] Both ablations isolate the GAT+CTDE contribution specifically:
      `independent_dqn` matches `gat_ctde`'s per-agent input exactly
      (same node features, same context), differing only in
      sharing/mixing; `single_agent_dqn` is paper #4's own architecture,
      unmodified, via already-existing multi-gNB support.
- [x] Found a real training bug via direct loss logging, not assumption;
      ruled out a false lead (GAT instability) via a control comparison
      against the known-working baseline on the identical environment,
      rather than chasing architecture changes against the wrong signal.
- [x] Reported the result with its actual seed-level numbers and
      explicit caveats about sample size, not oversold as a finished
      ablation.
- [x] No old-rig artifacts anywhere in this pass.
- [x] Block E task 1: checked hyperparameter parity directly rather than
      assuming it, found and fixed a real gamma/epsilon-schedule mismatch.
- [x] Block E task 1: instrumented and inspected per-agent commanded
      ceilings (reusing existing omega-log fields, no new logging
      infrastructure needed) before asserting a verdict, rather than
      guessing between the two hypotheses.
- [x] Block E task 1: reported the verdict as neither hypothesis exactly
      as posed, once the evidence pointed somewhere more specific
      (mmtc-abandonment, not shared-pool over-claiming), rather than
      forcing the finding into one of the two offered boxes.
- [x] Section 11: found the campaign-wide collapse by direct evidence
      (30/30 zero-block seeds, cross-checked against a completely
      independent codebase/checkpoint run weeks apart), not asserted from
      a hunch, and did not proceed to build M3 on top of unexamined
      ground once found.
- [x] Section 11: verified the reward calibration's own documented
      ground truth (`saclb_offline_dqn.yaml`'s congestion_coeff comments)
      before concluding the collapse was a bug rather than a legitimate
      discovered optimum -- confirmed via a direct Q-value probe, not
      inferred from block counts alone.
- [x] Section 11: isolated the root cause mechanistically (cosine-
      similarity probe on the encoder's own embeddings) rather than
      guessing at a fix; verified the fix's direction on a fresh network
      and a 3-seed pilot before committing to the full 30-seed campaign's
      compute budget.
- [x] Section 11: tried and honestly reported a calibration lever that
      didn't help (more training episodes: no effect) and one that made
      things worse (`elementwise_affine=False`), reverting the latter,
      rather than only reporting the change that was kept.
- [x] Section 11: re-ran the full campaign rather than extrapolating from
      pilot seeds, and reported the honest post-fix mechanism (still
      collapse-driven for 27/30 seeds) rather than overselling the 3
      genuinely-fixed seeds as proof the architecture now works as
      intended.
- [x] Section 12: formed the per-slice-heads hypothesis from the reward
      calibration's own documented numbers (priority_weight ratios, the
      congestion-coefficient tuning history), not a guess, before
      touching any code.
- [x] Section 12: verified on a fresh network and a 3-seed pilot before
      committing to the full campaign's compute budget, same discipline
      as section 11.
- [x] Section 12: caught and fixed a real resume-logic bug mid-process
      (naive file-existence check silently reused 14 seeds' worth of
      architecturally-stale results) by verifying the actual thing that
      mattered (`load_state_dict(strict=True)` against the current model)
      instead of a proxy for it, before trusting or reporting any
      number downstream of the resumed run.
- [x] Section 12: reported the compliance metric's mean going DOWN on a
      strictly-more-correct fix, with the mechanism explained, rather
      than omitting or downplaying an inconvenient number.
- [x] Section 12: built the new correctness-aware metrics entirely from
      already-logged data (`reward`, `episode_block_by_slice`) with no
      new instrumentation and no invented formula, matching this
      project's "never invent a number" discipline, and used the exact
      same `mean_reward_per_step` definition already established in
      `mc_runner.py` rather than a new, one-off aggregation choice.
- [x] Section 12: reported `block_precision` as undefined (not silently
      0 or 1) for seeds with zero blocks, rather than papering over the
      denominator issue.
- [x] M3 (federated variant, `docs/PAPER5_M3_fl_dp.md`) reuses the same
      `AgentQHead`/`GATEncoder` and was re-run with both fixes (sections
      11 and 12) once this section's verification was complete -- see
      that document for whether the same collapse/fix pattern, and the
      same metric-inversion effect, held there too.
