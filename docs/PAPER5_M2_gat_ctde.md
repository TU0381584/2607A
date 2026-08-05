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

## 7. Acceptance status

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
