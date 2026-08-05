# Paper #5 Status — Inventory & Gap Analysis

Read-only inventory pass. No code was modified to produce this document.
Every file listed in the task brief under `framework/qoe_oran_framework/`
(8 subdirectories + 12 top-level files) was read in full, along with the
sibling `framework/drl_slicing/` package, the two offline-environment
scripts named in the brief, and every `docs/STAGE*.md` / `docs/BRIDGE_AUDIT.md`
/ `docs/REPRODUCIBILITY.md` file that exists.

**Housekeeping note, confirmed before anything else:** a colleague's draft
instructions referenced `ACTION_PLAN_conference_and_journal.md` and a
package at `qoe_oran_framework/`. Neither is correct for this repo:
`ACTION_PLAN_conference_and_journal.md` does not exist anywhere in this
repository or its git history (re-confirmed this pass, `find . -iname
"ACTION_PLAN*"` returns nothing, matching `docs/AUDIT_stage0.md`'s own
prior confirmation), and the actual framework package lives at
`framework/qoe_oran_framework/` (there is also a sibling
`framework/drl_slicing/` package, described in §1). Per the author's
explicit decision, `docs/BRIDGE_AUDIT.md` and `docs/STAGE*.md` substitute
for the missing action-plan doc's stage cross-referencing (§3).

---

## 1. Per-module status: `framework/qoe_oran_framework/`

**Headline verdict, established once and applying to every module below:**
a repo-wide, case-insensitive search for graph-attention/GAT/GATConv,
CTDE, and federated-learning/FedAvg/FedProx terms across `framework/` and
`experiments/` (not just `qoe_oran_framework/`) returns **zero hits**
outside of unrelated third-party library internals under `venv/`
(e.g. `torch/cuda/_graph_annotations.py`, PIL/pygments matches on the
substring "graph"). **GAT: ABSENT. CTDE MARL: ABSENT. FL: ABSENT.**
Nowhere in this codebase, not stubbed, not partial — no file anywhere
under `framework/` even imports `torch_geometric` or defines an
attention-over-neighbours mechanism. Paper #3 (the review PDF, external
to this repo) is the only place any of these three things are described.

### configs/
7 YAML files (`saclb_live.yaml`, `saclb_live_lowlmax.yaml`,
`saclb_live_tightcaps.yaml`, `saclb_offline_dqn.yaml`,
`saclb_offline_live1gnb.yaml`, `saclb_offline_rainbow.yaml`,
`saclb_paper1_sac_only.yaml`). Plain `SacLbExperimentConfig` YAML —
slice specs, gNB list, reward weights, arrival/episode timing. **Single-
agent.** `config.py`'s `gnbs: List[GnbSpec]` field technically allows
listing more than one gNB (paper #2's LB-extension case), but as detailed
under `env.py` below, this never produces multiple independent agents —
it only widens one shared observation vector. GAT/CTDE/FL: absent.

### calibration/
`fit_iqx.py` (least-squares fit of per-slice Weber-Fechner/"IQX"
coefficients against P.1203/ACR objective MOS labels), `train_lstm.py`
(trains one LSTM per slice — `QoEMapper` — as a bounded-residual
correction on top of the closed-form prior), `acr_scoring.py`
(application-specific ACR-style scoring for URLLC/mMTC, since P.1203
doesn't apply to non-perceptual traffic), `video_client_model.py`
(simulated ABR/DASH client feeding the official `itu_p1203` reference
implementation for eMBB labels), `units.py` (PRB↔kbps/latency unit
conversions), `fitted_iqx_coeffs.json` (fitted output). **Single-slice-
scoped, not multi-agent** — each slice's coefficients/LSTM are fit
independently with no cross-slice or cross-gNB coupling. GAT/CTDE/FL:
absent.

### xapp/
`saclb_xapp.py` — the live xApp main loop. Explicitly single-physical-gNB
by hard design constraint (`parser.error` if the config lists != 1 gNB,
line ~76-79) and evaluation-only (loads a frozen checkpoint, never trains
live). **Single-agent.** GAT/CTDE/FL: absent.

### tests/
12 test files covering `action_mapping`, `calibration`, the closed-loop
KPM source, `comparators`, `env` (offline), `kpm_adapter`,
`live_kpm_source`, `mc_runner`, `omega_logger`, `policies` (shape checks),
`qoe_mapper`, `reward`. All exercise the single-agent `RANEnv`/policy
stack; no test constructs more than one policy instance or a graph
structure. GAT/CTDE/FL: absent, no test scaffolding exists for any of
the three.

### results/
Artifacts from this project's own early "Stage Zero" framework bring-up
smoke tests, **on the current rig** (all files dated 2026-07-14,
matching `BRINGUP_LOG.md`'s bring-up window for this fresh rig — see §4
for why these are flagged for exclusion anyway, on provenance-chain
grounds rather than literal old-rig grounds): `offline_live1gnb/{a2c,dqn}`
and `offline_live1gnb_qoe/{a2c,dqn}` (each a `summary.json` +
`rep_0/checkpoint.pt`), `qoe_mapper/` (3 trained per-slice LSTM `.pt`
files + `lstm_train_results.json`), `live/lb_only/stage9-smoke-001_omega_log.jsonl`
(one smoke-test log). All single-agent, single-gNB. GAT/CTDE/FL: absent.

### comparators/
`lb_only_baseline.py` (contextless "route/admit to lowest-utilisation
gNB" heuristic, no learning) and `sac_only.py` (paper #1's comparator —
just `DQNAdmissionPolicy` run against a `paper_variant: paper1`,
single-gNB config; not a separate algorithm). **Single-agent.**
GAT/CTDE/FL: absent.

### scripts/
`train_offline.py` (offline DQN/A2C/Rainbow training against
`ClosedLoopKpmSource`/`SyntheticKpmSource`), `run_live_mc.py` (live
Monte-Carlo runner — hard-requires `len(cfg.gnbs) == 1`, line ~63-67),
`run_saclb_live_testbed.sh`/`stop_saclb_live_testbed.sh` (shell
orchestration), `probe_e2_preconditions.py` (live E2 wire-protocol/KPM-
population precondition probe). All single-agent/single-gNB oriented.
GAT/CTDE/FL: absent.

### policies/
`dqn_admission.py`, `a2c_admission.py` (both thin wrappers fixing
`action_dim=2, n_branches=1` on `oranslice_drl.drl_policy`'s
`DQNPolicy`/`A2CPolicy` base classes — see `drl_slicing/` below),
`rainbow_admission.py` (Dueling + Double-Q + NoisyNet + PER, built fresh
for this project, confirmed via its own docstring that no Rainbow
implementation existed anywhere in the codebase before). **Each of the
three is a single shared policy instance**, called once per pending
request (binary accept/reject over a per-request state vector) — no
per-gNB or per-agent policy set, no centralized-critic/decentralized-
actor structure. GAT/CTDE/FL: absent.

### Top-level files (`env.py`, `action_mapping.py`, `mc_runner.py`, `replay_kpm_source.py`, `reward.py`, `omega_logger.py`, `types.py`, `config.py`, `kpm_adapter.py`, `live_kpm_source.py`, `qoe_mapper.py`, `_oranslice_path.py`)
**Important nuance for scoping paper #5's M2 (multi-agent) milestone:**
`config.py` does support `gnbs: List[GnbSpec]` (paper #2's multi-gNB
LB-extension case is already exercised in `saclb_offline_dqn.yaml`-style
configs), but `env.py`'s `encode_state()` (lines 27-44) concatenates
**every** gNB's per-slice features into **one flat vector**, plus a
single scalar fairness ratio — and `mc_runner.run_single`/`run_mc` drive
exactly **one shared policy instance** across all pending requests,
regardless of gNB count (`_select_actions()`, `mc_runner.py` lines
128-148). This is a **centralized single-agent** formulation over a
(possibly multi-gNB) observation space — not decentralized multi-agent
CTDE. There is no per-gNB agent, no GAT encoder over a gNB adjacency
structure, and no independent actor-per-gNB with a shared/centralized
critic anywhere in this stack. `reward.py` similarly computes one scalar
reward per step, shared across the whole cluster (no per-agent reward
decomposition). `live_kpm_source.py` is hard-wired to exactly one
physical gNB's UDP E2 agent. GAT/CTDE/FL: absent throughout.

### `framework/drl_slicing/` — the sibling package (checked per the task brief)

This is a **different, older, more general** package that `qoe_oran_framework`
imports from directly (`mc_runner.py`: `from oranslice_drl.drl_training
import ReplayBuffer`; `policies/dqn_admission.py` and `a2c_admission.py`:
`from oranslice_drl.drl_policy import DQNPolicy` / `A2CPolicy`), made
importable via `_oranslice_path.py`'s `ensure_oranslice_drl_importable()`.
It is the actual home of paper #4's core DQN/A2C algorithm implementations:

- `oranslice_drl/drl_policy.py` — `RLPolicy` ABC, `DQNPolicy`, `A2CPolicy`
  (the `QNetwork`/`ActorCriticNetwork` classes `qoe_oran_framework/policies/*.py`
  subclass directly, not copy).
- `oranslice_drl/drl_training.py` — `StateEncoder`, `ActionDecoder`
  (ratio-bin action decoding, unused by the admission-gate mapping),
  `ReplayBuffer` (the base uniform replay buffer `mc_runner.py` imports).
- `oranslice_drl/runner.py` — its **own**, separate orchestration loop
  (`Controller`/`RuleBasedSLAController`/`ThresholdHeuristicController`/
  `DQNController`/`A2CController`/`PPOController`, live-OAI-stack-aware via
  `_assert_live_oai_stack_running()`), distinct from and not used by
  `RANEnv`/`mc_runner.py` — confirmed there is no `step()`/`reset()`
  gym-like abstraction here (per `env.py`'s own module docstring: "No such
  step()/reset() abstraction exists in drl_slicing's runner.py").
- `oranslice_drl/{collectors,controllers,offline_warmstart,policy_io,types,
  reward,config}.py` — a parallel, PPO-inclusive controller/reward/config
  stack, generic slice-ratio-control (not binary admission-gate specific).

**No GAT/MARL/FL code exists here either** (repo-wide grep confirmed zero
hits). This package is the algorithm library `qoe_oran_framework` wraps
with its admission-control-specific env/reward/action-mapping layer — it
is not itself a multi-agent framework, and does not need to be treated as
one for paper #5 scoping purposes.

---

## 2. Offline simulation environment parameter inventory

All three items live inside `ClosedLoopKpmSource`
(`framework/qoe_oran_framework/replay_kpm_source.py`, class starts line
143) — the "meaningful-numbers" offline KPM source (as opposed to the
open-loop `SyntheticKpmSource`, whose numbers are explicitly documented
as wiring-smoke-test-only, not learned-behaviour evidence).

### `backlog_capacity`
- **Constructor default:** `replay_kpm_source.py:192` —
  `backlog_capacity: float = 200.0`, stored at `replay_kpm_source.py:218`
  (`self._backlog_capacity = backlog_capacity`).
- **Consumed at:** `replay_kpm_source.py:283` (`self._backlog[key] =
  min(self._backlog_capacity, unmet)`) and `:291` (`backlog_frac =
  self._backlog[key] / max(self._backlog_capacity, 1e-6)` — this is what
  drives the loss/`bler` channel, see below).
- **Current training-entrypoint value (what actually gets used):**
  `experiments/scripts/train_offline_live_scale.py:33` —
  `BACKLOG_CAPACITY = 2000.0` (module constant), wired to the
  `--backlog-capacity` CLI flag default at lines 44-45, passed into
  `ClosedLoopKpmSource(...)` at line 60.
- **A second, different default exists** in the validation-sweep helper:
  `experiments/scripts/live_scale_offline_env.py:51` —
  `DEFAULT_BACKLOG_CAPACITY = 200.0`, used by `make_env()` at line 61/70.
- **Tied to a live-measured value? NO.** Hardcoded, chosen purely so
  `accept_all`/`reject_all`/`threshold_like` produce differentiable
  outcomes (Stage 5's own validation criterion) — never checked against
  real live SLA-margin magnitude until Stage 13's investigation.
- **Agreement with `docs/STAGE13_recalibration_attempt.md`:** exact match,
  no drift. STAGE13 states the training script "gained a
  `--backlog-capacity` CLI flag (default unchanged at 2000...)" after its
  sweep found no single value satisfies both realism and differentiability
  — the current file confirms the default is still `2000.0`, i.e. STAGE13's
  recalibration was evaluated and NOT kept as the new default.

### `mean_offered_ratio`
- **`ClosedLoopKpmSource`'s own fallback default:**
  `replay_kpm_source.py:188` (parameter) / `:214` (`self._mean_offered_ratio
  = mean_offered_ratio or {s: 0.5 for s in slice_ids}`) — a generic,
  non-live-tied 0.5 fallback, used only if no caller supplies a value.
- **Current training-entrypoint value:**
  `experiments/scripts/live_scale_offline_env.py:44` —
  `MEAN_OFFERED_RATIO: Dict[str, float] = {"embb": 0.15, "urllc": 0.05,
  "mmtc": 0.05}`, imported into `train_offline_live_scale.py:27` and
  passed to `ClosedLoopKpmSource(...)` at `train_offline_live_scale.py:59`.
- **Tied to a live-measured value? YES, but of the wrong quantity.** The
  module docstring states these values come from "repeated, direct
  `probe_e2_preconditions.py` measurements on the real rig." STAGE13
  flags this precisely: these are **point-probe** measurements (an
  open-ceiling instantaneous demand reading), not "effective demand under
  a specific commanded ceiling, measured across a full episode" — one of
  STAGE13's two named unexplored next hypotheses for why recalibration
  didn't close the offline/live gap.

### Temporal-dynamics update (`0.1*(mean-offered)+noise`)
- **Location:** `replay_kpm_source.py`, inside `ClosedLoopKpmSource.poll()`,
  lines 261-264:
  ```
  261: mean = self._mean_offered_ratio[slice_id] * self._B * self._gnb_load_multiplier[gnb_id]
  262: drift = 0.1 * (mean - self._offered[key])
  263: noise = self._rng.normal(0.0, self._offered_volatility * self._B)
  264: self._offered[key] = max(0.0, self._offered[key] + drift + noise)
  ```
- **`offered_volatility` parameter default:** `replay_kpm_source.py:190`
  — `offered_volatility: float = 0.04`, hardcoded, not tied to any
  live-measured variance/burstiness statistic anywhere in the repo.
- **Not disputed by, but sharpened by, `docs/STAGE12_offline_online_gap.md`:**
  Stage 12 root-caused the offline/live gap specifically to this update's
  TEMPORAL STRUCTURE, not its mean: matching the mean-reverting walk's
  mean to real probe data (Stage 5's fix) does not reproduce real
  traffic's burstiness/autocorrelation — the `0.1` drift coefficient and
  `0.04` volatility constant are both uncalibrated constants, never fit
  against any live traffic autocorrelation/burstiness measurement. This
  is the mechanism STAGE13's backlog_capacity sweep then failed to fix
  (because it targeted the wrong knob, per its own honest conclusion).

**No drift found** between the current file contents and what
`docs/STAGE13_recalibration_attempt.md` describes — all values/state
match exactly.

---

## 3. Stage cross-reference (substituting for the missing action-plan doc)

**`ACTION_PLAN_conference_and_journal.md` does not exist anywhere in this
repository or its git history (confirmed via exhaustive search); this
section substitutes `docs/BRIDGE_AUDIT.md` and `docs/STAGE*.md` as the
closest existing equivalent, per the author's explicit decision.**

`docs/` contains `AUDIT_stage0.md`, `BRIDGE_AUDIT.md` (Stage 8),
`STAGE1_diffs.md` through `STAGE6_scarcity.md`, `STAGE9_hygiene.md`
through `STAGE15_n128_campaign.md` (STAGE7 and a standalone STAGE8 file
do not exist as separate docs — Stage 8's content is `BRIDGE_AUDIT.md`).
All were read this pass.

| Stage doc | What it established | Verdict for paper #5 |
|---|---|---|
| `AUDIT_stage0.md` | Confirmed `ACTION_PLAN_conference_and_journal.md` never existed; full claim-to-artifact trace of paper #4; found 2 real citation/prose bugs (eq.9→eq.3, IQX→Weber-Fechner) | Prior-art / provenance — establishes the audit discipline this document follows |
| `STAGE1_diffs.md` | Fixed the eq.9→eq.3 citation bug: paper #3's **eq.(9) is the FedAvg-FedProx FL objective**, eq.(3) is the QoE reward | **Direct prior art for #5's FL component** — pins the exact equation to implement/cite |
| `STAGE2_metrics.md` | Under the paper's own declared priority weights, baseline **beats** DQN on utility in the congested scenario — a real, non-manufactured finding | Caution for #5's reward/utility design — don't assume a learned policy wins on utility by construction |
| `STAGE3_oracle.md` | DQN's zero-collapse record vs. static-at-cap reached p=0.0149 (n=25 vs n=15), under the OLD (pre-Stage-5) calibration | Later superseded/revised by Stage 10 — historical baseline only |
| `STAGE4_instrumentation.md` | Real-hardware engineering budget: E2 round-trip 0.57ms median, DQN inference <70μs median, 18,562 params, control loop hits 5.000s cadence to within 0.7ms | Green light / budget reference — relevant ceiling for #5's added GAT-encoder + multi-agent inference latency overhead |
| `STAGE5_recalibration.md` | Fixed 3 real bugs (MOS throughput-scale, SLA-deadline mismatch, offline demand-scale 4-6x too small); established `live_scale_offline_env.py`/`MEAN_OFFERED_RATIO` | **Direct foundation M1 builds on** — this is the calibration baseline, not something to redo |
| `STAGE6_scarcity.md` | Live demand-driven multi-slice scarcity (real UEs oversubscribing the cell) judged infeasible on this rig: no spare per-slice UE config, and the rig already runs at ~200-400MB free even at 3-UE scale | **Blocker for #5's live multi-gNB/multi-UE validation** — new UE provisioning + memory-headroom fix needed before any live multi-agent contention scenario is attempted |
| `STAGE9_hygiene.md` | Submission hygiene pass; established the old-rig-checkpoint discipline (§6) this document's §4 reuses | Provenance / methodology precedent |
| `STAGE10_fullpower_reeval.md` | At full power (n=21→n=46), DQN-SLA's collapse-avoidance edge over static-at-cap did **not** clearly replicate (p=1.0) | Blocker / motivating finding — opened the offline/live-gap investigation chain Stage 11-13 continue |
| `STAGE11_checkpoint_sensitivity.md` | Live robustness varies enormously by training seed (13/21 to 21/21 fully compliant) despite uniformly healthy offline convergence across all 6 seeds | **Blocker/caveat for M1** — any recalibration fix must be validated across multiple seeds/checkpoints, not one, before being trusted |
| `STAGE12_offline_online_gap.md` | Root-caused the gap to KPM-source **temporal-dynamics** mismatch (burstiness/autocorrelation), not just mean-offered-ratio; offline held-out compliance ranking has **zero correlation** with live ranking | **Direct predecessor to the M1 question** — this is the "why" STAGE13 then tried and failed to fix via `backlog_capacity` alone |
| `STAGE13_recalibration_attempt.md` | See detailed summary below | **The single most relevant prior-art finding for M1** |
| `STAGE14_end_to_end_audit.md` | Full fabrication/anomaly sweep of every paper #4 number — zero fabricated data found, one labelling imprecision (median vs. mean latency claim) | Green light for dataset trust — not directly gating M1's technical question |
| `STAGE15_n128_campaign.md` | Extended live campaign to n=128/arm; DQN-QoE vs. baseline reached significance (p=0.0070); DQN-SLA vs. static-at-cap remains statistically tied even at 3x the sample | Context — reinforces that the SLA-only reward's edge is weak/absent while the QoE-reward's is real; relevant background, not a direct M1 blocker |

### `docs/STAGE13_recalibration_attempt.md` — detailed summary (most relevant to M1)

**Question:** does recalibrating `backlog_capacity` alone close the
offline/online rank-correlation gap Stage 12 found?

**What was done:** traced that `backlog_capacity=2000` (the value
`train_offline_live_scale.py` had been using) was chosen in Stage 5
purely so `accept_all`/`reject_all`/`threshold_like` produce
differentiable outcomes — never checked against real live SLA-margin
magnitude. At `backlog_capacity=2000`, mean offline SLA margin was
~-0.60 vs. real live's stable ~+0.70-0.75 for the same checkpoint. Swept
`backlog_capacity` 200→2000 using the same accept-all/reject-all
differentiability criterion: below ~1000, margin realism improves but
`accept_all`/`reject_all` become statistically indistinguishable
(environment loses the ability to teach anything); above ~1200, they
separate again but margin reverts to unrealistically negative. **No
single value satisfies both constraints** — a genuine structural property
(backlog accumulates every step with no decay mechanism other than being
served), not a tuning oversight. Chose `backlog_capacity=1200` as the
best compromise, **retrained all 6 checkpoints (seeds 256-261) from
scratch**, re-ran the same 100-episode held-out evaluation Stage 12 used.

**Result — not a fix.** Spearman rank correlation between live and
offline compliance: **ρ=0.10, p=0.86 at bc=2000 (original) → ρ=0.23,
p=0.67 at bc=1200 (recalibrated). Both are statistically indistinguishable
from zero at n=6.** Checkpoint 258 — one of three checkpoints with a
PERFECT live record — became the WORST performer offline under the
recalibrated environment, the same non-relationship Stage 12 found, just
reshuffled.

**Two concrete, unexplored next hypotheses named explicitly** (directly
relevant to scoping M1):
1. The offline loss/`bler` channel is **entirely derived from backlog
   fraction** (`bler = 0.02 + 0.3*backlog_frac`,
   `replay_kpm_source.py`'s `ClosedLoopKpmSource.poll()`) — real RF-level
   loss is presumably a more independent process, not a pure function of
   queue depth.
2. `mean_offered_ratio` itself may not represent the **effective demand
   a fixed ceiling actually experiences over a full episode** — the
   probes behind its current values (0.15/0.05/0.05) were point
   measurements under an open ceiling, not full-episode
   demand-under-a-specific-commanded-ceiling measurements.

A third, tangential finding (extreme SLA-margin values, ~3.5-8.5% of
live steps) was traced to a real physical RLC-max-RETX-style failure
event, confirmed NOT to implicate any reported manuscript number, and
not chased further.

**Bottom line: the offline/online gap is not explained by, or fixable
via, `backlog_capacity` alone** — closing it credibly needs either a
fundamentally different demand/loss model, or real experiments
specifically designed to measure "effective demand under a given
ceiling," not open-ceiling probes. This is exactly the starting point
M1 (recalibrating the offline environment against live SLA-margin
traces) needs to internalize before re-attempting the same class of fix.

---

## 4. Old-rig exclusion check

`docs/REPRODUCIBILITY.md`'s own "Old-rig-checkpoint check" section
(lines 69-80) establishes the pattern: every number the paper cites
traces to `experiments/results/live_campaign*` or
`experiments/results/offline*` (this rig, this project's sessions); the
only "old rig" mentions in the active `experiments/configs/saclb_campaign*.yaml`
files are header comments stating their values were "validated live
against this campaign's actual traffic profile, **not** inherited from
the old rig" — i.e. the phrase appears only to rule out reuse, confirmed
again independently in `docs/STAGE14_end_to_end_audit.md`'s fabrication
sweep (no config/script/doc under active use references an old-rig path;
no checkpoint file is a symlink).

Applying that same pattern to `framework/qoe_oran_framework/`, two things
were found that need flagging for paper #5:

1. **`framework/qoe_oran_framework/results/`** (`offline_live1gnb/`,
   `offline_live1gnb_qoe/`, `qoe_mapper/`, `live/lb_only/`) — all files
   dated 2026-07-14, matching this **current** rig's own initial bring-up
   window (`BRINGUP_LOG.md` documents this fresh rig's bring-up starting
   2026-07-14), so these are not literal old-rig (different physical
   desktop) artifacts. **However, they sit entirely outside the
   `experiments/results/live_campaign*`/`offline*` provenance chain**
   `REPRODUCIBILITY.md` validates number-by-number — they are early
   framework wiring-smoke-test artifacts from before the Stage 5
   calibration fixes and before the Stage 12/13 offline/live-gap findings
   existed. **Recommendation: exclude from paper #5 reuse without
   re-verification** — do not treat any checkpoint or summary under this
   path as calibrated or validated evidence.

2. **`framework/qoe_oran_framework/configs/saclb_live.yaml`** — carries
   multiple explicit, still-unresolved old-rig-inherited values, flagged
   in its own comments: line 40 ("copied from the old rig's numbers,
   though they turned out close"), line 47, line 56 ("left at the old
   rig's reference value, flagged"), line 59 (an sd-value correction
   note), and line 87 (`max_ratio_cap: 3  # UNVERIFIED against live
   demand on this rig -- old rig's reference value, no UE was ever
   attached to this slice here`). **Recommendation: do not reuse
   `saclb_live.yaml` for paper #5 without re-verifying these specific
   constants against this rig's own current live-measured demand** — the
   same discipline Stage 5's `MEAN_OFFERED_RATIO` recalibration already
   applied to the `experiments/` configs, not yet applied here.

No literal old-rig `.pt` checkpoint files or symlinks were found anywhere
in the repository via direct search.

---

## 5. Addendum: `experiments/REWORK_PLAN.md` (found after this document's
   first draft, not covered by the agent that produced §1-4)

A separate, pre-existing, **unapproved** staged plan (R0-R9) already exists
at `experiments/REWORK_PLAN.md`, written against the same false premise a
prior brief gave it ("existing scaffold implementing GAT+CTDE MARL+FL") --
which it independently debunked, reaching the identical conclusion as §1
above (GAT/CTDE/FL absent, single-agent throughout, confirmed by its own
grep). It ends with **"STOP -- awaiting approval before starting R1"** and
was apparently never resumed: its R3 assumes a `static_at_cap` arm does not
yet exist, but paper #4's current manuscript already has one, meaning this
plan predates paper #4's current state and was superseded by the actual
Stage 1-15 path the project took instead.

Its R1-R9 breakdown (phased demand scheduling, contention-gate-per-phase,
MOS re-fit, GAT+CTDE build, retrain, live campaign, analysis, rewrite) does
not include an item matching M1 (recalibrating the offline environment's
fidelity against live SLA-margin traces) -- M1 is new territory relative to
this plan, not a duplicate of any R-stage, and if anything is a reasonable
prerequisite to trust R6/R7's future retraining under any new GAT/CTDE arms.
Its two open questions (deadline-conflict check, R5 build-for-real-or-defer
decision) remain unresolved and are the author's call, not something this
document resolves.
