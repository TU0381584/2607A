# Rework plan (Stages R0–R9) — audit findings and task breakdown

Status: **R0 complete.** No runs, no training were performed. Everything below is
read-only repo archaeology plus a plan. Stop-gate: do not start R1 until approved.

---

## 0. Read this before approving anything else

The R0 brief's premise and this repo's actual state disagree on one load-bearing point:

> **"Existing scaffold: `qoe_oran_framework/` (~630 lines, 8 modules) implementing GAT +
> CTDE MARL + FL."**

This is not accurate. The framework is **2,622 lines across 13 core modules** (plus
`policies/`, `calibration/`, `comparators/`, `xapp/` subpackages), and **none of it
contains GAT, CTDE, or FL code** — confirmed by grepping every module for
`gat|GAT|ctde|CTDE|federat`. What exists is single-agent: `policies/dqn_admission.py`,
`policies/a2c_admission.py`, `policies/rainbow_admission.py`, each a standard
single-network admission policy with no multi-agent structure, no graph encoder, no
centralized-critic/decentralized-actor split.

**Consequence for R5:** "wire the existing GAT+CTDE modules to the live env adapter"
is not a wiring task. It's a from-scratch implementation of a graph-encoder + CTDE
architecture, plus its own training/debugging cycle, on top of a 300-episode×3-seed
retraining requirement. That's a materially different time cost than the brief implies.
I'd flag R5 as the single biggest schedule risk in this plan, independent of any
deadline you're working against.

I'm not blocking on this — R0 doesn't require resolving it, and you may already know
this and want R5 built for real. Flagging it now so the R1–R9 estimate below isn't
built on a false premise.

---

## 1. Repo map

**Orchestrator / entry points**
- Offline training: `framework/qoe_oran_framework/scripts/train_offline.py` →
  `mc_runner.run_mc()` (`framework/qoe_oran_framework/mc_runner.py`, 538 lines).
- Live evaluation: `framework/qoe_oran_framework/scripts/run_live_mc.py` → same
  `run_mc()`, swapping `LiveKpmSource` for `ClosedLoopKpmSource`/`ReplayKpmSource`.
- Campaign-level shell drivers (arm loops, seeds, logging, restart-on-failure):
  `experiments/scripts/run_phase2_training.sh`, `run_phase3_trial.sh`,
  `run_phase3_trial30.sh`, `run_phase_a_campaign.sh`.
- `mc_runner.build_policy(algorithm, cfg, **overrides)` (line 111) is the arm registry:
  `if algorithm == "dqn"/"a2c"/"rainbow"/"lb_only"`. No config-driven registry — adding
  an arm means adding another `if` branch here. `gat_ctde` is not a branch that exists.

**Config schema** (`framework/qoe_oran_framework/config.py`)
- `SliceSpec` already has `nominal_ratio`, `min_ratio_floor`, `max_ratio_cap` as
  first-class fields — R2/R3's `static_nominal` vs `static_at_cap` distinction is
  already representable in the existing schema (`nominal_ratio` vs `max_ratio_cap`),
  not something that needs new schema.
- `ArrivalConfig.ceiling_step_ratio` (default 5) is the ceiling-step size — see below.
- `QoeConfig` is optional (`None` unless a config opts in), added additively — this is
  the existing pattern R1's phase-schedule block should probably follow (a new optional
  `demand_schedule:` section, absent = old constant-rate behavior).
- All in `experiments/configs/*.yaml` territory — none of this is frozen source.

**Traffic generation (offered rate / packet size / burst duty cycle) — the one part
that lives outside `qoe_oran_framework/` entirely:**
`experiments/traffic/run_traffic_profiles.sh` + `experiments/configs/traffic_profiles.yaml`.
- eMBB: `iperf3 -u -b 4M -l 1200 -t 0` (sustained, UE1 default netns).
- URLLC: `iperf3 -u -b 300K -l 100 -t 0` (sustained, `ue3ns`).
- mMTC: bash loop, `iperf3 -u -b 50K -l 80 -t 2`, `sleep 6` (2s-on/6s-off, `ue2ns`).
- **This script only starts fixed-rate, run-forever iperf3 sessions.** There is no
  phase/schedule concept anywhere in it — R1's three-phase table (different rate *and*
  different mMTC duty cycle per phase, synchronized to step 0/20/40 of a 60-step
  episode) does not exist and has no analog to extend; it needs new logic to kill and
  restart each slice's iperf3 process at phase boundaries, timed against the Python
  step loop. This is buildable as an `experiments/traffic/` extension (not frozen
  source) but it's a real integration point: something has to signal phase transitions
  from the orchestrator (Python, 5s/step) to this bash-level process manager, which is
  new plumbing, not a config edit.
- Traffic profile is manually operated (`start|stop|status` CLI), so today a human runs
  this once per campaign and it holds constant for the whole thing — this is why the
  brief's "identical demand schedule across every arm/seed" requirement needs is
  currently trivially true, and will require actual work to keep true once demand
  varies within an episode.

**Ceiling command path (`min_ratio`/`max_ratio`)**
`action_mapping.AdmissionGate.apply()` → mutates `SliceCeiling.max_ratio` by
±`step_ratio` (config's `ceiling_step_ratio`) per accept/reject, clamped to
`[min_ratio_floor, max_ratio_cap]` → `LiveKpmSource.send_control()` →
`build_control_request()` → raw UDP protobuf `CONTROL` message to gNB port 6655,
applied directly to `gNB_MAC_INST` via the gNB's own `apply_slicing_ctrl()` (fire-and-
forget, no response). Ceiling-step size = `ArrivalConfig.ceiling_step_ratio`, default 5,
config-settable, not frozen.

**The `blocks` counter — traced to origin**
Two counters, kept deliberately separate (`action_mapping.py`'s module docstring is
explicit about this):
- `primary_blocks`: incremented exactly once per `action == 0` (reject) decision the
  *policy* makes — this is a **policy-internal admission-gate decision**, not anything
  the scheduler or PDU layer reports. There is no corresponding "packet dropped" event
  anywhere in the OAI stack for this counter; confirmed by the module's own docstring
  (`BLOCK_MAPPING_LIMITATION`) and by `LiveKpmSource.notify_rejected()` being a no-op on
  the real rig (a reject has no live-observable side effect beyond the ceiling nudge
  itself).
- `secondary_blocks`: a diagnostic-only, non-summed corroborating signal — flagged when
  the *previous* step's ceiling was already below *this* step's observed demand ratio,
  i.e. "the gNB couldn't have served this even absent a new reject." Never added to
  `primary_blocks`.
- This matches R8's instruction to rename `blocks/episode` → "agent-issued rejections"
  exactly — the code and its docstrings already describe it that way; only the
  paper's/plots' labeling needs to catch up.

**MOS mapper — inputs and calibration data**
`qoe_mapper.py`: IQX closed-form prior (eq. 3, per-slice coefficients in
`DEFAULT_IQX_COEFFS`) + optional per-slice LSTM residual refinement
(`QoEMapper`, bounded `tanh` correction on top of the IQX prior, never a from-scratch
regression). Inputs: `[latency_proxy_norm, staleness_norm, packet_loss,
throughput_norm, iqx_prior_mos]`, windowed (default 8 steps).
- Latency has no real E2SM-KPM field on this OAI build (confirmed against the .proto
  and MAC scheduler C source, documented in the module docstring) — `LatencyProxy` uses
  held-last-nonzero `dl_mac_buffer_occupation` with explicit staleness tagging as a
  substitute, not a synthetic zero.
- Calibration data (`calibration/fit_iqx.py`, `calibration/acr_scoring.py`,
  `calibration/video_client_model.py`): objective MOS labels generated from a **P.1203
  video-client model** (mmtc/urllc get closed-form ACR scoring functions instead) over
  a synthetically sampled throughput/latency/loss range — `THROUGHPUT_PRB_RANGE =
  (0.05, 5.0)` PRB/UE, chosen to match live-observed ceiling, with per-slice
  latency/packet-loss ranges deliberately widened past what one live session's light
  traffic produced, specifically so the fit spans "clearly good" to "clearly bad."
- **This is directly relevant to R4's premise.** The current calibration already
  acknowledges it's sampling a wider range than observed live traffic produces — R4's
  diagnosis step (sweep the mapper's inputs over what the R1 schedule can actually
  produce, and check where it saturates) is a legitimate, well-motivated next step
  given this, not a shot in the dark.

**`omega_log.jsonl` schema** — one JSON object/line, fields:
`role, method, objective, constraint, evidence{...}, limitation, run_id, episode, step,
timestamp_s, mode`. `evidence` (the payload plotting scripts read) includes: `seed,
reward, primary_block_count, secondary_block_count, accepted_counts{slice: n},
fairness_ratio, n_pending, ceilings{gnb:slice: {min_ratio,max_ratio}}, mean_mos,
mos_by_slice{slice: mos}, cost, sla_viol, per_slice_compliant{slice: bool},
per_slice_sla_margin{slice: margin}`. `limitation` is enforced non-empty at write time
(`OmegaTuple.__post_init__` raises otherwise) — this is a real, structural
no-silent-fabrication guarantee, not just a convention.

**Plotting scripts** — `experiments/plots/fig{1..7}_*.py` + `generate_results_tables.py`,
each reads `omega_log.jsonl` directly via `common.py`'s `read_omega_log()`. `fig7`
(QoE-reward decomposition) exists but is not currently referenced in `main.tex`.

**Health-check / restart mechanism** — `experiments/scripts/health_check.sh` +
`restart_ran_stack.sh`, invoked from the phase-campaign drivers. Exists, not frozen,
straightforward to keep threaded through R7.

---

## 2. GAT + CTDE + FL — confirmed absent, what wiring would actually mean

- No graph-neural-network code anywhere in the repo (no `torch_geometric`, no
  attention-over-nodes implementation, no adjacency/edge-feature construction).
- No CTDE structure (no centralized-critic/decentralized-actor split; the three
  existing policies are each independently-instantiated single agents with no shared
  critic or joint-state input).
- No FL code (no client/server aggregation, no weight-averaging round) — consistent
  with the brief's "FL stays disabled," except there's nothing to disable; it isn't
  built at all in this codebase, only presumably in the separate journal-paper repo
  the brief's docstring references.
- **What R5 actually requires:** a new GAT encoder (3-node fully-connected graph, the
  node/edge features the brief specifies are already all present in `omega_log.jsonl`'s
  `evidence` shape — good sign, the state the encoder needs is already being logged),
  a new centralized critic + 3 decentralized actors, a new training loop variant in
  `mc_runner.py` (or a parallel runner) to handle joint-vs-per-agent updates, and new
  `build_policy()` branches (`gat_ctde_sla`, `gat_ctde_qoe`). None of this touches
  frozen source — it's all new code under `qoe_oran_framework/policies/` or a new
  `qoe_oran_framework/marl/` subpackage — but it is new code, not integration.

---

## 3. R1–R9 task breakdown: files touched, wall-clock

| Stage | Files touched | Nature of work | Est. wall-clock |
|---|---|---|---|
| R1 | `experiments/configs/*.yaml` (new `demand_schedule:` block), `experiments/traffic/run_traffic_profiles.sh` (phase-timed iperf3 restarts — **new logic, not extension of existing start/stop**), new orchestrator hook to signal phase transitions, omega_logger evidence field for commanded rate | New plumbing between Python step loop and bash traffic process manager | 0.5–1 day dev + 1 smoke episode (5 min) |
| R2 | `experiments/scripts/` new demand-probe script, `experiments/calibration_report.md` (new) | 3× (open-ceiling probe × 3 phases), contention gate × 3 phases | ~0.5 day dev + a few hours of rig time (gate must physically re-verify per phase) |
| R3 | `experiments/configs/*.yaml` (new `static_at_cap` arm config using existing `max_ratio_cap` field — no schema change needed), analysis-layer script for `oracle_static` (post-hoc, no live time) | Mostly config + one analysis script | <0.5 day |
| R4 | `calibration/fit_iqx.py`, `calibration/train_lstm.py` re-run against R1's achievable range; new unit test; `experiments/mos_calibration_report.md` | Re-fit + acceptance test authoring | 0.5–1 day, contingent on whether the ≥0.5 MOS / monotone acceptance test passes on the first re-fit or needs iteration |
| R5 | New GAT encoder, new CTDE critic/actor split, new `mc_runner.py` training-loop branch, new `build_policy()` arms | **Net-new architecture**, not wiring (see §2) | Highest-variance item in this table — could be 2 days or a week+ depending on debugging. This is the one I'd cut first under time pressure, per the brief's own fallback note. |
| R6 | Retrain 6 arms × 3 seeds × 300 episodes offline | Compute-bound, but 2 of the 6 arms (gat_ctde_*) don't exist until R5 ships | Offline compute: hours, not rig-limited. Blocked on R5 for 2/6 arms. |
| R7 | Live campaign, 8 arms × 5 seeds × 5 episodes × 5 min/ep = 200 episodes × 5 min = **~17h** rig time (brief's own estimate), plus restarts/resets between every arm | Rig-time bound, matches brief's stated fallback (drop to 3 seeds, ~10h, if 17h is prohibitive) | ~17h nominal, ~10h fallback |
| R8 | New figs 2–7 regenerated, new stats (Wilcoxon, bootstrap CI, regret-vs-oracle), Table III rework | Analysis + plotting, no rig time | 1 day |
| R9 | `main.tex` full reframe, all `TODO(MANOJ)` sections, bib `VERIFY:` resolution, lit check for "first live E2 RL testbed" claim | Writing, much of it inherently yours (novelty/significance claims) | Depends on you; R8's outputs are the bottleneck before this can start |

**Total, full R0–R9, sequential:** on the order of **2.5–4 weeks** if R5 goes smoothly,
more if it doesn't — dominated by R5's from-scratch MARL implementation and R7's ~17h
of scarce, restart-prone rig time (which itself can't start until R1/R2/R3/R4/R6 all
land first).

**Minimum viable subset (R1, R2, R3, R7, R8)** as the brief itself suggests: **on the
order of 4–6 days** — R1 (~1 day) → R2 (~1 day incl. rig time) → R3 (~0.5 day) → R6
(offline retrain, 4 arms only, hours) → R7 pilot+full (with 6 arms not 8: drop
`gat_ctde_sla/qoe`) → R8 (~1 day). This is the version that's actually compatible with
a one-week clock.

---

## 4. Open questions before R1 can start

1. **Deadline check.** Last session's context had this paper essentially done (TODO
   sections + judgment calls only) with a one-week submission deadline. This rework
   plan is a different, larger project — full re-campaign, reframed contribution,
   ~17h of new rig time. Worth confirming this supersedes that deadline/paper rather
   than running in parallel with it.
2. **R5 scope decision** — build GAT-CTDE for real (multi-day, highest schedule risk),
   or treat it as explicitly out of scope for this pass (per the brief's own "drop R4
   and R5 last" / minimum-viable-subset guidance)?
3. Confirm who/what actually runs `run_traffic_profiles.sh` today during a live
   campaign (manual operator step per CAMPAIGN_LOG, not called from any committed
   orchestrator script) — R1's phase-schedule automation needs to hook into whatever
   that operational process actually is, which I don't have full visibility into from
   the repo alone.

**STOP — awaiting approval before starting R1.**
