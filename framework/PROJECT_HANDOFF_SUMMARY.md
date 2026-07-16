# ORANSlice Project Handoff Summary

**Purpose of this document:** a self-contained record of what this testbed set out to do, what was actually built and verified, what was learned (including the hard-won negative results), and what to carry over to a fresh Ubuntu + OAI build. Written to be read *without* the surrounding chat history.

**Date:** 2026-07-14
**Repo state at handoff:** commit `c01c7b2` on `feat/qoe-oran-framework-stage-zero-20260711-push` (local; not yet pushed — no git credentials in this sandbox).

---

## 1. Project objective

The end goal is an empirical realization (paper #4) of a QoE-aware, GNN-MARL-FL O-RAN network-slicing framework (paper #3, an IEEE Access survey), grounded by two earlier conference papers:

- **Paper #1 (MECON):** DRL-based slice admission control (DQN / A2C / Rainbow), single-gNB, 2 slices (URLLC, eMBB).
- **Paper #2 (ICRAIE):** Extends #1 with a 3rd slice (mMTC) and a multi-gNB load-balancing term.
- **Paper #3:** The survey this whole line of work targets — QoE-aware admission via a learned QoS→MOS mapper, GNN-based state representation, multi-agent RL, federated learning across gNBs.

Given real hardware constraints (one physical rig, one gNB), the work was staged:

- **Stage Zero** — reproduce papers #1/#2's admission-control setup against a *real* OAI testbed (not a simulator), establish the live E2 control loop, and calibrate a working baseline.
- **Stage One** — layer a QoE mapper on top of the frozen Stage Zero baseline (paper #3's first real component), calibrated against real quality-of-experience standards, without touching anything else ("change one thing").
- **Stage Two+ (not started)** — GNN state representation, multi-agent coordination, federated learning across gNBs — deferred until multiple physical gNBs exist.

---

## 2. Stage Zero: what was built and achieved

### What exists that didn't before
A complete `qoe_oran_framework/` package implementing the papers' admission-control MDP against **any** KPM source (replayed trace, synthetic, or live), including:
- `env.py` — gym-like `RANEnv` (`reset()`/`step()`)
- `kpm_adapter.py` — real E2SM-KPM UE samples → per-slice aggregate state
- `reward.py` — eq. 2 (paper #1) and its LB-extended form (paper #2)
- `action_mapping.py` — binary accept/reject → the real control primitive (see §4 below)
- `live_kpm_source.py` — a from-scratch, reverse-engineered UDP client for OAI's real `E2_AGENT` (see §4)
- `policies/` — DQN, A2C, Rainbow admission policies (reusing base classes from `drl_slicing/oranslice_drl`)
- `comparators/lb_only_baseline.py` — the papers' own "lowest-utilization-gNB" heuristic baseline
- `omega_logger.py` — every run logged as an `{role, method, objective, constraint, evidence, limitation}` tuple; every record's `limitation` field is non-empty by construction
- `mc_runner.py` — Monte-Carlo orchestration with fixed seeds, offline training and live evaluation modes
- Full offline→live pipeline: `scripts/train_offline.py`, `scripts/run_live_mc.py`, `xapp/saclb_xapp.py`, `scripts/run_saclb_live_testbed.sh`

### Key findings (real, verified against the physical rig)
- **First genuine live differentiation** between policies was achieved (LB-only baseline vs. learned policies showing different block rates on real traffic).
- SLA-compliance differentiation between DQN and A2C proved structurally hard to achieve given this rig's `Lmax`/offered-demand-volatility interaction — documented as an open limitation, not silently dropped.
- Full closeout report: `qoe_oran_framework/STAGE_ZERO_REPORT.md` / `.docx` (included in this handoff).

---

## 3. Stage One: what was built and achieved

### The QoE mapper (paper #3's first real component)
- **`qoe_mapper.py`**: `iqx_mos()` (eq. 3, the IQX closed-form QoS→MOS curve), a stateful `LatencyProxy` (holds the last real reading + staleness count — see §4), `RollingKpmWindow`, and `QoEMapper` (an LSTM that predicts a *bounded residual correction* on top of the IQX prior, not a from-scratch regression).
- **`calibration/`**: per-slice objective-label generators — a P.1203 reference-implementation bridge (via a small virtual ABR video client) for eMBB, and ACR-style task-success scoring for URLLC/mMTC (P.1203 doesn't apply to non-perceptual traffic) — plus `fit_iqx.py` (least-squares IQX coefficient fitting) and `train_lstm.py` (LSTM training on synthetic KPM-trajectory windows).
- **Calibration results (real, held-out, reported honestly):**

  | Slice | IQX prior MAE / r | LSTM-refined MAE / r |
  |---|---|---|
  | eMBB | 0.089 / 0.926 | 0.013 / 0.997 |
  | URLLC | 0.131 / 0.742 | 0.055 / 0.911 |
  | mMTC | 0.347 / 0.961 | 0.047 / 0.999 |

  URLLC's weaker IQX-alone fit is a genuine, documented finding (its near-binary deadline profile doesn't fit IQX's smooth log curve as well as the other two) — the LSTM refinement closes most of that gap.

- **`reward_mode="qoe"`**: eq. 9 (`r = α·MOS − β·cost − γ·SLA_viol`) threaded through `env.py`/`mc_runner.py`/all three entrypoints via a `--reward-mode` flag, a genuinely different reward *shape* from Stage Zero's eq. 2, not eq. 2 with one term swapped. `reward_mode="sla"` (the frozen baseline) is regression-tested to be **bit-for-bit unaffected** even when the new QoE-diagnostic machinery is present in a config.
- DQN and A2C retrained offline under the QoE reward, 300 episodes × 2 seeds each, results reproducible (max ~1.5% variance between seeds).

### Live pre-flight evidence (real testbed, not simulated)
- **Fig. 5 (the headline result):** the actual PRB ratio ceiling (`slicing_control_m` max_ratio) each policy commands, logged every step. `lb_only` never leaves the configured floor for any slice, the entire episode; the QoE-trained DQN rides every slice's ceiling to its cap within seconds and holds it there. This is the real controllable output, not an inferred effect.
- **Fig. 6 (the honest caveat):** that ceiling divergence has **not yet propagated** to backlog or inferred MOS at this rig's traffic scale — both policies show nearly identical downstream QoE. Reported as a genuine limitation, not hidden.
- Two root-cause investigations, both concluded with mechanism, not guesswork:
  - **Traffic-rate intensification** (~15–30× heavier packet rate) does not change admission counts (driven by a seeded synthetic arrival stream, decoupled from real traffic by construction) or scheduler PRB grants (`avg_prbs_dl` saturates at the ceiling regardless of offered load) — ruled out.
  - **UE-count scaling** (more simultaneously-attached real UEs, to create genuine multi-flow contention) hit a **confirmed** host-memory ceiling on this shared desktop rig (~1GB RSS per simulated UE, verified via `/proc/<pid>/status`, tied to OAI's real-time PHY buffer allocation at the configured bandwidth — not a tunable inefficiency) — ruled out on *this* machine, though it remains a valid lever on more capable hardware.
- A real bug was caught and fixed *before* it could waste live-rig time: MOS/cost/SLA-violation diagnostics were only computed under `reward_mode="qoe"`, so the baseline arm's live runs would have logged empty QoE data — fixed as a passive, reward-independent diagnostic, regression-tested to leave the baseline's actual reward unchanged.

### Why Stage One's live phase stopped here
A **genuine OAI binary segfault** started reproducing 100% of the time (even a single UE alone crashes within seconds of attach). Root-caused via `dmesg` + `addr2line` (correcting for the ELF's PIE load-segment offset) to two exact locations — `nr_dci_size` and `get_ul_tdalist` in `openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.c` — both reading a UE bandwidth-part config struct that's NULL or incomplete. Ruled out multi-UE races, stale AMF state (survived a full AMF restart + re-provisioning), and port/process conflicts before concluding this is a genuine logic bug in the vendored OAI build, not an environment issue. See §5 for why this motivates the fresh-build decision.

---

## 4. Key technical discoveries (carry these forward — they cost real time to learn)

1. **Real OAI's only xApp control primitive is `slicing_control_m{sst, sd, min_ratio, max_ratio}` — a per-slice PRB-ratio ceiling/floor.** There is no raw "accept/reject this one flow" hook. The papers' binary admission action has to be *realized* as ceiling nudging (accept → ceiling up, reject → ceiling down), not a literal per-flow accept. Capping a slice starves it (demand → backlog); it does not remove demand. This is the single most important architectural fact about this testbed, and will be true on any OAI version.

2. **OAI's E2 agent has a RIC-free UDP loop, but it is easy to reach for the wrong transport.** The correct, verified wire protocol (`live_kpm_source.py`'s docstring has the full detail): gNB listens on UDP `0.0.0.0:6655` for `INDICATION_REQUEST`/`CONTROL` protobufs, responds to UDP `127.0.0.1:6600`. Request/response, not subscribe-and-stream. **Sending a `SUBSCRIPTION`-type message crashes the gNB** (`assert(0!=0)` in `handle_subscription()`) — never send one. The `o-ran-e2sim/kpm_sim` pipeline and `xapp-oai/base-xapp`'s TCP/UDP transport were both tried first and found to require a real E2AP/SCTP RIC Subscription Request before relaying data — i.e. they need an actual RIC, which this rig doesn't have. `live_kpm_source.py` reimplements the socket calls directly and needs **no external xApp framework**. **This wire protocol must be re-verified on the new OAI version** — if the E2 agent's message format changed, this file needs updating, but the *discovery process* (check `e2_agent_app.c`/`e2_message_handlers.c` directly, don't trust example scripts) is the reusable part.

3. **There is no literal E2SM-KPM latency field.** The closest real signal is `dl_mac_buffer_occupation`, wired to the real MAC scheduler's backlog counter — genuine but **intermittent** (in one 3000-step live campaign: 99.9% of steps had a nonzero reading for eMBB, 53.3% for URLLC, 13.5% for mMTC). Treat it with a held/staleness-tagged proxy (`qoe_mapper.py`'s `LatencyProxy`), not a naive "0 this step = confirmed zero."

4. **Every "will the ceiling ever bind" question needs a real-demand baseline first.** Early config iterations left 10–20× headroom above real observed per-UE demand (~5 PRB/UE on this rig, `avg_prbs_dl` polled directly) — every policy just accepted everything, zero differentiation, for a long time before this was caught. Always poll real demand before setting ratio caps.

5. **Synthetic admission "requests" and real UE traffic are two decoupled layers.** `accepted_counts`/block counts come from a seeded synthetic arrival stream (`_synthesize_requests`); real UE traffic only affects what the E2SM-KPM layer *reports* (PRB usage, backlog). Pushing more real packets can never change admission counts directly — only the observed state that a policy conditions its decisions on.

6. **A software UE process (`nr-uesoftmodem`, `--rfsim`) costs ~1GB resident RAM at 106 PRB/numerology 1**, allocated at startup for real-time PHY buffers (FFT tables, HARQ soft-bit buffers) — confirmed via `/proc/<pid>/status`, not a leak or a tunable log-buffer setting. Budget hardware accordingly if genuine multi-UE contention testing matters to Stage One/Two — this rig (7.4GB RAM, shared with a live desktop session) tops out around 3–4 simulated UEs before swap thrashing degrades timing enough to risk exactly the kind of instability in §3.

---

## 5. Why a fresh build, not an in-place fix

The vendored OAI checkout is **v2.1.0 (February 2024)** — confirmed via `oai_ran/CHANGELOG.md` and the UE Docker image's baked-in commit (`c599e172`, 2024-02-16) — notably the exact release that *introduced* the E2 agent this whole pipeline depends on. A web search of upstream's commit history on the crashing file (`nr_mac_common.c`) found multiple segfault-prevention fixes dated through late 2025, suggesting real upstream progress on exactly this class of bug — but confirming a specific fix applies, and that a newer checkout's E2 agent is still wire-compatible with `live_kpm_source.py`, would require either a large, uncertain-payoff cherry-pick or a full migration. Given that, doing a clean fresh build (latest Ubuntu LTS + latest stable OAI with E2 agent support) and re-verifying the E2 wire protocol from scratch is the more tractable path than patching a 2+-year-old checkout blind.

**When standing up the new build, re-run the Stage Zero precondition checks before writing any new application code**: (1) confirm E2 agent is present and reachable (check for `E2_AGENT`/`e2_agent_app.c` in the source tree, not just the CMake option name — some builds gate it differently); (2) re-verify the wire protocol against the new version's `e2_message_handlers.c` directly; (3) re-poll real per-UE demand before setting any ratio caps; (4) re-characterize which KPM fields are actually populated live (§4.3) before wiring the QoE mapper's inputs.

---

## 6. What's in the migration bundle (`oranslice_migration_bundle.zip`)

**Fully reusable as-is (pure Python, RAN-version-independent):**
- `qoe_oran_framework/` — the complete framework: `env.py`, `config.py`, `reward.py`, `qoe_mapper.py`, `action_mapping.py`, `kpm_adapter.py`, `mc_runner.py`, `omega_logger.py`, `replay_kpm_source.py`, `types.py`, `calibration/`, `policies/`, `comparators/`, `scripts/`, `tests/` (140 tests), `configs/` (as reference — ratio caps/reward weights will need re-tuning against the new rig's real demand, per §4.4)
- `drl_slicing/oranslice_drl/` — base RL classes (`RLPolicy`, `QNetwork`, `ActorCriticNetwork`, `ReplayBuffer`) that `qoe_oran_framework`'s policies import
- `drl_slicing/scripts/` — UE fleet generation (`generate_ue_fleet_compose.py`, including this session's fixes for N-per-slice and custom contention profiles), subscriber provisioning, traffic-profile orchestration — the *patterns* are reusable even where OAI-version-specific flags need updating
- `qoe_oran_framework/results/qoe_mapper/` — trained LSTM checkpoints + `fitted_iqx_coeffs.json` (calibrated against P.1203/ACR standards, not this specific hardware — a reasonable warm-start prior, should be re-validated once live KPM behavior is re-characterized on the new build)
- Small offline-trained DQN/A2C checkpoints (`checkpoint.pt` + `summary.json` only, **not** the large raw `omega_log.jsonl` traces — those are historical record, already in git, not meant for redeployment) — useful as an architecture/hyperparameter reference, will need retraining against the new rig's actual dynamics

**Needs re-verification, not blind reuse:**
- `live_kpm_source.py` — wire protocol may differ on a newer OAI E2 agent (§4.2)
- All `configs/*.yaml` ratio caps / `Lmax` / reward weights — calibrated against *this* rig's specific real demand (§4.4)

**Not included** (intentionally left behind — this is exactly what's being replaced): `oai_ran/`, `oai_cn/`, `docker_open5gs/`, `protobuf-c/`, all live `omega_log.jsonl` traces, gNB/UE `.conf` files.

**Documents included:** this file, `qoe_oran_framework/STAGE_ZERO_REPORT.md`/`.docx`.
