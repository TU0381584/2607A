# Paper #5 (WPC) / Paper #4 (CACS26) — System Specs & Parameters

Compiled 2026-09-03 as a briefing document for external review. Every
value below is read directly from a live config file, the running
host, or a committed doc — nothing is inferred or estimated unless
explicitly marked "not measured" / "unverified" (this project's own
standing discipline: never invent a number).

## 1. Hardware

| | |
|---|---|
| Machine | Single laptop, one physical rig (no separate server) |
| CPU | Intel Core i5-1135G7 @ 2.40GHz (11th Gen), 4 physical cores / 8 logical (hyperthreaded) |
| RAM | 7.4 GiB total — below this project's own documented 8GB floor recommendation, explicitly flagged and accepted at bring-up time |
| Swap | 23 GiB |
| Disk | 156GB total, 136GB free at bring-up |
| OS | Ubuntu 24.04.4 LTS (noble), kernel 7.0.0-30-generic |
| GPU | torch build includes CUDA (`2.13.0+cu130`), but live checkpoint inference runs on CPU during live eval; not required for this project's workloads |

Per-process memory cost (measured directly, not estimated): each native
`nr-uesoftmodem` UE process costs roughly ~1GB resident RAM (real-time
PHY buffer allocation, not a leak or tunable); the gNB process itself
has been observed as high as ~2.17GB RSS under sustained multi-slice
live traffic. Running the full core (17 containers) + gNB + 3 UEs
leaves the machine with well under 1GB of headroom in several
documented bring-up sessions — a genuinely tight, not comfortable,
resource envelope for this workload.

This is a genuinely resource-constrained platform for what it runs
concurrently: a full 5G core (17 Docker containers), a native gNB, up
to 6 native UE processes (all software-simulated radio, no physical
RF), and a Python RL-policy inference/control loop — all on one
8-logical-core machine with well under 8GB of RAM.

## 2. Software stack

| Component | Version / detail |
|---|---|
| Docker | 29.7.2 |
| Docker Compose | v5.5.0 |
| Python (system + venv) | 3.12.3 |
| torch | 2.13.0 |
| numpy | 2.5.1 |
| scipy | 1.18.0 |
| matplotlib | 3.11.0 |
| RAN stack | `ORANSlice` fork (wineslab/ORANSlice, GitHub), rebased onto **OAI 2024.w28** — not vanilla upstream OAI, which has no RIC-free E2 agent (only the FlexRIC E2AP/SCTP agent requiring a real near-RT RIC). Pinned commit at bring-up: `b9bcc9b` (Dec 11 2025). |
| 5G core | Open5GS via `docker_open5gs` (a fork/vendored copy of the open5gs docker packaging, `open5gs_slicing` branch, pinned commit `3f82906`, Aug 22 2024), MongoDB-backed subscriber DB |
| protobuf-c | Built from source, v1.5.2 (not the apt package — avoids a version mismatch with the E2-agent's own generated C code) |
| Project git repo | `https://github.com/TU0381584/S4_S5.git`, branch `U24PC` (renamed from `paper5`) |
| Current head at time of writing | `82dd7112bba04557aa7131ca5b1032cd1e56836b` |

`ORANSlice/` itself is a separate git clone (of `wineslab/ORANSlice`),
gitignored by the main project repo — local modifications to it (e.g.
this project's own bug fixes) are tracked as patch files under
`docs/patches/` rather than committed into that clone directly.

## 3. RAN configuration (live testbed)

All radio is **rfsim** (software-simulated radio interface) — no
physical SDR/RF hardware. "Live" throughout this project means a real
running OAI gNB + real native UE processes + a real E2 control loop
over UDP, not a discrete-event simulator.

| Parameter | Value |
|---|---|
| Band | n78 (3.5 GHz range), `-C 3619200000` center frequency |
| Bandwidth | 106 PRB |
| Numerology | 1 (30 kHz SCS) |
| Duplex | TDD, SA (standalone) mode |
| gNB config file | `ORANSlice.gnb.sa.band78.fr1.106PRB.usrpx310.conf` |
| E2 agent | RIC-free, UDP-based, `openair2/E2_AGENT/` (ORANSlice's own, not FlexRIC) — listen port 6600, gNB-side port 6655 |
| E2 control primitive | `slicing_control_m` — writes per-slice `(min_ratio, max_ratio)` PRB ratio ceilings into `gNB_MAC_INST.SL_info`, applied by `apply_slicing_ctrl()` in `openair2/E2_AGENT/e2_message_handlers.c` |
| Multi-UE isolation | Linux network namespaces + veth pairs (`ue2ns`, `ue3ns`, ...) — required because native `nr-uesoftmodem` hardcodes its TUN interface name (`oaitun_ue1`) per process, so concurrent UEs collide without netns isolation. Namespaces do **not** survive a machine reboot and must be recreated each cold boot. |
| Core network | Open5GS, Docker Compose (`docker_open5gs/5g-sa-deploy-slicing.yaml`), 17 containers (amf, ausf, bsf, mongo, nrf, nssf, pcf, scp, smf-slice{1,2,3}, udm, udr, upf-slice{1,2,3}, webui), network `demo-open5gs-public-net` at `172.22.0.0/24` |
| Subscriber provisioning | 9 pre-provisioned IMSIs (776-784 suffix range), covering up to 3 UEs/slice × 3 slices, persisted in a Docker named volume (survives `docker compose down`/`up`, does not survive a full container/volume wipe) |

## 4. Network slices (NSSAI)

Three slices, all `sst=1` (this deployment does not distinguish by SST,
only by SD):

| Slice | SD (hex) | SD (yaml, decimal) | Native traffic profile (port, `experiments/configs/traffic_profiles.yaml`) |
|---|---|---|---|
| embb | `0xFFFFFF` (16777215 — the 3GPP "no SD configured" sentinel, matched to this gNB's actual default-slice wire value) | 16777215 | port 5201, sustained UDP, 4 Mbps, 1200B packets |
| urllc | `0x000001` | 1 | port 5202, sustained UDP, 300 Kbps, 100B packets |
| mmtc | `0x000002` | 2 | port 5203, bursty UDP (2s on / 6s off), 50 Kbps, 80B packets |

embb's ceiling is the only one matched against a wildcard SD value;
urllc and mmtc use specific, non-default SD values. (This asymmetry was
investigated as a candidate explanation for a slice-specific live
instability — see the limitations file — and ultimately ruled out: all
three slices were shown to fail under sustained live control, just with
different onset timing.)

Traffic is generated via `iperf3` against a dedicated `iperf3-target`
Docker container (`networkstatic/iperf3`, one server instance per
slice port, `172.22.0.50` on the core network) — real UDP flows through
the actual UE→gNB→UPF→bridge path, not synthetic/simulated demand.

## 5. RL / policy configuration

| Parameter | Value | Source |
|---|---|---|
| State vector dimensionality | 13-dim, single-gNB live config (`[urllc,embb,mmtc] × [prb_used_ratio, congestion_level, queue_len_norm]` + `slice_onehot(3) + gnb_onehot(1)`) | `saclb_live.yaml` |
| gNB PRB capacity (`B`) | 100 | `saclb_live.yaml` |
| Backlog scale (`Lmax`) | 10 | `saclb_live.yaml`, deliberately scaled down from an original 100 to make queue violations reachable within a live episode's step budget |
| Episode length | 60 steps, `step_seconds: 5.0` nominal | `saclb_live.yaml` |
| Reward | eq. 2-style (paper #1/#2 lineage): per-slice accept/violation terms + `congestion_coeff=1.5` + load-balance term `lb_coeff=1.0` (LB term is a no-op in single-gNB live runs, `fairness_ratio` trivially 1.0) | `saclb_live.yaml`, `reward.py` |
| Reward modes | `sla` (SLA-only) and `qoe` (QoE-mapper-weighted, eq. 9-style, per-slice IQX coefficients + trained per-slice LSTM QoE mappers) | `saclb_xapp.py --reward-mode` |
| Algorithms supported | DQN, A2C, Rainbow, LB-only heuristic baseline | `qoe_oran_framework/policies/` |
| MARL extensions (offline-only so far) | GAT-CTDE (graph-attention centralized-training/decentralized-execution), independent DQN, federated GAT-CTDE with differential-privacy (DP-SGD) noise sweep (`sigma` 0.0-4.0) | `qoe_oran_framework/marl/` |

### Checkpoint lineages (do not conflate — two structurally different
seed-numbering/config lineages exist in this project)

1. **Seeds 256-261** (`dqn_sla`), single-gNB, from the original Paper #4
   / early live-anchor work. Subject of an early recalibration attempt
   (M1) that found the offline/live compliance-rank mismatch could not
   be closed within that parameter family (loss surface flat, target
   unreachable).
2. **Seeds 900+** (`single_agent_dqn`, `gat_ctde`, `independent_dqn`,
   federated variants), the M2-onward lineage used for essentially all
   of this project's more recent quantitative results, live and
   offline, including everything in this briefing's milestones file
   from M2 onward.
   - **"Original" checkpoint**: `experiments/results/m8_live_anchor/offline_train/single_agent_dqn/seed900/train/dqn/offline_train/rep_0/checkpoint.pt` — trained under the original `ClosedLoopKpmSource` offline simulator.
   - **"Recalibrated" checkpoint**: `experiments/results/m34_realistic_retrain_v2/seed900/train/dqn/offline_train/rep_0/checkpoint.pt` — retrained under `RealisticServedKpmSource` (M34), a non-frozen offline environment using served-PRB values measured live at 3-UE and 6-UE anchors (`SERVED_PRB_3UE = {urllc:5, embb:13, mmtc:5}`, `SERVED_PRB_6UE = {urllc:10, embb:45, mmtc:10}`), interpolated by a slowly mean-reverting random walk. `_v2` fixes a real implementation bug in the first version (offered demand was left fixed while only served capacity scaled).

## 6. Live evaluation tooling

| Script | Purpose |
|---|---|
| `experiments/scripts/m33_live_state_probe.py` | The live xApp reproduction used for every live pilot in this project's recent history — loads a checkpoint, drives real E2 control decisions, logs per-decision state + omega (evidence) records to JSONL |
| `experiments/scripts/restart_ran_stack.sh` | The validated "full stack restart" recipe (gNB + UE1 + UE2 + UE3) — established because hot-restarting a single UE into an already-running stack was found not to reliably recover from certain RF-simulation failures |
| `experiments/scripts/state_vector_probe.py` | Non-invasive state-vector logging wrapper, reused across offline and live congestion-characterization work |
| `m2_correctness_metrics.py` / `m4_correctness_metrics.py` / `m6_correctness_metrics.py` | Shared statistical primitives — `bootstrap_ci` (95% CI, 10,000-resample percentile bootstrap, this project's standard), `per_seed_metrics`, per-gNB/per-pending-request reward normalization |

### Correctness-aware metrics (used throughout, in preference to
`sla_compliance_all_slices` alone)

- `mean_reward_per_step` / normalized variants (per-gNB, per-pending-request)
- `block_precision` — fraction of blocks correctly targeting the lowest-priority slice (mmtc); undefined (not zero) when total_blocks=0
- Paired Wilcoxon signed-rank tests for direction + significance (p<0.05) per comparison, distinguishing a "significance flip" (strong disagreement between metrics) from "direction-only" (weak, small-n noise)

This distinction is itself a documented, quantified finding of this
project (M35): compliance disagreed with the correctness-aware metric
pair in 44/49 (89.8%) of re-analyzed comparisons across the project's
own earlier campaigns.

## 7. Repository layout

```
oranslice_rig/
├── Papers_4-5/
│   ├── Paper_4/            # CACS26 (submitted 2026-08-31)
│   └── Paper_5/
│       ├── WPC/            # active manuscript (25 pages as of this writing)
│       └── IEEE_Access/    # paused, standing instruction not to edit
├── framework/
│   ├── qoe_oran_framework/ # the frozen evaluation-critical package
│   │   ├── env.py, reward.py, action_mapping.py, mc_runner.py,
│   │   │   replay_kpm_source.py, config.py, types.py, kpm_adapter.py,
│   │   │   live_kpm_source.py, qoe_mapper.py, omega_logger.py
│   │   │   (frozen — new logic goes in new files, never edited in place)
│   │   ├── marl/           # GAT-CTDE, federated, DP, disruption (project-owned, extended freely)
│   │   ├── policies/       # DQN, A2C, Rainbow
│   │   ├── configs/        # saclb_*.yaml
│   │   └── xapp/           # saclb_xapp.py, the live control-loop entry point
│   └── drl_slicing/        # sibling package, earlier Papers #1/#2 lineage
├── experiments/
│   ├── scripts/            # ~100+ one-off and reusable milestone scripts
│   ├── configs/            # traffic_profiles.yaml etc.
│   ├── results/            # 65GB, 59 top-level campaign directories
│   ├── logs/
│   └── plots/              # figure-generation scripts
├── ORANSlice/              # separate git clone, gitignored, the actual RAN stack build
├── docker_open5gs/         # 5G core Docker Compose deployment
└── docs/                   # ~30 milestone/stage writeups (see milestones file)
```

## 8. Standing project constraints (not written in any repo file —
relayed verbally across sessions, included here for completeness)

1. Do not modify frozen framework source under `qoe_oran_framework/`
   (the list in §7 above) — new logic goes in new files or in the
   `marl/` subpackage, which is project-owned and extended freely.
2. Never reuse results from an old/different physical rig as if they
   were this rig's evidence.
3. A mandatory contention gate (`phase1_contention_gate.py`) must pass
   before any *live* training/eval run — offline-only compute is
   exempt.
4. Scope discipline across the paper series — a mechanism proposed for
   one paper doesn't get retrofitted into an earlier one's scope.
5. Never invent a number — an unmeasured value is marked
   `TODO(MEASURE)` (or, in this document, "not measured"/"unverified"),
   never guessed.
