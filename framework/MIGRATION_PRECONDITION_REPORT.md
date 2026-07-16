# Migration Precondition Verification Report

**Date:** 2026-07-14
**Scope:** Fresh-build migration away from the segfaulting OAI v2.1.0 (Feb 2024) checkout, per PROJECT_HANDOFF_SUMMARY.md §5. This report covers everything verifiable without a live gNB, plus the exact gating steps that remain for the physical rig.

---

## Executive summary

The single most consequential finding is that **"latest stable OAI with E2 agent support" is not a single target — it is a fork in the road.** Vanilla upstream OAI's only E2 agent is the FlexRIC E2AP/SCTP agent (`openair2/E2AP/`, built with `-DE2_AGENT=ON`), which requires a real near-RT RIC — precisely the architecture the old testbed investigated and ruled out (handoff §4.2). The RIC-free UDP protobuf `openair2/E2_AGENT` loop and the `slicing_control_m` control primitive that the entire migrated framework depends on come from the **ORANSlice fork (wineslab)**, which is what the old testbed actually was.

The good news: **ORANSlice `main` has been rebased onto OAI 2024.w28** (README: "The customized RAN slicing and E2 Agent code are integrated into OAI 2024.w28 release"; last commit Dec 11 2025) — roughly five months of upstream fixes past the crashing v2.1.0 base — and its E2 agent was verified in this session to be **wire-identical** to what `live_kpm_source.py` implements. The recommended target is therefore **ORANSlice `main` (OAI 2024.w28 base) on Ubuntu 24.04 LTS**, not vanilla upstream. Moving to genuinely current upstream OAI would require rewriting the transport against FlexRIC's xApp SDK and finding a replacement slicing control path — a Stage-Two-scale rework, not a migration.

The migrated framework itself is fully healthy: **140/140 tests pass**, including the six wire-protocol transport tests, which now run against a `ran_messages_pb2.py` regenerated from the *new* checkout's `ran_messages.proto` — a direct serialize/parse compatibility proof against the target tree.

---

## Precondition checklist status

| # | Precondition | Status | Where verified |
|---|---|---|---|
| 1 | New build has E2 agent support | **VERIFIED (source)** — with a critical caveat on *which* agent; see §1 below | `executables/nr-softmodem.c`, `openair2/E2_AGENT/` on ORANSlice main |
| 2 | Re-derive wire protocol from `e2_agent_app.c`/`e2_message_handlers.c` | **VERIFIED (source + protobuf round-trip)** — unchanged vs. old rig | Same files, new checkout; 6 transport tests pass against regenerated pb2 |
| 3 | Poll real per-UE demand before setting caps | **PENDING — live rig required.** Automated in new probe script (P3) | `scripts/probe_e2_preconditions.py` |
| 4 | Re-characterize which KPM fields populate live | **Schema verified; population PENDING — live rig required.** Automated (P4) | Proto field-by-field check + probe script |
| 5 | `slicing_control_m` still the control primitive | **VERIFIED (source)** — schema and MAC application path unchanged | `ran_messages.proto`, `apply_slicing_ctrl()` |

The old rig's fatal segfault gate (single-UE attach crash in `nr_dci_size`/`get_ul_tdalist`) is addressed under §6.

---

## 1. E2 agent presence (precondition 1)

On ORANSlice `main`, `openair2/E2_AGENT/` contains `e2_agent_app.c`, `e2_agent_app.h`, `e2_message_handlers.c/.h`, and `oai-oran-protolib/` — the full RIC-free UDP agent. `nr-softmodem.c` includes `E2_AGENT/e2_agent_app.h` unconditionally and calls `e2_agent_init()` unconditionally at gNB boot for any non-PNF mode, immediately after `create_gNB_tasks()`. No CMake flag, no config section — same behavior as the old rig.

The dormant same-named trap the old docstring warned about is present too: a separate FlexRIC agent gated behind `#ifdef E2_AGENT` (lines ~560 and ~727–740), activated only if compiled with `-DE2_AGENT` and given an `e2_agent = {near_ric_ip_addr, sm_dir}` conf section. Leave it uncompiled; it is irrelevant to this pipeline. If a future grep for "E2 agent" during debugging lands in `openair2/E2AP/` or FlexRIC code, that is the wrong agent.

By contrast, **upstream OAI develop has no `openair2/E2_AGENT/` at all** — only the FlexRIC E2AP/SCTP agent requiring a running RIC. This is why the ORANSlice fork, not vanilla upstream, is the migration target.

## 2. Wire protocol re-derivation (precondition 2)

Read directly from the new checkout's handlers, as the handoff prescribes, and compared against `live_kpm_source.py`'s assumptions. Every element matches:

Ports are unchanged — `e2_agent_app.h` defines `E2AGENT_IN_PORT 6655` and `E2AGENT_OUT_PORT 6600`. The dispatch in `e2_message_handlers.c` handles `INDICATION_REQUEST` (one response per request — request/response, not subscribe-and-stream), `CONTROL` (fire-and-forget, routed to `apply_slicing_ctrl()` for `SLICING_CONTROL`), and — **still — `SUBSCRIPTION` → `handle_subscription()` → `LOG_E("Not implemented") + assert(0!=0)`**. The subscription landmine is alive on the new base: sending one crashes the gNB, exactly as on the old rig. `live_kpm_source.py` never builds one; keep it that way.

Beyond source reading, the compatibility was proven mechanically: `ran_messages_pb2.py` was regenerated with protoc from the new checkout's `ran_messages.proto`, and the framework's six loopback transport tests (fake gNB peer, real UDP sockets, real serialize/parse of `INDICATION_REQUEST`/`INDICATION_RESPONSE`/`CONTROL`) pass against it. What this cannot prove is the behavior of the *running* agent — that is the probe script's P2 check on the rig.

## 3–4. Demand polling and KPM field population (preconditions 3–4)

These are live-only by nature. What was verified now: the `ue_info_m` proto message on the new base carries every field the framework reads, under identical names — `rnti`, `nssai_sST`/`nssai_sD`, `avg_prbs_dl`, `dl_mac_buffer_occupation`, `dl_total_bytes`, `dl_errors`, `dl_bler` — so `parse_indication_response()` and `kpm_adapter.py` need no changes.

What remains, automated in the new `qoe_oran_framework/scripts/probe_e2_preconditions.py`: per-slice population rates for each KPM field over N live polls (the old rig's `dl_mac_buffer_occupation` intermittency — 99.9%/53.3%/13.5% for eMBB/URLLC/mMTC — must be re-measured before the QoE mapper's `LatencyProxy` staleness handling is trusted on new hardware), and per-UE `avg_prbs_dl` demand statistics so ratio caps are set to actually bind (§4.4's hard-won lesson: caps with 10–20× headroom produce zero policy differentiation and look like implementation failure).

## 5. Control primitive (precondition 5)

`slicing_control_m{required sst; optional sd; required min_ratio; required max_ratio}` is byte-identical in the new proto (`SLICING_CONTROL = 8` in the parameter enum), and `handle_control()` routes it to `apply_slicing_ctrl()` into the live MAC slicing policy. The papers' accept/reject abstraction must therefore still be realized as ceiling nudging, with all the backlog-not-demand-removal consequences Stage Zero documented. Nothing about the Stage Zero control-surface analysis is invalidated by the new base. The `rrmPolicyJson.patch` xApp-free fallback (MAC reads `rrmPolicy.json` periodically) also still exists in `doc/` as a debugging aid.

## 6. The segfault that forced this migration

The crashing functions were located in the new base: `get_ul_tdalist` (line ~3078) and `nr_dci_size` (line ~3116) of `openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.c`. Both were **substantially rewritten between v2.1.0 and 2024.w28** — `nr_dci_size` has a changed signature (the serving-cell-info refactor) and the region now carries numerous explicit NULL guards on the UE config structures (`pusch_Config`, `dmrs_*`, `csi_MeasConfig`, `supplementaryUplink`, `pdsch_Config`) — the exact class of "reading a BWP config struct that's NULL or incomplete" the old crash was root-caused to.

This is strong but not conclusive: source inspection cannot prove the absence of the specific bug on this rig's configuration. The bring-up script therefore includes a mandatory **30+ minute single-UE attach soak** as a hard gate before any framework work — the old crash reproduced within seconds of attach, so a clean soak is a meaningful discriminator. If it crashes, the `dmesg` + `addr2line` procedure (with the PIE load-segment offset correction) from the handoff applies unchanged.

## 7. Framework test results and changes made

The migration bundle extracted cleanly. With dependencies installed (numpy, scipy, torch, pyyaml, matplotlib, protobuf, and `itu-p1203` — note: **not on PyPI**, install via `pip install git+https://github.com/itu-p1203/itu-p1203.git`), the suite ran 134 passed / 1 module-skip standalone, then **140/140** once `XAPP_OAI_PROTO_DIR` pointed at the regenerated protobuf.

Two deliberate, minimal changes were made to the bundle, both included in the delta package:

1. `tests/test_live_kpm_source.py` — the proto dir was hardcoded to the old rig's path (`/home/w5/ors/...`); it now honors `XAPP_OAI_PROTO_DIR` (same env var, same default, as `live_kpm_source.py` itself). One `os.environ.get` — no behavioral change on the old path.
2. `scripts/probe_e2_preconditions.py` — new; automates the live half of preconditions 2/3/4 (+ opt-in 5) on the rig, reusing `LiveKpmSource` unmodified.

Nothing else in the framework was touched. Carried checkpoints (DQN/A2C under both reward modes) and the QoE-mapper calibration (`fitted_iqx_coeffs.json`, LSTM weights) remain reference/warm-start material only, per the handoff — retrain/re-validate after the probe re-characterizes live KPM behavior.

## 8. Hardware note for the new rig

This sandbox VM (3.9 GB RAM, 1 core, no swap) is below the floor for hosting the live stack — the handoff's confirmed ~1 GB RSS per rfsim UE plus the gNB softmodem makes 8 GB the practical minimum, and 16 GB is what unlocks the genuine multi-UE contention testing (3–4+ UEs) that the old 7.4 GB shared desktop could not run — which was one of the two ruled-out levers for the Fig. 6 ceiling-divergence-propagation limitation. If the new physical rig's spec is a free variable, RAM is the single highest-leverage line item for Stage One's open questions.

Ubuntu 24.04 LTS is explicitly in the 2024.w28 build system's supported-distribution list (`cmake_targets/tools/build_helper`), despite ORANSlice's docs mentioning 22.04 — either works.

## 9. Recommended sequence on the rig

Encoded stage-by-stage in `rig_bringup.sh` (each stage gates the next): clone (ORANSlice main + `docker_open5gs` `open5gs_slicing` branch; record the commit hash) → build (incl. protobuf-c prerequisite and regenerating `ran_messages_pb2.py` from the checkout — do not carry the old rig's copy) → core up + per-slice subscriber provisioning → gNB → single UE → **30-min attach soak (segfault gate)** → framework tests (expect 140) → live probe (P2 round-trip, P4 field population, P3 demand; then opt-in P5 control) → re-tune ratio caps/`Lmax` from measured demand → short `lb_only` smoke → only then resume DQN/A2C retraining and QoE-mode work.
