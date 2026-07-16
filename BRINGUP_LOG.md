# ORANSlice Testbed Bring-up Log

Repo root: `~/oranslice_rig/`. Stages per `CLAUDE_CODE_BRINGUP_PROMPT.md` / `rig_bringup.sh`. Each entry: stage, timestamp, pass/fail, evidence.

---

## Stage 0 — Machine audit
**Time:** 2026-07-14 20:53 (all times local/machine clock)
**Result:** PARTIAL FAIL (flagged to user, approved to proceed with reduced scope)

- RAM: 7,765,664 kB total (**7.4 GiB**) — under the 8 GB hard floor specified for gNB + 1 rfsim UE.
  - `free -h`: 7.4Gi total, 900Mi free, 2.7Gi available, 4.0Gi swap configured (60Ki used)
- CPU: 8 cores, Intel i5-1135G7 @ 2.40GHz
- Disk: 156G filesystem on `/`, 136G free (9% used)
- OS: Ubuntu 24.04.4 LTS — matches expectation exactly
- Input files located at `~/Downloads/` (prompt said `~/uploads/`, which does not exist on this machine):
  - `oranslice_migration_bundle.zip` (+ already-extracted `oranslice_migration_bundle/`)
  - `oranslice_migration_delta.zip` (+ already-extracted `oranslice_migration_delta/`)
  - `PROJECT_HANDOFF_SUMMARY.md` present directly inside the extracted bundle (no docx-only situation)
  - `MIGRATION_PRECONDITION_REPORT.md`, `rig_bringup.sh` present directly inside the extracted delta

**Decision (user, 2026-07-14):** RAM shortfall flagged via AskUserQuestion. User chose: **proceed anyway, single-UE only** — Stage 8 multi-UE scale-out will be curtailed accordingly (this rig's 7.4 GiB matches almost exactly the old rig's "7.4GB shared desktop" constraint noted in `MIGRATION_PRECONDITION_REPORT.md` §8, which reports that config topped out around 3-4 UEs before swap thrashing risked instability — this fresh rig will be treated the same way, defaulting to 1 UE for the soak/probe stages and only cautiously trying more in Stage 8 if headroom allows).

## Stage 1 — Prerequisites
**Result:** PASS

- apt: build-essential, cmake, ninja-build, git, python3-pip/venv, protobuf-compiler, libprotoc-dev, libprotobuf-c-dev, protobuf-c-compiler, autoconf/automake/libtool, pkg-config — all installed.
- **protobuf-c built from source** (v1.5.2, via `git clone https://github.com/protobuf-c/protobuf-c && autogen.sh && configure && make && sudo make install && ldconfig`) per the ORANSlice README's documented path — not just the apt package — to avoid version mismatches with the E2 agent's generated C code. Installed to `/usr/local`.
- Docker: installed via Docker's official apt repo (docker-ce, docker-ce-cli, containerd.io, docker-buildx-plugin, docker-compose-plugin). `sudo systemctl enable --now docker` — active. User added to `docker` group (requires new login shell to take effect automatically; used `sg docker -c "..."` as a same-session workaround). Verified with `docker run hello-world`.
- sudo access: this shell had no interactive TTY for password prompts. User granted passwordless sudo via `/etc/sudoers.d/010-oranslice-bringup` (their own action, with their password) — noted here since it's a standing change to the machine outside `~/oranslice_rig/`, done at explicit user request.
- Python: dedicated venv at `~/oranslice_rig/venv` (Ubuntu 24.04 blocks system-wide pip via PEP 668). Installed numpy, scipy, torch, pyyaml, pytest, matplotlib, protobuf. `itu-p1203` pending (git-based install, not on PyPI).

## Stage 2 — Clone + build RAN
**Result:** PASS

- `git clone https://github.com/wineslab/ORANSlice.git` → branch `main`, commit **`b9bcc9b17fbecfc1041072a7b8d0f01ae874aba2`** (Thu Dec 11 2025), matches `MIGRATION_PRECONDITION_REPORT.md`'s "last commit Dec 11 2025" claim.
- `git clone -b open5gs_slicing https://github.com/wineslab/docker_open5gs.git` → commit `3f829063e60fc573e65c9f27977b73c5057aa9d8` (Thu Aug 22 2024).
- `./build_oai -I` (sudo, one-time dependency install incl. ASN1C, SIMDE headers, libsctp-dev, libssl-dev, etc.) — completed, "BUILD SHOULD BE SUCCESSFUL".
- `./build_oai --ninja --gNB --nrUE` (no `-w USRP`, per handoff — rfsim needs none) — **caveat**: the script's own "BUILD SHOULD BE SUCCESSFUL" message is unreliable on this run; the sudo `-I` step had left `cmake_targets/log/` owned by root, so this non-sudo build's log-file write (and the script's own grep-for-errors check against that log) failed with "Permission denied" and silently fell through to the success message without actually verifying. Caught this because the top-level `nr-softmodem`/`nr-uesoftmodem` binaries were initially missing from `ran_build/build/`. Re-ran `ninja nr-softmodem nr-uesoftmodem` directly in that build dir — it resumed and *actually* completed real compilation/linking (final steps: linking `libL2_LTE_NR.a`, then `nr-uesoftmodem`, then `nr-softmodem`). Confirmed both binaries now exist, are PIE ELF executables with debug info (`not stripped` — useful for Stage 6 addr2line if needed), owned by `kmanojp`, sizes 165MB (`nr-softmodem`) / 55MB (`nr-uesoftmodem`). Fixed the ownership issue going forward: `sudo chown -R $USER:$USER cmake_targets/log/`.
  - **Lesson for future runs of this script**: don't trust `build_oai`'s printed "BUILD SHOULD BE SUCCESSFUL" at face value if any step before it touched `cmake_targets/log/` as a different user (e.g. via `sudo`) — verify the actual target binaries exist.
- Regenerated Python protobuf: `cd oai_ran/openair2/E2_AGENT/oai-oran-protolib && mkdir -p builds && protoc --python_out=builds ran_messages.proto` → `builds/ran_messages_pb2.py` produced. **Diffed byte-identical against the delta's reference copy** (`oranslice_migration_delta/oai-oran-protolib-builds/ran_messages_pb2.py`) — confirms the precondition-verification session's regenerated copy was indeed built from this exact checkout, consistent with `MIGRATION_PRECONDITION_REPORT.md` §2's claims.
- Created `~/oranslice_rig/env.sh` exporting `XAPP_OAI_PROTO_DIR` (pointing at the `builds/` dir above), `ORANSLICE_HOME`, `FRAMEWORK_DIR`.

## Stage 3 — Extract framework + offline tests
**Result:** PASS — exactly 140 passed

- Unzipped `oranslice_migration_bundle.zip` into `~/oranslice_rig/framework/`.
- Unzipped delta on top: overlaid `qoe_oran_framework/scripts/probe_e2_preconditions.py` (new) and `qoe_oran_framework/tests/test_live_kpm_source.py` (overwritten) — confirmed these were the only two paths touched, matching `MIGRATION_PRECONDITION_REPORT.md` §7's stated diff.
- Copied `rig_bringup.sh`, `MIGRATION_PRECONDITION_REPORT.md`, `PROJECT_HANDOFF_SUMMARY.md` into the framework root, per instructions. (Also kept copies at `~/oranslice_rig/` root, read during preliminary review.)
- Ignored delta's `oai-oran-protolib-builds/` except as the diff-reference used above.
- `itu-p1203` installed via `pip install git+https://github.com/itu-p1203/itu-p1203.git` (not on PyPI, per instructions) into `~/oranslice_rig/venv`, along with numpy 2.5.1, scipy 1.18.0, torch 2.13.0+cu130, protobuf 7.35.1, pyyaml, pytest, matplotlib.
- With `XAPP_OAI_PROTO_DIR` (from `env.sh`) set and the venv active: `python3 -m pytest qoe_oran_framework/tests/ -q` → **`140 passed in 18.72s`** — exact pass condition met, no skips.

---

## Stage 4 — Core up
**Status:** IN PROGRESS

**Finding worth flagging now (not papering over):** `docker_open5gs`'s `open5gs_slicing` branch and this exact ORANSlice `main` checkout both ship **two** S-NSSAIs out of the box, not three (eMBB/URLLC/mMTC) as `PROJECT_HANDOFF_SUMMARY.md` describes for the old rig's paper #2 3-slice work:
- `amf/amf.yaml` (docker_open5gs): `plmn_support.s_nssai` = `[{sst:1}, {sst:1, sd:000002}]`
- gNB conf `ORANSlice.gnb.sa.band78.fr1.106PRB.usrpx310.conf`: `snssaiList = ({sst=1,sd=0xFFFFFF}, {sst=1,sd=0x000002})`
- Only `nrUE_slice1.conf` (imsi `...776`, sst=1, no sd → default/wildcard SD, dnn=`oai`) and `nrUE_slice2.conf` (imsi `...777`, sst=1, sd=0x2, dnn=`oai2`) exist in this checkout — **no `nrUE_slice3.conf`**.

This is the RAN/core pairing as actually shipped by wineslab's fork, verified by reading the files rather than assumed. Per the hard rule not to modify frozen framework source and per Stage 4's own instruction to "verify, don't assume" against what the compose actually creates: proceeding with **2 slices** for the live bring-up (Stages 4-9), not fabricating a 3rd. This will be called out again in Stage 10 as a deviation from the handoff's premise. It does not block anything structurally — the framework's control primitive (`slicing_control_m`) and env/adapter code are slice-count-agnostic; `configs/*.yaml` reference 3 slices, which will need trimming to 2 in Stage 9, along with the ratio-cap retuning already planned there.

- `.env`: left MCC/MNC (001/01), bridge subnet (172.22.0.0/24) as shipped — verified no conflict with existing docker0 (172.17.0.0/16) or host LAN (192.168.50.0/24). `DOCKER_HOST_IP` confirmed unused by any active yaml/script in this deploy path (grep across `*.yaml`/`*.sh` — no hits), so left at its stale example value harmlessly.
- Building `docker_open5gs_slicing` image from `base/Dockerfile_Slicing` (compiles Open5GS v2.7.2 from source) — succeeded, `docker_open5gs_slicing:latest`, 1.24GB.
- `docker compose -f 5g-sa-deploy-slicing.yaml up -d` — all 15 containers (mongo, webui, nrf, scp, ausf, udr, nssf, bsf, udm, pcf, amf, smf-slice1, smf-slice2, upf-slice1, upf-slice2) created and Up.
- AMF log confirms: NGAP server listening `172.22.0.10:38412` (SCTP), registered with NRF, SMF-slice1/SMF-slice2/PCF registering — core is live.
- Subscriber provisioning: the framework's `drl_slicing/scripts/provision_open5gs_subscribers.sh` pattern hardcodes APN=`internet` in its `add_ue_with_slice` call, but this rig's UE confs use DNN `oai` (slice1) / `oai2` (slice2) — a real mismatch, not something to force. Per Stage 4's "verify, don't assume" instruction, provisioned directly via `open5gs-dbctl add_ue_with_slice` inside the `amf` container instead of running the script as-is (script left unmodified — it's not part of the frozen `qoe_oran_framework/`, but editing it wasn't necessary once doing it directly):
  - IMSI `001010000010776` (matches `nrUE_slice1.conf`): ki/opc from that conf, apn=`oai`, sst=1, **sd=`ffffff`** (3GPP's "no SD" sentinel — matches gNB conf's `sd=0xFFFFFF` for this slice and the AMF's `s_nssai` entry that omits `sd` entirely).
  - IMSI `001010000010777` (matches `nrUE_slice2.conf`): same ki/opc, apn=`oai2`, sst=1, sd=`000002`.
  - Needed a one-time `mongosh` shim inside the `amf` container (only legacy `mongo` client present, `open5gs-dbctl` invokes `mongosh`) — same workaround the provisioning script itself would have applied.
  - Verified via `open5gs-dbctl showfiltered`: both IMSIs present with correct APN/key/opc.
- **Pass: AMF running and listening; subscribers visible in DB.** ✓

---

## Stage 5 — gNB + one UE, stable attach
**Status:** IN PROGRESS

- Fixed `ORANSlice.gnb.sa.band78.fr1.106PRB.usrpx310.conf` against the actual deployed core (original had stale example values, not this rig's): `amf_ip_address` `192.168.70.132` → `172.22.0.10` (AMF_IP); `GNB_INTERFACE_NAME_FOR_NG_AMF`/`NGU` `demo-oai` → `demo-open5gs` (the actual host-side bridge interface name docker compose created — confirmed via `ip addr`/`docker network inspect`, network `demo-open5gs-public-net`, subnet 172.22.0.0/24); `GNB_IPV4_ADDRESS_FOR_NG_AMF`/`NGU` `192.168.70.129/24` → `172.22.0.1/24` (host's real address on that bridge). PLMN (001/01) and `snssaiList` already matched the core (§ Stage 4 finding) — no change needed there. Original backed up as `.orig`.
- **Second real gap found in the Stage 2 build, only visible at runtime**: first gNB launch attempt failed immediately — `dlopen(libparams_libconfig.so)` not found, config module couldn't load. Root cause: `build_oai --ninja --gNB --nrUE`'s original invocation (Stage 2) requested targets `nr-softmodem nr-cuup nr-uesoftmodem params_libconfig coding rfsimulator dfts`, but that run was the one whose log-write hit "Permission denied" — evidently it aborted before ninja did real work. My Stage-2 recovery (`ninja nr-softmodem nr-uesoftmodem`) only pulled in *link-time* dependencies of those two binaries (which is why `libldpc*.so`/`libdfts.so` exist), not `params_libconfig`/`coding`/`rfsimulator` — these are runtime `dlopen()`-loaded plugins, not link-time dependencies, so ninja never built them as a side effect. Building them explicitly now (`ninja nr-cuup params_libconfig coding rfsimulator`).
  - **Lesson reinforcing Stage 2's**: verifying "the two top-level binaries exist" was not sufficient proof the build was complete — this class of build (dlopen plugins) needs its full target list actually verified, not inferred from the executables linking successfully.
- gNB launched in tmux session `gnb`: `sudo ./nr-softmodem -O <fixed conf> --sa --rfsim`, log at `~/oranslice_rig/logs/gnb.log`. Came up clean: RU/L1 threads created, rfsimulator listening as server, **E2 agent initialized and sending heartbeats** (`[E2_AGENT] E2 agent heartbeat`, 31 logged over the launch window). NGAP: `Registered new gNB[0]`, `NGAP_REGISTER_GNB_CNF: associated AMF 1` — confirmed on the AMF side too (`docker logs amf`: `gNB-N2 accepted[172.22.0.1]`, `[Added] Number of gNBs is now 1`).
- UE launched in tmux session `ue` (`nrUE_slice1.conf`, matches provisioned IMSI `...776` exactly — no edits needed): `sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --sa -O <conf> --rfsim --rfsimulator.serveraddr 127.0.0.1`, log at `~/oranslice_rig/logs/ue.log`. RSRP steady at -42 dBm (strong rfsim signal). NAS: `REGISTRATION ACCEPT` → `RegistrationComplete` → `PduSessionEstablishRequest` → `NAS_CONN_ESTABLI_CNF` → `Interface oaitun_ue1 successfully configured, ip address 192.168.100.2`. (Benign warning logged: NAS IEIs `0x21`/`0x5e` "not handled when extracting list of allowed NSSAI" — didn't block attach; noting for Stage 10 in case it matters later.)
- Connectivity: `ping -I oaitun_ue1 8.8.8.8` → **5/5 received, 0% loss, ~20ms avg RTT**, through the real PDU session and UPF (`upf-slice1`'s `ogstun` interface confirmed at `192.168.100.1/24`, matching the UE's assigned `192.168.100.2/24`).
- **Pass: UE got a PDU session IP and can ping the internet through it; gNB log shows the E2 agent alive.** ✓

---

## Stage 6 — SEGFAULT SOAK GATE
**Status:** IN PROGRESS — restarted 2026-07-14 21:58:02 local (first attempt invalid, see below)

- Light continuous traffic: `ping -I oaitun_ue1 -i 1 8.8.8.8` (1/sec) through the live PDU session, backgrounded, logging to `~/oranslice_rig/logs/soak_ping.log`.
- **Caught a false-pass on the first attempt, worth recording**: the first soak watcher used plain `dmesg -w` (no sudo) piped through `timeout 1900`. This shell has `kernel.dmesg_restrict` enabled, so unprivileged `dmesg -w` fails immediately with "Operation not permitted" — the pipeline exited in under a second, and the watcher's unconditional trailing `echo` printed a clean-looking "ended" signal anyway, which would have read as "30 minutes, zero crashes" after only ~39 seconds. Caught by checking actual elapsed wall-clock time against the epoch timestamp recorded at soak start, rather than trusting the watcher's own "done" signal. This is exactly the "silence is not success" failure mode — a filter that's silent when the thing it's watching is broken looks identical to a filter that's silent because nothing bad happened.
  - Fix: `sudo dmesg -w` (passwordless sudo already in place from Stage 1). Re-armed the soak with an added independent liveness check — a 5-minute heartbeat loop that (a) confirms the `dmesg` watcher subprocess is still alive, (b) confirms both `nr-softmodem` and `nr-uesoftmodem` processes are still running, and only then prints `SOAK_ALIVE`; any failure of any of those three conditions prints an `ALERT` and exits immediately rather than staying silent.
- gNB (tmux `gnb`) and UE (tmux `ue`) left running untouched throughout, per the hard rule on long-running processes.
- Will record: zero-crash pass (all `SOAK_ALIVE` heartbeats + final `SOAK_COMPLETE_30MIN_NO_CRASH`), or (if it crashes) dmesg capture + `addr2line` against the faulting address corrected for the PIE load-segment offset from `/proc/<pid>/maps`, then STOP per instructions — no source patches attempted.

**Result: PASS.** `SOAK_ALIVE` heartbeats confirmed healthy at 300s/600s/900s/1200s/1500s/1800s (every 5 min through the full 30-min mark) — both processes running, dmesg watcher healthy, every time. The monitor's own hard timeout (2100s) cut it off mid-way through its final sleep cycle before it could print the scripted `SOAK_COMPLETE_30MIN_NO_CRASH` line, so verified directly instead of trusting an absence of a final message:
- Elapsed wall-clock at verification: **35 min 35 sec** since soak start (exceeds the 30-min floor).
- `pgrep` confirms both `nr-softmodem` and `nr-uesoftmodem` (and their tmux/bash wrapper processes) still running, same PIDs as at launch (146585 / 147006 respectively) — no restart, no crash-and-respawn.
- `sudo dmesg | grep -iE "segfault|nr-softmodem|nr-uesoftmodem|general protection|killed process"` → **zero matches**.
- Ping log: 2170+ ICMP sequences sent over the window; a handful of individual sequence numbers missing from the tail (e.g. 2182, 2186) — ordinary internet jitter/loss on the public 8.8.8.8 path, not a RAN-process failure (traffic resumed normally on the next sequence each time; this is not the same failure class as a gNB/UE segfault, which would have shown up in dmesg and killed the tunnel outright).
- gNB log: **773 `E2_AGENT] E2 agent heartbeat` lines** logged over the soak, steadily incrementing throughout — the E2 loop never stalled or died.
- **The old rig's crash (`nr_dci_size`/`get_ul_tdalist` NULL derefs) did not reproduce.** Consistent with `MIGRATION_PRECONDITION_REPORT.md` §6's finding that both functions were substantially rewritten with NULL guards between v2.1.0 and this checkout's 2024.w28 base.

---

## Stage 7 — Live E2 precondition probe
**Result: PASS** (P2 and P5 succeeded; P3/P4 recorded)

Ran with gNB + UE (slice1) up and the soak's ping traffic still flowing.

**P2 (wire protocol round-trip):** `python3 -m qoe_oran_framework.scripts.probe_e2_preconditions --polls 120` → `[P2] PASS: got INDICATION_RESPONSE with 1 UE sample(s)`.

**P4 (per-slice KPM field population, 120/120 successful polls, single attached UE on slice1 = sst1/sd16777215):**

| Field | Population rate |
|---|---|
| `avg_prbs_dl` | 100.0% |
| `dl_mac_buffer_occupation` | **0.0%** |
| `dl_total_bytes` | 100.0% |
| `dl_errors` | 0.0% |
| `dl_bler` | 100.0% |

Comparison to the old rig's reference numbers (eMBB/URLLC/mMTC: 99.9%/53.3%/13.5% for `dl_mac_buffer_occupation`): **this rig's slice1 shows 0.0%**, not intermittent-but-present — a real, honest measurement, not a fudge. Read in context: this probe ran under `ping -i 1` traffic (a few small packets/sec), nowhere near enough to build a nonzero MAC downlink backlog on a 106-PRB cell — the scheduler drains that little traffic within the same TTI it arrives, so the backlog counter has nothing to report. The old rig's 99.9%/53.3%/13.5% figures were presumably measured under materially heavier live traffic (its own report doesn't state the load, only the sample count). **This means `dl_mac_buffer_occupation`'s intermittency characterization from the old rig cannot be assumed to transfer as-is** — it is a function of offered load, not just the RAN build, and needs re-measurement under realistic traffic (Stage 8/9's heavier profiles) before the QoE mapper's `LatencyProxy` staleness handling is trusted here. `dl_errors`/`dl_bler` both report (bler always populated, errors never nonzero — clean rfsim channel, no errors expected).

**P3 (real per-UE demand):** `rnti=32988: mean=5.17 PRB, max=12.00 PRB, n=120` under light ping traffic. **Matches the handoff's own reference point** ("~5 PRB/UE on this rig" from the old rig's finding #4) closely — a good sanity check that the measurement methodology transferred correctly even though the absolute rig differs.

**P5 (opt-in control check, values from the gNB's actual configured floor/ceiling only):** The gNB conf's `SliceConf` points at a stale, nonexistent `rrmPolicy.json` path (`/home/wineslab/ORANSlice/rrmPolicy.json`, an old-rig artifact) — but the periodic policy-reload code path that would read it (`nr_update_slice_policy()`) is commented out by default in this checkout (only enabled by applying `doc/rrmPolicyJson.patch`, which was **not** applied — not part of any stage instruction). Read `gnb_config.c`'s slice-init code directly (lines ~1315-1330) to find the actual live values: every slice initializes to **`min_ratio=0, max_ratio=100`** in code, regardless of the dead `SliceConf` path. Used those as "the gNB conf's configured floor/ceiling":
```
python3 -m qoe_oran_framework.scripts.probe_e2_preconditions --polls 5 \
  --send-control --sst 1 --sd 16777215 --min-ratio 0 --max-ratio 100
```
gNB log confirms: `[E2_AGENT] Control message received` / `Slicing_Ctrl_Msg Applied: NSSAI.SST 1, NSSAI.SD 16777215, min_ratio 0, max_ratio 100` — exact match, P5 **PASS**.

---

## Stage 8 — Multi-UE to capacity
**Result: CURTAILED AT 1 UE — by user decision (Stage 0), confirmed correct by live measurement, not just deferred to policy.**

Before adding a second UE, checked actual headroom with the real workload running (core + gNB + 1 UE + soak traffic), rather than assuming the Stage 0 estimate still held:
- `free -h`: **7.4Gi total, 490Mi free, 3.0Gi "available" (reclaimable cache), 1.0Gi of 4.0Gi swap already in use.**
- Real (non-wrapper) process RSS: `nr-softmodem` main process **1,369,124 KB ≈ 1.34 GB**; `nr-uesoftmodem` main process **441,044 KB ≈ 431 MB**. (Notably the UE RSS here is lower than the handoff's ~1GB/UE figure — possibly config-dependent — but the gNB alone is heavier than expected and dominates.)
- Docker core containers: modest, ~450MB combined across all 15 containers.

**With only the single required UE attached, this rig is already swapping.** That's concrete evidence, not just caution: adding a second UE (whichever slice) would push further into swap on a machine already below the Stage 0 floor, directly risking the swap-thrashing-induced instability class `PROJECT_HANDOFF_SUMMARY.md` finding #6 explicitly warns about — the same class of instability Stage 6 spent 35+ minutes proving does *not* currently happen. Scaling up here would trade a proven-stable state for an unproven, evidence-contraindicated one.

Per the user's explicit Stage 0 decision ("proceed anyway, single UE only... skip/limit Stage 8"), and now confirmed by this rig's own telemetry rather than assumed from the old rig's numbers: **not launching additional UEs.** The single-UE P3/P4 demand numbers from Stage 7 stand as this rig's measured demand baseline for Stage 9's cap calibration — there will be no "updated P3/P4 tables per slice" beyond what Stage 7 already recorded, since no second UE was added.

**Deviation from the handoff's premise, worth flagging for Stage 10:** the old rig's `PROJECT_HANDOFF_SUMMARY.md` describes hitting its UE-count ceiling around 3-4 UEs on "this shared desktop rig (~1GB RSS per simulated UE ... tied to OAI's real-time PHY buffer allocation)." This fresh rig hits swap with a single UE — either this desktop session carries a heavier baseline load than the old rig's, or the gNB's own footprint (1.34GB, not accounted for in the old rig's "~1GB/UE" framing, which was about UE RSS specifically) was underweighted in the original headroom math. Both are worth noting as real, rig-specific findings rather than assuming the old rig's capacity numbers transfer.

---

## Stage 9 — Calibrate caps, then lb_only smoke
**Status: caps DONE (authorized config-YAML edit); smoke test BLOCKED — stopping to report, not guessing around it**

### Cap calibration (config-YAML edit, within the explicitly authorized scope)
Edited `qoe_oran_framework/configs/saclb_live.yaml` only (no other framework source touched):
1. **Dropped the `urllc` slice entry (sd=1) entirely.** No MAC slice with `(sst=1, sd=1)` exists on this gNB — its 3 configured slices are `id0 sst=0/sd=0` (reserved SRB-only default), `id1 sst=1/sd=0xFFFFFF`, `id2 sst=1/sd=2` (confirmed via the gNB's own boot log: `+++++++ Configured slices at MAC +++++++`). `apply_slicing_ctrl()` (`e2_message_handlers.c:183`) silently no-ops when no slice matches — verified by reading the function directly rather than assumed — so a phantom `urllc` entry could never "bind" on real hardware; keeping it would have corrupted the smoke test's signal, not just been inert.
2. **Fixed `embb`'s `sd: 0` → `sd: 16777215` (0xFFFFFF).** This is the actual wire value for "no SD configured" on this checkout (3GPP's SD-absent sentinel) — confirmed both by the gNB's MAC slice-init log and by Stage 7's live probe output (`sst1/sd16777215`). This was **load-bearing, not cosmetic**: `env.py:183` sends `slicing_control_m` using the YAML's exact `(sst, sd)` pair every step — with the old value, every "embb" ceiling command would have silently targeted a nonexistent slice.
3. **`max_ratio_cap` values**: left `embb` at `4` — re-verified as still valid against this rig's own Stage 7 measurement (mean 5.17 PRB, max 12.00 PRB), not just inherited from the old rig's coincidentally-similar ~5.0 PRB/UE reference. Left `mmtc` at `3` but **flagged explicitly in the YAML's own comment as unverified on this rig** (no UE was ever attached to slice2 — Stage 8 curtailed) rather than silently presenting an inherited number as freshly measured.
4. Verified the edited config still loads cleanly: `load_saclb_config()` → `embb sst=1 sd=16777215 floor=1 cap=4`, `mmtc sst=1 sd=2 floor=1 cap=3`.

Old → new, with justification:

| Slice | Field | Old | New | Why |
|---|---|---|---|---|
| urllc | (entire entry) | present, sd=1 | **removed** | No matching MAC slice on this gNB; CONTROL would permanently no-op |
| embb | sd | 0 | **16777215** | Wrong wire value; real "no-SD" sentinel is 0xFFFFFF, not 0 |
| embb | max_ratio_cap | 4 | 4 (unchanged, re-verified) | Still below this rig's measured mean 5.17 / max 12.00 PRB |
| mmtc | sd | 2 | 2 (unchanged, correct) | Already matched real slice2 |
| mmtc | max_ratio_cap | 3 | 3 (unchanged, **unverified**) | No live measurement available on this rig (Stage 8 curtailed) |

### Smoke test: blocked, stopping to report per the hard rules
Per the stage instruction, read `run_saclb_live_testbed.sh`'s flags before running it (including a `--dry-run` to see its exact command sequence without executing). Found **structural mismatches between what this frozen script assumes and the rig as built in Stages 1-8**, none fixable within the "config YAMLs only" edit authorization:

1. **Path layout mismatch.** The script resolves `ROOT_DIR` as its own grandparent directory and expects `$ROOT_DIR/oai_ran/` and `$ROOT_DIR/docker_open5gs/` as siblings under `framework/`. This rig — built exactly per `rig_bringup.sh` and every prior stage — has them at `~/oranslice_rig/ORANSlice/oai_ran/` and `~/oranslice_rig/docker_open5gs/`, siblings under `~/oranslice_rig/` (one level up, and `oai_ran` nested inside `ORANSlice/`), not under `framework/`. Confirmed via the dry-run's printed paths (e.g. `.../framework/oai_ran/cmake_targets/...`, which doesn't exist).
2. **Different, colliding core deployment.** The script always runs `docker compose -f sa-deploy.yaml up -d` — a *different* compose file from the `5g-sa-deploy-slicing.yaml` built and verified in Stage 4. `sa-deploy.yaml` uses the plain `docker_open5gs` image (from `base/Dockerfile`, not `Dockerfile_Slicing`) but the **same container names** (`mongo`, `webui`, `amf`, etc.) as my already-running, soak-verified stack — running it would either collide with the running containers or require tearing down Stage 4-7's verified-good deployment for a differently-built one, not something to do unilaterally.
3. **Own gNB + UE fleet lifecycle.** The script launches its own `nr-softmodem` via `nohup` and a 3-UE containerized fleet (`generate_ue_fleet_compose.py --ue-count 3 ... --oai-ue-image docker_oai_nrue:local --oai-launch-mode local-build`) — conflicting with the already-running, 35+-minute-soak-verified gNB/UE in tmux, and directly re-opening the RAM-safety question Stage 8 just closed (3 more UEs on a machine already swapping with one). The referenced `docker_oai_nrue:local` image was also never built in any stage.

None of this is a "config YAML" problem — it's the script's own hardcoded paths and orchestration logic (`qoe_oran_framework/scripts/run_saclb_live_testbed.sh`, frozen source per the hard rules). I did not edit it and did not tear down the verified-stable stack to force-fit its assumptions. Reported this to the user rather than guessing a workaround, per "If something appears broken, report it; don't patch it" and "ask before ... any deviation from the stage order."

**User decision:** invoke the xApp entrypoint (`qoe_oran_framework/xapp/saclb_xapp.py`) directly against the already-running, verified stack, skipping the wrapper script's redundant/mismatched bring-up steps 1-7 entirely (core, gNB, and UE from Stages 4-7 are already up and soak-verified) — this achieves the stage's actual intent (a live lb_only run producing omega logs against the real rig) without editing frozen source or tearing down anything.

### Smoke run (direct xApp invocation)
```
PYTHONPATH="$PWD" python3 qoe_oran_framework/xapp/saclb_xapp.py \
  --config qoe_oran_framework/configs/saclb_live.yaml --algorithm lb_only --gnb-id gnb-0 \
  --episodes 1 --seed 256 --run-id stage9-smoke-001 \
  --omega-jsonl qoe_oran_framework/results/live/lb_only/stage9-smoke-001_omega_log.jsonl --reward-mode sla
```
Default ports (`--xapp-listen-port 6600`, `--gnb-listen-port 6655`) already match the wire protocol verified in Stages 5-7. 1 episode = 60 steps × 5s/step ≈ 5 minutes — the shortest sensible campaign per the stage's own instruction. Completed cleanly, exit code 0.

**Result: PASS, with an honest, explainable caveat — not a clean unconditional pass.**
- Omega log written: `qoe_oran_framework/results/live/lb_only/stage9-smoke-001_omega_log.jsonl`, 60 per-step records + 1 episode-rollup summary, every record's `limitation` field non-empty by construction (per the framework's own logging contract).
- **`embb` (the slice with the real attached UE): `max_ratio` stayed pinned at its configured floor (1) for all 60/60 steps, zero accepts the entire episode — exactly matches lb_only's documented signature** ("never leaves the configured floor for any slice, the entire episode" — `PROJECT_HANDOFF_SUMMARY.md` §3 Fig. 5).
- **`mmtc` did NOT stay at floor**: `max_ratio` rose from 1 → 2 → 3 (its cap) over the episode, because it accepted 51/60 synthetic admission requests, and `AdmissionGate`'s ceiling rises `+ceiling_step_ratio` per accept. Root cause, verified by reading the actual mechanism rather than assumed: `LbOnlyHeuristic` only rejects (and thus pins the ceiling at floor) when it reads real saturation/backlog for a slice. `embb` has that — a real UE has been continuously attached and generating traffic since Stage 5. `mmtc` does not — **no UE was ever attached to slice2 on this rig**, a direct, traced consequence of Stage 8's RAM-driven curtailment to 1 UE, not a new problem or a framework defect. With zero real backlog ever observed on `mmtc`, the heuristic has no basis to ever reject a synthetic `mmtc` request, so its ceiling behaves exactly like an *unconstrained* slice would — which is the logically correct behavior given this rig's actual topology, just not the specific "stays at floor for every slice" signature the stage instruction described (that signature implicitly assumes every configured slice has real contention, which requires the multi-UE scale-out Stage 8 couldn't safely do here).
- eMBB's live diagnostics are also coherent with everything measured earlier: `sla_viol=0.9833` / `per_slice_compliant.embb=false` almost every step — consistent with `latency_budget_ms=45`/`loss_budget_pct=2.0` being tight relative to a slice pinned at `max_ratio=1` (1% of PRB budget) the entire time, i.e. the ceiling calibration from earlier in this stage is visibly doing its job (creating real, binding scarcity), not sitting inert.

---

## Stage 10 — Final report

### Commit hashes
- **ORANSlice**: `b9bcc9b17fbecfc1041072a7b8d0f01ae874aba2`, branch `main` (Thu Dec 11 2025) — matches `MIGRATION_PRECONDITION_REPORT.md`'s "last commit Dec 11 2025" claim exactly.
- **docker_open5gs**: `3f829063e60fc573e65c9f27977b73c5057aa9d8`, branch `open5gs_slicing` (Thu Aug 22 2024).

### Machine specs
Ubuntu 24.04.4 LTS, 8-core i5-1135G7 @ 2.40GHz, **7.4 GiB RAM** (below the 8 GB floor the handoff specified — flagged to the user at Stage 0, who approved proceeding single-UE-only), 4.0 GiB swap, 156 GB disk (136 GB free, ~6.9 GB used by the full rig at close).

### Stage-by-stage pass/fail
| Stage | Result |
|---|---|
| 0 — Machine audit | **PARTIAL** — RAM 7.4 GiB < 8 GiB floor; user approved proceeding single-UE-only |
| 1 — Prerequisites | PASS |
| 2 — Clone + build RAN | PASS (after recovering from an incomplete first build — see Stage 2/5 notes) |
| 3 — Framework + offline tests | **PASS — 140/140** |
| 4 — Core up | PASS (2 real slices, not 3 — see deviations below) |
| 5 — gNB + 1 UE stable attach | PASS — PDU session IP acquired, pinged 8.8.8.8 0% loss, E2 agent alive |
| 6 — Segfault soak gate | **PASS — 35+ min, zero crashes** (old rig's `nr_dci_size`/`get_ul_tdalist` bug did not reproduce) |
| 7 — Live E2 precondition probe | **PASS** — P2 and P5 succeeded; P3/P4 recorded (see table below) |
| 8 — Multi-UE to capacity | **Curtailed at 1 UE** — evidence-backed (already swapping with 1 UE), not merely policy-deferred |
| 9 — Calibrate caps + lb_only smoke | **PASS with caveat** — caps recalibrated and a real sd-routing bug fixed; smoke run completed, omega log written, floor-pinning signature confirmed for the slice with real traffic |

### P3/P4 tables (Stage 7, single UE on the `embb` slice, light ping traffic)

| KPM field | Population rate |
|---|---|
| `avg_prbs_dl` | 100.0% |
| `dl_mac_buffer_occupation` | 0.0% |
| `dl_total_bytes` | 100.0% |
| `dl_errors` | 0.0% |
| `dl_bler` | 100.0% |

Per-UE demand: `rnti=32988: mean=5.17 PRB, max=12.00 PRB, n=120`. No live measurement exists for `mmtc` (slice2) — Stage 8 never attached a UE to it.

### Cap changes (Stage 9)
| Slice | Field | Old | New |
|---|---|---|---|
| urllc | entry | present (sd=1) | **removed** — no matching real MAC slice |
| embb | sd | 0 | **16777215** (0xFFFFFF) — was silently routing CONTROL to a nonexistent slice |
| embb | max_ratio_cap | 4 | 4 (re-verified against this rig's own measurement) |
| mmtc | sd | 2 | 2 (unchanged, already correct) |
| mmtc | max_ratio_cap | 3 | 3 (unchanged, **explicitly flagged unverified** — no live measurement available) |

### Everything that differed from `PROJECT_HANDOFF_SUMMARY.md` / `MIGRATION_PRECONDITION_REPORT.md`'s expectations
1. **Input files were at `~/Downloads/`, not `~/uploads/`** as the bring-up prompt stated (Stage 0).
2. **RAM is 7.4 GiB, under the stated 8 GB floor** — this fresh rig turned out to share almost exactly the old rig's constraint, not comfortably clear it (Stage 0/8).
3. **This ORANSlice+docker_open5gs pairing ships 2 real S-NSSAIs, not 3** (eMBB/URLLC/mMTC per the old rig's paper #2 work) — verified by reading the AMF config, gNB config, and the checkout's own `nrUE_slice*.conf` files directly, not assumed (Stage 4). This propagated forward: Stage 9's config had to drop `urllc` entirely and fix a real sd-routing bug in `embb`.
4. **`build_oai`'s "BUILD SHOULD BE SUCCESSFUL" message was unreliable** after a `sudo`-owned `log/` directory caused a silent permission failure partway through — verifying the actual target binaries (and, later, the dlopen-loaded runtime plugins specifically) was necessary; the printed message alone was not proof (Stage 2/5).
5. **The gNB conf's AMF address and NG interface names were stale placeholders** (`192.168.70.132`, `demo-oai`) that needed correcting to this rig's actual deployed core and bridge interface (`172.22.0.10`, `demo-open5gs`) before the gNB could associate (Stage 5).
6. **The gNB alone uses ~1.34 GB RSS**, not accounted for in the handoff's "~1GB RSS per UE" framing — combined with real UE RSS (~431 MB here, lower than the old rig's ~1GB/UE figure), this rig hits swap with a single required UE, forcing Stage 8's curtailment (Stage 8).
7. **`dl_mac_buffer_occupation` measured 0.0% here** vs. the old rig's 99.9%/53.3%/13.5% reference — plausibly a function of offered traffic load (this probe ran under light `ping -i 1` only) rather than a RAN-build difference; flagged as needing re-measurement under heavier traffic before the QoE mapper's `LatencyProxy` staleness handling is trusted here (Stage 7).
8. **`qoe_oran_framework/scripts/run_saclb_live_testbed.sh` assumes a different repo layout and a different (colliding) core deployment**, and launches its own conflicting gNB+3-UE fleet — structural mismatches with the rig as actually built per every other instruction, not fixable via config-YAML edits. Reported to the user rather than patched; user directed running the xApp entrypoint directly against the already-verified stack instead (Stage 9).
9. **Minor**: UE attach logged two benign NAS-IEI warnings ("0x21"/"0x5e" not handled when extracting allowed NSSAI") that didn't block attach or anything downstream (Stage 5) — noted in case it matters for future NSSAI-list-related work.

### Verdict
**Cleared to resume DQN/A2C offline retraining and QoE-mode work**, with two caveats the next session should read first:
- The `mmtc` slice's `max_ratio_cap` and any live behavior involving it are **unverified on this rig** — no UE was ever attached to it. Treat any `mmtc` live results with that in mind, or attach a UE there first (RAM permitting) before trusting them.
- `dl_mac_buffer_occupation`'s intermittency characterization should be re-measured under realistic (not just `ping`) traffic before the QoE mapper's staleness handling is trusted live on this rig.

No DQN/A2C/QoE runs were started by me, per instructions — the above is the actual, live rig state as of this report, ready for that work to begin.

---

## Post-Stage-10 addendum: 3rd slice (urllc) + scale-out to 3 UEs

**Requested after the Stage 10 report above.** User asked for all 3 traffic types (urllc/eMBB/mMTC) and at least 3 UEs, overriding Stage 4/8's 2-slice/1-UE scope. Documenting this as an addendum rather than rewriting history above.

### Session-boundary finding (not a RAN crash)
Before starting this work, found the gNB/UE processes and tmux server from Stages 5-9 were gone — no live process matched (an earlier `pgrep` "still running" check right after this addendum's AMF restart was actually a **pgrep self-match artifact**: it matched its own invoking shell command line, which contained the literal string "nr-softmodem", not a real process). `sudo dmesg` showed **zero segfault/crash signatures** across the gap. Docker containers survived (managed by the persistent docker daemon), but the tmux server and native RAN processes did not — consistent with a session/sandbox boundary between conversation turns, not a software fault. Stage 6's stability finding stands; this was an environment artifact, cleanly recovered by relaunching everything fresh.

### Adding the 3rd core slice (urllc, sst=1/sd=1)
This checkout's `docker_open5gs` `open5gs_slicing` branch ships dedicated SMF/UPF pairs per slice (2 by default), and its `open5gs_init_slicing.sh` dispatcher only recognizes `smf-1`/`smf-2`/`upf-1`/`upf-2` — no generic N-slice support. Extended it properly rather than hacking a shared SMF:
- Created `smf/smf3.yaml`, `smf/smf_init3.sh`, `upf/upf3.yaml`, `upf/upf_init3.sh` (mirroring the slice2 pattern exactly: sd=000001, dnn=`oai3`).
- Added `smf-3`/`upf-3` dispatch branches to `base/open5gs_init_slicing.sh`.
- Added `.env` vars: `SMF_IP3=172.22.0.40`, `UPF_IP3`/`UPF_ADVERTISE_IP3=172.22.0.41`, `UE_IPV4_INTERNET3=192.168.102.0/24` (verified non-conflicting with existing ranges first).
- Added `smf-slice3`/`upf-slice3` services to `5g-sa-deploy-slicing.yaml`.
- Rebuilt `docker_open5gs_slicing` (fully cached except the final `COPY`, seconds not minutes) and brought the new containers up.
- **Two small bugs caught and fixed**: the new `smf_init3.sh`/`upf_init3.sh` I wrote lacked the execute bit (`Write` doesn't preserve/set it) — `chmod +x` fixed it. The one-time `mongosh` shim from Stage 4 was gone because `docker compose up -d` recreated the `amf` container — reapplied it (harmless, same workaround the framework's own provisioning script would have needed).
- Added the 3rd S-NSSAI to `amf/amf.yaml`'s `plmn_support.s_nssai` list, restarted `amf`.
- **A real gap found only by testing**: AMF logged `No SMF Instance` / `HTTP response error [403]` then `[400]` for the new slice — traced to **NSSF** (`nssf/nssf.yaml`), which independently gatekeeps slice selection and had its own hardcoded 2-entry `nsi` list, never touched until this point. Added the 3rd entry, restarted `nssf`. The first retry after that still failed (`400`) — an AMF-side NSSF-client cache hadn't refreshed yet; a second retry a few minutes later succeeded cleanly (`SCP-discover` found NSSF and SMF-slice3 correctly, PDU session established). Not a config bug, just eventual consistency after the NSSF restart.
- Added the 3rd S-NSSAI (`{sst=1;sd=0x000001;}`) to the gNB's `snssaiList`, restarted the gNB (confirmed via its boot log: `Slice id = 3 [ sst = 1, sd = 000001 ]`).
- Created `nrUE_slice3.conf` (imsi `...778`, dnn=`oai3`, sst=1/sd=0x1) and provisioned the matching subscriber via `open5gs-dbctl add_ue_with_slice`.

### Real bug found and fixed: native multi-UE TUN interface collision
Launching a second bare `nr-uesoftmodem` process on the same host **is not supported out of the box**: the TUN interface name is hardcoded to `oaitun_ue` + (local UE index + 1) (`openair3/NAS/UE/ESM/esm_ebr_context.c`), so every independent process tries to create `oaitun_ue1` regardless of which slice/IMSI it's for. UE2's first attempt logged `Error opening socket oaitun_ue1 (16:Device or resource busy)` then cascading `fd -1` errors — but still printed a **misleading "Interface oaitun_ue1 successfully configured" success message** despite the broken TUN path, and it silently clobbered UE1's already-working interface (UE1's ping went from 0% to 100% loss immediately after). This is why the framework's own `generate_ue_fleet_compose.py` puts each UE in its own Docker container (separate network namespace) — the officially-designed answer to this exact problem, requiring a `docker_oai_nrue:local` image never built in any stage.

**Fix, lighter than a full container image build**: Linux network namespaces + veth pairs, one per additional UE (`ue2ns`/`ue3ns`, veth `10.99.2.0/30` and `10.99.3.0/30`), running each `nr-uesoftmodem` via `sudo ip netns exec <ns> ...` with `--rfsimulator.serveraddr` pointed at the veth's host-side IP (the gNB's rfsimulator already listens on `0.0.0.0:4043`, confirmed via `ss -tlnp`, so this needed no gNB-side change). Cleanly resolved the collision — all 3 UEs (default netns for UE1, `ue2ns`, `ue3ns`) now run simultaneously with independent `oaitun_ue1` devices that don't collide.

### Result: all 3 UEs live simultaneously
- UE1 (embb, sd=0xFFFFFF, default netns): `192.168.100.3`
- UE2 (mmtc, sd=2, `ue2ns`): `192.168.200.3`
- UE3 (urllc, sd=1, `ue3ns`): `192.168.102.2`
- All three: **0% packet loss** pinging 8.8.8.8 through their respective PDU sessions.
- RAM under all 3 UEs + 17-core (3×SMF+3×UPF+11 other) core containers + gNB: plateaued around 600Mi-1.2Gi free / 1.4Gi swap in use — tight (as Stage 8 predicted) but **not thrashing**: swap usage held steady rather than climbing, and a 3-minute targeted watch (dmesg segfault grep + a timing-advance MAC warning that appeared on one UE) showed the warning rate fluctuating in a steady 30-78/200-lines band rather than escalating, heartbeats climbing continuously throughout (661→715 over 3 min), zero segfault signatures. Treat this as **stable but with less margin than the single-UE configuration** — not a clean bill of health for extended unattended runs, a reasonable state for active testing.
- Updated P3/P4 (120 polls, all 3 UEs live simultaneously):

| Slice (sst/sd) | avg_prbs_dl | dl_mac_buffer_occupation | dl_total_bytes | dl_errors | dl_bler | Demand (mean/max PRB) |
|---|---|---|---|---|---|---|
| urllc (1/1) | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% | 5.00 / 5.00 |
| embb (1/16777215) | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% | 5.00 / 5.00 |
| mmtc (1/2) | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% | 5.00 / 5.00 |

All three slices show **identical** demand (mean=max=5.00 PRB) — consistent with each UE being idle-attached with no application traffic generator running (just NAS/RRC-level default allocation), not yet real differentiated load per slice. `dl_mac_buffer_occupation` remains 0.0% across all three, reinforcing Stage 7's finding that this field's population is a function of real offered traffic load, not something this bring-up has exercised yet.

### Still open, worth resolving before trusting live multi-slice results further
- `saclb_live.yaml`'s `slices:` list currently only has `embb`/`mmtc` (Stage 9's trim) — does **not** yet include the newly-added `urllc` entry. Re-adding it (sd=1, matching this addendum's real gNB slice) would need a `max_ratio_cap` decision now backed by real data (mean 5.00 PRB, same as the other two) rather than guesswork, unlike Stage 9's original mmtc caveat.
- No traffic generator was run against any of the 3 UEs here — all P3/P4 numbers reflect idle-attached demand. Real per-slice differentiation (the kind Stage 9's smoke test showed for embb vs. mmtc) requires actual traffic, same lesson as Stage 7/9.
- RAM margin is real but thin with all 3 UEs up — no additional UEs should be added without re-checking `free -h` first.


## Preliminary reading (required before Stage 1)
- Read `PROJECT_HANDOFF_SUMMARY.md` in full (§4 six technical findings, §5 fresh-build preconditions) — done.
- Read `MIGRATION_PRECONDITION_REPORT.md` in full — done. Key takeaways:
  - Target = ORANSlice `main` (OAI 2024.w28 base), NOT vanilla upstream OAI (no E2_AGENT there).
  - Wire protocol (UDP 6655 in / 6600 out, request/response, SUBSCRIPTION crashes gNB) re-verified via source read + protobuf round-trip on the new checkout as of 2026-07-14 — still needs live re-verification (Stage 7 P2).
  - Segfault root cause (`nr_dci_size`/`get_ul_tdalist` NULL derefs) addressed by NULL guards added between v2.1.0 and 2024.w28 — Stage 6 soak is the live proof, not assumed from source reading alone.
  - Framework: 140/140 tests passed against regenerated `ran_messages_pb2.py` in the precondition-verification session.

Staged into `~/oranslice_rig/`: `rig_bringup.sh`, `PROJECT_HANDOFF_SUMMARY.md`, `MIGRATION_PRECONDITION_REPORT.md`, this log.

---
