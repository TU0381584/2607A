# 3-UE / 3-slice Experimental Campaign Log

Repo root: `~/oranslice_rig/`. Campaign scope per the handover prompt (conference
paper + IEEE Transactions journal paper). This log follows `BRINGUP_LOG.md`'s
discipline: every phase, config hash, seed, wall-clock time, pass/fail, anomaly.

---

## Session start — 2026-07-16

Read `PROJECT_HANDOFF_SUMMARY.md`, `MIGRATION_PRECONDITION_REPORT.md`,
`BRINGUP_LOG.md` in full per the handover's instructions. Key state inherited
from bring-up:
- ORANSlice main (OAI 2024.w28), commit `b9bcc9b17fbecfc1041072a7b8d0f01ae874aba2`.
- 140/140 framework tests passed.
- 3 real MAC slices confirmed on this gNB: id1 sst=1/sd=0xFFFFFF (embb),
  id2 sst=1/sd=0x000002 (mmtc), id3 sst=1/sd=0x000001 (urllc) — the 3rd
  (urllc) was added in a post-Stage-10 addendum (core SMF3/UPF3, NSSF, gNB
  snssaiList, `nrUE_slice3.conf`).
- `dl_mac_buffer_occupation` measured 0.0% on this rig in all prior probes —
  every prior measurement was under idle-attach or light-ping traffic only,
  never real per-slice-differentiated load. This campaign's traffic profiles
  are the first real test of whether that field populates under genuine load.
- `saclb_live.yaml` (framework config) only lists `embb`/`mmtc` slices (Stage
  9 trim, before urllc's core slice existed) — needs a `urllc` entry added
  under `experiments/configs/` (not editing the frozen file in place; new
  campaign configs live under `experiments/configs/`).
- `rrmPolicy.json` periodic-reload path (`nr_update_slice_policy()`) is
  commented out by default on this checkout (Stage 7 P5 finding) — the
  built-in RRM baseline realization is confirmed NON-functional. Baseline
  arm uses the handover's documented fallback (see Phase 0/1 section below
  once designed).

### Session-boundary finding (recurring, not new)
Exactly as the Stage-10 addendum found before: tmux server, all native
`nr-softmodem`/`nr-uesoftmodem` processes, and (this time, unlike last time)
all Docker containers were gone at this session's start — confirmed via
`tmux ls` (no server), `pgrep` (no matches), `docker ps` (empty). This is a
sandbox/session-boundary artifact, not a rig crash: `dmesg` showed nothing,
and all persistent config/data (docker volumes, gNB/UE `.conf` file edits,
Mongo subscriber records, the 3rd-slice core additions) survived intact.
Every long-lived process must be relaunched at the start of any new working
session — this is now an expected, not exceptional, step.

### Stack re-stand-up (this session)
1. `docker compose -f 5g-sa-deploy-slicing.yaml up -d` — all 17 containers
   up cleanly (mongo, webui, nrf, scp, ausf, udr, nssf, bsf, udm, pcf, amf,
   3x smf-slice{1,2,3}, 3x upf-slice{1,2,3}). Confirmed Mongo subscriber
   records for all 3 IMSIs (...776 embb, ...777 mmtc, ...778 urllc) survived
   in the volume.
2. gNB launched in tmux session `gnb`: binaries found at
   `ORANSlice/oai_ran/cmake_targets/ran_build/build/` (note: NOT
   `ORANSlice/oai_ran/ran_build/...` — the correct path includes
   `cmake_targets/`). Boot log confirms all 4 slices configured (id0
   reserved sst=0/sd=0, id1 embb, id2 mmtc, id3 urllc), NGAP registered
   with AMF (`associated AMF 1`, AMF log confirms `gNB-N2 accepted`), E2
   agent heartbeats present.
3. UE1 (embb, default netns): attached cleanly, `oaitun_ue1` @
   192.168.100.2, 0% loss to 8.8.8.8.
4. UE2 (mmtc) and UE3 (urllc): launched inside dedicated network namespaces
   (`ue2ns`/`ue3ns`, veth pairs `10.99.2.0/30` / `10.99.3.0/30`), reusing
   the exact pattern from the Stage-10 addendum (native `nr-uesoftmodem`
   processes on one host can't share a TUN interface name — hardcoded to
   `oaitun_ue1` regardless of which process — so each additional UE needs
   its own netns). Both attached cleanly: UE2 @ 192.168.200.2, UE3 @
   192.168.102.2, both 0% loss to 8.8.8.8.
5. All 3 UEs simultaneously live, `sudo dmesg` shows zero segfault/crash
   signatures. RAM: 181Mi free / 1.3Gi swap in use with all 3 UEs +
   17-container core + gNB running — tight, matching the addendum's
   "stable but with less margin than single-UE" finding. No additional
   processes should be added without re-checking `free -h` first.

**Status: full stack live.** Proceeding to traffic-profile design (per-slice,
pinned to a config, logged, never varied between arms), then Phase 0 sanity
re-check.

---

### Traffic profile design + iperf3 target setup

Per-slice traffic generation (native processes, since UEs run as bare
`nr-uesoftmodem` + netns, not docker containers -- `drl_slicing/scripts/`'s
docker-exec-based traffic patterns don't apply directly here, only their
per-slice shape parameters were reused as a starting point):
- A dedicated `iperf3-target` Docker container was attached to the core's
  own `demo-open5gs-public-net` bridge (verified reachable from all 3 UEs'
  PDU sessions: UE -> gNB -> UPF -> docker bridge -> container, confirmed
  via ping from all 3 netns before relying on it). One iperf3 server
  instance per slice, on ports 5201 (embb)/5202 (urllc)/5203 (mmtc), all
  DOWNLINK (`--reverse`, server-sends-to-UE) since the KPM fields of
  interest are DL counters.
- Config pinned at `experiments/configs/traffic_profiles.yaml`: embb
  sustained UDP, urllc sustained small-packet UDP, mmtc bursty (2s on/6s
  off) small-packet UDP. Driver: `experiments/traffic/run_traffic_profiles.sh`.

**Calibration finding:** embb was initially set to 15 Mbps ("sustained
high-throughput"). Measured real (unconstrained-ceiling) demand at that
rate: ~26-34 PRB, a ~7-8x oversubscription against the calibrated cap=4 --
so extreme that ceiling changes within [floor,cap] would never move
backlog (permanently saturated regardless of policy), which would have
destroyed Phase 2/3's learning signal exactly the way the OLD rig's Fig 6
problem did, just from the opposite direction (over- rather than
under-contention). Retuned to 4 Mbps: re-measured organic (ceiling wide
open) demand = ~5 PRB, giving a mild ~1.25x oversubscription against
cap=4 -- consistent with the ORIGINAL Stage 9 calibration philosophy
(cap set just below observed mean demand). embb bitrate finalized at 4M in
both the traffic config and this log. urllc/mmtc's demand floors (both
~5 PRB, matching embb's own floor at low bitrates -- apparently a
scheduler minimum-grant floor on this clean rfsim channel, not a bug)
were left at their existing designs (urllc 300K/100B sustained, mmtc
50K/80B bursty).

**IMPORTANT diagnostic note for future measurements**: `avg_prbs_dl` is a
SERVED metric, gated by whatever ceiling is currently active, not raw
offered demand -- if a non-default ceiling is already in force (e.g. left
over from a prior probe's `--send-control`), P3 readings will reflect the
ceiling, not organic demand. Always reset to a wide-open ceiling
(min=0,max=100, the gNB's own boot default) and let any prior backlog
fully drain before trusting a P3 demand reading as "organic."

### Phase 0 sanity check (initial pass, under real traffic)

`probe_e2_preconditions.py --polls 60` under real per-slice traffic (before
the embb retune) showed, for the first time on this rig, REAL
differentiated `dl_mac_buffer_occupation` population: embb 80.0%, urllc
31.7%, mmtc 1.7% (vs. BRINGUP_LOG's 0.0%/0.0%/0.0% under idle/light-ping
only). This is the expected, deliberate effect of adding real traffic, not
drift/regression -- Phase 0 PASSES: no unexplained deviation, only the
anticipated traffic-driven change BRINGUP_LOG itself flagged as needed
("re-measure under heavier traffic before trusting the QoE mapper's
staleness handling"). Re-run after the embb retune as part of the Phase 1
work below.

### Operational finding: hot-restarting a single UE into an already-running
### 3-UE stack is unreliable; a full cold restart is not

During the first Phase 1 contention-gate attempt (embb pinned to
min=max=1 for ~45s under 15 Mbps traffic, then restored), embb's
`dl_mac_buffer_occupation` genuinely spiked (documented below), but UE1
(embb) then hit `[RLC] max RETX reached on DRB 1` repeatedly and lost all
connectivity (even to the AMF/internet, not just the iperf3 target) --
a radio-bearer-level failure, not a segfault (dmesg clean throughout,
`nr-softmodem`/`nr-uesoftmodem` processes never died).

Four consecutive attempts to hot-restart *only* UE1 (kill + relaunch)
while the gNB, UE2, and UE3 kept running all failed the same way: NAS
registration completed, but the rfsimulator TCP channel's socket
send/receive queues grew unboundedly (`ss` showed hundreds of KB queued,
climbing) and the PDU session never finished (`oaitun_ue1` stayed DOWN,
no IP). Root cause, evidenced by `ss`/`top`: the freshly (re)starting UE
process needs a burst of real-time catch-up processing that this rig
cannot reliably grant it while gNB + 2 other UE softmodems are already
running (gNB alone steady-state ~160% CPU; RAM chronically sits at
150-250MB free / ~1.8GB swap in use with all 3 UEs attached, even at pure
idle -- matching the Stage-10 addendum's own "stable but with less margin"
caveat, now shown to not extend to "stable under mid-session disruption").

**Fix that worked**: stop ALL of gNB+UE1+UE2+UE3 (not just the failing
one), then cold-restart the full RAN stack in the original sequence (gNB,
settle, UE1, settle, UE2, settle, UE3), same as the initial stand-up.
Succeeded immediately, 0% loss to 8.8.8.8 on all 3 UEs.

**Operational rule adopted for the rest of this campaign**: if any single
UE needs restarting for any reason once the 3-UE stack is live, restart
the WHOLE RAN stack (gNB + all 3 UEs) from a clean stop, not just the
affected UE. Budget ~2 minutes settle time for a full restart. Avoid
unnecessary churn generally (e.g. don't recreate the iperf3-target
container while UEs are attached and live -- do it before UEs attach, or
accept a full RAN restart afterward).

**Second operational finding**: UE PDU session IPs are NOT stable across
restarts (Open5GS's IP pool advances each time) -- UE1 alone went
`192.168.100.2 -> .100.3 -> .100.6` across this session's restarts.
`experiments/traffic/run_traffic_profiles.sh` was fixed to auto-detect
each UE's current IP (`ip addr show oaitun_ue1` / the netns equivalent)
at every invocation rather than hardcoding it -- a hardcoded IP silently
targets a dead address after any future restart with no error (iperf3
client just fails to connect).

### Ratio-to-PRB mapping: empirical, not the assumed 1:1

Original Stage 9 calibration assumed `max_ratio` (the config's ratio
units, 0-100 scale) maps roughly 1:1 onto real PRBs at this rig's 106-PRB/
numerology-1 configuration (e.g. cap=4 approx 4 real PRB). Live measurement
under this campaign's real traffic shows that assumption does NOT hold:
pinning embb's `max_ratio` to 4 empirically allowed ~15-22 real PRB served
(measured via `avg_prbs_dl` during the contention-gate tests below), not
~4. **The correct, evidence-based approach (per the handoff's own repeated
lesson -- "always poll real demand before setting ratio caps," "verify,
don't assume") is to empirically sweep candidate `max_ratio` values and
read off the resulting served PRB directly, not to reason about the
internal ratio formula.** embb's organic (ceiling-wide-open) demand
settled to **~15 PRB mean (range ~5-23 across polls, likely reflecting
residual-backlog transients from recent ceiling changes rather than true
demand volatility)** at the finalized 4 Mbps traffic rate. urllc/mmtc's
organic demand sits at a ~5 PRB floor (matching the scheduler's apparent
per-UE minimum grant on this clean rfsim channel, seen consistently
across light and moderate traffic alike).

### Phase 1 CONTENTION GATE: PASS

Final clean run (`experiments/scripts/phase1_contention_gate.py`, trace at
`experiments/logs/phase1/embb_final_20260716_204703.jsonl`), embb slice
(sst=1, sd=16777215), under the finalized 4 Mbps campaign traffic:

| Phase | mean `dl_mac_buffer_occupation` | n |
|---|---|---|
| baseline (ceiling wide open) | 411.3 | 15 |
| **pinned** (min=max=1) | **6,601,079.0** (max 10,023,785) | 30 |
| recovery (restored to min=1,max=20) | 2,504,530.4 | 20 |

**Verdict: PASS.** Pinning embb's ceiling far below its organic demand
raised backlog by >4 orders of magnitude within the pinned window (well
within one episode's worth of wall-clock at this campaign's 5s/step
cadence), and restoring to a ceiling closer to (but still below) organic
demand produced real, measurable drainage (6.6M -> 2.5M) rather than
permanent saturation -- both directions of the required
ceiling-down-implies-backlog-up / ceiling-up-implies-relief relationship
are demonstrated on this rig, under this campaign's real traffic, with
raw data preserved for the eventual Fig-5/backlog-CDF figures. (An earlier,
now-superseded attempt at 15 Mbps embb traffic and restore-to-cap=4 also
showed the pinned-phase spike but no recovery, because cap=4 was itself
still far below the heavier 15 Mbps demand -- superseded by the retune
above, not disregarded: that run is what surfaced the ratio-to-PRB
mapping finding in the first place.)

### Calibrated caps finalized for this campaign (superseding Stage 9's)

| Slice | sst/sd | organic demand (mean PRB) | `max_ratio_cap` | rationale |
|---|---|---|---|---|
| embb | 1/16777215 | ~15 (range 5-23) | **12** | slightly below mean -- binding but not permanently saturating (validated live: cap=20 gave real relief, cap=1 gave severe starvation; 12 sits meaningfully below the ~15 mean while leaving less headroom than 20) |
| urllc | 1/1 | ~5 (floor) | **4** | mirrors embb's original Stage 9 philosophy (cap just below observed floor); this rig's urllc core slice did not exist at Stage 9, so this is a fresh calibration, not inherited |
| mmtc | 1/2 | ~5 (floor) | **3** | unchanged from Stage 9 (already validated against this same ~5 PRB floor) |

Written into the campaign's own config, `experiments/configs/saclb_campaign.yaml`
(the frozen `qoe_oran_framework/configs/saclb_live.yaml` is left untouched
per the hard rule -- only new files under `experiments/` carry campaign-
specific values). gNB left resting at embb min=1/max=12 (the calibrated
value) after the Phase 1 gate test concluded; traffic generators stopped
to let the rig idle while configs/scripts are written up.

Config validated: `load_saclb_config()` loads `saclb_campaign.yaml`
cleanly, all 3 slices present with the correct sst/sd/floor/cap values,
`cfg.qoe is not None`.

### Baseline arm: design, implementation, live smoke-test PASS

The handover's `baseline` arm (static per-slice RRM ratio, no learning, no
per-step control) cannot use the framework's `rrmPolicy.json` periodic-
reload path (confirmed non-functional, BRINGUP_LOG.md Stage 7 P5) nor the
framework's `lb_only` comparator (that IS an active heuristic -- it moves
the ceiling every step based on observed saturation/quota, per
`comparators/lb_only_baseline.py` -- not a static ratio). Implemented as a
new script, `experiments/scripts/run_baseline_static.py`, which:
- reuses `RANEnv`/`run_single` from the frozen `mc_runner.py` UNMODIFIED
  (only imported), for byte-identical logging/reward/diagnostics code
  paths against every other arm -- critical for the paired comparison;
- passes a tiny local `AlwaysAcceptPolicy` (every request decided
  "accept") together with `algorithm="baseline_static"` (a new string, not
  `"lb_only"`) so the omega log's `method` field records honest
  provenance and doesn't pick up the inapplicable
  `LB_ONLY_ROUTING_LIMITATION`;
- relies on a one-line campaign-config variant,
  `experiments/configs/saclb_campaign_baseline.yaml`
  (`arrivals.ceiling_step_ratio: 0` instead of `1`, everything else
  byte-identical to `saclb_campaign.yaml`), so `AdmissionGate.apply()` can
  never move a slice's ceiling away from its `reset_ceilings()` initial
  value (`min_ratio_floor` / `nominal_ratio`) for the whole episode,
  regardless of any accept/reject decision.

**Realization used (recording per the handover's explicit requirement):**
realization "A" -- periodic reassertion of an unchanging static value, not
"send once at episode start then nothing further." The gNB does receive a
`slicing_control_m` most steps (verified in the gNB log), but every one
carries the identical min/max ratio for that slice throughout the episode
-- `AdmissionGate.apply()`'s `changed[key] = ceiling` fires whenever any
pending request touches that (gNB, slice) key, even when the resulting
value is unchanged, and this is a framework-internal detail not bypassable
without a source edit. Static VALUE, not static WIRE TRAFFIC -- this
satisfies "no learning, no per-step control" (the ratio itself never
adapts) without needing frozen-source changes.

**Live smoke test** (5 steps, 1s cadence, throwaway config, against the
live gNB): PASSED. Omega log confirms `method="baseline_static"` on every
record and the exact SAME ceiling (`embb max_ratio=3`, `urllc max_ratio=2`,
`mmtc max_ratio=2` -- each slice's calibrated `nominal_ratio`) on all 5
steps; gNB log independently confirms `Slicing_Ctrl_Msg Applied` with
those identical values repeated, never drifting. QoE diagnostics
(mean_mos_by_slice, cost, sla_viol) populated correctly even under
`reward_mode="sla"`, confirming the passive-diagnostic path (env.py's
`cfg.qoe is not None` branch) works for this arm too.

### Third recurrence of UE1 RLC max-RETX -- escalating, not resolved by a restart alone

Shortly after the baseline smoke test (traffic generators already
stopped, rig otherwise idle), a routine health check found ALL 3 UEs had
lost connectivity again (100% loss to both 8.8.8.8 and the AMF itself),
with UE1's log again showing repeated `[RLC] max RETX reached on DRB 1`.
This is the SAME failure signature as the first occurrence (which was
during 15 Mbps sustained traffic) and the second (during hot-restart
attempts) -- but this time it occurred with no traffic generators running
and no UE restart in progress, after roughly 40 minutes of cumulative
gNB+3-UE uptime carrying a long sequence of live probes, contention-gate
control messages, and the baseline smoke test.

`dmesg` remained clean throughout (no segfault -- all 16 expected
processes were still alive when this was noticed) and RAM was not
acutely exhausted at the moment of failure (~400MB free), so this is not
the same simple "sudden OOM/swap-storm" story as the earlier two
episodes. The common thread across all three occurrences is cumulative
session duration and total control/traffic activity on a rig that runs
chronically close to its RAM ceiling (150-400MB free is the recurring
range with all 3 UEs attached, regardless of what else is happening) --
consistent with intermittent missed real-time deadlines in the rfsim
PHY/MAC pipeline under sustained memory pressure, not a one-off trigger
tied to any single traffic level or action.

**Finding, stated plainly for planning purposes:** this rig has now shown
it cannot be assumed to sustain a live gNB+3-UE session indefinitely --
UE1 (the embb slice) in particular has failed 3 times within roughly one
hour of cumulative uptime, each requiring a FULL stack restart (not a
per-UE restart, see above) to recover. This directly bears on Phase 3's
feasibility: a live evaluation campaign that assumes many consecutive
hours of uninterrupted 3-UE operation is not a safe assumption on this
hardware as currently observed. All RAN processes (gNB + 3 UEs) were
stopped at the end of this session to let the rig rest; the Docker core
(all 17 containers) was left running (stable throughout, unaffected by
any of these RLC episodes). RAM with only the core up: 2.9 GiB free (vs.
150-400 MB with gNB+3UEs attached) -- confirms the core alone is not the
resource pressure; the RAN processes are.

**Reported to the user for a scope/mitigation decision before committing
further live-rig wall-clock to Phase 2/3** -- see the session status
update. Not treated as a silent per-incident restart-and-continue matter
any further, given the 3rd recurrence within one session.

**User decision**: short episodes + periodic health-checked restarts
(recommended option). Implemented as three new scripts:
- `experiments/scripts/restart_ran_stack.sh` -- full stop + sequenced
  relaunch (gNB, settle+verify, UE1, settle+verify, UE2, settle+verify,
  UE3, settle+verify, connectivity check), the exact procedure that
  recovered the rig every time above. Never restarts a single UE alone.
- `experiments/scripts/health_check.sh` -- process count + all-3-UEs-
  reachable + recent-dmesg-segfault check, exit 0/1.
- `experiments/scripts/run_live_eval_arm.py` -- orchestrates ONE arm/seed
  ("rep") of live evaluation in small episode batches (default batch size
  2), health-checking before every batch and invoking
  `restart_ran_stack.sh` (up to 2 retries) if unhealthy, before continuing.
  Wraps the frozen `saclb_xapp.py` (learned arms) or this campaign's own
  `run_baseline_static.py` (baseline arm) as subprocesses -- no frozen
  source touched. All batches for one rep append to the SAME
  `omega_log.jsonl` (OmegaLogger's own append-mode behavior).
  **Honestly documented limitation** (see the script's own docstring):
  each batch reseeds the policy/env RNG from scratch
  (`base_seed*1000 + batch_index`, deterministic and logged in
  `batch_manifest.jsonl`), so the raw omega log's `episode`/`step`/
  `global_step` fields are only continuous WITHIN a batch, not across the
  whole rep -- Phase 4 analysis must reconstruct cross-batch episode
  ordering from `batch_manifest.jsonl`, not assume the raw fields are
  already rep-global. This is the direct, necessary cost of the
  short-episodes-plus-restarts mitigation and is recorded here so a
  future reader (or me, later) doesn't misread the raw log as one
  continuous seeded trajectory.

---

## Phase 2 — Offline retraining (synthetic ClosedLoopKpmSource, no live rig)

Config: `experiments/configs/saclb_offline_campaign.yaml` -- same slice
ORDER (embb, urllc, mmtc -- verified identical, load-bearing for
checkpoint compatibility, see the config's own comment) and same
reward-shape hyperparameters (Lmax, congestion_coeff, priority_weight,
violation_penalty, latency/loss budgets) as
`experiments/configs/saclb_campaign.yaml` (the live-eval config) --
`max_ratio_cap` values differ deliberately (pinned at nominal_ratio per
`train_offline.py`'s `OVERSUBSCRIPTION_FACTOR=1.25` synthetic-demand
philosophy, same approach as the prior session's
`saclb_offline_live1gnb.yaml`). Verified via direct `load_saclb_config()`
call: both configs produce `state_dim=9`, `request_state_dim=13`, slice
order `['embb','urllc','mmtc']` in both -- checkpoints trained on the
offline config WILL load and mean the same thing against the live config.

Timing check: 10 episodes (dqn, sla-mode) completed in ~4s wall-clock
(offline mode has no live step-cadence sleep) -- 300 episodes/seed/arm
estimated at ~2 min; all 4 learned arms x 3 seeds = 12 runs, ~24 min
total. Entirely synthetic (ClosedLoopKpmSource) -- no live-rig involvement,
no RLC-failure risk.

Launched in background: `experiments/scripts/run_phase2_training.sh`
(2 algorithms x 2 reward modes x 3 seeds = 12 runs, 300 episodes each,
seeds 256/257/258, results under `experiments/results/offline/seed<N>/`).

**Bug caught mid-run and fixed, data discarded and rerun (not silently
patched over):** `train_offline.py`'s `results_dir` argument does not
include `reward_mode` in its directory scheme --
`mc_runner.run_mc`'s `rep_dir = results_dir/algorithm/f"offline_{source}"/rep_0`
depends only on `algorithm` and `--source` (always `closed_loop` here),
NOT on `--reward-mode`. My first driver script version passed the SAME
`--results-dir` (`.../seed256`) for both `--reward-mode sla` and
`--reward-mode qoe` runs of the same algorithm+seed, so the qoe run's
`rep_0/` directory silently collided with the sla run's: `OmegaLogger`
opens in append mode, so the qoe run's episodes got appended into the
SAME `omega_log.jsonl` as the sla run (mixing two different reward-mode
runs' rows in one file, no way to cleanly separate them after the fact),
and `checkpoint.pt` was fully overwritten (not appended) -- destroying the
sla-mode-trained weights.

Caught by inspecting the actual checkpoint files present partway through
the run (only found `dqn`/`a2c` checkpoints, no reward-mode subdirectory,
right after `dqn`'s qoe-mode runs had just completed following its
sla-mode runs) rather than trusting the driver's own "done" printouts.
**Killed the training driver immediately** (before `a2c`'s qoe-mode runs
could similarly clobber its already-completed, still-clean sla-mode
results) and deleted the corrupted `dqn` sla/qoe-mixed results entirely --
not "recovered" or partially salvaged, since the omega log's mixed rows
could not be reliably un-mixed after the fact. `a2c`'s sla-mode results
(3 seeds) were confirmed untouched (its qoe-mode runs had not started
yet) but were deleted anyway for a clean, uniform re-run rather than a
run matrix with mismatched provenance guarantees between arms.

**Fix**: `results_dir` now folds in reward_mode BEFORE seed
(`experiments/results/offline/<sla|qoe>/seed<N>/<algo>/...`), making
collision structurally impossible. Re-launched all 12 runs from scratch
under the fixed path scheme.

**Secondary finding**: the re-launched training ran ~3x slower per episode
(3+ min vs. the original ~1 min) because the RAN stack (gNB+3 UEs) was
running concurrently (brought up in preparation for the Phase 3 trial) --
confirms offline training and live RAN operation genuinely compete for
this rig's CPU. Stopped the RAN stack again while training finishes (will
do a fresh `restart_ran_stack.sh` right before the Phase 3 trial) -- both
faster and avoids adding unnecessary CPU pressure on top of the
already-documented RLC-failure risk while nothing live-rig-related
actually needs to be running.

### Phase 2 result: PASS, all 12 checkpoints saved, convergence verified

All 12 runs completed cleanly (~13 min total wall-clock with the RAN
stack stopped). Checkpoints confirmed present at all 12 expected paths.
Inter-seed variance on the mean of each seed's final-10-episode reward
(the "old rig: ~1.5%" benchmark the handover cites):

| Arm | Final-10-ep mean reward, per seed (256/257/258) | Inter-seed variance |
|---|---|---|
| dqn_sla | -4.728 / -4.633 / -4.705 | **0.87%** |
| a2c_sla | -4.583 / -4.468 / -4.505 | **1.07%** |
| dqn_qoe | -0.489 / -0.492 / -0.492 | **0.27%** |
| a2c_qoe | -0.487 / -0.486 / -0.487 | **0.13%** |

All 4 well within the 1.5% benchmark -- **Phase 2 PASS**.

`experiments/plots/fig1_training_convergence.py` generated (2x2 layout,
one subplot per arm, NOT overlaid -- eq.2/sla and eq.9/qoe reward
magnitudes are on genuinely different scales, confirmed real by an
initial overlaid attempt that visually flattened both qoe curves to
near-invisible flat lines near the top of the sla-scaled axis; the
handover's Phase 4 spec explicitly permits "2x2 or overlaid" for this
reason). Real finding visible in the figure: **A2C converges almost
immediately (within ~10 episodes) under both reward modes, while DQN
takes ~150 episodes to plateau under sla-mode** (both converge fast under
qoe-mode) -- worth carrying into the paper's training-behavior discussion.

---

## Phase 3 — Live evaluation campaign

Per the user's decision above (short episodes + health-checked restarts),
and a follow-up decision to run a ~20-minute pipeline TRIAL before
committing to the full campaign: short-episode config variants created
(`experiments/configs/saclb_campaign_trial.yaml` /
`saclb_campaign_baseline_trial.yaml`, 30 steps x 3s = 90s/episode instead
of the real campaign's 60 x 5s = 5min) and a trial driver
(`experiments/scripts/run_phase3_trial.sh`) that runs 1 episode per arm
(all 5 arms) through the real orchestrator
(`run_live_eval_arm.py`), using seed256's checkpoints as representative
weights for the 4 learned arms. This run's numbers are a PIPELINE SMOKE
TEST only (short episodes, single seed) -- not evidence for the paper.

### Trial result: PASS -- pipeline validated end-to-end, restart mechanism proven live

Ran 21:37:34-21:52:17 (14m43s total for 5 arms x 1 episode each, comfortably
under the 20-minute budget).

**Data quality: excellent, exactly the expected signature.** embb ceiling
trajectory over the episode:
- `baseline`: `[3,3,3,...]` -- pinned at its nominal_ratio for all 30
  steps, never moves. Confirms the static-ratio realization works
  correctly under real live conditions, not just the earlier synthetic
  smoke test.
- `dqn_sla` / `dqn_qoe`: `[5,6,7,8,8,9,10,11,11,11,12,12,12,...]` -- rides
  the ceiling up from the floor to the calibrated cap (12) within ~10
  steps and holds there. This is the Fig-4 headline signature (learned
  policy actively pushes the ceiling; baseline never leaves its floor)
  showing up correctly on the very first live trial.

**Operational finding, load-bearing for the wall-clock estimate below:**
the health check failed and required a full stack restart before **4 of
the 5** arm transitions (only the very first, `baseline`, ran without a
preceding restart). Every restart succeeded on the first attempt and the
subsequent episode then completed cleanly (all 5 omega logs have the
full expected 31 rows -- 30 steps + 1 rollup -- with `returncode: 0`).
**The health-check-and-restart mechanism worked exactly as designed,
fully automatically, with no manual intervention needed** -- this is the
mitigation the user chose, now proven live rather than just implemented.

This does mean the rig's real failure rate under actual per-slice
traffic load is HIGHER than the ~3-times-per-hour figure estimated
earlier (which was mostly observed under idle/light-ping conditions) --
closer to needing a restart roughly once per episode-scale unit of
activity under real load. Reported to the user with a revised wall-clock
estimate before launching the full campaign.

---

## Session resume (2026-07-18)

Machine had rebooted since the last session (docker containers "Exited
137 hours ago", RAN processes gone -- expected/handled). New finding:
**`ue2ns`/`ue3ns` network namespaces do NOT survive a reboot** (`ip netns
list` came back empty) -- `restart_ran_stack.sh` was not yet
self-healing for this, so UE2's launch hung with no PDU session. Fixed:
the script now checks for and idempotently recreates both netns + veth
pairs if missing, before launching anything. Verified working on this
same resume (UE2/UE3 attached cleanly after the fix).

### 30-minute END-TO-END trial (real cadence, real checkpoints) -- run before committing to the full 8h campaign

Per the user's request for a real (not compressed) dry run: 1 FULL-LENGTH
episode (60 steps x 5s = 5 min) per arm, all 5 arms, via
`experiments/scripts/run_phase3_trial30.sh` (real `saclb_campaign.yaml` /
seed-256 checkpoints, not the trial-config variants). Total wall-clock
21:31:52-14:04:05 = **32m13s** for 5 real episodes + 4 automatic
health-check restarts (one before nearly every arm transition, same
pattern as the earlier compressed trial) -- restart mechanism worked
flawlessly again, zero manual intervention.

**Bug caught before reporting anything (verify, don't trust a chart on
sight):** `fig5_backlog.py`'s CDF used `per_slice_sla_margin` assuming it
was pre-clipped to [0,1] per `RunSummary`'s docstring description --
actual raw values go far outside that range under real contention (seen:
mean ~-944,615 for a badly-backlogged slice, matching Phase 1's
multi-order-of-magnitude backlog finding). On a linear axis this
compressed the entire informative region into an invisible sliver,
making the CDF look like baseline was "comfortable" (~1.0) the whole
episode -- the OPPOSITE of the true result, and directly contradicted by
the compliance numbers below. Caught by cross-checking the figure against
`fig2`'s numbers before reporting; fixed by clipping the display (not the
underlying data) at -1.5 with the clipping stated on the axis label.

### Trial result (n=1 episode/seed per arm -- directional, not final evidence)

**SLA compliance (fig2) -- dramatic, consistent with the corrected
backlog CDF (fig5):**

| Arm | eMBB | URLLC | mMTC |
|---|---|---|---|
| baseline | 1.7% | 1.7% | 3.3% |
| dqn_sla / a2c_sla / dqn_qoe / a2c_qoe | ~100% | ~100% | ~100% |

Mechanism, directly visible in `fig4` (ceiling trajectory, baseline vs.
dqn_sla): baseline's ceiling sits flat at its nominal ratio (embb=3,
urllc=2, mmtc=2) for all 60 steps; dqn_sla rides eMBB's ceiling from ~4
to the cap (12) within ~13 steps and holds there, and URLLC/mMTC jump to
their caps (4/3) immediately. This is the real, mechanistic reason the
compliance numbers differ so starkly -- not a reward-shaping artifact.

**Inferred MOS (fig6) -- mixed, NOT uniformly favorable to the learned
arms, reported plainly rather than omitted:** eMBB/mMTC MOS improves
somewhat under the learned arms, but **URLLC MOS is pinned at the exact
floor (1.0) for all 4 learned arms** (dqn_sla/a2c_sla mean=1.0000 exactly,
dqn_qoe mean=1.0047, a2c_qoe mean=1.0000), while baseline's URLLC MOS is
also low on average (1.06) but shows real variance up to 4.83 at some
steps. This does not contradict the compliance result (compliance is a
backlog/loss-budget measure; MOS is a separately-calibrated QoE
inference) but is a genuine limitation worth carrying into the paper
rather than hiding -- plausibly related to the already-documented
"URLLC's weaker IQX-alone fit" finding from Stage One calibration.

**URLLC blocking (fig3) -- one clear outlier:** baseline and 3 of 4
learned arms (dqn_sla, a2c_sla, dqn_qoe) show 0 URLLC blocks this
episode; **a2c_qoe blocked all three slices heavily this one episode**
(embb=47, urllc=35, mmtc=40) -- a real, single-episode data point, not
noise-filtered yet (n=1). Plausibly connects to a previously-documented
framework-level finding (an old rig investigation noted A2C's tendency
to over-reject under certain Lmax/reward configurations) -- flagged as
something to watch across more seeds/episodes in the full campaign, not
dismissed as a fluke without more data.

**Overall read**: strong, mechanistically-explained evidence that the
learned arms outperform the static baseline on SLA compliance and
backlog control -- a genuine, non-cherry-picked result. MOS and
a2c_qoe's blocking behavior are real nuances the full campaign needs
enough seeds/episodes to characterize properly (n=1 cannot distinguish a
systematic a2c_qoe issue from single-episode variance). Reported to the
user with this full picture before proceeding to the 8h campaign.

User then had this packaged for a conference-paper handoff
(`HANDOFF_FOR_PAPER.md` + a zip on the Desktop), explicitly flagging that
n=1 trial figures are "too basic" to be convincing -- correct, and stated
as such in the handoff doc. New handover: run the full campaign, adjudicate
the two flagged anomalies with pre-registered decision rules, produce
`RESULTS_REPORT.md`, publication figures, and an IEEEtran paper scaffold.

---

## Phase A — Full live campaign

### Pre-flight (2026-07-18, continued)

Rig brought back up (`restart_ran_stack.sh`) after the trial -- clean on
first try. `iperf3-target` needed a fresh recreate (the container's port
5201 server had wedged after ~4.5h holding a stale session from the
earlier trial: "the server is busy running a test" -- same failure mode
as previously documented, same fix). Traffic confirmed flowing (embb
~4.1 Mbps). Probe diff against CAMPAIGN_LOG's calibrated values:

| Slice | Calibrated organic demand | This pre-flight | Drift? |
|---|---|---|---|
| embb | ~15 PRB mean (range 5-23) | 15.60 mean / 19 max | none, within range |
| urllc | ~5 PRB floor | 5.00 mean / 5.00 max | none |
| mmtc | ~5 PRB floor | 5.00 mean / 5.00 max | none |

No material drift -- **pre-flight PASS**.

### Campaign design

- **Grid**: 5 arms x 3 seeds (950/951/952, continuing the trial's seed
  950) x 5 episodes/arm/seed = 75 episodes total.
- **Checkpoints**: fixed to each learned arm's seed-256 offline-trained
  checkpoint (one representative, well-converged checkpoint per arm,
  consistent with "evaluate frozen weights" -- the seed varying across
  950/951/952 is the LIVE evaluation seed controlling env RNG/synthetic
  arrivals, not which offline checkpoint is loaded).
- **Arm order interleaved per seed** (rotation, not fixed), so no arm
  always runs at the same point in the rig's uptime/drift curve:
  - seed 950: baseline, dqn_sla, a2c_sla, dqn_qoe, a2c_qoe
  - seed 951: dqn_sla, a2c_sla, dqn_qoe, a2c_qoe, baseline
  - seed 952: a2c_sla, dqn_qoe, a2c_qoe, baseline, dqn_sla
- **Between every arm**: `experiments/scripts/drain_backlog.sh` opens all
  3 slices' ceilings wide (min=0,max=100), waits 20s, re-probes -- so no
  arm inherits the previous arm's queue state.
- **Crash-safe progress**: `experiments/scripts/run_phase_a_campaign.sh`
  appends one plain-text line to
  `experiments/results/live_campaign/PROGRESS.log` per completed
  (arm, seed) BEFORE moving on, and skips any (arm, seed) already marked
  DONE there on restart -- so a session crash mid-campaign loses at most
  the in-flight (arm, seed), not prior accounting.
- Uses the already-proven `run_live_eval_arm.py` orchestrator unchanged
  (health-check + auto-restart between episode batches, batch size 2).

Estimated wall-clock: 75 episodes x 5 min = 6.25h pure episode time, plus
drain (20s x ~15 arm-transitions ~= 5 min) plus restart overhead (~once
per episode transition at the trial's observed rate, ~1.5-2 min each) --
consistent with the ~8h estimate already given to and approved by the
user.

Launched in tmux session `campaign` at 18:08:18.

### Progress checkpoints (updated as (arm, seed) pairs complete)

- **baseline/seed950: DONE** (1738s = ~29 min, incl. one health-check
  restart before the first batch). On pace with the ~8h estimate
  (15 arm-seed pairs x ~29 min each ~= 7.25h + inter-seed overhead).
- **dqn_sla/seed950: DONE** (1725s = ~29 min, consistent with baseline's
  timing). 2/15 arm-seed pairs complete.
- **a2c_sla/seed950: DONE** (1738s). 3/15 complete.
- **dqn_qoe/seed950: DONE** (1726s). 4/15 complete. All 4 pairs so far
  ~1725-1738s each -- very consistent, restart mechanism firing reliably
  (one restart per arm transition, same pattern as both trials) with no
  anomalies.
- **a2c_qoe/seed950: DONE** (1727s). **5/15 complete -- seed 950's full
  5-arm rotation finished cleanly**, ~144 min total for the first seed
  pass (matches the ~29min/arm-seed x 5 estimate closely). Now starting
  seed 951's rotation (dqn_sla, a2c_sla, dqn_qoe, a2c_qoe, baseline order).
  a2c_qoe/950's actual blocking numbers will be checked against the n=1
  trial's 47/35/40 finding once Phase B analysis runs on the full dataset
  (not spot-checked mid-campaign, to avoid biasing later interpretation).
- **dqn_sla/951, a2c_sla/951, dqn_qoe/951, a2c_qoe/951: DONE** (1738s,
  1728s, 1725s, 1731s). 9/15 complete. Pace remains steady (~29 min/pair
  including restarts); no anomalies, no stop conditions triggered.
  a2c_qoe/951 needed more health-check restarts within its own batches
  than other arms typically do -- consistent with (not yet confirmed as
  causally linked to) the trial's a2c_qoe anomaly; noted for Phase B, not
  interpreted mid-campaign.
- **baseline/951: DONE** (1716s). **10/15 complete -- seed 951's full
  5-arm rotation finished cleanly.** Two of three seeds done, ~5h elapsed
  (18:08-23:02), on pace for completion around 01:00-01:30. Now starting
  seed 952's rotation (a2c_sla, dqn_qoe, a2c_qoe, baseline, dqn_sla
  order).
- **a2c_sla/952, dqn_qoe/952: DONE** (1721s, 1733s). 12/15 complete.
  Remaining: a2c_qoe/952 (in progress), baseline/952, dqn_sla/952.

### Phase A ended early: two genuine crashes, stop condition triggered

**Timeline of the final stretch:** after 13/15 arm-seed pairs completed
cleanly (baseline, a2c_sla, dqn_qoe, a2c_qoe all fully done across all 3
seeds; dqn_sla done for seeds 950/951), `baseline/seed952` failed partway
through its final batch (4/5 episodes already written) when the health
check found all 3 UEs unreachable and the automated
`restart_ran_stack.sh` recovery **itself failed** (UE1 did not attach
within 20s even on a fresh launch) -- the first time in ~6h45m of
continuous operation this recovery mechanism has failed outright. The
campaign driver logged `FAILED arm=baseline seed=952` per its crash-safe
design and moved to the next arm, `dqn_sla/seed952`, which immediately
hit the same "UE unreachable" health check and its own restart attempt
also failed, so it was logged `FAILED` with zero episodes completed. The
campaign loop then reached the end of seed 952's rotation and printed
`PHASE A CAMPAIGN COMPLETE` (all 15 slots attempted, not all succeeded --
this is the driver behaving exactly as designed, not a silent success).

**Manual investigation (not further blind retries) found TWO DISTINCT,
genuine crash signatures in the vendored OAI/ORANSlice source within the
same recovery window:**

1. **A real segfault** in `nr-uesoftmodem`, thread `Tpool3_-1`, captured
   via `dmesg` and resolved with `addr2line` against the kernel's own
   reported file-offset (`nr-uesoftmodem[41a205,...]` -- the kernel's
   bracket notation already gives the PIE-corrected file offset, no
   manual `/proc/<pid>/maps` correction needed this time):
   ```
   addr2line -f -C -e nr-uesoftmodem 0x41a205
   -> nr_ue_periodic_srs_scheduling
      openair2/LAYER2/NR_MAC_UE/nr_ue_scheduler.c:1140
   ```
   Root cause read directly from source: the function guards against
   `current_UL_BWP->srs_Config` being NULL, then unconditionally
   dereferences `srs_config->srs_ResourceSetToAddModList->list.count` --
   but `srs_Config` can be non-NULL while `srs_ResourceSetToAddModList`
   itself is NULL (plausibly during a BWP/config reconfiguration
   transient), which is exactly the same CLASS of bug as the old rig's
   `nr_dci_size`/`get_ul_tdalist` NULL-derefs from the original bring-up
   (a config sub-struct that's present-but-incomplete during a transient
   window) -- just a different specific function, apparently not covered
   by the NULL-guard fixes that landed between v2.1.0 and this 2024.w28
   base for the *other* functions.

2. **A second, different crash** on the very next restart attempt: an
   explicit `AssertFatal` (not a kernel segfault -- a controlled abort),
   in `lockGet_ul_iterator()`,
   `openair2/LAYER2/NR_MAC_UE/nr_ue_scheduler.c:223`:
   ```
   Assertion (is_nr_UL_slot(tdd_config, slot_tx, mac->frame_type) != 0) failed!
   UL config_request called at wrong slot 7
   ```
   This happened AFTER a full, clean RA procedure (PRACH -> RAR ->
   RRCSetup -> RRC_CONNECTED) -- the UE actually attached successfully,
   then the gNB immediately sent an `RRC Release`, and the UE's own
   scheduler crashed shortly after on a UL config request for a slot the
   TDD configuration says isn't an uplink slot. `nr-softmodem` (gNB) was
   NOT affected either time -- E2 agent heartbeats continued uninterrupted
   throughout both crashes; both are UE-side (`nr-uesoftmodem`) bugs.

**Why this is a stop condition, not "just another restart cycle":** every
prior restart in this campaign (dozens, across 13 clean arm-seed pairs)
recovered via the SAME mechanism on the first or second attempt with no
crash signature at all (the earlier RLC-max-retx pattern is a link
degradation, not a process crash). Two DIFFERENT genuine crashes on two
CONSECUTIVE fresh-restart attempts, after ~6h45m of continuous heavy
operation, is a materially different and more severe failure mode --
matching the handover's explicit stop condition ("rig instability
materially worse than the characterized... restart pattern") and
separately, on its own, the "any gNB/UE crash or segfault" stop condition
(interpreted to cover the UE process here, since both crashes are
architecturally the same class of real-time-scheduling/NULL-deref bug
the handoff docs already treat as gNB-adjacent). Both bugs are in the
frozen, vendored `ORANSlice/oai_ran/` C source -- not something I can or
should patch.

**Data completeness as of stopping (verified directly from omega log row
counts, 61 rows/episode):**

| Arm | seed 950 | seed 951 | seed 952 | Valid seeds (>=3 ep) |
|---|---|---|---|---|
| baseline | 5/5 | 5/5 | **4/5** | 3 (meets floor) |
| dqn_sla | 5/5 | 5/5 | **0/5 (missing)** | **2 (BELOW the 3-seed floor)** |
| a2c_sla | 5/5 | 5/5 | 5/5 | 3 (complete) |
| dqn_qoe | 5/5 | 5/5 | 5/5 | 3 (complete) |
| a2c_qoe | 5/5 | 5/5 | 5/5 | 3 (complete) |

13/15 arm-seed pairs are complete and clean (65 valid episodes' worth of
data across 4 of 5 arms, all with full 3-seed coverage). Only `dqn_sla`
falls short of the campaign's own stop-condition floor (3 seeds x >=3
episodes) -- its seed-952 rep has zero data, not a partial one.

**Rig left in a safe state**: all native RAN processes (gNB + 3 UEs)
stopped cleanly (no lingering broken processes); Docker core left
running (unaffected by either crash, stable throughout the entire ~6h45m
run exactly as in every prior session). Traffic generators stopped.
**Stopping here to report to the user rather than attempting further
unilateral recovery or silently treating dqn_sla as n=2**, per the
handover's explicit instruction.

### Resume + successful retry: Phase A now 15/15 complete

Resumed with a genuinely fresh rig restart (not continuing the crashed
6h45m session -- docker core and all RAN processes had stopped between
sessions, netns self-healed correctly). `iperf3-target` recreated fresh.
Discarded `baseline/seed952`'s partial 4/5 episodes (not patched/topped
up -- redone cleanly from scratch, consistent with this campaign's
established discipline for handling any partial/incomplete data) and
reran the campaign driver, which correctly skipped all 13 already-`DONE`
pairs (via `PROGRESS.log`) and retried only `baseline/seed952` and
`dqn_sla/seed952`.

Both completed cleanly on the fresh rig: `baseline/seed952` (1720s, 5/5
episodes) and `dqn_sla/seed952` (1723s, 5/5 episodes) -- ordinary
health-check-restart cycles throughout, no crash signatures, same
well-characterized pattern as the other 13 pairs. **Neither of the two
crash bugs (SRS-scheduling segfault, UL-iterator AssertFatal) recurred**
on the fresh rig, consistent with them being genuine but low-probability
/ cumulative-uptime-correlated bugs in the vendored source rather than a
deterministic, immediately-reproducible fault.

**Phase A final result: 15/15 arm-seed pairs complete, verified directly
from omega log row counts (305 rows = 5 episodes x 61 rows/episode,
exactly, for all 15):**

| Arm | seed 950 | seed 951 | seed 952 |
|---|---|---|---|
| baseline | 5/5 | 5/5 | 5/5 |
| dqn_sla | 5/5 | 5/5 | 5/5 |
| a2c_sla | 5/5 | 5/5 | 5/5 |
| dqn_qoe | 5/5 | 5/5 | 5/5 |
| a2c_qoe | 5/5 | 5/5 | 5/5 |

75/75 episodes, full 3-seed x 5-episode coverage for all 5 arms. `dmesg`
clean (no segfaults) since the resume. Rig stopped safely (RAN processes
+ traffic off, Docker core stable throughout). **Proceeding to Phase B
(analysis + anomaly adjudication) with the complete dataset.**

---

## Final summary: what the n=3 campaign confirmed vs. changed relative to the n=1 trial

The n=1 trial's core qualitative claim held up: all four learned arms
achieve robust, consistent SLA compliance (100.0±0.0% across 45 episodes)
via the same ceiling-riding mechanism, while the static baseline does not
match that reliability. But the trial's specific *numbers* were
misleading at n=1 and the n=3 campaign corrected them honestly rather
than confirming them by coincidence:
- **Baseline's ~1.7-3.3% compliance did not replicate.** The real,
  3-seed picture is bimodal (60%/100%/60%) with a mean of ~74% and large
  variance -- the trial's single episode sampled the worse mode. The
  corrected finding (reliability vs. an inconsistent baseline, not
  rescuing an always-failing one) is arguably a *more* interesting result
  for the paper, not a weaker one.
- **The "URLLC MOS pinned at exactly 1.0" finding did not replicate as
  stated.** Real values range 1.2-1.7 across arms with real (if modest)
  policy-driven differentiation. The underlying real finding -- eMBB and
  URLLC both show low, largely policy-independent inferred MOS while
  mMTC is high, and baseline's URLLC MOS variance is the largest of any
  arm/slice -- is genuine and now properly characterized instead of
  overstated as a hard floor.
- **The `a2c_qoe` mass-blocking anomaly DID replicate, decisively.**
  100% of its 15 episodes across all 3 seeds show 28-49 blocks on all 3
  slices simultaneously, while every algorithm/reward control (`a2c_sla`,
  `dqn_qoe`) shows exactly zero. This is the strongest-evidence finding
  of the whole campaign precisely because n=1 could not have
  distinguished it from noise, and n=3 removes all doubt.

Two genuine crashes in the vendored OAI source (a segfault and a
separate AssertFatal, both in `nr_ue_scheduler.c`, both root-caused via
dmesg+addr2line+source reading) interrupted the campaign once after
~6h45m of continuous operation; the affected arm-seed pairs were
discarded and redone from scratch on a freshly restarted rig, completing
cleanly with no recurrence. Final dataset: 15/15 arm-seed pairs, 75/75
episodes, fully verified. `RESULTS_REPORT.md`, all Phase 4 figures, and
`paper_conf/` (6-page IEEEtran scaffold, compiles cleanly, author-owned
sections left as explicit TODO blocks) are complete and committed.
