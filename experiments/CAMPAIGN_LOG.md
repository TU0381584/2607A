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
