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

---

## 2026-07-19 -- Phase 1 follow-up: re-visualization of existing S1 data (zero rig time)

A proposed multi-phase follow-up plan (static-max diagnostic probe, a
harder scarcity+dynamics S2 scenario, retraining, and a second ~10-14h
live campaign) arrived via a pasted "handover" attributed to a separate
Claude session. Given the plan's real rig-time cost and irreversibility
(a second live campaign, a frozen scenario redesign, a paper
restructure) and the fact that it arrived as third-party pasted content
rather than direct instruction, only the zero-rig-time, data-safe piece
was authorized to proceed: re-visualizing the EXISTING S1 campaign data
to state its results more decisively without collecting anything new.
Everything else in that plan (static_max probe, S2 scenario design,
retraining, second campaign) remains on hold, not started.

**What changed, all from the same omega logs already in
`experiments/results/live_campaign/` (no new data collected):**

1. **`experiments/plots/fig2_sla_compliance.py` rewritten.** The original
   grouped-bar mean±std figure collapsed a genuinely bimodal per-episode
   distribution into a misleading "73.7%±18.6%" summary. The new figure
   is a two-panel per-episode reliability plot: top panel is a strip/dot
   plot, one point per episode (n=15/arm, deterministically jittered, no
   RNG), showing every learned arm's episodes flat at 100.0% against
   baseline's real bimodal split (majority near 100%, a genuine cluster
   at 0%); bottom panel is fraction-of-episodes-fully-compliant and
   worst-single-episode-compliance per arm. Confirmed by direct
   computation: baseline's worst episode is 0.0% compliant on every
   slice; every learned arm's worst episode is 100.0%.
2. **`experiments/plots/fig5_backlog.py` rewritten.** Added a top row
   showing one representative episode's per-step SLA margin (seed 950,
   episode 1 -- same seed/episode/run_id convention as fig4, first
   `run_id` in file order) on a symlog axis: baseline (bad seed)
   collapses to its floor (~-1e6) within ~5 steps and never recovers,
   while DQN/SLA stays flat near +1.0 for the whole episode. The
   original pooled CDF (all 3 seeds x 5 episodes, clipped at -1.5) is
   kept unchanged as the bottom row -- it still earns its space as the
   aggregate complement to the single-episode illustration.
3. **`experiments/plots/fig4_ceiling_trajectories.py` updated in place**
   (kept, not replaced, per the follow-up's own instruction): each
   per-slice panel now annotates its calibrated `max_ratio_cap` (embb=12,
   urllc=4, mmtc=3, sourced from `saclb_campaign.yaml`) as a dotted
   reference line, so the ceiling-riding mechanism reads against the
   actual ceiling-of-the-ceiling.
4. **`experiments/plots/generate_results_tables.py` updated**: added
   "Worst ep. (%)" (minimum per-episode compliance, pooled across ALL
   episodes/seeds -- not a mean of per-rep means) and "P5 margin" (5th
   percentile of pooled per-step SLA margin) columns to the Markdown,
   LaTeX (Table III), and JSON outputs, alongside the existing mean±std
   columns. Re-ran against the same seeds (950/951/952); numbers
   unchanged elsewhere, new columns added.
5. **`experiments/RESULTS_REPORT.md` §2 and `experiments/figures_manifest.md`**
   updated to describe the new figures/columns and carry the new numbers.
6. **`paper_conf/main.tex`** captions for Fig. 2 (compliance) and Fig. 4
   (backlog, per this repo's float-order numbering) rewritten to
   describe the new panels; one added sentence in §V-A citing the new
   worst-episode/P5-margin columns; a pre-existing stray Markdown
   `**bold**` marker in prose (would have rendered as literal asterisks)
   fixed to `\emph{}`. No claims changed -- same underlying data, stated
   more precisely.
7. Recompiled cleanly: `pdflatex` x2 + `bibtex` + `pdflatex` x1, 6 pages,
   no undefined references, no errors. New figures visually verified in
   the compiled PDF (Table III widened to 8 columns fits within
   `table*`; both new figures legible at single/double-column size).

**Verdict: this re-visualization makes the S1 reliability result more
decisive using only existing, already-collected data.** No new claims
were introduced -- the underlying numbers (100.0% vs. bimodal 60-100%
compliance, ~-1e6 vs. ~0.7-1.0 margin) were already in the n=3 dataset;
only their presentation changed. Static-max probe, S2 hard-scenario
design, retraining, and a second live campaign remain on hold pending
explicit authorization.

---

## 2026-07-20 -- Objectives v3 (Design-only): A1/B1 investigation, NOT frozen -- major finding, decision needed

A third pasted "handover" (same third-party-attributed pattern as the
2026-07-19 one) proposed three new evaluation objectives (A: stochastic
admission control under overload: B: resource efficiency/beta-cost sweep;
C: QoE-vs-QoS dissociation via a real ABR client) requiring retraining and
up to ~10h of live-rig time. Given the cost and the pattern, only a
"design-only" scope was authorized: A1 (overload arrival-process design +
freeze) and B1 (beta-cost sweep design + freeze) via offline
simulation/config only -- explicitly NO retraining, NO live rig time.
Everything else (A2/A3, B2/B3, Objective C, and the live campaigns)
remains on hold.

**A1 finding #1 -- frozen-code gap (STOP condition per the plan's own
rules).** Read `types.AdmissionRequest` (`framework/qoe_oran_framework/types.py`):
fields are `request_id, slice_id, gnb_id, arrival_step, synthetic` only --
no resource_demand, lifetime, or per-request SLA weight. `action_mapping.AdmissionGate.apply()`
always nudges a slice's ceiling by a fixed `ceiling_step_ratio` per
accept/reject, regardless of any notion of request "size"; there is no
occupancy/lifetime model (a request is a one-shot decision, not a
session that holds and later releases capacity). `config.ArrivalConfig`
exposes exactly 3 knobs (`synthetic_arrivals_per_step` -- one GLOBAL
count, not per-slice; `max_pending_per_step`; `ceiling_step_ratio`), and
`env._synthesize_requests` assigns each synthetic arrival to a UNIFORM
RANDOM slice -- there is no per-slice arrival-rate weight in the config
surface. **Conclusion: A1's literal ask ("heterogeneous request classes
per slice with distinct resource demands, lifetimes, and SLA weights")
is NOT achievable via config alone.** What IS achievable without touching
frozen source: (a) a global offered-load sweep via `synthetic_arrivals_per_step`;
(b) per-SLICE (not per-request) value asymmetry via `SliceSpec`'s
already-config-driven `priority_weight`/`accept_reward`/`violation_penalty`/
`nominal_ratio`/`min_ratio_floor`/`max_ratio_cap` fields -- these can
realize "URLLC high-value/scarce, eMBB bulky, mMTC numerous/tiny" in
spirit, at slice granularity, not per-request granularity. Reported here
rather than silently narrowed.

**A1 finding #2 -- "capacity" units mismatch.** The plan's A1 says
"capacity = measured, from the live rig's calibrated demand mapping."
But per PROJECT_HANDOFF_SUMMARY.md finding #5 (confirmed by reading
`env.py`/`replay_kpm_source.py` directly): the synthetic admission-request
stream and real UE traffic are decoupled layers with their OWN abstract
capacity notion (`GnbSpec.prb_capacity=100`=B, per-slice `nominal_ratio`
as "% of B", entirely separate from the live rig's real `avg_prbs_dl`
PRB measurements). "Sustainable capacity" for A1's load sweep must be
defined and measured within the admission-MDP's own B/nominal_ratio
abstraction, not via live PRB polling -- the plan's literal instruction
does not match the architecture.

**A1/B1 finding #3 -- MAJOR, confirmed via a zero-training offline
diagnostic (not retraining; 3 fixed non-learning policies x 3 seeds x 10
episodes each, `ClosedLoopKpmSource` + `RANEnv` as-is, no frozen-code
edits, ~seconds of compute): the EXISTING offline training environment
(`experiments/configs/saclb_offline_campaign.yaml`, used for ALL 4
learned arms' already-completed 300-episode x 3-seed offline training)
has almost NO admission-policy leverage over SLA outcomes, for three
compounding, identified reasons:**
1. `nominal_ratio`/`max_ratio_cap` for embb/urllc/mmtc are 3/2/2 (tiny,
   inherited-looking from the LIVE campaign's PRB-cap numbers without
   re-deriving them against the offline `ClosedLoopKpmSource`'s own
   demand/backlog dynamics, which routinely reach the hundreds) -- the
   admission ceiling's entire serving-capacity range (1-3 units) is a
   rounding error against backlog once it accumulates even slightly, so
   accept vs. reject choices barely move backlog at all past the first
   few steps of an episode.
2. `Lmax: 10` in the same config (the divisor for `queue_len_norm`) is
   similarly tiny relative to that same backlog scale, so the SLA
   violation-severity metric (`per_slice_sla_margin`/`sla_viol`) saturates
   at its worst value almost immediately in every episode, regardless of
   policy.
3. `ClosedLoopKpmSource`'s `backlog_capacity` default (200.0, not
   currently overridden by any experiments/-level config) further clips
   the raw signal early.

   **Direct evidence** (existing offline `dqn_qoe`/`a2c_qoe` training logs,
   300 episodes x 3 seeds x 2 algos, zero new compute -- just re-read):
   `sla_viol` sits above 0.9 in 99.9% of all 108,000 pooled steps, with
   NO improvement from the first to the last training quartile (0.997 ->
   1.000), and `mean_mos` is pinned near-floor (~1.07-1.08) the entire
   time. **Confirmed causally**, not just correlationally, via the new
   diagnostic: accept-all, reject-all, and a crude threshold-like policy
   were run through the SAME environment at the current config
   (backlog_capacity=200, Lmax=10) and, separately, at backlog_capacity
   raised 10x (2000) and Lmax raised up to 80x (800) -- across every
   combination tested, `sla_viol` and `backlog_mean` were STATISTICALLY
   INDISTINGUISHABLE between accept-all and reject-all (e.g. at
   backlog_capacity=2000/Lmax=800: sla_viol 0.492 vs. 0.473, backlog_mean
   850.7 vs. 831.8 -- a <2.5% gap between the two most extreme opposite
   admission strategies possible). Only the reward's "cost" term
   (accepted-count-driven, from eq.9) differentiated between policies;
   the actual SLA/backlog/MOS outcome barely did.

   **Implication:** this is a pre-existing characteristic of the
   ALREADY-COMPLETED offline training underpinning the S1 live campaign,
   not a new problem introduced by this session -- it does NOT retroactively
   invalidate S1 (S1's live evaluation used the REAL rig with a separately,
   already-validated PRB-cap calibration chain, per Phase 1's contention
   gate). But it means what DQN/A2C "learned" offline may have been driven
   substantially by minimizing the cost term rather than genuine SLA-outcome
   optimization, and it means Objective A/B's "restore the hard admission
   problem" premise requires recalibrating this nominal_ratio/cap/Lmax/
   backlog_capacity scale relationship as a PRECONDITION, not merely adding
   heterogeneous arrival classes on top of the current (structurally
   saturated) calibration -- a bigger lift than either objective's own A1/B1
   sub-task anticipated.

**B1 retroactive beta-sweep (zero new compute, existing qoe-reward offline
logs only, pooled 108,000 steps across dqn_qoe/a2c_qoe x 3 seeds):**
recomputed eq.9's `alpha*mos_norm - beta*cost - gamma*sla_viol` for
beta in {0.2 (current), 0.5, 1.0, 1.5, 2.0, 3.0, 5.0} using the ALREADY-LOGGED
mos_norm/cost/sla_viol per step (alpha=1.0, gamma=0.5 held fixed). Even at
beta=5.0 (25x current), beta*cost's mean contribution (0.227) stays well
below gamma*sla_viol's mean contribution (0.500, itself pinned near its
ceiling per finding #3 above) -- **this data cannot support a real B1
freeze decision**, because "compliance" in this dataset is already pinned
at its structural floor for reasons unrelated to beta (finding #3), so
"does raising beta destabilize compliance" is untestable against it. A
trustworthy B1 sweep needs the finding-#3 recalibration resolved first.

**Not frozen. Nothing retrained. No rig time used.** Reported back to
the user with the above findings and a request for a decision on how to
proceed (options: authorize an experiments/-level recalibration of
nominal_ratio/cap/Lmax/backlog_capacity as a new prerequisite design step;
narrow A1 to a slice-level-only heterogeneity design and accept the
current calibration's limits; or hold pending further review).

**User decision: recalibrate first.** Proceeded with a design iteration
(still zero training, zero rig time -- constructor kwargs + in-memory
`SliceSpec` overrides on the existing frozen `ClosedLoopKpmSource`/`RANEnv`,
no frozen-code edits):

**Recalibration #1 (iteration 1 of the plan's own 2-iteration budget):**
rescaled `nominal_ratio`/`min_ratio_floor`/`max_ratio_cap` from the
existing config's 2-3 units to tens-of-units matching the papers' own "%
of B=100" convention (embb 10/50/65, urllc 5/20/30, mmtc 5/15/20 --
floor/nominal/cap), keeping the EXISTING per-slice value asymmetry
(`priority_weight`/`violation_penalty`: urllc(5.0/8.0) > embb(3.5/5.0) >
mmtc(0.3/2.5)) unchanged since it was already reasonable. Redefined
"sustainable capacity" as the ceiling achievable AT `max_ratio_cap` (per
A1's literal wording), not `nominal_ratio` (the existing frozen
`train_offline.py`'s convention) -- offered demand = 1.5x that cap.
Swept `Lmax` in {30,60,150,300,600} and `backlog_capacity` in {300,600}
against the same 3-policy diagnostic (accept-all/reject-all/threshold-like).

**Result: the scale fix works, and reveals a genuinely rich mechanism --
but only for eMBB so far.** At Lmax=300, backlog_capacity=600,
oversub=1.5x cap (10 episodes x 3 seeds, per-slice, pooled):

| Policy | eMBB margin | eMBB compliant | eMBB block rate | URLLC/mMTC margin | URLLC/mMTC compliant |
|---|---|---|---|---|---|
| accept_all | +0.362 | 71.8% | 0% | -0.86 to -0.91 | 4.7%/6.2% |
| reject_all | +0.971 | 100.0% | 100% | -0.93 to -0.95 | 2.3%/3.6% |
| threshold_like | +0.840 | 99.9% | **7.9%** | -0.89 to -0.92 | 4.3%/5.6% |

For eMBB this is exactly the mechanism Objective A wants: threshold_like
achieves reject-all's near-full compliance at 1/12th the blocking cost --
genuine evidence a smart selective policy can beat both naive extremes,
not just tie one of them. **For URLLC/mMTC, all three policies remain
similarly poor (2-6% compliant) regardless of policy** -- the same
uniform 1.5x-of-cap oversubscription and Lmax that works for eMBB leaves
these two slices in an apparently-unwinnable regime for any tested
policy. Not yet root-caused (candidate explanations: their much smaller
absolute cap/demand scale interacting differently with the
`notify_rejected` relief formula, which relieves `offered/n_ues` per
reject -- same `n_ues` for every slice regardless of scale; or Lmax/
oversub genuinely need to be set per-slice, not uniformly, matching the
existing config's own already-asymmetric philosophy for other fields).

**Iteration 2 of 2, spent on URLLC/mMTC -- superseded by a much bigger
finding.** First tried bringing urllc/mmtc's absolute cap/floor scale up
to eMBB's range (urllc 8/35/55, mmtc 8/30/45, floor/nominal/cap) under
the same Lmax=300/oversub=1.5, keeping their existing higher
priority_weight/violation_penalty. Result: NO improvement -- all three
policies remained statistically indistinguishable for urllc/mmtc
(2-6% compliant regardless of policy), even at matched absolute scale to
eMBB. This ruled out "absolute scale mismatch" as urllc/mmtc's problem
and prompted checking the mechanism directly rather than guessing another
parameter.

**MAJOR FINDING, corrects the eMBB result reported above: eMBB's
admission-ceiling control has been completely non-functional in the
offline `ClosedLoopKpmSource` environment for the ENTIRE project's offline
training history (all 4 learned arms, both reward modes, every seed) --
a frozen-code / config interaction bug, verified directly and
reproducibly, not inferred:**

`replay_kpm_source.py` (frozen) hardcodes `_SD_FOR_SLICE = {"embb": 0,
"urllc": 1, "mmtc": 2}` and its reverse map, used by `ClosedLoopKpmSource
.send_control()` to figure out which slice's ceiling to update from the
`sd` argument it receives. But `experiments/configs/saclb_offline_campaign.yaml`
(and every other campaign config, correctly mirroring the LIVE rig's real
NSSAI convention) sets embb's `sd: 16777215` (0xFFFFFF, "the confirmed
real no-SD sentinel on this gNB" per that config's own comment) -- a
value `_SD_FOR_SLICE_REVERSE` has no entry for. Direct verification (60
scripted accept-all steps against the real `ClosedLoopKpmSource`/`RANEnv`,
zero training): `send_control()` was called 40 times for embb over 60
steps (confirmed via `sent_controls`), yet `_ceiling_ratio[('gnb-0','embb')]`
never moved off its `initial_ceiling_ratio` default of 100.0 the entire
time -- while urllc (sd=1) and mmtc (sd=2) correctly updated to their
configured caps in the same run. `_SD_FOR_SLICE`/`_SD_FOR_SLICE_REVERSE`
are used ONLY inside `replay_kpm_source.py` (grep-confirmed) -- `live_kpm_source.py`
does not use them at all, so **this does NOT affect S1's live-evaluation
results** (those go through the real E2 `slicing_control_m` wire protocol
against real OAI, unaffected). It DOES mean: every offline-trained
checkpoint's embb-slice admission decisions, for the entire project so
far, were made against a permanently-wide-open (ceiling=100, never
constrained) synthetic embb environment during training -- whatever an
agent "learned" to do for embb offline could never have had any real
effect on embb's simulated SLA outcome, by construction of this bug.

**This retroactively corrects this session's own eMBB recalibration
result reported above.** The clean accept-all/reject-all/threshold-like
differentiation observed for eMBB was NOT the ceiling-riding mechanism
(which cannot function for eMBB in this environment) -- it was entirely
attributable to the `notify_rejected` relief pathway (which fires from
block decisions directly, independent of `send_control()`/ceiling state).
eMBB's apparent "fix" was real in its measured numbers but was validating
a different, accidental mechanism than the one A1/B1 need. URLLC/mMTC's
resistance to every recalibration attempt in this session is now
explained too: they don't have this specific sd-mismatch bug (their sd
values happen to match `_SD_FOR_SLICE`'s hardcoded {1,2}), so their
ceiling control DOES function -- but under the SAME Lmax/oversub
parameters, the ceiling range still couldn't produce differentiation on
its own, and their fix genuinely required a different diagnosis than the
one this session had budget to complete before the sd bug was found.

**This is a genuine frozen-code bug (inside `replay_kpm_source.py`), not
a calibration or design choice -- a stop condition per this session's own
integrity rules ("if an objective turns out to require frozen-code
changes, STOP and report options; do not patch frozen code").** Reported
to the user with three options: (a) authorize a minimal, disclosed
frozen-code fix (make `_SD_FOR_SLICE`/`_SD_FOR_SLICE_REVERSE` read from
`cfg.slice_by_id`'s actual configured `sd` values instead of a hardcoded
{0,1,2}, a small and well-understood change); (b) work around it without
touching frozen code, by changing the OFFLINE config's embb `sd` to 0
(breaking its intentional parity with the live config's real NSSAI
convention, and needing a documented rationale for why offline/live
configs would then intentionally diverge on this field); (c) hold pending
further review. **Not fixed. Nothing retrained. No rig time used. A1/B1
remain un-frozen pending this decision.**

**User decision: fix the frozen code.** Implemented, minimally and with
full backward compatibility:
- `replay_kpm_source.py`, `ClosedLoopKpmSource.__init__`: added an
  optional `sd_for_slice: Optional[Dict[str,int]] = None` parameter,
  defaulting to the old module-level `_SD_FOR_SLICE` dict when not
  supplied (so every existing test/caller that doesn't pass real config
  SD values is byte-identical in behavior). `poll()`/`send_control()`
  now read `self._sd_for_slice`/`self._sd_for_slice_reverse` (built from
  the constructor arg) instead of the module-level globals directly.
  `SyntheticKpmSource` (the explicitly-non-authoritative open-loop smoke
  path) and `notify_rejected()` (already keyed by slice_id, not sd) were
  left untouched -- out of the bug's blast radius.
- `scripts/train_offline.py`'s `kpm_source_factory`: now passes
  `sd_for_slice={slice_id: spec.sd for slice_id, spec in cfg.slice_by_id.items()}`
  when constructing `ClosedLoopKpmSource`, so future offline training
  runs use the config's REAL sd values instead of the broken default.

**Verified, not assumed:** full test suite (`pytest qoe_oran_framework/tests/`)
-- 134 passed, 1 skipped (pre-existing skip, unrelated), zero
regressions. Direct re-run of the earlier reproduction script: after
the fix, `embb`'s ceiling now correctly rides to its configured cap
(3.0, under the OLD unscaled config) after 60 accept-all steps, instead
of staying stuck at 100.0 -- bug confirmed fixed at the mechanism level.

**Re-ran the full 3-policy diagnostic with the fix + Recalibration #1's
scale (nominal/cap/floor at tens-of-units) -- corrects the earlier
report again:** with the ceiling genuinely functional for all 3 slices
now, eMBB's PREVIOUSLY-clean differentiation collapsed back to
near-total, undifferentiated violation (~2% compliant for all 3
policies) at the SAME Lmax=300/backlog_capacity=600/oversub=1.5 that
looked good before the fix -- confirming that earlier "success" really
was 100% a bug artifact (relief-only), not a real result, exactly as
predicted. Re-swept Lmax/backlog_capacity/oversub post-fix (same
zero-training methodology): found a working combination that
differentiates all 3 slices simultaneously and monotonically:
`backlog_capacity=1000, oversub_of_cap=1.2, Lmax=1000` (10 episodes x 3
seeds, pooled):

| Policy | eMBB compliant | URLLC compliant | mMTC compliant |
|---|---|---|---|
| accept_all | 20% | 53% | 77% |
| threshold_like | 12% | 17% | 25% |
| reject_all | 11% | 12% | 22% |

With a genuinely functioning ceiling, accept_all is now consistently
the BEST policy for raw compliance (not reject_all, as the earlier
buggy/relief-only run suggested) -- intuitive once the ceiling actually
works: riding to cap maximizes served capacity, which dominates reject's
one-off relief bonus. This reframes what a learned policy needs to beat:
raw SLA compliance is NOT where accept-all is weak (it's the strongest
naive baseline on that axis); accept-all's real cost is the reward's
`beta*cost` congestion-penalty term (accepting unconditionally racks up
cost regardless of whether the request was worth it) -- meaning
Objective A's "selective rejection beats naive extremes" story and
Objective B's efficiency story are now the SAME underlying tension in
this environment, not two separate objectives to design independently.
Worth flagging back to whoever scopes A2/A3's design next.

**B1 implication:** the retroactive beta-sweep run earlier in this
session used offline logs collected under BOTH the scale-mismatch AND
the sd-mapping bug -- now doubly superseded. Any real B1 sweep needs
fresh short offline runs under this corrected environment (a small
amount of retraining), which stays out of scope for a design-only
session.

**Status: A1's core validity premise (a genuine overload regime where
accept/reject choices produce real, non-saturated, non-trivial
differentiation) is now demonstrated end-to-end, zero-training, for all
3 slices simultaneously, with a real bug fixed along the way.** Not yet
frozen as a final campaign config -- reported back to the user with the
above numbers and the A1/B1-conflation finding before writing a final
`saclb_admission_v3.yaml` and calling A1 "frozen."

**User: "execute your proposal."** Formalized the A/B merge
(`experiments/NOTE_admission_objective_merge.md`) into real, reusable,
script-generated artifacts -- no scratch/`/tmp` code, no in-memory-only
overrides, matching this project's own "every number script-generated"
discipline:
- `experiments/configs/saclb_admission_efficiency_v1.yaml` -- the frozen
  offline config for the merged "admission efficiency under overload"
  objective (nominal/floor/cap at tens-of-units, Lmax=1000, existing
  per-slice priority_weight/violation_penalty asymmetry unchanged; beta
  left at 0.2 as an explicit placeholder pending a real sweep, documented
  in-file as not yet decided).
- `experiments/scripts/admission_efficiency_env.py` -- non-frozen factory
  (backlog_capacity=1000.0, oversub_of_cap=1.2 x max_ratio_cap, real
  sd_for_slice wiring) -- the single source of truth for this environment
  going forward, replacing the session's earlier scratch diagnostics.
- `experiments/scripts/run_admission_efficiency_baselines.py` -- runs
  accept_all, reject_all, and the framework's own unmodified
  `LbOnlyHeuristic` (static-threshold) against the frozen config, writes
  a Markdown validity report.

**Ran it for real** (3 seeds x 10 episodes, zero training) --
`experiments/results/admission_efficiency/baseline_validity.md`:
**PASS.** Real, non-saturated, monotonic compliance differentiation
(accept_all 16-70% > static_threshold 16-44% > reject_all 10-18%,
per-slice) AND an inverted reward ordering (reject_all has the best mean
reward, +0.025, despite the worst compliance, because it never pays the
cost/congestion penalty; accept_all is worst, -0.382, despite the best
compliance) -- confirming the merge finding directly: no naive policy
manages compliance and cost jointly, which is exactly the room a learned
policy needs to be worth training.

**A1's validity check is now formally PASSED against a real, frozen,
script-generated config -- not scratch numbers.** `saclb_admission_efficiency_v1.yaml`
supersedes `saclb_offline_campaign.yaml` for any future admission-control
training. B1's beta value remains an explicit placeholder (0.2, unswept).
**Nothing retrained. No rig time used.** Offline retraining (3 seeds x
300 episodes/arm) against this config, and any eventual live confirmation
subset, remain separately un-authorized next steps.
