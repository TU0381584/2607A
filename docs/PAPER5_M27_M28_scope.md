# Paper #5 M27/M28: scope note

Status: **M27 fully complete (both N=19 and N=7) and written into the
manuscript -- see docs/PAPER5_M27_scaling_reframe.md for the full
write-up, including a genuinely new finding at N=7 (single-agent DQN
collapses far more than M6's original 3-seed pilot suggested). M28
preparatory work complete (configs, orchestration script, a validated
candidate checkpoint, gNB2's E2 ports resolved); a first live attempt
with the user present achieved genuine 2-gNB registration and two
working radio links briefly, but did not hold under sustained real
traffic load (real link failures, not just slowness) and was aborted
cleanly -- see the dedicated section below for the full, honest
account. M28's live demo is not yet achieved.** The original scoping message for M27 ("offline scaling
reframe") and M28 ("live multi-gNB demo") was a one-line description
each in an earlier pasted instruction block; the fuller text did not
survive context compaction. This doc records the interpretation this
session is proceeding on, so it can be corrected on review rather than
silently assumed correct.

## M27: offline scaling reframe -- interpretation

M6 (`docs/PAPER5_M6_topology.md`) already ran the N=7/19 topology
scaling campaign (GAT-CTDE / independent DQN / single-agent DQN,
fully-connected / ring / hex) entirely against `ClosedLoopKpmSource`,
the same offline simulator M32-M34 (this session, commits 78e11a8 /
bc36bae / 639ec89) found to be fundamentally miscalibrated: its
congestion_level is capped near 0.03-0.09 by construction (served is
bounded by the admission ceiling, which stays far below real demand),
while live measurement on this rig shows real congestion at 0.23-0.26
(3 UEs) and 0.60-0.69 (6 UEs) -- both single-gNB numbers M6's own
multi-gNB work never had reason to know about.

"Reframe" is read here as: rerun the SAME topology/arm sweep against
`RealisticServedKpmSource` (already built and validated single-gNB,
generic over `gnb_ids` so it needed no change to extend to N>1) and
check whether M6's headline finding -- single-agent DQN collapses
totally at N=19, GAT-CTDE partially (~35%), independent DQN never
fully collapses but never cleanly differentiates either -- holds,
strengthens, or changes under congestion levels that actually resemble
what this rig's own live measurements show, rather than the frozen
simulator's synthetic proxy.

This is NOT a live claim and does not need the rig. It is a second
application of the same fix M34 already validated single-gNB, extended
to the multi-gNB axis this paper's own architecture (GAT-CTDE) is
actually about.

## M28: live multi-gNB demo -- interpretation, a real constraint found, and resolved

**Update:** the gNB2 E2-port blocker described below is now resolved (compile-only, no live hardware touched) -- see the Log section's final entry.


M26 (already in this session's history, before compaction) verified
this rig can run 2 concurrent gNBs with real E2 control and 0% packet
loss for a sustained (15-minute) window, but that a 3rd gNB caused a
severe, user-observed slowdown requiring immediate abort. 2 gNBs is
this rig's confirmed ceiling, not a lower bound to be pushed past.

Read the most direct way, M28 asks for a live demonstration using that
verified 2-gNB capability. A real constraint found while starting this:
the frozen live xApp (`qoe_oran_framework/xapp/saclb_xapp.py`) hard-
requires exactly one gNB (`if len(cfg.gnbs) != 1: parser.error(...)`),
and no existing checkpoint is dimension-matched to N=2 (existing
checkpoints are N=1, state_dim=13, or N=3, state_dim=34). A genuine
2-gNB live demo of a SHARED multi-agent policy (not two independent
single-gNB policies running side by side, which would not exercise
GAT-CTDE's actual coordination mechanism) needs: (a) a new N=2 offline
training config, (b) a checkpoint trained against it, and (c) a new,
non-frozen orchestration script generalizing saclb_xapp.py's loop to
two simultaneous E2 connections and one shared policy instance.

Given this rig's own documented history of severe live instability the
moment multi-gNB concurrency is pushed at all, and that the user is
unreachable for the duration of this session, the plan is:
**build and validate everything up to the point of actually firing two
live gNBs at once (config, checkpoint, orchestration script, offline-
verified control flow), but hold the actual live 2-gNB run for when the
user is back to watch it**, rather than take that specific class of
hardware risk unsupervised. Anything short of live execution -- config,
training, script correctness -- proceeds without waiting.

## Log

- Session start: this doc written, M27 script under construction.
- `experiments/scripts/m27_scaling_reframe.py` built and smoke-tested (N=7 and N=19, all 3 arms) -- monkeypatches `m6_run_experiment.make_kpm_source_factory` to use `RealisticServedKpmSource`, delegates everything else (CLI, resumability, results-writing) to `m6_run_experiment.main()` unchanged. Timing: N=19, all 3 arms, 1 seed, full 100-train/20-eval budget = 369s (matches M6's own historic per-seed pace closely).
- Launched the primary N=19 campaign (12 seeds 900-911, all 3 topologies, all 3 arms, `--resume-seeds`), running as a background sequential loop across topologies (`fully_connected` -> `ring` -> `hex`), ~3.7h estimated.
- In parallel, built M28's remaining prep pieces: `saclb_offline_live2gnb.yaml` / `saclb_live2gnb.yaml` (N=2 configs, dimension-matched: state_dim=19, request_state_dim=24, verified by loading), `multi_gnb_live_kpm_source.py` (wraps N LiveKpmSource instances behind one KpmSource -- offline-verified: construction, duplicate-port and N<2 validation, all pass; never opened a real remote connection), `m28_live_gat_ctde_2gnb.py` (new orchestration script, mirrors saclb_xapp.py's structure with GatCtdeMarlPolicy + run_episodes_marl instead of the single-agent path -- imports cleanly, not yet run against real hardware). Launched a 5-seed GAT-CTDE+independent_dqn training campaign against `saclb_offline_live2gnb.yaml` to produce actual live-demo checkpoint candidates.
- Early signal (3/12 seeds, N=19 fully_connected, gat_ctde, under the recalibrated simulator): precision looks different from M6's original bimodal near-ceiling/collapse pattern -- seed 900/902 sit at ~0.50 (mediocre, not collapsed, not clean either), seed 901 shows a genuinely new pattern (8857 total blocks, but 0 ever correctly hit mmTC -- consistently wrong target, not a collapse to always-accept). Too early to conclude anything from 3 seeds of 1 topology; noted here as a preliminary observation to revisit once the full sample is in, not a finding.
- M28 checkpoint training complete (5 seeds x 2 arms, N=2, `saclb_offline_live2gnb.yaml`, RealisticServedKpmSource from the start). GAT-CTDE: seeds 900/901/902 all show perfect precision (1.0000, 2308/2339/2345 blocks); seeds 903/904 collapse completely (0 blocks) -- a 3/5 non-collapse rate, roughly in line with M6's own ~35% GAT-CTDE collapse finding at a much larger N, though N=2 isn't a direct comparison point. independent_dqn: 4/5 seeds strong (0.93-1.00), one (901) mediocre (0.498), zero full collapses -- also matches M6's established "independent DQN never fully collapses, never cleanly differentiates either" pattern. **M28's checkpoint prerequisite is done**: `experiments/results/m28_live_checkpoint/gat_ctde/seed900/train/checkpoint.pt` (or 901/902) is a validated, non-collapsed, perfect-precision candidate ready for the eventual live demo. Searched for the actual gNB2 E2 port values used in M26's build (checked ORANSlice's own git history, file timestamps, binary strings) -- not recoverable, the source patch was reverted and never committed. The live 2-gNB run still needs: (a) a fresh, deliberate port decision/patch for gNB2, and (b) the user present per this doc's own stated caution. Everything else for M28 is ready.

- N=19 campaign completed (~3.87h, all 108 cells): single-agent DQN and GAT-CTDE's collapse rates statistically unchanged from M6's original findings (91.7% [75.0%,100%] vs 100%; 33.3% [11.1%,58.3%] vs pooled 35.4% [25.5%,45.8%]); independent DQN shows a small, new difference (8.3% [0.0%,25.0%] vs original 0/36). Written into paper5_wpc/main.tex as a new subsection extending §8.3 (new Fig. 9), committed (commit 941ff07) along with docs/PAPER5_M27_scaling_reframe.md. Raw omega logs (4.4GB) deliberately left uncommitted -- matches this project's own precedent that M6's original N=19 campaign was never bulk-committed either; a compact derived_correctness_metrics.json preserves every per-seed number instead.
- N=7 campaign completed (~2.12h). Genuinely new finding, not just a replication: independent DQN replicates cleanly (0/36); GAT-CTDE's N=7 collapse rate (33.3%) now matches its own N=19 rate almost exactly; single-agent DQN collapses in 18/36 cells (50.0%), directly contradicting M6's original n=3 pilot ("holds 1.000 at N=7"). Cannot cleanly attribute this to the recalibration vs. the original pilot simply being too small to see a ~50% rate -- reported as an open, honest attribution question, not resolved one way. Written into the manuscript (commit 89bd60b) as an extension of §8.4, with a forward-reference added at §7's own "never fully collapses (0/36)" claim to avoid an unflagged internal contradiction. Fig. 9 is now 2 panels (N=7, N=19). **M27 is fully complete.**
- M28: prep complete, checkpoint candidate validated (commit f312a9d). Remaining work is only the live execution step, held for the user's return.
- All work through this point committed: f312a9d (M28 prep), 941ff07 (M27 N=19), d6be488 (scope doc update), 89bd60b (M27 N=7). Manuscript compiles clean throughout (25 pages, 0 errors, 0 undefined refs as of 89bd60b).
- **gNB2 E2-port blocker resolved.** Patched e2_agent_app.h (E2AGENT_IN_PORT 6655->6656, E2AGENT_OUT_PORT 6600->6601), rebuilt nr-softmodem, saved the result as the new nr-softmodem-gnb2 (removing the old, ambiguous-port one from M26), reverted the header (confirmed byte-identical to original via git diff), rebuilt nr-softmodem again for gNB1 and confirmed it byte-identical to the preserved nr-softmodem-gnb1-orig reference copy. Compile-only throughout -- no live gNB was started. saclb_live2gnb.yaml's header updated to reflect these as confirmed, not placeholder, values. **M28's only remaining blocker is now genuinely just "the user needs to be present for the first live 2-gNB run" -- nothing else is outstanding.**

## First live 2-gNB attempt (2026-08-28): real registration success, real load failure -- honest result, not yet a working demo

With the user present, attempted the actual live run. Two new, real technical findings on the way to a genuine 2-gNB registration, both now fixed and documented for next time:

1. **gNB2 needs a real host-level IP, not just a "free" docker-range address.** `GNB_IPV4_ADDRESS_FOR_NG_AMF=172.22.0.3/24` in gnb2.conf failed outright (`sctp_bindx() ... errno 99 Cannot assign requested address`) because gNB2 is a native host process, like gNB1, and the host only owned 172.22.0.1 (the bridge's own gateway address) -- .3 was free inside the docker network's address space but not actually assigned to any interface the host could bind to. Fixed by adding it as a secondary address on the same bridge interface (`sudo ip addr add 172.22.0.3/24 dev demo-open5gs`) -- additive, reversible, removed again during teardown.
2. **A second UE in the host namespace collides with the first.** OAI's UE TUN device name (`oaitun_ue1`) is hardcoded per-process regardless of which gNB it talks to -- this project already knew this required netns isolation for multiple UEs on ONE gNB (BRINGUP_LOG.md), but it turns out to apply identically across gNBs too: launching gNB2's own UE (UE7) directly in the host namespace silently reconfigured/corrupted gNB1's own already-working UE1 interface. Fixed the same way as ever: UE7 needed its own netns (`ue7ns`), reaching gNB2's rfsim server via the veth's host-side address, exactly the established UE2-6 pattern, just pointed at gNB2 instead of gNB1.

With both fixed, genuine success followed: gNB2 registered ("Number of gNBs is now 2"), UE7 attached to it specifically (confirmed via gNB2's own RRC log, a distinct RNTI, real RSRP), and both UEs showed 0% packet loss with real traffic flowing on both simultaneously -- the first time this rig has ever run two complete, independent live radio links at once.

It did not hold under sustained load. The user reported real, severe slowdown (not just a number on a dashboard); investigation found genuine iowait spikes and swap activity (the established signature this project treats as real trouble, not mere tightness) once both UEs' traffic was flowing. Tried capping the RF processes' CPU via a cgroup (`oranslice-rf.slice`) -- first attempt used `CPUQuota` (CFS bandwidth control) and had **no effect at all**, because ~21 of gNB1's own threads run under `SCHED_RR` (real-time scheduling), which cgroup v2's `cpu.max` does not govern; real-time threads bypass it entirely. Switched to `cpuset` (`AllowedCPUs`), which restricts cores regardless of scheduling class and did measurably work (confirmed via `mpstat -P ALL` and raw `cpu.stat` deltas), first at 7/8 cores then 6/8 cores per the user's own live feedback as the perceived slowdown continued. Even at 6/8 cores, UE7 then UE1 each independently developed 100% packet loss while their underlying processes stayed alive -- a real link-level failure, not merely reduced throughput. A full restart of just gNB2+UE7 (leaving gNB1/UE1 running) recovered UE7 briefly, but the same failure mode reappeared and then spread to gNB1/UE1 too. Aborted and tore down cleanly on the user's confirmation, matching this project's standing practice for real (not merely perceived) trouble.

**Honest conclusion, not smoothed over:** this rig's confirmed 2-gNB stability (M26) was established for concurrent *registration and build load*, not for two complete live radio links each carrying real per-slice traffic simultaneously -- a genuinely heavier combination this session did not previously exercise. Two genuinely reusable fixes came out of this regardless (the bridge IP requirement, the cross-gNB UE netns requirement) and are captured above and in the relevant config/script comments for the next attempt.

### Root-cause follow-up: likely a genuine hardware ceiling, not a fixable config mistake

Investigated further (offline, reading OAI's own source, no live hardware) rather than leave "raw CPU vs. something else" open. Two findings, read together, point to a specific, well-supported conclusion:

1. **Not a hardcoded-core-collision bug.** Read `thread_top_init()` (`common/utils/system.c`) and its actual call sites: despite the comment "CPU 0 is reserved for UHD threads, CPU 1 is reserved for all RX_TX threads," the function's own body never actually calls `pthread_setaffinity_np` with the `affinity` argument it's passed -- that argument is dead for pinning purposes (only used in a log string built from the THREAD's already-inherited mask). The generic `threadCreate()` path (used for e.g. the E2 agent's own TASK_SCTP/TASK_NGAP threads, confirmed in tonight's own console logs) passes `affinity=0xffffffff` -- an explicit "no pinning, full mask" sentinel, not a specific colliding core number. So gNB1 and gNB2 are not fighting over the SAME hardcoded core indices; ruled this out directly from source, not assumed.
2. **This CPU is 4 physical cores, not 8.** `lscpu` confirms `nproc`'s reported "8" is 4 physical cores x 2 hardware threads (Intel i5-1135G7, mobile Tiger Lake). For compute-intensive, latency-sensitive DSP work like real-time RF signal processing, two hyperthreads sharing one physical core contend heavily for the same execution units and cache -- "8 logical cores" was never really 8 independent units of DSP throughput. OAI's own design assumption (dedicate specific cores to its most timing-critical `SCHED_FIFO`-at-maximum-priority threads, per the comment above) implicitly expects a machine with several cores it can have mostly to itself; it does not need to hardcode which ones if it's the only such workload running.

Put together: running TWO complete, independent, maximum-RT-priority RF stacks simultaneously -- each individually already designed around "give me dedicated cores" -- on a 4-physical-core mobile laptop CPU is a genuinely demanding configuration, independent of any scheduling or cgroup arrangement. This also explains why the cpuset restriction made things WORSE rather than better: squeezing the same real-time thread demand onto fewer logical cores increases contention per physical core, it does not relieve it, and the kernel's own global RT bandwidth cap (`sched_rt_runtime_us=950000/1000000`, confirmed via `/proc/sys/kernel/`) reserves only 5% non-RT headroom PER CORE regardless of how many cores are made available to the cgroup.

**Revised, honest recommendation for the next attempt:** do not apply CPU-affinity/cpuset restriction to the gNB processes at all -- give them the full, unrestricted logical-core pool, since restricting it demonstrably worsens real-time contention rather than protecting headroom. If a live 2-gNB demo is still wanted, the more promising path is reducing REAL-TIME DEMAND rather than reallocating cores: fewer UEs (down to the minimum 1 per gNB, already tried), lower-bitrate or intermittent traffic instead of continuous 4Mbps streams on both links simultaneously, or accepting that this rig's genuine ceiling for sustained live demonstration is coordinated E2 control/registration across 2 gNBs (already proven, M26 and again tonight) rather than two complete live RF links under real sustained load at once -- which would itself be a legitimate, reportable hardware-capacity finding rather than an unresolved failure, consistent with this project's own standard of reporting a genuine limit honestly rather than treating it as a bug still to be found.
