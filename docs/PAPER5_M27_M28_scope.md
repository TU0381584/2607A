# Paper #5 M27/M28: scope note

Status: **M27 complete and written into the manuscript for N=19 (see
docs/PAPER5_M27_scaling_reframe.md); N=7 extension launched, running.
M28 preparatory work complete (configs, orchestration script, a
validated candidate checkpoint); live execution deliberately held for
the user's return.** The original scoping message for M27 ("offline
scaling reframe") and M28 ("live multi-gNB demo") was a one-line
description each in an earlier pasted instruction block; the fuller
text did not survive context compaction. This doc records the
interpretation this session is proceeding on, so it can be corrected on
review rather than silently assumed correct.

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

## M28: live multi-gNB demo -- interpretation and a real constraint found

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
- N=7 extension launched (same script, same 12 seeds, 3 topologies, 3 arms; ~2h estimated). Original M6 baseline at N=7 is a much smaller n=3 pilot sample (GAT-CTDE 1.000, single-agent DQN 1.000, independent_dqn 0.787, all topologies) -- M6's own writeup explicitly deprioritized a larger N=7 resample in favor of resolving N=19's collapse-rate question, so this is a genuinely new, more powered measurement at N=7, not a like-for-like large-sample comparison the way N=19 was. Report this honestly when writing it up -- do not imply the original N=7 number carries the same statistical weight N=19's pooled estimate does.
- M28: prep complete, checkpoint candidate validated (commit f312a9d). Remaining work is only the live execution step, held for the user's return.
