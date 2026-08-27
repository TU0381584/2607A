# Paper #5 M27: offline scaling reframe

Status: **complete for N=19 (the headline case). GAT-CTDE and
single-agent DQN's collapse rates are robust to the M34 recalibration;
independent DQN shows a small, previously-unobserved collapse rate the
original simulator's narrower conditions never surfaced. N=7 not yet
run (optional extension, not required to answer M27's question).**

## What M27 asks

M32-M34 (this session, commits 78e11a8/bc36bae/639ec89/43a60b5) found
that M6's own topology-scaling campaign (`docs/PAPER5_M6_topology.md`)
was run entirely against `ClosedLoopKpmSource`, an offline simulator
whose congestion_level is structurally capped near 0.03-0.09 (served
PRB is bounded by the admission ceiling, which the trained policy keeps
far below real demand) -- while live measurement on this rig shows real
congestion at 0.23-0.26 (3 UEs) and 0.60-0.69 (6 UEs). M27 asks whether
M6's own findings -- single-agent DQN collapses totally at N=19 (15/15
across two samples), GAT-CTDE collapses partially (pooled estimate
35.4% [25.5%, 45.8%]), independent DQN never fully collapses (0/36) but
never cleanly differentiates either (0.29-0.49 precision band) -- were
an artifact of that miscalibration, or hold up once each simulated
gNB's congestion actually resembles this rig's own measured live range.

## Method

`experiments/scripts/m27_scaling_reframe.py` reruns M6's exact
`m6_run_experiment.py` orchestration (same CLI, same `--config-path`,
same `--topology`, same resumability) with one substitution:
`RealisticServedKpmSource` (M34, already generic over `gnb_ids`, so no
change was needed to extend it from N=1 to N=19) instead of
`ClosedLoopKpmSource`, via a monkeypatch of
`m6_run_experiment.make_kpm_source_factory` -- not a file edit, and not
a reimplementation of the training/eval loop, resumability, or
results-writing, all of which are reused unchanged. Each simulated gNB
reuses the SAME single-cell served-PRB anchors M34 measured on the real
rig (not a new, invented multi-gNB number) -- the same already-validated
per-cell calibration M6 itself already applies per-gNB via
`gnb_load_multiplier` for heterogeneity.

Primary sample: 12 seeds (900-911, matching M6's own primary sample
exactly for direct comparability), 3 topologies (fully-connected, ring,
hex), 3 arms, N=19, full 100-train/20-eval episode budget. Total
wall-clock: 13,945s (~3.87h) for all 108 (arm, topology, seed) cells.

## Result: two of three arms robust, one shows a real, small difference

| Arm | Original M6 (ClosedLoopKpmSource) | M27 recalibrated (RealisticServedKpmSource) |
|---|---|---|
| single-agent DQN | 15/15 collapsed (100%) | 33/36 collapsed, 91.7% [75.0%, 100%] |
| GAT-CTDE | 35.4% [25.5%, 45.8%] (pooled, 3 samples) | 33.3% [11.1%, 58.3%] (12 seeds x 3 topologies) |
| independent DQN | 0/36 collapsed (never) | 3/36 collapsed, 8.3% [0.0%, 25.0%] |

Collapse rate computed identically to M6's own convention: bootstrap
over the 12 independent seeds (not the 36 cells), since collapse status
correlates within a seed across its three topologies -- confirmed
directly here too: `independent_dqn`'s and `single_agent_dqn`'s results
are byte-identical across all three topologies for the same seed (they
never consume adjacency, so nothing about the environment's own random
seed sequence differs when only `--topology` changes), which is not a
bug, it is exactly the expected behaviour for arms that never read a
graph structure.

**GAT-CTDE's and single-agent DQN's collapse rates are statistically
indistinguishable from M6's original findings** -- both CIs overlap
extensively with the original estimates, and the non-collapsed GAT-CTDE
seeds show the same qualitative pattern M6 originally reported (mostly
near-ceiling precision, with a low-precision or exactly-zero-precision
seed appearing at every topology). This means M6's headline finding
-- GAT-CTDE fails less often than single-agent DQN, GAT-CTDE's collapse
rate has real seed-to-seed variability, non-collapsed seeds are not
uniformly reliable -- was not an artifact of the original simulator's
miscalibrated congestion range. It holds under a corrected environment
whose congestion matches this rig's own live measurements.

**Independent DQN is the one place the recalibration surfaces something
new.** M6's original claim (0/36, "never fully collapses") does not
survive unchanged: 3/36 cells here show a genuine, total collapse (0
blocks). The rate is still small (8.3%, CI includes 0), and the CI on
this recalibrated estimate does not exclude M6's original 0% -- so this
is not a contradiction at the current sample size, but it is a real,
observed difference in the raw counts, reported here rather than
smoothed over. The most direct reading: independent DQN's resistance to
total collapse is real but not absolute, and the original simulator's
narrower congestion range happened not to surface the rare cases where
it fails, the same way it hid the live collapse M32-M34 diagnosed for
single-agent DQN specifically.

## What this means for the paper

This is a validating result, not a retraction: M6's own topology-scaling
section does not need to be walked back. The recalibration was worth
doing precisely because it could have gone either way -- and for the
paper's two headline comparison points (GAT-CTDE vs. single-agent DQN),
it did not change the story. The one genuine update worth making is
softening independent DQN's "never fully collapses" claim to "rarely,
not never" -- a small, honest correction, not a different finding.

## What was not done

- N=7 (M6's replication/secondary scale) was not rerun under the
  recalibrated environment -- N=19 is the headline case this project's
  own reporting emphasizes, and answering M27's core question (does the
  recalibration change M6's story) did not require it. A natural
  extension if more confidence at a second scale is wanted later.
- This is entirely an offline result. No live multi-gNB claim is made
  or implied here (that is M28's separate, still-pending question) --
  this rig has one physical gNB in its default configuration, and the
  2-gNB capability M26 verified is not exercised here at all (M27 is
  pure offline compute).
