# Paper #5 M27: offline scaling reframe

Status: **complete for both N=19 and N=7. At N=19, GAT-CTDE and
single-agent DQN's collapse rates are robust to the M34 recalibration;
independent DQN shows a small, previously-unobserved collapse rate. At
N=7, the picture is more interesting: GAT-CTDE's collapse rate is now
statistically indistinguishable from its own N=19 rate, and
single-agent DQN -- which M6's original n=3 pilot reported as fully
reliable at N=7 -- collapses half the time in this properly-powered
12-seed resample. Reported honestly below, including what can and
cannot be attributed to the recalibration versus the original pilot
simply being too small to see this.**

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
hex), 3 arms, full 100-train/20-eval episode budget, run at both N=19
(13,945s, ~3.87h) and N=7 (7,621s, ~2.12h) -- 216 (arm, topology, seed)
cells total.

## Result at N=19: two of three arms robust, one shows a real, small difference

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

## Result at N=7: the original "clean" story does not survive a properly-powered resample

M6's own N=7 number was never a large sample: a 3-seed pilot
(`docs/PAPER5_M6_topology.md`), explicitly never resampled at scale
because that document's own judgement was that N=19's collapse-rate
question was the higher-value target ("everything else in M6's original
scope... already has a fairly clear, if modest, answer from what has
run so far"). In that pilot, GAT-CTDE and single-agent DQN both held
1.000 precision (0/3 collapsed each) and independent DQN sat at 0.787
precision (also 0/3 collapsed) -- point estimates only, no CI ever
reported for N=7 specifically.

| Arm | Original M6 ($n{=}3$ pilot, point estimate) | M27 recalibrated (12 seeds x 3 topologies) |
|---|---|---|
| single-agent DQN | 0/3 collapsed (0%) | 18/36 collapsed, 50.0% [25.0%, 75.0%] |
| GAT-CTDE | 0/3 collapsed (0%) | 12/36 collapsed, 33.3% [8.3%, 58.3%] |
| independent DQN | 0/3 collapsed (0%) | 0/36 collapsed, 0.0% [0.0%, 0.0%] |

Independent DQN replicates cleanly (still never collapses, now with 12x
the sample). GAT-CTDE's N=7 collapse rate (33.3%) is now statistically
indistinguishable from its own N=19 rate (33.3%) -- collapse-proneness
at this scale looks like a property of the architecture under real
congestion, not something that only emerges once the cluster gets large.

**Single-agent DQN is the genuinely new finding here.** The original
pilot's "holds 1.000 at N=7" does not survive a properly-powered
resample: half of the 36 cells here show complete collapse. This cannot
be cleanly attributed to the recalibration alone versus the original
pilot simply being too small ($n{=}3$) to ever see a real ~50% rate --
both are true simultaneously and cannot be disentangled without an
equally-sized 12-seed `ClosedLoopKpmSource` resample at N=7, which this
session did not run (it would not have answered M27's actual question,
which is about the recalibration, not about re-deriving M6's original
numbers at higher power). What this result does establish, without that
missing comparison: single-agent DQN's collapse tendency is not an
N=19-specific phenomenon under the recalibrated, live-congestion-matched
environment -- it is already substantial at N=7.

## What this means for the paper

Mixed, and reported as such rather than smoothed into one verdict. At
N=19 -- the scale carrying this paper's actual comparative claims --
GAT-CTDE's and single-agent DQN's collapse rates are statistically
unchanged by the recalibration, so the paper's headline comparison does
not need to be walked back. At N=7, the recalibration (or the larger
sample it came with -- both are true) reveals real complexity the
original 3-seed pilot could not: single-agent DQN is not reliably safe
at small scale either, and GAT-CTDE's collapse-proneness is present
from N=7 onward, not something that appears only at large N. Two
concrete updates worth making to the paper: independent DQN's "never
fully collapses" claim softens to "rarely, not never" (both N=7 and
N=19 now show this consistently: 0/36 and 3/36), and any implication
that N=7 is a "safe," collapse-free scale for single-agent DQN should be
removed -- it was never well-supported, only under-sampled.

## What was not done

- An equally-sized (12-seed) `ClosedLoopKpmSource` resample at N=7 was
  not run, so the N=7 finding above cannot cleanly separate "the
  recalibration changed the dynamics" from "the original 3-seed pilot
  was always too small to see this" -- both contribute, in unknown
  proportion. Flagged explicitly above rather than picking one
  explanation without the data to support it.
- This is entirely an offline result. No live multi-gNB claim is made
  or implied here (that is M28's separate, still-pending question) --
  this rig has one physical gNB in its default configuration, and the
  2-gNB capability M26 verified is not exercised here at all (M27 is
  pure offline compute).
