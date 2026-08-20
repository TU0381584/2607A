# Paper #5: why the M3 privacy threshold's exact location shifts between seed samples

Status: **complete. Answered, not left open. Written into paper5/main.tex
(Section IX and Conclusion).**

## What this asks

Section IX's independent-seed replication found block precision's
degradation under DP noise is real in both the primary (900-909) and
replication (1000-1009) samples, but the exact noise level ($\sigma$)
it first appears at differs: perfect through $\sigma=1.0$ in the
primary sample, already degrading at $\sigma=0.5$ in the replication
one. The paper flagged this as an open question: does that reflect
genuine sensitivity to which ten seeds are drawn, or a factor not yet
identified?

## Method

M3's raw per-seed eval logs (needed to look at each checkpoint's own
trajectory across the $\sigma$ sweep, not just the pooled precision
number) were cleaned up earlier as disposable working data (only
`checkpoint.pt` + `campaign_results.json` were ever tracked for
`m3_campaign`, matching m2/m4's convention). Regenerated them
eval-only from the already-committed checkpoints
(`experiments/scripts/m3_eval_only_from_checkpoint.py`, no training, no
DP noise reapplied since noise only ever affects training-time
gradients, not eval-time greedy action selection) -- smoke-tested (1
seed, 5 episodes) before the full run, checkpoint file verified
byte-identical afterward. All 5 sigma levels, seeds 900-909, 50
episodes each -- 50 cells, all present, none missing. The replication
sample's raw logs (seeds 1000-1009, part of the wider 1000-1029
`fresh_seed_retrain` set) were already intact on disk.

For every seed, at every $\sigma$, read `total_blocks` (0 = fully
collapsed that run) directly from the regenerated/existing eval logs
(`m2_correctness_metrics.per_seed_metrics`, reused, not reimplemented).

## Result

**Per-seed block counts are not a monotonic function of $\sigma$.**
Example, primary sample:

| seed | $\sigma$=0.0 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|
| 900 | 0 | 0 | 0 | 3506 | 3506 |
| 902 | 3468 | 3468 | **0** | 3468 | **0** |
| 906 | 3488 | 3488 | **0** | 3488 | 6911 |
| 909 | 3507 | **0** | 3507 | 469 | 3507 |

Seed 900 collapses at low $\sigma$ and recovers at high $\sigma$; seed
902 collapses at $\sigma=1.0$ specifically but not at the $\sigma=2.0$
level right next to it; seed 909 collapses only at $\sigma=0.5$ and is
fine everywhere else, including higher noise levels. This is not
noise -- **checked directly, not assumed**: checkpoint files across
sigma levels for the same seed are confirmed genuinely different
(distinct MD5 hashes, ruling out a duplication/symlink bug before
trusting the pattern as real), and each $\sigma$ level's training run
uses its own independently-sampled Gaussian noise draw (DP-SGD adds
fresh noise every gradient step; $\sigma$ scales the noise's magnitude
but each level's actual draw is statistically independent of every
other level's, not a nested or cumulative sequence).

**A handful of seeds are robust across the entire tested range**
(replication seeds 1003, 1006, 1008: identical, non-collapsed block
counts at every one of the 5 sigma levels) **while most are not**,
collapsing at some seemingly-arbitrary subset of levels and not others.

## Conclusion

The "threshold location" a pooled, single-sample analysis reports is
an **order statistic**, not a population parameter: it is the minimum
collapse-onset point across whichever ten independent noise
realisations that specific sample happened to draw. Two disjoint
ten-seed samples will, by the ordinary logic of order statistics over
a noisy, non-monotonic per-seed process, generally report different
"first sign of degradation" points even if both are honest, correctly-
computed readings of the same underlying (noisy) generative process --
exactly what was observed. This confirms the paper's own speculation
("genuine sensitivity to which ten seeds are drawn") directly, with
seed-level mechanistic evidence, rather than leaving it as an
unconfirmed possibility. It also means a "true" threshold location,
in the sense of a fixed $\sigma$ value below which every seed is
always safe, may not be a well-posed quantity for this system at
all -- the honest description is a noisy, seed-dependent collapse
process whose *population-level* degradation is real and monotonic in
expectation (Section IX's own mean-precision curve shows this clearly)
even though no individual seed's trajectory is.

## What was not done

- Did not attempt to characterise the DISTRIBUTION of collapse-onset
  points more precisely (e.g. fitting a survival curve across many more
  seeds) -- the qualitative mechanism (non-monotonic, seed-and-noise-
  realisation-dependent collapse) is the answer the flagged question
  asked for; a full distributional characterisation would be new scope
  beyond what was asked.
- Did not investigate WHY some specific seeds (1003, 1006, 1008) are
  robust across the whole range while most are not -- a real, secondary
  question this investigation surfaced but did not chase further.
