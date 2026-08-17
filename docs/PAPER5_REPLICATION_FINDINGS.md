# Paper #5: independent-seed replication findings (M2-M4)

Full run: `experiments/scripts/reproduce_paper5_fresh_seeds.sh`, seeds
1000-1029 (M2) / 1000-1009 (M3, M4) -- disjoint from the committed
900-929/900-909 range, torch-seeding fix in place throughout (see
`docs/PAPER5_REPRODUCIBILITY.md`'s torch-seeding-gap section; this run
was launched *after* that fix, so it is itself fully seed-reproducible).
30-seed M2 campaign + 10-seed x 5-sigma M3 sweep + 330-cell M4 campaign,
~8.6 hours wall-clock, zero errors, `compare_reproduction.py`-style
cell counts all present.

This document is not asking "does the same seed reproduce the same
number" (`docs/PAPER5_REPRODUCIBILITY.md` already answers that: yes,
after the torch fix). It asks the harder, more valuable question: does
the paper's *statistical story* hold up under a completely independent
sample of seeds the analysis was never tuned against? Answer: **mostly
yes, with one specific claim that does not survive and has been
retracted** (independent DQN's supposed immunity to agent churn --
`docs/PAPER5_M4_disruption.md`'s RETRACTED note has the full account).

## M2: both headline reward comparisons replicate

| Comparison | Committed (seeds 900-929) | Fresh (seeds 1000-1029) |
|---|---|---|
| GAT-CTDE vs. independent DQN | +1.461 [0.550,2.782], p<0.0001, 27/0/3 | +0.767 [0.390,1.233], p<0.0001, 29/0/1 |
| GAT-CTDE vs. single-agent DQN | +0.195 [-0.017,0.442], p=0.0577, 20/3/7 | +0.445 [-0.049,1.279], p=0.1109, 20/2/8 |
| Differentiated seeds (any block) | 21/30 | 23/30 |
| Of those, mmtc-only blocking | 20/21 | 23/23 |

Both of the paper's actual claims replicate cleanly: GAT-CTDE decisively
beats the independent-DQN ablation (p<0.0001 in both samples, effect
size differs but same order of magnitude and direction, 29/30 and 27/30
seeds favor it), and its edge over single-agent DQN specifically remains
small and short of conventional significance in both samples (p=0.058
and p=0.111) -- the SAME win/tie/loss split almost exactly (20 wins
both times). The differentiated-shedding rate (how many seeds learn
genuine, correctly-targeted mmTC blocking) is close between samples
(21/30 vs. 23/30) and, notably, the fresh sample's differentiated seeds
are 100% mmtc-only (23/23) versus the committed sample's 20/21 -- if
anything, a cleaner result, not a weaker one.

## M3: federation-cost finding replicates; privacy threshold's exact location does not

| Quantity | Committed (seeds 900-909) | Fresh (seeds 1000-1009) |
|---|---|---|
| Federation cost (paired diff) | +0.133 [-0.138,0.470], p=0.8125, 2/5/3 | -0.080 [-0.241,0.001], p=1.0000, 1/8/1 |
| block_precision, sigma=0.0/0.5/1.0/2.0/4.0 | 1.000/1.000/1.000/0.834/0.584 | 1.000/0.876/0.876/0.876/0.637 |

**Federation cost replicates decisively**: both samples show
essentially no measurable cost to federating (p=0.81 and p=1.00, both
dominated by ties -- 5/10 and 8/10 seeds tie exactly). This is now
confirmed under two independent samples, not one.

**The privacy threshold's existence replicates; its exact location
does not.** Both samples show the same qualitative shape -- precision
starts at or near 1.000 and degrades as sigma increases, ending
substantially degraded by sigma=4.0 (0.584 vs. 0.637, close). But WHERE
the degradation starts differs: the committed sample holds perfect
precision all the way through sigma=1.0 and only drops at sigma=2.0;
the fresh sample starts dropping already at sigma=0.5. Both are real,
directly-measured results (not a bug -- verified the fresh sample's
bit-identical values across sigma=0.5/1.0/2.0 trace to the same
seeded-noise-generator-plus-discrete-decision-boundary mechanism already
validated for the committed data, not a new anomaly). **The honest
claim is "a threshold exists," not "the threshold is at sigma=2.0"** --
the paper should not pin an exact sigma value as if it were a universal
property of the architecture; it is specific to which 10 seeds happened
to be drawn.

## M4: dropout and the coordination-dependent arms' churn cost replicate; independent DQN's churn "immunity" does not

Dropout: every arm, every severity, both samples -- significant
(p<=0.03), accelerating with severity (not linear). This is the paper's
main M4 claim and it replicates cleanly; see
`docs/PAPER5_M4_disruption.md`'s dropout table plus the fresh numbers
below for the side-by-side.

| Arm | Committed churn sev1/sev2/sev3 (cost, p) | Fresh churn sev1/sev2/sev3 (cost, p) |
|---|---|---|
| GAT-CTDE | 0.059(p=.004) / 0.200(p=.004) / 0.529(p=.004) | 0.098(p=.004) / 0.309(p=.004) / 0.820(p=.004) |
| Federated | 0.073(p=.004) / 0.240(p=.004) / 0.585(p=.004) | 0.083(p=.004) / 0.262(p=.004) / 0.706(p=.004) |
| Independent DQN | 0.000(p=.084) / 0.168(p=.106) / 0.640(p=.065) | 0.097(p=.002) / 0.657(p=.002) / 1.223(p=.002) |

GAT-CTDE and the federated arm's own churn costs replicate closely in
both direction and magnitude. **Independent DQN's churn cost does NOT
replicate as "not significant"** -- the fresh sample shows it highly
significant at every severity (p=0.002, 10/10 seeds hurt), and its
*magnitude* in the fresh sample is larger than GAT-CTDE's or the
federated arm's at matched severities (1.223 vs. 0.820/0.706 at sev3),
the opposite of "immune." The direction was always consistent across
both samples (churn hurts, never helps) -- what changed is that the
committed sample's borderline p-values (0.065-0.106, n=10) got written
up as a confirmed architectural property in an earlier pass of
`docs/PAPER5_M4_disruption.md`. That was sample noise, not a finding,
and the write-up there now has an explicit RETRACTED note rather than a
silent edit.

Spike: both samples show `block_precision` flat and near-ceiling
through the highest tested multiplier (no threshold, matching the
paper's own framing that spike is a "genuinely different, cleaner
story" than dropout/churn). The normalized-reward "spike helps" signal
replicates for GAT-CTDE, independent DQN, and the federated arm
(p<=0.014 in both samples) but is noticeably weaker for single-agent
DQN in the fresh sample (p=0.084, not significant, versus the committed
sample's p=0.002 at every severity) -- direction consistent, but this
specific arm's significance is sample-dependent and should be reported
as such if it appears in a paper figure/table, not stated as
unconditionally significant for all four arms.

## What this means for paper #5

The paper's actual planned M4 claims (dropout is a real, accelerating,
threshold-like cost across arms; federation is free; a real privacy
threshold exists) are now backed by two independent seed samples, not
one -- report them with that strength. Do NOT write the independent-DQN
churn-immunity claim into the paper in any form; it was retracted before
ever reaching `main.tex` (M4's own section was deliberately not drafted
into the paper yet, per the original M4 plan's scope boundary -- this
replication check caught the error while it was still cheap to fix).
If a privacy-threshold sigma value is quoted in prose, hedge it
explicitly ("held through at least sigma=1.0 in one 10-seed sample,
degraded starting as early as sigma=0.5 in another" or similar) rather
than stating one number as if it generalizes. If single-agent DQN's
spike behavior is reported, note the significance is sample-dependent
rather than presenting it as uniformly significant.
