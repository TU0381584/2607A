# M35: unified metric-disagreement ledger

Renumbered from the reviewer's own "M28" (see below) as part of a
4-milestone plan an external Claude Web review recommended for WPC:
M35 (this doc, formerly "M28") -> M36 (formerly "M29", sim-to-real
congestion characterisation, rig) -> M37 (formerly "M30", recalibrated-
sim generalisation to an unseen live load, rig) -> M38 (formerly "M31",
properly-powered live correctness campaign, rig). Gated and sequenced
per the reviewer's own instruction: stop at each gate, await go before
the next.

**Renumbering reason:** this project already has an M28 (the aborted
live 2-gNB GAT-CTDE demo, 2026-08-28, `docs/PAPER5_M27_M28_scope.md`)
and M32-M34 (later live-fix work). The reviewer's plan reused M28-M31
for unrelated work; reusing those numbers here would create two
unrelated "M28"s in the doc trail. User confirmed renumbering to
M35-M38 (2026-08-31).

## Scope (M35 only, no rig, re-analysis)

Enumerate every paired arm/condition comparison already reported in
M2 (centralised), M3 (federated+DP), M4 (disruption), M6 (topology
scaling), and recompute each under three metrics from the SAME already-
saved eval logs: `sla_compliance_all_slices`, the campaign's own
correctness-aware reward metric, and `block_precision`. Flag where
compliance disagrees with the other two in direction or significance
verdict. No retraining, no checkpoint access, no new campaigns.

## Method

New script: `experiments/scripts/m35_metric_disagreement_ledger.py`.
Reuses, does not reimplement, the project's own existing metric
primitives:
- `m2_correctness_metrics.per_seed_metrics` / `bootstrap_ci` (M2, M3)
- `m4_correctness_metrics.per_seed_metrics_normalized` (M4 -- divides
  reward by that step's own `n_pending`, the fix that already exists
  for demand-volume confounds under disruption)
- `m6_correctness_metrics.per_seed_metrics_per_gnb` (M6 -- divides
  reward by cluster size, the fix that already exists for the N
  confound)

`sla_compliance_all_slices` is recomputed directly from each eval log's
own `episode_sla_compliance_all_slices` rollup field (mean across
episodes), rather than read from each campaign's cached
`campaign_results.json` / `m6_results.json`. This was a deliberate
choice made after finding M6's own cache (`m6_pilot/n19_hex_capfix/
m6_results.json`) only has `gat_ctde` entries cached for some combos,
missing `independent_dqn`/`single_agent_dqn` despite their raw logs
being present on disk -- computing every campaign's compliance the same
way, directly from the logs, avoids that inconsistency rather than
mixing cached and recomputed values across campaigns.

A comparison is flagged **disagree** if compliance's direction (sign of
the mean paired Wilcoxon difference) or significance verdict (p<0.05)
does not match the reward metric's and/or block_precision's. Two kinds
are distinguished, since they carry very different evidential weight:
- **significance_flip**: one metric significant, the other not -- the
  strong form, a real finding.
- **direction_only**: both metrics non-significant but point opposite
  ways -- the weak form, expected sampling noise at small n, not the
  same strength of claim.

## GATE M35 report

Output: `experiments/results/m35_metric_disagreement_ledger.csv` (49
rows: 25 scored comparisons, 24 gaps).

**Compliance disagreed with the correctness-aware pair in 23 of 25
comparisons (92.0%).**
- 15 of those 23 are a **significance flip** (the strong form).
- 8 of those 23 are **direction-only** splits between two
  already-non-significant results (the weak form, small-n noise, mostly
  M6's N=7 topologies at n=3 seeds).

Concrete examples of the strong form, cross-checked against numbers
already published in this project's own docs to validate the script
before trusting its output:
- M2, gat_ctde vs independent_dqn: compliance p=0.067 (not
  significant) vs reward p=2.7e-05 (highly significant) -- reward
  direction/magnitude here (+, large) matches
  `docs/PAPER5_M2_gat_ctde.md`'s own corrected finding "p<0.0001"; no
  compliance-based version of this specific pair existed anywhere in
  the repo before this ledger.
- M4, gat_ctde and independent_dqn, all 6 dropout/churn severities:
  reward and precision both find the disruption cost significant
  (p<=0.003) in every case; compliance finds it significant in only 1
  of 6 (gat_ctde dropout_sev3, p=0.031). independent_dqn's churn
  reward p-values here (0.064-0.105) land almost exactly on the
  ORIGINAL, since-retracted range `docs/PAPER5_M4_disruption.md`
  reports before its independent-seed replication overturned it
  (p=0.0645-0.1055) -- expected, since this ledger deliberately uses
  the same official 900-909 seeds the original (pre-retraction) finding
  used, not the replication batch.
- M6, gat_ctde vs single_agent_dqn at N=19, all three topologies:
  compliance is significant (p=0.007-0.015) while the per-gNB-normalized
  reward is not (p=0.74-0.91) -- this is the *opposite* pattern from
  M2/M4 (here compliance claims a difference the correctness-aware
  metric doesn't support) and matches this project's own already-
  published conclusion that N=19's compliance metric trends toward 0
  for every arm as a structural artifact, while the real signal lives in
  block precision/collapse rate, not reward margin
  (`docs/PAPER5_M6_topology.md`, quoted in the M35 scoping report).

**Gap list (24 comparisons, M4 only):** spike (all 4 arms) and
single_agent_dqn / fl_gat_ctde_sigma0.0 (dropout, churn) at the
official 900-909 seeds have `sla_compliance_all_slices` only --
`experiments/results/m4_campaign/`'s raw per-seed `omega_log.jsonl`
files no longer exist on disk (only the aggregated
`campaign_results.json` survives; git history shows only that JSON was
ever committed for this path). A different, independently-drawn seed
batch (1000-1009) exists with full arm/kind coverage under
`experiments/results/fresh_seed_retrain/m4_campaign/`, but is not
git-tracked and was NOT substituted into this ledger, per this
milestone's own instruction not to recompute from a different sample
without flagging it -- doing so silently would repeat exactly the
seed-set-swap mistake `docs/PAPER5_M4_disruption.md`'s own churn
retraction already documents (same direction, different verdict, across
900-909 vs 1000-1029).

## Status

Awaiting go before M36 (sim-to-real congestion characterisation --
requires the live rig).
