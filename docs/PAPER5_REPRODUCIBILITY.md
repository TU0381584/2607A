# Paper #5 reproducibility appendix (M1-M4)

Same purpose as `docs/REPRODUCIBILITY.md` (paper #4's own audit trail),
for paper #5: for every number in `paper5/main.tex` and every milestone
doc's headline finding, the exact script/config/seed that produced it,
plus a single command that reproduces the entire M1-M4 process from a
fresh checkout end to end.

## One-command reproduction

```bash
experiments/scripts/reproduce_paper5_full.sh [OUT_ROOT]
```

Runs M1 -> M2 -> M3 -> M4 sequentially into an isolated output root
(default `experiments/results/reproduction_check/`) that never touches
or overwrites the committed `experiments/results/{m1_recalibration,
m2_campaign,m3_campaign,m4_campaign}/` directories. M4 is run against
the *freshly-produced* M2/M3 checkpoints from the same script run (via
`m4_run_experiment.py`'s `--m2-campaign-dir`/`--m3-campaign-dir`
overrides), not the committed ones -- a full run of this script all the
way through is a genuine, from-scratch verification of the entire
chain, not just each stage in isolation against pre-existing artifacts.

M1 is cheap (grid search + held-out eval against already-frozen,
already-committed live checkpoints -- no training, a few minutes). M2
and M3 are expensive: both are real training campaigns. M2 is 30 seeds
x 3 arms x (300 train + 50 eval) episodes; one `gat_ctde` seed alone
was timed at 317s, so the full campaign is on the order of 5-8 hours
on the machine this was developed on (8 cores, single-threaded CPU
torch). M3 (federated, 3 client networks per condition) is comparable
or somewhat larger: 10 seeds x 5 sigma levels. M4 is eval-only (no
gradient steps) and fast regardless of M2/M3's checkpoints' provenance
-- the full 330-cell sweep took 1070s (~18 minutes) against the
committed checkpoints, and should be similar against fresh ones.

Every M2/M3/M4 sub-stage is merge-safe/resumable
(`campaign_results.json` skips cells already present) -- if the script
is interrupted (machine restart, session timeout), re-running it with
the **same** `OUT_ROOT` picks up exactly where it left off, it does not
restart from zero.

## Independent-seed replication (not just same-seed reproduction)

```bash
experiments/scripts/reproduce_paper5_fresh_seeds.sh [OUT_ROOT] [SEED_BASE]
```

A stronger check than `reproduce_paper5_full.sh`'s same-seed
determinism verification: retrains M2 (30 seeds), M3 (10 seeds x 5
sigma), and M4 (10 seeds) from scratch under a **disjoint** seed range
(default base 1000, i.e. seeds 1000-1029/1000-1009 instead of
900-929/900-909) -- testing whether the paper's statistical findings
(paired significance, the accelerating dropout/churn severity curves)
hold up under an independent sample, not merely that the same seed
reproduces the same number. `m2_seed_campaign.py`/`m3_privacy_sweep.py`
both gained a `--seed-base` override for this (default unchanged, so
every existing invocation is unaffected); `m4_seed_campaign.py` already
had a `--seeds` override. M4's stage runs against this same script's own
freshly-trained M2/M3 checkpoints via `--m2-campaign-dir`/
`--m3-campaign-dir`, not the committed ones.

**M1 is deliberately not part of this script.** It evaluates paper #4's
real, historical live-hardware checkpoints (training seeds 256-261)
against already-recorded live traffic from an actual OpenAirInterface
testbed run -- there is no "fresh seed" retraining of a live testbed
session to redo from this codebase. M1's own reproducibility (does the
grid search / held-out eval reproduce on a clean run) is already fully
covered by `reproduce_paper5_full.sh`.

Output lands in an isolated `experiments/results/fresh_seed_retrain/`
by default -- separate from both the committed results and the
same-seed `reproduction_check/` directory. Since this and the same-seed
reproduction are both real training campaigns competing for the same
CPU budget, they are chained to run sequentially (fresh-seed starts
automatically once the same-seed run exits), not in parallel.

**Verify the result:**

```bash
python3 experiments/scripts/compare_reproduction.py --repro-root <OUT_ROOT>
```

Reports exact-match / mismatch per (arm, seed) or (arm, condition, seed)
cell against the committed results. Fixed-seed RL training/eval in this
codebase is expected to reproduce **exactly** (bit-identical
`sla_compliance_all_slices`), not just approximately -- confirmed
already for M1 (`m1_extract_live_traces.py`'s own live-compliance
numbers reproduce the milestone doc's numbers exactly on a fresh
extraction) and for M4 (`m2_reeval_gat_ctde.py`'s clean eval re-run of
`gat_ctde` reproduced the independently-derived "true last block"
numbers exactly, seed for seed, twice over, during the eval-log
contamination fix in `docs/PAPER5_M2_gat_ctde.md` section 14). Any
divergence found by this comparison is a real finding to investigate,
not noise to average away.

## Provenance table

| Item | Source | Script | Config | Seeds |
|---|---|---|---|---|
| M1 live traces (95.7/61.9/100/100/90.5/100% per-checkpoint compliance, 4.17% hardware-outlier drop) | `experiments/results/live_campaign_v2/dqn_sla` (seed 256, paper #4's own data) + `experiments/results/live_checkpoint_sensitivity/dqn_sla_seed{257-261}` | `experiments/scripts/m1_extract_live_traces.py` | n/a (reads existing omega logs) | 256-261 (checkpoints, not training seeds) |
| M1 grid search (best fit: bc=3200, drift=0.1, vol=0.04, ar1=0.0; loss 1.542-1.562 across top-5) | frozen checkpoints 256/257/258 | `experiments/scripts/m1_fit_recalibration.py` | `saclb_offline_dqn.yaml` (via `RecalibratedClosedLoopKpmSource`) | 256, 257, 258 |
| M1 held-out eval (baseline 9/9/14/13/20/9%; recalibrated bit-identical) | all 6 frozen checkpoints | `experiments/scripts/m1_run_held_out_eval.py --backlog-capacity {2000,3200} --drift-coef 0.1 --offered-volatility 0.04 --ar1-coef 0.0` | as above | 5001-5010 (held-out, 10 ep/seed) |
| M1 Spearman rho=0.097, p=0.855 | held-out eval outputs above | one-off analysis, same doc | as above | as above |
| M1b loss/backlog coupling (monotonically worse, gate not met) | frozen checkpoints 256/257/258 | `experiments/scripts/m1b_loss_backlog_coupled_source.py` (used by a grid-search variant of `m1_fit_recalibration.py`) | `loss_noise_std` in {0,0.05,0.1,0.2} x `loss_noise_ar1` in {0,0.5,0.85}, demand params at M1's best fit | 256, 257, 258 |
| Table I / Fig. 2 collapse-reduction counts (0/30 -> 3/30 -> 21/30 differentiated) | `experiments/results/m2_campaign/gat_ctde/seed{900-929}/eval/omega_log.jsonl` | `experiments/scripts/m2_seed_campaign.py --arms gat_ctde` (LayerNorm-only and per-slice-heads are the two states of `ctde_policy.py`/`gat_encoder.py` at the time each pass ran -- current code IS the per-slice-heads/final state; the 0/30 and 3/30 pre-fix numbers are historical, recoverable from git history commit `a756044`/pre-`d3096a8`, not from a live script flag) | `saclb_offline_dqn.yaml` | 900-929 |
| Table I (all three arms' compliance/reward/precision) | `experiments/results/m2_campaign/{gat_ctde,independent_dqn,single_agent_dqn}/seed{900-929}/eval/omega_log.jsonl` | `experiments/scripts/m2_seed_campaign.py` (full 3-arm run) -> `experiments/scripts/m2_campaign_analysis.py` (compliance) + `experiments/scripts/m2_correctness_metrics.py` (reward/precision) | `saclb_offline_dqn.yaml` | 900-929 |
| Paired GAT-CTDE vs. independent-DQN (+1.461, p<0.0001, 27/0/3) and vs. single-agent DQN (+0.195, p=0.0577, 20/3/7), mean_reward_per_step | same M2 eval logs | `experiments/scripts/m2_correctness_metrics.py` | as above | 900-929 |
| Fig. 3 (per-seed paired slope + distribution) | same M2 data | `experiments/plots/paper5_fig3_m2_campaign.py` | n/a (reads `campaign_results.json` + eval logs) | 900-929 |
| Fig. 2 (collapse-reduction bars) | hardcoded `DIFFERENTIATED = [0, 3, 21]` (historical + current campaign counts, see row above) | `experiments/plots/paper5_fig2_collapse_reduction.py` | n/a | 900-929 (for the 21 count) |
| Federation cost (+0.133, p=0.8125), M3 privacy sweep (block_precision 1.000/1.000/1.000/0.834/0.584 at sigma=0/0.5/1/2/4) | `experiments/results/m3_campaign/fl_gat_ctde_sigma*/seed{900-909}/eval/omega_log.jsonl`, centralized reference from the M2 `gat_ctde` logs above | `experiments/scripts/m3_privacy_sweep.py` -> `experiments/scripts/m3_correctness_metrics.py` | `saclb_offline_dqn.yaml` | 900-909 |
| Fig. 4 (federation-cost slope + privacy-utility curve) | same M3 data | `experiments/plots/paper5_fig4_m3_privacy.py` | n/a | 900-909 |
| M4 dropout/churn/spike per-condition results (`docs/PAPER5_M4_disruption.md`'s result tables) | `experiments/results/m4_campaign/<arm>/<kind>_sev<N>/seed{900-909}/eval/omega_log.jsonl`, checkpoints from M2/M3 above (never retrained) | `experiments/scripts/m4_seed_campaign.py` -> `experiments/scripts/m4_correctness_metrics.py` | `saclb_offline_dqn.yaml` | 900-909 |
| Fig. 1 (architecture diagram) | n/a -- hand-authored TikZ, not data-driven | `paper5/figures/fig1_gat_ctde_fl_architecture.tex` | n/a | n/a |

## Known, already-disclosed reproducibility caveats

- **M2's `gat_ctde` collapse-rate history (0/30, 3/30) is not re-derivable
  by a script flag.** These are the original-encoder and LayerNorm-only
  states of code that has since been replaced in place (the per-slice
  Q-heads fix). The current repository only runs the final,
  per-slice-heads architecture. The two superseded states are preserved
  in git history (`git show a756044:...` for the original 30/30, the
  commit immediately before `d3096a8` for the LayerNorm-only 27/30) --
  see `docs/PAPER5_M2_gat_ctde.md` sections 11-12 for the exact
  diagnostic chain and commit references. Re-deriving these would
  require checking out those historical commits and re-running M2's
  campaign under each -- not automated here, since it would mean running
  three full 30-seed campaigns instead of one.
- **The M2 eval-log append-contamination bug** (`docs/PAPER5_M2_gat_ctde.md`
  section 14): fixed in `m2_run_experiment.py` (`_clear_seed_dir`, called
  before any fresh per-seed run). A from-scratch run via this doc's
  one-command path cannot reproduce that bug (each seed's directory is
  always cleared before writing), so a fresh reproduction's `gat_ctde`
  numbers are expected to match the *already-corrected* committed
  numbers, not the originally-published (contaminated) ones.
- **The M4 volume-confound metric issue** (`docs/PAPER5_M4_disruption.md`):
  `m4_correctness_metrics.py`'s primary metric is already the
  volume-normalized one; raw `mean_reward_per_step` is reported
  alongside for continuity only and is expected to show the same
  spike-inflation artifact on a fresh run too (it is a property of the
  reward function, not of any particular run).
- **Old-rig-checkpoint check**: every M1-M4 number traces to
  `experiments/results/{live_campaign_v2,live_checkpoint_sensitivity,
  m1_recalibration,m2_campaign,m3_campaign,m4_campaign}/` -- this
  project's own results directories. M1's live traces reuse paper #4's
  already-existing, already-verified live data (never regenerated, per
  `docs/REPRODUCIBILITY.md`'s own old-rig check for that data's
  provenance); M2-M4 are 100% offline-simulation, entirely this
  session's own output. No number traces to an old-rig checkpoint.
- **Wall-clock is machine-dependent.** All timing estimates above were
  measured on this project's own development machine (8 cores, CPU-only
  torch). A faster machine (more cores, GPU) would complete the full
  pipeline faster; a slower one, slower. The merge-safe/resumable design
  of every M2/M3/M4 sub-stage means total wall-clock budget is not a
  correctness concern, only a convenience one.
