# Reproducibility appendix (paper #4)

Not part of the 4-page manuscript (no room, and it would reintroduce
exactly the tool/script-name jargon the house-style rewrite removed).
Kept here as the author's own audit trail: for every number, figure,
and table in `paper_conf/main.tex`, the exact script/config/seed that
produced it, so a reviewer question or a future re-run doesn't require
re-deriving provenance from scratch.

**Superseded (2026-07-30, see below): the original Table I / Fig. 1 / Fig.
2 / $p=0.0149$ row below used the OLD (pre-Stage-5) calibration and a
smaller sample. The manuscript's current Table I / Fig. 1 / Fig. 2 use
the rows further down this table (the n=21, `live_campaign_v2` data).
Kept here unedited for provenance of what was previously reported.**

| Manuscript item (superseded) | Source data | Script | Config | Seeds |
|---|---|---|---|---|
| Table I (live SLA compliance) — baseline, DQN (SLA/QoE) rows | `experiments/results/live_campaign/{baseline,dqn_sla,dqn_qoe}/**/omega_log.jsonl` | `experiments/scripts/metrics_stage2.py` → `docs/stage2_metrics_raw.json` | `saclb_campaign.yaml` | 950, 951, 952 |
| Table I — static-at-cap row | `experiments/results/live_campaign/static_at_cap/**/omega_log.jsonl` | `experiments/scripts/metrics_stage3.py` → `docs/stage3_metrics_raw.json` | `saclb_campaign_static_at_cap.yaml` | 950, 951, 952 |
| Fisher exact test, $p=0.0149$ (static-at-cap vs.\ DQN collapse rate) | `docs/stage3_metrics_raw.json` → `fisher_vs_dqn_sla_combined` | `experiments/scripts/metrics_stage3.py` | as above | 950–954 (DQN reverification adds 953, 954) |

**Superseded (2026-07-30/31, n=46/arm): the top-up + 10h reverification
campaigns' numbers were current until the Stage 15 n=128 campaign
(2026-08-01/03, `docs/STAGE15_n128_campaign.md`) extended the same
dataset further. Kept below, struck through in spirit, for provenance;
current numbers are in the table after.**

| Manuscript item | Source data | Script | Config | Seeds |
|---|---|---|---|---|
| Table (all 4 arms, n=46 each, superseded) | `experiments/results/live_campaign_v2/{baseline,dqn_sla,dqn_qoe,static_at_cap}/**/omega_log.jsonl` | `experiments/scripts/metrics_stage5_v2.py` (`ARM_SEEDS` = 950–960) → `docs/stage5_v2_campaign_metrics_raw.json` | `saclb_campaign_v2.yaml` / `saclb_campaign_baseline_v2.yaml` / `saclb_campaign_static_at_cap_v2.yaml` | 950–952 (2 ep/seed) + 953–960 (5 ep/seed) = 46/arm |
| Fisher exact tests, superseded: dqn\_sla vs.\ static\_at\_cap ($p=1.0$, 44/46 vs.\ 44/46), dqn\_sla vs.\ baseline ($p=0.68$), dqn\_qoe vs.\ baseline ($p=0.12$) | `docs/stage5_v2_campaign_metrics_raw.json` (pre-Stage-15 version, see git history) | `experiments/scripts/metrics_stage5_v2.py` | as above | 950–960 |

**Current (2026-08-01/03 Stage 15 n=128 campaign, same dataset extended
with 17 more seeds, `docs/STAGE15_n128_campaign.md`):**

| Manuscript item | Source data | Script | Config | Seeds |
|---|---|---|---|---|
| Table I (all 4 arms, n=128 each; renumbered from Table II after the 2026-08-03 4.5-page trim dropped the old Table I) | `experiments/results/live_campaign_v2/{baseline,dqn_sla,dqn_qoe,static_at_cap}/**/omega_log.jsonl` | `experiments/scripts/metrics_stage5_v2.py` (`ARM_SEEDS` = 950–977) → `docs/stage5_v2_campaign_metrics_raw.json` | `saclb_campaign_v2.yaml` / `saclb_campaign_baseline_v2.yaml` / `saclb_campaign_static_at_cap_v2.yaml` | 950–952 (2 ep/seed) + 953–976 (5 ep/seed) + 977 (2 ep/seed) = 128/arm |
| Ceiling trajectories, baseline vs.\ dqn\_qoe -- cut from the manuscript in the 2026-08-03 4.5-page trim (superseded in the paper by the DQN-SLA-vs-DQN-QoE companion figure below; kept here for provenance) | live_campaign_v2 omega logs, episode 1, seed 950 | `experiments/plots/fig4_ceiling_trajectories.py --live-root experiments/results/live_campaign_v2 --best-arm dqn_qoe` | `saclb_campaign_v2.yaml` | 950 |
| Fig. 2 (SLA compliance, per-episode, 4 arms; renumbered from Fig. 4 after the trim) | live_campaign_v2 omega logs | `experiments/plots/fig2_sla_compliance.py --live-root experiments/results/live_campaign_v2 --seeds 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977` | v2 configs as above | 950–977 |
| Fisher exact tests: dqn\_sla vs.\ static\_at\_cap ($p=1.0$, 126/128 vs.\ 126/128), dqn\_sla vs.\ baseline ($p=0.10$ raw / $p=0.205$ Holm), dqn\_qoe vs.\ baseline ($p=0.0070$ raw / $p=0.0209$ Holm) -- Holm-Bonferroni applied across the 3-test vs.-baseline family (`holm_bonferroni()`, `metrics_stage2.py`) | `docs/stage5_v2_campaign_metrics_raw.json` → `fisher_vs_static_at_cap` / `fisher_vs_baseline` (`p_value` raw, `p_value_holm` adjusted) | `experiments/scripts/metrics_stage5_v2.py` | as above | 950–977 |
| Fig. 1 (DQN-SLA vs.\ DQN-QoE commanded ceiling, seed 955 ep.\ 1, the episode DQN-SLA collapses; renumbered from the "Fig.~3 companion" after the trim, now the only ceiling-trajectory figure in the paper) | `experiments/results/live_campaign_v2/{dqn_sla,dqn_qoe}/{sla,qoe}/rep_seed955/omega_log.jsonl`, run\_ids `dqn_sla_sla_seed955_batch0`/`dqn_qoe_qoe_seed955_batch0` | `experiments/plots/fig_sla_vs_qoe_ceiling.py` | as above | 955 |
| Congested live pilot (all arms 100% compliant; QoE-reward URLLC MOS 4.83 vs.\ 2.44/2.66) | `experiments/results/live_congested_pilot/{baseline_congested,dqn_sla_congested,dqn_qoe_congested}/**/omega_log.jsonl` | one-off aggregation, same `_read_omega`/`per_slice_compliant`/`mos_by_slice` fields as `metrics_stage2.py` (no dedicated script written — see `docs/STAGE10_fullpower_reeval.md`) | `saclb_offline_congested_v1.yaml` / `saclb_offline_congested_v1_baseline.yaml` (new, one-line ceiling_step_ratio=0 variant) | 950 (1 seed, 2 episodes/arm — pilot scale, not yet statistically powered) |
| Checkpoint sensitivity (Section IV-A new paragraph: 3/5 perfect, 1/5 matches original, 1/5 collapses to 13/21) | `experiments/results/live_checkpoint_sensitivity/dqn_sla_seed{257,258,259,260,261}/sla/**/omega_log.jsonl` | `experiments/scripts/run_checkpoint_sensitivity.sh` (new) + one-off aggregation script, same fields as above | `saclb_campaign_v2.yaml`, checkpoints from `experiments/results/offline_v2_reverify/sla/seed{257-261}/...` (Stage 10's retrain) | 950–952 (2 ep/seed) + 953–955 (5 ep/seed) = 21/checkpoint; see `docs/STAGE11_checkpoint_sensitivity.md` for the data-corruption bug caught and fixed on 2 of these blocks |
| Table II (congested scenario, offline, pooled 165 episodes/arm) | original 3-seed run (`docs/stage2_metrics_raw.json`) pooled with an 8-new-seed extension (`experiments/results/congested_vs_baseline_v7_reverify/results.json`), by step-count-weighted average (not simple mean, since seed batches have different episode counts) | `experiments/scripts/eval_congested_vs_baseline.py --seeds 953 954 955 956 957 958 959 960 --episodes-per-seed 15` (must be run with `cwd=framework/` — see `docs/STAGE10_fullpower_reeval.md` section 6 for the relative-path bug this caught) | `saclb_offline_congested_v1.yaml`, same checkpoints as originally trained (not retrained) | 950–952 (orig, 15 ep/seed) + 953–960 (new, 15 ep/seed) = 165 ep/arm |
| Priority-weighted utility (27.2 / 21.0 / 19.0, §IV-C prose) | same congested re-evaluation | `metrics_stage2.py`'s `weighted_u`, weights from `saclb_offline_congested_v1.yaml`'s `priority_weight` field | as above | as above |
| E2 round-trip latency (0.57 ms median) | live measurement against the real gNB E2 agent | `experiments/scripts/measure_e2_latency.py` → `docs/stage4_e2_latency_raw.json` | n/a (direct measurement) | n/a |
| DQN inference time (under 70 $\mu$s), parameter count (18,562) | offline, on the real deployment checkpoints, CPU | `experiments/scripts/metrics_stage4_offline.py` → `docs/stage4_metrics_offline_raw.json` | `dqn_sla`/`dqn_qoe` seed-256 checkpoints | n/a |
| Control-loop cadence (5.0007 s vs.\ 5.000 s configured) — mentioned in `docs/STAGE4_instrumentation.md`, cut from the lean manuscript for space | live measurement | `experiments/scripts/profile_run_single_overhead.py` | `saclb_campaign.yaml` | n/a |

## Fixed during this stage

`experiments/plots/fig4_ceiling_trajectories.py`'s own docstring usage
example and its `--live-root`/`--seed` argparse defaults pointed at
`experiments/results/live` (seed 256) — a directory that does not
exist on this rig; the real data has always lived at
`experiments/results/live_campaign` (seed 950 for the figure actually
embedded in the paper), per `experiments/CAMPAIGN_LOG.md`'s own
description of how this figure was produced. Running the script
exactly as its own usage example instructed would have failed with a
missing-directory error. **This was a documentation/reproducibility
bug in the script, not a correctness bug in the embedded figure**:
regenerating with the corrected defaults reproduces
`paper_conf/figures/fig4_ceiling_trajectories.png` byte-for-byte
(verified this stage, `md5sum` match) — confirming the figure already
in the paper is exactly what it claims to be. Defaults corrected to
`--live-root experiments/results/live_campaign --seed 950`.

## Old-rig-checkpoint check

Every number above traces to `experiments/results/live_campaign*` or
`experiments/results/offline*` — this rig's own results directories,
all populated this project's sessions. `experiments/configs/saclb_campaign.yaml`'s
own header comment explicitly documents that its `max_ratio_cap`
values were "validated live against this campaign's actual traffic
profile, not inherited from the old rig or reasoned about from the
ratio formula" — the one place old-rig data is mentioned at all in the
current config/campaign files, and it is mentioned specifically to
rule out reuse, not to reuse it. No number in the paper traces to an
old-rig checkpoint.
