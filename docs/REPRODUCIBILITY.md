# Reproducibility appendix (paper #4)

Not part of the 4-page manuscript (no room, and it would reintroduce
exactly the tool/script-name jargon the house-style rewrite removed).
Kept here as the author's own audit trail: for every number, figure,
and table in `paper_conf/main.tex`, the exact script/config/seed that
produced it, so a reviewer question or a future re-run doesn't require
re-deriving provenance from scratch.

| Manuscript item | Source data | Script | Config | Seeds |
|---|---|---|---|---|
| Table I (live SLA compliance) — baseline, DQN (SLA/QoE) rows | `experiments/results/live_campaign/{baseline,dqn_sla,dqn_qoe}/**/omega_log.jsonl` | `experiments/scripts/metrics_stage2.py` → `docs/stage2_metrics_raw.json` | `saclb_campaign.yaml` | 950, 951, 952 |
| Table I — static-at-cap row | `experiments/results/live_campaign/static_at_cap/**/omega_log.jsonl` | `experiments/scripts/metrics_stage3.py` → `docs/stage3_metrics_raw.json` | `saclb_campaign_static_at_cap.yaml` | 950, 951, 952 |
| Fig. 1 (ceiling trajectories) | same live-campaign omega logs, baseline vs.\ dqn\_sla, episode 1 | `experiments/plots/fig4_ceiling_trajectories.py` (defaults fixed this stage — see below) | `saclb_campaign.yaml` | 950 |
| Fig. 2 (SLA compliance, per-episode) | live-campaign omega logs, 4 arms | `experiments/plots/fig2_sla_compliance.py --arms baseline dqn_sla dqn_qoe static_at_cap` | `saclb_campaign.yaml` / `saclb_campaign_static_at_cap.yaml` | 950, 951, 952 |
| Fisher exact test, $p=0.0149$ (static-at-cap vs.\ DQN collapse rate) | `docs/stage3_metrics_raw.json` → `fisher_vs_dqn_sla_combined` | `experiments/scripts/metrics_stage3.py` | as above | 950–954 (DQN reverification adds 953, 954) |
| Table II (congested scenario) | re-evaluation of frozen offline checkpoints | `experiments/scripts/eval_congested_vs_baseline.py`, re-run via `metrics_stage2.py` → `docs/stage2_metrics_raw.json` | `saclb_offline_congested_v1.yaml` | held-out episodes, same checkpoints as originally trained |
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
