# Stage 14 — End-to-end validation audit

User: *"check my entire simulation flow from end-to-end for any errors,
anomalies, fake data and whatnot. my work needs to be fully validated
and reproducible."*

Scope: every number currently in `paper_conf/main.tex`, traced from the
raw `omega_log.jsonl` / checkpoint / config artifacts through the
metrics scripts that produce it, independently recomputed (not just
re-read from a doc), plus a scan for fabrication, silent fallback, and
known-bug recurrence.

## Method

For each reported number: (1) find its source script + config + seed
list (`docs/REPRODUCIBILITY.md`), (2) re-run that script fresh against
the raw logs right now, or independently recompute it from the raw
logs in a fresh script when no single script produces it directly
(e.g. the pooled congested numbers), (3) diff against both the
manuscript and `REPRODUCIBILITY.md`'s claim. Separately: hash/mtime/
content spot-checks on the raw logs themselves for fabrication
signatures, a determinism test, and a check that this project's two
previously-known bug classes (the `cwd=framework/` relative-path bug,
old-rig-checkpoint reuse) don't touch anything currently cited.

## Result: every number re-verified matches exactly

| Manuscript number | Re-derived value | Method |
|---|---|---|
| Table (live SLA compliance, 4 arms, 46 ep/arm): 91.7/91.4/91.4, 97.4/95.7/95.7, 96.6/95.7/95.8, 100/100/100 | identical | fresh run, `metrics_stage5_v2.py` |
| Episodes fully compliant: 42/46, 44/46, 44/46, 46/46 | identical | same run |
| Fisher exact: dqn\_sla vs static\_at\_cap $p=1.0$; dqn\_sla vs baseline $p=0.68$; dqn\_qoe vs baseline $p=0.12$ | identical | same run |
| Historical $p=0.0149$ (static-at-cap vs.\ DQN-SLA, Stage 3) | $p=0.014936$ | recomputed Fisher exact from `docs/stage3_metrics_raw.json`'s raw counts (11/15 vs.\ 25/25) — this number is not stored precomputed anywhere, had to be derived from the counts to check it |
| Congested (offline), pooled 165 ep/arm: baseline 19.7/32.7/19.8, DQN-SLA 27.7/7.7/9.4, DQN-QoE 25.0/8.2/11.6 | identical | step-count-weighted pool of `docs/stage2_metrics_raw.json` (n=2700/arm) + `congested_vs_baseline_v7_reverify/results.json` (n=7200/arm) → n=9900=165 episodes, written fresh, not previously saved anywhere as a script |
| Priority-weighted congested score: baseline 24.9 vs.\ 19.1/17.9 for DQN | identical | `weighted_u` applied to the pooled numbers above with `CONGESTED_PRIORITY_WEIGHT` from `metrics_stage2.py`, matching `saclb_offline_congested_v1.yaml`'s own `priority_weight` fields |
| Congested live pilot MOS: baseline 2.66, SLA 2.44, QoE 4.83, 100% compliance | identical | recomputed directly from `live_congested_pilot/*/omega_log.jsonl` (n=2 episodes/arm) |
| Checkpoint sensitivity: 13/21, 21/21, 21/21, 19/21, 21/21 (seeds 257–261) | identical | recomputed directly from `live_checkpoint_sensitivity/*/omega_log.jsonl` |
| E2 round-trip median 0.57 ms | 0.566 ms | `docs/stage4_e2_latency_raw.json`, correctly reported as median |
| DQN params 18,562 (\"under 20,000\") | 18,562 | `docs/stage4_metrics_offline_raw.json` |
| 106 PRBs, real gNB config | confirmed | `docker_open5gs/oai/gnb.sa.band78.fr1.106PRB.usrpb210.conf`: `dl_carrierBandwidth = 106` |

Nothing needed correcting in the manuscript from this pass.

## One real finding: an imprecise claim, not fabricated data

**"the DQN model itself ... selects an action in under 70 $\mu$s on
CPU"** (`main.tex`, Deployment Feasibility) understates the spread.
`docs/stage4_metrics_offline_raw.json`'s own percentiles, already
sitting right there in the artifact this claim is sourced from:

| | median | mean | p90 |
|---|---|---|---|
| DQN-SLA | 67.9 $\mu$s | 70.2 $\mu$s | 73.1 $\mu$s |
| DQN-QoE | 68.3 $\mu$s | 75.4 $\mu$s | 89.3 $\mu$s |

Median and min are under 70 $\mu$s for both arms; mean and p90 are not
(DQN-QoE's mean is 75.4 $\mu$s, p90 89.3 $\mu$s). The manuscript's
"under 70 $\mu$s" reads as a blanket bound but is really the median.
Not invented -- the real number picked was just the most flattering
one, unlabelled as such. **Fix is a one-line wording change** ("a
median of under 70 $\mu$s" or "typically under 70 $\mu$s, up to
~90 $\mu$s at p90") -- flagging here rather than silently editing the
manuscript, since the exact phrasing is the author's call.

## Fabrication / anomaly sweep: nothing found

- **Hashes**: all 44 `live_campaign_v2` omega logs have distinct MD5s
  -- no copy-pasted/duplicated episode data across seeds or arms.
- **Timestamps**: file mtimes span 2026-07-29 16:10 through 2026-07-31
  05:35 (real 37.5-hour, multi-day campaign, irregular gaps of
  9–30 min between consecutive seed/arm blocks) -- not a batch-generated
  or instantaneous artifact.
- **Content spot-check**: raw per-step records carry an honest,
  system-generated `limitation` field disclosing real sensor gaps
  (e.g. "`dl_mac_buffer_occupation` was 0 for all UEs; used
  `(dl_errors+dl_bler)` as a backlog proxy") -- the opposite of
  fabrication, an explicit fallback-and-say-so discipline that's
  consistent project-wide (`qoe_mapper.py`'s `LatencyProxy` does the
  same for its own real, documented sensor sparsity: 99.9%/53.3%/13.5%
  nonzero-reading rates for eMBB/URLLC/mMTC).
- **Determinism**: re-ran the same checkpoint/seed/episode-count
  offline evaluation twice, independently, minutes apart --
  byte-identical `omega_log.jsonl` output both times.
- **`cwd=framework/` bug** (this project's recurring failure mode,
  first caught in Stage 10): confirmed the live-campaign launcher,
  `experiments/scripts/run_live_eval_arm.py`, explicitly passes
  `cwd=FRAMEWORK_DIR` to its subprocess call -- the bug's precondition
  never holds for any data behind a currently-cited number. Where it
  *has* bitten (offline analysis scripts run ad hoc), the failure mode
  is a hard `FileNotFoundError` crash (confirmed by reading
  `env.py`'s `torch.load` call site -- no silent `except`), not silent
  bad data -- consistent with every previous catch of this bug being a
  missing-file/short-episode-count problem, not a corrupted-value one.
- **Old-rig checkpoints**: no config, script, or doc under active use
  references an old-rig path; `saclb_campaign*.yaml`'s own header
  comment states its ceiling caps were "validated live against this
  campaign's actual traffic profile, not inherited from the old rig."
  No checkpoint file in `experiments/results/` is a symlink.
- **Contention gate**: `experiments/logs/phase1/embb_contention_gate_
  20260716_202627_clean.jsonl` timestamps to 2026-07-16; the earliest
  `offline_v2` checkpoint (training) is 2026-07-29 04:08, and
  `live_campaign_v2` starts 2026-07-29 16:10 -- gate strictly precedes
  both, as required.
- **Seed hygiene**: training (256–261), held-out offline eval
  (5001–5010), and live eval (950–960) occupy disjoint integer ranges
  by construction; no script reuses a training seed for evaluation.

## Known, already-disclosed reproducibility gaps (not new findings)

`REPRODUCIBILITY.md` already flags these itself -- repeating them here
because the user asked specifically about reproducibility, not just
correctness:

- The congested live pilot's numbers and the checkpoint-sensitivity
  aggregation were each produced by a one-off script that was not
  saved -- the *data* is real and I re-verified it directly against
  the raw logs above, but re-deriving it from scratch today means
  re-writing that aggregation rather than re-running a checked-in
  script. Low risk (both are simple mean/threshold reductions I just
  reproduced independently in this pass), but worth writing a small
  permanent script for either it's revisited.

## Bottom line

No fabricated, mocked, or anomalous data found anywhere in the path
behind a currently-cited number. Every percentage, count, and p-value
in the manuscript reproduces exactly from the raw logs, using either
the project's own metrics scripts (re-run fresh) or an independent
recomputation where no single script existed. The one issue is a
labelling imprecision on the inference-latency claim, not a
correctness or fabrication problem, and is a one-line fix left for the
author to phrase.
