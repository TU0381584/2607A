# Figures Manifest

Maps each Phase 4 figure file to the data source run IDs it was generated
from, plus a one-sentence caption draft. All figures are produced by
scripts in `experiments/plots/` (committed, re-runnable, no hand-made
figures) reading only from omega logs.

| # | Figure file | Script | Data source (run IDs / seeds) | Caption draft |
|---|---|---|---|---|
| 1 | `fig1_training_convergence.{pdf,png}` | `fig1_training_convergence.py` | offline training, `experiments/results/offline/<sla,qoe>/seed{256,257,258}/{dqn,a2c}/offline_closed_loop/rep_0/` | Training convergence (mean reward per step vs. episode, mean ± seed std across 3 seeds), 2×2 layout (SLA/QoE reward magnitudes are on different scales, not overlaid). A2C converges within ~10 episodes under both reward modes; DQN takes ~150 episodes under the SLA reward. |
| 2 | `fig2_sla_compliance.{pdf,png}` | `fig2_sla_compliance.py` | live eval, `experiments/results/live_campaign/<arm>/<mode>/rep_seed{950,951,952}/`, n=15 episodes/arm | **Re-visualized (Phase 1 follow-up, 2026-07-19), replaces the original mean±std bar chart.** Top: one dot per episode (per-episode SLA compliance %, mean across 3 slices), deterministically jittered — every learned arm's 15 episodes sit flat at 100.0%; baseline shows a genuine bimodal split (majority ~100%, a real cluster at 0%), not a uniform ~73% mean. Bottom: fraction of episodes fully compliant and worst single-episode compliance per arm — baseline's worst episode is 0.0% on every slice vs. 100.0% for every learned arm. Mean±std bars hid this; see RESULTS_REPORT.md §4 for the original reading and STATUS_AND_NEXT_STEPS-era rationale for why this re-visualization was done. |
| 3 | `fig3_urllc_blocking.{pdf,png}` | `fig3_urllc_blocking.py` | live eval, same as above, n=15 episodes pooled/arm | Box plot of URLLC blocks/episode, pooled across 3 seeds × 5 episodes. Baseline and 3 of 4 learned arms show exactly 0 in every episode; `a2c_qoe` blocks 35–47/episode in 100% of its episodes — a systematic, A2C-specific anomaly (RESULTS_REPORT.md §5.1), not noise. |
| 4 | `fig4_ceiling_trajectories.{pdf,png}` | `fig4_ceiling_trajectories.py` | live eval, `baseline` vs. `dqn_sla`, seed 950, episode 1 (`run_id=dqn_sla_sla_seed950_batch0`) | Commanded PRB ceiling (max_ratio) per slice over one representative episode: baseline's ceiling is flat at its static nominal ratio for all 60 steps; DQN (SLA reward) rides eMBB's ceiling from floor to the calibrated cap (12) within ~13 steps and jumps URLLC/mMTC to cap immediately, holding both for the rest of the episode. **Updated (Phase 1 follow-up, 2026-07-19):** each panel now annotates its calibrated `max_ratio_cap` (embb=12, urllc=4, mmtc=3, from `saclb_campaign.yaml`) as a dotted reference line — the mechanism figure now shows the ceiling against the actual ceiling-of-the-ceiling. |
| 5 | `fig5_backlog.{pdf,png}` | `fig5_backlog.py` | live eval, all 5 arms, n=3 seeds (all steps pooled); plus `baseline` vs. `dqn_sla`, seed 950, episode 1 for the new top row | **Re-visualized (Phase 1 follow-up, 2026-07-19), adds a time-series row above the original CDF.** Top: per-slice SLA margin over one representative episode (seed 950, episode 1), baseline vs. DQN/SLA, symlog y-axis — baseline collapses to its floor (≈−1e6) within ~5 steps and never recovers, while the learned arm stays flat near +1.0 for the whole episode; this makes the orders-of-magnitude event visible as an event, not just a pooled statistic. Bottom: CDF of per-slice SLA margin (continuous backlog-severity proxy, clipped at −1.5 for display — raw values reach ≈−1e6 under baseline's worst-case contention), unchanged from the original figure. Baseline's CDF sits at the clip floor almost the entire time; all 4 learned arms sit mostly in the comfortable region (~0.7–1.0). |
| 6 | `fig6_inferred_mos.{pdf,png}` | `fig6_inferred_mos.py` | live eval, all 5 arms, n=3 seeds | Inferred per-slice MOS, mean ± std across seeds. eMBB and URLLC sit low (~1.2–1.7/5) and are overwhelmingly policy-independent (eMBB MOS is identical to 3 decimals across all 4 learned arms); mMTC sits high (~4.6–4.8/5), also policy-independent. Baseline's URLLC MOS has by far the largest variance of any arm/slice (std 0.331 vs. 0.05–0.12 for learned arms), consistent with its seed-dependent compliance (RESULTS_REPORT.md §5.2). |
| 7 | `fig7_qoe_decomposition.{pdf,png}` | `fig7_qoe_decomposition.py` | offline training, `dqn_qoe`/`a2c_qoe`, seeds 256/257/258 | eq.9 reward decomposition (α·MOS / β·cost / γ·SLA_viol) over training episodes for the two QoE-reward arms. |

## Table sources

| Table | Script | Notes |
|---|---|---|
| T1 (testbed/campaign parameters) | hand-written, `paper_conf/tables/table1_params.tex` | Values traceable to `experiments/configs/*.yaml` and `CAMPAIGN_LOG.md`'s calibration sections — no campaign data needed. |
| T2 (headline results) | auto-generated, `experiments/plots/generate_results_tables.py` → `paper_conf/tables/table2_results.tex` | Do not hand-edit; re-run the script if the underlying omega logs change. **Updated (Phase 1 follow-up, 2026-07-19):** adds "Worst ep. (%)" (min per-episode compliance, pooled across all episodes/seeds — not a mean-of-per-rep-means) and "P5 margin" (5th percentile of pooled per-step SLA margin) columns alongside the existing mean±std columns. |

## Notes / caveats carried into every caption

- All error bands/bars are mean ± std across **n=3 seeds** (950, 951, 952),
  stated explicitly in each figure's caption text.
- Fig 5 uses `per_slice_sla_margin` (Lmax-normalized), NOT raw
  `dl_mac_buffer_occupation` bytes — the standard per-step omega evidence
  dict does not carry the raw byte value. Phase 1's own contention-gate
  trace (`experiments/logs/phase1/embb_final_*.jsonl`) is the citation for
  the raw-byte contention-gate claim specifically, not this figure.
- **Fig 4 caught a real bug during generation**: episode numbers in
  `run_live_eval_arm.py`-orchestrated omega logs are only unique WITHIN a
  batch (each health-checked batch reseeds its own local episode counter
  at 1 — see `CAMPAIGN_LOG.md`'s documented batch-reseeding caveat).
  Filtering by episode number alone silently overlaid 2-3 distinct
  episodes from different batches onto one trajectory plot; fixed by
  also constraining on `run_id` (`fig4_ceiling_trajectories.py`'s
  `load_episode_ceilings()` now requires a matching `run_id`, defaulting
  to the first one encountered). Any future figure needing a *specific
  single episode's* time series (not an aggregate across all episodes)
  must do the same — Figs 2/3/5/6/7 are unaffected because they aggregate
  across all episodes/rollups rather than selecting one.
- Live-eval reps that span multiple health-checked batches have
  per-batch-reseeded RNG streams — see `batch_manifest.jsonl` per rep for
  the actual episode provenance if a reviewer asks about continuity
  within a rep.
