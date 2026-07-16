# Figures Manifest

Maps each Phase 4 figure file to the data source run IDs it was generated
from, plus a one-sentence caption draft. All figures are produced by
scripts in `experiments/plots/` (committed, re-runnable, no hand-made
figures) reading only from omega logs. **This file is filled in with real
run IDs once Phase 3 live evaluation data exists** -- currently a
skeleton reflecting the plotting scripts already built and smoke-tested
against Phase 2 offline training data.

| # | Figure file | Script | Data source (run IDs / seeds) | Caption draft |
|---|---|---|---|---|
| 1 | `fig1_training_convergence.{pdf,png}` | `fig1_training_convergence.py` | offline training, `experiments/results/offline/<sla,qoe>/seed{256,257,258}/{dqn,a2c}/offline_closed_loop/rep_0/` | Training convergence (mean reward per step vs. episode, mean ± seed std across 3 seeds) for the four learned arms under both reward modes. |
| 2 | `fig2_sla_compliance.{pdf,png}` | `fig2_sla_compliance.py` | live eval, `experiments/results/live/<arm>/<mode>/rep_seed{256,257,258}/` | Per-slice (eMBB/URLLC/mMTC) SLA compliance (%), mean ± std across seeds, for all 5 arms. |
| 3 | `fig3_urllc_blocking.{pdf,png}` | `fig3_urllc_blocking.py` | live eval, same as above | Distribution (box plot) of URLLC blocks per episode, pooled across seeds, by arm. |
| 4 | `fig4_ceiling_trajectories.{pdf,png}` | `fig4_ceiling_trajectories.py` | live eval, baseline vs. best learned arm (TBD after Phase 3 results), one representative episode/seed | Commanded PRB ceiling (max_ratio) per slice over one representative episode: baseline's static/floor signature vs. the best learned arm's adaptive trajectory. |
| 5 | `fig5_backlog.{pdf,png}` | `fig5_backlog.py` | live eval, all 5 arms | CDF of per-slice SLA margin (continuous backlog-severity proxy) by arm -- the figure connecting Phase 1's contention-gate finding to the campaign's actual results. |
| 6 | `fig6_inferred_mos.{pdf,png}` | `fig6_inferred_mos.py` | live eval, all 5 arms | Inferred per-slice MOS (passive QoE diagnostic, logged for every arm regardless of reward mode), mean ± std across seeds. |
| 7 | `fig7_qoe_decomposition.{pdf,png}` | `fig7_qoe_decomposition.py` | offline training, dqn_qoe/a2c_qoe, seeds 256/257/258 | eq.9 reward decomposition (α·MOS / β·cost / γ·SLA_viol) over training episodes for the two QoE-reward arms. |

## Notes / caveats carried into every caption

- All error bands/bars are mean ± std across **n seeds** (state n explicitly
  in the final caption text -- currently 3, pending the Phase 3 scope
  decision).
- Fig 5 uses `per_slice_sla_margin` (Lmax-normalized), NOT raw
  `dl_mac_buffer_occupation` bytes -- the standard per-step omega evidence
  dict does not carry the raw byte value. Phase 1's own contention-gate
  trace (`experiments/logs/phase1/embb_final_*.jsonl`) is the citation for
  the raw-byte contention-gate claim specifically, not this figure.
- Live-eval reps that span multiple health-checked batches
  (`experiments/scripts/run_live_eval_arm.py`) have per-batch-reseeded RNG
  streams -- see `batch_manifest.jsonl` per rep for the actual episode
  provenance if a reviewer asks about continuity within a rep.
