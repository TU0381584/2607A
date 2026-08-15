# Paper #5, M3 — federated GAT-CTDE + differential privacy

**Superseded framing note (read this before anything below):** this
document's original "Question" cited Block E's section-9 campaign numbers
(+0.284, 27/3/0) as the starting point. While building this M3 arm,
bit-identical eval compliance across a noise-multiplier sweep led to
discovering that `gat_ctde`'s entire section-9 campaign rests on an
architecture-level training collapse (`GATEncoder` had no normalization,
causing embedding magnitude rather than pattern to drive Q-values,
universally collapsing to an "always accept" no-op policy on 30/30
seeds) — full diagnosis, fix, and re-verification in
`docs/PAPER5_M2_gat_ctde.md` section 11. The fix (`GATEncoder` now has
per-layer LayerNorm) reduces but does not eliminate the collapse (27/30
seeds still reach it) and the centralized comparison point below now uses
section 11's **post-fix** numbers, not the original section-9 ones. This
M3 arm reuses the identical `GATEncoder`, was built and its first
(pre-fix) campaign run *before* this discovery, and was fully re-run
after the fix landed.

**Third framing update:** the author then asked for a way to make the
fix actually reliable ("ensure the AI is doing what it is supposed to
do"), leading to `docs/PAPER5_M2_gat_ctde.md` section 12's per-slice-
Q-head fix (22/30 centralized seeds now genuinely differentiated, up
from 3/30) and new correctness-aware metrics (`mean_reward_per_step`,
`block_precision`) since `sla_compliance_all_slices` was found to reward
the wrong behavior. This M3 arm inherited that fix automatically (same
`AgentQHead` class, imported not duplicated) and was re-run a third time
— see Result for the final numbers on both the compliance-based and
correctness-aware metrics.

**Question:** starting from the post-fix GAT-CTDE result (centralized
per-step joint training over the 3-gNB stress environment beats both
ablations with a statistically solid margin — see
`docs/PAPER5_M2_gat_ctde.md` section 11: +0.238 [0.150, 0.325] compliance
over `single_agent_dqn`, Wilcoxon p=0.0001, 25/3/2 over 30 seeds, with the
honest caveat that 27/30 of those seeds still reach the same collapse,
just less universally than before), does the same architecture still hold
most of that gain under a **federated** training regimen (each gNB trains
locally, weights synced only periodically via FedAvg, no per-step central
data pipe), and what does adding **differential privacy** (DP-SGD-style
per-client gradient clip+noise) cost on top of that?

Two separable costs, reported separately (see Method): the **federation
cost** (centralized gat_ctde vs. FL with no DP noise) and the **privacy
cost** (FL with no DP noise vs. FL at increasing noise multiplier) — a
privacy-utility curve is only interpretable once these two are not
conflated.

**Framing note (per the Block F brief):** all 3 gNBs in this repo's
topology belong to one operator's cluster (`topology.py`'s own
fully-connected, no-real-topology-supplied default), so "federated
learning across clients" is not modelling genuine cross-organization data
separation — that would need multiple *operators* unwilling to pool raw
data, which nothing in this project's config space represents. The
motivation led with here is **operational, not data-sovereignty privacy**:
a real disaggregated O-RAN deployment has per-site near-RT RICs with
limited/intermittent backhaul to any central trainer, so training that
tolerates periodic (not per-step) synchronization is a real deployment
constraint even under one operator. The DP mechanism is still real and
still costs real utility (reported below) — it is evaluated here because
Block F asks for it and because "how much does formal privacy cost on top
of federation" is a well-posed, answerable question regardless of whether
the privacy need itself is synthetic in this single-operator setting.

## Method

1. **`FederatedGatPolicy`**
   (`framework/qoe_oran_framework/marl/fl_ctde_policy.py`, purely additive,
   no frozen-source edits) — each of the `n_agents` gNBs owns a private
   `LocalGatQNetwork` (the same `GATEncoder` + `AgentQHead` pair
   `GatCtdeNetwork` uses, minus the `QMixer`, which `ctde_policy.py`'s own
   `train_step` already doesn't use — dropping it here isolates
   "federated training regimen" as the only architectural difference from
   `gat_ctde`, the same isolation principle M2's `independent_dqn`
   ablation used on a different axis). All clients start from identical
   server-broadcast initial weights. Every client's GAT encoder still
   consumes the full joint node-feature snapshot as input (same
   environment observability every other M2 arm gets — this is public
   per-slice PRB/queue occupancy the environment already broadcasts to
   whichever policy is running, not another client's private training
   data). What never crosses a client boundary: each client's own replay
   transitions (train_step splits every batch by owning-agent, mirroring
   `independent_dqn`'s existing partition) and each client's own gradient
   (clipped, and optionally DP-noised, **before** that client's own
   `optimizer.step()` — the privatization happens at the local update
   itself, not only at the round boundary).

2. **Aggregation** (`framework/qoe_oran_framework/marl/fl_aggregation.py`):
   every `local_steps_per_round` local `train_step()` calls, the server
   averages all clients' online `state_dict`s (`fedavg_aggregate`, equal
   weighting — every client runs the same amount of local training) and
   broadcasts the average back to every client. FedAvg and FedProx (Li et
   al. 2020) share this exact server step; FedProx differs only in the
   client's local loss, which gains a proximal term
   `(mu/2)*||local_params - last_global_snapshot||^2` pulling local
   weights toward the last-broadcast global model
   (`FederatedGatPolicy._local_loss`). **FedProx was smoke-tested (3
   episodes, mu=0.01, no crash, checkpoint round-trips, sane compliance)
   but not run as a full campaign** — Li et al.'s FedProx is specifically
   designed to correct *client heterogeneity* (systematically different
   local data distributions/compute across clients), and this
   environment's topology (`topology.py`'s own documented default: every
   gNB an interchangeable peer, identical slice specs, fully-connected,
   no real adjacency data anywhere in this repo) has none — running a
   FedProx-vs-FedAvg campaign here would be testing a fix for a problem
   this environment doesn't have, which this project's own "never invent
   a number" discipline extends naturally to: don't manufacture a
   heterogeneity scenario nowhere else in the repo specifies, just to have
   something for FedProx to fix.

3. **DP-SGD mechanism**
   (`framework/qoe_oran_framework/marl/dp_sgd.py`): standard per-client
   grad-norm clip to `dp_clip_norm` (=1.0, the same clip norm the non-DP
   arms already use, so DP and non-DP differ only in whether noise is
   added, not in clip behaviour) followed by Gaussian noise
   `N(0, (noise_multiplier * dp_clip_norm)^2)` added to every gradient
   tensor, applied every local `train_step()` for every client (Abadi et
   al. 2016). **noise_multiplier is the primary reported privacy knob.**
   No DP-accounting library is available in this environment (`opacus` —
   checked, not installed); rather than invent an epsilon via an ad hoc
   formula, `dp_sgd.zcdp_epsilon` implements exactly one well-established
   closed-form bound (Bun & Steinke 2016: Gaussian mechanism satisfies
   rho-zCDP with rho=1/(2*sigma^2) per release, composed additively over
   the client's own logged step count with **no subsampling-amplification
   credit claimed**, converted to (epsilon,delta)-DP via their Prop. 1.3).
   This is a real, conservative (loose) upper bound, reported alongside
   noise_multiplier, not in place of it.

4. **Campaign** (`experiments/scripts/m3_privacy_sweep.py`,
   `m3_campaign_analysis.py`): FedAvg arm only (see point 2), 5 noise
   levels {0.0, 0.5, 1.0, 2.0, 4.0} (0.0 = FL-only, no-DP control, isolating
   the federation cost from the privacy cost) x the **same 10 seeds**
   (900-909) that are the first 10 of the (post-fix) 30-seed centralized
   `gat_ctde` campaign — chosen specifically so the "benchmarked against
   the centralized CTDE result" comparison Block F asks for reuses that
   campaign's already-completed per-seed results rather than re-running
   that arm, and so the federation-cost comparison is a true paired
   comparison (same 10 env realizations, only the training regimen
   differs). Same per-run training budget as the centralized campaign
   (300 train / 50 eval episodes) for direct comparability — a shorter FL
   budget would conflate "federation costs compliance" with "less
   absolute training," which this design specifically avoids.
   `local_steps_per_round=50` throughout (one aggregation round roughly
   every 50 replay-buffer samples drawn, i.e. several rounds per training
   episode at this environment's request rate — not swept, since Block
   F's deliverable is the noise-multiplier curve, not a round-frequency
   ablation). Same bootstrap-CI (10,000 resamples) / Wilcoxon-signed-rank
   methodology as `m2_campaign_analysis.py`, applied twice: (a)
   centralized `gat_ctde` vs. FL/no-DP (federation cost), (b) FL/no-DP vs.
   each FL/DP sigma level (privacy cost).

   Pilot timing (2 seeds, reduced 100-train/20-eval budget, sigma=1.0):
   155s total, confirming the harness runs cleanly end to end (DP steps
   logged correctly, aggregation rounds fire, checkpoints round-trip)
   before committing to the full campaign's compute budget.

   **Run twice.** The first full campaign (all 5 sigma levels, run in two
   sittings — sigma 0.0/0.5 then 1.0/2.0/4.0, both committed) used the
   pre-fix `GATEncoder` (no normalization) and is invalidated by the same
   collapse `docs/PAPER5_M2_gat_ctde.md` section 11 diagnoses — that data
   was discovered, not this arm's own separate finding, and is what
   triggered the section-11 investigation in the first place. Deleted
   from the working tree (recoverable from git history, commit
   `8d54b51`) and the full 5-level x 10-seed sweep was re-run from
   scratch against the fixed encoder once section 11's fix was verified.
   The numbers below are the post-fix run.

## Result

**Second superseding note:** the numbers below are from a THIRD full
sweep, after `docs/PAPER5_M2_gat_ctde.md` section 12's per-slice-Q-head
fix (author-requested, after being shown the section-11 LayerNorm-only
fix still collapsed on 27/30 seeds: "find a way to ensure the AI is doing
what it is supposed to do"). `AgentQHead` is imported, not duplicated, by
this arm's `LocalGatQNetwork`, so the fix applied automatically. The
compliance-based sweep below (`m3_campaign_analysis.py`) is kept for
completeness but should be read alongside the correctness-aware sweep
that follows it (`m3_correctness_metrics.py`, same `mean_reward_per_step`
/ `block_precision` definitions section 12 introduced) — section 12
found the compliance metric structurally rewards the wrong behavior, and
that effect shows up here too, more subtly.

**Compliance-based sweep** (n=10 seeds per level, 95% bootstrap CIs,
Wilcoxon signed-rank for paired comparisons):

| Arm / σ | n | mean compliance | 95% CI |
|---|---|---|---|
| centralized `gat_ctde` (same 10 seeds) | 10 | 0.341 | [0.251, 0.435] |
| FL, σ=0.0 (no DP) | 10 | 0.136 | [0.034, 0.247] |
| σ=0.5 | 10 | 0.200 | [0.076, 0.327] |
| σ=1.0 | 10 | 0.244 | [0.089, 0.412] |
| σ=2.0 | 10 | 0.094 | [0.006, 0.202] |
| σ=4.0 | 10 | 0.245 | [0.115, 0.375] |

Federation cost: +0.204 [0.051, 0.361], Wilcoxon p=0.065 (borderline, 7
centralized wins / 0 ties / 3 FL wins). Privacy cost per level vs. FL/
no-DP: none reach significance at n=10 (p=0.31–0.69), and the sign
flips between levels (σ=0.5/1.0/4.0 read as *higher* than no-DP, σ=2.0
*lower*) — **this is noise, not a trend**, and per-seed values are no
longer bit-identical across σ levels the way the pre-per-slice-heads run
showed (per-slice heads gives training enough real sensitivity to the
exact DP noise draw that different seeds land in different basins at
different σ, rather than one collapse-or-not coin flip per seed). Reading
a "privacy-utility curve" directly off this table would be reading noise.

**Correctness-aware sweep** (`experiments/scripts/
m3_correctness_metrics.py`, `mean_reward_per_step` + `block_precision`,
same definitions as `docs/PAPER5_M2_gat_ctde.md` section 12):

| Arm / σ | n | mean_reward_per_step | 95% CI | block_precision | seeds w/ any block |
|---|---|---|---|---|---|
| centralized `gat_ctde` | 10 | **14.362** | [14.075, 14.668] | **1.000** [1.000, 1.000] | 7/10 |
| FL, σ=0.0 | 10 | 13.940 | [13.806, 14.071] | 1.000 [1.000, 1.000] | 6/10 |
| σ=0.5 | 10 | 14.078 | [13.868, 14.315] | 1.000 [1.000, 1.000] | 5/10 |
| σ=1.0 | 10 | 14.194 | [13.825, 14.623] | 1.000 [1.000, 1.000] | 5/10 |
| σ=2.0 | 10 | 12.450 | [10.395, 13.986] | **0.834** [0.612, 1.000] | 9/10 |
| σ=4.0 | 10 | 13.301 | [11.974, 14.212] | **0.584** [0.251, 0.917] | 6/10 |

Federation cost: +0.422 [0.134, 0.727], Wilcoxon p=0.055 (borderline, 6
wins for centralized / 1 tie / 3 for FL) — a real, small, consistent-
direction cost to federating, present but not reaching significance at
n=10. Privacy cost per level vs. FL/no-DP: none individually significant
(p=0.22–0.50), consistent with the small sample, **but `block_precision`
shows a real, visible, monotonic-ish pattern the reward number alone
does not**: perfect precision (1.000) through σ=1.0, then a sharp,
qualitative drop at σ=2.0 (0.834) and σ=4.0 (0.584). Every block at
σ≤1.0 correctly targets mmtc; by σ=4.0, over 40% of blocks land on the
wrong slice.

## Honest conclusion

**Two different, complementary pictures, and the correctness-aware one
is the one to trust for "is the AI still doing the right thing."**
`mean_reward_per_step` (the actual training objective) stays roughly
stable across the whole noise sweep — DP noise does not crater the
policy outright, even at σ=4.0. But `block_precision` — whether the model
still blocks the *correct* slice when it blocks anything — degrades
sharply past σ=1.0. This is a genuinely different, more informative
finding than compliance gave in the pre-per-slice-heads run: **DP noise
doesn't erase differentiated behavior in one shot; it erodes the
precision of that behavior, gradually, past a real threshold** (between
σ=1.0 and σ=2.0 in this environment). That threshold, not a flat "any
noise destroys it" cliff, is the honest privacy-utility story now that
the underlying fix (section 12) gives training enough capacity for a
real, graded response to noise rather than one binary collapse-or-not
outcome per seed.

Federation itself still reads as a small, real, borderline-significant
cost (both on compliance, p=0.065, and reward, p=0.055) rather than
"free" — a change from the section-11 (LayerNorm-only) M3 run, which
found federation entirely free. Plausible reading: section 12's fix gives
the centralized arm's per-step joint training a real edge (finer-grained
gradient signal reaches the shared representation faster than periodic
FedAvg rounds can), whereas the earlier, cruder LayerNorm-only fix wasn't
precise enough for that edge to show up above noise. Neither run's
federation-cost finding reaches significance at n=10 — both are
"borderline, consistent direction," not proven.

The compliance-based sweep in the Result section above is kept for
completeness and cross-reference, but its per-level pattern (σ=0.5/1.0/
4.0 reading higher than σ=0.0, σ=2.0 lower) does not track the
correctness-aware sweep's much cleaner pattern at all — direct evidence,
not just section 12's argument, that compliance is the wrong metric to
read a privacy-utility trend from here.

## What this means for paper #5

**Report `block_precision` as the primary privacy-utility curve, not
compliance.** It shows a real, interpretable, threshold-like effect
(perfect through σ=1.0, degrading sharply after) that the compliance
numbers do not reliably reproduce. Frame the federation-cost finding as
real-but-unproven-at-this-sample-size on both metrics, not "free" (the
section-11 M3 run's claim) — a real change from the earlier writeup,
driven by the more effective fix giving the centralized arm more
headroom to actually use its per-step training advantage.

For M4 (disruption-resilience, Block G): use the same correctness-aware
metrics (`mean_reward_per_step`, `block_precision`), not compliance
alone, from the start — this document had to discover that the hard way
twice. Given the threshold-like pattern found here (graceful up to a
point, then a real drop), M4 should look for a similar threshold in
disruption severity rather than assuming either pure graceful decay or a
pure collapse-on-any-disruption cliff.

## Acceptance status

- [x] No frozen `qoe_oran_framework/` source modified.
- [x] Federation and privacy costs kept separable by design (FL/no-DP
      control arm included, not just DP sigma levels), not conflated into
      a single "FL+DP vs centralized" number.
- [x] Same training budget as the centralized comparison point (Block E),
      not a shorter one that would confound "federated" with "less
      trained."
- [x] Reused Block E's existing per-seed centralized results rather than
      re-deriving a second, possibly-divergent centralized number.
- [x] No DP-accounting library available; used one real, correctly-cited,
      explicitly-conservative closed-form bound rather than inventing or
      approximating an epsilon number.
- [x] FedProx built and smoke-tested (functional), but not campaigned —
      reasoned explicitly from this environment's lack of client
      heterogeneity rather than run anyway for the sake of coverage.
- [x] Discovered the pre-fix campaign's collapse via this arm's own
      bit-identical-compliance anomaly, reported it rather than
      re-running until numbers looked better, and re-ran the full
      5-level x 10-seed sweep from scratch once `docs/PAPER5_M2_gat_ctde.md`
      section 11's fix was verified — did not build the writeup on top of
      invalidated data.
- [x] Noticed the "privacy cost" paired diff had the wrong sign (DP
      noise *raising* mean compliance) and traced it to the same
      underlying collapse mechanism rather than reporting a
      privacy-utility curve at face value — the metric rewards the
      degenerate policy over the correct one, so naive compliance-vs-ε
      reporting would have been actively misleading.
- [x] Verified the bit-identical-across-σ>0 pattern against real
      evidence (nonzero, consistent `dp_step_count`; block-count
      cross-check) rather than either assuming a bug or assuming it was
      fine.
- [x] Stated plainly where results don't reach statistical significance
      at n=10 (both the federation-cost and privacy-cost paired tests),
      rather than leading with the qualitative pattern as if it were
      confirmed at this sample size.
- [x] Re-ran the full 5-level x 10-seed sweep a THIRD time after
      section 12's per-slice-heads fix (author-requested, not
      self-initiated scope creep) rather than leaving the writeup on the
      now-superseded LayerNorm-only numbers.
- [x] Built the same correctness-aware metrics (`mean_reward_per_step`,
      `block_precision`) this arm needed for the same reason section 12
      needed them, reusing the exact function (`m2_correctness_metrics.
      per_seed_metrics`) rather than a re-implemented, possibly-divergent
      copy.
- [x] Reported that federation is no longer "free" under the better fix
      (a real change from the section-11 M3 run's own conclusion) rather
      than leaving the earlier, now-inconsistent claim standing.
- [x] Noticed the compliance-based and correctness-aware sweeps tell
      visibly different stories at the SAME data and reported both,
      naming the correctness-aware one as more trustworthy with a
      concrete reason (the compliance curve's sign flips between levels
      in a way the correctness curve's monotonic-ish pattern does not),
      rather than picking whichever story was more convenient.
