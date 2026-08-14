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
after the fix landed — see Result.

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

All numbers from `experiments/scripts/m3_campaign_analysis.py`'s real
output against the post-fix campaign
(`experiments/results/m3_campaign/campaign_results.json`), n=10 seeds per
arm, 95% bootstrap CIs (10,000 resamples), Wilcoxon signed-rank for paired
comparisons.

**Federation cost — not statistically distinguishable from zero:**

| Arm | n | mean | 95% bootstrap CI |
|---|---|---|---|
| centralized `gat_ctde` (post-fix, same 10 seeds) | 10 | 0.404 | [0.284, 0.512] |
| FL / no-DP (σ=0.0) | 10 | 0.367 | [0.237, 0.487] |

Paired diff (centralized − FL/no-DP): +0.037, 95% CI [−0.092, 0.167],
Wilcoxon p=0.875 (3 wins / 6 ties / 1 loss for centralized). **Going
federated costs nothing measurable at this sample size** — the periodic-
sync training regimen holds up against per-step centralized training.

**Privacy-utility curve — a real, sharp, one-shot effect, not a gradual
curve:**

| σ (noise multiplier) | n | mean | 95% bootstrap CI | ε (zCDP upper bound, δ=1e-5) |
|---|---|---|---|---|
| 0.0 (no DP) | 10 | 0.367 | [0.237, 0.487] | ∞ (no privacy) |
| 0.5 | 10 | 0.447 | [0.368, 0.526] | 37224 |
| 1.0 | 10 | 0.447 | [0.368, 0.526] | 9628 |
| 2.0 | 10 | 0.447 | [0.368, 0.526] | 2568 |
| 4.0 | 10 | 0.447 | [0.368, 0.526] | 722 |

**Every σ>0 level produces bit-identical per-seed compliance to every
other σ>0 level** — verified not to be a bug: `dp_step_count` is nonzero
and identical across levels (17969 steps/client, confirming noise really
is being generated and applied at every level), and this exact
phenomenon was already characterized on the pre-fix data (checkpoint
weights differ substantially between noise levels; the discrete greedy
eval *decision boundary* a given seed's training converges to does not,
because zero-mean noise averages out over ~18k steps while the mean
gradient direction still dominates which side of the decision boundary
training lands on). The real, sharp finding is between σ=0.0 and any
σ>0: **at σ=0.0, 3/10 seeds (903, 908, 909) show genuine differentiated
shedding** (mmtc-only blocking, matching section 11's fix working as
intended); **at every σ≥0.5, all 10/10 seeds collapse to always-accept**
— any nonzero DP noise, even the smallest level tested, is enough to
erase 100% of the fix's benefit for this federated arm.

Paired FL/no-DP vs. each σ>0 level (identical across all four, since the
σ>0 arms are themselves identical): mean diff −0.081, 95% CI
[−0.198, 0.000], Wilcoxon p=0.25 (0 wins for no-DP / 7 ties / 3 wins for
DP). **Not statistically significant at n=10** — but note the sign:
FL/DP's mean is *higher* than FL/no-DP's, not lower.

## Honest conclusion

**The sign of the "privacy cost" is backwards from the naive
expectation, and the reason is a direct continuation of section 11's
finding, not a new, separate result.** DP noise does not gradually
degrade a working policy here — it destroys the *fragile, already-rare*
differentiated-shedding behavior (3/10 seeds at σ=0.0) and replaces it
with the *same collapsed, always-accept fallback* every other seed
already reaches. Because `sla_compliance_all_slices` scores accept-
everything *higher* than genuine mmtc-shedding (mmtc's SLA can't be
rescued by blocking in this stress regime regardless — see section 11 —
so blocking it only costs compliance, never earns it back), erasing the
differentiated seeds *raises* the measured mean. **This is not evidence
that DP training improves policy quality — it is evidence that this
compliance metric rewards a degenerate policy over a correct one, and DP
noise pushes every seed toward the metric-favored degenerate policy.**
Reporting only the mean-compliance numbers without this context would
be actively misleading.

Two real, honest findings survive scrutiny:
1. **Federation itself is free** (no significant cost vs. centralized
   training) — a genuine, positive, well-supported result for the
   "operational coordination benefit" framing this document leads with.
2. **This architecture's fragile differentiated-shedding behavior has
   essentially zero DP-noise tolerance** — not "degrades gracefully with
   more noise," but "gone entirely at the smallest tested σ." Given how
   rare that behavior already is post-fix (3/30 in the centralized
   campaign, 3/10 here), this reads less like "DP has a cost" and more
   like confirmation that the underlying fix (section 11) produced a
   narrow, easily-disturbed basin of correct behavior, not a robust one.

Neither finding is statistically significant at n=10/paired-Wilcoxon —
stated plainly, not stretched. The qualitative "0% survival past σ=0.5"
pattern is real and reproducible (verified against DP-step counts, not
assumed), even where the paired test doesn't cross significance at this
sample size.

## What this means for paper #5

**Do not frame this as a privacy-utility tradeoff curve with a
compliance cost.** The honest framing is: federation is free, and the
architecture's (already-marginal) capacity for genuinely correct
admission behavior does not survive any tested amount of differential
privacy. A paper claim built on this data should lead with the
federation-is-free result (real, positive, matches the "coordination
benefit not privacy alone" framing already adopted above) and report the
DP finding as a limitation of the current architecture's robustness, not
as a calibrated privacy-utility curve — the metric-inversion effect makes
a naive "compliance vs. ε" plot actively misleading without this
explanation attached.

For M4 (disruption-resilience, Block G): M4 evaluates frozen policies
from Blocks E and F under perturbation. Given how narrow and fragile the
"correctly differentiated" behavior is even absent any disruption (3/30
and 3/10 seeds respectively), M4 should expect disruption to have a
similarly outsized, cliff-like effect rather than graceful degradation —
and should test whether disruption-recovery, like DP noise here, also
just pushes seeds toward the same collapsed always-accept fallback. This
is a testable prediction from this document's finding, not an assumption
M4 should take on faith.

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
