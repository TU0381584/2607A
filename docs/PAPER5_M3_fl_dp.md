# Paper #5, M3 — federated GAT-CTDE + differential privacy

**Question:** starting from the Block E-verified GAT-CTDE result (centralized
per-step joint training over the 3-gNB stress environment beats both
ablations with a statistically solid margin — see
`docs/PAPER5_M2_gat_ctde.md` section 9: +0.284 [0.214, 0.355] compliance
over `single_agent_dqn`, Wilcoxon p<0.0001, 27/3/0 over 30 seeds), does the
same architecture still hold most of that gain under a **federated**
training regimen (each gNB trains locally, weights synced only
periodically via FedAvg, no per-step central data pipe), and what does
adding **differential privacy** (DP-SGD-style per-client gradient
clip+noise) cost on top of that?

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
   (900-909) that are the first 10 of Block E's 30-seed centralized
   `gat_ctde` campaign — chosen specifically so the "benchmarked against
   the centralized CTDE result from Block E" comparison Block F asks for
   reuses Block E's already-completed per-seed results rather than
   re-running that arm, and so the federation-cost comparison is a true
   paired comparison (same 10 env realizations, only the training regimen
   differs). Same per-run training budget as Block E (300 train / 50 eval
   episodes) for direct comparability — a shorter FL budget would
   conflate "federation costs compliance" with "less absolute training,"
   which this design specifically avoids. `local_steps_per_round=50`
   throughout (one aggregation round roughly every 50 replay-buffer
   samples drawn, i.e. several rounds per training episode at this
   environment's request rate — not swept, since Block F's deliverable is
   the noise-multiplier curve, not a round-frequency ablation).
   Same bootstrap-CI (10,000 resamples) / Wilcoxon-signed-rank methodology
   as `m2_campaign_analysis.py`, applied twice: (a) centralized `gat_ctde`
   vs. FL/no-DP (federation cost), (b) FL/no-DP vs. each FL/DP sigma level
   (privacy cost).

   Pilot timing (2 seeds, reduced 100-train/20-eval budget, sigma=1.0):
   155s total, confirming the harness runs cleanly end to end (DP steps
   logged correctly, aggregation rounds fire, checkpoints round-trip)
   before committing to the full campaign's compute budget.

## Result

TODO(MEASURE) — full campaign (`experiments/scripts/m3_privacy_sweep.py`)
running in the background as of 2026-08-13; this section is filled in from
`experiments/scripts/m3_campaign_analysis.py`'s real output once it
completes, not estimated in advance.

## Honest conclusion

TODO(MEASURE) — pending the Result section above.

## What this means for paper #5

TODO(MEASURE) — pending the Result section above; in particular, this
determines how M4's disruption-resilience harness should frame the
FL/DP arm (Block G explicitly runs disruption evaluation against
"policies from Blocks E and F," so M4 cannot start in earnest on the FL
arm until this verdict is in).

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
- [ ] Result / Honest conclusion / paper-#5 implications — pending
      campaign completion (see Result section).
