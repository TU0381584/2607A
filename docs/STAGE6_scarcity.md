# Stage 6 — Demand-driven scarcity, assessed and deferred

**Goal, as scoped:** replace the congested/URLLC-preservation scenario's
artificial scarcity mechanism (a fiat `shared_pool_prb` budget in the
offline simulator) with genuine demand-driven contention — real UEs
actually competing for capacity on the physical rig — if it can be
sustained; otherwise document the ceiling honestly rather than fake it.

**Finding: not attempted live. The resource constraint is real, checked
directly, not assumed.**

## What the current mechanism actually is

`experiments/scripts/shared_pool_kpm_source.py`'s `SharedPoolCongestedKpmSource`
adds a real shared-PRB-pool constraint ACROSS slices in the offline
simulator: each slice's requested PRBs (`min(demand, ceiling)`) are
summed per gNB, and if the sum exceeds `shared_pool_prb` (8.0 in the
evaluation actually reported — `metrics_stage2.py`'s `congested_metrics`
call), every slice's served amount is scaled down proportionally. This is
a genuine constraint in the sense that the policy can't just "raise every
ceiling and win" — but the scarcity itself is a single scalar parameter,
not real UEs generating real, independent demand that happens to
oversubscribe the cell.

## Why real UEs weren't added

1. **No spare per-slice UE configuration exists.** Only 3 UE configs are
   present (`ORANSlice/oai_ran/targets/PROJECTS/GENERIC-NR-5GC/CONF/nrUE_slice{1,2,3}.conf`),
   one per slice, each already in use. A genuine 2nd UE on any slice
   needs a new IMSI, a new subscriber record, a new network namespace,
   and a new PDU session wired through the same real gNB — new
   engineering work, not a config-value change.
2. **The rig's memory headroom is tight even at the current 3-UE scale.**
   Checked directly, right now, with nothing running:
   `free -h` → 1.4 GiB free / 3.0 GiB available (out of 7.4 GiB total).
   `CAMPAIGN_LOG.md`'s own history records this rig running with only
   ~195 MiB free / 1.6 GiB available once the full 3-UE stack, traffic
   generators, and Docker core were all up simultaneously. Each
   `nr-uesoftmodem` process observed at ~400–500 MB RSS this session.
   Adding even one more real UE is a genuine risk of pushing an
   already-tight system into swap or OOM territory, on a rig that has
   already shown real instability today (repeated `iperf3` crashes, at
   least one health-check failure requiring a restart) during ordinary
   3-UE operation.
3. **This is the explicitly lowest-priority stage in the original rework
   plan** ("Stage 6 offline-only is survivable" — the plan's own
   sequencing notes), unlike Stage 5's recalibration work, which gated
   everything downstream. Spending further live rig time here, on a rig
   already used heavily across Stages 3–5 in this session, was judged
   not worth the resource risk relative to the payoff.

## Decision

**The offline synthetic shared-pool mechanism stays as the
congested-scenario's method**, clearly labeled in the paper as a
simulated (not live-hardware) scarcity constraint — which the manuscript
already does honestly (Section IV-C's own framing distinguishes the
congested scenario from the live campaign). No paper claim changes as a
result of this stage; this document exists so the decision not to
attempt live demand-driven scarcity is recorded and justified, not
silently absent.

## Acceptance status

- [x] Demand-driven live scarcity assessed for feasibility (UE config
      availability, real memory headroom) before any rig time was spent.
- [x] Found genuinely infeasible at acceptable risk on this rig's current
      resource envelope — not assumed, checked directly (`free -h`,
      config directory listing).
- [x] Documented plainly rather than faked or silently skipped, per the
      stage's own explicit fallback instruction.
- [ ] Live demand-driven scarcity itself — not done; would need new UE
      provisioning work and a memory-headroom fix (or a smaller
      incremental test, e.g. one additional UE) as a genuinely separate,
      future undertaking if ever revisited.
