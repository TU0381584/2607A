# Stage 8 — Bridge audit

Cross-checks paper #4's manuscript (`paper_conf/main.tex`, current
lean/house-style draft, 4pp) against the actual PDFs of papers #1
(MECON), #2 (ICRAIE), and #3 (IEEE Access review) on
`~/Desktop/`, and against the framework's real config/results
artifacts. One real inconsistency was found and fixed during this
audit (Table 1 below); everything else checked out.

---

## 1. Backward traceability: paper #4 vs. paper #3 (the review)

| Claim in #4 | Checked against #3 | Status |
|---|---|---|
| "validated entirely in simulation" describes the majority of the field's DRL-driven SAC work | #3's abstract: derives "recurring gaps... namely the absence of direct QoE inference, privacy preservation, topology awareness, and validated deployment" from a 67-study systematic review | Consistent |
| #3 provides "preliminary experimental grounding" from #1/#2 | #3's contribution (4): "Preliminary experimental grounding from our two prior studies on slice admission control and load balancing, establishing an incremental validation path" | Consistent, verbatim match of intent |
| Future work: "multi-agent, multi-gNB setting proposed in our systematic review" | #3's contribution (5): "A novel QoE-aware GAT-MARL-FL framework for MEC-enabled O-RAN, combining a graph attention network (GAT) encoder, distributed multi-agent DRL, and privacy-preserving FL" | Consistent as an intentional simplification (paper #4's lean draft doesn't name GAT/FL explicitly, to keep jargon down; the citation carries the reader to the specifics) |
| #3's own five gaps (QoE inference, privacy, topology, validated deployment, scalability) | Paper #4's lean draft no longer carries a gap-by-gap scorecard (removed in the house-style rewrite as review-paper-specific apparatus) | **Deliberate scope change, not an inconsistency** — the gap-scorecard content still exists in `docs/STAGE7`-era git history (commit `e5795e1`) if a fuller version is ever needed for a different venue |

## 2. Forward traceability: paper #4 vs. paper #5 (planned journal follow-up)

Paper #4's "Conclusion and Future Work" commits to three items, each of
which is a direct precondition for #5 rather than something #5 itself
needs to re-derive:

1. **Fully powered live re-evaluation under the corrected QoE
   calibration** — #5 should not need to re-diagnose the calibration
   bug described in Section IV-B; it can cite paper #4 and start from
   corrected numbers (still pending full statistical power per
   `docs/STAGE5_recalibration.md`).
2. **Live reproduction of the congested, multi-slice scenario** —
   currently offline-only (Section IV-C); #5's multi-agent framework
   will need this as a live baseline to compare against, not just the
   offline numbers in Table II of #4.
3. **Multi-agent, multi-gNB extension** — this **is** #5, scoped
   directly from paper #4's single-agent, single-gNB result and #3's
   proposed GAT-MARL-FL framework. No gap between what #4 promises and
   what #3 already proposed.

No claims in #4 pre-empt or foreclose #5's contribution; #4 is
explicitly scoped as single-gNB/single-agent throughout (see
`\cite{survey-paper3}` used only for the forward-looking framework
mention, never as though multi-agent work were already done).

## 3. Corpus continuity, papers #1–#4 (#5 not yet written)

| Item | #1 / #2 | #4 | Status |
|---|---|---|---|
| Author list, order, affiliation, ORCID | Kunasegran, Pang* (corresponding), Phang; Taylor's University; ORCIDs 0009-0005-3169-1873 / 0000-0001-8407-5648 / 0000-0002-7877-8766 | Identical, copied verbatim from the PDFs | Consistent |
| Acknowledgment | MOHE FRGS/1/2024/ICT06/TAYLOR/02/1 (both #1 and #2; #2 additionally thanks JESTEC/ICRAIE sponsorship, specific to that venue) | MOHE FRGS line only (JESTEC line correctly *not* copied — it's ICRAIE-specific, and #4's venue isn't fixed) | Consistent, correct scoping |
| Spelling convention | British throughout ("optimise," "realise," "prioritise," "modelled," "behaviour") | Checked with a targeted grep for American variants (`optimize`, `realize`, `behavior`, etc.) — zero matches | Consistent |
| Reward equation structure | #1 eq.(2): $r=\sum_k(\omega_k R_k-\lambda_k)-\mu C_t\|a_t\|_1$; #2 eq.(2) adds $-\beta F_t$ for its load-balancing term | #4 eq.(2): same SLA base plus $+\eta\,\mathrm{MOS}_t$ for the QoE-aware variant ($\eta=0$ recovers the SLA-only reward) | Consistent extension pattern (swap #2's fairness term for a QoE term, matching the paper's own stated purpose) |
| State formulation | #1 eq.(1): 2 slices, no fairness term; #2 eq.(1): 3 slices + fairness indicator $F_t$ | #4 eq.(1): general form over $\mathcal{S}=\{$eMBB,URLLC,mMTC$\}$, no fairness term (no LB objective in #4) | Consistent |
| gNB capacity | #1/#2 (pure simulation): 100 PRBs nominal, by construction of their `RANEnv` | #4 (real hardware): **found stated as "100 PRBs" in the first house-style draft — wrong.** The real rig's physical cell is 106 PRBs at $\mu=1$ numerology; "100" is only the framework's internal ceiling-ratio normalisation constant ($B$ in `env.py`), not a PRB count. This is the same units confusion Stage 5 diagnosed and fixed in the QoE calibration (`docs/STAGE5_recalibration.md`, Bug 1) — it had crept back in during the house-style rewrite because 100 PRBs is the correct, unrelated convention in #1/#2's *simulated* environment. **Fixed during this audit**: Section III-A now says 106 PRBs ($\mu=1$) for the physical cell, and the state-equation's $B$ is now defined as "the total resource budget on the ceiling's normalised 0–100 scale" rather than implying it equals the PRB count. |
| Live SLA numbers (Table I) | N/A (not in #1/#2) | 73.7/73.4/73.8 (baseline), 83.1/73.4/73.9 (static-at-cap), 100.0/100.0/100.0 (both DQN arms) | Re-verified against `docs/stage2_metrics_raw.json` / `docs/stage3_metrics_raw.json` this session — exact match |
| Congested-scenario numbers (Table II) | N/A | 22.6/34.5/19.0 (baseline), 30.9/7.9/9.2 (DQN-SLA), 27.0/8.2/10.8 (DQN-QoE); utility 27.2/21.0/19.0 | Matches `docs/STAGE2_metrics.md` exactly; utility arithmetic re-derived independently this session (weighted eMBB loss $\approx$93.1 vs. weighted URLLC gain $\approx$41.5 vs. weighted mMTC loss $\approx$2.9, net $\approx-6.2$pp on $U$ — matches the observed 27.2$\to$21.0 drop) |
| Engineering numbers | N/A | E2 round trip 0.57 ms median, DQN inference under 70 $\mu$s, under 20,000 parameters | Matches `docs/STAGE4_instrumentation.md` (0.566 ms, 67.9/68.3 $\mu$s, 18,562 params) |
| Fisher exact significance | N/A | $p=0.0149$ | Matches `docs/STAGE3_oracle.md`'s combined (25/25 vs. 11/15) comparison |

## 4. Acceptance status

- [x] Backward traceability to #3 checked claim-by-claim; one
      deliberate scope reduction noted (gap scorecard dropped per
      house-style rewrite), no factual inconsistency.
- [x] Forward traceability to #5 checked: #4's three future-work items
      are direct, non-overlapping preconditions for #5; no claim in #4
      forecloses #5's contribution.
- [x] Corpus continuity checked across authorship, acknowledgment,
      spelling convention, equation structure, and every numeric claim
      in #4 against its source artifact.
- [x] One real inconsistency found (gNB capacity stated as 100 PRBs,
      a carry-over from #1/#2's simulated-environment convention
      rather than this rig's real 106-PRB cell) and fixed directly in
      `paper_conf/main.tex` during this audit, not left for Stage 9.
