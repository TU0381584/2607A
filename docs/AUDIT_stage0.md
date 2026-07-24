# Stage 0 Audit — Paper #4 (conference draft)

Read-only audit. Nothing in `paper_conf/`, `experiments/`, or `framework/`
was modified to produce this document. `ACTION_PLAN_conference_and_journal.md`
was named in the Stage 0 brief but does not exist anywhere in this repo
(`find . -iname "ACTION_PLAN*"` returns nothing) — flagged as missing, not
silently skipped.

Source snapshot: `paper_conf/main.tex` (447 lines, commit `688a510`),
`paper_conf/refs.bib` (15 entries), review PDF at
`/home/kmanojp/Desktop/IA_CROTUCDNS5GN_corrected.pdf`.

---

## 1. File inventory — section-by-section completion state

| Section | Lines | State |
|---|---|---|
| Title/author | 15–20 | Written; 2 `\authorTODO{}` markers remain (author order confirmation, corresponding-author email) — these are genuinely author-only decisions, not content gaps. |
| Abstract | 24–51 | Written, numbers current as of the epsilon-fix retrain (commit 688a510). |
| I. Introduction | 57–102 | Written. 4 contributions + explicit scope statement. |
| II. System Architecture and Testbed | 104–135 | Written. |
| III. Problem Formulation | 137–182 | Written. 3 equations (SLA reward, QoE reward, shared-pool rationing). |
| IV-A. Live Testbed | 184–227 | Written. |
| IV-B. Sim-to-Real Transfer | 229–256 | Written; explicitly frames the eMBB-only transfer as a limitation, not a result. |
| IV-C. Congested/URLLC-Prioritized Admission | 258–338 | Written; includes an in-text correction narrating the epsilon-decay bug and retraining (lines 293–331). |
| V. Discussion (5-gaps table + next-steps) | 340–398 | Written. |
| VI. Limitations | 400–414 | Written. |
| VII. Conclusion | 416–441 | Written. |
| Tables I–IV | `tables/*.tex` + inline | All populated, no placeholder values found. |
| Figures 1–5 | `figures/*.pdf` | All regenerated from current data (see §5 below for exactly which script produced which). |

No section is a stub. The two `\authorTODO` markers are the only remaining
placeholders, and both require Manoj's input specifically (not derivable
from any artifact).

---

## 2. Claim-to-artifact trace table

Every numeric claim in the current draft, traced to its source.

| Claim (as written) | Value | Source file | Source key/line | Reproducible? |
|---|---|---|---|---|
| 67 core studies, 6,286 records screened | 67 / 6,286 | review PDF | PRISMA figure, abstract | Y (external artifact, not regenerable from this repo) |
| Paper #1: URLLC SLA ≈99.5%, 100 episodes | 99.5% | review PDF | Table row for ref [8] (line 869) | Y (external) |
| Paper #2: 99.63/88.41/97.64% per-slice | 99.63/88.41/97.64 | review PDF | Table row for ref [9] (line 872) | Y (external) |
| Live: DQN 100.0±0.0% vs baseline 73.6±18.6% | 100.0/73.6/18.6 | `experiments/RESULTS_REPORT.md` | line 47 (table row) | Y — traces to `experiments/results/live_campaign/*/omega_log.jsonl` |
| Baseline worst episode 0.0% | 0.0 | `experiments/RESULTS_REPORT.md` | same table, "Worst ep." column | Y |
| Baseline per-seed 60.3/100.0/60.7% | 60.3/100.0/60.7 | `experiments/RESULTS_REPORT.md` | §4 table (line ~102–105) | Y |
| Bootstrap 95% CI [53.4, 93.4] | 53.4/93.4 | `experiments/RESULTS_REPORT.md` | (CI reported alongside the mean±std row) | Y |
| eMBB/URLLC MOS ≈1.2–1.5, mMTC ≈4.6–4.8 | Table II values | `paper_conf/tables/table2_results.tex` | rows | Y — traces to same omega logs, `mean_mos_by_slice` field |
| Sim2real: eMBB 100→100, URLLC/mMTC 0→100 | 0/100 | `experiments/plots/fig8_sim2real_parity.py` stderr output | script prints exact values on run | Y, regenerated this session |
| Congested baseline 22.6/34.5/19.0 | unchanged pre/post epsilon-fix | `experiments/results/congested_vs_baseline_v6_epsilonfix/results.json` | `baseline.compliance_pct` | Y |
| Congested DQN(SLA) 30.9/7.9/9.2 | post-fix | same file | `dqn_sla.compliance_pct` | Y |
| Congested DQN(QoE) 27.0/8.2/10.8 | post-fix | same file | `dqn_qoe.compliance_pct` | Y |
| A2C epsilon-schedule-unaffected claim | epsilon N/A for A2C | `framework/drl_slicing/oranslice_drl/drl_policy.py` | A2CPolicy has no `self.epsilon` attribute at all | Y — verified by grep, not assumed |
| DQNAdmissionPolicy epsilon=1.0 after ~18,000 steps | 18,000 | direct checkpoint inspection this session (`torch.load(...)['train_step_count']`) | `experiments/results/offline_congested_preepsilonfix/{sla,qoe}/seed256/dqn/checkpoint.pt` | Y, but **this artifact is a backup of a checkpoint no longer used for any reported number** — kept only for provenance |
| Ratio=4 permits 15–22 real PRBs (non-linear mapping) | 15–22 | `experiments/CAMPAIGN_LOG.md` (Phase 1 calibration section) | not independently re-verified this session | Y (pre-existing artifact, not re-derived) |
| Contention gate: 411→6.6M→2.5M bytes | — | referenced in earlier draft, **not present in the current main.tex** (Table III / contention-gate table was cut in the 6-page rewrite) | `experiments/logs/phase1/embb_final_20260716_204703.jsonl` | Y, but currently unused in the paper |

**Everything traced.** No number in the current draft is untraceable to a
committed artifact. The one item worth flagging: the epsilon=1.0
finding is reported in prose (line 297–298) but the checkpoint that
proves it is a backup directory (`offline_congested_preepsilonfix/`),
not something a fresh clone would regenerate by running the pipeline
end-to-end — a reader auditing this claim needs that specific backup
directory, which is committed (commit `688a510`) but not obviously
signposted from the paper text itself. Low severity, but worth a
one-line pointer in the reproducibility notes if one gets added back.

---

## 3. TODO(MANOJ) / VERIFY: markers, file:line

**main.tex:**
- `main.tex:18` — `\authorTODO{author order/inclusion matches papers #1/#2/#3's byline...}`
- `main.tex:19` — `\authorTODO{confirm corresponding author and email...}`

**refs.bib** (6 markers, all pre-existing, none touched this session):
- `refs.bib:10` — `oranslice` entry: VERIFY citation format WinesLab/Northeastern prefers
- `refs.bib:17` — `oai` entry: VERIFY canonical OAI citation vs. project URL
- `refs.bib:25` — `oran-e2ap` entry: VERIFY exact spec number/version
- `refs.bib:33` — `oran-e2sm-kpm` entry: VERIFY exact spec number/version
- `refs.bib:41` — `itu-p1203` entry: VERIFY exact edition/amendment
- `refs.bib:49` — `iqx-hypothesis` entry: VERIFY exact volume/issue/page
- `refs.bib:91` — `survey-paper3` entry: VERIFY final volume/issue/DOI (paper #3 is stated as accepted; this is purely a publication-metadata lag, not a content risk)

None of these seven items is a research-content gap — they are bibliographic
metadata that only becomes knowable once each source is looked up or #3
formally publishes. None should block submission by itself, but all seven
should be closed before camera-ready.

---

## 4. Metric definitions — verbatim from code, with prose divergence flagged

**SLA-only reward** (`framework/qoe_oran_framework/reward.py:103-124`,
`compute_step_reward`):
```python
service_term += spec.priority_weight * spec.accept_reward * n_accepted
if violations.violated.get(slice_id, False):
    violation_term += spec.violation_penalty
...
congestion_term = weights.congestion_coeff * mean_congestion * total_accepted
reward = service_term - violation_term - congestion_term
```
Matches `main.tex` Eq. 1 (`eq:sla`) term-for-term. **No divergence.**

**QoE-aware reward** (`reward.py:182-232`, `compute_qoe_reward`):
```python
mos_norm = (mean_mos - 1.0) / 4.0   # [1,5] -> [0,1]
sla_viol = min(1.0, sum(severities)/len(severities))  # severities = max(0, -margin)
reward = qoe_weights.alpha * mos_norm - qoe_weights.beta * cost - qoe_weights.gamma * sla_viol
```
Matches `main.tex` Eq. 2 (`eq:qoe`) structurally. **Divergence found:**
main.tex cites this as "eq.~9 of survey-paper3". Verified against the
actual review PDF (`/home/kmanojp/Desktop/IA_CROTUCDNS5GN_corrected.pdf`,
extracted text line 384): the QoE reward `rt = α·MOS(QoS) − β·cost − ξ·SLAviol`
is the review's **Eq. (3)**, not Eq. (9). Eq. (9) in the review is the
FedAvg-FedProx federated-learning objective (`min Fk(θ) + ||θ-θ(r)||²`,
extracted text line 505) — an entirely different equation. **This is a
real, confirmed citation error, not a stylistic quibble; a reviewer who
opens #3 to check Eq. 9 will find FedAvg, not the QoE reward.** Additionally,
the review paper uses **ξ (xi)** for the SLA-violation coefficient
(line 384); `main.tex` uses **γ (gamma)** for the same term. Both should be
corrected together (§ recommendation below).

**MOS mapper** (`framework/qoe_oran_framework/qoe_mapper.py:92-107`, `iqx_mos`):
```python
q_deg = coeffs.gamma * latency + coeffs.delta * packet_loss + coeffs.epsilon / throughput
mos = coeffs.alpha - coeffs.beta * np.log1p(np.maximum(q_deg, 0.0))
return np.clip(mos, 1.0, 5.0)
```
This is `MOS = a − b·ln(1+Q_deg)` — a **logarithmic Weber–Fechner-law
form**. Verified against the review PDF (line 458): this is the review's
own **Eq. (4)**, explicitly named "Weber–Fechner Law (WFL)" in the
surrounding prose (line 442), NOT the classic exponential IQX form
(`a·exp(−b·Q)+c`) that `iqx-hypothesis`~\cite{} and the paper's own prose
("IQX closed-form prior") imply. **Confirmed, real divergence**, present
in three places: (1) the code's own module docstring
(`qoe_mapper.py:4-8`) mislabels this formula as "eq.(3)" of the review —
it is actually eq.(4); (2) `main.tex` line 161 calls it an "IQX closed-form
prior" when the implemented and review-cited form is Weber-Fechner: IQX
and WFL are different QoE-mapping families in the QoE literature (both
cited in the review's related-work survey, lines 440-452, as *alternative*
approaches — the review explicitly chose WFL, not IQX); (3) the
`\cite{iqx-hypothesis}` reference (Fiedler et al., "The IQX Hypothesis") is
therefore citing the wrong family of prior work for the formula actually
in use.

**MOS scale.** `main.tex` Eq. 2 declares $\widehat{\text{MOS}}_t\in[0,1]$.
Table II reports raw values 1.22–4.78. These are NOT contradictory — the
code normalizes (`mos_norm = (mean_mos-1)/4`) before it enters the reward
— but **the paper text never states this normalization step**, so a reader
comparing Eq. 2's $[0,1]$ claim against Table II's 1–5-scale numbers has no
way to reconcile them without reading the code. Confirmed gap, cheap fix
(one clause).

**SAC acronym collision — NOT CONFIRMED.** The Stage 1 brief's claim
that review Table 6's "Sanaei et al." row abbreviates "soft actor–critic"
as "SAC" does not hold up against the actual PDF text: that row spells
out "Energy-aware soft actor–critic + STP" in full (line 573) and never
abbreviates it. A full-text search of the extracted PDF for the literal
string `(SAC)` returns exactly one hit, at line 823: "slice admission
control (SAC)" — the review paper does not appear to define SAC as
Soft Actor-Critic anywhere in the extracted text. **Recommend NOT
"fixing" this** (there is nothing to fix) but noting in Stage 1 execution
that this specific sub-claim didn't verify, so the acceptance test isn't
"searched and replaced" but "searched, and found already
non-colliding" — worth confirming with a second full-text pass (OCR of
figures/tables text embedded as images can be lossy) before fully
closing it.

**Paper #1 title — resolved, not a TODO.** Review PDF reference list
entry `[8]` (extracted text line 1046): *"Optimising resource allocation
in 5G networks: Balancing URLLC and eMBB traffic under gNB congestion,"
2025 Multimedia University Engineering Conference (MECON), 2025, pp.
1–6.* Current `refs.bib` `mecon-paper1` entry reads *"...in 5G **RAN**
Networks..."* — the word "RAN" is not in the accepted paper #3's own
reference list. This is fully resolved (not a VERIFY item): drop "RAN"
from the title in `refs.bib`.

---

## 5. Offline simulator vs. live rig — exact config diff

Three offline training configs exist; each targets a different section:

| Config | Used for | eMBB cap | URLLC cap | mMTC cap | Headroom vs. nominal |
|---|---|---|---|---|---|
| `saclb_campaign.yaml` | **Live eval** (Sections IV-A/IV-B), nominal_ratio 3/2/2 | 12 | 4 | 3 | Real headroom (4×/2×/1.5×) |
| `saclb_offline_campaign.yaml` | **Offline training** for the IV-A/IV-B checkpoints | 3 | 2 | 2 | **Zero** (cap == nominal for all three slices) |
| `saclb_offline_congested_v1.yaml` | Section IV-C (congested) training | 12 | 4 | 3 | Matches live — this was the fix applied when IV-C's train/eval mismatch was diagnosed |

This is the exact, already-disclosed root cause of Section IV-B's
sim-to-real finding (eMBB transfers, URLLC/mMTC don't): the config that
actually trained the deployed checkpoints pins ALL THREE slices at
zero elastic headroom, not just URLLC/mMTC as the prose implies at first
read — eMBB's offline cap (3) also equals its own nominal_ratio (3) in
`saclb_offline_campaign.yaml`. eMBB's *live* transfer success is therefore
not because its offline training had adequate headroom either; it
transfers exactly by coincidence of that slice's demand/ceiling
relationship, not because eMBB's offline environment was "representative"
in the way the current prose (line 249, "whose offline config *is*
calibrated against live-representative headroom" — this exact sentence
was cut in the 6-page rewrite, but the underlying claim in the current
text, "eMBB transfers exactly," still implies a favorable-config
explanation that the actual `saclb_offline_campaign.yaml` file does not
support). **Recommend re-checking this specific claim in Stage 5**
(recalibration) rather than accepting the current prose's implicit
explanation.

Other parameters (`B=100`, `Lmax=10`, `step_seconds=5.0`,
`steps_per_episode=60`, `synthetic_arrivals_per_step=2`) are identical
across all three configs — confirmed via direct diff, not assumed.

Section IV-C's config additionally adds a genuine shared-PRB-pool
constraint (`shared_pool_prb=8`, `SharedPoolCongestedKpmSource`) absent
from the other two, which compute each slice's served PRBs
independently — this is the mechanism Eq. 3 (`eq:pool`) describes, and it
is real (verified in `experiments/scripts/shared_pool_kpm_source.py`),
not a paper-only construct.

---

## Summary for Stage 1 hand-off

Confirmed, real, fixable defects (proceed with Stage 1 as scoped):
- (b) eq. 9 → eq. 3 citation error, plus γ→ξ symbol mismatch — **confirmed against the primary source**, both errors are real.
- (a) IQX → Weber-Fechner citation/prose mismatch — **confirmed against both the code and the primary source**; the fix should cite the review's own Eq. 4 and either drop or properly re-scope the `iqx-hypothesis` citation.
- (c) MOS scale — real gap, needs one clarifying clause, not a structural change.
- (e) Paper #1 title — **resolved outright**, not a VERIFY item; the exact string is above.
- (f) "closes the validated-deployment gap" — present in 2 places (line 30 "This paper closes that gap", line 348 "This paper closes the validated-deployment gap"); Stage 1's recommended softening ("narrows"/"first real-stack realisation") is a reasonable edit given rfsim is a radio simulator, not over-the-air.

Not confirmed, recommend skipping or re-verifying before acting:
- (d) SAC/Soft-Actor-Critic acronym collision — **not found** in the extracted PDF text. Do not spend a fix-cycle on this without a second, targeted look (possible OCR/text-extraction loss from a table cell).

New item found during this audit, not in the original Stage 1 list:
- `qoe_mapper.py`'s own module docstring mislabels the Weber-Fechner
  formula as the review's "eq.(3)" — it's eq.(4). This is a code-comment
  fix, zero effect on the paper text, but worth a one-line correction
  while Stage 1 is already touching this exact equation for the paper.
