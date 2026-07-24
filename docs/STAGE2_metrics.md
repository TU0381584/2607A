# Stage 2 — Metrics layer

Recomputed from existing logs (live campaign) and from one deterministic
re-evaluation of the already-frozen congested-scenario checkpoints (the
existing `eval_congested_vs_baseline.py` only ever persisted the *mean*
per-step margin, not the raw distribution Stage 2 needs — see
`experiments/scripts/metrics_stage2.py`'s module docstring for why a
re-run was unavoidable; no training occurred, same seeds, same
checkpoints, same config). Raw output: `docs/stage2_metrics_raw.json`.

**Read this document before touching Table III's prose.** Item 1 below
does not confirm the result Stage 2's acceptance test assumed.

---

## 1. Priority-weighted SLA utility, U

Defined as $U = \sum_s w_s \cdot \text{compliance}_s \,/\, \sum_s w_s$,
using each experiment's own declared `priority_weight` (Table I):
eMBB=3.5, URLLC=5.0, mMTC=0.3 — identical in both the live-campaign
config (`saclb_campaign.yaml`) and the congested config
(`saclb_offline_congested_v1.yaml`), confirmed by direct diff.

**Live campaign** (no surprise — DQN already leads on every single slice):

| Arm | U (priority-wtd) | Unweighted mean |
|---|---|---|
| baseline | 73.5% | 73.6% |
| DQN (SLA) | 100.0% | 100.0% |
| DQN (QoE) | 100.0% | 100.0% |

**Congested scenario — the actual result does not match what Stage 2 assumed:**

| Arm | URLLC | eMBB | mMTC | U (priority-wtd) | Unweighted mean |
|---|---|---|---|---|---|
| baseline | 22.6 | 34.5 | 19.0 | **27.2%** | 25.4% |
| DQN (SLA) | 30.9 | 7.9 | 9.2 | **21.0%** | 16.0% |
| DQN (QoE) | 27.0 | 8.2 | 10.8 | **19.0%** | 15.3% |

**Baseline wins on the priority-weighted utility too — it does not just
win on the unweighted mean.** The reason is arithmetic, not a modeling
error: eMBB's declared priority weight (3.5) is more than two-thirds of
URLLC's (5.0), so the ~26-point compliance eMBB gives up
(34.5%→7.9%/8.2%) costs more weighted utility than the ~4-8 point URLLC
gain (22.6%→27.0%/30.9%) buys back, under the weights actually in Table
I. This held under both DQN reward modes.

**Sensitivity check, weighting by `violation_penalty` instead** (the
per-slice weight that actually multiplies the *violation* term in the
reward — URLLC's penalty was deliberately raised to 20.0 specifically to
protect it, vs. eMBB's 5.0 and mMTC's 2.5 — arguably a more mechanistically
faithful weight for a *compliance* utility than `priority_weight`, which
in the actual reward only multiplies the *accepted-volume* term, not
compliance):

| Arm | U (violation_penalty-wtd) |
|---|---|
| baseline | 24.5% |
| DQN (SLA) | 24.8% (a 0.3-point win — essentially a tie) |
| DQN (QoE) | 22.1% (still a loss) |

Under this alternative weighting DQN(SLA) edges baseline by a margin too
small to call a win, and DQN(QoE) still loses. **Neither weighting
scheme supports the "DQN wins on the declared objective" framing.** This
is a real, calibrated finding, not a bug in the metric: it means the
paper's own Section IV-C narrative ("a genuine, learned priority
tradeoff") currently asserts value that a utility computation against
the paper's own declared weights does not support. Two honest ways to
proceed, both flagged for Stage 7 (this is a narrative decision, out of
Stage 2's scope):
1. Report plainly that the tradeoff is real and reproducible but not
   utility-improving under the current priority weights — a legitimate,
   still-interesting finding (DQN learns to protect the highest-raw-gain
   slice, not the highest-utility outcome) — or
2. If the intent really is "URLLC should dominate the utility
   calculation," the config's own `priority_weight` values (Table I) are
   the thing to revisit, not the metric.

I have not made this call; it changes what Section IV-C claims and is
therefore Stage 7's decision, with the numbers in hand.

---

## 2. Violation severity distributions (median / IQR / P90 of the
continuous per-step SLA margin, `reward.py`'s `per_slice_sla_margin`)

**Live campaign** (n=900 steps/arm/slice):

| Arm | Slice | Median | IQR | P90 |
|---|---|---|---|---|
| baseline | eMBB | 0.7 | 1,002,378 | 0.7 |
| baseline | URLLC | 0.7 | 150,783 | 0.7 |
| baseline | mMTC | 0.7 | 99.0 | 0.7 |
| DQN (either) | eMBB | 1.0 | ~0 | 1.0 |
| DQN (either) | URLLC/mMTC | 0.7 | ~0 | ~1.0 |

Baseline's enormous IQR (eMBB: over 1,000,000) alongside a *median* of
0.7 confirms the already-known bimodal pattern quantitatively: most
steps are fine, but a real minority sit in the catastrophic
backlog-blowup regime documented elsewhere in the paper (P5 margin
≈−1×10⁶). DQN's near-zero IQR is the numeric signature of "always at
ceiling, always compliant" — consistent with, not new information beyond,
the 100.0±0.0% compliance figure already in Table II.

**Congested scenario** (n=2,700 steps/arm/slice) — this is where severity
adds real information the raw compliance percentage does not:

| Arm | Slice | Median | P90 |
|---|---|---|---|
| baseline | eMBB | −0.5 | **+0.976** |
| DQN (SLA) | eMBB | −0.5 | **−0.059** |
| DQN (QoE) | eMBB | −0.5 | **−0.056** |
| baseline | URLLC | −0.5 | +0.593 |
| DQN (SLA) | URLLC | −0.5 | **+0.960** |
| DQN (QoE) | URLLC | −0.5 | +0.939 |
| baseline | mMTC | −0.5 | +0.447 |
| DQN (SLA/QoE) | mMTC | −0.5 | ≈0.0 |

All medians are −0.5 (every arm is in violation more than half the time
on every slice — the scenario is genuinely, uniformly hard). The P90
column is where the trade-off actually shows: baseline's eMBB P90 is
comfortably positive (+0.976 — its *good* tail is genuinely good); both
DQN arms' eMBB P90 is *negative* (−0.06), meaning even DQN's best 10% of
eMBB steps are still in violation. Conversely, DQN(SLA)'s URLLC P90
(+0.96) is much healthier than baseline's (+0.59). This is a sharper,
more honest picture of the trade-off than the compliance percentages
alone: DQN doesn't just lower eMBB's average compliance, it eliminates
eMBB's good tail entirely, while giving URLLC a good tail baseline
doesn't have.

---

## 3. Significance testing

**Live campaign** — episode-level Fisher exact test (episodes fully
SLA-compliant, all 3 slices, out of 15/arm): DQN 15/15 vs. baseline
11/15 (both reward modes). **p = 0.0996** (two-sided Fisher exact,
scipy). Odds ratio is undefined (a zero cell). **This does not clear
p<0.05** — consistent with, not a correction of, the paper's own existing
honesty about the Wilcoxon test not clearing significance either (main
text, Section IV-A): n=15 is genuinely too small for the categorical,
practically-decisive gap (0.0% worst-episode baseline vs. 100.0% worst
DQN) to also be statistically decisive by a paired/exact test. Report
both, as the paper already does for Wilcoxon.

**Congested scenario** — the same episode-level test is **degenerate**:
0 of 45 episodes are fully compliant (any single slice, for the whole
episode) for *any* arm, including baseline. This is expected, not a
bug — congested-scenario compliance is a within-episode step *fraction*
(22–35% of steps), never a per-episode binary the way the live
campaign's is, so "episodes fully compliant" simply cannot discriminate
here (0/45 vs 0/45 everywhere, p=1.0, uninformative). **Substituted a
step-level chi-square test of independence** on the actual
compliant/total step counts (n=2,700 steps/arm/slice — large enough for
chi-square to be safe), arm vs. baseline:

| Comparison | Slice | χ² | p |
|---|---|---|---|
| DQN(SLA) vs baseline | eMBB | 569.4 | 7.4×10⁻¹²⁶ |
| DQN(SLA) vs baseline | URLLC | 47.0 | 7.2×10⁻¹² |
| DQN(SLA) vs baseline | mMTC | 107.3 | 3.8×10⁻²⁵ |
| DQN(QoE) vs baseline | eMBB | 555.9 | 6.5×10⁻¹²³ |
| DQN(QoE) vs baseline | URLLC | 13.6 | 2.3×10⁻⁴ |
| DQN(QoE) vs baseline | mMTC | 71.9 | 2.2×10⁻¹⁷ |

Every per-slice difference is real (not noise) at this sample size — the
large χ² values mostly reflect n=2,700 giving enormous power, not that
the *effect sizes* are all large; report alongside the raw compliance
percentages, not instead of them.

---

## 4. Tables and figures updated

- **Table II** (live campaign): added a `U` column.
- **Table III** (congested): added a `U (priority-wtd)` column. Full
  severity distributions did not fit the existing table without
  crowding it past readability at the paper's current page budget — the
  P90 summary (§2 above) is reported in this document and is a candidate
  for a compact addition in Stage 7's restructuring pass, not force-fit
  here.
- **Figs 3 and 5**: left the plotted content as-is (per-slice compliance
  bars remain the clearest visual for this data); U and severity are
  reported in the tables and this document rather than re-encoded
  visually, to avoid a Stage-2-only change silently altering figures
  that Stage 7 will restructure anyway.
- `U` is now defined in `main.tex` (Problem Formulation, as Eq. 4)
  *before* Table II/III use it, per the acceptance test.

## Acceptance test — actual status

- [x] U is defined in the paper before it is used.
- [ ] ~~"Table III shows DQN winning on U while losing on unweighted
      mean"~~ — **does not hold**. Table III shows DQN losing on U under
      both candidate weightings, and the text (once Stage 7 touches
      Section IV-C's prose) needs to say so explicitly rather than the
      reverse. Flagging this deviation from the stage brief rather than
      manufacturing the assumed result.
