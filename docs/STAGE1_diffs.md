# Stage 1 — Cross-paper consistency fixes

Text/LaTeX only. No code changes (the one code-comment error found in
Stage 0 — `qoe_mapper.py`'s docstring mislabeling its formula as the
review's eq.(3) instead of eq.(4) — was left untouched: it is inside
`qoe_oran_framework/`, which the standing hard constraints for this
rework explicitly forbid modifying. Flagged, not fixed; still open.)

All six items below were re-verified against the actual review PDF
(`/home/kmanojp/Desktop/IA_CROTUCDNS5GN_corrected.pdf`) in Stage 0 before
any edit was made here, per that stage's own findings.

---

### (a) MOS mapper contradiction — FIXED

The implementation (`qoe_oran_framework/qoe_mapper.py:92-107`) and the
review's own eq. (4) are both the logarithmic Weber–Fechner form
(`MOS = a - b*ln(1+Q_deg)`), not the exponential IQX-hypothesis form the
paper's prose implied.

**File:** `paper_conf/main.tex`, ~line 161 (Problem Formulation)

Before:
```
the mean inferred per-slice MOS (IQX closed-form prior~\cite{iqx-hypothesis}
refined by a per-slice LSTM, calibrated against ITU-T
P.1203~\cite{itu-p1203} objective labels for eMBB and ACR-style
task-success scoring for URLLC/mMTC), and
```
After:
```
the mean inferred per-slice MOS, normalized from its native P.1203/ACR 1--5
scale via $(\text{MOS}-1)/4$ (Table~\ref{tab:results} reports the raw,
un-normalized 1--5 values). MOS itself is estimated with a
Weber--Fechner-law closed-form prior -- eq.~4 of~\cite{survey-paper3},
$\text{MOS}=a-b\ln(1+Q_{\text{deg}})$, distinct from the exponential
IQX-hypothesis QoE-mapping family~\cite{iqx-hypothesis} -- refined by a
per-slice LSTM, calibrated against ITU-T P.1203~\cite{itu-p1203}
objective labels for eMBB and ACR-style task-success scoring for
URLLC/mMTC, and
```
`iqx-hypothesis` is kept as a citation but now correctly scoped as "the
family this paper does NOT use," rather than mislabeling it as the
formula in use. Same correction applied to the Discussion (~line 361,
"a calibrated IQX+LSTM mapper" → "a calibrated Weber-Fechner+LSTM
mapper").

### (b) Wrong equation cross-reference — FIXED

Review PDF confirms: eq. (3) is the QoE reward
(`rt = α·MOS(QoS) − β·cost − ξ·SLAviol`, extracted-text line 384); eq. (9)
is the FedAvg-FedProx federated-learning objective (line 505) — a
completely different equation. Symbol: the review uses **ξ**, not γ, for
the SLA-violation weight.

**File:** `paper_conf/main.tex`, Eq. `\ref{eq:qoe}` and surrounding prose;
`paper_conf/tables/table1_params.tex` row 17.

Before: `(eq.~9 of~\cite{survey-paper3})`, equation used `\gamma`,
`\alpha{=}1.0,\beta{=}0.2,\gamma{=}0.5`; Table I: `QoE weights $\alpha,\beta,\gamma$`

After: `(eq.~3 of~\cite{survey-paper3})`, equation uses `\xi`,
`\alpha{=}1.0,\beta{=}0.2,\xi{=}0.5` (matching~\cite{survey-paper3}'s own
symbol); Table I: `QoE weights $\alpha,\beta,\xi$`

Also fixed the same `\gamma` → `\xi` in the abstract's inline reward
formula (line 32).

### (c) MOS scale inconsistency — FIXED

Folded into the same edit as (a)/(b) above: the sentence now states
explicitly that $\widehat{\text{MOS}}_t\in[0,1]$ is
$(\text{MOS}-1)/4$ applied to the raw P.1203/ACR 1–5 scale that
Table II actually reports. Equation, table, and code
(`reward.py`'s `mos_norm = (mean_mos - 1.0) / 4.0`) now agree in the
paper text, not just in the code.

### (d) SAC acronym collision — PARTIALLY ACTIONED, with a caveat

**Stage 0 finding, restated:** the specific claim (review Table 6's
"Sanaei et al." row abbreviates "soft actor–critic" as "SAC") did **not**
verify against the extracted PDF text — that row spells the term out in
full, and the only parenthetical `(SAC)` in the whole review is "slice
admission control (SAC)". I flagged this as unconfirmed and recommended
not spending a fix cycle on it without a second look (possible
PDF-text-extraction loss from a table cell).

**What I did anyway:** spelled out "slice admission control" everywhere
in `main.tex` regardless, and removed the "(SAC)" abbreviation
definition entirely (5 sites: lines 62, 76, 357, 383, 429 in the
pre-edit file). Rationale: SAC is Soft Actor-Critic in a large fraction
of the RL literature independent of whether this specific review's
Table 6 collides with it — spelling it out is a costless precaution
regardless of whether the originally-cited collision itself checks out.
If you want this reverted (i.e., keep the "(SAC)" shorthand) because the
Table 6 claim didn't hold up, say so; it's a trivial revert.

### (e) Bibliography mismatch for paper #1 — FIXED, fully resolved (not a TODO)

Review PDF's own reference list, entry [8] (extracted-text line 1046):
*"Optimising resource allocation in 5G networks: Balancing URLLC and
eMBB traffic under gNB congestion,"* 2025 MECON, pp. 1–6. The word "RAN"
is not present.

**File:** `paper_conf/refs.bib`, `mecon-paper1` entry.

Before: `title = {Optimising Resource Allocation in {5G} {RAN} Networks: ...}`
After: `title = {Optimising Resource Allocation in {5G} Networks: ...}`

### (f) Over-claim on the deployment gap — FIXED, 6 sites

Every instance of "closes"/"Closed"/"closing" the validated-deployment
gap replaced with "narrows"/"Narrowed"/"narrowing", each now paired with
an explicit real-vs-simulated enumeration (protocol stack, MAC
scheduler, E2 control loop, core network = real; radio channel = rfsim
= simulated). Sites fixed:

1. Abstract (line 30): "closes that gap" → "narrows that gap --
   realizing the control loop on real protocol-stack, MAC-scheduler, and
   E2-agent hardware, with the radio channel still simulated (rfsim) --"
2. Discussion §V-A (line 357): "\textbf{closes} the validated-deployment
   gap for the SAC formulation" → "\textbf{narrows} the validated-deployment
   gap for the slice admission-control formulation: the protocol stack,
   MAC scheduler, E2 control loop, and core network are real; the radio
   channel is still rfsim-simulated."
3. Table `tab:gaps` row (line 383): "\textbf{Closed} for SAC: real E2
   loop, real gNB" → "\textbf{Narrowed} for slice admission control: real
   E2 loop, real MAC scheduler, real protocol stack; radio channel still
   simulated (rfsim)"
4. Conclusion (line 431): "closing the validated-deployment gap" →
   "narrowing the validated-deployment gap", with the same real/simulated
   enumeration added to the preceding clause.

---

## Compile check

`pdflatex` + `bibtex` × 3 passes: **6 pages, 15 citations, zero undefined
references, zero multiply-defined labels.** No new overfull/underfull
warnings beyond the pre-existing cosmetic ones (the `\authorTODO` fbox
overflow, which is intentional and removed before submission; two
sub-1-inch table-column overflows already present before this stage).

## Acceptance test

- [x] Paper compiles.
- [x] All six fixes present with before/after (this document).
- [x] No remaining reference to IQX describing the implemented formula,
      or to eq. 9, without justification — `iqx-hypothesis` is still
      cited once, explicitly scoped as "the family not used here."
