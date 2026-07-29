# Stage 9 — Submission hygiene

Final pass before this manuscript is ready to leave the staged-rework
pipeline. Six checks, each done directly rather than assumed.

## 1. Final TODO sweep

`grep`'d `paper_conf/main.tex` and `paper_conf/refs.bib` for
`TODO`/`FIXME`/`XXX`. One hit, and it's intentional: the
`\authorTODO{...}` stub in Section II (Literature Review), left for
Manoj to fill in per the explicit instruction to leave that section
out of this pass. No stray or forgotten markers anywhere else.

## 2. Clean compile check

Rebuilt from a bare copy of `main.tex` + `refs.bib` + `figures/`
(no reused `.aux`/`.bbl`/`.log`) with the real sequence:
`pdflatex → bibtex → pdflatex → pdflatex`. Result: **4 pages**, zero
errors, zero undefined references, zero "Label(s) may have changed"
residue after the third pass. The bibliography's numbering matches the
in-text citation order (`[1]`–`[6]`) exactly.

## 3. Figure/data freshness verification

Both embedded figures were checked, not assumed current:

- `fig2_sla_compliance.pdf` — regenerated this session (Stage 7/8) from
  `experiments/results/live_campaign`, seeds 950–952, with the
  `static_at_cap` arm's label corrected to drop a stray "(oracle)"
  suffix (`experiments/plots/common.py`). Current.
- `fig4_ceiling_trajectories.pdf` — not touched this session, dated
  2026-07-19 (pre-dates the current staged rework). Verified by
  regenerating from its own defaults after fixing them (item 5 below)
  and comparing `md5sum` against the file embedded in the paper:
  **byte-identical**. The figure is exactly what it claims to be
  (baseline vs.\ DQN, seed 950, episode 1, from
  `experiments/results/live_campaign`), not stale or orphaned data.

## 4. Reference audit

Of the 15 stub entries in `refs.bib`, 6 are actually `\cite`'d in the
current lean manuscript (the other 9 remain available, unused, for
whenever the Literature Review section is written). Checked each of
the 6 for outstanding `VERIFY:` notes:

- **`oranslice` — resolved this stage.** Fetched the repository's own
  citation guidance (`github.com/wineslab/ORANSlice`): it names a
  specific paper (Cheng et al., "ORANSlice: An Open-Source 5G Network
  Slicing Platform for O-RAN," ACM MobiCom '24, DOI
  10.1145/3636534.3701544) as the preferred citation, not the bare
  repository URL. `refs.bib` updated to a proper `@inproceedings`
  entry; this fork's own commit (`b9bcc9b`) kept as a note for
  provenance, not as the citation itself.
- **`oran-e2ap` — left as `VERIFY`, deliberately.** The exact O-RAN
  WG3 E2AP spec version/year (e.g.\ `O-RAN.WG3.E2AP-v03.00`) lives
  behind the O-RAN Alliance's member specification portal; I could not
  confirm the exact version number from public sources without
  guessing a document ID, which the project's own standing
  instruction (never invent/guess a citation detail) rules out.
  **Needs Manoj's direct access to the O-RAN specification portal
  before submission.**
- **`survey-paper3` — left as `VERIFY`, deliberately.** The note
  ("final volume/issue/DOI once formally published") records the
  paper's own publication status, which only the author knows
  first-hand. **Needs Manoj to fill in once #3's IEEE Access
  publication record is final.**
- **`mecon-paper1`, `icraie-paper2`, `dqn`** — no outstanding `VERIFY`
  notes; page ranges and venue names already confirmed against the
  actual conference PDFs during Stage 8's bridge audit.

## 5. Reproducibility appendix

Written up separately as `docs/REPRODUCIBILITY.md` rather than folded
into the manuscript (no page budget for it, and it would reintroduce
script/tool-name jargon the house-style rewrite deliberately removed).
Maps every number, figure, and table in the paper to its exact
source script, config, and seed. One real bug found and fixed while
building this map: `fig4_ceiling_trajectories.py`'s own usage example
and argparse defaults pointed at a nonexistent directory
(`experiments/results/live`, should be `.../live_campaign`) and a
stale seed (256, an offline-training seed, not a live-evaluation one).
Fixed to `experiments/results/live_campaign` / seed 950, matching how
the script has actually been invoked throughout the project — see
`docs/REPRODUCIBILITY.md` for the verification that this doesn't
change the embedded figure at all.

## 6. Old-rig-checkpoint confirmation

Every number in the paper traces to `experiments/results/live_campaign*`
or `experiments/results/offline*` (this rig, this project's sessions) —
detailed in `docs/REPRODUCIBILITY.md`. The only "old rig" mentions
anywhere in the active config/campaign files are in
`saclb_campaign.yaml`'s own header comment, explicitly stating its
calibrated values were "validated live against this campaign's actual
traffic profile, not inherited from the old rig" — i.e.\ the phrase
appears only to rule out reuse, never to describe reuse. Confirms the
project's standing hard constraint ("do not report or reuse any result
from old-rig checkpoints") was honored throughout.

## Acceptance status

- [x] TODO sweep: one intentional stub (Lit Review), nothing else.
- [x] Clean compile: 4 pages, zero errors, from a bare rebuild.
- [x] Figure freshness: both figures verified current, one confirmed
      byte-identical to a from-scratch regeneration.
- [x] Reference audit: 1 of 3 outstanding `VERIFY` notes resolved
      (`oranslice`); the other 2 are genuinely author-only information
      (O-RAN spec portal access, #3's own publication record) and are
      flagged, not guessed.
- [x] Reproducibility appendix written (`docs/REPRODUCIBILITY.md`);
      one real script-default bug found and fixed as part of building
      it.
- [x] Old-rig-checkpoint constraint confirmed honored across every
      number in the paper.
