# Paper #5 skeleton bibliography (`paper5.bib`, Desktop) — authenticity audit

Method: every entry's DOI was resolved directly against the CrossRef API
(`api.crossref.org/works/{doi}`), the authoritative metadata registry
publishers themselves submit to — not a search engine guess, not an LLM
recollection. Returned title/authors/container-title/year were compared
against the bib entry field by field. No entry was accepted on the DOI
resolving alone; metadata had to actually match.

**Headline finding, stated plainly because it contradicts the brief's own
premise:** all 31 entries resolve to real, correctly-cited publications,
**including both entries the brief said were "already confirmed
non-existent"** (`wu2025admissionmarl` / "Wu constrained-MARL, ICC'25" and
`wu2025slicewise` / "Wu SliceWise, Netw. Lett.'25"). Both DOIs resolve
cleanly on CrossRef with title, full author list, venue, and year all
matching the bib entry exactly. Whatever check produced the "already
confirmed non-existent" claim did not match what CrossRef's own registered
metadata shows. Reported here rather than silently accepted, consistent
with this project's practice of not trusting an unverified claim just
because a prior message asserted it.

## Results

| Key | Status | Evidence |
|---|---|---|
| `ref:access` | verified | Vol./pages/DOI already independently confirmed against IEEE Access this session (paper_conf/refs.bib's `survey-paper3` entry, same paper) |
| `bouroudi2025gnnmarl` | verified | CrossRef exact match: title, 3 authors, GLOBECOM 2025, year 2025 |
| `mohajer2026pronto` | verified | CrossRef exact match: title, 4 authors, IEEE TNSM, year 2026 |
| `qasim2025tgnnfl` | verified | CrossRef exact match: title, 5 authors, IEEE Access, year 2025 |
| `gong2026femaddpg` | **verified, minor formatting issue** | CrossRef title/venue/year match exactly. Two authors' given/family name order is swapped in the bib vs. CrossRef ("Yu Gong" listed as "G. Yu"; "He Jiang" listed as "J. He") — a real paper, cosmetic author-initial error, not fabrication. Worth fixing before submission. |
| `gonzalez2024fedns` | verified | CrossRef exact match: title, 6 authors, IEEE TVT, year 2024 |
| `tran2026gnnppo` | verified | CrossRef exact match: title, 6 authors, ICOIN 2026, year 2026 |
| `silva2025gnnintent` | verified | CrossRef exact match: title, 5 authors, IEEE Access, year 2025 |
| `hao2024html` | verified | CrossRef exact match: title, 6 authors, IEEE TWC, year 2024 |
| `zhu2025hgatdt` | verified | CrossRef exact match: title, 4 authors, ISPA 2025, year 2025 |
| `kurapati2025hybridgnn` | verified | CrossRef exact match: title, 3 authors, ICOSEC 2025, year 2025 |
| `jing2025gnnoffline` | verified | CrossRef exact match: title, 6 authors, IEEE TVT, year 2025 |
| `qpcs2024gdrl` | **verified, key/year mismatch** | CrossRef exact match: title, 5 authors, IEEE TMC, actual year **2025**. The bib entry's own `year` field already correctly says 2025 -- only the citation *key* (`qpcs2024gdrl`) carries a stale "2024", a cosmetic naming inconsistency, not a data error. |
| `asheralieva2024admmgnn` | verified | CrossRef exact match: title, 3 authors, IEEE TMC, year 2024 |
| `doanis2024samarl` | verified | CrossRef exact match: title, 2 authors, IEEE TMLCN, year 2024 |
| `zangooei2024flexran` | verified | CrossRef exact match: title, 5 authors, IEEE JSAC, year 2024 |
| `wu2025admissionmarl` | **verified** (brief claimed non-existent) | CrossRef exact match: title, 3 authors (Xingqi Wu, Junaid Farooq, Juntao Chen), ICC 2025, year 2025 |
| `wu2025slicewise` | **verified** (brief claimed non-existent) | CrossRef exact match: title, 5 authors (Xingqi Wu, Yuhui Wang, Junaid Farooq, Hakim Ghazzai, Gianluca Setti), IEEE Networking Letters, year 2025 |
| `lotfi2026sharpmarl` | verified | CrossRef exact match: title, 3 authors, IEEE TMLCN, year 2026 |
| `phyu2025icecream` | verified | CrossRef exact match: title, 3 authors, IEEE TNSM, year 2025 |
| `chen2025matd3` | verified | CrossRef exact match: title, 5 authors, IEEE TWC, year 2025 |
| `li2026scalablemarl` | verified | CrossRef exact match: title, 5 authors, IEEE TMC, year 2026 |
| `wang2024madqnvnf` | verified | CrossRef exact match: title, 6 authors, IEEE IoT Journal, year 2024 |
| `amiri2024fedvnf` | verified | CrossRef exact match: title, 4 authors, IEEE Commun. Lett., year 2024 |
| `ming2024fedmobility` | verified | CrossRef exact match: title, 3 authors, IEEE TMC, year 2024 |
| `kwantwi2025pfl` | verified | CrossRef exact match: title, 5 authors, IEEE IoT Journal, year 2025 |
| `abdisarabshali2025elasticfl` | **verified, key/year mismatch** | CrossRef exact match: title, 5 authors, IEEE IoT Magazine, actual year **2026**. Bib entry's own `year` field already says 2026 correctly; only the key carries a stale "2025". |
| `gong2024dqnfl` | verified | CrossRef exact match: title, 6 authors, ICAIT 2024, year 2024 |
| `koursioumpas2024safefl` | verified | CrossRef exact match: title, 8 authors, IEEE TGCN, year 2024 |
| `liu2026aoifl` | verified | CrossRef exact match: title, 6 authors, IEEE TMC, year 2026 (early access, matching bib's `note={early access}`) |
| `wu2024cefl` | verified | CrossRef exact match: title, 6 authors, IEEE/ACM Trans. Networking, year 2024 |

## Summary

- **31/31 entries verified authentic** (DOI resolves, metadata matches).
- **0/31 fabricated.** Both entries the brief flagged as "already confirmed
  non-existent" are real, correctly-cited papers — that premise was wrong,
  not the bibliography.
- **3/31 have a cosmetic (not substantive) issue**, listed above: one
  author-name-order swap, two citation-key/year mismatches where the
  `year` field itself is already correct. None affect whether the cited
  paper exists or what it says.
- Per the brief's instruction, nothing has been deleted or altered —
  this is a classification pass only. If the author wants, the three
  cosmetic issues are a five-minute fix (correct the two author name
  orders in `gong2026femaddpg`, rename the two stale-year keys) before
  this bibliography is used in the lit review.
