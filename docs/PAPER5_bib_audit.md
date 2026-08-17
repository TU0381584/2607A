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

## Addendum: seven specifically-named prior-art works (M5 brief)

The reviewing brief (routed through Claude Web) named seven prior-art
works by topic/venue/year rather than by citation key, to be located,
verified, and positioned against explicitly in the lit review. None of
these overlap with the 31-entry skeleton above (that skeleton was
deliberately scoped to 2024-2026 only, per `ref:access`'s own review
window; several of these seven predate that window). Same method as
above: found via web search, confirmed against CrossRef's own
registered metadata, not accepted on title-match alone.

| Brief's description | Resolved paper | Venue | Year | DOI | Status |
|---|---|---|---|---|---|
| GAT+MARL base-station-agent slicing, "the 2021 canonical work" | Shao, Li, Hu, Wu, Zhao, Zhang, "Graph Attention Network-Based Multi-Agent Reinforcement Learning for Slicing Resource Management in Dense Cellular Network" | IEEE Trans. Vehicular Technology | 2021 | `10.1109/TVT.2021.3103416` | verified |
| Topology-generalizable GAT-MARL admission agents | Ahmadi, Moayyedi, Sulaiman, Salahuddin, Boutaba, Saleh, "Generalizable 5G RAN/MEC Slicing and Admission Control for Reliable Network Operation" | IEEE Trans. Network and Service Management | 2024 | `10.1109/TNSM.2024.3437217` | verified -- fetched the actual PDF (not just CrossRef metadata) to confirm it is genuinely graph-attention-based, not just topically adjacent |
| Coordinated MARL slicing+admission control, "IEEE TNSM 2022" | Sulaiman, Moayyedi, Ahmadi, Salahuddin, Boutaba, Saleh, "Coordinated Slicing and Admission Control Using Multi-Agent Deep Reinforcement Learning" | IEEE Trans. Network and Service Management | 2023 (CrossRef print record: vol. 20, issue 2, pp. 1110-1124) | `10.1109/TNSM.2022.3222589` | verified, year note: DOI itself is 2022-dated (early access); CrossRef's authoritative print-issue year is 2023 -- the brief's "2022" is the online-early-access date, cited here as 2023 to match CrossRef |
| Federated DRL coordinating O-RAN slicing xApps, "GLOBECOM 2022" | Zhang, Zhou, Erol-Kantarci, "Federated Deep Reinforcement Learning for Resource Allocation in O-RAN Slicing" | GLOBECOM 2022, pp. 958-963 | 2022 | `10.1109/GLOBECOM48099.2022.10001658` | verified |
| FL+DP for O-RAN slicing | Yasin, Yu, Wang, "Differential Privacy Federated Edge Learning-assisted for Securing RAN Intelligent Controller in O-RAN 6G Communications" | 2025 IEEE VTS Asia Pacific Wireless Commun. Symp. (APWCS) | 2025 | `10.1109/APWCS67981.2025.11151868` | verified |
| Adversarial/disruption resilience for RAN slicing, "2026" | Tashman, Cherkaoui, "Adversarial Attacks in AI-Driven RAN Slicing: SLA Violations and Recovery" | 2026 Intl. Wireless Commun. and Mobile Computing (IWCMC) | 2026 | `10.1109/IWCMC69287.2026.11580033` | verified |
| Attention-based MARL for O-RAN slicing, "2026" | Fatehi, Rahmani Ghourtani, Sonee, Yadav, Russo, Ahmadi, Calinescu, "Interpretable Attention-Based Multi-Agent PPO for Latency Spike Resolution in 6G RAN Slicing" | ICC 2026 | 2026 | `10.1109/ICC59461.2026.11586913` | verified |

**7/7 located and verified, 0/7 fabricated.** Note the Waterloo group
(Sulaiman/Ahmadi/Moayyedi/Salahuddin/Boutaba/Saleh) authored both the
TNSM 2022/2023 coordinated-admission paper and its 2024
topology-generalization follow-up -- these read as one coherent line
of work and are positioned that way in the lit review rather than as
two independent citations.

All 38 entries (31 skeleton + 7 here) are merged into `paper5/refs.bib`
for the actual manuscript; see `paper5/main.tex`'s Literature Review
section for how each is positioned.
