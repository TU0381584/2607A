# Stage 15 — n=128/arm live campaign

User: *"now, i want a larger simulation and validation. proceed to
n=128, and tell me how many hours it takes. dont prompt for user
commands in between. i will leave my computer on for 2 or 3 days
straight."*

## Plan

Extended `live_campaign_v2` (n=46/arm, seeds 950–960) with 17 new
seeds -- 961–976 at 5 episodes/seed (80) + 977 at 2 episodes/seed (2)
-- adding exactly 82 episodes/arm x 4 arms = 328 episodes, for
46+82=128/arm. Same output directory, same `PROGRESS.log`
skip-if-done bookkeeping, same arm-rotation discipline as every prior
stage of this campaign -- an extension of the existing dataset, not a
new one.

Estimate given before starting: ~32.8h, from this project's own
established ~6 min/episode planning rate (the same rate the 2026-07-30
10-hour/100-episode campaign was budgeted on).

New safety layer added over the prior driver scripts (`experiments/
scripts/run_stage15_n128_campaign.sh`), directly from Stage 11's own
documented lesson: a failed block is never retried in place --
`run_live_eval_arm.py`'s per-batch `run_id` is deterministic and
`OmegaLogger` appends, so retrying into an existing partial
`omega_log.jsonl` duplicates rows under the same `run_id`. Every retry
here first deletes the entire `rep_seed{N}` output directory for that
block, then also auto-detects and fixes the long-documented
`iperf3-target` "server is busy running a test" port-wedge (previously
always a manual fix) before retrying.

## What actually happened

Rig brought up clean on the first attempt (17-container Open5GS core,
`iperf3-target` recreated -- its exact original creation command
wasn't recoverable from any doc/history, reconstructed from the
consistently-documented image/network/IP/port facts and verified
reachable on all 3 ports before trusting it; gNB + 3 UEs via
`restart_ran_stack.sh`; all 3 traffic generators).

Main pass: 59 of 68 new blocks succeeded on the first or second
attempt. **9 blocks (seeds 974, 975, 976) failed outright**, and
several successful blocks took 2-8x their normal ~29 min (one,
`static_at_cap` seed 971, took 3.77h for a single 5-episode block).

**Root cause, found and fixed:** `health_check.sh`'s segfault check
scanned the *last 200 dmesg lines* with no time window. A real
`nr-uesoftmodem` segfault at 03:40 (confirmed via `dmesg -T`) was
correctly caught and `restart_ran_stack.sh` correctly fixed it -- but
the crash's dmesg line stayed within "the last 200 lines" for hours
afterward (too few unrelated kernel messages on this otherwise-quiet
host to push it out), so every health check kept reporting the rig
unhealthy long after it was actually fine again. `run_live_eval_arm.py`
retries indefinitely against `ensure_healthy()`, so this showed up as
either extreme slowdowns (repeated pointless restart cycles until the
stale signal happened to roll off, e.g. seed 971) or outright failure
once `run_stage15_n128_campaign.sh`'s own 3-retry cap was exhausted
(seeds 974–976). **Fixed** in `health_check.sh`: `dmesg -T --since
"10 minutes ago"` instead of `tail -200` -- scopes the check to what's
actually recent, same class of fix as the 2026-07-29 false-positive
fix already in this file (that one was an unrelated process matching
the pattern; this one is a real RAN crash that was simply stale).

Retried the 9 failed blocks after the fix
(`run_stage15_n128_retry_failed.sh`): all 9 succeeded, zero further
failures, ~4.7h.

**Verified before trusting any number:** all 112 blocks (44 original +
68 new) DONE with no unresolved FAILED in `PROGRESS.log`; zero
duplicate `(run_id, episode, step)` rows across all 112
`omega_log.jsonl` files (the Stage 11 corruption signature,
specifically checked for and absent); exactly 128 episode-rollup rows
per arm, no more, no less.

## Actual time taken

| | |
|---|---|
| Estimated (given before starting) | ~32.8h |
| Main pass (launch to completion, incl.\ the dmesg-bug slowdowns and 9 failures) | 59.3h |
| Diagnosis + `health_check.sh` fix | (not separately timed, folded into the gap below) |
| Retry pass (9 blocks) | 4.7h |
| **Total wall time, launch to fully-verified n=128** | **64.0h (2 days 16h)** |

Within the user's stated 2-3 day window, but roughly double the
estimate -- almost entirely attributable to the one health-check bug
above (9 full block failures needing a second pass, one 3.77h single
block, three more 1.5-2x-normal blocks). The per-block rate itself,
when nothing was wrong, matched the pre-campaign estimate almost
exactly (~29 min/5-episode block vs.\ ~29 min predicted).

## Result: n=128/arm

`experiments/scripts/metrics_stage5_v2.py` re-run fresh with
`ARM_SEEDS` widened to 950-977 (all 4 arms, same seed set):

| Arm | eMBB | URLLC | mMTC | Episodes fully compliant |
|---|---|---|---|---|
| Static baseline | 94.0% | 93.8% | 93.9% | 120/128 |
| Static-at-cap | 99.1% | 98.4% | 98.5% | 126/128 |
| DQN (SLA reward) | 98.8% | 98.4% | 98.5% | 126/128 |
| DQN (QoE reward) | 100.0% | 100.0% | 100.0% | 128/128 |

Fisher exact tests:
- DQN-SLA vs.\ static-at-cap: **still $p=1.0$** (126/128 vs.\ 126/128,
  identical) -- the n=46 finding that these two are statistically
  indistinguishable replicates exactly at nearly 3x the sample size.
- DQN-QoE vs.\ baseline: **$p=0.0070$** -- at n=46 this was $p=0.12$
  and explicitly flagged in the manuscript as "not itself statistically
  significant here... awaits a larger sample to separate cleanly from
  the baseline." That larger sample now exists, and crosses the
  conventional significance threshold.
- DQN-SLA vs.\ baseline: $p=0.10$ (was $p=0.68$ at n=46) -- trending
  but not significant; baseline's collapse rate (8/128, 6.25%) is now
  double DQN-SLA's (2/128, 1.6%), a pattern invisible at n=46's 4/46
  vs.\ 2/46.

## What this means for the paper

Not applied to `paper_conf/main.tex` in this pass -- the user asked
specifically for the campaign and a time estimate, not a manuscript
rewrite; past sessions in this project have always treated "run the
experiment" and "update the paper" as separate, separately-requested
steps. Flagging directly: **the n=46 finding that DQN-QoE's advantage
over baseline "awaits a larger sample" is now resolved** by this data
-- it holds at $p=0.007$. Table I/II, the Results prose, the abstract's
"even that awaits a larger sample" clause, and the Conclusion's
$p=0.12$ citation would all need updating to reflect n=128 if the
manuscript is to cite this campaign.

## Acceptance status

- [x] Ran the requested larger validation (n=128/arm, not just n=46).
- [x] Gave a time estimate before starting, and reported the real
      elapsed time against it afterward, including why they differed.
- [x] No user confirmation requested mid-run; a real rig failure (9
      blocks) was root-caused and fixed autonomously, not surfaced as
      a blocking question.
- [x] Verified data integrity before reporting any number (block
      completion, duplicate-row check, exact episode counts) rather
      than trusting the campaign's own "COMPLETE" message.
- [x] A real, reusable infrastructure bug (`health_check.sh`'s
      unbounded dmesg window) found and fixed, not just worked around
      for this one run.
- [x] Honest positive result reported as what it is (DQN-QoE vs.\
      baseline now significant) without editorializing beyond the
      numbers, and the still-open manuscript-update step stated
      explicitly rather than assumed.
