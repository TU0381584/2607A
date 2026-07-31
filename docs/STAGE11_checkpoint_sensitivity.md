# Stage 11 — Checkpoint sensitivity: is one trained DQN-SLA policy representative?

User: *"proceed with no.1 [live-evaluate the other trained checkpoints].
go all the way, and dont prompt for user inputs."* Full autonomous run,
same standing instruction as Stage 10.

**Question:** Stage 10's offline reverification showed training itself
converges consistently across 6 seeds (Q1→Q4 reward improvement
similar for all). But only ONE trained checkpoint (seed256) has ever
been evaluated live for `dqn_sla`. Is seed256's tie with static-at-cap
(44/46 vs 44/46, the finding that overturned Stage 3's original
p=0.0149 claim) representative of what this training pipeline
typically produces, or a property of that one specific checkpoint?

## Method

Live-evaluated all 5 OTHER already-trained DQN-SLA checkpoints (seeds
257–261, from Stage 10's offline reverification) against the existing
`static_at_cap` record (n=46, unchanged), using the exact same seed
structure as every other arm (950–952 at 2 ep/seed, 953–955 at 5
ep/seed = 21 episodes/checkpoint). `experiments/scripts/
run_checkpoint_sensitivity.sh` (new).

## Operational notes (rig instability, real this time — not a false alarm)

Two blocks failed on the first pass (`dqn_sla_seed260`/`dqn_sla_seed261`,
both seed=954): a genuine health-check failure escalated to
`restart_ran_stack.sh` itself failing its own post-restart connectivity
check. Root cause both times, on manual follow-up: the `iperf3-target`
container's port had wedged again (`"the server is busy running a
test"` — the same long-documented failure mode this project has hit
repeatedly; fix is always the same, recreate the container).

**A real mistake made and caught during the fix:** retrying the two
failed blocks by simply re-invoking the orchestrator (without first
clearing the stale, partial `omega_log.jsonl` from the failed attempt)
corrupted the data — `run_live_eval_arm.py`'s per-batch `run_id` is
DETERMINISTIC (`{arm}_{mode}_seed{N}_batch{i}`, not timestamped), and
`OmegaLogger` opens in append mode, so the retry's fresh batch0/batch1
episodes landed in the SAME file under the SAME `run_id` as the failed
attempt's partial episodes, inflating `seed260`'s episode count to 25
(should be 21) with duplicate-labelled data. **Caught before any
number was trusted**, by checking `Counter(run_id)` per omega log and
noticing `batch0`/`batch1` had 4 entries instead of 2. Fixed by
deleting both contaminated `rep_seed954` directories entirely (not
attempting a surgical row-level fix, which would have required trusting an
assumption about append order under concurrent-looking but actually
sequential writes) and re-running those two blocks cleanly from a
freshly restarted, freshly health-checked stack. Verified clean after
the redo: exactly 2/2/1 rows per batch, matching the intended 5-episode
protocol.

## Result

| Checkpoint (training seed) | Live compliance | vs.\ static-at-cap (44/46) |
|---|---|---|
| 256 (original, already reported) | 44/46 (95.7%) | tied, $p=1.0$ |
| 257 | **13/21 (61.9%)** | **significantly worse, $p=0.0009$** |
| 258 | 21/21 (100%) | not distinguishable, $p=1.0$ |
| 259 | 21/21 (100%) | not distinguishable, $p=1.0$ |
| 260 | 19/21 (90.5%) | not distinguishable, $p=0.58$ |
| 261 | 21/21 (100%) | not distinguishable, $p=1.0$ |

**Seed257's failure is a genuine checkpoint property, not a hardware
artifact or an evaluation-order effect.** Its own per-eval-seed
breakdown: 0/2, 0/2, 0/2 (seeds 950–952), then 3/5, 5/5, 5/5 (seeds
953–955) — a real, systematic collapse concentrated in its FIRST three
eval blocks. This could look like a "rig was still warming up" artifact
(seed257 was the first checkpoint tested this session) — **ruled out**
by checking seeds 258/259/260/261's own 950–952 blocks (tested later,
after seed257): all are 2/2, 2/2, 2/2, uniformly clean. If the early
blocks were systematically bad regardless of checkpoint, every arm
would show it there; only seed257 does.

## Interpretation

**Training itself converges consistently across seeds (Stage 10), but
the resulting LIVE robustness of the converged policy does not.** 4 of
6 independently-trained DQN-SLA checkpoints (256, 258/259/261-perfect,
260) land at or above static-at-cap's rate; one (257) fails
catastrophically, worse than even the static baseline (91.7%). None of
this variance is visible in the offline training reward curve, which
looked equally healthy for all 6 seeds. **Offline training convergence
is not a reliable predictor of live robustness for this policy class
under this reward** — a materially sharper and more specific finding
than Stage 10's original, vaguer "investigate why" Future Work item.

This also reframes Stage 10's own headline finding: seed256's tie with
static-at-cap was not a worst-case or best-case draw — it sits in the
middle of a real distribution that spans a catastrophic failure (257)
to three perfect records (258/259/261). The honest, defensible
statement is not "DQN-SLA ties with static-at-cap" (implying that's
what any trained instance would do) but "an individual trained DQN-SLA
policy's live robustness varies substantially by training seed, in a
way offline metrics do not surface, and the specific checkpoint this
paper evaluated throughout happens to land in the middle of that
range."

## Manuscript impact

Added to Section IV-A (a new paragraph after the static-at-cap
discussion) and Future Work item (1) sharpened accordingly — see
`paper_conf/main.tex`. Table I / the abstract's headline claim (about
`dqn_qoe` and the overall n=46 comparison) are unaffected; this is an
additional, clearly-scoped robustness check on the SLA-only reward
specifically, not a replacement for the main comparison.

## Acceptance status

- [x] Question posed precisely (representative vs.\ checkpoint-specific)
      before running anything.
- [x] Same seed structure as the rest of the campaign, for direct
      comparability.
- [x] Real rig instability encountered and fixed (iperf3-target
      recreate) — not glossed over.
- [x] A real data-corruption mistake (naive retry into an
      append-mode, deterministic-run-id log) caught before any number
      was trusted, root-caused, and fixed by clean re-run rather than
      a surgical patch.
- [x] Finding reported at its actual nuance (real variance, not a clean
      yes/no answer) rather than forced into a simpler story.
- [x] Manuscript and Future Work updated to match.
