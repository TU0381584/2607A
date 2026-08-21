# Paper #5 M6: topology-scaling campaign

Status: **pilot complete (3 seeds/arm, full 6-combination grid, real
300/50 episode budget); full 30-seed campaign not yet run.**
Heterogeneous load (Part 2) is resolved, no frozen-source edit needed.
The pilot's own first read of "does GAT-CTDE's edge grow with N"
turned out to be confounded and reversed on correction (Part 3) --
read that section before trusting any raw-reward number anywhere else
in this doc or in an early draft of the pilot report.

## What M6 asks

Does GAT-CTDE's paired edge over single-agent DQN (established at
N=3, fully-connected, in M2) grow with cluster size N and with
topology sparsity, or is the M2 result specific to a small,
fully-connected graph? Extend to N in {3, 7, 19} with ring and
hex-grid adjacency, plus per-gNB heterogeneous load, keeping reward/
action space/per-request TD scheme identical so any measured
difference is attributable to topology alone.

## Part 1: N + adjacency (built, smoke-tested, unblocked)

### Design

- `framework/qoe_oran_framework/marl/topology.py` gained two new pure
  functions, `ring_edges(n)` and `hex_grid_edges(n)`, both returning an
  edge list consumed by the existing, unmodified `build_adjacency` --
  no change to that function or to anything M2/M3/M4 depend on.
  - `ring_edges`: an N-cycle, defined for any N. Degenerate at N=3 (a
    3-cycle already touches every pair, so ring == fully-connected
    there) -- not a bug, just means sparsity only differentiates from
    fully-connected once N>=4.
  - `hex_grid_edges`: concentric hex rings around one centre cell via
    axial coordinates (ring 1 = +6 cells -> N=7; ring 2 = +12 cells ->
    N=19) -- the standard cellular frequency-reuse cluster sizes, which
    is why M6's own N choices land exactly on 7 and 19. Defined only
    for N in {7, 19}; raises rather than guessing for any other N.
  - Verified directly (not assumed): computed degree sequences for both
    match hand-checkable hex geometry exactly -- N=7 gives one
    degree-6 centre + six degree-3 outer cells; N=19 gives 7
    degree-6 interior cells (centre + all 6 ring-1 cells) + 6
    degree-4 ring-2 edge cells + 6 degree-3 ring-2 corner cells, all
    symmetric adjacency matrices.
- Two new YAML configs, `configs/saclb_offline_dqn_n7.yaml` and
  `..._n19.yaml`: diffed directly against the N=3 baseline
  (`saclb_offline_dqn.yaml`) to confirm the ONLY changes are the
  `gnbs:` list (extended to 7/19 entries, same `prb_capacity: 100`
  each) and `arrivals.synthetic_arrivals_per_step` (scaled from 3 to
  7/19, preserving ~1 arrival/gNB/step -- unscaled, N=19 would dilute
  per-node contention roughly 6x relative to N=3, confounding "does N
  matter" with "is the stress regime still actually a stress regime
  at this N," which is not the question M6 asks). Every slice/reward/
  episode parameter is byte-identical to the N=3 config.
- New `experiments/scripts/m6_run_experiment.py`, mirroring
  `m2_run_experiment.py`'s exact per-arm training/eval loop (same
  torch-seed-before-construction fix, same seed-dir-clearing bug
  guard, same checkpoint-architecture-verified resume check) with two
  new parameters: `--config-path` (selects N) and `--topology`
  (fully_connected/ring/hex, consumed only by the `gat_ctde` arm's
  adjacency construction -- `independent_dqn` and `single_agent_dqn`
  never consume an adjacency matrix at all, so topology only reaches
  the environment/contention level for those two arms, never a
  graph-structure level).

### Verification

1. **Regression check**: ran seed 8801, 5 train/3 eval episodes,
   `gat_ctde` arm, N=3/fully_connected through BOTH
   `m2_run_experiment.py` and the new `m6_run_experiment.py`. Byte-
   identical eval omega log, byte-identical train omega log, and
   `torch.equal`-identical checkpoint weights. The new script
   introduces zero behavioural change for the case it must reproduce
   exactly.
2. **Smoke test**: 1 seed (8801), 5 train/3 eval episodes, all three
   arms, at N=7/ring, N=7/hex, and N=19/hex (3 arms). All ran to
   completion with no errors; compliance values were non-degenerate
   (neither 0 nor 1 across the board) at every combination tested.
3. **Timing probe**: 1 seed, FULL M2-scale episode budget (300
   train/50 eval), `gat_ctde` arm only, at N=19/hex (the most
   expensive single combination: largest N, and `gat_ctde` is the only
   arm whose per-step cost scales with graph attention over N nodes).
   Result: TODO(MEASURE) -- launched, not yet complete as of this
   doc's first commit; see the addendum below once it finishes.

## Part 2: per-gNB heterogeneous load (RESOLVED -- correcting the earlier "blocked" finding)

**This section originally said heterogeneous load was blocked without
a frozen-source edit. That was wrong** -- the investigation that
produced it checked `GnbSpec.prb_capacity` (genuinely dead, see below)
and `env.py`'s arrival-assignment RNG (genuinely frozen and uniform),
but missed a third path that was there the whole time. Corrected here
rather than silently edited, per this project's own established
practice (Section IV-E's eval-log bug, the M4 churn-immunity
retraction) of leaving a record of what was wrong and why, not just
the fix.

**`replay_kpm_source.ClosedLoopKpmSource` -- the class that actually
generates offered demand per (gNB, slice) -- already has a first-class
`gnb_load_multiplier: Optional[Dict[str, float]]` constructor
parameter, purpose-built for exactly this.** Its own docstring states
the reason it exists: "Without genuine heterogeneity across gNBs there
is nothing for the LB (load-balance) term to actually resolve." Each
gNB's offered demand is `mean_offered_ratio * gnb_load_multiplier[gnb_id]`.
More surprising: **if the caller doesn't supply one, `ClosedLoopKpmSource`
auto-generates one from `seed`** (uniform in [0.6, 1.4], first gNB
pinned to 1.0) -- and neither `m2_run_experiment.py` nor the M6 script
above ever passed this argument explicitly, so **every M2/M3/M4 seed
run to date already has implicit, seeded per-gNB load heterogeneity
baked in**, just never deliberately controlled or reported as an
experimental variable. This does not invalidate any committed M2/M3/M4
result (the heterogeneity was there, consistently, for every arm and
every seed alike -- it is not a confound between arms since all three
arms see the same per-seed KPM source) but it is worth knowing: this
paper's existing multi-gNB results were never actually homogeneous-load
results in the first place.

`GnbSpec.prb_capacity` remains genuinely dead code (confirmed: no
reference anywhere outside `config.py`) and the arrival-assignment
RNG inside `env.py` remains frozen and uniform -- neither of those
findings was wrong. The path that resolves this was simply a third
one, in a different frozen file (`replay_kpm_source.py`) than the two
already checked, consumed the same way `mean_offered_ratio`/`B`/
`backlog_capacity` already are: as a constructor argument passed from
project-owned code (`make_kpm_source_factory`), not a frozen-file
edit.

### What M6 actually does with this

`make_kpm_source_factory` (both `m2_run_experiment.py` and
`m6_run_experiment.py`) gains a `gnb_load_multiplier_mode` parameter:
- `"default"`: unchanged behaviour -- the existing seeded-random
  [0.6, 1.4] auto-generation, matching every already-committed M2/M3/M4
  result exactly (verified: regression check below).
- `"homogeneous"`: explicit override, every gNB's multiplier forced to
  1.0 -- giving M6 a genuine, deliberate homogeneous-vs-heterogeneous
  comparison pair that did not exist before, since "default" was never
  actually homogeneous to begin with.

This is the pair M7's FedProx-under-heterogeneity task needs: FedProx
against `"homogeneous"` (no real client heterogeneity, matching the M3
doc's own reasoning for why a full FedProx sweep wasn't run there) vs.
FedProx against `"default"`/heterogeneous (genuine per-client
difference for FedProx's proximal term to correct for).

### Verification

1. **Regression check, repeated after adding the parameter**: same
   seed 8801/N=3/fully_connected/5-train/3-eval comparison as Part 1,
   `gnb_load_multiplier_mode="default"` (the parameter's default) vs.
   `m2_run_experiment.py` unchanged -- still byte-identical eval log.
   Adding the parameter did not silently change default behaviour.
2. **The override actually reaches the environment**: same seed 8801,
   `"default"` vs. `"homogeneous"`, otherwise identical arguments --
   612 of 612 compared train-log lines differ (different reward,
   different accepted counts, different ceiling trajectory from step 1
   onward). Confirms the multiplier is not silently ignored anywhere
   downstream.

## Part 3: the pilot's collapse-at-scale signal, chased down

The pilot (3 seeds/arm, all 6 combinations, full 300/50 episode
budget) finished in 15{,}964s (~4.4h; 295.6s/cell average, giving a
real, not extrapolated, full-30-seed-campaign estimate of ~44.3h main
+ ~14.8h for a 10-seed independent-replication pass ~= 59h/~2.5 days
sequential). Its first read looked like two findings: (a) GAT-CTDE's
reward edge over single-agent DQN grows with N (+0.4-0.5 at N=7 vs.
+0.67 at N=19), and (b) compliance collapses toward ~0 for every arm
at N=7/19 versus N=3's established ~0.15 baseline. Chased (a) and (b)
down before trusting either, and both needed correcting.

### (b) is a metric artifact, not an architecture failure -- confirmed by reading the frozen source

`reward.py::check_violations`'s own docstring says plainly: "Per-slice
SLA violation flags for one step (cluster-wide, OR'd across gNBs)."
The actual loop (`for slice_states in cluster_state.per_gnb.values():
... if queue_violation or loss_violation: violated[slice_id] = True`)
confirms it: a slice counts as violated for the WHOLE cluster if ANY
single one of the N gNBs is out of budget on it that step. The
probability that at least one of N gNBs is briefly non-compliant on
any given step rises with N by construction, independent of policy
quality -- so `sla_compliance_all_slices` trending toward 0 as N grows
from 3 to 7 to 19 is expected from the metric's own OR-aggregation
definition, not evidence the architecture (or any arm) is collapsing
worse at scale. This is a second, N-specific instance of exactly the
failure mode Section V of the paper already leads with (a standard
metric that cannot be trusted at face value in this problem class) --
not corrected here (it is frozen source, and the point of this
project's correctness-aware metrics was always to have an alternative
that does not inherit this specific flaw, not to patch the flawed
one), but recorded as a second, independent argument for the paper's
own central thesis, this time on the cluster-size axis rather than the
disruption-severity axis M4 already covers.

### (a) was itself confounded -- raw reward is not comparable across different N, confirmed by reading the frozen source

`action_mapping.py`'s `apply_actions()`: `accepted_counts[slice_id]`
is a single dict, initialized once per step and incremented once per
accepted request across ALL gNBs in that step's combined request list
-- not per gNB. `reward.py::compute_step_reward`'s `service_term =
sum_slice(priority_weight * accept_reward * n_accepted)` therefore
scales with however many gNBs there are, mechanically, the identical
shape of confound (accept-volume inflating raw reward) M4's
`mean_reward_per_pending_request` already had to fix for demand
spikes -- just showing up on the N axis instead of the severity axis
this time. Raw `mean_reward_per_step` comparisons across different N
values are not valid for the same reason raw reward wasn't valid
across spike severities.

**Fix**: `m6_correctness_metrics.py`'s `mean_reward_per_gnb` divides
by N before averaging -- same discipline, new instance. Re-running the
pilot's GAT-CTDE vs. single-agent comparison through it:

| Combo | Raw reward diff (WRONG, confounded) | Per-gNB-normalized diff (n=3, directional) |
|---|---|---|
| N=7, any topology | +0.25 to +0.49 | +0.035 to +0.070 |
| N=19, any topology | +0.672 | +0.035 |

**The "grows with N" reading does not survive correction -- if
anything it shrinks slightly (N=7's ~0.07 to N=19's ~0.035), the
opposite of what the confounded raw numbers suggested.** This would
have been a real, wrong finding if it had gone straight into a 30-seed
campaign write-up on the strength of the naive metric alone -- exactly
the kind of error this paper's own reproduction/replication and
metric-integrity discipline exists to catch before publication, not
after.

### What does hold up: block precision, not reward margin

`block_precision` (mmTC-only-blocking fraction) is untouched by either
confound above -- it was already volume-invariant, the same reason M4
chose it as its own primary metric. In the pilot: GAT-CTDE holds
1.000 precision at both N=7 and N=19 across every topology.
Single-agent DQN holds 1.000 at N=7 but goes fully UNDEFINED (zero
blocks in eval -- full collapse to always-accept) at N=19, across all
three N=19 topologies. Independent DQN sits at 0.787 (N=7) and 0.544
(N=19) -- degraded, not collapsed. **The signal this pilot actually
supports is not "GAT-CTDE's reward edge grows with N" (corrected away
above) but "GAT-CTDE resists always-accept collapse at N=19 in a way
single-agent DQN's flattened-state policy does not, in this 3-seed
sample"** -- a reliability/collapse-resistance claim, not a
reward-margin claim, and still only n=3, not yet a powered result.

### Implication for the full campaign

Any full-scale M6 run must analyze through `mean_reward_per_gnb`
(this file) from the start, not raw `mean_reward_per_step` -- and
should report `sla_compliance_all_slices` trending toward 0 with N as
an expected metric artifact requiring the explanation above, not a
finding, exactly parallel to how the paper already handles
`sla_compliance_all_slices` at the M2/M3 stage. The collapse-resistance
signal (block precision holding for GAT-CTDE, going undefined for
single-agent DQN at N=19) is the more promising thread to power up
first, ahead of committing the full ~59h to the reward-margin question
this pilot just showed was confounded in its naive form.

## Part 4: is the N=19 collapse-resistance signal itself trustworthy? Checked directly -- mostly yes, with one real confound found and fixed

Asked (correctly) not to trust the pilot's remaining signal without
more work, so before recommending anything be scaled up: checked
whether training had genuinely converged (not just run out of budget
mid-improvement), then went to the strongest available evidence --
direct Q-value inspection, the same standard this project's own
original N=3 collapse diagnosis used -- and separately audited my own
N=19 config for a scaling error rather than assuming the config I
wrote was correct.

### Training had converged, not just run out of episodes

Both arms' per-50-episode block-count trajectories at N=19 flatten out
over the last ~100-150 episodes (gat\_ctde: 250.3 -> 242.5 -> 247.6;
single\_agent\_dqn: 48.1 -> 27.9 -> 32.7) rather than still falling
sharply at episode 300 -- consistent with genuine convergence to two
different policies, not one arm simply having less effective training
time than the other within the same 300-episode budget.

### Direct Q-value probe: the collapse is real, and so is GAT-CTDE's differentiation

Loaded seed 900's N=19/hex checkpoints for both arms and ran a fresh
greedy eval episode with a hook computing $Q(\text{accept}) -
Q(\text{reject})$ for every one of that episode's real pending-request
decisions (720 for each arm) -- not a synthetic probe state, the
actual states each policy saw:

| Arm | urllc | embb | mmtc |
|---|---|---|---|
| single\_agent\_dqn | +46.4 (100% accept-preferred) | +39.0 (100%) | **+36.5 (100% accept-preferred)** |
| gat\_ctde | +13.1 (100% accept-preferred) | +3.7 (100%) | **-1.9 (0% accept-preferred, i.e. 100% reject-preferred)** |

single\_agent\_dqn: $Q(\text{accept}) > Q(\text{reject})$ for every
single one of 720 decisions, on all three slices including mmTC -- the
identical unanimous-positive-gap signature the original N=3 collapse
diagnosis used to confirm that one was genuine, not assumed. gat\_ctde:
positive (accept-preferred) for urllc/embb, NEGATIVE (reject-preferred)
for mmTC, unanimously, on all 241 mmTC decisions in the episode --
exactly the reward-optimal differentiated-shedding signature
Section III-B defines. This is not a difference in an aggregate
statistic that could hide seed noise or a measurement quirk: it is two
policies making opposite, internally consistent decisions on the
identical problem, confirmed at the individual-decision level.

### But the N=19 config itself had a real bug -- found by auditing my own scaling choice, not assumed correct

Checked whether the "~1 arrival/gNB/step" scaling target Part 1 claims
was actually delivered, rather than trusting the YAML I wrote. It was
not, at N=19: `experiments/results/m6_pilot/n19_hex/gat_ctde/seed900/eval`'s
`n_pending` was pinned at EXACTLY 12 every single step (zero variance)
-- the unmistakable signature of a hard cap, not organic arrival
variation. Root cause: `synthetic_arrivals_per_step` was scaled
3->7->19 as Part 1 describes, but `max_pending_per_step` (the hard
per-step cap) was left at the N=3 baseline's 12 in both new configs.
19 > 12, so N=19 silently truncated every step's intended 19 arrivals
down to 12 -- delivering ~0.63 pending/gNB/step, not the claimed
~1.0 (N=3 measured 1.148; N=7, never actually hitting its own
un-scaled 12 cap since 7 < 12, measured 1.148 too, matching exactly).
**N=7's pilot cells were unaffected** (confirmed: real variance,
mean 8.04, never pinned) but **N=19's were run under an unintended,
gentler-than-designed stress regime.**

Fixed: both configs' `max_pending_per_step` now scale with the same
ratio as `synthetic_arrivals_per_step` (the original 12/3=4x headroom
margin, preserved: N=7 -> 28, N=19 -> 76). Verified the fix: a 3-episode
smoke run at the corrected N=19 config now shows real variance
(mean n\_pending=22.1, std=4.3, min=19, max=76, per-gNB rate 1.164 --
matching N=3/N=7's ~1.15 to within noise) instead of being pinned.

### Net conclusion

The collapse-resistance signal survives every check thrown at it so
far -- genuine convergence, not undertraining; genuine, unanimous,
decision-level Q-value confirmation on both arms, not an aggregate
artifact -- but it was measured under an environment that was
accidentally less stressed than intended at N=19 specifically. **The
qualitative direction (single-agent DQN collapses, GAT-CTDE
differentiates, at N=19) is well-supported by what actually ran; the
N=19 pilot cells still need to be re-run against the corrected config
before this is reported as a finding about the intended stress
regime**, since a gentler-than-designed environment accidentally
producing this contrast is a different, weaker claim than the intended
regime producing it. Re-running the 9 N=19 cells (3 seeds x 3 arms,
now under the fixed config) is far cheaper than the full campaign --
recommended as the next concrete step, ahead of any further scaling.

## Part 5: N=19 re-run against the fix -- signal confirmed, stronger than before

Re-ran all 9 corrected cells (3 seeds x {fully\_connected, ring, hex},
into new `n19_*_capfix` output dirs -- the original, confounded run is
left in place for comparison, not overwritten). 10{,}244s total
(~2.85h), matching the earlier per-cell cost estimate closely.

**Every part of the collapse-resistance signal survives, and several
parts got MORE decisive, not less:**

- `single_agent_dqn`: **0 total blocks across all 9 cells** (all 3
  seeds x all 3 topologies) -- complete collapse persists even under
  the full, properly-scaled stress level (previously capped at 12
  pending/step, now up to 76). block\_precision is undefined (nan) in
  every cell, not just most of them.
- `gat_ctde`: block\_precision = 1.000 in every cell, now over a much
  larger number of actual blocking decisions (43{,}639 total blocks
  across the 9 cells, vs. a much smaller count under the artificially
  capped run) -- more evidence, not less, of consistent, correctly-
  targeted differentiation.
- Direct Q-value re-probe (same seed-900/hex checkpoints, now
  corrected): `single_agent_dqn` -- $Q(\text{accept}) > Q(\text{reject})$
  on 100% of 1347 real decisions (up from 720, since the environment
  now delivers more real traffic), all three slices, mean mmTC gap
  +29.6. `gat_ctde` -- unanimous accept-preference for urllc (+128.4)
  and embb (+37.3), unanimous REJECT-preference for mmTC (0% accept-
  preferred, mean -16.2, a much larger and more confident gap than the
  buggy run's -1.9). The differentiated-shedding signature got
  sharper under the intended, harsher stress level, not weaker.
- Per-gNB-normalized reward diff (GAT-CTDE vs. single-agent): +0.065
  [+0.000, +0.100], consistent across all three N=19 topologies --
  close to N=7's own +0.035 to +0.070 range from Part 3, not
  meaningfully larger. **The reward-margin question still reads as a
  null once corrected (no clear growth from N=7 to N=19); the
  collapse-resistance question does not** -- these are two different
  axes with two different answers, and only one of them shows an N
  effect in this pilot.

**Conclusion**: the N=19 collapse-resistance finding is no longer
provisional-pending-a-known-confound. It has now been confirmed twice
(once under the accidentally-gentler buggy environment, once under the
corrected one, with the signal strengthening under correction rather
than weakening or reversing) and at the individual-decision level via
direct Q-value inspection both times, matching this project's own
established collapse-diagnosis bar. It is still an n=3 pilot -- not yet
a Wilcoxon-powered, 30-seed claim -- but it is no longer resting on an
uncorrected config bug. Recommended next step: scale the
collapse-resistance axis specifically (N=19, all three arms, more
seeds) ahead of the broader reward-margin/topology-sparsity campaign,
which this pilot's own corrected numbers suggest is likely to be a
null result across the board.

## Part 6: scaled to 6 seeds (900-905) -- the clean n=3 story does not survive, and a real analysis-script bug was found in the process

User's instruction: bring the 30-seed rescale down to a 6-hour, 6-seed
budget (900-905), all three N=19 topologies. Two operational problems
came up during the run itself (both caught and fixed, not silently
absorbed): an orphaned process from an earlier, superseded 30-seed
launch survived a `pkill` and ran concurrently with the correct 6-seed
job for a while, contending for CPU; and because `independent_dqn`/
`single_agent_dqn` have no resume guard (only `gat_ctde` does -- Part 1
already notes this as a known inefficiency, which turned out to also
be a correctness risk), the two concurrent processes raced on
`independent_dqn/seed900`'s directory, leaving it half-deleted
mid-retrain. Both seed900/independent\_dqn (the one corrupted cell,
confirmed via a directory-by-directory audit of every other cell) and
the orphan process itself were fixed/killed before any analysis ran.

### The aggregate block\_precision number was itself hiding the real story -- a bug in this project's own new script

`m6_correctness_metrics.py`'s block\_precision computation sums
`mmtc_blocks`/`total_blocks` across every seed FIRST, then divides
once -- the same pattern `m2_correctness_metrics.py` already uses. At
n=3 this looked clean (1.000 for gat\_ctde, every topology). At n=6 the
aggregate number dropped to 0.50-0.67, and re-checking per-seed
(not pooled) revealed why: seed 901's gat\_ctde total\_blocks was 0 in
ALL THREE topologies, in both the n=3 AND n=6 runs -- it was collapsed
from the very first pilot, silently. A seed contributing (0, 0) to a
summed ratio changes neither the numerator nor the denominator, so it
disappears from the aggregate instead of flagging as undefined the way
a single-seed calculation would. This is not a new bug specific to
n=19 -- `m2_correctness_metrics.py`'s own block\_precision has the
identical structural blind spot, quietly relying on the fact that M2's
established write-up already reports collapse rate as its own
first-class number (21/30, 9/30) alongside precision, rather than
trusting the pooled ratio alone to surface it. M6's analysis had not
yet adopted that same discipline; it now does (below).

### Per-seed collapse/precision breakdown, n=6, all three N=19 topologies

| Arm | Collapsed (0 blocks) | Precision >=0.99 | Other (mixed/wrong-target) |
|---|---|---|---|
| single\_agent\_dqn | **6/6** (all topologies) | 0/6 | 0/6 |
| gat\_ctde, fully\_connected | 2/6 (seeds 901, 904) | 3/6 (900, 902, 903) | 1/6 (905, precision 0.003) |
| gat\_ctde, ring | 2/6 (901, 904) | 2/6 (900, 902) | 2/6 (903: precision **0.000**; 905: 0.501) |
| gat\_ctde, hex | 2/6 (901, 904) | 2/6 (900, 902) | 2/6 (903: 0.000; 905: 0.501) |
| independent\_dqn (topology-invariant) | 0/6 | 0/6 | 6/6, precision range 0.29-0.49 |

**Seed 903 is the clearest single data point that topology sparsity
does matter, contradicting Part 3/5's n=3-based conclusion that it
didn't**: identical seed, identical initial weights, identical
environment stream -- perfect precision (1.0) under fully\_connected,
zero precision (0.0, blocking 43{,}687-43{,}708 requests with not one
of them mmTC) under ring or hex. Seed 905 shows the same qualitative
pattern (0.003 fully\_connected vs. 0.501 ring/hex), smaller in
magnitude. The n=3 pilot's apparent topology-invariance (Parts 3 and 5)
was a real reading of those specific seeds, not a fabrication -- but it
was sample luck: none of seeds 900-902 happen to be topology-sensitive,
and two of the three new seeds (903, 905) are.

### What the finding actually is, corrected

Not "GAT-CTDE resists collapse, single-agent DQN doesn't" (Part 5's
framing). The corrected version: **single-agent DQN's collapse is
total and unanimous (6/6, every topology) -- a real, N=19-specific
escalation from whatever its baseline collapse tendency is at smaller
N. GAT-CTDE's OWN collapse rate at N=19 (2/6, ~33%) sits close to its
already-established ~30% (9/30) collapse rate from the N=3 campaign --
N does not appear to make GAT-CTDE collapse MORE often. What N=19 adds
for GAT-CTDE specifically is a new, previously-unseen partial failure
mode among its non-collapsed seeds (blocking heavily but targeting the
wrong slice), and that new failure mode's frequency depends on
topology.** independent\_dqn is a third, distinct profile: it never
fully collapses but never cleanly differentiates either, consistently
mediocre (0.29-0.49) regardless of topology.

None of this is yet a powered, 30-seed claim -- six seeds is enough to
show the n=3 story was incomplete, not enough to state the corrected
one with real confidence. The honest summary for now: single-agent
DQN's total collapse at N=19 is the most robust single finding in this
whole pilot (6/6, zero exceptions, confirmed twice under two different
environment configs, confirmed at the Q-value level). Everything about
GAT-CTDE's OWN reliability and topology-sensitivity at this scale needs
more seeds before it is a claim rather than a lead.

## Part 7: requested audit before scaling further -- one more real bug found, Part 6's conclusion survives verification (not assumption)

Asked explicitly to audit methodology/results/framework for glitches
before spending more compute. Systematic pass: full completeness sweep
over every cell in the pilot's entire history (not just the latest
run), re-derivation of the per-gNB reward normalization math, a
targeted re-check of `gnb_load_multiplier_mode` threading, and a
direct re-verification (not re-assertion) that N=7 was genuinely never
affected by the arrival-cap bug.

**Found: the same class of bug as Part 6's independent\_dqn/seed900
corruption, this time in `gat_ctde`.** The orphaned 30-seed process
(Part 6) was not confined to `independent_dqn`/`single_agent_dqn` --
it raced the legitimate 6-seed job on `gat_ctde` too, for the three
seeds both processes' seed lists overlapped on late in the run (903,
904, 905, `fully_connected` only -- ring/hex were never touched, same
reasoning as Part 6). `gat_ctde`'s resume guard prevented the
900-902 case (already-complete data verified and reused, not
re-cleared), but seeds 903-905 had no pre-existing checkpoint for
either process to resume from, so both raced to clear-and-train them
independently. Caught by a full-history rollup-count sweep across
every combo in the pilot (not just the newest one): exactly these 3
cells showed 100 rollup episodes instead of 50; every other cell,
including the entire original 6-combination pilot and the earlier
3-seed capfix recheck, showed a clean 50. Confirmed via the same
episode-index-reset check this project's original append-contamination
bug used: sequence 1..50 then 1..50 again, in all three.

Training logs for these seeds were clean (single, un-interleaved
300-episode sequence, no resets) -- only the eval phase was
double-appended -- but rather than trying to determine which of the
two interleaved eval halves the on-disk checkpoint actually
corresponds to, retrained and re-evaluated all three from a full clean
slate (cheap: 3 seeds, 1 arm, 1 topology, ~1390s).

**The corrected numbers**: block *counts* for these three cells are
now exactly half what Part 6 reported (e.g. seed903/fully\_connected:
21{,}791 vs. the contaminated 43{,}582) -- consistent with the
contamination having been two near-identical stacked runs, not two
different ones. **Precision ratios, collapse status, and every
qualitative claim in Part 6 are unchanged**: seed901/904 still fully
collapsed (0 blocks); seed903 still shows perfect precision (1.000) at
fully\_connected and zero (0.000) at ring/hex, the topology-dependence
finding; seed905 still shows near-zero precision (0.003) at
fully\_connected vs. partial (0.501) at ring/hex. Part 6's conclusion
holds -- now because it was checked, not because the contaminated data
happened to average out kindly (which it did here, but that was not
knowable in advance and is not a reason to have skipped the check).

**Also found, not a data bug but a housekeeping one**: the orphan had
already completed a further 28 seeds (906-929) for
`gat_ctde`/`fully_connected` before it was killed -- never overlapping
the legitimate 6-seed job's own range, so single-writer and verified
clean by the same sweep, but outside every documented scope in this
file. Not deleted (real, clean, already-paid-for compute), but flagged
here so it is never mistaken for part of the reported 6-seed sample if
someone later globs `seed*` in that directory without checking this
note.

**Verified clean, no action needed**: N=7's arrival distribution
(mean 8.04, std 1.00, natural variance up to a max of 12 -- never
pinned, unlike N=19's pre-fix signature), the per-gNB reward
normalization arithmetic (division is linear, so dividing before vs.
after the within-episode mean is equivalent -- traced, not just
asserted), and `gnb_load_multiplier_mode`'s threading through all
three arm functions (identical pattern to the already-empirically-
tested `gat_ctde` path; not itself re-run empirically for the other
two arms since `homogeneous` mode is not yet used by any committed
result).

**Also fixed while auditing** (root cause, not just symptom): added
real resume logic to `run_independent_dqn_arm`/`run_single_agent_dqn_arm`
(previously absent -- Part 1 already flagged this as a known
inefficiency; Part 7 shows it was also the exact mechanism that made
concurrent-process corruption possible for those two arms). Both new
resume paths reuse the existing checkpoint's own `load_checkpoint`
(a real `load_state_dict` call, not a weaker existence check) and
`_reload_eval_compliance` now refuses to resume from any eval log
whose rollup count doesn't match exactly or whose episode numbering
isn't a clean, non-decreasing sequence -- the same contamination
signature Parts 6-7 found, now a standing guard against resuming from
one. Verified before use: resumed 6 already-complete cells in 2s
(vs. ~2400s to retrain), reloaded compliance values matched previously
recorded ones exactly.

## Part 8: scaled to 12 seeds (900-911), fully audited clean -- the corrected, current picture

12-seed run (17{,}644s total, resume logic now correctly skipping
already-complete cells throughout) passed a full audit with zero
findings: every one of 108 cells (3 topologies x 3 arms x 12 seeds)
has a matching checkpoint and a clean, exactly-50-rollup, non-reset
eval log; a training-log spot check on 9 of the newly-trained cells
confirmed clean, single, un-interleaved 300-episode sequences; no
concurrent process was ever running alongside this one (checked before
launch and confirmed after completion).

| Arm | Collapsed (0 blocks) | High precision (>=0.99) | Other (mixed/wrong-target) |
|---|---|---|---|
| single\_agent\_dqn | **12/12, every topology** | 0/12 | 0/12 |
| gat\_ctde, fully\_connected | 4/12 | 6/12 | 2/12 |
| gat\_ctde, ring | 4/12 | 5/12 | 3/12 |
| gat\_ctde, hex | 3/12 | 6/12 | 3/12 |
| independent\_dqn (topology-invariant) | 0/12 | 0/12 | 12/12, range 0.29-0.49 |

**What is now well-supported, not just a lead**: single-agent DQN's
total collapse at N=19 (12/12, zero exceptions, three topologies, two
independent seed batches, confirmed at the Q-value level in Part 5).
GAT-CTDE's own collapse rate at N=19 (25-33%) sits close to its
established ~30% N=3 rate -- N does not appear to make GAT-CTDE
collapse noticeably more often, but it is not immune either, and even
among its non-collapsed seeds only 5-6/12 (not all of them) show
clean, correctly-targeted precision.

**Correction to how this should be reported**: the paired reward-margin
comparison (GAT-CTDE vs. single-agent DQN, per-gNB-normalized) is
**not statistically significant at n=12** for any topology (Wilcoxon
$p=0.74$-$0.91$) and the mean difference is now slightly *negative*
for GAT-CTDE in all three. Mechanically sensible, not a red flag:
single-agent DQN's always-accept collapse collects the reward's
service term on every request with no rejection cost, while GAT-CTDE
pays a real service-term cost every time it correctly rejects mmTC --
at this N, that cost is not clearly recovered in the aggregate reward
signal, even though it is exactly the reward-optimal action per this
project's own established calibration. **The honest framing is: this
project's evidence for GAT-CTDE's advantage at N=19 lives entirely in
block precision / collapse resistance, not in the reward-margin
metric** -- reporting a reward-margin advantage here would not be
supported by what actually ran.

Independent-seed replication (disjoint from 900-911) launched next to
check whether this corrected picture holds on seeds it was not tuned
against, per this project's own reproduction-vs-replication discipline
-- see the addendum below once it lands.

## Part 9: independent-seed replication (1000-1002) -- one finding replicates cleanly, one does not

Ran 3 fresh seeds (1000-1002, disjoint from the 900-911 primary
sample), all three N=19 topologies, all three arms (9925s total).
Audited with the same rigor as every prior stage before trusting it:
all 27 cells checked for both train (300 rollups) and eval (50
rollups, no resets) completeness, zero problems found, zero concurrent
processes.

| Arm | fully\_connected | ring | hex |
|---|---|---|---|
| single\_agent\_dqn | 3/3 collapsed | 3/3 collapsed | 3/3 collapsed |
| gat\_ctde | 3/3 collapsed | 2/3 collapsed, 1/3 other (0.005) | 2/3 collapsed, 1/3 high-precision (1.0) |
| independent\_dqn | 0/3 collapsed, precisions [0.333, 0.323, 0.341] | same | same |

**single-agent DQN's total collapse replicates perfectly**: 3/3 on
every topology, joining the primary sample's 12/12 for a combined
15/15 across two disjoint seed batches, zero exceptions. This is now
about as solid a finding as anything in this paper.

**GAT-CTDE's own collapse rate does NOT clearly replicate at the
primary sample's ~30% level.** The replication sample collapsed 7 of 9
(arm, topology) cells (78%) -- notably higher than the primary
12-seed sample's 11 of 36 (31%). At n=3 per topology this could be
small-sample noise (if the true rate really is ~30%, 3/3 collapsed has
roughly a 3% chance by chance alone -- unlikely but not
disqualifying), or it could mean the primary sample's ~30% reading
itself doesn't generalize as cleanly as Part 8 suggested. Both
samples agree GAT-CTDE collapses sometimes and single-agent DQN
collapses always -- that qualitative ordering replicates -- but the
EXACT collapse rate for GAT-CTDE is not yet a number this project
should quote with confidence. More seeds, specifically aimed at
pinning down GAT-CTDE's own collapse rate at N=19 (not the broader
topology-sparsity or reward-margin questions, which Part 8 already
found to be weak or null), is the honest next step -- not yet done,
flagged here rather than assumed.

**independent\_dqn's mediocre-but-never-collapsed profile replicates**
reasonably: precisions 0.32-0.34 in the replication sample, within the
same general range as the primary sample's 0.29-0.49, still
topology-invariant as expected (independent\_dqn never consumes an
adjacency matrix).

### Where this leaves M6, honestly (superseded by Part 10 below)

Two findings now stand on real, audited, twice-independently-sampled
evidence: single-agent DQN's total collapse at N=19, and the
qualitative ordering (single-agent always collapses, GAT-CTDE
sometimes does, independent\_dqn never fully collapses but never
cleanly differentiates either). Two things do not yet stand on solid
evidence: GAT-CTDE's exact collapse rate at N=19 (33% vs. 78% across
the two samples so far), and the reward-margin/topology-sparsity
questions M6 originally set out to answer (Part 8: null on both, at
n=12). Nothing here is fabricated or assumed -- every number in this
document was read from an audited log, and every correction (Parts
6, 7, 9) is a record of a real check catching a real problem, not a
hedge. The single most valuable next increment, if this is continued,
is more seeds on the collapse-rate question specifically -- everything
else in M6's original scope (N=7, topology sparsity as its own axis,
the reward-margin question) already has a fairly clear, if modest,
answer from what has run so far.

## Part 10 (M7): a third sample resolves the collapse-rate discrepancy

Ran under M7 (docs/PAPER5_M7_heterogeneity.md), specifically to answer
Part 9's open question: 6 more seeds (2000-2005, disjoint from both
900-929 and 1000-1029), all three N=19 topologies, gat\_ctde only
(single\_agent\_dqn's 15/15 total collapse and independent\_dqn's
topology-invariant mediocre profile were already well-supported and
not what needed narrowing). Audited the same way as every prior stage:
all 18 new cells checked for exactly 300 train / 50 eval rollups, no
concurrent process running before or during.

**This third sample: 11/18 collapsed (61.1\%)** -- lands between the
primary sample's 31\% and the replication sample's 78\%, neither
confirming nor repeating either one exactly.

**Combined across all three samples (63 cells, 21 independent seeds):
29/63 collapsed, 46.0\%.** Because collapse status for the same seed
across its three topologies is correlated (a seed that collapses at
one topology is more likely to collapse at the others -- confirmed
directly: of the 21 seeds, several collapse at all 3 topologies or none,
not a uniform scatter), the pooled 63-cell ratio understates the real
uncertainty. Bootstrapping at the seed level instead (10,000 resamples
of the 21 seeds' own per-seed collapse fraction across their 3
topologies, not the 63 pooled cells) gives **95\% CI [28.6\%, 65.1\%]**.

| Sample | Seeds | Collapsed | Rate |
|---|---|---|---|
| Primary (900-911) | 12 | 11/36 | 30.6\% |
| Replication (1000-1002) | 3 | 7/9 | 77.8\% |
| Extension (2000-2005) | 6 | 11/18 | 61.1\% |
| **Combined** | **21** | **29/63** | **46.0\%, CI [28.6\%, 65.1\%]** |

**This is now the number to quote**: GAT-CTDE's own collapse rate at
N=19 is approximately 46\% (roughly a coin flip), with real,
substantial seed-to-seed variability -- not a tight, precisely-located
rate the way single-agent DQN's 100\% or independent\_dqn's
never-collapses profile are. Neither of the two smaller samples (31\%,
78\%) was wrong; both were real, honest readings of small samples that
happened to land on opposite sides of a genuinely wide distribution.
The qualitative ordering established in Part 9 (single-agent always
collapses, GAT-CTDE collapses roughly half the time, independent\_dqn
never fully collapses) is unchanged and now rests on a properly-powered
estimate rather than two disagreeing small ones.

## Part 11: a second extension (2006-2014) narrows the estimate further -- 36.7%, not 46.0%

Continuing the same investigation with 9 more seeds (2006-2014,
disjoint from every prior sample), gat\_ctde only, all three
topologies, same script/infra as Part 10's extension. Audited the same
way as every prior stage: all 27 new cells checked for exactly 300
train / 50 eval rollups, no anomalies found, no concurrent process
running before or during (14{,}834s total, ~4.1h).

**This second extension: 4/27 collapsed (14.8\%)** -- notably lower
than the first extension's 61.1\%, itself a real, useful data point
about just how wide seed-to-seed variability is at this N. Treating
both extension batches as one unified 15-seed sample (2000-2014, same
method, same script, launched in two batches only because of a
mid-session time-budget decision, not a methodological difference):
**15/45 collapsed (33.3\%)**, close to the primary sample's 31\% and
notably below the replication sample's 78\%.

**Combined across all four samples now available (90 cells, 30
independent seeds): 33/90 collapsed, 36.7\%.** Seed-level bootstrap 95%
CI (10,000 resamples, same seed-level-not-pooled-cell methodology as
Part 10, since collapse status correlates within a seed across its
three topologies): **[22.2\%, 52.2\%]** -- narrower than Part 10's
three-sample estimate (46.0\% [28.6\%, 65.1\%]), and consistent with
it (Part 10's own interval already contained this final estimate, not
contradicted by it).

| Sample | Seeds | Collapsed | Rate |
|---|---|---|---|
| Primary (900-911) | 12 | 11/36 | 30.6\% |
| Replication (1000-1002) | 3 | 7/9 | 77.8\% |
| Extension (2000-2014) | 15 | 15/45 | 33.3\% |
| **Combined** | **30** | **33/90** | **36.7\%, CI [22.2\%, 52.2\%]** |

**Updated honest read**: three of the four samples (primary, and both
extension batches individually: 61.1\% and 14.8\%, averaging to the
combined extension's 33.3\%) cluster reasonably close to the
combined 36.7\% estimate; the replication sample's 78\% now reads as
the outlier in hindsight, not because it was measured wrong (n=3 is
just genuinely noisy) but because more data eventually clarified which
side of the distribution most samples land on. This is itself a useful
methodological point, not just a numeric update: a single small sample
that happens to land far from the eventual estimate cannot be
distinguished, at the time it is drawn, from a genuinely different
population -- only more independent sampling resolves that, and in
this case resolved it toward the smaller, more commonly-observed rate
rather than the larger one. GAT-CTDE's N=19 collapse rate is now best
understood as roughly one-in-three with real seed-to-seed spread, a
number this project is prepared to quote with confidence, unlike the
31\%/78\%/46\% readings that preceded it. Written into
`paper5/main.tex` Section VII/X (abstract, Fig. 8, Section X-D,
Conclusion) -- all instances of the superseded 46.0\%/21-seed numbers
updated, none left stale.

## Part 12: a third extension (2015-2024), per direct user request -- 37.5%, converging

Direct user instruction: "run more seeds for GAT-CTDE, up to 6 hours of
compute time." 10 more seeds (2015-2024, disjoint from every prior
sample), gat\_ctde only, all three topologies, same script/infra as
Parts 10-11's extensions (`m6_gatctde_collapse_rate_extension3.sh`).
Sized against Part 11's own measured per-cell cost (14,834s / 27 cells
= ~549s/cell average, up to 713s/cell for ring) to fit a 6-hour budget
with margin. Launched only after confirming via `pgrep` that no other
`m6_run_experiment.py` process was running. Completed in 13,595s
(~3.8h), all 30 new cells verified present (non-empty eval omega logs)
before analysis.

**This third batch: 12/30 collapsed (40.0\%)** -- close to the combined
estimate so far (36.7\%), not a new outlier. Treating all three
extension batches as one unified 25-seed sample (2000-2024, same
method/script throughout, batched only for time-budget reasons each
time): **27/75 collapsed (36.0\%)**.

**Combined across primary, replication, and the full three-batch
extension (120 cells, 40 independent seeds): 45/120 collapsed, 37.5\%.**
Seed-level bootstrap 95% CI (10,000 resamples, same
seed-level-not-pooled-cell methodology as Parts 10-11): **[25.0\%,
50.8\%]** -- narrower than Part 11's estimate (36.7\% [22.2\%, 52.2\%]:
30.0-point-wide interval down to 25.8 points), and consistent with it
(Part 11's own interval already contained this final estimate).

| Sample | Seeds | Collapsed | Rate |
|---|---|---|---|
| Primary (900-911) | 12 | 11/36 | 30.6\% |
| Replication (1000-1002) | 3 | 7/9 | 77.8\% |
| Extension (2000-2024, 3 batches) | 25 | 27/75 | 36.0\% |
| **Combined** | **40** | **45/120** | **37.5\%, CI [25.0\%, 50.8\%]** |

**Updated honest read**: the point estimate has now barely moved across
the last two extensions (36.7\% to 37.5\%, well within each other's CI)
while the interval keeps narrowing as more seeds land close to it --
exactly the behaviour expected once a noisy small-sample estimate
starts converging on the true population rate, rather than a sign that
more sampling is still needed to find where the number "really" is.
GAT-CTDE's N=19 collapse rate is best stated as **roughly 37-38%**,
with the replication sample's 78\% now clearly the outlier of the four
samples/batches drawn (primary 30.6\%, extension batches 61.1\%,
14.8\%, 40.0\% -- individually noisy, as expected at n=6-10, but
averaging to 36.0\% and agreeing with the primary sample far more than
with the replication one). Written into `paper5/main.tex` (abstract,
Fig. 8, Section X-D, Conclusion, and the FedProx-adjacent Conclusion
mention) -- all instances of the superseded 36.7\%/30-seed point
estimate updated to 37.5\%/40 seeds, the prior estimate retained in one
place only as an explicit historical comparison, not left as a stale
current number.

## Part 13: a fourth extension (2025-2042), per direct user request -- 36.2%, still converging

Direct user instruction: "run another seed batch for 9 more hours." 18
more seeds (2025-2042, disjoint from every prior sample), gat\_ctde
only, all three topologies, same script/infra as Parts 10-12's
extensions (`m6_gatctde_collapse_rate_extension4.sh`). Sized against
the blended per-cell cost of the two prior extensions (extension2:
14,834s/27 cells = ~549s/cell; extension3: 13,595s/30 cells =
~453s/cell; blended ~500s/cell) to fit a 9-hour budget with margin.
Launched only after confirming via `pgrep` that no other
`m6_run_experiment.py` process was running. Completed in 24,359s
(~6.8h, all three topology blocks landing at a consistent ~8,100s
each -- close to extension3's rate, not extension2's slower one), all
54 new cells verified present (non-empty eval omega logs) before
analysis.

**This fourth batch: 18/54 collapsed (33.3\%)** -- close to the
combined estimate so far (37.5\%), not a new outlier. Treating all four
extension batches as one unified 43-seed sample (2000-2042, same
method/script throughout, batched only for time-budget reasons each
time): **45/129 collapsed (34.9\%)**.

**Combined across primary, replication, and the full four-batch
extension (174 cells, 58 independent seeds): 63/174 collapsed, 36.2\%.**
Seed-level bootstrap 95% CI (10,000 resamples, same
seed-level-not-pooled-cell methodology as Parts 10-12): **[25.9\%,
46.6\%]** -- narrower than Part 12's estimate (37.5\% [25.0\%, 50.8\%]:
25.8-point-wide interval down to 20.7 points), and consistent with it
(Part 12's own interval already contained this final estimate).

| Sample | Seeds | Collapsed | Rate |
|---|---|---|---|
| Primary (900-911) | 12 | 11/36 | 30.6\% |
| Replication (1000-1002) | 3 | 7/9 | 77.8\% |
| Extension (2000-2042, 4 batches) | 43 | 45/129 | 34.9\% |
| **Combined** | **58** | **63/174** | **36.2\%, CI [25.9\%, 46.6\%]** |

**Updated honest read**: the point estimate has held essentially
steady across the last three extensions (36.7\% to 37.5\% to 36.2\%,
each well within the others' CIs) while the interval keeps narrowing
(36.5, then 30.0, then 25.8, now 20.7 points wide) as more seeds
consistently land near the same value -- the signature of a converging
estimate, not one still searching for where the true rate sits.
GAT-CTDE's N=19 collapse rate is best stated as **roughly 36%**, with
the replication sample's 78\% remaining the clear outlier across all
four extension batches now drawn (61.1\%, 14.8\%, 40.0\%, 33.3\% --
individually noisy at n=6-18, but averaging to 34.9\% and tracking the
primary sample far more closely than the replication one). Written
into `paper5/main.tex` (abstract, Fig. 8, Section X-D, Conclusion,
and the FedProx-adjacent Conclusion mention) -- all instances of the
superseded 37.5\%/40-seed point estimate updated to 36.2\%/58 seeds,
the prior estimate retained in one place only as an explicit historical
comparison, not left as a stale current number.
