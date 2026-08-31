# M36: sim-to-real congestion characterization (live half blocked, offline half done)

Second of the external-review gated sequence (M35-M38, renumbered from
the reviewer's own M28-M31 -- see `docs/PAPER5_M35_metric_disagreement_ledger.md`
for why). M36 asks: using the existing non-invasive state-vector wrapper
(`state_vector_probe.py`, the same one `docs/PAPER5_M8_live_anchor.md`
used to diagnose the original sim-to-real mismatch), log the seed-900
single-agent-DQN policy's `congestion_level` feature per admission
decision, offline across every demand lever, and live at UE counts
{1,2,3,4,5,6} with real per-slice UDP traffic, >=10 episodes each.

## Offline half: done, committed

`experiments/scripts/m36_offline_congestion_sweep.py`,
`experiments/results/m36_congestion_ranges_offline.csv`, commit
`f59fe1d`. Used the M8-trained, dimension-matched checkpoint
(`experiments/results/m8_live_anchor/offline_train/single_agent_dqn/seed900/`
-- NOT M2's seed900, which is 34-dim and fails to load into this
13-dim-state config, confirmed via the same `RuntimeError` M8's own doc
already documents for this exact mismatch). Swept `mean_offered_ratio`
(0.1-1.0) x `synthetic_arrivals_per_step` (2/4/8). Result: p50 pinned at
0.080 identically across every ratio level (the ceiling clamp dominates
typical decisions completely), p99/max scale with ratio (0.24 to 1.0,
only in the rare tail). `synthetic_arrivals_per_step` confirmed to have
no meaningful effect, matching M8's own finding.

## Live half: attempted, blocked on a real infrastructure regression

### Bring-up: fully successful
- Docker containers didn't exist at all (fully removed, not just
  stopped) -- rebuilt from scratch via `docker compose -f
  5g-sa-deploy-slicing.yaml up -d`. All 17 containers up clean.
- Subscriber DB **survived** the container recreation (Docker named
  volume persisted) -- all 9 pre-provisioned IMSIs (776-784, covering up
  to 9 UEs / 3 per slice) were already present, confirmed via
  `showfiltered`, no re-provisioning actually needed (my own
  provisioning attempt correctly failed with duplicate-key errors,
  which is how this was discovered).
- gNB: `sudo ./nr-softmodem -O <106PRB conf> --sa --rfsim`, rfsimulator
  listening on `0.0.0.0:4043`, E2 agent heartbeating, zero errors.
- UE1 (embb, default netns, `nrUE_slice1.conf`): attached cleanly (RSRP
  -42dBm, matching every prior successful bring-up), `oaitun_ue1` at
  192.168.100.2, **0% packet loss pinging 8.8.8.8** (real internet
  reachability confirmed).
- Memory checked after every single step (core, gNB, UE1): stayed
  healthy throughout, 2.5-4.9Gi available, never dropped near the
  historical "tight" territory BRINGUP_LOG documents for 3+ UEs. Zero
  crash/OOM signatures in dmesg across the entire session.

### The blocker: iperf3-target unreachable from UE1, cause not found
Needed real per-slice UDP traffic per M36's own explicit requirement (an
idle-attached UE's `congestion_level` sits flat at the floor value --
confirmed directly: 318 consecutive live decisions all read exactly
0.05, zero variance, before I caught this and stopped that run rather
than let it burn the full 10-episode budget on invalid data).

The `iperf3-target` container setup is **extensively documented as
working** across at least 5 prior stages (STAGE3/5/10/11/15,
`experiments/configs/traffic_profiles.yaml`'s own creation-command
comment: "verified: UE -> gNB -> UPF -> docker bridge -> this container
... no host-level NAT/routing changes needed"). Recreated it with the
exact documented command (`docker run -d --name iperf3-target --network
demo-open5gs-public-net --ip 172.22.0.50 ...`, then the multi-port
variant from `run_stage15_n128_campaign.sh`'s own wedge-recovery
routine). It did not work this session.

**Diagnosed and ruled out, in order:**
1. Wrong subnet/network -- confirmed `demo-open5gs-public-net` is
   exactly what UPF's own compose definition uses.
2. ARP failure -- `ip neigh show` inside `upf-slice1` shows a resolved
   (if stale) entry for 172.22.0.50.
3. Masquerade rule absent -- present (`MASQUERADE ... 192.168.100.0/24
   -> 0.0.0.0/0`), and its packet counter **does increment** on every
   ping/iperf3 attempt from UE1 (confirmed via `iptables -t nat -L -v`
   before/after), meaning UE1's outbound packets ARE reaching and
   matching this rule.
4. IP forwarding disabled -- checked, enabled (`1`).
5. FORWARD chain blocking -- checked, default-ACCEPT policy, no
   blocking rule.
6. Wrong UPF instance -- confirmed via AMF logs (`DNN[oai]
   S_NSSAI[SST:1 SD:0xffffff]`) and each UPF's own `ogstun` subnet that
   `upf-slice1` (172.22.0.8) is the correct, only instance handling
   UE1's session.
7. Reverse-path filtering -- host bridge and UPF both show `rp_filter=2`
   (loose); temporarily set to `0` on both as a test, no change,
   reverted back to `2` afterward.
8. UPF-to-target connectivity in isolation -- `docker exec upf-slice1
   ping 172.22.0.50` succeeds cleanly (0% loss) when run directly from
   the UPF container itself.

**What this leaves**: outbound UE1->UPF->(masqueraded)->target packets
are provably being generated and matched by the correct NAT rule, and
the UPF container can reach the target fine on its own, but the
round trip does not complete for UE-originated traffic specifically. A
host-level `tcpdump` on the docker bridge during a live ping attempt
captured **zero** ICMP packets, which is hard to reconcile with the
incrementing NAT counter and deserves more investigation than I could
give it with the tools available inside these minimal containers (no
`tcpdump`, no `conntrack` in either `upf-slice1` or `iperf3-target`).

**Time-boxed at ~90 minutes of live-rig time** before stopping rather
than continuing to consume the session's limited unsupervised window on
a single infrastructure puzzle. This is a genuine, reproducible blocker
on this session's rig state, not a configuration mistake in how I
invoked the documented, previously-working setup -- but I could not
resolve it with the diagnostic tools available.

## Update, same session: iperf3-target blocker resolved (transient), 1-UE done, 2-UE hit a different, real rig instability

After the ~90-minute blocker above, tried once more with better
tooling: installed `tcpdump`/`conntrack` into `upf-slice1` (`apt-get
install`, this image is Debian-based, not Alpine) and did a fresh
gNB+UE1 restart before retesting. **The exact same setup then worked
immediately** -- `tcpdump` on both `upf-slice1` interfaces
(`eth0`/`ogstun`) during a ping showed a clean, complete round trip,
and a real 6-second UDP iperf3 test (4 Mbps, 1200B, matching the embb
profile) confirmed 0% loss. Root cause was never conclusively
identified (the fresh restart could have cleared stale ARP/conntrack
state, or something else entirely) -- recorded honestly as unresolved
rather than claiming a diagnosis the evidence doesn't support.

**1-UE congestion probe: completed successfully.** 10 episodes, real
sustained embb UDP traffic (4 Mbps) throughout. `congestion_level`
pinned flat at 0.140 (n=3600 decisions, zero variance) -- up from the
idle-attached floor of 0.050 measured earlier, confirming real traffic
genuinely moves this feature. Block precision 1.000, zero collapsed
episodes. Committed: `experiments/results/m36_live/ue1/`, commit
`e1b1f3a`.

One live E2 `TimeoutError` (`no INDICATION_RESPONSE from gNB E2 agent
within 30.0s`) occurred on the first launch attempt for this run,
before the successful one -- not previously documented anywhere in
this project's history. Treated as a one-off retriable condition
(matching `run_live_eval_arm.py`'s own established health-check-and-
restart discipline); the immediate retry succeeded cleanly with no
further recurrence during the full 1-UE campaign.

**2-UE attempt: blocked by a real, recurring UE1 radio-link failure,
not a new bug.** Adding UE2 (mmtc, `ue2ns`) itself went cleanly --
netns/veth isolation confirmed correct (UE1's and UE2's `oaitun_ue1`
interfaces are properly separate, different ifindexes, no collision),
both UEs independently reached 0% packet loss to the internet right
after attach. Shortly after starting sustained traffic on both UEs
together, UE1 dropped to 100% packet loss with `[RLC] max RETX reached
on DRB 1` repeating in its log -- **this exact failure signature is
already documented** in this project's own `run_live_eval_arm.py`
docstring as a real, prior production incident ("UE1/embb hit an RLC
max-RETX failure 3 times within ~1 hour of cumulative uptime"), with
`restart_ran_stack.sh` built specifically because hot-restarting a
single UE into a running stack doesn't reliably fix it -- only a full
stop-everything-and-relaunch-in-sequence does.

Did exactly that (full kill of gNB+UE1+UE2, fresh relaunch of gNB, then
UE1, then UE2, each connectivity-checked before proceeding) -- both UEs
came up clean, 0% loss. Launched the 2-UE probe. **Within a few
minutes of both UEs carrying real traffic simultaneously, UE1 failed
the same way again** (100% loss, same RLC max-RETX signature). This is
the second occurrence of a real, previously-documented instability
within one session, not a new failure mode and not something traceable
to a mistake in this session's own procedure (the restart followed the
project's own validated recipe exactly).

**Decision: stopped rather than attempt a third restart cycle.** Two
recurrences of a known-but-supposedly-rare failure within roughly an
hour suggested the rig itself may have been in a degraded state after
several hours of continuous operation (thermal, resource
fragmentation from repeated process churn, or simply this failure mode
being less rare under sustained 2-UE real traffic than the single prior
documented incident suggested) rather than one-off bad luck. The
partial 2-UE state/omega logs collected during the failure window were
discarded (not committed) rather than kept as data, since a KPM/reward
signal measured while UE1's radio link is actively failing reflects a
link-failure artifact, not the demand-driven congestion this milestone
is trying to characterize -- keeping it would risk contaminating the
eventual UE-count-vs-congestion curve with a qualitatively different
phenomenon.

## State at end of this entry: fully torn down, clean

All RAN processes, traffic generators, and the `iperf3-target`
container stopped/removed; core network containers stopped (not
removed -- DB volume and container definitions preserved for a fast
resume). `ue2ns` network namespace left in place (cheap to keep,
matches `restart_ran_stack.sh`'s own idempotent-creation convention).
Verified at teardown: zero RAN/traffic processes, zero running
containers, memory fully recovered to 5.3Gi available, **zero
crash/OOM signatures in dmesg across the entire ~4-hour session**
(confirming the RLC failures are a radio-link/RRC-layer protocol
issue, not a kernel-level crash, memory exhaustion, or the M28 2-gNB
incident's resource-contention pattern -- this was single-gNB
throughout).

**Correction (2026-09-01):** that "zero RAN/traffic processes" claim
was wrong. Four `sudo`-launched `iperf3 --reverse` bash while-loops
from the mmtc traffic generator (started 19:23-19:32, i.e. *before*
the 19:42 container teardown) were never actually killed -- they sat
retrying every 6s against the by-then-stopped `iperf3-target` for
over 5 hours before being caught and killed the next session. No live
RF was involved (`nr-softmodem`/`nr-uesoftmodem` were correctly not
running), so this was wasted CPU/log churn, not a hardware-risk gap --
but the teardown verification that produced the "zero" claim above did
not actually check for `sudo`-owned background loops, only the
processes this session's own kill commands targeted directly. Fixed:
killed via `sudo kill -9`, confirmed clean.

## Recommended next steps (for whoever resumes this)
1. **Before the next attempt, consider whether this rig needs a cooldown
   period or reboot** rather than an immediate retry -- two RLC
   max-RETX recurrences within about an hour, after several hours of
   continuous gNB/UE operation across this session's earlier work, is
   worth treating as a real signal rather than assumed-independent bad
   luck, even though it isn't conclusively diagnosed as thermal or
   resource-related.
2. **Re-attempt 2-UE (and beyond) fresh**, ideally with the user present
   given the now-twice-observed instability under 2-UE combined real
   traffic specifically (1-UE alone ran a full 10-episode, real-traffic
   campaign without a single RLC error) -- this may be a genuine
   2-UE-specific finding worth its own investigation rather than purely
   an unlucky rig state, and distinguishing those two possibilities
   likely needs closer, real-time observation than a single unattended
   session allows.
3. If it recurs a third time under the same conditions, that upgrades
   from "worth flagging" to "a real, reportable finding about this
   rig's live multi-UE ceiling" analogous to M28's 2-gNB hardware
   ceiling -- write it up rather than keep treating it as transient.
4. Once 2-UE is stable for a full campaign, continue the UE-count
   sequence (3, 4, 5, 6) exactly as originally planned, with a
   health/memory/connectivity check after every addition, matching the
   discipline that got 1-UE and the initial 2-UE attach (before traffic
   load) both cleanly verified.
