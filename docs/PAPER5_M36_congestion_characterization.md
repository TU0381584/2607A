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

## State at end of this entry: fully torn down, clean

UE1, gNB, `iperf3-target` all stopped; core network containers stopped
(not removed -- `docker compose stop`, not `down`, so the subscriber DB
volume and container definitions are preserved for a fast resume).
Verified: zero RAN processes, zero running containers, memory back to
4.9Gi available, zero crash/OOM signatures in dmesg across the entire
~2-hour session. Chosen deliberately over leaving a half-working stack
running unattended for several more hours with no way to make further
M36 progress on it.

## Recommended next steps (for whoever resumes this)
1. **Bring the core+gNB+UE1 back up** (`docker compose ... start`, then
   gNB, then UE1 -- containers and DB are already there, this should be
   fast) and get a **second pair of eyes / `tcpdump`-capable tooling**
   on the UE1-to-iperf3-target path specifically, since every rule-level
   check passed but the actual packet doesn't arrive. Installing
   `tcpdump` into the minimal Alpine-based containers
   (`apk add tcpdump` inside `upf-slice1`/`iperf3-target`, if their
   image has `apk`) would let the exact drop point be found in minutes
   rather than inferred indirectly.
2. **Alternative if that stays blocked**: skip the docker-internal
   target entirely and generate real per-slice load a different way --
   e.g. a UDP echo/sink script run directly in the UE's own default
   netns (loopback-adjacent, sidesteps the UPF-to-docker-bridge hop
   entirely) if the actual requirement is "real, sustained per-slice
   demand the KPM layer measures", not specifically "traffic to this
   particular container".
3. Once real traffic is confirmed flowing (checked directly via
   `dl_mac_buffer_occupation` going non-zero, per BRINGUP_LOG's own
   established verification method, before trusting anything
   downstream), redo the 1-UE congestion probe from scratch (the
   invalid idle-attached run was not saved as a result), then proceed
   UE-count by UE-count exactly as this doc's still-open plan describes,
   with a health/memory check after every addition.
