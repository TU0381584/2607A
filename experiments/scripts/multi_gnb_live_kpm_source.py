#!/usr/bin/env python3
"""MultiGnbLiveKpmSource: wraps N independent LiveKpmSource instances
(one real E2 connection per physical gNB) behind the single KpmSource
protocol RANEnv expects (poll/send_control/notify_rejected/close) --
M28 prep, see docs/PAPER5_M27_M28_scope.md.

Each LiveKpmSource already tags every UE sample with its own fixed
gnb_id at construction (live_kpm_source.py, frozen, unmodified) and
binds its own xapp_listen_port/gnb_listen_port pair, so two instances
with DIFFERENT port pairs can talk to two DIFFERENT physical gNBs
concurrently without touching that file at all -- this wrapper only
adds the fan-out/fan-in RANEnv needs across them.

NOT yet live-tested: gNB2's actual E2 ports are unconfirmed (see
saclb_live2gnb.yaml's header) -- this class is offline-verified only
(construction, protocol shape) pending that confirmation and the
user's presence for the first real 2-gNB live attempt, per this
session's own stated caution around this rig's documented multi-gNB
instability history.
"""
from typing import Dict, List

from qoe_oran_framework.live_kpm_source import LiveKpmSource
from qoe_oran_framework.types import UeSample


class MultiGnbLiveKpmSource:
    def __init__(self, gnb_specs: Dict[str, Dict], recv_timeout_s: float = 30.0):
        """gnb_specs: {gnb_id: {"host": ..., "xapp_listen_port": ..., "gnb_listen_port": ...}}
        one entry per physical gNB, each needing its own distinct
        xapp_listen_port (to avoid two gNBs' INDICATION_RESPONSEs
        colliding on the same socket)."""
        if len(gnb_specs) < 2:
            raise ValueError(
                f"MultiGnbLiveKpmSource is for N>=2 gNBs; got {len(gnb_specs)}. "
                "Use LiveKpmSource directly for a single gNB (saclb_xapp.py already does)."
            )
        ports = [spec["xapp_listen_port"] for spec in gnb_specs.values()]
        if len(set(ports)) != len(ports):
            raise ValueError(f"duplicate xapp_listen_port across gNBs: {ports} -- each gNB needs its own")

        self._sources: Dict[str, LiveKpmSource] = {}
        for gnb_id, spec in gnb_specs.items():
            self._sources[gnb_id] = LiveKpmSource(
                gnb_id=gnb_id,
                host=spec.get("host", "127.0.0.1"),
                xapp_listen_port=spec["xapp_listen_port"],
                gnb_listen_port=spec["gnb_listen_port"],
                recv_timeout_s=recv_timeout_s,
            )

    def poll(self) -> List[UeSample]:
        samples: List[UeSample] = []
        for source in self._sources.values():
            samples.extend(source.poll())
        return samples

    def send_control(self, gnb_id: str, sst: int, sd: int, min_ratio: int, max_ratio: int) -> None:
        if gnb_id not in self._sources:
            raise KeyError(f"send_control targeted unknown gnb_id {gnb_id!r}; known: {list(self._sources)}")
        self._sources[gnb_id].send_control(gnb_id, sst, sd, min_ratio, max_ratio)

    def notify_rejected(self, gnb_id: str, slice_id: str, n_rejected: int) -> None:
        if gnb_id in self._sources:
            self._sources[gnb_id].notify_rejected(gnb_id, slice_id, n_rejected)

    def close(self) -> None:
        for source in self._sources.values():
            source.close()
