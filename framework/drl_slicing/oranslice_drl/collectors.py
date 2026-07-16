import csv
import json
import os
import random
import subprocess
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .types import EnvState, SliceMetrics
from .config import SliceTarget
from .types import SliceAction


class MetricsCollector(ABC):
    @abstractmethod
    def get_state(self, step: int) -> EnvState:
        raise NotImplementedError

    def on_action(self, actions: List[SliceAction], targets: Dict[str, SliceTarget], step: int) -> None:
        return


class CsvReplayCollector(MetricsCollector):
    def __init__(self, csv_path: str, known_slice_ids: List[str], step_seconds: int) -> None:
        self.rows = self._read_rows(csv_path)
        self.known_slice_ids = known_slice_ids
        self.step_seconds = step_seconds

    def _read_rows(self, csv_path: str) -> List[dict]:
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [row for row in reader]

    def get_state(self, step: int) -> EnvState:
        if not self.rows:
            raise RuntimeError("CSV replay collector has no rows")
        index = min(step, len(self.rows) - 1)
        row = self.rows[index]

        state = EnvState(timestamp_s=float(row.get("timestamp_s", step * self.step_seconds)))
        default_throughput = float(row.get("throughput_kbps", 0.0))
        default_latency = float(row.get("latency_ms", 0.0))
        default_loss = float(row.get("loss_pct", 0.0))
        default_offered = float(row.get("offered_load_kbps", default_throughput))

        for slice_id in self.known_slice_ids:
            prefix = f"{slice_id}_"
            throughput = float(row.get(prefix + "throughput_kbps", default_throughput))
            latency = float(row.get(prefix + "latency_ms", default_latency))
            loss = float(row.get(prefix + "loss_pct", default_loss))
            offered = float(row.get(prefix + "offered_load_kbps", default_offered))
            state.slices[slice_id] = SliceMetrics(
                slice_id=slice_id,
                throughput_kbps=throughput,
                latency_ms=latency,
                loss_pct=loss,
                offered_load_kbps=offered,
            )
        return state


class PrometheusCollector(MetricsCollector):
    def __init__(
        self,
        base_url: str,
        metric_queries: Dict[str, str],
        known_slice_ids: List[str],
        step_seconds: int,
        enable_action_feedback: bool = False,
        action_feedback_gain: float = 0.35,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.metric_queries = metric_queries
        self.known_slice_ids = known_slice_ids
        self.step_seconds = step_seconds
        self.enable_action_feedback = enable_action_feedback
        self.action_feedback_gain = max(0.0, float(action_feedback_gain))
        self.last_actions: Dict[str, SliceAction] = {}
        self._upf_prev_sample = None
        self._upf_rate_cache = None
        self._ue_slice_map = self._load_ue_slice_map()
        self._ue_prev_samples: Dict[str, Tuple[float, int]] = {}
        self._ue_rate_cache: Optional[Tuple[float, Dict[str, float]]] = None
        self._live_rate_cache: Optional[Tuple[float, Dict[str, float], Dict[str, float]]] = None

    def on_action(self, actions: List[SliceAction], targets: Dict[str, SliceTarget], step: int) -> None:
        self.last_actions = {action.slice_id: action for action in actions}

    def _query(self, query: str, metric_name: str = "") -> Dict[str, float]:
        encoded = urllib.parse.urlencode({"query": query})
        url = f"{self.base_url}/api/v1/query?{encoded}"
        with urllib.request.urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))

        data = payload.get("data", {}).get("result", [])
        out: Dict[str, float] = {}
        for item in data:
            metric = item.get("metric", {})
            slice_id = metric.get("slice_id") or metric.get("snssai") or metric.get("slice")
            if not slice_id:
                continue
            value = float(item.get("value", [0, 0])[1])
            out[slice_id] = value

        if metric_name in {"throughput_kbps", "offered_load_kbps"}:
            out = self._maybe_apply_upf_counter_fallback(query, out, metric_name)
        return out

    def _read_upf_packet_rates(self) -> tuple:
        now = time.time()
        if self._upf_rate_cache is not None:
            cache_ts, rx_rate, tx_rate = self._upf_rate_cache
            if (now - cache_ts) < 0.5:
                return rx_rate, tx_rate

        rx_packets = tx_packets = None
        try:
            result = subprocess.run(
                ["docker", "exec", "upf", "cat", "/proc/net/dev"],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("ogstun:"):
                    fields = stripped.split(":", 1)[1].split()
                    rx_packets = int(fields[1])
                    tx_packets = int(fields[9])
                    break
        except Exception:
            rx_packets = None
            tx_packets = None

        if rx_packets is None or tx_packets is None:
            self._upf_rate_cache = (now, 0.0, 0.0)
            return 0.0, 0.0

        rx_rate = 0.0
        tx_rate = 0.0
        if self._upf_prev_sample is not None:
            prev_ts, prev_rx, prev_tx = self._upf_prev_sample
            dt = max(now - prev_ts, 1e-3)
            rx_rate = max(0.0, float(rx_packets - prev_rx) / dt)
            tx_rate = max(0.0, float(tx_packets - prev_tx) / dt)

        self._upf_prev_sample = (now, rx_packets, tx_packets)
        self._upf_rate_cache = (now, rx_rate, tx_rate)
        return rx_rate, tx_rate

    @staticmethod
    def _repo_root() -> Path:
        # drl_slicing/oranslice_drl/collectors.py -> ORANSlice/
        return Path(__file__).resolve().parents[2]

    def _load_ue_slice_map(self) -> Dict[str, str]:
        candidates: List[Path] = []
        env_path = os.getenv("ORANSLICE_UE_PROFILE_CSV", "").strip()
        if env_path:
            candidates.append(Path(env_path))

        repo_root = self._repo_root()
        candidates.append(repo_root / "docker_open5gs" / "generated" / "ue_fleet_profiles.csv")
        candidates.append(repo_root / "generated" / "ue_fleet_profiles.csv")

        for candidate in candidates:
            if not candidate.is_file():
                continue

            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    mapping: Dict[str, str] = {}
                    for row in reader:
                        container = (
                            row.get("container_name")
                            or row.get("service_name")
                            or ""
                        ).strip()
                        slice_id = (row.get("slice_id") or "").strip()
                        if container and slice_id in self.known_slice_ids:
                            mapping[container] = slice_id
                    if mapping:
                        return mapping
            except Exception:
                continue

        return {}

    def _read_ue_offered_rates(self) -> Dict[str, float]:
        now = time.time()
        if self._ue_rate_cache is not None:
            cache_ts, offered_map = self._ue_rate_cache
            if (now - cache_ts) < 0.5:
                return dict(offered_map)

        offered_kbps: Dict[str, float] = {slice_id: 0.0 for slice_id in self.known_slice_ids}
        if not self._ue_slice_map:
            self._ue_rate_cache = (now, offered_kbps)
            return offered_kbps

        for container, slice_id in self._ue_slice_map.items():
            tx_packets = None
            try:
                result = subprocess.run(
                    ["docker", "exec", container, "cat", "/proc/net/dev"],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                for line in result.stdout.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("uesimtun0:") or stripped.startswith("oaitun_ue"):
                        fields = stripped.split(":", 1)[1].split()
                        tx_packets = int(fields[9])
                        break
            except Exception:
                tx_packets = None

            if tx_packets is None:
                continue

            tx_rate = 0.0
            prev_sample = self._ue_prev_samples.get(container)
            if prev_sample is not None:
                prev_ts, prev_tx = prev_sample
                dt = max(now - prev_ts, 1e-3)
                tx_rate = max(0.0, float(tx_packets - prev_tx) / dt)

            self._ue_prev_samples[container] = (now, tx_packets)
            offered_kbps[slice_id] += tx_rate * 12.0

        self._ue_rate_cache = (now, offered_kbps)
        return dict(offered_kbps)

    def _read_live_e2e_slice_rates(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        now = time.time()
        if self._live_rate_cache is not None:
            cache_ts, throughput_map, offered_map = self._live_rate_cache
            if (now - cache_ts) < 0.5:
                return dict(throughput_map), dict(offered_map)

        offered_map = self._read_ue_offered_rates()
        rx_rate_pps, _ = self._read_upf_packet_rates()
        total_served_kbps = max(0.0, rx_rate_pps * 12.0)
        total_offered_kbps = max(0.0, sum(offered_map.values()))

        throughput_map: Dict[str, float] = {slice_id: 0.0 for slice_id in self.known_slice_ids}
        normalized_offered: Dict[str, float] = {slice_id: float(offered_map.get(slice_id, 0.0)) for slice_id in self.known_slice_ids}

        if total_offered_kbps > 0.0:
            served_ratio = max(0.0, min(1.0, total_served_kbps / total_offered_kbps))
            for slice_id in self.known_slice_ids:
                throughput_map[slice_id] = normalized_offered[slice_id] * served_ratio
        elif total_served_kbps > 0.0:
            # If UE counters are temporarily unavailable, retain a non-zero fallback
            # by distributing served load equally across known slices.
            equal_share = total_served_kbps / max(len(self.known_slice_ids), 1)
            for slice_id in self.known_slice_ids:
                throughput_map[slice_id] = float(equal_share)
                normalized_offered[slice_id] = float(equal_share)

        self._live_rate_cache = (now, throughput_map, normalized_offered)
        return dict(throughput_map), dict(normalized_offered)

    def _distribute_total_by_weights(self, weighted_map: Dict[str, float], total: float) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for slice_id in self.known_slice_ids:
            weights[slice_id] = max(0.0, float(weighted_map.get(slice_id, 0.0)))

        weight_sum = sum(weights.values())
        if weight_sum <= 0.0:
            equal = 1.0 / max(len(self.known_slice_ids), 1)
            return {slice_id: float(total * equal) for slice_id in self.known_slice_ids}

        return {slice_id: float(total * (weights[slice_id] / weight_sum)) for slice_id in self.known_slice_ids}

    def _maybe_apply_upf_counter_fallback(
        self,
        query: str,
        out: Dict[str, float],
        metric_name: str,
    ) -> Dict[str, float]:
        lowered = query.lower()
        is_upf_pkt_query = (
            "fivegs_ep_n3_gtp_indatapktn3upf" in lowered
            or "fivegs_ep_n3_gtp_outdatapktn3upf" in lowered
        )
        if not is_upf_pkt_query:
            return out

        # Some Open5GS builds export the N3 packet metrics but keep them pinned
        # at zero. If that happens, use live UPF interface counters as fallback.
        if sum(out.values()) > 0.05:
            return out

        live_throughput, live_offered = self._read_live_e2e_slice_rates()
        if metric_name == "throughput_kbps" and sum(live_throughput.values()) > 0.0:
            return live_throughput
        if metric_name == "offered_load_kbps" and sum(live_offered.values()) > 0.0:
            return live_offered

        rx_rate_pps, tx_rate_pps = self._read_upf_packet_rates()
        if metric_name == "throughput_kbps":
            total_kbps = rx_rate_pps * 12.0
        else:
            # In some setups traffic is predominantly uplink (UE -> DN), which can
            # keep tx packet counters near zero. Use whichever direction is live.
            total_kbps = max(rx_rate_pps, tx_rate_pps) * 12.0

        if total_kbps <= 0.0:
            return out
        return self._distribute_total_by_weights(out, total_kbps)

    def _maybe_apply_latency_loss_fallback(
        self,
        throughput_map: Dict[str, float],
        offered_map: Dict[str, float],
        latency_map: Dict[str, float],
        loss_map: Dict[str, float],
    ) -> tuple:
        lat_values = [float(latency_map.get(slice_id, 0.0)) for slice_id in self.known_slice_ids]
        loss_values = [float(loss_map.get(slice_id, 0.0)) for slice_id in self.known_slice_ids]

        if not lat_values or not loss_values:
            return latency_map, loss_map

        # Guarded fallback: if exporter values are pinned at clearly pathological
        # levels for all slices, derive SLA surrogates from served-demand ratio.
        if not (min(lat_values) > 80.0 and min(loss_values) > 2.5):
            return latency_map, loss_map

        fixed_latency: Dict[str, float] = {}
        fixed_loss: Dict[str, float] = {}
        for slice_id in self.known_slice_ids:
            offered = max(float(offered_map.get(slice_id, throughput_map.get(slice_id, 0.0))), 1e-6)
            throughput = max(0.0, float(throughput_map.get(slice_id, 0.0)))
            served_ratio = max(0.0, min(1.2, throughput / offered))
            congestion = max(0.0, 1.0 - served_ratio)

            fixed_latency[slice_id] = 10.0 + 55.0 * congestion
            fixed_loss[slice_id] = 0.05 + 2.0 * congestion

        return fixed_latency, fixed_loss

    @staticmethod
    def _render_query(query: str, step: int) -> str:
        # Allow configs to inject step-indexed behavior (for deterministic scenarios)
        # without requiring Prometheus wall-clock alignment.
        return query.replace("__STEP__", str(step))

    def _normalized_shares(self) -> Dict[str, float]:
        if not self.last_actions:
            equal = 1.0 / max(len(self.known_slice_ids), 1)
            return {slice_id: equal for slice_id in self.known_slice_ids}

        raw_shares: Dict[str, float] = {}
        for slice_id in self.known_slice_ids:
            action = self.last_actions.get(slice_id)
            if action is None:
                raw_shares[slice_id] = 1.0
                continue
            raw_shares[slice_id] = max(0.01, (action.min_ratio + action.max_ratio) / 200.0)

        total = sum(raw_shares.values())
        if total <= 0.0:
            equal = 1.0 / max(len(self.known_slice_ids), 1)
            return {slice_id: equal for slice_id in self.known_slice_ids}

        return {slice_id: raw_shares[slice_id] / total for slice_id in self.known_slice_ids}

    def _apply_action_feedback(self, state: EnvState) -> None:
        if not self.enable_action_feedback:
            return
        if not self.last_actions:
            return

        shares = self._normalized_shares()
        baseline_share = 1.0 / max(len(self.known_slice_ids), 1)

        for slice_id in self.known_slice_ids:
            metrics = state.slices[slice_id]
            relative_share = shares[slice_id] / max(baseline_share, 1e-6)
            bias = 1.0 + self.action_feedback_gain * (relative_share - 1.0)
            bias = max(0.35, min(2.2, bias))

            throughput = max(0.0, metrics.throughput_kbps * bias)
            metrics.throughput_kbps = float(min(metrics.offered_load_kbps, throughput))

            latency_scale = max(0.25, min(3.0, 1.0 / max(bias, 1e-6)))
            loss_scale = max(0.2, min(4.0, 1.0 / max(0.8 * bias + 0.2, 1e-6)))

            metrics.latency_ms = float(max(0.1, metrics.latency_ms * latency_scale))
            metrics.loss_pct = float(max(0.0, metrics.loss_pct * loss_scale))

    def get_state(self, step: int) -> EnvState:
        throughput_query = self._render_query(self.metric_queries["throughput_kbps"], step)
        latency_query = self._render_query(self.metric_queries["latency_ms"], step)
        loss_query = self._render_query(self.metric_queries["loss_pct"], step)
        offered_query = self._render_query(
            self.metric_queries.get("offered_load_kbps", self.metric_queries["throughput_kbps"]),
            step,
        )

        throughput_map = self._query(throughput_query, metric_name="throughput_kbps")
        latency_map = self._query(latency_query, metric_name="latency_ms")
        loss_map = self._query(loss_query, metric_name="loss_pct")
        offered_map = self._query(offered_query, metric_name="offered_load_kbps")
        latency_map, loss_map = self._maybe_apply_latency_loss_fallback(
            throughput_map,
            offered_map,
            latency_map,
            loss_map,
        )

        state = EnvState(timestamp_s=step * self.step_seconds)
        for slice_id in self.known_slice_ids:
            state.slices[slice_id] = SliceMetrics(
                slice_id=slice_id,
                throughput_kbps=float(throughput_map.get(slice_id, 0.0)),
                latency_ms=float(latency_map.get(slice_id, 0.0)),
                loss_pct=float(loss_map.get(slice_id, 0.0)),
                offered_load_kbps=float(offered_map.get(slice_id, throughput_map.get(slice_id, 0.0))),
            )

        self._apply_action_feedback(state)
        return state


class OnlineRanRandomCollector(MetricsCollector):
    """
    Online randomized RAN traffic collector with action-feedback dynamics.
    Simulates eMBB/URLLC load variation and KPI response to allocation actions.
    """

    def __init__(
        self,
        known_slice_ids: List[str],
        targets: Dict[str, SliceTarget],
        step_seconds: int,
        random_seed: int,
        total_capacity_kbps: float = 30000.0,
    ) -> None:
        self.known_slice_ids = known_slice_ids
        self.targets = targets
        self.step_seconds = step_seconds
        self.total_capacity_kbps = total_capacity_kbps
        self.rng = random.Random(random_seed)

        self.last_actions: Dict[str, SliceAction] = {}
        self.channel_quality: Dict[str, float] = {sid: 1.0 for sid in known_slice_ids}
        self.prev_offered: Dict[str, float] = {sid: 0.0 for sid in known_slice_ids}

    def on_action(self, actions: List[SliceAction], targets: Dict[str, SliceTarget], step: int) -> None:
        self.last_actions = {action.slice_id: action for action in actions}

    def _slice_profile(self, slice_id: str) -> str:
        target = self.targets[slice_id]
        return "urllc" if target.latency_budget_ms <= 25.0 else "embb"

    def _random_offered_load(self, slice_id: str) -> float:
        profile = self._slice_profile(slice_id)
        if profile == "urllc":
            base = self.rng.uniform(800.0, 4500.0)
            burst = self.rng.uniform(1500.0, 3500.0) if self.rng.random() < 0.22 else 0.0
        else:
            base = self.rng.uniform(7000.0, 19000.0)
            burst = self.rng.uniform(2500.0, 7000.0) if self.rng.random() < 0.28 else 0.0
        offered = 0.65 * self.prev_offered[slice_id] + 0.35 * (base + burst)
        self.prev_offered[slice_id] = offered
        return max(offered, 100.0)

    def _allocation_share(self, slice_id: str, actions_by_slice: Dict[str, SliceAction]) -> float:
        if slice_id not in actions_by_slice:
            return 1.0 / max(len(self.known_slice_ids), 1)
        action = actions_by_slice[slice_id]
        return max(0.01, (action.min_ratio + action.max_ratio) / 200.0)

    def get_state(self, step: int) -> EnvState:
        state = EnvState(timestamp_s=step * self.step_seconds)
        actions_by_slice = self.last_actions

        shares = {
            sid: self._allocation_share(sid, actions_by_slice)
            for sid in self.known_slice_ids
        }
        total_share = sum(shares.values())
        if total_share <= 0:
            total_share = float(len(self.known_slice_ids))

        for slice_id in self.known_slice_ids:
            profile = self._slice_profile(slice_id)

            offered_load = self._random_offered_load(slice_id)
            share = shares[slice_id] / total_share
            allocated_capacity = max(200.0, self.total_capacity_kbps * share)

            q_prev = self.channel_quality[slice_id]
            q_new = 0.82 * q_prev + 0.18 * self.rng.uniform(0.72, 1.18)
            q_new = max(0.55, min(1.25, q_new))
            self.channel_quality[slice_id] = q_new

            raw_capacity = allocated_capacity * q_new
            throughput = min(offered_load, raw_capacity)

            queue_pressure = max(0.0, offered_load - throughput) / max(allocated_capacity, 1.0)
            if profile == "urllc":
                latency = 4.0 + 18.0 * queue_pressure + self.rng.uniform(0.2, 2.0)
                loss = 0.05 + 2.2 * queue_pressure + self.rng.uniform(0.02, 0.35)
            else:
                latency = 14.0 + 26.0 * queue_pressure + self.rng.uniform(0.5, 3.5)
                loss = 0.15 + 1.3 * queue_pressure + self.rng.uniform(0.03, 0.45)

            state.slices[slice_id] = SliceMetrics(
                slice_id=slice_id,
                throughput_kbps=float(max(0.0, throughput)),
                latency_ms=float(max(0.5, latency)),
                loss_pct=float(max(0.0, min(loss, 15.0))),
                offered_load_kbps=float(offered_load),
            )

        return state
