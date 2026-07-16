"""RANEnv: a gym-like reset()/step() environment implementing the paper
#1/#2 admission-control MDP against any KpmSource (replay, synthetic, or
live). No such step()/reset() abstraction exists in drl_slicing's runner.py
(confirmed during Stage Zero design survey) -- this is new.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .action_mapping import BLOCK_MAPPING_LIMITATION, AdmissionGate
from .config import SacLbExperimentConfig
from .kpm_adapter import build_cluster_state
from .qoe_mapper import (
    QOE_FEATURE_DIM, LatencyProxy, QoEMapper, RollingKpmWindow, build_qoe_features, iqx_mos,
)
from .replay_kpm_source import KpmSource
from .reward import QOE_DIAGNOSTIC_ONLY_LIMITATION, compute_qoe_reward, compute_step_reward
from .types import AdmissionRequest, ClusterState, StepResult, UeSample

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def encode_state(cluster_state: ClusterState, cfg: SacLbExperimentConfig) -> np.ndarray:
    """eq. 1: per-gNB, per-slice [U_k(t)/B, C_t, L_k(t)/Lmax], concatenated
    across gNBs in cfg.gnb_ids order and slices in cfg.slices order, with
    the cluster-wide fairness ratio F_t appended once as a global scalar
    (paper #2 only; omitted when only one gNB is configured, matching
    paper #1's single-gNB formulation which has no F_t term)."""
    features: List[float] = []
    for gnb_id in cfg.gnb_ids:
        slice_states = cluster_state.per_gnb.get(gnb_id, {})
        for spec in cfg.slices:
            agg = slice_states.get(spec.slice_id)
            if agg is None:
                features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([agg.prb_used_ratio, agg.congestion_level, agg.queue_len_norm])
    if len(cfg.gnbs) > 1:
        features.append(cluster_state.fairness_ratio)
    return np.array(features, dtype=np.float32)


def state_dim(cfg: SacLbExperimentConfig) -> int:
    dim = len(cfg.gnbs) * len(cfg.slices) * 3
    if len(cfg.gnbs) > 1:
        dim += 1
    return dim


def encode_request_context(request: AdmissionRequest, cfg: SacLbExperimentConfig) -> np.ndarray:
    """One-hot [slice_onehot..., gnb_onehot...] identifying which pending
    request a decision is for.

    The papers' MDP considers one request per step; Stage Zero's RANEnv
    instead batches every request pending at a step (see env.py module
    docstring / build_order notes) so a single shared cluster observation
    can cover several simultaneous admission decisions. Since
    oranslice_drl.RLPolicy.select_action takes one state vector and returns
    one action, each request in the batch is decided by concatenating this
    per-request identity onto the shared cluster observation and calling
    select_action once per request -- this is how a single reused policy
    instance makes slice/gNB-differentiated decisions within one step,
    rather than needing a bespoke multi-request policy interface.
    """
    slice_oh = [1.0 if request.slice_id == s.slice_id else 0.0 for s in cfg.slices]
    gnb_oh = [1.0 if request.gnb_id == gid else 0.0 for gid in cfg.gnb_ids]
    return np.array(slice_oh + gnb_oh, dtype=np.float32)


def request_state_dim(cfg: SacLbExperimentConfig) -> int:
    return state_dim(cfg) + len(cfg.slices) + len(cfg.gnb_ids)


def encode_full_request_state(
    obs: np.ndarray, request: AdmissionRequest, cfg: SacLbExperimentConfig
) -> np.ndarray:
    return np.concatenate([obs, encode_request_context(request, cfg)]).astype(np.float32)


class RANEnv:
    def __init__(
        self, cfg: SacLbExperimentConfig, kpm_source: KpmSource, seed: Optional[int] = None,
        reward_mode: str = "sla",
    ):
        """reward_mode: "sla" (default, eq.2, Stage Zero's frozen baseline,
        unaffected by any of Stage One's QoE machinery below) or "qoe"
        (eq.9, Stage One's "+QoE-driven-ratio" ablation arm -- requires
        cfg.qoe to be set). See reward.py's compute_step_reward vs
        compute_qoe_reward for why these are two distinct reward shapes,
        not one reward with a swapped term.
        """
        if reward_mode not in ("sla", "qoe"):
            raise ValueError(f"reward_mode must be 'sla' or 'qoe', got {reward_mode!r}")
        if reward_mode == "qoe" and cfg.qoe is None:
            raise ValueError("reward_mode='qoe' requires cfg.qoe to be set (no 'qoe:' section in config)")

        self.cfg = cfg
        self.kpm_source = kpm_source
        self.reward_mode = reward_mode
        self.gate = AdmissionGate(
            cfg.slice_by_id, cfg.gnb_ids, step_ratio=cfg.arrivals.ceiling_step_ratio
        )
        self._rng = np.random.RandomState(seed if seed is not None else cfg.random_seed)
        self._step_in_episode = 0
        self._episode_idx = 0
        self._global_step = 0
        self._seen_rntis: set = set()
        self._pending: List[AdmissionRequest] = []
        self._last_cluster_state: Optional[ClusterState] = None
        self.include_lb_term = cfg.paper_variant == "paper2"

        self._latency_proxy: Optional[LatencyProxy] = None
        self._kpm_window: Optional[RollingKpmWindow] = None
        # One QoEMapper PER SLICE, not one shared model -- calibration/
        # train_lstm.py trains a separate LSTM per slice (each slice's
        # objective-label generator and IQX prior differ enough that a
        # single shared model would blur very different QoS-to-MOS
        # relationships together). A slice absent from cfg.qoe.mapper_checkpoint
        # falls back to its IQX prior alone (no LSTM refinement).
        self._qoe_mapper: Dict[str, QoEMapper] = {}
        self._latency_scale_s: Dict[str, float] = {}
        # Built whenever cfg.qoe is configured, NOT gated on reward_mode=="qoe":
        # a reward_mode="sla" run (the frozen ratio-control baseline) still
        # needs MOS/cost/sla_viol computed as PASSIVE diagnostics so the
        # Stage One ablation can report "inferred-vs-objective MOS alignment"
        # for BOTH arms on equal footing -- see step()'s reward_mode=="sla"
        # branch, which discards compute_qoe_reward's scalar reward and keeps
        # only its diagnostic sub-fields. This does not change the baseline's
        # actions, ceilings, or reward signal in any way (CHANGE ONE THING).
        if cfg.qoe is not None:
            qcfg = cfg.qoe
            self._latency_proxy = LatencyProxy(max_staleness=qcfg.max_staleness)
            self._kpm_window = RollingKpmWindow(window=qcfg.window, feature_dim=QOE_FEATURE_DIM)
            # queue_len_norm==1.0 (the SLA-violation threshold, see
            # reward.check_violations) is mapped onto EACH slice's own
            # already-configured latency_budget_ms -- ties the abstract,
            # Lmax-normalised backlog scale to a real per-slice deadline
            # that's already load-bearing elsewhere in this config, instead
            # of inventing a second, disconnected latency constant.
            self._latency_scale_s = {s.slice_id: s.latency_budget_ms / 1000.0 for s in cfg.slices}
            if torch is not None:
                for slice_id, ckpt_path in qcfg.mapper_checkpoint.items():
                    if not ckpt_path:
                        continue
                    mapper = QoEMapper(window=qcfg.window, hidden=qcfg.lstm_hidden)
                    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                    mapper.load_state_dict(state)
                    mapper.eval()
                    self._qoe_mapper[slice_id] = mapper

    def reset(self) -> np.ndarray:
        self._step_in_episode = 0
        self._episode_idx += 1
        self.gate.reset_ceilings()
        ue_samples = self.kpm_source.poll()
        self._last_cluster_state = self._build_cluster_state(ue_samples)
        self._pending = self._synthesize_requests(ue_samples)
        return encode_state(self._last_cluster_state, self.cfg)

    def pending_requests(self) -> List[AdmissionRequest]:
        return list(self._pending)

    @property
    def last_cluster_state(self) -> Optional[ClusterState]:
        return self._last_cluster_state

    def step(self, actions: List[int]) -> StepResult:
        if self._last_cluster_state is None:
            raise RuntimeError("step() called before reset()")

        self._global_step += 1
        apply_result = self.gate.apply(
            self._pending, actions, self._last_cluster_state, self._global_step
        )

        for (gnb_id, slice_id), ceiling in apply_result.ceilings.items():
            spec = self.cfg.slice_by_id[slice_id]
            self.kpm_source.send_control(
                gnb_id, spec.sst, spec.sd, ceiling.min_ratio, ceiling.max_ratio
            )

        rejected_counts: Dict[Tuple[str, str], int] = {}
        for block in apply_result.primary_blocks:
            key = (block.gnb_id, block.slice_id)
            rejected_counts[key] = rejected_counts.get(key, 0) + 1
        for (gnb_id, slice_id), n_rejected in rejected_counts.items():
            self.kpm_source.notify_rejected(gnb_id, slice_id, n_rejected)

        if self.reward_mode == "qoe":
            mos_by_slice = self._compute_mos_by_slice(self._last_cluster_state)
            reward, reward_info = compute_qoe_reward(
                self._last_cluster_state,
                self.cfg.slice_by_id,
                apply_result.accepted_counts,
                mos_by_slice,
                self.cfg.qoe.reward,
            )
        else:
            reward, reward_info = compute_step_reward(
                self._last_cluster_state,
                self.cfg.slice_by_id,
                apply_result.accepted_counts,
                self.cfg.reward,
                include_lb_term=self.include_lb_term,
            )
            # Passive QoE diagnostics: computed and logged for ablation
            # comparability against reward_mode="qoe" runs, but this scalar
            # compute_qoe_reward() return value is discarded -- the actual
            # `reward` above (eq.2, compute_step_reward) is what drives this
            # run's actions/training, completely unaffected. See __init__'s
            # comment on why self._qoe_mapper etc. are built whenever
            # cfg.qoe is set, not gated on reward_mode.
            if self.cfg.qoe is not None:
                mos_by_slice = self._compute_mos_by_slice(self._last_cluster_state)
                _, qoe_diag = compute_qoe_reward(
                    self._last_cluster_state,
                    self.cfg.slice_by_id,
                    apply_result.accepted_counts,
                    mos_by_slice,
                    self.cfg.qoe.reward,
                )
                reward_info["mean_mos"] = qoe_diag["mean_mos"]
                reward_info["mos_by_slice"] = qoe_diag["mos_by_slice"]
                reward_info["cost"] = qoe_diag["cost"]
                reward_info["sla_viol"] = qoe_diag["sla_viol"]
                existing_limitation = reward_info.get("limitation", "")
                reward_info["limitation"] = (
                    (existing_limitation + "; " if existing_limitation else "") + QOE_DIAGNOSTIC_ONLY_LIMITATION
                )

        ue_samples = self.kpm_source.poll()
        next_cluster_state = self._build_cluster_state(ue_samples)
        self._last_cluster_state = next_cluster_state
        self._pending = self._synthesize_requests(ue_samples)

        self._step_in_episode += 1
        done = self._step_in_episode >= self.cfg.episode.steps_per_episode

        limitations = list(next_cluster_state.limitations) + [BLOCK_MAPPING_LIMITATION]
        if "limitation" in reward_info:
            limitations.append(reward_info["limitation"])

        # Full current ceiling SNAPSHOT (not just apply_result.ceilings,
        # which only holds entries that changed THIS step) -- the actual
        # slicing_control_m max_ratio commanded to the gNB, per (gNB,
        # slice), every step. This is the real controllable output each
        # policy produces; block/accept counts are only its downstream
        # effect. Logged so live runs can plot the ceiling trajectory
        # directly, not just infer it from block counts.
        ceilings_snapshot: Dict[str, Dict[str, int]] = {}
        for gnb_id in self.cfg.gnb_ids:
            for slice_id in self.cfg.slice_by_id:
                c = self.gate.ceiling_for(gnb_id, slice_id)
                ceilings_snapshot[f"{gnb_id}:{slice_id}"] = {
                    "min_ratio": c.min_ratio, "max_ratio": c.max_ratio,
                }

        info: Dict[str, Any] = {
            "episode": self._episode_idx,
            "step": self._step_in_episode,
            "global_step": self._global_step,
            "primary_blocks": [vars(b) for b in apply_result.primary_blocks],
            "secondary_blocks": [vars(b) for b in apply_result.secondary_blocks],
            "primary_block_count": len(apply_result.primary_blocks),
            "secondary_block_count": len(apply_result.secondary_blocks),
            "accepted_counts": apply_result.accepted_counts,
            "fairness_ratio": next_cluster_state.fairness_ratio,
            "ceilings": ceilings_snapshot,
            "reward_breakdown": reward_info,
            "limitations": limitations,
        }
        return StepResult(
            obs=encode_state(next_cluster_state, self.cfg), reward=reward, done=done, info=info
        )

    def close(self) -> None:
        self.kpm_source.close()

    def _compute_mos_by_slice(self, cluster_state: ClusterState) -> Dict[str, float]:
        """Stage One's QoE-mapper inference path: per (gNB, slice), track
        the held/staleness-tagged latency proxy and rolling KPM window
        (see qoe_mapper.py module docstring for why latency needs this
        treatment -- there's no literal E2SM-KPM latency field on this OAI
        build), compute the IQX closed-form prior, then refine it through
        the trained LSTM if a checkpoint was loaded (cfg.qoe.mapper_checkpoint),
        else fall back to the IQX prior alone. Returns one MOS per slice,
        aggregated (mean) across gNBs -- reward.compute_qoe_reward then
        aggregates across slices for the single scalar reward term.
        """
        assert self._latency_proxy is not None and self._kpm_window is not None
        qcfg = self.cfg.qoe
        mos_by_slice: Dict[str, float] = {}
        for slice_id in self.cfg.slice_by_id:
            per_gnb_mos = []
            for gnb_id in self.cfg.gnb_ids:
                agg = cluster_state.per_gnb.get(gnb_id, {}).get(slice_id)
                if agg is None:
                    continue
                # raw_queue_len_norm is Lmax-normalised (see kpm_adapter.py);
                # queue_len_norm==1.0 is DEFINED here as "at this slice's own
                # configured latency budget" -- see __init__'s comment.
                raw_norm = agg.raw_queue_len_norm
                held_norm, staleness = self._latency_proxy.update(gnb_id, slice_id, raw_norm)
                latency_s = held_norm * self._latency_scale_s.get(slice_id, 0.0)

                coeffs = qcfg.iqx_coeffs.get(slice_id)
                prior_mos = float(iqx_mos(
                    latency=latency_s, packet_loss=agg.loss_proxy,
                    throughput=max(agg.prb_used_ratio, 1e-3), coeffs=coeffs,
                )) if coeffs is not None else 3.0

                mapper = self._qoe_mapper.get(slice_id)
                if mapper is not None and torch is not None:
                    feat = build_qoe_features(
                        latency_norm=held_norm, staleness=staleness, max_staleness=qcfg.max_staleness,
                        packet_loss=agg.loss_proxy, throughput_norm=agg.prb_used_ratio,
                        iqx_prior_mos=prior_mos,
                    )
                    window = self._kpm_window.push(gnb_id, slice_id, feat)
                    with torch.no_grad():
                        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                        mos = float(mapper(x).item())
                else:
                    mos = prior_mos
                per_gnb_mos.append(mos)
            mos_by_slice[slice_id] = (sum(per_gnb_mos) / len(per_gnb_mos)) if per_gnb_mos else 3.0
        return mos_by_slice

    def _build_cluster_state(self, ue_samples: List[UeSample]) -> ClusterState:
        return build_cluster_state(
            ue_samples, self.cfg.gnb_ids, self.cfg.slice_by_id, self.cfg.B, self.cfg.Lmax,
            timestamp_s=float(self._global_step),
        )

    def _synthesize_requests(self, ue_samples: List[UeSample]) -> List[AdmissionRequest]:
        """One request per newly-observed RNTI (real UE attach), plus a
        config-driven number of synthetic background arrivals so the RL
        problem stays well-posed even when live UE churn is low. Seeded via
        self._rng for reproducibility."""
        requests: List[AdmissionRequest] = []
        default_gnb = self.cfg.gnb_ids[0] if self.cfg.gnb_ids else ""
        for ue in ue_samples:
            if ue.rnti not in self._seen_rntis:
                self._seen_rntis.add(ue.rnti)
                slice_id = _resolve_slice_id(ue.nssai_sd, self.cfg)
                if slice_id is None:
                    continue
                requests.append(
                    AdmissionRequest(
                        request_id=f"ue:{ue.rnti}:{self._global_step}",
                        slice_id=slice_id,
                        gnb_id=ue.gnb_id or default_gnb,
                        arrival_step=self._global_step,
                        synthetic=False,
                    )
                )

        n_synthetic = self.cfg.arrivals.synthetic_arrivals_per_step
        slice_ids = [s.slice_id for s in self.cfg.slices]
        gnb_ids = self.cfg.gnb_ids
        for i in range(n_synthetic):
            if not slice_ids or not gnb_ids:
                break
            slice_id = slice_ids[self._rng.randint(0, len(slice_ids))]
            gnb_id = gnb_ids[self._rng.randint(0, len(gnb_ids))]
            requests.append(
                AdmissionRequest(
                    request_id=f"synthetic:{self._global_step}:{i}",
                    slice_id=slice_id,
                    gnb_id=gnb_id,
                    arrival_step=self._global_step,
                    synthetic=True,
                )
            )

        max_pending = self.cfg.arrivals.max_pending_per_step
        if max_pending and len(requests) > max_pending:
            requests = requests[:max_pending]
        return requests


def _resolve_slice_id(nssai_sd: int, cfg: SacLbExperimentConfig) -> Optional[str]:
    from .config import SD_TO_SLICE_ID

    slice_id = SD_TO_SLICE_ID.get(int(nssai_sd))
    if slice_id in cfg.slice_by_id:
        return slice_id
    return None
