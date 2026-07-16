import json
from pathlib import Path

import numpy as np
import pytest

from qoe_oran_framework.config import load_saclb_config
from qoe_oran_framework.env import (
    RANEnv,
    encode_full_request_state,
    encode_state,
    request_state_dim,
    state_dim,
)
from qoe_oran_framework.replay_kpm_source import ReplayKpmSource, SyntheticKpmSource
from qoe_oran_framework.types import AdmissionRequest, UeSample

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_load_saclb_config_paper1():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    assert cfg.paper_variant == "paper1"
    assert cfg.gnb_ids == ["gnb-0"]
    assert set(cfg.slice_by_id.keys()) == {"urllc", "embb"}
    assert cfg.reward.lb_coeff == 0.0


def test_load_saclb_config_paper2():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_offline_dqn.yaml"))
    assert cfg.paper_variant == "paper2"
    assert cfg.gnb_ids == ["gnb-0", "gnb-1", "gnb-2"]
    assert set(cfg.slice_by_id.keys()) == {"urllc", "embb", "mmtc"}
    assert cfg.reward.lb_coeff == 1.0


def test_state_dim_matches_encode_state_length_single_gnb():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    source = SyntheticKpmSource(seed=1, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=1)
    obs = env.reset()
    assert obs.shape == (state_dim(cfg),)
    # single gNB -> no trailing fairness scalar (paper #1 has no F_t term)
    assert state_dim(cfg) == 1 * 2 * 3


def test_state_dim_includes_fairness_scalar_for_multi_gnb():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_offline_dqn.yaml"))
    assert state_dim(cfg) == 3 * 3 * 3 + 1


def test_reset_then_step_returns_well_formed_step_result():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_offline_dqn.yaml"))
    source = SyntheticKpmSource(seed=7, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=7)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)

    pending = env.pending_requests()
    assert len(pending) > 0
    actions = [1] * len(pending)  # accept everything
    result = env.step(actions)

    assert result.obs.shape == (state_dim(cfg),)
    assert isinstance(result.reward, float)
    assert result.info["step"] == 1
    assert result.info["episode"] == 1
    assert result.info["primary_block_count"] == 0  # all accepted
    assert "limitations" in result.info
    assert any("slicing_control_m" in msg for msg in result.info["limitations"])


def test_step_before_reset_raises():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    source = SyntheticKpmSource(seed=1, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=1)
    with pytest.raises(RuntimeError):
        env.step([])


def test_rejecting_everything_produces_primary_blocks():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    source = SyntheticKpmSource(seed=3, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=3)
    env.reset()
    pending = env.pending_requests()
    result = env.step([0] * len(pending))
    assert result.info["primary_block_count"] == len(pending)


def test_step_notifies_kpm_source_of_rejections_grouped_by_gnb_and_slice():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    source = SyntheticKpmSource(seed=3, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    calls = []
    source.notify_rejected = lambda gnb_id, slice_id, n_rejected: calls.append((gnb_id, slice_id, n_rejected))
    env = RANEnv(cfg, source, seed=3)
    env.reset()
    pending = env.pending_requests()
    env.step([0] * len(pending))  # reject everything pending this step

    by_key = {}
    for gnb_id, slice_id, n in calls:
        by_key[(gnb_id, slice_id)] = by_key.get((gnb_id, slice_id), 0) + n
    expected = {}
    for req in pending:
        expected[(req.gnb_id, req.slice_id)] = expected.get((req.gnb_id, req.slice_id), 0) + 1
    assert by_key == expected


def test_episode_ends_after_configured_steps():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    cfg.episode.steps_per_episode = 3
    source = SyntheticKpmSource(seed=5, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=5)
    env.reset()
    done_flags = []
    for _ in range(3):
        pending = env.pending_requests()
        result = env.step([1] * len(pending))
        done_flags.append(result.done)
    assert done_flags == [False, False, True]


def test_seeded_env_is_reproducible():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))

    def run():
        source = SyntheticKpmSource(seed=42, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
        env = RANEnv(cfg, source, seed=42)
        obs0 = env.reset()
        pending = env.pending_requests()
        result = env.step([1] * len(pending))
        return obs0, result.obs, result.reward, len(pending)

    obs0_a, obs1_a, reward_a, n_a = run()
    obs0_b, obs1_b, reward_b, n_b = run()
    assert np.allclose(obs0_a, obs0_b)
    assert np.allclose(obs1_a, obs1_b)
    assert reward_a == reward_b
    assert n_a == n_b


def test_request_state_dim_and_encoding_shape():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_offline_dqn.yaml"))
    obs = np.zeros(state_dim(cfg), dtype=np.float32)
    req = AdmissionRequest("r1", "urllc", "gnb-1", arrival_step=1)
    full = encode_full_request_state(obs, req, cfg)
    assert full.shape == (request_state_dim(cfg),)
    # one-hot tail: 3 slices + 3 gnbs = 6 extra dims, urllc(slot0)/gnb-1(slot1) set
    tail = full[state_dim(cfg):]
    assert tail.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_env_over_replay_kpm_source(tmp_path):
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    trace_path = tmp_path / "trace.jsonl"
    rows = [
        {
            "ue_samples": [
                {
                    "rnti": 1, "timestamp_s": 0.0, "nssai_sst": 1, "nssai_sd": 1,
                    "avg_prbs_dl": 10.0, "gnb_id": "gnb-0",
                }
            ]
        },
        {
            "ue_samples": [
                {
                    "rnti": 2, "timestamp_s": 1.0, "nssai_sst": 1, "nssai_sd": 0,
                    "avg_prbs_dl": 20.0, "gnb_id": "gnb-0",
                }
            ]
        },
    ]
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    source = ReplayKpmSource(str(trace_path), loop=True)
    env = RANEnv(cfg, source, seed=1)
    env.reset()
    pending = env.pending_requests()
    assert any(r.slice_id == "urllc" for r in pending)  # real UE from row 0
    result = env.step([1] * len(pending))
    assert result.done is False
    assert len(source.sent_controls) >= 1


def test_sla_mode_reward_unaffected_by_passive_qoe_diagnostics():
    """reward_mode="sla" against a config WITH a qoe: section must produce
    the identical reward compute_step_reward alone would give (the frozen
    ratio-control baseline's actions/training signal must not change),
    while ALSO surfacing mean_mos/mos_by_slice/cost/sla_viol as read-only
    diagnostics -- see env.py's RANEnv.__init__/step() comments on why
    these are needed for the Stage One ablation to compare both arms on
    equal footing."""
    from qoe_oran_framework.reward import compute_step_reward

    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_offline_live1gnb.yaml"))
    assert cfg.qoe is not None  # this test is only meaningful if qoe: is configured

    source = SyntheticKpmSource(seed=3, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=3, reward_mode="sla")
    env.reset()
    pending = env.pending_requests()
    actions = [1] * len(pending)

    cluster_state_before = env.last_cluster_state
    result = env.step(actions)
    rb = result.info["reward_breakdown"]

    # Diagnostics present and plausible.
    assert 1.0 <= rb["mean_mos"] <= 5.0
    assert set(rb["mos_by_slice"].keys()) == set(cfg.slice_by_id.keys())
    assert rb["cost"] >= 0.0
    assert 0.0 <= rb["sla_viol"] <= 1.0
    assert "diagnostics" in rb["limitation"] or "NOT part of this run's actual reward" in rb["limitation"]

    # Reward itself matches a standalone compute_step_reward call on the
    # same pre-step cluster_state/accepted_counts -- i.e. bit-identical to
    # what this run would have produced with no qoe: section at all.
    expected_reward, _ = compute_step_reward(
        cluster_state_before, cfg.slice_by_id, result.info["accepted_counts"],
        cfg.reward, include_lb_term=env.include_lb_term,
    )
    assert result.reward == pytest.approx(expected_reward)


def test_step_logs_full_ceiling_snapshot_every_step():
    """info["ceilings"] must be the FULL current (gNB, slice) ceiling state
    every step, not just entries action_mapping.ApplyResult.ceilings
    touched this step (which omits slices with no pending request this
    step) -- this is what lets a live run plot the real slicing_control_m
    trajectory each policy commands, not just infer it from block counts."""
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_offline_live1gnb.yaml"))
    source = SyntheticKpmSource(seed=5, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=5, reward_mode="sla")
    env.reset()
    pending = env.pending_requests()

    result = env.step([1] * len(pending))  # accept everyone
    ceilings = result.info["ceilings"]
    expected_keys = {f"{gnb_id}:{slice_id}" for gnb_id in cfg.gnb_ids for slice_id in cfg.slice_by_id}
    assert set(ceilings.keys()) == expected_keys
    for key, c in ceilings.items():
        assert "min_ratio" in c and "max_ratio" in c
        slice_id = key.split(":")[1]
        spec = cfg.slice_by_id[slice_id]
        assert spec.min_ratio_floor <= c["max_ratio"] <= spec.max_ratio_cap

    # A slice with zero pending requests this step must still appear (full
    # snapshot, not just this-step deltas).
    pending2 = env.pending_requests()
    result2 = env.step([1] * len(pending2))
    assert set(result2.info["ceilings"].keys()) == expected_keys
