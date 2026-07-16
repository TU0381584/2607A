import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qoe_oran_framework.policies.a2c_admission import A2CAdmissionPolicy
from qoe_oran_framework.policies.dqn_admission import DQNAdmissionPolicy
from qoe_oran_framework.policies.rainbow_admission import (
    PrioritizedReplayBuffer,
    RainbowAdmissionPolicy,
)

STATE_DIM = 10


def _fixed_batch(batch_size=16, state_dim=STATE_DIM, seed=0, with_weights=False):
    rng = np.random.RandomState(seed)
    states = rng.randn(batch_size, state_dim).astype(np.float32)
    next_states = rng.randn(batch_size, state_dim).astype(np.float32)
    actions = rng.randint(0, 2, size=batch_size).astype(np.int64)
    rewards = rng.randn(batch_size).astype(np.float32)
    dones = np.zeros(batch_size, dtype=np.float32)
    batch = {
        "states": states, "actions": actions, "rewards": rewards,
        "next_states": next_states, "dones": dones,
        "action_info": np.full(batch_size, np.nan, dtype=np.float32),
    }
    if with_weights:
        batch["weights"] = np.ones(batch_size, dtype=np.float32)
    return batch


# --- DQN ---

def test_dqn_admission_select_action_returns_binary_action():
    policy = DQNAdmissionPolicy(state_dim=STATE_DIM)
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action, q_value = policy.select_action(state, training=False)
    assert action in (0, 1)
    assert isinstance(q_value, float)


def test_dqn_admission_paper_hyperparameters_applied():
    policy = DQNAdmissionPolicy(state_dim=STATE_DIM)
    assert policy.gamma == 0.95
    assert policy.learning_rate == 0.001
    assert policy.action_dim == 2
    assert policy.n_branches == 1


def test_dqn_admission_train_step_reduces_loss_on_fixed_batch():
    policy = DQNAdmissionPolicy(state_dim=STATE_DIM, epsilon_decay=1.0)
    batch = _fixed_batch()
    losses = [policy.train_step(batch)["loss"] for _ in range(50)]
    assert losses[-1] < losses[0]


def test_dqn_admission_on_episode_end_syncs_target():
    policy = DQNAdmissionPolicy(state_dim=STATE_DIM)
    batch = _fixed_batch()
    policy.train_step(batch)
    policy.on_episode_end()
    q_params = list(policy.q_network.parameters())
    t_params = list(policy.target_network.parameters())
    for qp, tp in zip(q_params, t_params):
        assert torch.allclose(qp, tp)


def test_dqn_admission_checkpoint_roundtrip(tmp_path):
    policy = DQNAdmissionPolicy(state_dim=STATE_DIM)
    path = str(tmp_path / "dqn.pt")
    policy.save_checkpoint(path)
    policy2 = DQNAdmissionPolicy(state_dim=STATE_DIM)
    policy2.load_checkpoint(path)
    for p1, p2 in zip(policy.q_network.parameters(), policy2.q_network.parameters()):
        assert torch.allclose(p1, p2)


# --- A2C ---

def test_a2c_admission_select_action_returns_binary_action():
    policy = A2CAdmissionPolicy(state_dim=STATE_DIM)
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action, value = policy.select_action(state, training=True)
    assert action in (0, 1)


def test_a2c_admission_paper_hyperparameters_applied():
    policy = A2CAdmissionPolicy(state_dim=STATE_DIM)
    assert policy.gamma == 0.95
    assert policy.learning_rate == 0.001


def test_a2c_admission_train_step_runs_and_returns_metrics():
    policy = A2CAdmissionPolicy(state_dim=STATE_DIM)
    batch = _fixed_batch()
    metrics = policy.train_step(batch)
    assert "loss" in metrics and "entropy" in metrics


def test_a2c_admission_checkpoint_roundtrip(tmp_path):
    policy = A2CAdmissionPolicy(state_dim=STATE_DIM)
    path = str(tmp_path / "a2c.pt")
    policy.save_checkpoint(path)
    policy2 = A2CAdmissionPolicy(state_dim=STATE_DIM)
    policy2.load_checkpoint(path)
    for p1, p2 in zip(policy.network.parameters(), policy2.network.parameters()):
        assert torch.allclose(p1, p2)


# --- Rainbow ---

def test_rainbow_admission_select_action_returns_binary_action():
    policy = RainbowAdmissionPolicy(state_dim=STATE_DIM)
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action, q_value = policy.select_action(state, training=False)
    assert action in (0, 1)


def test_rainbow_admission_paper_hyperparameters_applied():
    policy = RainbowAdmissionPolicy(state_dim=STATE_DIM)
    assert policy.gamma == 0.95
    assert policy.per_alpha == 0.6
    assert policy.per_beta == 0.4


def test_rainbow_admission_train_step_reduces_loss_on_fixed_batch():
    policy = RainbowAdmissionPolicy(state_dim=STATE_DIM)
    batch = _fixed_batch(with_weights=True)
    losses = [policy.train_step(batch)["loss"] for _ in range(50)]
    assert losses[-1] < losses[0]


def test_rainbow_admission_on_episode_end_syncs_target():
    policy = RainbowAdmissionPolicy(state_dim=STATE_DIM)
    batch = _fixed_batch(with_weights=True)
    policy.train_step(batch)
    policy.on_episode_end()
    for qp, tp in zip(policy.q_network.parameters(), policy.target_network.parameters()):
        assert torch.allclose(qp, tp)


def test_rainbow_admission_anneal_beta_moves_toward_one():
    policy = RainbowAdmissionPolicy(state_dim=STATE_DIM)
    policy.anneal_beta(0.5)
    assert 0.4 < policy.per_beta < 1.0
    policy.per_beta = 0.4
    policy.anneal_beta(1.0)
    assert policy.per_beta == pytest.approx(1.0)


def test_rainbow_admission_checkpoint_roundtrip(tmp_path):
    policy = RainbowAdmissionPolicy(state_dim=STATE_DIM)
    path = str(tmp_path / "rainbow.pt")
    policy.save_checkpoint(path)
    policy2 = RainbowAdmissionPolicy(state_dim=STATE_DIM)
    policy2.load_checkpoint(path)
    for p1, p2 in zip(policy.q_network.parameters(), policy2.q_network.parameters()):
        assert torch.allclose(p1, p2)


# --- Prioritized replay buffer ---

def test_per_add_and_sample_shapes():
    buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
    for i in range(20):
        buf.add(np.zeros(STATE_DIM), i % 2, float(i), np.ones(STATE_DIM), False)
    batch = buf.sample(8, beta=0.4)
    assert batch["states"].shape == (8, STATE_DIM)
    assert batch["actions"].shape == (8,)
    assert len(batch["indices"]) == 8


def test_per_sampling_favors_high_priority_transitions():
    buf = PrioritizedReplayBuffer(capacity=100, alpha=1.0)
    for i in range(10):
        buf.add(np.full(STATE_DIM, i), 0, 0.0, np.zeros(STATE_DIM), False)
    # crank one transition's priority far above the rest
    buf.priorities[3] = 1000.0
    counts = np.zeros(10)
    rng_state = np.random.get_state()
    np.random.seed(0)
    for _ in range(200):
        batch = buf.sample(1, beta=0.4)
        counts[batch["indices"][0]] += 1
    np.random.set_state(rng_state)
    assert counts[3] > counts.sum() * 0.5  # dominates sampling


def test_per_update_priorities_changes_future_sampling():
    buf = PrioritizedReplayBuffer(capacity=100, alpha=1.0)
    for i in range(5):
        buf.add(np.zeros(STATE_DIM), 0, 0.0, np.zeros(STATE_DIM), False)
    buf.update_priorities(np.array([2]), np.array([50.0]))
    assert buf.priorities[2] > buf.priorities[0]


def test_per_raises_when_insufficient_transitions():
    buf = PrioritizedReplayBuffer(capacity=100)
    buf.add(np.zeros(STATE_DIM), 0, 0.0, np.zeros(STATE_DIM), False)
    with pytest.raises(ValueError):
        buf.sample(5)
