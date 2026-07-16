"""A2C admission policy: binary accept/reject over a per-request state.

Reuses oranslice_drl.drl_policy.A2CPolicy/ActorCriticNetwork directly, the
same way dqn_admission.py reuses DQNPolicy -- see that module's docstring.
Table I (papers #1/#2, A2C column): gamma=0.95, alpha=0.001, no replay
buffer / target network (both n/a for A2C per the papers' own table).
"""

from .._oranslice_path import ensure_oranslice_drl_importable

ensure_oranslice_drl_importable()

from oranslice_drl.drl_policy import A2CPolicy  # noqa: E402

PAPER_TABLE_I_A2C_DEFAULTS = dict(
    gamma=0.95,
    learning_rate=0.001,
)


class A2CAdmissionPolicy(A2CPolicy):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        device: str = "cpu",
        **overrides,
    ):
        params = {**PAPER_TABLE_I_A2C_DEFAULTS, **overrides}
        super().__init__(
            state_dim=state_dim,
            action_dim=2,
            n_branches=1,
            hidden_dim=hidden_dim,
            device=device,
            **params,
        )
