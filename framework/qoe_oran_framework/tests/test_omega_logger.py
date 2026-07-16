import pytest

from qoe_oran_framework.omega_logger import OmegaLogger, OmegaTuple, read_omega_jsonl


def _tuple(**overrides):
    base = dict(
        role="urllc-admission", method="DQNAdmissionPolicy",
        objective="minimize URLLC block rate subject to SLA and PRB budget",
        constraint="gNB capacity B=100 PRB, URLLC quota<=30",
        evidence={"block_count": 1, "rho": 0.58},
        limitation="episode horizon reduced from 300 to 50 for live MC due to wall-clock cost",
        run_id="run-1", episode=1, step=1, timestamp_s=0.0, mode="offline_synthetic",
    )
    base.update(overrides)
    return OmegaTuple(**base)


def test_empty_limitation_is_rejected():
    with pytest.raises(ValueError):
        _tuple(limitation="")


def test_whitespace_only_limitation_is_rejected():
    with pytest.raises(ValueError):
        _tuple(limitation="   ")


def test_empty_evidence_is_rejected():
    with pytest.raises(ValueError):
        _tuple(evidence={})


def test_valid_tuple_constructs():
    tup = _tuple()
    assert tup.mode == "offline_synthetic"


def test_jsonl_round_trip(tmp_path):
    path = str(tmp_path / "omega.jsonl")
    logger = OmegaLogger(path)
    logger.log(_tuple(step=1))
    logger.log(_tuple(step=2, evidence={"block_count": 0, "rho": 0.6}))
    logger.close()

    rows = read_omega_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["step"] == 1
    assert rows[1]["evidence"]["block_count"] == 0
    assert all(row["limitation"] for row in rows)


def test_logger_appends_across_instances(tmp_path):
    path = str(tmp_path / "omega.jsonl")
    with OmegaLogger(path) as logger:
        logger.log(_tuple(step=1))
    with OmegaLogger(path) as logger:
        logger.log(_tuple(step=2))
    rows = read_omega_jsonl(path)
    assert len(rows) == 2


def test_logger_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "omega.jsonl")
    with OmegaLogger(path) as logger:
        logger.log(_tuple())
    assert read_omega_jsonl(path)
