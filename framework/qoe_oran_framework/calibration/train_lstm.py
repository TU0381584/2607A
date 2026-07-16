"""Trains QoEMapper (the LSTM temporal-refinement layer) per slice, against
the same objective calibration labels fit_iqx.py uses (P.1203 bridge for
eMBB, ACR-style scoring for URLLC/mMTC).

Unlike fit_iqx.py's i.i.d. condition sampling (appropriate for fitting a
memoryless closed-form curve), the LSTM's whole purpose is temporal
refinement, so training examples here are WINDOWS drawn from synthetic
per-slice TRAJECTORIES (a random walk through the (latency, loss,
throughput) condition space, in the same spirit as
replay_kpm_source.ClosedLoopKpmSource's own offered-demand random walk) --
this matches how the mapper is actually queried at inference time
(env.py::_compute_mos_by_slice pushes one step's features onto a rolling
window each call), rather than training on independent single points and
hoping the LSTM generalises to genuine temporal structure it never saw.

Each trajectory step's target is the OBJECTIVE label for that step's raw
(latency, loss, throughput) condition (P.1203/ACR, not the IQX prior) --
the LSTM is trained to predict the objective MOS directly from a window of
raw + IQX-prior features, and its own forward() applies the bounded-residual
correction on top of the prior internally (see qoe_mapper.py).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from ..qoe_mapper import IqxCoeffs, QoEMapper, build_qoe_features, iqx_mos
from .acr_scoring import acr_score_mmtc, acr_score_urllc
from .fit_iqx import LATENCY_S_RANGE_BY_SLICE, PACKET_LOSS_RANGE_BY_SLICE, THROUGHPUT_PRB_RANGE
from .units import prb_to_kbps
from .video_client_model import p1203_mos_from_throughput_trace

TRAIN_LSTM_LIMITATION = (
    "QoEMapper LSTM is trained on synthetic per-slice random-walk "
    "trajectories through the calibration condition space (not real KPM "
    "traces), with per-step objective labels from the same P.1203/ACR "
    "references fit_iqx.py uses -- inherits every modelling assumption "
    "documented there. See calibration/train_lstm.py module docstring."
)


@dataclass
class LstmTrainResult:
    slice_id: str
    mae_train: float
    mae_test: float
    pearson_r_test: float
    n_train_windows: int
    n_test_windows: int
    epochs: int


def _random_walk_trajectory(
    n_steps: int, slice_id: str, rng: np.random.RandomState, volatility: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-reverting random walk through (latency, loss, throughput),
    matching ClosedLoopKpmSource's own offered-demand dynamics in spirit --
    gives the LSTM genuine temporal correlation to learn from, unlike i.i.d.
    samples."""
    lat_lo, lat_hi = LATENCY_S_RANGE_BY_SLICE[slice_id]
    loss_lo, loss_hi = PACKET_LOSS_RANGE_BY_SLICE[slice_id]
    thr_lo, thr_hi = THROUGHPUT_PRB_RANGE

    def walk(lo, hi):
        mean = rng.uniform(lo, hi)
        x = np.zeros(n_steps)
        x[0] = np.clip(rng.uniform(lo, hi), lo, hi)
        span = hi - lo
        for t in range(1, n_steps):
            drift = 0.1 * (mean - x[t - 1])
            noise = rng.normal(0.0, volatility * span)
            x[t] = np.clip(x[t - 1] + drift + noise, lo, hi)
        return x

    return walk(lat_lo, lat_hi), walk(loss_lo, loss_hi), walk(thr_lo, thr_hi)


def _objective_labels(slice_id: str, latency_s: np.ndarray, packet_loss: np.ndarray, throughput_prb: np.ndarray) -> np.ndarray:
    if slice_id == "urllc":
        return acr_score_urllc(latency_s, packet_loss)
    if slice_id == "mmtc":
        return acr_score_mmtc(packet_loss, latency_s)
    if slice_id == "embb":
        throughput_kbps = prb_to_kbps(throughput_prb)
        mos = np.zeros(len(throughput_kbps))
        for i, kbps in enumerate(throughput_kbps):
            mos[i], _ = p1203_mos_from_throughput_trace([float(kbps)] * 3)  # short session per step, cheaper than 10 segs
        return mos
    raise ValueError(f"unknown slice_id {slice_id!r}")


def build_windowed_dataset(
    slice_id: str, coeffs: IqxCoeffs, n_trajectories: int, traj_len: int, window: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (X, y): X is [n_windows, window, 5] feature tensors, y is
    [n_windows] objective MOS targets, one per trajectory step (after the
    first `window-1` warm-up steps of each trajectory)."""
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    for traj_idx in range(n_trajectories):
        latency_s, packet_loss, throughput_prb = _random_walk_trajectory(traj_len, slice_id, rng)
        objective = _objective_labels(slice_id, latency_s, packet_loss, throughput_prb)
        prior = iqx_mos(latency_s, packet_loss, np.maximum(throughput_prb, 1e-3), coeffs)

        feat_history: List[np.ndarray] = []
        for t in range(traj_len):
            feat = build_qoe_features(
                latency_norm=latency_s[t], staleness=0, max_staleness=20,
                packet_loss=packet_loss[t], throughput_norm=throughput_prb[t],
                iqx_prior_mos=float(prior[t]),
            )
            feat_history.append(feat)
            if len(feat_history) >= window:
                X_list.append(np.stack(feat_history[-window:], axis=0))
                y_list.append(objective[t])
    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.float32)


def train_slice_mapper(
    slice_id: str, coeffs: IqxCoeffs, window: int = 8, hidden: int = 32,
    n_trajectories: int = 60, traj_len: int = 40, epochs: int = 60,
    lr: float = 1e-3, test_frac: float = 0.25, seed: int = 0,
) -> Tuple["QoEMapper", LstmTrainResult]:
    if torch is None:
        raise RuntimeError("torch is required to train QoEMapper")

    X, y = build_windowed_dataset(slice_id, coeffs, n_trajectories, traj_len, window, seed=seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_tr, y_tr = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.float32)
    X_te, y_te = torch.tensor(X[test_idx], dtype=torch.float32), torch.tensor(y[test_idx], dtype=torch.float32)

    torch.manual_seed(seed)
    model = QoEMapper(window=window, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    batch_size = 32
    n_train = X_tr.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
            opt.zero_grad()
            pred = model(X_tr[batch_idx])
            loss = loss_fn(pred, y_tr[batch_idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_tr = model(X_tr).numpy()
        pred_te = model(X_te).numpy()
    mae_tr = float(np.mean(np.abs(pred_tr - y_tr.numpy())))
    mae_te = float(np.mean(np.abs(pred_te - y_te.numpy())))
    if np.std(pred_te) > 1e-9 and np.std(y_te.numpy()) > 1e-9:
        r_te = float(np.corrcoef(pred_te, y_te.numpy())[0, 1])
    else:
        r_te = float("nan")

    result = LstmTrainResult(
        slice_id=slice_id, mae_train=mae_tr, mae_test=mae_te, pearson_r_test=r_te,
        n_train_windows=len(train_idx), n_test_windows=len(test_idx), epochs=epochs,
    )
    return model, result


def train_all_slices(iqx_coeffs_by_slice: Dict[str, IqxCoeffs], out_dir: str) -> Dict[str, LstmTrainResult]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for slice_id, coeffs in iqx_coeffs_by_slice.items():
        model, result = train_slice_mapper(slice_id, coeffs)
        results[slice_id] = result
        torch.save(model.state_dict(), os.path.join(out_dir, f"qoe_mapper_{slice_id}.pt"))
        print(f"{slice_id}: MAE(train)={result.mae_train:.4f} MAE(test)={result.mae_test:.4f} "
              f"pearson_r(test)={result.pearson_r_test:.4f} "
              f"(n_train={result.n_train_windows}, n_test={result.n_test_windows})")
    with open(os.path.join(out_dir, "lstm_train_results.json"), "w") as f:
        json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
    return results


if __name__ == "__main__":
    from ..qoe_mapper import DEFAULT_IQX_COEFFS
    import os

    fitted_path = os.path.join(os.path.dirname(__file__), "fitted_iqx_coeffs.json")
    if os.path.exists(fitted_path):
        with open(fitted_path) as f:
            fitted = json.load(f)
        coeffs_by_slice = {
            slice_id: IqxCoeffs(**fitted[slice_id]["coeffs"]) for slice_id in fitted
        }
        print("Using fitted IQX coefficients from fitted_iqx_coeffs.json")
    else:
        coeffs_by_slice = dict(DEFAULT_IQX_COEFFS)
        print("fitted_iqx_coeffs.json not found -- using DEFAULT_IQX_COEFFS priors "
              "(run fit_iqx.py first for calibrated coefficients)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "qoe_mapper")
    train_all_slices(coeffs_by_slice, out_dir)
