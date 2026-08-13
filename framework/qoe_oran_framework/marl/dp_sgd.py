"""DP-SGD-style per-client gradient clipping + calibrated Gaussian noise
(Abadi et al. 2016), applied to one client's local gradient in-place
immediately before its optimizer.step() -- the mechanism M3's brief asks
for ("DP-SGD-style per-client gradient clipping plus calibrated Gaussian
noise, with the noise multiplier as a swept knob"). Framework-agnostic:
takes any iterable of parameters with populated .grad, no dependency on
FederatedGatPolicy or any other class in this package.

No DP-accounting library is available in this environment (opacus is not
installed -- checked before writing this file). Rather than invent an
epsilon via an ad hoc formula, `zcdp_epsilon` below implements exactly one
well-established, closed-form bound: the Gaussian mechanism satisfies
rho-zCDP with rho = 1/(2*noise_multiplier**2) per release (Bun & Steinke
2016, consequence of their Gaussian-mechanism zCDP characterization),
composed additively over `steps` releases (basic composition, no
subsampling-privacy-amplification credited even though this project's
replay-buffer sampling would in principle earn some -- omitted because
claiming amplification correctly requires knowing the exact sampling
scheme's privacy amplification theorem, which is not implemented here),
then converted to (epsilon, delta)-DP via Bun & Steinke 2016 Proposition
1.3: epsilon(delta) = rho + 2*sqrt(rho * ln(1/delta)). This is a real,
citable, conservative (loose, not optimistic) upper bound on epsilon --
not a tight moments-accountant/RDP-with-subsampling figure a tool like
Opacus would produce. The privacy-utility analysis should treat the
noise multiplier itself as the primary, unambiguous reported knob, and
this epsilon as a secondary, explicitly-labelled conservative bound.
"""
import math
from typing import Iterable, Optional

import torch


def clip_and_noise_(
    parameters: Iterable[torch.nn.Parameter],
    clip_norm: float,
    noise_multiplier: float,
    generator: Optional[torch.Generator] = None,
) -> float:
    """In-place: clips the global grad-norm across `parameters` to
    `clip_norm`, then (if noise_multiplier > 0) adds i.i.d. Gaussian noise
    N(0, (noise_multiplier*clip_norm)^2) to every grad tensor -- the
    standard DP-SGD mechanism, sensitivity-normalized so noise_multiplier
    is the reusable, sensitivity-independent knob (Abadi et al. 2016,
    Algorithm 1). Returns the pre-clip global grad norm (for logging/
    diagnostics, not used in the mechanism itself). No-op (identity clip
    at clip_norm, no noise) when noise_multiplier == 0 and clip_norm is
    the same value already used for the non-DP arms' clip_grad_norm_ call
    -- so DP and non-DP arms differ ONLY in whether noise is added, not in
    clip behaviour, isolating the privacy cost cleanly.
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    total_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params))
    clip_coef = min(1.0, clip_norm / (float(total_norm) + 1e-6))
    for p in params:
        p.grad.detach().mul_(clip_coef)
        if noise_multiplier > 0:
            noise = torch.normal(
                mean=0.0, std=noise_multiplier * clip_norm, size=p.grad.shape,
                generator=generator, device=p.grad.device,
            )
            p.grad.detach().add_(noise)
    return float(total_norm)


def zcdp_epsilon(steps: int, noise_multiplier: float, delta: float) -> float:
    """Conservative (epsilon, delta)-DP upper bound for `steps` Gaussian-
    mechanism releases at the given noise_multiplier, via zCDP composition
    (Bun & Steinke 2016) -- see module docstring for the exact bound and
    its caveats. Returns float('inf') for noise_multiplier == 0 (no
    privacy) rather than raising, so a noise-multiplier sweep that
    includes a 0.0 (no-DP) control arm can call this uniformly."""
    if noise_multiplier <= 0:
        return float("inf")
    if steps <= 0:
        return 0.0
    rho = steps / (2.0 * noise_multiplier ** 2)
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
