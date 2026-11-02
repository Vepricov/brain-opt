"""SoftSignum and SoftMuon — smooth relaxations of sign-based optimizers.

Method of the BRAIn Lab (Veprikov, Feoktistov, Belinsky, Zainulin,
"Softsign: Smooth Sign in Your Optimizer For Better Parameter
Heterogeneity Handling"). This is a paper-based implementation.

SoftSignum replaces ``sign(m)`` of Signum by ``tanh(tau * m)``. Until the
transition point ``alpha_sign`` of training the temperature is infinite and
the update equals Signum. After it, ``tau`` is chosen so that the quantile
``q_p`` of the momentum magnitudes is mapped to ``1 - eps``:
``tau = artanh(1 - eps) / q_p``, clipped from below by 1, where ``p`` is the
transition progress in ``[0, 1]``. The magnitude distribution is
approximated once, at the transition point, by a folded Cauchy law with the
median and median absolute deviation of the momentum.

SoftMuon applies the algebraic soft sign ``x / sqrt(1 + x^2)`` to the
singular values of the Muon momentum:
``Phi_tau(M) = M (M^T M + tau^-2 I)^{-1/2}``; for ``tau -> inf`` it is the
Muon direction ``U V^T``. Temperatures are kept per parameter tensor.
"""
import math
from typing import Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

from .muon import Muon

_Params = Union[Iterable[torch.Tensor], Iterable[dict]]


def folded_cauchy_quantile(p: float, mu: float, sigma: float) -> float:
    """Quantile of ``|X|`` for ``X ~ Cauchy(mu, sigma)``, found by bisection."""
    if p <= 0.0:
        return 0.0
    p = min(p, 1.0 - 1e-6)
    sigma = max(sigma, 1e-12)

    def cdf(x: float) -> float:
        return (math.atan((x - mu) / sigma) + math.atan((x + mu) / sigma)) / math.pi

    lo, hi = 0.0, abs(mu) + sigma
    while cdf(hi) < p:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def robust_stats(values: torch.Tensor) -> Tuple[float, float]:
    """Median and median absolute deviation of a tensor (sampled if large)."""
    flat = values.detach().reshape(-1).float()
    if flat.numel() > 1_000_000:
        idx = torch.randint(flat.numel(), (1_000_000,), device=flat.device)
        flat = flat[idx]
    mu = flat.median()
    sigma = (flat - mu).abs().median()
    return float(mu), float(sigma)


def temperature(progress: float, alpha_sign: float, stats: Optional[Tuple[float, float]],
                saturation: float, inverse_at_saturation: float) -> float:
    """Temperature schedule of the paper (Algorithm "Temperature Schedule")."""
    if progress < alpha_sign or stats is None:
        return math.inf
    p = (progress - alpha_sign) / max(1.0 - alpha_sign, 1e-12)
    q = folded_cauchy_quantile(p, *stats)
    if q <= 0.0:
        return math.inf
    return max(1.0, inverse_at_saturation / q)


class SoftSignum(Optimizer):
    """Signum with a temperature-controlled ``tanh`` instead of ``sign``.

    Args:
        params: iterable of parameters or parameter group dicts.
        lr: learning rate.
        total_steps: number of optimizer steps ``N`` of the run.
        momentum: EMA coefficient ``beta`` of the momentum.
        weight_decay: decoupled weight-decay coefficient.
        alpha_sign: fraction of training spent in the pure Signum regime.
        saturation: tolerance ``eps``; ``tanh(tau * q_p) = 1 - eps``.
    """

    def __init__(
        self,
        params: _Params,
        lr: float = 1e-3,
        total_steps: int = 1000,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        alpha_sign: float = 0.5,
        saturation: float = 0.05,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= alpha_sign <= 1.0:
            raise ValueError("alpha_sign must be in [0, 1]")
        if not 0.0 < saturation < 1.0:
            raise ValueError("saturation must be in (0, 1)")
        defaults = dict(lr=lr, total_steps=total_steps, momentum=momentum,
                        weight_decay=weight_decay, alpha_sign=alpha_sign, saturation=saturation)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            wd = group["weight_decay"]
            alpha_sign = group["alpha_sign"]
            inverse = math.atanh(1.0 - group["saturation"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SoftSignum does not support sparse gradients")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                    state["stats"] = None
                buf = state["momentum_buffer"]
                buf.mul_(beta).add_(p.grad, alpha=1 - beta)
                progress = state["step"] / group["total_steps"]
                if progress >= alpha_sign and state["stats"] is None:
                    state["stats"] = robust_stats(buf)
                tau = temperature(progress, alpha_sign, state["stats"], group["saturation"], inverse)
                state["tau"] = tau
                state["step"] += 1

                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                update = buf.sign() if math.isinf(tau) else torch.tanh(buf * tau)
                p.data.add_(update, alpha=-lr)
        return loss


def soft_orthogonalize(m: torch.Tensor, tau: float) -> torch.Tensor:
    """``M (M^T M + tau^-2 I)^{-1/2}`` computed on the smaller Gram matrix."""
    x = m.float()
    transposed = x.size(0) < x.size(1)
    if transposed:
        x = x.T
    gram = x.T @ x
    evals, evecs = torch.linalg.eigh(gram)
    inv_sqrt = (evals.clamp_min(0.0) + tau ** -2).rsqrt()
    out = x @ (evecs * inv_sqrt) @ evecs.T
    return out.T if transposed else out


class SoftMuon(Muon):
    """Muon whose spectral sign is relaxed after ``alpha_sign`` of training.

    Accepts every :class:`brain_opt.Muon` argument plus ``total_steps``,
    ``alpha_sign`` and ``saturation`` (see :class:`SoftSignum`). Before the
    transition the update is the Newton-Schulz Muon direction; the singular
    values of each momentum matrix at the transition define its
    temperature schedule.
    """

    def __init__(self, params: _Params, lr: float = 0.02, total_steps: int = 1000,
                 alpha_sign: float = 0.5, saturation: float = 0.05, **muon_kwargs) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 <= alpha_sign <= 1.0:
            raise ValueError("alpha_sign must be in [0, 1]")
        if not 0.0 < saturation < 1.0:
            raise ValueError("saturation must be in (0, 1)")
        super().__init__(params, lr=lr, **muon_kwargs)
        self.total_steps = total_steps
        self.alpha_sign = alpha_sign
        self.saturation = saturation
        # algebraic soft sign phi(x) = x / sqrt(1 + x^2); phi^-1(y) = y / sqrt(1 - y^2)
        y = 1.0 - saturation
        self._inverse = y / math.sqrt(1.0 - y * y)

    def _direction(self, p: torch.Tensor, g_update: torch.Tensor, ns_steps: int) -> torch.Tensor:
        state = self.state[p]
        step = state.get("soft_step", 0)
        state["soft_step"] = step + 1
        progress = step / self.total_steps
        if progress >= self.alpha_sign and state.get("stats") is None:
            state["stats"] = robust_stats(torch.linalg.svdvals(g_update.float()))
        tau = temperature(progress, self.alpha_sign, state.get("stats"), self.saturation, self._inverse)
        state["tau"] = tau
        if math.isinf(tau):
            return super()._direction(p, g_update, ns_steps)
        return soft_orthogonalize(g_update, tau)
