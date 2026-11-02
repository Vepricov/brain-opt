"""Distributed SGD with Markovian compression of client gradients.

Methods of the BRAIn Lab paper on Markovian compression (Veprikov,
Solodkin et al.): the ``BanLast(K, m)`` and ``KAWASAKI(K, b, pi, m)``
random-sparsification compressors whose coordinate choice is a Markov chain
over the last ``K`` steps, used inside Markovian QSGD (``MQSGD``,
Algorithm 1) and Accelerated Markovian QSGD (``AMQSGD``, Algorithm 2).
``QSGD`` with the memoryless ``Rand-m`` compressor is the known baseline.

Every client keeps its own compressor state. The network is simulated in
one process, as in :mod:`brain_opt.federated`; ``history`` records the
global loss and the number of coordinates sent per round.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .federated import (
    FederatedClient,
    FederatedResult,
    LossFn,
    _default_loss,
    _global_loss,
    _gradient_vector,
    _parameters_to_vector,
    _validate_clients,
    _vector_to_parameters,
)

COMPRESSORS = ("randm", "banlast", "kawasaki")


class RandSparsifier:
    """Random sparsification ``Q(x) = d/m * x * 1(nu)`` with a Markovian rule for ``nu``.

    ``kind="randm"`` draws ``nu`` uniformly and independently (Rand-m);
    ``kind="banlast"`` forbids coordinates sent in the last ``history`` steps;
    ``kind="kawasaki"`` divides the probability of a coordinate by
    ``forgetting ** count`` of its choices in the last ``history`` steps and
    normalizes by the L1 norm.
    """

    def __init__(self, dim: int, m: int, kind: str = "randm", history: int = 1,
                 forgetting: float = 2.0, generator: Optional[torch.Generator] = None) -> None:
        if kind not in COMPRESSORS:
            raise ValueError(f"compressor must be one of {COMPRESSORS}")
        if not 1 <= m <= dim:
            raise ValueError("m must be in [1, dim]")
        if history < 0:
            raise ValueError("history must be non-negative")
        if kind == "banlast" and dim < (history + 1) * m:
            raise ValueError("BanLast requires dim >= (history + 1) * m")
        if kind == "kawasaki" and forgetting <= 1.0:
            raise ValueError("KAWASAKI requires forgetting rate b > 1")
        self.dim, self.m, self.kind = dim, m, kind
        self.forgetting = forgetting
        self.generator = generator
        self.past: Deque[torch.Tensor] = deque(maxlen=max(history, 1) if kind != "randm" else 1)
        self.counts = torch.zeros(dim)
        self.history = history

    def choose(self) -> torch.Tensor:
        if self.kind == "randm" or self.history == 0:
            idx = torch.randperm(self.dim, generator=self.generator)[: self.m]
        else:
            if self.kind == "banlast":
                probs = (self.counts == 0).float()
            else:
                probs = self.forgetting ** (-self.counts)
            idx = torch.multinomial(probs / probs.sum(), self.m, replacement=False,
                                    generator=self.generator)
            if len(self.past) == self.past.maxlen:
                self.counts[self.past[0]] -= 1
            self.past.append(idx)
            self.counts[idx] += 1
        return idx

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.choose().to(x.device)
        out = torch.zeros_like(x)
        out[idx] = x[idx] * (self.dim / self.m)
        return out


@dataclass(frozen=True)
class CompressedConfig:
    """Configuration of compressed distributed SGD.

    ``ratio`` is the share ``m / d`` of coordinates a client sends per round.
    ``theta``, ``eta``, ``beta``, ``p`` are the momenta of AMQSGD; the
    defaults ``theta=0``, ``beta=0``, ``p=1`` reduce it to Nesterov-type
    extrapolation with coefficient ``eta - 1``.
    """

    rounds: int = 100
    lr: float = 0.1
    ratio: float = 0.1
    compressor: str = "kawasaki"
    history: int = 3
    forgetting: float = 2.0
    batch_size: Optional[int] = None
    theta: float = 0.0
    eta: float = 1.5
    beta: float = 0.0
    p: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 < self.ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        if self.compressor not in COMPRESSORS:
            raise ValueError(f"compressor must be one of {COMPRESSORS}")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < self.p <= 1.0:
            raise ValueError("p must be in (0, 1]")


def run_qsgd(model: nn.Module, clients: Sequence[FederatedClient],
             config: Optional[CompressedConfig] = None, loss_fn: Optional[LossFn] = None,
             ) -> FederatedResult:
    """QSGD (``compressor="randm"``) or Markovian QSGD (``"banlast"``, ``"kawasaki"``)."""
    return _run_compressed(model, clients, config or CompressedConfig(), loss_fn, accelerated=False)


def run_amqsgd(model: nn.Module, clients: Sequence[FederatedClient],
               config: Optional[CompressedConfig] = None, loss_fn: Optional[LossFn] = None,
               ) -> FederatedResult:
    """Accelerated Markovian QSGD (Algorithm 2 of the paper)."""
    return _run_compressed(model, clients, config or CompressedConfig(), loss_fn, accelerated=True)


def _run_compressed(model: nn.Module, clients: Sequence[FederatedClient], cfg: CompressedConfig,
                    loss_fn: Optional[LossFn], *, accelerated: bool) -> FederatedResult:
    _validate_clients(clients)
    loss = loss_fn or _default_loss
    gen = torch.Generator().manual_seed(cfg.seed)
    x = _parameters_to_vector(model)
    dim = x.numel()
    m = max(1, int(round(cfg.ratio * dim)))
    compressors = [RandSparsifier(dim, m, cfg.compressor, cfg.history, cfg.forgetting, gen)
                   for _ in clients]
    x_f = x.clone()
    probe = copy.deepcopy(model)
    total = float(sum(c.aggregation_weight for c in clients))
    history: List[Dict[str, Any]] = []

    for round_idx in range(cfg.rounds):
        x_g = cfg.theta * x_f + (1 - cfg.theta) * x if accelerated else x
        _vector_to_parameters(probe, x_g)
        g = torch.zeros_like(x)
        for client, compress in zip(clients, compressors):
            grad = _gradient_vector(probe, client, loss, cfg.batch_size, gen)
            g.add_(compress(grad), alpha=client.aggregation_weight / total)
        if accelerated:
            x_f_next = x_g - cfg.p * cfg.lr * g
            x = (cfg.eta * x_f_next + (cfg.p - cfg.eta) * x_f
                 + (1 - cfg.p) * (1 - cfg.beta) * x + (1 - cfg.p) * cfg.beta * x_g)
            x_f = x_f_next
        else:
            x = x - cfg.lr * g
        _vector_to_parameters(model, x_f if accelerated else x)
        history.append({
            "round": round_idx + 1,
            "coordinates_sent": m * len(clients),
            "train_loss": _global_loss(model, clients, loss),
        })
    return FederatedResult(model=model, history=history)
