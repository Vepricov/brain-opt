"""Federated optimization simulators.

The module intentionally stays single-process and dependency-light. It is
meant for algorithm studies and platform demos where the network is
simulated by delayed client updates rather than implemented through RPC or
``torch.distributed``.
"""
from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FederatedClient:
    """One simulated client with a local supervised dataset."""

    x: torch.Tensor
    y: torch.Tensor
    weight: Optional[float] = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("client x and y must have the same first dimension")
        if self.x.shape[0] == 0:
            raise ValueError("client dataset must be non-empty")
        if self.weight is not None and self.weight <= 0:
            raise ValueError("client weight must be positive")

    @property
    def num_samples(self) -> int:
        return int(self.x.shape[0])

    @property
    def aggregation_weight(self) -> float:
        return float(self.weight if self.weight is not None else self.num_samples)


@dataclass(frozen=True)
class FedAvgConfig:
    """Configuration for synchronous FedAvg."""

    rounds: int = 10
    clients_per_round: Optional[int] = None
    local_steps: int = 1
    lr: float = 0.1
    batch_size: Optional[int] = None
    seed: int = 0


@dataclass(frozen=True)
class AsyncConfig:
    """Configuration for asynchronous SGD-style simulations."""

    updates: int = 100
    local_steps: int = 1
    lr: float = 0.1
    server_lr: float = 1.0
    max_pending: int = 8
    min_delay: int = 1
    max_delay: int = 5
    staleness_power: float = 1.0
    batch_size: Optional[int] = None
    seed: int = 0


@dataclass
class FederatedResult:
    """Result returned by federated simulators."""

    model: nn.Module
    history: List[Dict[str, Any]]


def run_fedavg(
    model: nn.Module,
    clients: Sequence[FederatedClient],
    config: Optional[FedAvgConfig] = None,
    loss_fn: Optional[LossFn] = None,
) -> FederatedResult:
    """Run synchronous FedAvg on a list of simulated clients.

    Each round samples ``clients_per_round`` clients, trains local copies
    for ``local_steps`` SGD steps, and replaces the server parameters by a
    sample-count weighted average of local parameters.
    """
    cfg = config or FedAvgConfig()
    _validate_clients(clients)
    _validate_fedavg_config(cfg, len(clients))
    loss = loss_fn or _default_loss
    gen = torch.Generator().manual_seed(cfg.seed)
    history: List[Dict[str, Any]] = []

    for round_idx in range(cfg.rounds):
        ids = _sample_clients(len(clients), cfg.clients_per_round, gen)
        local_vectors = []
        weights = []
        for client_id in ids:
            client = clients[client_id]
            local_model = _local_sgd(
                model,
                client,
                loss,
                lr=cfg.lr,
                steps=cfg.local_steps,
                batch_size=cfg.batch_size,
                generator=gen,
            )
            local_vectors.append(_parameters_to_vector(local_model))
            weights.append(client.aggregation_weight)

        new_vector = _weighted_average(local_vectors, weights)
        _vector_to_parameters(model, new_vector)
        history.append({
            "round": round_idx + 1,
            "clients": ids,
            "train_loss": _global_loss(model, clients, loss),
        })

    return FederatedResult(model=model, history=history)


def run_async_sgd(
    model: nn.Module,
    clients: Sequence[FederatedClient],
    config: Optional[AsyncConfig] = None,
    loss_fn: Optional[LossFn] = None,
) -> FederatedResult:
    """Run asynchronous server-side SGD with delayed client gradients."""
    cfg = config or AsyncConfig(local_steps=1)
    return _run_async(model, clients, cfg, loss_fn, mode="gradient")


def run_async_local_sgd(
    model: nn.Module,
    clients: Sequence[FederatedClient],
    config: Optional[AsyncConfig] = None,
    loss_fn: Optional[LossFn] = None,
) -> FederatedResult:
    """Run asynchronous local SGD with delayed client deltas."""
    cfg = config or AsyncConfig(local_steps=3)
    if cfg.local_steps <= 0:
        raise ValueError("Async-Local SGD requires local_steps > 0")
    return _run_async(model, clients, cfg, loss_fn, mode="local")


def _run_async(
    model: nn.Module,
    clients: Sequence[FederatedClient],
    cfg: AsyncConfig,
    loss_fn: Optional[LossFn],
    *,
    mode: str,
) -> FederatedResult:
    _validate_clients(clients)
    _validate_async_config(cfg, len(clients))
    loss = loss_fn or _default_loss
    gen = torch.Generator().manual_seed(cfg.seed)
    pending: List[Tuple[int, int, Dict[str, Any]]] = []
    history: List[Dict[str, Any]] = []
    clock = 0
    scheduled = 0
    applied = 0
    server_version = 0

    while applied < cfg.updates:
        while scheduled < cfg.updates and len(pending) < cfg.max_pending:
            client_id = _random_client(len(clients), gen)
            snapshot = _parameters_to_vector(model)
            update = _client_update(
                model,
                snapshot,
                clients[client_id],
                loss,
                cfg,
                gen,
                mode=mode,
            )
            delay = _random_delay(cfg, gen)
            heapq.heappush(pending, (
                clock + delay,
                scheduled,
                {
                    "client_id": client_id,
                    "delta": update,
                    "server_version": server_version,
                },
            ))
            scheduled += 1

        finish_time, _, event = heapq.heappop(pending)
        clock = finish_time
        staleness = max(0, server_version - int(event["server_version"]))
        staleness_weight = 1.0 / ((1.0 + staleness) ** cfg.staleness_power)
        current = _parameters_to_vector(model)
        current = current + cfg.server_lr * staleness_weight * event["delta"]
        _vector_to_parameters(model, current)
        applied += 1
        server_version += 1
        history.append({
            "update": applied,
            "client": event["client_id"],
            "time": clock,
            "staleness": staleness,
            "staleness_weight": staleness_weight,
            "train_loss": _global_loss(model, clients, loss),
        })

    return FederatedResult(model=model, history=history)


def _client_update(
    server_model: nn.Module,
    snapshot: torch.Tensor,
    client: FederatedClient,
    loss_fn: LossFn,
    cfg: AsyncConfig,
    generator: torch.Generator,
    *,
    mode: str,
) -> torch.Tensor:
    local_model = copy.deepcopy(server_model)
    _vector_to_parameters(local_model, snapshot)
    if mode == "gradient":
        grad = _gradient_vector(local_model, client, loss_fn, cfg.batch_size, generator)
        return -cfg.lr * grad
    if mode == "local":
        trained = _local_sgd(
            local_model,
            client,
            loss_fn,
            lr=cfg.lr,
            steps=cfg.local_steps,
            batch_size=cfg.batch_size,
            generator=generator,
        )
        return _parameters_to_vector(trained) - snapshot
    raise ValueError(f"unknown async mode: {mode}")


def _local_sgd(
    model: nn.Module,
    client: FederatedClient,
    loss_fn: LossFn,
    *,
    lr: float,
    steps: int,
    batch_size: Optional[int],
    generator: torch.Generator,
) -> nn.Module:
    local_model = copy.deepcopy(model)
    opt = torch.optim.SGD(local_model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = _batch(client, batch_size, generator, _model_device(local_model))
        opt.zero_grad()
        loss = loss_fn(local_model(x), y)
        loss.backward()
        opt.step()
    return local_model


def _gradient_vector(
    model: nn.Module,
    client: FederatedClient,
    loss_fn: LossFn,
    batch_size: Optional[int],
    generator: torch.Generator,
) -> torch.Tensor:
    x, y = _batch(client, batch_size, generator, _model_device(model))
    model.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    pieces = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            pieces.append(torch.zeros_like(p).reshape(-1))
        else:
            pieces.append(p.grad.detach().reshape(-1))
    return torch.cat(pieces)


def _parameters_to_vector(model: nn.Module) -> torch.Tensor:
    pieces = [p.detach().reshape(-1) for p in model.parameters() if p.requires_grad]
    if not pieces:
        raise ValueError("model has no trainable parameters")
    return torch.cat(pieces).clone()


def _vector_to_parameters(model: nn.Module, vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(vector[offset:offset + n].view_as(p))
            offset += n
    if offset != vector.numel():
        raise ValueError("parameter vector has the wrong size")


def _batch(
    client: FederatedClient,
    batch_size: Optional[int],
    generator: torch.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = client.x.to(device)
    y = client.y.to(device)
    n = client.num_samples
    if batch_size is None or batch_size >= n:
        return x, y
    idx = torch.randint(n, (batch_size,), generator=generator)
    return x[idx.to(x.device)], y[idx.to(y.device)]


def _global_loss(
    model: nn.Module,
    clients: Sequence[FederatedClient],
    loss_fn: LossFn,
) -> float:
    model.eval()
    total = 0.0
    weight_sum = 0.0
    device = _model_device(model)
    with torch.no_grad():
        for client in clients:
            x = client.x.to(device)
            y = client.y.to(device)
            weight = client.aggregation_weight
            total += weight * float(loss_fn(model(x), y).item())
            weight_sum += weight
    model.train()
    return total / weight_sum


def _weighted_average(vectors: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("sum of aggregation weights must be positive")
    out = torch.zeros_like(vectors[0])
    for vector, weight in zip(vectors, weights):
        out.add_(vector, alpha=float(weight) / total)
    return out


def _sample_clients(
    num_clients: int,
    clients_per_round: Optional[int],
    generator: torch.Generator,
) -> List[int]:
    count = num_clients if clients_per_round is None else clients_per_round
    if count >= num_clients:
        return list(range(num_clients))
    return torch.randperm(num_clients, generator=generator)[:count].tolist()


def _random_client(num_clients: int, generator: torch.Generator) -> int:
    return int(torch.randint(num_clients, (1,), generator=generator).item())


def _random_delay(cfg: AsyncConfig, generator: torch.Generator) -> int:
    return int(torch.randint(cfg.min_delay, cfg.max_delay + 1, (1,), generator=generator).item())


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _default_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def _validate_clients(clients: Sequence[FederatedClient]) -> None:
    if not clients:
        raise ValueError("at least one client is required")


def _validate_fedavg_config(cfg: FedAvgConfig, num_clients: int) -> None:
    if cfg.rounds <= 0:
        raise ValueError("rounds must be positive")
    if cfg.local_steps <= 0:
        raise ValueError("local_steps must be positive")
    if cfg.lr <= 0:
        raise ValueError("lr must be positive")
    if cfg.clients_per_round is not None and not 1 <= cfg.clients_per_round <= num_clients:
        raise ValueError("clients_per_round must be in [1, num_clients]")
    if cfg.batch_size is not None and cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _validate_async_config(cfg: AsyncConfig, num_clients: int) -> None:
    if cfg.updates <= 0:
        raise ValueError("updates must be positive")
    if cfg.lr <= 0:
        raise ValueError("lr must be positive")
    if cfg.server_lr <= 0:
        raise ValueError("server_lr must be positive")
    if cfg.max_pending <= 0:
        raise ValueError("max_pending must be positive")
    if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
        raise ValueError("delay range must satisfy 0 <= min_delay <= max_delay")
    if cfg.staleness_power < 0:
        raise ValueError("staleness_power must be non-negative")
    if cfg.batch_size is not None and cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_clients <= 0:
        raise ValueError("at least one client is required")
