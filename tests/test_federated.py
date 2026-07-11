import torch
import torch.nn as nn

import brain_opt
from brain_opt import (
    AsyncConfig,
    FedAvgConfig,
    FederatedClient,
    run_async_local_sgd,
    run_async_sgd,
    run_fedavg,
)


def _clients(n_clients=10, samples_per_client=24):
    gen = torch.Generator().manual_seed(7)
    true_w = torch.tensor([[1.5], [-2.0], [0.5]])
    true_b = torch.tensor([0.3])
    clients = []
    for i in range(n_clients):
        x = torch.randn(samples_per_client, 3, generator=gen) + 0.1 * i
        y = x @ true_w + true_b
        clients.append(FederatedClient(x=x, y=y, name=f"client-{i}"))
    return clients


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(3, 1)


def _loss(model, clients):
    with torch.no_grad():
        losses = [((model(c.x) - c.y) ** 2).mean().item() for c in clients]
    return sum(losses) / len(losses)


def test_fedavg_supports_different_client_counts():
    clients = _clients()
    small_model = _model()
    large_model = _model()
    small_initial = _loss(small_model, clients)
    large_initial = _loss(large_model, clients)

    small = run_fedavg(
        small_model,
        clients,
        FedAvgConfig(rounds=25, clients_per_round=2, local_steps=2, lr=0.05, seed=1),
    )
    large = run_fedavg(
        large_model,
        clients,
        FedAvgConfig(rounds=25, clients_per_round=8, local_steps=2, lr=0.05, seed=1),
    )

    assert len(small.history) == 25
    assert len(large.history) == 25
    assert small.history[-1]["train_loss"] < 0.2 * small_initial
    assert large.history[-1]["train_loss"] < 0.2 * large_initial


def test_async_sgd_tracks_staleness_and_reduces_loss():
    clients = _clients()
    model = _model()
    initial = _loss(model, clients)

    result = run_async_sgd(
        model,
        clients,
        AsyncConfig(
            updates=80,
            lr=0.03,
            server_lr=1.0,
            max_pending=6,
            min_delay=1,
            max_delay=4,
            seed=2,
        ),
    )

    assert len(result.history) == 80
    assert any(row["staleness"] > 0 for row in result.history)
    assert result.history[-1]["train_loss"] < 0.3 * initial


def test_async_local_sgd_tracks_staleness_and_reduces_loss():
    clients = _clients()
    model = _model()
    initial = _loss(model, clients)

    result = run_async_local_sgd(
        model,
        clients,
        AsyncConfig(
            updates=50,
            local_steps=2,
            lr=0.05,
            server_lr=1.0,
            max_pending=5,
            min_delay=1,
            max_delay=4,
            seed=3,
        ),
    )

    assert len(result.history) == 50
    assert any(row["staleness"] > 0 for row in result.history)
    assert result.history[-1]["train_loss"] < 0.3 * initial


def test_federated_public_api():
    for sym in [
        "FederatedClient",
        "FederatedResult",
        "FedAvgConfig",
        "AsyncConfig",
        "run_fedavg",
        "run_async_sgd",
        "run_async_local_sgd",
    ]:
        assert hasattr(brain_opt, sym), f"brain_opt.{sym} missing"
