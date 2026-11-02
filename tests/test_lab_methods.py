"""Checks of the BRAIn Lab methods: SoftSignum, SoftMuon, Markovian compression, WeightLoRA.

Local algorithm checks on toy problems, not quality benchmarks.
"""

import json
import math

import pytest
import torch
from torch import nn

import brain_opt
from brain_opt.compression import CompressedConfig, RandSparsifier, run_amqsgd, run_qsgd
from brain_opt.federated import FederatedClient
from brain_opt.softsign import (SoftMuon, SoftSignum, folded_cauchy_quantile,
                                soft_orthogonalize, temperature)
from brain_opt.training import TrainingConfig, run_training
from brain_opt.weight_lora import WeightLoRA, WeightLoRAConfig, hard_threshold
from tests.test_training import local_inputs  # noqa: F401  (pytest fixture)


def _regression(seed: int = 0, n: int = 64, d: int = 8):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=gen)
    w = torch.randn(d, 1, generator=gen)
    return x, x @ w


def _train(opt_factory, steps: int = 200) -> float:
    torch.manual_seed(0)
    x, y = _regression()
    model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
    opt = opt_factory(model.parameters())
    first = None
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        first = first if first is not None else float(loss)
        loss.backward()
        opt.step()
    return float(nn.functional.mse_loss(model(x), y)) / first


def test_folded_cauchy_quantile_matches_cdf():
    for mu, sigma in ((0.0, 1.0), (0.3, 0.05), (2.0, 0.5)):
        for p in (0.1, 0.5, 0.9):
            q = folded_cauchy_quantile(p, mu, sigma)
            cdf = (math.atan((q - mu) / sigma) + math.atan((q + mu) / sigma)) / math.pi
            assert cdf == pytest.approx(p, abs=1e-6)
    assert folded_cauchy_quantile(0.0, 1.0, 1.0) == 0.0


def test_temperature_schedule_is_infinite_then_decreasing():
    inverse = math.atanh(0.95)
    assert math.isinf(temperature(0.2, 0.5, None, 0.05, inverse))
    stats = (0.0, 0.01)
    taus = [temperature(a, 0.5, stats, 0.05, inverse) for a in (0.55, 0.7, 0.9, 0.99)]
    assert all(a >= b for a, b in zip(taus, taus[1:]))
    assert taus[-1] >= 1.0


def test_softsignum_equals_signum_before_transition():
    torch.manual_seed(1)
    a = nn.Linear(4, 3)
    b = nn.Linear(4, 3)
    b.load_state_dict(a.state_dict())
    soft = SoftSignum(a.parameters(), lr=0.01, total_steps=100, momentum=0.9, alpha_sign=0.5)
    buffers = [torch.zeros_like(p) for p in b.parameters()]
    x = torch.randn(5, 4)
    for _ in range(10):
        soft.zero_grad()
        a(x).pow(2).sum().backward()
        soft.step()
        b.zero_grad()
        b(x).pow(2).sum().backward()
        with torch.no_grad():  # Signum with the EMA momentum of the paper
            for p, m in zip(b.parameters(), buffers):
                m.mul_(0.9).add_(p.grad, alpha=0.1)
                p.add_(m.sign(), alpha=-0.01)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q)


def test_softsignum_leaves_sign_regime_and_trains():
    ratio = _train(lambda ps: SoftSignum(ps, lr=0.01, total_steps=200, alpha_sign=0.3))
    assert ratio < 0.2
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    opt = SoftSignum(model.parameters(), lr=0.01, total_steps=10, alpha_sign=0.5)
    for _ in range(10):
        opt.zero_grad()
        model(torch.randn(4, 3)).sum().backward()
        opt.step()
    assert all(math.isfinite(opt.state[p]["tau"]) for p in model.parameters())


def test_soft_orthogonalize_limits():
    torch.manual_seed(2)
    m = torch.randn(6, 4)
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    torch.testing.assert_close(soft_orthogonalize(m, 1e6), u @ vh, atol=1e-4, rtol=1e-4)
    tau = 3.0
    expected = u @ torch.diag(tau * s / torch.sqrt(1 + tau ** 2 * s ** 2)) @ vh
    torch.testing.assert_close(soft_orthogonalize(m, tau), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(soft_orthogonalize(m.T, tau), expected.T, atol=1e-5, rtol=1e-5)


def test_softmuon_trains_and_relaxes():
    ratio = _train(lambda ps: SoftMuon(ps, lr=0.02, total_steps=200, alpha_sign=0.3))
    assert ratio < 0.2


@pytest.mark.parametrize("name", ["SoftSignum", "SoftMuon"])
def test_factory_knows_lab_optimizers(name):
    model = nn.Linear(4, 4)
    opt = brain_opt.get_optimizer(model.parameters(), name=name, lr=1e-3, total_steps=10)
    assert type(opt).__name__ == name


def test_banlast_never_repeats_inside_window():
    gen = torch.Generator().manual_seed(3)
    comp = RandSparsifier(dim=20, m=4, kind="banlast", history=3, generator=gen)
    recent = []
    for _ in range(30):
        idx = set(comp.choose().tolist())
        assert len(idx) == 4
        for past in recent[-3:]:
            assert not idx & past
        recent.append(idx)


def test_kawasaki_prefers_fresh_coordinates_and_is_scaled():
    gen = torch.Generator().manual_seed(4)
    comp = RandSparsifier(dim=10, m=2, kind="kawasaki", history=2, forgetting=50.0, generator=gen)
    x = torch.arange(1.0, 11.0)
    out = comp(x)
    sent = out.nonzero().flatten()
    assert len(sent) == 2
    torch.testing.assert_close(out[sent], x[sent] * 5.0)
    repeats = 0
    previous = set(sent.tolist())
    for _ in range(50):
        now = set(comp.choose().tolist())
        repeats += bool(now & previous)
        previous = now
    assert repeats < 10


@pytest.mark.parametrize("kwargs", [
    {"dim": 10, "m": 0}, {"dim": 10, "m": 4, "kind": "banlast", "history": 2},
    {"dim": 10, "m": 2, "kind": "kawasaki", "forgetting": 1.0}, {"dim": 10, "m": 2, "kind": "topk"},
])
def test_invalid_compressor(kwargs):
    with pytest.raises(ValueError):
        RandSparsifier(**kwargs)


def _clients():
    x, y = _regression(n=80)
    return [FederatedClient(x[i::4], y[i::4], name=f"c{i}") for i in range(4)]


@pytest.mark.parametrize("compressor", ["randm", "banlast", "kawasaki"])
def test_mqsgd_reduces_loss(compressor):
    torch.manual_seed(0)
    model = nn.Linear(8, 1)
    cfg = CompressedConfig(rounds=300, lr=0.05, ratio=0.3, compressor=compressor, history=2)
    result = run_qsgd(model, _clients(), cfg)
    assert result.history[-1]["train_loss"] < 0.1 * result.history[0]["train_loss"]
    assert result.history[0]["coordinates_sent"] == 4 * round(0.3 * 9)


def test_amqsgd_reduces_loss():
    torch.manual_seed(0)
    model = nn.Linear(8, 1)
    cfg = CompressedConfig(rounds=300, lr=0.05, ratio=0.3, compressor="kawasaki", history=2)
    result = run_amqsgd(model, _clients(), cfg)
    assert result.history[-1]["train_loss"] < 0.1 * result.history[0]["train_loss"]


def test_hard_threshold():
    torch.testing.assert_close(hard_threshold(torch.tensor([0.1, -3.0, 2.0, 0.5]), 2),
                               torch.tensor([0.0, -3.0, 2.0, 0.0]))


@pytest.mark.parametrize("kwargs", [
    {"keep": 0}, {"selection_steps": 1000}, {"lr": float("nan")}, {"omega_lr": 0},
])
def test_invalid_weightlora_config(kwargs):
    with pytest.raises(ValueError):
        WeightLoRAConfig(**kwargs)


def test_weightlora_keeps_k_adapters(local_inputs, tmp_path):  # noqa: F811
    pytest.importorskip("peft")
    base, train, val, _ = local_inputs
    output = tmp_path / "result"
    selection = WeightLoRAConfig(total_steps=6, keep=1, selection_steps=3, lr=0.005)
    config = TrainingConfig(mode="weightlora", steps=6, rank=2, lora_alpha=4, seq_len=8,
                            train_blocks=8, val_blocks=3, batch_size=2, lr=0.005,
                            weight_lora=selection)
    result = run_training(str(base), str(train), str(val), str(output), config)
    assert result["status"] == "COMPLETED" and result["reload_verified"] is True
    assert result["final_trainable_parameters"] < result["initial_trainable_parameters"]
    evidence = json.loads((output / "checkpoint" / "weightlora_selection.json").read_text())
    assert len(evidence["final"]["adapters"]) == 1
    adapter = json.loads((output / "checkpoint" / "adapter_config.json").read_text())
    assert len(adapter["target_modules"]) == 1


@pytest.mark.parametrize("optimizer", ["SoftSignum", "SoftMuon"])
def test_lm_training_with_lab_optimizers(local_inputs, tmp_path, optimizer):  # noqa: F811
    pytest.importorskip("peft")
    base, train, val, _ = local_inputs
    config = TrainingConfig(mode="lora", optimizer=optimizer, steps=6, rank=2, lora_alpha=4,
                            seq_len=8, train_blocks=8, val_blocks=3, batch_size=2, lr=0.005)
    result = run_training(str(base), str(train), str(val), str(tmp_path / "out"), config)
    assert result["status"] == "COMPLETED" and result["reload_verified"] is True
