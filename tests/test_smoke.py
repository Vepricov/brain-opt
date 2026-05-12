"""Smoke tests for brain_opt.

Every optimizer, given a single user-facing ``lr=1e-3``, should drive a
small MLP regression loss meaningfully down. Convergence speed varies by
family (Muon/SGD/Shampoo fast, Lion intentionally slow), so per-method
step budgets and reduction targets are calibrated to the *literature*
optimum, which is what :func:`brain_opt.scale_lr` maps ``1e-3`` to.
"""
import math

import pytest
import torch
import torch.nn as nn

import brain_opt
from brain_opt import (
    AdamW,
    Lion,
    Muon,
    SGD,
    SOAP,
    Shampoo,
    SignSGD,
    get_optimizer,
    scale_lr,
)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(16, 32)
        self.ln = nn.LayerNorm(32)
        self.l2 = nn.Linear(32, 16)
        self.act = nn.GELU()

    def forward(self, x):
        return self.l2(self.act(self.ln(self.l1(x))))


def _problem(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(128, 16, generator=g)
    W = torch.randn(16, 16, generator=g)
    Y = X @ W
    return X, Y


def _run(opt_factory, n_steps: int):
    torch.manual_seed(42)
    model = _MLP()
    X, Y = _problem()
    opt = opt_factory(model.parameters())
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses[0], losses[-1]


# ----------------------------------------------------------- scale_lr -------

def test_scale_lr_multipliers():
    """User-passed lr=1e-3 should map to each method's effective lr."""
    assert scale_lr("SGD",     1e-3) == pytest.approx(1e-2)
    assert scale_lr("AdamW",   1e-3) == pytest.approx(1e-3)
    assert scale_lr("Adam",    1e-3) == pytest.approx(1e-3)  # alias
    assert scale_lr("SignSGD", 1e-3) == pytest.approx(1e-3)
    assert scale_lr("Signum",  1e-3) == pytest.approx(1e-3)  # alias
    assert scale_lr("Lion",    1e-3) == pytest.approx(1e-3)
    assert scale_lr("Muon",    1e-3) == pytest.approx(1e-2)
    assert scale_lr("Shampoo", 1e-3) == pytest.approx(1e-1)
    assert scale_lr("SOAP",    1e-3) == pytest.approx(3e-3)


def test_scale_lr_case_insensitive():
    assert scale_lr("lion",    1e-3) == scale_lr("Lion",    1e-3)
    assert scale_lr("sign-sgd", 1e-3) == scale_lr("SignSGD", 1e-3)
    assert scale_lr("MUON",    1e-3) == scale_lr("Muon",    1e-3)


def test_scale_lr_unknown_raises():
    with pytest.raises(KeyError):
        scale_lr("Nope", 1e-3)


# ----------------------------------------------------- factory convergence --

# Every entry uses USER_LR=1e-3 via the factory.
# Step budget and acceptable end-ratio reflect each method's behaviour at
# its canonical lr on a small MLP regression task.
USER_LR = 1e-3

_CASES = [
    # name,      kwargs,                 n_steps,  max_end_ratio
    ("SGD",      dict(momentum=0.9),       500,    0.10),
    ("AdamW",    dict(),                   500,    0.20),
    ("SignSGD",  dict(momentum=0.9),       500,    0.20),
    ("Lion",     dict(),                   500,    0.10),
    ("Muon",     dict(adamw_lr_ratio=0.05), 500,   0.05),
    ("Shampoo",  dict(epsilon=1.0, update_freq=1), 500, 0.30),
    ("SOAP",     dict(weight_decay=0.0,
                      precondition_frequency=5,
                      precondition_1d=True), 500, 0.30),
]


@pytest.mark.parametrize("name,kwargs,n_steps,max_end_ratio", _CASES)
def test_converges_via_factory_unified_lr(name, kwargs, n_steps, max_end_ratio):
    """Each method, given USER_LR=1e-3 via the factory, makes meaningful
    progress within its step budget."""
    factory = lambda p: get_optimizer(p, name=name, lr=USER_LR, **kwargs)
    loss0, lossN = _run(factory, n_steps=n_steps)
    assert math.isfinite(lossN), f"{name} produced non-finite loss"
    ratio = lossN / loss0
    assert ratio <= max_end_ratio, (
        f"{name} at user_lr={USER_LR} ({n_steps} steps) only reached "
        f"{ratio:.3f}× initial loss (max allowed {max_end_ratio})"
    )


# ------------------------------------------------------ factory plumbing ----

def test_factory_resolves_all_names():
    model = _MLP()
    for name in [
        "SGD", "sgd",
        "AdamW", "adamw", "Adam", "adam",
        "SignSGD", "sign-sgd", "Signum",
        "Lion", "lion",
        "MUON", "muon",
        "Shampoo", "shampoo",
        "SOAP", "soap",
    ]:
        opt = get_optimizer(model.parameters(), name=name, lr=1e-3)
        opt.zero_grad()
        loss = ((model(torch.randn(8, 16)) - 0) ** 2).mean()
        loss.backward()
        opt.step()


def test_factory_unknown_raises():
    model = _MLP()
    with pytest.raises(KeyError):
        get_optimizer(model.parameters(), name="NotAnOptimizer", lr=1e-3)


def test_factory_auto_scale_can_be_disabled():
    """``auto_scale_lr=False`` should forward the raw lr."""
    model = _MLP()
    opt = get_optimizer(model.parameters(), name="Lion",
                        lr=5e-4, auto_scale_lr=False)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_factory_auto_scale_is_default():
    """Default factory call should apply the scale multiplier."""
    model = _MLP()
    opt = get_optimizer(model.parameters(), name="Muon", lr=1e-3)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-2)


# ----------------------------------------------------------- public API -----

def test_public_api():
    for sym in ["SGD", "AdamW", "SignSGD", "Lion", "Muon", "Shampoo",
                "SOAP", "get_optimizer", "scale_lr", "LR_MULTIPLIERS"]:
        assert hasattr(brain_opt, sym), f"brain_opt.{sym} missing"
    assert brain_opt.__version__
