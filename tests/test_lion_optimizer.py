import importlib.util
import os
from pathlib import Path
import unittest

import torch


OPTIMIZERS_SOURCE = Path(
    os.environ.get(
        "RL_MUON_OPTIMIZERS_SOURCE",
        Path(__file__).resolve().parents[1]
        / "routed-scale-source"
        / "verl"
        / "utils"
        / "optimizers.py",
    )
)
SPEC = importlib.util.spec_from_file_location("candidate_optimizers", OPTIMIZERS_SOURCE)
assert SPEC is not None and SPEC.loader is not None
OPTIMIZERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPTIMIZERS)
Lion = OPTIMIZERS.Lion


def _reference_step(parameter, gradient, moment, lr, weight_decay, betas):
    beta1, beta2 = betas
    update = (beta1 * moment + (1 - beta1) * gradient).sign()
    parameter = parameter * (1 - lr * weight_decay) - lr * update
    moment = beta2 * moment + (1 - beta2) * gradient
    return parameter, moment


def test_two_steps_match_canonical_lion_reference():
    initial = torch.tensor([1.0, -2.0, 0.5])
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = Lion([parameter], lr=0.01, weight_decay=0.1, betas=(0.9, 0.99))
    reference = initial.clone()
    moment = torch.zeros_like(reference)

    for gradient in (torch.tensor([2.0, -0.5, 0.25]), torch.tensor([-4.0, -0.1, 1.0])):
        parameter.grad = gradient.clone()
        optimizer.step()
        reference, moment = _reference_step(reference, gradient, moment, 0.01, 0.1, (0.9, 0.99))
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], moment, rtol=0, atol=0)


def test_all_parameter_shapes_use_lion_and_configured_betas():
    parameters = [
        torch.nn.Parameter(torch.ones(3)),
        torch.nn.Parameter(torch.ones(2, 3)),
        torch.nn.Parameter(torch.ones(4, 2)),
    ]
    optimizer = Lion(parameters, lr=1e-6, weight_decay=0.01, betas=(0.9, 0.99))
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 0.5)
    optimizer.step()

    assert optimizer.param_groups[0]["betas"] == (0.9, 0.99)
    assert set(optimizer.state) == set(parameters)
    assert all("exp_avg" in optimizer.state[parameter] for parameter in parameters)


def test_state_dict_round_trip_preserves_next_update():
    first = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    first_optimizer = Lion([first], lr=0.02, weight_decay=0.01)
    first.grad = torch.tensor([2.0, -3.0])
    first_optimizer.step()

    second = torch.nn.Parameter(first.detach().clone())
    second_optimizer = Lion([second], lr=0.02, weight_decay=0.01)
    second_optimizer.load_state_dict(first_optimizer.state_dict())
    next_gradient = torch.tensor([-4.0, 0.5])
    first.grad = next_gradient.clone()
    second.grad = next_gradient.clone()
    first_optimizer.step()
    second_optimizer.step()

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_sparse_gradients_fail_closed():
    parameter = torch.nn.Parameter(torch.ones(3))
    parameter.grad = torch.sparse_coo_tensor(torch.tensor([[0, 2]]), torch.tensor([1.0, -1.0]), (3,))
    optimizer = Lion([parameter], lr=0.01, weight_decay=0.0)

    with unittest.TestCase().assertRaisesRegex(RuntimeError, "sparse"):
        optimizer.step()
