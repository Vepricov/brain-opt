import importlib.util
import os
from pathlib import Path

import torch


OPTIMIZERS_SOURCE = Path(
    os.environ.get(
        "RL_MUON_OPTIMIZERS_SOURCE",
        Path(__file__).resolve().parents[1] / "routed-scale-source" / "verl" / "utils" / "optimizers.py",
    )
)
SPEC = importlib.util.spec_from_file_location("candidate_optimizers", OPTIMIZERS_SOURCE)
assert SPEC is not None and SPEC.loader is not None
OPTIMIZERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPTIMIZERS)
MuonWithAuxAdamW = OPTIMIZERS.MuonWithAuxAdamW
RMSMatchedMomentumWithAuxAdamW = OPTIMIZERS.RMSMatchedMomentumWithAuxAdamW


def _parameters():
    hidden = torch.nn.Parameter(
        torch.tensor(
            [
                [1.0, -2.0, 0.5],
                [0.25, 1.5, -0.75],
            ],
            dtype=torch.float32,
        )
    )
    auxiliary = torch.nn.Parameter(torch.tensor([0.5, -1.0], dtype=torch.float32))
    return [("model.layers.0.self_attn.q_proj.weight", hidden), ("model.layers.0.input_layernorm.weight", auxiliary)]


def _step(optimizer_cls):
    named_parameters = _parameters()
    before = [parameter.detach().clone() for _, parameter in named_parameters]
    named_parameters[0][1].grad = torch.tensor(
        [[3.0, 0.25, -0.5], [0.1, -2.0, 0.75]], dtype=torch.float32
    )
    named_parameters[1][1].grad = torch.tensor([0.4, -0.2], dtype=torch.float32)
    optimizer = optimizer_cls(
        named_parameters,
        lr=0.01,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        muon_adjust_lr_fn="match_rms_adamw",
    )
    optimizer.step()
    deltas = [parameter.detach() - initial for (_, parameter), initial in zip(named_parameters, before)]
    return optimizer, deltas


def test_hidden_update_matches_muon_rms_without_copying_its_direction():
    _, muon_deltas = _step(MuonWithAuxAdamW)
    _, control_deltas = _step(RMSMatchedMomentumWithAuxAdamW)

    torch.testing.assert_close(control_deltas[0].norm(), muon_deltas[0].norm(), rtol=2e-3, atol=1e-6)
    cosine = torch.nn.functional.cosine_similarity(control_deltas[0].flatten(), muon_deltas[0].flatten(), dim=0)
    assert cosine < 0.999


def test_auxiliary_adamw_route_is_identical_to_muon_composite():
    _, muon_deltas = _step(MuonWithAuxAdamW)
    control, control_deltas = _step(RMSMatchedMomentumWithAuxAdamW)

    torch.testing.assert_close(control_deltas[1], muon_deltas[1], rtol=0, atol=0)
    assert "rms_matched_momentum" in control.parameter_routes
