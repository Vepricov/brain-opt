import importlib.util
import os
from pathlib import Path
import types
import unittest

import torch


SOURCE = Path(
    os.environ.get(
        "RL_MUON_LION_CALIBRATION_SOURCE",
        Path(__file__).resolve().parents[1] / "calibrate_lion_actor_lr.py",
    )
)
SPEC = importlib.util.spec_from_file_location("lion_calibration", SOURCE)
assert SPEC is not None and SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATION)


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, input_ids, attention_mask):
        batch = input_ids.shape[0]
        return types.SimpleNamespace(logits=self.fixed_logits.expand(batch, -1, -1))


def test_exact_full_categorical_kl_matches_direct_statewise_calculation_and_mask():
    old_logits = [[[2.0, 0.0], [1.0, -1.0], [0.5, 0.5], [3.0, -2.0]]]
    new_logits = [[[1.0, 1.0], [0.0, 0.0], [1.5, -0.5], [-1.0, 2.0]]]
    baseline = FixedLogitModel(old_logits)
    candidate = FixedLogitModel(new_logits)
    sequences = torch.tensor([[0, 1, 0, 1]])
    attention = torch.ones_like(sequences)
    response_mask = torch.tensor([[1.0, 0.0]])

    actual = CALIBRATION.occupied_state_categorical_kl(
        baseline, candidate, sequences, attention, prompt_width=2, response_mask=response_mask
    )
    old_logp = torch.log_softmax(torch.tensor(old_logits)[:, 1:3], dim=-1)
    new_logp = torch.log_softmax(torch.tensor(new_logits)[:, 1:3], dim=-1)
    direct = (old_logp.exp() * (old_logp - new_logp)).sum(-1).reshape(-1)[:1]

    torch.testing.assert_close(actual, direct)
    assert actual.numel() == 1


def test_candidate_selection_rejects_unsafe_q95_even_when_mean_matches():
    trials = [
        {"learning_rate": 1e-6, "mean_relative_error": 0.0, "safe": False, "mean": 0.001, "q95": 0.005},
        {"learning_rate": 9e-7, "mean_relative_error": 0.05, "safe": True, "mean": 0.00095, "q95": 0.003},
    ]
    assert CALIBRATION.select_candidate(trials)["learning_rate"] == 9e-7


def test_candidate_selection_fails_closed_without_safe_match():
    trials = [
        {"learning_rate": 1e-6, "mean_relative_error": 0.0, "safe": False},
        {"learning_rate": 5e-7, "mean_relative_error": 0.4, "safe": True},
    ]
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "no Lion learning rate"):
        CALIBRATION.select_candidate(trials)
