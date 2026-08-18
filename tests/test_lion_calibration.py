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
        baseline,
        candidate,
        sequences,
        attention,
        prompt_width=2,
        response_mask=response_mask,
    )
    old_logp = torch.log_softmax(torch.tensor(old_logits)[:, 1:3], dim=-1)
    new_logp = torch.log_softmax(torch.tensor(new_logits)[:, 1:3], dim=-1)
    direct = (old_logp.exp() * (old_logp - new_logp)).sum(-1).reshape(-1)[:1]

    torch.testing.assert_close(actual, direct)
    assert actual.numel() == 1


def test_candidate_selection_requires_mean_and_q95_to_match_adam():
    adam = {"mean": 0.006, "q95": 0.025}
    trials = [
        {"learning_rate": 1e-6, "mean": 0.006, "q95": 0.030},
        {"learning_rate": 9e-7, "mean": 0.0057, "q95": 0.024},
    ]
    assert CALIBRATION.select_candidate(trials, adam)["learning_rate"] == 9e-7
    assert trials[0]["q95_relative_error"] > CALIBRATION.MATCH_RELATIVE_TOLERANCE
    assert trials[1]["mean_relative_error"] <= CALIBRATION.MATCH_RELATIVE_TOLERANCE
    assert trials[1]["q95_relative_error"] <= CALIBRATION.MATCH_RELATIVE_TOLERANCE


def test_candidate_selection_fails_closed_without_joint_match():
    adam = {"mean": 0.006, "q95": 0.025}
    trials = [
        {"learning_rate": 1e-6, "mean": 0.006, "q95": 0.030},
        {"learning_rate": 5e-7, "mean": 0.003, "q95": 0.0125},
    ]
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "no Lion learning rate"):
        CALIBRATION.select_candidate(trials, adam)


def test_calibration_names_actual_metric_and_keeps_kl_reduction_on_device():
    source = SOURCE.read_text()
    assert CALIBRATION.CALIBRATION_METRIC == (
        "exact_full_categorical_KL_old_to_new_on_occupied_response_states"
    )
    kl_function = source[
        source.index("def occupied_state_categorical_kl") : source.index(
            "def summarize_kl"
        )
    ]
    assert ".cpu()" not in kl_function
    assert 'device = torch.device("cuda")' in source
    assert '"model_compute_device": "cuda"' in source
    assert "old policy logprobs mutated during calibration" in source
    assert "reference policy logprobs mutated during calibration" in source


def test_both_non_adam_routes_share_frozen_gradients_and_joint_matching():
    source = SOURCE.read_text()
    assert "MuonWithAuxAdamW" in source
    assert "muon_trials" in source and "lion_trials" in source
    assert 'select_candidate(muon_trials, adam_kl, "Muon")' in source
    assert 'select_candidate(lion_trials, adam_kl, "Lion")' in source
    assert source.count("restore_with_gradients(actor, baseline_state, gradients)") >= 3
