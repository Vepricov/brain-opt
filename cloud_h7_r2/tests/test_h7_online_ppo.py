import json
import tempfile
import unittest
from pathlib import Path

import torch

from h7_online_ppo import (
    _config_from_json,
    build_actor_optimizer,
    functional_value_change_rms,
    route_direction,
)


class TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return self.proj(input_ids.float()).squeeze(-1)


class H7OnlinePpoTest(unittest.TestCase):
    def test_actor_route_uses_zero_weight_decay_adamw(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = build_actor_optimizer([parameter], 3e-6)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)

    def test_exact_protocol_config_rejects_wrong_batch_size(self):
        sha = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            evaluate = root / "eval.jsonl"
            train.write_text('{"prompt":"x"}\n')
            evaluate.write_text('{"prompt":"y"}\n')
            config = {
                "train_prompts_path": str(train),
                "eval_prompts_path": str(evaluate),
                "actor_model_revision": sha,
                "reference_model_revision": sha,
                "critic_model_revision": sha,
                "reward_model_id": "reward/model",
                "reward_model_revision": sha,
                "evaluator_model_id": "evaluator/model",
                "evaluator_model_revision": sha,
                "total_updates": 20,
                "ppo_epochs": 1,
                "eval_every": 5,
                "batch_size": 3,
                "actor_adam_lr": 3e-6,
                "critic_adam_lr": 3e-6,
            }
            path = root / "config.json"
            path.write_text(json.dumps(config))

            with self.assertRaisesRegex(ValueError, "batch_size=2"):
                _config_from_json(path)

    def test_functional_value_trial_does_not_mutate_critic(self):
        critic = TinyCritic()
        sequences = torch.tensor([[[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]]])
        attention = torch.ones(1, 3)
        mask = torch.ones(1, 2)
        with torch.no_grad():
            baseline = critic(sequences, attention)[:, 0:-1]
        before = critic.proj.weight.detach().clone()

        measured = functional_value_change_rms(
            critic,
            sequences=sequences,
            attention=attention,
            prompt_width=1,
            response_mask=mask,
            baseline_values=baseline,
            updates={"proj.weight": torch.ones_like(critic.proj.weight)},
            scale=0.25,
        )

        self.assertGreater(measured, 0.0)
        self.assertTrue(torch.equal(critic.proj.weight, before))

    def test_only_predeclared_routes_are_accepted(self):
        gradient = {"weight": torch.eye(2)}
        factors = {"weight": {"left": torch.ones(2), "right": torch.ones(2)}}

        for route in ("raw_muon", "own_polar_d01", "own_polar_d1"):
            self.assertEqual(set(route_direction(route, gradient, factors)), {"weight"})
        with self.assertRaisesRegex(ValueError, "unknown critic route"):
            route_direction("swapped", gradient, factors)


if __name__ == "__main__":
    unittest.main()

