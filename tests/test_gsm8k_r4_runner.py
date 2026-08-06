import unittest
import json
import tempfile
from pathlib import Path

from collect_gsm8k_r4_result import build_result


ROOT = Path(__file__).resolve().parents[1]


class Gsm8kR4RunnerTest(unittest.TestCase):
    def test_bootstrap_does_not_duplicate_large_wheels_in_pip_cache(self):
        script = (ROOT / "bootstrap_cloud_gsm8k.sh").read_text()

        self.assertIn("export PIP_NO_CACHE_DIR=1", script)
        self.assertIn("pip install --user --no-cache-dir", script)

    def test_terminal_markers_are_written_as_one_atomic_log_line(self):
        bootstrap = (ROOT / "bootstrap_cloud_gsm8k.sh").read_text()
        runner = (ROOT / "run_cloud_gsm8k.sh").read_text()

        self.assertIn("printf 'RL_MUON_TERMINAL {", bootstrap)
        self.assertIn("printf 'RL_MUON_TERMINAL {", runner)
        self.assertNotIn("printf 'RL_MUON_TERMINAL '\n  cat", bootstrap)
        self.assertNotIn("printf 'RL_MUON_TERMINAL '\n  cat", runner)

    def test_scientific_result_requires_validation_endpoint_and_auc(self):
        runner = (ROOT / "run_cloud_gsm8k.sh").read_text()

        self.assertIn("collect_gsm8k_r4_result.py", runner)

    def test_result_contains_exact_validation_trajectory_and_auc(self):
        metric = "val-core/openai/gsm8k/reward/mean@1"
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for route in ("adam_adam", "muon_actor", "muon_critic"):
                run = root / route
                run.mkdir()
                rows = [
                    {"step": 1, "data": {metric: 0.25}},
                    {"step": 3, "data": {metric: 0.75,
                                           "actor/grad_norm": 1.5}},
                ]
                (run / "metrics.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows))

            result = build_result(root, "full", 0, "commit", 3)

        route = result["routes"]["muon_actor"]
        self.assertEqual([1, 3], [point["step"]
                                  for point in route["validation_points"]])
        self.assertEqual(0.75, route["final_validation"])
        self.assertEqual(0.5, route["validation_auc"])
        self.assertEqual(1.5, route["terminal_metrics"]["actor/grad_norm"])


if __name__ == "__main__":
    unittest.main()
