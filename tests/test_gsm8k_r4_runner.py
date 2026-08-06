import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
