import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CloudH7R2ProfileTest(unittest.TestCase):
    def test_profile_uses_cloud_base_versions_and_fresh_namespace(self):
        profile = json.loads((ROOT / "cloud_h7_r2" / "profile.json").read_text())

        self.assertEqual(profile["python"], "3.10.14")
        self.assertEqual(profile["torch"], "2.1.2+cu121")
        self.assertEqual(profile["transformers"], "4.44.0")
        self.assertEqual(profile["datasets"], "2.20.0")
        self.assertEqual(profile["accelerate"], "0.33.0")
        self.assertEqual(profile["numpy"], "1.26.4")
        self.assertEqual(profile["campaign_root"], "/home/jovyan/rl_muon/h7_online_cloud_r2")

    def test_bootstrap_is_fail_closed_and_runs_compatibility_checks(self):
        script = (ROOT / "bootstrap_cloud_h7_r2.sh").read_text()

        self.assertIn("cloud_h7_r2/profile.json", script)
        self.assertIn("h7_online_cloud_r2", script)
        self.assertIn('state_root="$campaign_root/bootstrap"', script)
        self.assertIn("unittest discover", script)
        self.assertIn("cloud_h7_r2/tests", script)
        self.assertIn("AutoConfig.from_pretrained", script)
        self.assertIn("tiny_model.save_pretrained", script)
        self.assertIn("AutoModelForCausalLM.from_pretrained", script)
        self.assertIn("status=0", script)
        self.assertIn('"state":"failed"', script)
        self.assertIn('if ! mkdir "$state_root"', script)

    def test_runner_uses_r2_bootstrap_and_preserves_scientific_gates(self):
        script = (ROOT / "run_cloud_h7_r2.sh").read_text()

        self.assertIn("h7_online_cloud_r2", script)
        self.assertIn("cloud_h7_r2/profile.json", script)
        self.assertIn("waiting_for_bootstrap", script)
        self.assertIn("raw_muon own_polar_d01 own_polar_d1", script)
        self.assertIn("progress update schedule mismatch", script)
        self.assertIn("strict budget mismatch", script)

    def test_collector_reads_only_the_fresh_r2_namespace(self):
        script = (ROOT / "collect_cloud_h7_r2.sh").read_text()

        self.assertIn("h7_online_cloud_r2", script)
        self.assertNotIn("h7_online_cloud_r1", script)
        self.assertIn("payload.failed.json", script)
        self.assertIn('"state":"complete"', script)

    def test_r2_loader_uses_transformers_4_torch_dtype_keyword(self):
        source = (ROOT / "cloud_h7_r2" / "src" / "llm_ppo.py").read_text()

        self.assertIn("torch_dtype=torch.float32", source)
        self.assertIn("torch_dtype=frozen_dtype", source)
        self.assertNotIn("            dtype=frozen_dtype,", source)


if __name__ == "__main__":
    unittest.main()
