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
        self.assertIn("pip install --no-cache-dir --ignore-installed", script)

    def test_terminal_markers_are_written_as_one_atomic_log_line(self):
        bootstrap = (ROOT / "bootstrap_cloud_gsm8k.sh").read_text()
        runner = (ROOT / "run_cloud_gsm8k.sh").read_text()

        self.assertIn("printf 'RL_MUON_TERMINAL {", bootstrap)
        self.assertIn("printf 'RL_MUON_TERMINAL {", runner)
        self.assertNotIn("printf 'RL_MUON_TERMINAL '\n  cat", bootstrap)
        self.assertNotIn("printf 'RL_MUON_TERMINAL '\n  cat", runner)

    def test_runner_emits_heartbeat_during_silent_model_startup(self):
        runner = (ROOT / "run_cloud_gsm8k.sh").read_text()

        self.assertIn("RL_MUON_HEARTBEAT", runner)
        self.assertIn("while sleep 60", runner)
        self.assertIn('kill "$heartbeat_pid"', runner)

    def test_verl_patch_omits_unsupported_vllm_085_logprobs_mode(self):
        patch = (ROOT / "0001-feat-add-role-routed-Muon-optimizer-for-GSM8K-PPO.patch").read_text()

        self.assertIn('if _VLLM_VERSION >= version.parse("0.9.0"):', patch)
        self.assertIn('args["logprobs_mode"] = self.config.logprobs_mode', patch)

    def test_runner_repairs_reused_campaign_vllm_argv_under_lock(self):
        runner = (ROOT / "run_cloud_gsm8k.sh").read_text()

        self.assertIn("vllm_async_server.py", runner)
        self.assertIn("fcntl.LOCK_EX", runner)
        self.assertIn('replace(logprobs_needle, "")', runner)
        self.assertIn('hasattr(engine_client, "reset_mm_cache")', runner)
        self.assertIn('getattr(self.engine, "wait_for_requests_to_drain", None)', runner)
        self.assertIn("self.engine.output_processor.request_states", runner)
        self.assertIn("_process_weights_after_loading as process_weights_after_loading", runner)
        self.assertIn('if multi_modal_data:', runner)
        self.assertIn('prompt_kwargs["multi_modal_data"] = multi_modal_data', runner)
        self.assertIn("weight_utils_path", runner)
        self.assertIn("compile(updated", runner)

    def test_runner_repairs_reused_campaign_tensordict_version_under_lock(self):
        runner = (ROOT / "run_cloud_gsm8k.sh").read_text()
        requirements = (ROOT / "requirements-gsm8k.txt").read_text()

        self.assertIn("tensordict==0.10.0", requirements)
        self.assertIn("pyvers==0.1.0", requirements)
        self.assertIn('.tensordict010-compat.lock', runner)
        self.assertIn('Version(tensordict.__version__) < Version("0.10")', runner)
        self.assertIn("from tensordict.tensorclass import NonTensorData, NonTensorStack", runner)

    def test_runner_avoids_flash_attention_remove_padding_path(self):
        runner = (ROOT / "run_cloud_gsm8k.sh").read_text()

        self.assertIn("actor_rollout_ref.model.use_remove_padding=False", runner)
        self.assertIn("critic.model.use_remove_padding=False", runner)

    def test_runner_repairs_trainer_padding_helpers_without_flash_attention(self):
        runner = (ROOT / "run_cloud_gsm8k.sh").read_text()

        self.assertIn("attention_utils.py", runner)
        self.assertIn("compatible_attention_import", runner)
        self.assertIn("from verl.utils.npu_flash_attn_utils import", runner)
        self.assertIn('if exc.name != "flash_attn"', runner)
        self.assertIn("compile(attention_updated", runner)

    def test_full_run_disables_large_periodic_checkpoints(self):
        runner = (ROOT / "run_cloud_gsm8k.sh").read_text()

        full_branch = runner.split('  full)\n', 1)[1]
        full_branch = full_branch.split('    ;;', 1)[0]
        self.assertIn('extra_args=(trainer.save_freq=-1)', full_branch)

    def test_routed_modes_overlay_pinned_source_and_run_only_control_route(self):
        runner = (ROOT / "run_cloud_gsm8k.sh").read_text()

        self.assertIn("smoke|full|routed-smoke|routed-full", runner)
        self.assertIn('if [[ "$mode" == routed-* ]]', runner)
        self.assertIn('routes=(routed_scale_adam_actor)', runner)
        self.assertIn('"${routes[@]}" || finish $?', runner)
        for relative_path in (
            "routed-scale-source/verl/utils/optimizers.py",
            "routed-scale-source/verl/workers/config/optimizer.py",
            "routed-scale-source/examples/ppo_trainer/run_qwen2_5_0_5b_gsm8k_optimizer_ablation.sh",
        ):
            self.assertTrue((ROOT / relative_path).is_file())

    def test_runner_fail_closes_unless_campaign_python_is_selected(self):
        runner = (ROOT / "run_cloud_gsm8k.sh").read_text()

        self.assertIn('venv_python="$venv_root/bin/python3"', runner)
        self.assertIn('$(command -v python3) != "$venv_python"', runner)
        self.assertIn('cloudpickle==3.1.1', runner)

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
                    {"step": 0, "data": {metric: 0.25}},
                    {"step": 3, "data": {metric: 0.75,
                                           "actor/grad_norm": 1.5,
                                           "actor/ppo_kl": 0.01,
                                           "actor/pg_clipfrac": 0.02,
                                           "critic/vf_clipfrac": 0.03}},
                ]
                (run / "metrics.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows))

            result = build_result(root, "full", 0, "commit", 3)

        route = result["routes"]["muon_actor"]
        self.assertEqual([0, 3], [point["step"]
                                  for point in route["validation_points"]])
        self.assertEqual(0.75, route["final_validation"])
        self.assertEqual(0.5, route["validation_auc"])
        self.assertEqual(1.5, route["terminal_metrics"]["actor/grad_norm"])
        self.assertEqual(0.01, route["safety_points"][0]["metrics"]["actor/ppo_kl"])

    def test_result_can_be_bound_to_only_the_routed_scale_control(self):
        metric = "val-core/openai/gsm8k/reward/mean@1"
        control = "routed_scale_adam_actor"
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = root / control
            run.mkdir()
            rows = [
                {"step": 0, "data": {metric: 0.25}},
                {"step": 1, "data": {
                    metric: 0.5,
                    "actor/ppo_kl": 0.01,
                    "actor/pg_clipfrac": 0.02,
                    "critic/vf_clipfrac": 0.03,
                }},
            ]
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))

            result = build_result(
                root, "routed-smoke", 0, "commit", 1, (control,))

        self.assertEqual([control], list(result["routes"]))
        self.assertEqual(1, result["routes"][control]["terminal_step"])


if __name__ == "__main__":
    unittest.main()
