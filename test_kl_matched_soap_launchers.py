"""Static launcher contract tests that do not require the target GPU runtime."""

from pathlib import Path

ROOT = Path(__file__).parent


def test_production_launcher_has_exact_seed_step_validation_and_checkpoint_contract():
    seed_runner = (ROOT / "run_kl_matched_soap_seed.sh").read_text()
    campaign = (ROOT / "launch_kl_matched_soap_seeds.sh").read_text()
    runner = (ROOT / "run_matched_soap_config_adamw.sh").read_text()
    assert "0 25 50 75 100 125 150" in seed_runner
    assert "EXPECTED_STEP=150" in seed_runner
    assert "SAVE_FREQ=25" in seed_runner
    assert "TEST_FREQ=25" in seed_runner
    assert "for seed in 0 1 2" in campaign
    assert "critic.optim.optimizer=AdamW" in runner
    assert "critic.optim.optimizer_impl=torch.optim" in runner
    assert (
        "fisher_prompt_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]"
        in runner
    )


def test_smoke_is_exactly_one_step_then_real_auto_resume_to_step_two():
    smoke = (ROOT / "smoke_kl_matched_soap_resume.sh").read_text()
    runner = (ROOT / "run_matched_soap_config_adamw.sh").read_text()
    assert smoke.index("EXPECTED_STEP=1") < smoke.index("EXPECTED_STEP=2")
    assert "global_step_1/actor/optim_world_size_1_rank_0.pt" in smoke
    assert "global_step_2/actor/optim_world_size_1_rank_0.pt" in smoke
    assert "trainer.resume_mode=auto" in runner
    assert "+trainer.save_initial_checkpoint=True" in runner
    assert "VERL_ROOT=${RL_MUON_VERL_ROOT:-$repo_root/vendor/verl}" in runner


def test_fsdp_proposal_hook_is_after_gradient_clipping_and_before_parameter_step():
    engine = (
        ROOT / "vendor/verl/verl/workers/engine/fsdp/transformer_impl.py"
    ).read_text()
    optimizer_step = engine[
        engine.index("    def optimizer_step(self):") : engine.index(
            "    def lr_scheduler_step(self):"
        )
    ]
    clip = optimizer_step.index("clip_grad_norm_")
    step = optimizer_step.index("self.optimizer.step()")
    assert clip < step
    assert "latest_telemetry" in optimizer_step
