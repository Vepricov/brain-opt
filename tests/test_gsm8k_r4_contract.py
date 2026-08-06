from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_bootstrap_preserves_pinned_protobuf_and_reports_inner_terminal_state():
    script = (ROOT / "bootstrap_cloud_gsm8k.sh").read_text()

    assert 'RL_MUON_SOURCE_COMMIT:?RL_MUON_SOURCE_COMMIT is required' in script
    assert 'RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required' in script
    assert 'pip install --user --no-deps -e "$verl_root"' in script
    assert "RL_MUON_TERMINAL" in script
    assert "test_muon_optimizer_r4_geometry.py" in script


def test_runner_uses_fresh_r4_namespace_and_reports_inner_terminal_state():
    script = (ROOT / "run_cloud_gsm8k.sh").read_text()

    assert 'RL_MUON_SOURCE_COMMIT:?RL_MUON_SOURCE_COMMIT is required' in script
    assert 'RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required' in script
    assert ".local-gsm8k-vllm085-r4" in script
    assert "RL_MUON_TERMINAL" in script
    assert "gsm8k_ppo_r3" not in script
