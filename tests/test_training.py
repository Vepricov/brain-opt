"""Offline export tests use a tiny random model, not a quality benchmark."""

import json
import subprocess
import sys

import pytest
import torch

from brain_opt.simplex_lora import SimplexLoRAConfig
from brain_opt.training import (TrainingConfig, pack_text, resolve_device,
                                restore_parameter_dtypes, run_training)


@pytest.fixture
def local_inputs(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    base = tmp_path / "model-without-architecture-in-name"
    base.mkdir()
    vocab = {"<unk>": 0, "<eos>": 1, "<pad>": 2,
             "small": 3, "models": 4, "learn": 5, "from": 6, "text": 7,
             "validation": 8, "data": 9, ".": 10}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>")
    tokenizer.save_pretrained(base)
    # Transformers 5 saves the generic backend class under a different name;
    # keep this WordLevel fixture from being reinterpreted as a Qwen BPE.
    tokenizer_config = base / "tokenizer_config.json"
    saved = json.loads(tokenizer_config.read_text())
    saved["tokenizer_class"] = "PreTrainedTokenizerFast"
    tokenizer_config.write_text(json.dumps(saved))
    torch.manual_seed(8)
    transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, tie_word_embeddings=False)).save_pretrained(base)
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text("small models learn from text .\n" * 30)
    val.write_text("validation data from small models .\n" * 20)
    return base, train, val, tokenizer


@pytest.mark.parametrize("mode, optimizer", [
    ("full", "AdamW"), ("full", "Lion"), ("lora", "AdamW"),
    ("lora", "Muon"), ("simplexlora", "AdamW"),
])
def test_training_reload(local_inputs, tmp_path, mode, optimizer):
    pytest.importorskip("peft")
    base, train, val, _ = local_inputs
    before = (base / "model.safetensors").read_bytes()
    output = tmp_path / "result"
    simplex = SimplexLoRAConfig(total_steps=5, initial_rank=2, rank_updates=1,
                               steps_per_update=2, lr=0.005, warmup_steps=0) if mode == "simplexlora" else None
    config = TrainingConfig(mode=mode, optimizer=optimizer, steps=5, rank=2, lora_alpha=4,
                            seq_len=8, train_blocks=8, val_blocks=3, batch_size=2,
                            accumulation=2, lr=0.005, simplex=simplex)
    result = run_training(str(base), str(train), str(val), str(output), config)
    assert result["status"] == "COMPLETED"
    assert result["reload_verified"] is True
    assert result["reload_loss"] == pytest.approx(result["val_loss"], abs=1e-5)
    assert result["training_peak_allocated_bytes"] is None
    assert result["input_tokens_seen"] == 160
    assert len(result["history"]) == 5
    assert (base / "model.safetensors").read_bytes() == before
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["config"]["auto_scale_lr"] is False
    assert protocol["dataset"]["train_sha256"] != protocol["dataset"]["validation_sha256"]
    if mode != "full":
        assert (output / "checkpoint" / "adapter_config.json").exists()
        assert not (output / "checkpoint" / "model.safetensors").exists()
    if mode == "simplexlora":
        evidence = json.loads((output / "checkpoint" / "simplexlora_rank_history.json").read_text())
        assert evidence["final"]["step"] == 5
        assert evidence["final"]["adapter_parameter_budget_respected"]


@pytest.mark.parametrize("kwargs", [
    {"mode": "qlora"}, {"steps": 0}, {"batch_size": True}, {"seq_len": 1},
    {"dtype": "float16"}, {"lr": float("nan")}, {"optimizer": "unknown"},
    {"target_modules": "q_proj"}, {"mode": "simplexlora"},
    {"simplex": SimplexLoRAConfig()}, {"weight_decay": -1},
])
def test_invalid_training_config(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_no_synthetic_data_fallback(local_inputs, tmp_path):
    _, _, _, tokenizer = local_inputs
    empty = tmp_path / "empty.txt"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="insufficient real text"):
        pack_text(empty, tokenizer, 2, 8)


def test_no_overwrite_or_input_output_overlap(local_inputs, tmp_path):
    base, train, val, _ = local_inputs
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep"
    marker.write_text("untouched")
    with pytest.raises(FileExistsError):
        run_training(str(base), str(train), str(val), str(existing), TrainingConfig())
    assert marker.read_text() == "untouched"
    with pytest.raises(ValueError, match="must not contain"):
        run_training(str(base), str(train), str(val), str(base / "nested"), TrainingConfig())


def test_identical_validation_rejected(local_inputs, tmp_path):
    base, train, _, _ = local_inputs
    with pytest.raises(ValueError, match="must be different"):
        run_training(str(base), str(train), str(train), str(tmp_path / "out"), TrainingConfig())


def test_cli(local_inputs, tmp_path):
    base, train, val, _ = local_inputs
    config = tmp_path / "run.json"
    config.write_text(json.dumps({
        "base_checkpoint": str(base), "train_text": str(train), "validation_text": str(val),
        "output_dir": str(tmp_path / "cli-out"),
        "config": {"mode": "full", "steps": 1, "seq_len": 8, "train_blocks": 4, "val_blocks": 2},
    }))
    run = subprocess.run([sys.executable, "-m", "brain_opt.cli", "train", str(config)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["reload_verified"]


def test_optimizer_import_does_not_load_optional_dependencies():
    code = "import sys, brain_opt; assert not {'peft', 'transformers', 'llmcompressor'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_scratch_does_not_require_or_load_weights(local_inputs, tmp_path):
    base, train, val, _ = local_inputs
    (base / "model.safetensors").unlink()
    cfg = TrainingConfig(mode="full", initialization="scratch", steps=2, seq_len=8,
                         train_blocks=4, val_blocks=2)
    result = run_training(str(base), str(train), str(val), str(tmp_path / "scratch"), cfg)
    assert result["reload_verified"]
    protocol = json.loads((tmp_path / "scratch" / "protocol.json").read_text())
    assert protocol["input_weights_used"] is False


@pytest.mark.parametrize("mode", ["lora", "simplexlora"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_merged_export_is_standalone(local_inputs, tmp_path, mode, dtype):
    pytest.importorskip("peft")
    base, train, val, _ = local_inputs
    simplex = SimplexLoRAConfig(total_steps=3, initial_rank=2, rank_updates=1,
                               steps_per_update=1, lr=0.001, warmup_steps=0) if mode == "simplexlora" else None
    cfg = TrainingConfig(mode=mode, dtype=dtype, export_merged=True, steps=3, rank=2,
                         seq_len=8, train_blocks=4, val_blocks=2, lr=0.001, simplex=simplex)
    output = tmp_path / "merged"
    result = run_training(str(base), str(train), str(val), str(output), cfg)
    assert result["reload_verified"]
    assert (output / "adapter" / "adapter_config.json").exists()
    assert (output / "checkpoint" / "model.safetensors").exists()
    assert not (output / "checkpoint" / "adapter_config.json").exists()
    merged = json.loads((output / "merge_check.json").read_text())
    if dtype == "float32":
        assert merged["absolute_loss_difference"] < 1e-5
    assert merged["quality_gate_applied"] is False


@pytest.mark.parametrize("kwargs", [
    {"initialization": "random"}, {"initialization": "scratch", "mode": "lora"},
    {"export_merged": "false"}, {"auto_scale_lr": "false"}, {"seed": -1},
])
def test_initialization_and_flags_are_explicit(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_cuda_without_index_resolves_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert resolve_device("cuda") == torch.device("cuda:3")
    assert resolve_device("cuda:0") == torch.device("cuda:0")
    assert resolve_device("cpu") == torch.device("cpu")


def test_cuda_unavailable_has_explicit_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        resolve_device("cuda")


def test_different_files_with_identical_used_tokens_rejected(local_inputs, tmp_path):
    base, train, val, _ = local_inputs
    val.write_text(train.read_text() + "unused additional validation suffix\n")
    with pytest.raises(ValueError, match="all validation token blocks"):
        run_training(str(base), str(train), str(val), str(tmp_path / "contaminated"),
                     TrainingConfig(mode="full", seq_len=8, train_blocks=8, val_blocks=4))


def test_repeat_seed_reproduces_learning(local_inputs, tmp_path):
    base, train, val, _ = local_inputs
    config = TrainingConfig(mode="full", initialization="scratch", steps=2,
                            seq_len=8, train_blocks=4, val_blocks=2)
    first = run_training(str(base), str(train), str(val), str(tmp_path / "repeat-a"), config)
    second = run_training(str(base), str(train), str(val), str(tmp_path / "repeat-b"), config)
    assert first["history"] == second["history"]
    assert first["val_loss"] == second["val_loss"]


def test_merge_dtype_restore_preserves_nonpersistent_fp32_buffers():
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    model.register_buffer("inv_freq", torch.tensor([0.123456789]), persistent=False)
    original = {name: p.dtype for name, p in model.named_parameters()}
    frequencies = model.inv_freq.clone()
    model.weight.data = model.weight.data.float()
    restore_parameter_dtypes(model, original)
    assert model.weight.dtype == torch.bfloat16
    assert model.inv_freq.dtype == torch.float32
    assert torch.equal(model.inv_freq, frequencies)


def test_merge_dtype_restore_rejects_changed_parameter_names():
    with pytest.raises(ValueError, match="parameter names"):
        restore_parameter_dtypes(torch.nn.Linear(2, 2), {})
