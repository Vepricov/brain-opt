import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import pytest

import cross_dataset_screen
from collect_cross_dataset_screen import build_result, validate_manifest
from cross_dataset_screen import (
    DATASETS,
    ROUTES,
    AmbiguousAnswerError,
    ContractError,
    adapt_arc_easy,
    adapt_records,
    adapt_svamp,
    compute_score,
    model_identity,
    parse_arc_easy_answer,
    parse_svamp_answer,
    validate_gate_metrics,
    validate_held_out,
    validate_screen_metrics,
    verl_identity,
    verify_identity_artifact,
    write_identity_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def test_reward_module_loads_without_preinsertion_in_sys_modules():
    module_path = ROOT / "cross_dataset_screen.py"
    spec = importlib.util.spec_from_file_location("dynamic_cross_dataset_reward", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert module.compute_score(
        "cross_dataset/svamp",
        "work\nFinal answer: 2",
        "2",
        {
            "data_source": "cross_dataset/svamp",
            "dataset": "svamp",
            "split": "validation",
            "index": 0,
            "id": "fixture-0",
        },
    ) == 1.0


def test_reward_accepts_verl_runtime_metadata_without_weakening_provenance_schema():
    extra_info = _reward_extra("cross_dataset/svamp")
    extra_info.update(
        {
            "num_turns": 1,
            "rollout_reward_scores": {},
            "raw_prompt": [{"role": "user", "content": "fixture"}],
        }
    )

    assert compute_score(
        "cross_dataset/svamp",
        "work\nFinal answer: 2",
        "2",
        extra_info,
    ) == 1.0

    with pytest.raises(ContractError, match="provenance schema"):
        compute_score(
            "cross_dataset/svamp",
            "work\nFinal answer: 2",
            "2",
            {**extra_info, "unexpected_runtime_field": True},
        )


def test_verl_critical_paths_match_pinned_v080_runtime_layout():
    assert set(cross_dataset_screen.VERL_CRITICAL_PATHS) == {
        "verl/trainer/main_ppo.py",
        "verl/trainer/ppo/ray_trainer.py",
        "verl/utils/attention_utils.py",
        "verl/utils/model.py",
        "verl/workers/engine_workers.py",
        "verl/workers/engine/fsdp/transformer_impl.py",
        "verl/workers/rollout/vllm_rollout/utils.py",
        "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
        "verl/workers/rollout/vllm_rollout/vllm_rollout.py",
    }


def _reward_extra(source):
    dataset = source.removeprefix("cross_dataset/")
    extra = {
        "data_source": source,
        "dataset": dataset,
        "split": "validation",
        "index": 0,
        "id": "fixture-0",
    }
    if dataset == "arc_easy":
        extra.update(
            {"choice_labels": ["A", "B"], "original_choice_labels": ["1", "2"]}
        )
    return extra


def test_svamp_adapter_is_deterministic_and_uses_held_out_source_schema():
    raw = {
        "ID": "chal-2",
        "Body": "Ada has 2 apples.",
        "Question": "How many apples?",
        "Answer": "2.0",
    }
    row = adapt_svamp(raw, "validation", 7)

    assert row["data_source"] == "cross_dataset/svamp"
    assert row["reward_model"] == {"style": "rule", "ground_truth": "2"}
    assert row["extra_info"] == {
        "data_source": "cross_dataset/svamp",
        "dataset": "svamp",
        "split": "validation",
        "index": 7,
        "id": "chal-2",
    }
    assert row["prompt"][0]["content"].endswith("'Final answer: <number>'.")


def test_arc_adapter_canonicalizes_numeric_choice_labels():
    row = adapt_arc_easy(
        {
            "id": "Mercury-1",
            "question": "Which one?",
            "choices": {"label": ["1", "2", "3"], "text": ["red", "green", "blue"]},
            "answerKey": "2",
        },
        "train",
        0,
    )

    assert row["data_source"] == "cross_dataset/arc_easy"
    assert row["reward_model"]["ground_truth"] == "B"
    assert "A. red\nB. green\nC. blue" in row["prompt"][0]["content"]
    assert row["extra_info"]["original_choice_labels"] == ["1", "2", "3"]


def test_adapt_records_sorts_ids_and_reindexes_deterministically():
    records = [
        {"ID": "z", "Body": "B", "Question": "Q?", "Answer": "1"},
        {"ID": "a", "Body": "B", "Question": "Q?", "Answer": "2"},
    ]
    rows = adapt_records("svamp", records, "train")
    assert [(row["extra_info"]["id"], row["extra_info"]["index"]) for row in rows] == [
        ("a", 0),
        ("z", 1),
    ]


def test_held_out_validation_rejects_id_leakage():
    record = {"ID": "same", "Body": "B", "Question": "Q?", "Answer": "1"}
    train = [adapt_svamp(record, "train", 0)]
    validation = [adapt_svamp(record, "validation", 0)]
    with pytest.raises(ContractError, match="overlap"):
        validate_held_out(train, validation)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work\nFinal answer: 0.5", "1/2"),
        ("work\nFinal answer: -3/2\n", "-3/2"),
        ("no marker", None),
    ],
)
def test_svamp_parser_is_exact(text, expected):
    value = parse_svamp_answer(text)
    assert (str(value) if value is not None else None) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Final answer: 1\nFinal answer: 1",
        "Final answer: 1\nextra",
        "Final answer: 1 or 2",
        "Final answer: 1e3",
        "Final answer: 1/0",
    ],
)
def test_svamp_parser_fails_closed_on_ambiguous_or_noncanonical_answers(text):
    with pytest.raises(AmbiguousAnswerError):
        parse_svamp_answer(text)


def test_arc_parser_and_rewards_are_exact_and_source_bound():
    assert parse_arc_easy_answer("reason\nFinal answer: B") == "B"
    assert (
        compute_score(
            "cross_dataset/arc_easy",
            "reason\nFinal answer: B",
            "B",
            _reward_extra("cross_dataset/arc_easy"),
        )
        == 1.0
    )
    assert (
        compute_score(
            "cross_dataset/svamp",
            "reason\nFinal answer: 2.00",
            "2",
            _reward_extra("cross_dataset/svamp"),
        )
        == 1.0
    )
    with pytest.raises(AmbiguousAnswerError):
        parse_arc_easy_answer("Final answer: b")
    with pytest.raises(ContractError, match="source mismatch"):
        compute_score(
            "cross_dataset/arc_easy",
            "Final answer: A",
            "A",
            _reward_extra("cross_dataset/svamp"),
        )


@pytest.mark.parametrize(
    ("source", "completion", "ground_truth"),
    [
        ("cross_dataset/svamp", "Final answer: 1\nFinal answer: 1", "1"),
        ("cross_dataset/svamp", "final answer: 1", "1"),
        ("cross_dataset/svamp", "Final answer: 1.", "1"),
        ("cross_dataset/svamp", "Final answer: 1e3", "1000"),
        ("cross_dataset/svamp", "missing", "1"),
        ("cross_dataset/arc_easy", "Final answer: b", "B"),
        ("cross_dataset/arc_easy", "Final answer: B.", "B"),
        ("cross_dataset/arc_easy", "", "B"),
    ],
)
def test_malformed_model_completions_deterministically_score_zero(
    source, completion, ground_truth
):
    extra_info = _reward_extra(source)
    assert compute_score(source, completion, ground_truth, extra_info) == 0.0
    assert compute_score(source, completion, ground_truth, extra_info) == 0.0


@pytest.mark.parametrize(
    ("source", "ground_truth", "extra_info", "message"),
    [
        ("cross_dataset/svamp", "1/0", _reward_extra("cross_dataset/svamp"), "ground truth"),
        ("cross_dataset/arc_easy", "b", _reward_extra("cross_dataset/arc_easy"), "ground truth"),
        ("unknown", "1", {"data_source": "unknown"}, "data source"),
        ("cross_dataset/svamp", "1", None, "extra_info"),
        ("cross_dataset/svamp", "1", {}, "source mismatch"),
        ("cross_dataset/svamp", "1", {"data_source": "wrong"}, "source mismatch"),
        ("cross_dataset/svamp", "1", {"data_source": "cross_dataset/svamp"}, "schema"),
    ],
)
def test_malformed_dataset_contract_and_provenance_remain_fatal(
    source, ground_truth, extra_info, message
):
    with pytest.raises(ContractError, match=message):
        compute_score(source, "Final answer: 1", ground_truth, extra_info)


def _write_metrics(
    path: Path, source: str, *, baseline=0.25, bad_safety=None, terminal=50
):
    metric = f"val-core/{source}/reward/mean@1"
    rows = [
        {"step": 0, "data": {metric: baseline}},
        {
            "step": 1,
            "data": {
                "actor/ppo_kl": 0.01,
                "actor/pg_clipfrac": 0.02,
                "critic/vf_clipfrac": 0.03,
            },
        },
    ]
    if terminal >= 25:
        rows.append({"step": 25, "data": {metric: 0.5}})
    if terminal >= 50:
        rows.append({"step": 50, "data": {metric: 0.75}})
    if bad_safety is not None:
        rows[1]["data"]["actor/ppo_kl"] = bad_safety
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("bad_step", [0.9, True, "0"])
def test_metric_steps_require_exact_nonnegative_integers(tmp_path, bad_step):
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(metrics, "cross_dataset/svamp")
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[0]["step"] = bad_step
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ContractError, match="invalid metrics row"):
        validate_gate_metrics(metrics, "cross_dataset/svamp")


@pytest.mark.parametrize(
    "bad_source",
    [
        "attacker-cross_dataset/svamp",
        "cross_dataset/svamp-attacker",
        "attacker/cross_dataset/svamp/embedded",
    ],
)
def test_validation_metric_source_must_match_exactly(tmp_path, bad_source):
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(metrics, bad_source)

    with pytest.raises(ContractError, match="source mismatch"):
        validate_gate_metrics(metrics, "cross_dataset/svamp")


def test_gate_requires_positive_baseline_and_real_finite_step_one(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(metrics, "cross_dataset/svamp")
    gate = validate_gate_metrics(metrics, "cross_dataset/svamp")
    assert gate["baseline_reward"] == 0.25
    assert gate["safety"]["actor/ppo_kl"] == 0.01

    _write_metrics(metrics, "cross_dataset/svamp", baseline=0)
    with pytest.raises(ContractError, match="must be positive"):
        validate_gate_metrics(metrics, "cross_dataset/svamp")
    _write_metrics(metrics, "cross_dataset/svamp", bad_safety=math.inf)
    with pytest.raises(ContractError, match="non-finite"):
        validate_gate_metrics(metrics, "cross_dataset/svamp")
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[1]["data"].pop("critic/vf_clipfrac")
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ContractError, match="step-1 safety"):
        validate_gate_metrics(metrics, "cross_dataset/svamp")


def test_screen_metrics_enforce_max_step_and_validation_schedule(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(metrics, "cross_dataset/arc_easy")
    result = validate_screen_metrics(metrics, "muon_actor", "cross_dataset/arc_easy")
    assert result["terminal_step"] == 50
    assert [point["step"] for point in result["validation_points"]] == [0, 25, 50]

    _write_metrics(metrics, "cross_dataset/arc_easy", terminal=25)
    with pytest.raises(ContractError, match="step 50"):
        validate_screen_metrics(metrics, "muon_actor", "cross_dataset/arc_easy")
    _write_metrics(metrics, "cross_dataset/svamp")
    with pytest.raises(ContractError, match="source mismatch"):
        validate_screen_metrics(metrics, "muon_actor", "cross_dataset/arc_easy")
    with pytest.raises(ContractError, match="unexpected route"):
        validate_screen_metrics(metrics, "muon_critic", "cross_dataset/svamp")


def test_screen_metrics_reject_validation_outside_preregistered_schedule(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(metrics, "cross_dataset/svamp")
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows.append(
        {
            "step": 10,
            "data": {"val-core/cross_dataset/svamp/reward/mean@1": 0.4},
        }
    )
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ContractError, match="validation schedule"):
        validate_screen_metrics(metrics, "adamw_actor", "cross_dataset/svamp")


def _write_provenance(
    root, phase, route, dataset, commit, manifest_sha, calibration_sha,
    model_sha, verl_sha,
):
    optimizer = {
        "adamw_actor": "AdamW",
        "muon_actor": "MuonWithAuxAdamW",
        "lion_actor": "Lion",
    }[route]
    provenance = {
        "protocol": "cross-dataset-actor-screen-v1",
        "phase": phase,
        "route": route,
        "dataset": dataset,
        "data_source": DATASETS[dataset].source,
        "seed": 0,
        "source_commit": commit,
        "manifest_sha256": manifest_sha,
        "critic_optimizer": "AdamW",
        "preserve_old_logprobs": True,
        "preserve_reference_logprobs": True,
        "actor_optimizer": optimizer,
        "actor_learning_rate": 1e-5 if route == "lion_actor" else 1e-6,
        "actor_route_description": {
            "adamw_actor": "AdamW on all actor parameters",
            "muon_actor": "Muon on hidden attention/MLP matrices plus AdamW auxiliaries",
            "lion_actor": "Lion on all actor parameters; strict no-Adam actor route",
        }[route],
        "actor_parameter_routing": {
            "adamw_actor": "all_actor_parameters",
            "muon_actor": "hidden_attention_mlp_matrices_muon;all_auxiliaries_adamw",
            "lion_actor": "all_actor_parameters_lion_no_adam",
        }[route],
        "actor_uses_adamw_auxiliaries": route == "muon_actor",
        "calibration_sha256": calibration_sha,
        "calibration_metric": "exact_full_categorical_KL_old_to_new_on_occupied_response_states",
        "model_snapshot_sha256": model_sha,
        "verl_implementation_sha256": verl_sha,
    }
    (root / "run-provenance.json").write_text(json.dumps(provenance))


def _write_calibration(
    path, dataset, commit, manifest_sha, train_sha, model_sha, verl_sha
):
    adam = {"mean": 0.01, "q95": 0.02, "occupied_states": 8}
    matched = {**adam, "mean_relative_error": 0.0, "q95_relative_error": 0.0}
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "complete",
                "protocol": "per_dataset_same_frozen_batch_one_production_equivalent_ppo_update",
                "dataset": dataset,
                "data_source": DATASETS[dataset].source,
                "seed": 0,
                "source_commit": commit,
                "manifest_sha256": manifest_sha,
                "calibration_metric": "exact_full_categorical_KL_old_to_new_on_occupied_response_states",
                "kl_direction": "old_policy_to_updated_policy",
                "model_compute_device": "cuda",
                "adam_actual_full_categorical_kl": adam,
                "chosen_muon_learning_rate": 1e-6,
                "chosen_muon_actual_full_categorical_kl": matched,
                "chosen_lion_learning_rate": 1e-5,
                "chosen_lion_actual_full_categorical_kl": matched,
                "gates": {"passed": True},
                "identities": {
                    "train_file_sha256": train_sha,
                    "model_snapshot_sha256": model_sha,
                    "verl_implementation_sha256": verl_sha,
                    "rollout_sha256": "1" * 64,
                    "old_logprobs_sha256": "2" * 64,
                    "reference_logprobs_sha256": "3" * 64,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_implementation_identities(tmp_path, monkeypatch, *, include_gitlink=False):
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text('{"model_type":"qwen2"}\n')
    (model_root / "tokenizer.json").write_text('{"version":"1"}\n')
    (model_root / "tokenizer_config.json").write_text('{"padding_side":"left"}\n')
    (model_root / "model-00001-of-00002.safetensors").write_bytes(b"weights-1")
    (model_root / "model-00002-of-00002.safetensors").write_bytes(b"weights-2")
    (model_root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        )
    )

    verl_root = tmp_path / "verl-checkout"
    overlay_root = tmp_path / "routing-overlay"
    for relative in (
        *cross_dataset_screen.VERL_CRITICAL_PATHS,
        *cross_dataset_screen.ROUTING_OVERLAY_PATHS,
    ):
        payload = f"# {relative}\n".encode()
        target = verl_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if relative in cross_dataset_screen.ROUTING_OVERLAY_PATHS:
            overlay = overlay_root / relative
            overlay.parent.mkdir(parents=True, exist_ok=True)
            overlay.write_bytes(payload)
    dangling = verl_root / ".claude/skills/issue"
    dangling.parent.mkdir(parents=True, exist_ok=True)
    dangling.symlink_to("../../.agent/skills/issue")
    subprocess.run(["git", "init", "-q", str(verl_root)], check=True)
    subprocess.run(["git", "-C", str(verl_root), "add", "."], check=True)
    if include_gitlink:
        gitlink_oid = "1" * 40
        subprocess.run(
            [
                "git", "-C", str(verl_root), "update-index", "--add", "--cacheinfo",
                f"160000,{gitlink_oid},recipe",
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(verl_root), "-c", "user.name=Test", "-c",
         "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(verl_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    monkeypatch.setattr(cross_dataset_screen, "PINNED_VERL_COMMIT", commit)
    model = model_identity(model_root)
    verl = verl_identity(verl_root, overlay_root)
    write_identity_artifact(tmp_path / "model-identity.json", model)
    write_identity_artifact(tmp_path / "verl-identity.json", verl)
    return model_root, verl_root, overlay_root, model["identity_sha256"], verl["identity_sha256"]


def test_verl_identity_hashes_dangling_symlink_target(tmp_path, monkeypatch):
    _, verl_root, overlay_root, _, original = _write_implementation_identities(tmp_path, monkeypatch)
    dangling = verl_root / ".claude/skills/issue"
    assert dangling.is_symlink()
    dangling.unlink()
    dangling.symlink_to("../../.agent/skills/other-issue")
    changed = verl_identity(verl_root, overlay_root)
    assert changed["identity_sha256"] != original


def test_verl_identity_hashes_uninitialized_gitlink(tmp_path, monkeypatch):
    _, _, _, _, identity = _write_implementation_identities(
        tmp_path, monkeypatch, include_gitlink=True
    )
    assert isinstance(identity, str)

    entry = cross_dataset_screen._tracked_entry_identity(
        tmp_path, "recipe", "160000", "1" * 40
    )
    assert entry == {
        "path": "recipe",
        "kind": "gitlink",
        "index_mode": "160000",
        "git_oid": "1" * 40,
        "worktree_state": "uninitialized",
    }


def test_tracked_gitlink_rejects_symlink_to_directory(tmp_path):
    (tmp_path / "outside").mkdir()
    (tmp_path / "recipe").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ContractError, match="invalid gitlink worktree path"):
        cross_dataset_screen._tracked_entry_identity(
            tmp_path, "recipe", "160000", "1" * 40
        )


def test_tracked_gitlink_rejects_populated_non_repository(tmp_path):
    (tmp_path / "recipe").mkdir()
    (tmp_path / "recipe/attacker.py").write_text("payload")
    with pytest.raises(ContractError, match="invalid gitlink worktree path"):
        cross_dataset_screen._tracked_entry_identity(
            tmp_path, "recipe", "160000", "1" * 40
        )


def test_tracked_gitlink_rejects_initialized_checkout(tmp_path):
    recipe = tmp_path / "recipe"
    subprocess.run(["git", "init", "-q", str(recipe)], check=True)
    (recipe / "tracked").write_text("payload")
    subprocess.run(["git", "-C", str(recipe), "add", "tracked"], check=True)
    subprocess.run(
        [
            "git", "-C", str(recipe), "-c", "user.name=Test", "-c",
            "user.email=test@example.invalid", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    with pytest.raises(ContractError, match="initialized gitlink is not allowed"):
        cross_dataset_screen._tracked_entry_identity(
            tmp_path, "recipe", "160000", "1" * 40
        )


@pytest.mark.parametrize(
    ("mode", "oid"),
    [
        ("nonsense", "1" * 40),
        ("100644", "not-an-oid"),
        ("100644", "1" * 39),
    ],
)
def test_tracked_entry_rejects_invalid_index_metadata(tmp_path, mode, oid):
    (tmp_path / "ordinary").write_text("payload")
    with pytest.raises(ContractError, match="invalid VERL index"):
        cross_dataset_screen._tracked_entry_identity(tmp_path, "ordinary", mode, oid)


def test_tracked_entry_rejects_index_mode_worktree_mismatch(tmp_path):
    (tmp_path / "ordinary").write_text("payload")
    with pytest.raises(ContractError, match="index mode/worktree mismatch"):
        cross_dataset_screen._tracked_entry_identity(
            tmp_path, "ordinary", "120000", "1" * 40
        )


@pytest.mark.parametrize(("index_mode", "worktree_mode"), [("100644", 0o755), ("100755", 0o644)])
def test_tracked_entry_rejects_executable_bit_mismatch(tmp_path, index_mode, worktree_mode):
    path = tmp_path / "ordinary"
    path.write_text("payload")
    path.chmod(worktree_mode)
    with pytest.raises(ContractError, match="executable-bit mismatch"):
        cross_dataset_screen._tracked_entry_identity(
            tmp_path, "ordinary", index_mode, "1" * 40
        )


def test_parse_tracked_entries_rejects_malformed_and_unmerged_records():
    with pytest.raises(ContractError, match="invalid VERL index entry"):
        cross_dataset_screen._parse_tracked_entries(b"malformed\0")
    with pytest.raises(ContractError, match="unmerged VERL index entry"):
        cross_dataset_screen._parse_tracked_entries(
            f"100644 {'1' * 40} 1\tconflicted\0".encode()
        )


def test_verl_identity_rejects_file_symlink_overlay_collision(tmp_path, monkeypatch):
    _, verl_root, overlay_root, _, _ = _write_implementation_identities(tmp_path, monkeypatch)
    relative = cross_dataset_screen.ROUTING_OVERLAY_PATHS[0]
    active = verl_root / relative
    overlay = overlay_root / relative
    payload = active.read_text()
    overlay.unlink()
    overlay.symlink_to(payload)
    with pytest.raises(ContractError, match="routing overlay is not active"):
        verl_identity(verl_root, overlay_root)


def _write_manifest(tmp_path, dataset="svamp"):
    source = DATASETS[dataset].source
    spec = DATASETS[dataset]
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    train.write_bytes(b"train artifact")
    validation.write_bytes(b"validation artifact")
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "data_source": source,
        "repository": spec.repository,
        "config": spec.config,
        "revision": spec.revision,
        "seed": 0,
        "train_split": spec.train_split,
        "validation_split": spec.validation_split,
        "files": {
            "train.parquet": {
                "rows": spec.expected_train_rows,
                "sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
            },
            "validation.parquet": {
                "rows": spec.expected_validation_rows,
                "sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
            },
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest, root: manifest.pop("schema_version"), "schema"),
        (lambda manifest, root: manifest.update({"extra": True}), "schema"),
        (lambda manifest, root: manifest["files"].pop("train.parquet"), "file entries"),
        (
            lambda manifest, root: manifest["files"].update(
                {"extra.parquet": {"rows": 1, "sha256": "0" * 64}}
            ),
            "file entries",
        ),
        (
            lambda manifest, root: manifest["files"]["train.parquet"].update(
                {"rows": 1}
            ),
            "row-count",
        ),
        (
            lambda manifest, root: (root / "train.parquet").unlink(),
            "missing dataset artifact",
        ),
        (
            lambda manifest, root: (root / "train.parquet").write_bytes(b"tampered"),
            "hash mismatch",
        ),
    ],
)
def test_manifest_validation_rejects_missing_extra_tampered_and_wrong_rows(
    tmp_path, mutation, message
):
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    mutation(manifest, tmp_path)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ContractError, match=message):
        validate_manifest(manifest_path, "svamp")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", 1, "seed"),
        ("data_source", "wrong/source", "data_source"),
        ("route", "lion_actor", "route"),
        ("source_commit", "other", "source_commit"),
        ("calibration_sha256", "0" * 64, "calibration_sha256"),
        ("model_snapshot_sha256", "0" * 64, "model_snapshot_sha256"),
        ("verl_implementation_sha256", "0" * 64, "verl_implementation_sha256"),
        ("actor_learning_rate", 2e-6, "calibrated actor learning rate"),
    ],
)
@pytest.mark.parametrize("dataset", ["svamp", "arc_easy"])
def test_result_integration_binds_routes_seed_source_and_manifest(
    tmp_path, monkeypatch, dataset, field, value, message
):
    source = DATASETS[dataset].source
    manifest_path = _write_manifest(tmp_path, dataset)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    train_sha = hashlib.sha256((tmp_path / "train.parquet").read_bytes()).hexdigest()
    model_root, verl_root, overlay_root, model_sha, verl_sha = (
        _write_implementation_identities(tmp_path, monkeypatch)
    )
    calibration_sha = _write_calibration(
        tmp_path / "calibration.json", dataset, "commit", manifest_sha, train_sha,
        model_sha, verl_sha,
    )
    for route in ROUTES:
        gate = tmp_path / "gates" / route
        gate.mkdir(parents=True)
        _write_provenance(
            gate, "gate", route, dataset, "commit", manifest_sha, calibration_sha,
            model_sha, verl_sha,
        )
        _write_metrics(gate / "metrics.jsonl", source, terminal=1)
    for route in ROUTES:
        root = tmp_path / route
        root.mkdir()
        _write_provenance(
            root, "screen", route, dataset, "commit", manifest_sha, calibration_sha,
            model_sha, verl_sha,
        )
        _write_metrics(root / "metrics.jsonl", source)

    result = build_result(
        tmp_path, dataset, 0, "commit", manifest_path,
        model_root, verl_root, overlay_root,
    )
    assert list(result["routes"]) == list(ROUTES)
    assert result["validation_steps"] == [0, 25, 50]
    assert result["critic_optimizer"] == "AdamW"
    assert result["routes"]["muon_actor"]["actor_optimizer"] == "MuonWithAuxAdamW"
    assert result["routes"]["lion_actor"]["calibration_sha256"] == calibration_sha
    assert set(result["gates"]) == set(ROUTES)
    assert result["model_snapshot_sha256"] == model_sha
    assert result["verl_implementation_sha256"] == verl_sha
    assert result["gates"]["adamw_actor"]["model_snapshot_sha256"] == model_sha
    assert result["routes"]["muon_actor"]["actor_route_description"].endswith(
        "AdamW auxiliaries"
    )
    assert "strict no-Adam" in result["routes"]["lion_actor"]["actor_route_description"]

    provenance_path = tmp_path / "muon_actor" / "run-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance[field] = value
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ContractError, match=message):
        build_result(
            tmp_path, dataset, 0, "commit", manifest_path,
            model_root, verl_root, overlay_root,
        )


def test_identity_artifacts_recompute_bytes_and_reject_arbitrary_strings(
    tmp_path, monkeypatch
):
    model_root, verl_root, overlay_root, model_sha, verl_sha = (
        _write_implementation_identities(tmp_path, monkeypatch)
    )
    assert verify_identity_artifact(
        tmp_path / "model-identity.json", model_identity(model_root)
    ) == model_sha
    assert verify_identity_artifact(
        tmp_path / "verl-identity.json", verl_identity(verl_root, overlay_root)
    ) == verl_sha

    shard = model_root / "model-00002-of-00002.safetensors"
    shard.unlink()
    with pytest.raises(ContractError, match="missing indexed model weight shard"):
        model_identity(model_root)
    shard.write_bytes(b"weights-2")

    (model_root / "model-00002-of-00002.safetensors").write_bytes(b"tampered")
    with pytest.raises(ContractError, match="identity artifact mismatch"):
        verify_identity_artifact(
            tmp_path / "model-identity.json", model_identity(model_root)
        )

    artifact = json.loads((tmp_path / "verl-identity.json").read_text())
    artifact["identity_sha256"] = "0" * 64
    (tmp_path / "verl-identity.json").write_text(json.dumps(artifact))
    with pytest.raises(ContractError, match="identity artifact mismatch"):
        verify_identity_artifact(
            tmp_path / "verl-identity.json", verl_identity(verl_root, overlay_root)
        )

    critical = verl_root / cross_dataset_screen.VERL_CRITICAL_PATHS[0]
    critical.write_text("# modified active trainer\n")
    assert verl_identity(verl_root, overlay_root)["identity_sha256"] != verl_sha


def test_runner_and_launcher_wire_preregistered_protocol():
    runner = (ROOT / "run_cross_dataset_screen.sh").read_text()
    launcher = (
        ROOT
        / "routed-scale-source/examples/ppo_trainer/run_qwen2_5_0_5b_cross_dataset_screen.sh"
    ).read_text()

    first_loop = runner.index("for route in adamw_actor muon_actor lion_actor")
    second_loop = runner.index("for route in adamw_actor muon_actor lion_actor", first_loop + 1)
    gate_loop = runner[first_loop:second_loop]
    screen_loop = runner[second_loop:]
    assert 'run_route gate "$route"' in gate_loop
    assert '--gate-route "$route"' in gate_loop
    assert 'run_route gate "$route" "$run_root/gates/$route" || finish $?' in gate_loop
    assert '--gate-route "$route" || finish $?' in gate_loop
    assert "run_route screen" not in gate_loop
    assert 'run_route screen "$route"' in screen_loop
    assert "run_route gate" not in screen_loop
    assert "calibrate_lion_actor_lr.py" in runner
    assert '--train-file "$data_root/train.parquet"' in runner
    assert 'chmod 0444 "$calibration"' in runner
    assert "torch.cuda.is_available()" in runner
    assert "model numerical compute on CPU is forbidden" in runner
    assert '[[ "$PHASE" == gate && "$ROUTE" != adamw_actor ]]' not in launcher
    assert 'trainer.total_training_steps="$TOTAL_STEPS"' in launcher
    assert "trainer.val_before_train=True" in launcher
    assert "trainer.test_freq=25" in launcher
    assert "critic.optim.optimizer=AdamW" in launcher
    assert "critic.optim.optimizer_impl=torch.optim" in launcher
    assert "algorithm.use_kl_in_reward=True" in launcher
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu" in launcher
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu" in launcher
    assert "custom_reward_function.name=compute_score" in launcher
    assert "hidden_attention_mlp_matrices_muon;all_auxiliaries_adamw" in launcher
    assert "all_actor_parameters_lion_no_adam" in launcher
    assert "CALIBRATION_SHA256" in launcher
    assert "verify_identity_artifact" in launcher
    assert "model_identity(model_root)" in launcher
    assert "verl_identity(verl_root, overlay_root)" in launcher


def test_compat_shell_parses_and_hash_capture_uses_artifact_files():
    subprocess.run(["bash", "-n", str(ROOT / "cloud_verl_compat.sh")], check=True)
    runner = (ROOT / "run_cross_dataset_screen.sh").read_text()
    assert 'manifest_hash_file="$preflight_root/manifest.sha256"' in runner
    assert 'hash_path.write_text(manifest_sha256 + "\\n")' in runner
    assert 'manifest_sha256=$(<"$manifest_hash_file")' in runner
    assert 'rm -f "$manifest_hash_file"' in runner
    assert 'write_identity_artifact(run_root / "model-identity.json", model)' in runner
    assert 'json.loads(model_path.read_text())' in runner
    assert 'identity_values_file="$run_root/identity-values.txt"' in runner
    assert 'values_path.write_text(' in runner
    assert 'read -r model_identity_sha256 verl_identity_sha256 extra <"$identity_values_file"' in runner
    assert 'calibration_values_file="$run_root/calibration-values.txt"' in runner
    assert 'read -r muon_lr lion_lr calibration_sha256 extra <"$calibration_values_file"' in runner
    assert 'identity_values=$("$venv_python"' not in runner
    assert 'calibration_values=$("$venv_python"' not in runner
    assert "contextlib.redirect_stdout" not in runner

    direct_runner = (ROOT / "run_cross_dataset_direct_transfer.sh").read_text()
    assert 'identity_values_file="$run_root/identity-values.txt"' in direct_runner
    assert "values_path.write_text(" in direct_runner
    assert 'read -r model_identity_sha256 verl_identity_sha256 extra <"$identity_values_file"' in direct_runner
    assert 'read -r model_identity_sha256 verl_identity_sha256 < <(' not in direct_runner

    launcher = (
        ROOT
        / "routed-scale-source/examples/ppo_trainer/run_qwen2_5_0_5b_cross_dataset_screen.sh"
    ).read_text()
    assert "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in launcher
    assert "+critic.model.override_config.attn_implementation=sdpa" in launcher
    assert 'GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}' in launcher
    assert 'actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"' in launcher
    assert '"rollout_gpu_memory_utilization": float(gpu_memory_utilization)' in launcher
