import hashlib
import json
import math
from pathlib import Path

import pytest

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
    parse_arc_easy_answer,
    parse_svamp_answer,
    validate_gate_metrics,
    validate_held_out,
    validate_screen_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


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
            {"data_source": "cross_dataset/arc_easy"},
        )
        == 1.0
    )
    assert (
        compute_score(
            "cross_dataset/svamp",
            "reason\nFinal answer: 2.00",
            "2",
            {"data_source": "cross_dataset/svamp"},
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
            {"data_source": "cross_dataset/svamp"},
        )


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
    root, phase, route, dataset, commit, manifest_sha, calibration_sha
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
    }
    (root / "run-provenance.json").write_text(json.dumps(provenance))


def _write_calibration(path, dataset, commit, manifest_sha, train_sha):
    adam = {"mean": 0.01, "q95": 0.02, "occupied_states": 8}
    matched = {**adam, "mean_relative_error": 0.0, "q95_relative_error": 0.0}
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
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
        ("actor_learning_rate", 2e-6, "calibrated actor learning rate"),
    ],
)
@pytest.mark.parametrize("dataset", ["svamp", "arc_easy"])
def test_result_integration_binds_routes_seed_source_and_manifest(
    tmp_path, dataset, field, value, message
):
    source = DATASETS[dataset].source
    manifest_path = _write_manifest(tmp_path, dataset)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    train_sha = hashlib.sha256((tmp_path / "train.parquet").read_bytes()).hexdigest()
    calibration_sha = _write_calibration(
        tmp_path / "calibration.json", dataset, "commit", manifest_sha, train_sha
    )
    for route in ROUTES:
        gate = tmp_path / "gates" / route
        gate.mkdir(parents=True)
        _write_provenance(
            gate, "gate", route, dataset, "commit", manifest_sha, calibration_sha
        )
        _write_metrics(gate / "metrics.jsonl", source, terminal=1)
    for route in ROUTES:
        root = tmp_path / route
        root.mkdir()
        _write_provenance(
            root, "screen", route, dataset, "commit", manifest_sha, calibration_sha
        )
        _write_metrics(root / "metrics.jsonl", source)

    result = build_result(tmp_path, dataset, 0, "commit", manifest_path)
    assert list(result["routes"]) == list(ROUTES)
    assert result["validation_steps"] == [0, 25, 50]
    assert result["critic_optimizer"] == "AdamW"
    assert result["routes"]["muon_actor"]["actor_optimizer"] == "MuonWithAuxAdamW"
    assert result["routes"]["lion_actor"]["calibration_sha256"] == calibration_sha
    assert set(result["gates"]) == set(ROUTES)
    assert result["routes"]["muon_actor"]["actor_route_description"].endswith(
        "AdamW auxiliaries"
    )
    assert "strict no-Adam" in result["routes"]["lion_actor"]["actor_route_description"]

    provenance_path = tmp_path / "muon_actor" / "run-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance[field] = value
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ContractError, match=message):
        build_result(tmp_path, dataset, 0, "commit", manifest_path)


def test_runner_and_launcher_wire_preregistered_protocol():
    runner = (ROOT / "run_cross_dataset_screen.sh").read_text()
    launcher = (
        ROOT
        / "routed-scale-source/examples/ppo_trainer/run_qwen2_5_0_5b_cross_dataset_screen.sh"
    ).read_text()

    loop = runner[runner.index("for route in adamw_actor muon_actor lion_actor") :]
    assert loop.index('run_route gate "$route"') < loop.index('--gate-route "$route"')
    assert loop.index('--gate-route "$route"') < loop.index('run_route screen "$route"')
    assert 'run_route gate "$route" "$run_root/gates/$route" || finish $?' in loop
    assert '--gate-route "$route" || finish $?' in loop
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
