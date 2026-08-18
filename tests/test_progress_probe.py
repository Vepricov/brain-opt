import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_progress_probe_accepts_explicit_route(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    metrics_root = run_root / "rms_matched_momentum_actor" / "output"
    metrics_root.mkdir(parents=True)
    (metrics_root / "metrics.jsonl").write_text(
        json.dumps({"step": 7, "data": {"actor/ppo_kl": 0.01}}) + "\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "probe_lion_progress.py"),
            "--run-root",
            str(run_root),
            "--route",
            "rms_matched_momentum_actor",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    snapshot = json.loads(completed.stdout.removeprefix("RL_MUON_PROGRESS "))
    assert list(snapshot["routes"]) == ["rms_matched_momentum_actor"]
    assert snapshot["routes"]["rms_matched_momentum_actor"]["step"] == 7
