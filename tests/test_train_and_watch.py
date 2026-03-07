from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "flapping_rl" / "train_and_watch.py"
SPEC = importlib.util.spec_from_file_location("train_and_watch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
train_and_watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_and_watch)


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "case"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_latest_checkpoint_uses_numeric_sort(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_2.pt").write_text("")
    (run_dir / "model_10.pt").write_text("")

    latest = train_and_watch._latest_checkpoint(run_dir)

    assert latest == (run_dir / "model_10.pt").resolve()


def test_needs_final_eval_when_latest_suite_row_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)
    model_0 = (run_dir / "model_0.pt")
    model_1 = (run_dir / "model_1.pt")
    model_0.write_text("")
    model_1.write_text("")
    _write_summary(
        eval_dir / "summary.csv",
        [
            {"checkpoint": str(model_0.resolve()), "case": "suite"},
            {"checkpoint": str(model_1.resolve()), "case": "calm"},
        ],
    )

    assert train_and_watch._needs_final_eval(run_dir) is True


def test_needs_final_eval_false_when_latest_suite_row_exists(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)
    model_0 = (run_dir / "model_0.pt")
    model_1 = (run_dir / "model_1.pt")
    model_0.write_text("")
    model_1.write_text("")
    _write_summary(
        eval_dir / "summary.csv",
        [
            {"checkpoint": str(model_0.resolve()), "case": "suite"},
            {"checkpoint": str(model_1.resolve()), "case": "suite"},
        ],
    )

    assert train_and_watch._needs_final_eval(run_dir) is False
