from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "flapping_rl" / "eval_path_tracking_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("eval_path_tracking_checkpoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
eval_path_tracking_checkpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_path_tracking_checkpoint)


def test_checkpoint_index_from_name_accepts_standard_rsl_rl_checkpoint() -> None:
    assert eval_path_tracking_checkpoint._checkpoint_index_from_name("model_20.pt") == 20


def test_checkpoint_index_from_name_rejects_non_numeric_bc_checkpoint() -> None:
    assert eval_path_tracking_checkpoint._checkpoint_index_from_name("model_bc.pt") == -1
