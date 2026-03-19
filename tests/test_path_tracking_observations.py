from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "direct"
    / "flapping_bot"
    / "path_tracking_env.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("path_tracking_env_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preview_observation_has_five_xyz_points():
    module = _load_module()
    query = {
        "preview_points_body_xyz": torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
        "lateral_error_m": torch.tensor([0.2]),
        "height_error_m": torch.tensor([0.5]),
        "curvature_m_inv": torch.tensor([0.01]),
        "progress_s": torch.tensor([3.0]),
    }
    obs = module._build_preview_observation(query)
    assert obs.shape[-1] >= 15
