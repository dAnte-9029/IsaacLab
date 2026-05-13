from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tools" / "plot_polygon_2d.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_polygon_2d", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_closed_polygon_appends_first_point_once() -> None:
    module = _load_module()
    points = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]

    closed = module.closed_polygon(points)

    assert closed == [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.0, 0.0)]


def test_closed_polygon_keeps_existing_closure() -> None:
    module = _load_module()
    points = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.0, 0.0)]

    closed = module.closed_polygon(points)

    assert closed == points


def test_validate_points_rejects_too_few_vertices() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="at least three"):
        module.validate_points([(0.0, 0.0), (1.0, 0.0)])
