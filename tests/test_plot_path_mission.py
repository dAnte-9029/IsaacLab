from __future__ import annotations

import ast
from pathlib import Path


SCRIPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "flapping_px4"
    / "plot_path_mission.py"
)


def _load_module() -> ast.Module:
    return ast.parse(SCRIPT_FILE.read_text(encoding="utf-8"))


def test_plot_path_mission_parser_exposes_ground_track_y_zoom_margin() -> None:
    module = _load_module()
    add_argument_names: set[str] = set()

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            add_argument_names.add(first_arg.value)

    assert "--ground_track_y_zoom_margin_m" in add_argument_names


def test_plot_path_mission_does_not_add_invisible_reference_series_to_altitude_axis() -> None:
    module = _load_module()

    invisible_plot_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "plot"
        and any(
            keyword.arg == "alpha"
            and isinstance(keyword.value, ast.Constant)
            and float(keyword.value.value) == 0.0
            for keyword in node.keywords
        )
    ]

    assert invisible_plot_calls == []
