"""Continuous-path math and replay tests, independent of Isaac Sim."""

from dataclasses import fields, replace
import importlib.util
import json
import math
from pathlib import Path
import sys
import subprocess

import pytest
import torch

from flapping_bot.path_tracking.continuous_curve import (
    CONTINUOUS_CURVE_CONTRACT as CONTRACT,
    ContinuousCurveSpec,
    build_continuous_curves,
    continuous_curve_diagnostic_grid,
    load_continuous_curves,
    sample_continuous_curve_specs,
    save_continuous_curves,
    screen_continuous_curves,
)

ROOT = Path(__file__).resolve().parents[1]


def _pure_module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "source/flapping_bot/flapping_bot/direct/flapping_bot" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


spatial = _pure_module("pure_rl_spatial_path")
observation = _pure_module("pure_rl_observation")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA tensor device unavailable; native PhysX remains CPU-only"
))])
def test_frame_arc_length_and_analytic_derivatives(dtype, device):
    batch = build_continuous_curves((ContinuousCurveSpec(),), dtype=dtype, device=device)
    assert batch.points_world_m.shape == (1, 1201, 3)
    for field in fields(batch):
        value = getattr(batch, field.name)
        assert value.dtype == dtype and value.device.type == device and torch.isfinite(value).all()
    frame = torch.stack((batch.tangent_world, batch.lateral_normal_world, batch.vertical_normal_world), dim=-1)
    identity = torch.eye(3, device=device, dtype=dtype).expand_as(frame)
    torch.testing.assert_close(frame.transpose(-1, -2) @ frame, identity, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.linalg.det(frame), torch.ones_like(batch.slope_rad), atol=1e-6, rtol=1e-6)
    # Independent finite differences verify the three-dimensional arc convention.
    chord = batch.points_world_m[:, 1:] - batch.points_world_m[:, :-1]
    torch.testing.assert_close(chord.norm(dim=-1), torch.full_like(chord[:, :, 0], 0.25), atol=3e-5, rtol=0)
    torch.testing.assert_close(chord[:, :, 2] / 0.25,
                               torch.sin(batch.slope_rad[:, 1:]), atol=0.001, rtol=0)
    for values, derivative in ((batch.slope_rad, batch.slope_derivative_rad_per_m),
                               (batch.heading_rate_rad_per_m, batch.heading_rate_derivative_rad_per_m2)):
        torch.testing.assert_close((values[:, 2:] - values[:, :-2]) / 0.5, derivative[:, 1:-1], atol=1e-5, rtol=0.003)
    assert batch.slope_rad[0, 0] == batch.heading_rate_rad_per_m[0, 0] == batch.turn_activity[0, 0] == 0


def test_grid_is_frozen_and_all_96_paths_pass_screening():
    specs = continuous_curve_diagnostic_grid()
    assert len(specs) == len({s.curve_id for s in specs}) == 96
    assert specs == continuous_curve_diagnostic_grid()
    reports = screen_continuous_curves(build_continuous_curves(specs, dtype=torch.float64))
    assert all(r["screen_passed"] for r in reports)
    assert all(r["qualification"] == "sampled_kinematic_screen_only" for r in reports)
    assert CONTRACT.completion_progress_m / CONTRACT.episode_duration_s == 7.0


def test_recipe_save_load_replays_exactly_and_does_not_trust_stored_screen(tmp_path):
    specs = continuous_curve_diagnostic_grid()[:3]
    path = tmp_path / "curves.json"
    save_continuous_curves(path, specs)
    assert load_continuous_curves(path) == specs
    for dtype in (torch.float32, torch.float64):
        before = build_continuous_curves(specs, dtype=dtype)
        after = build_continuous_curves(load_continuous_curves(path), dtype=dtype)
        for field in fields(before):
            assert torch.equal(getattr(before, field.name), getattr(after, field.name))
    with pytest.raises(FileExistsError):
        save_continuous_curves(path, specs)
    data = json.loads(path.read_text())
    data["curves"][0]["slope_bias_rad"] = -0.12
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="screen rejected"):
        load_continuous_curves(path)
    data["contract"]["version"] = "unknown"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incompatible"):
        load_continuous_curves(path)


@pytest.mark.parametrize("change", [
    {"heading_rate_amplitude_rad_per_m": 0.04}, {"slope_amplitude_rad": 0.3},
    {"initial_altitude_m": 2.0}, {"slope_bias_rad": -0.12},
    {"heading_wavelength_m": 0.0}, {"slope_phase_rad": float("nan")},
    {"horizontal_mirror": 0}, {"chirp": -1.0},
])
def test_invalid_geometry_and_metadata_fail_closed(change):
    with pytest.raises(ValueError):
        build_continuous_curves((replace(ContinuousCurveSpec(), **change),))


def test_sampling_is_repeatable_and_single_resets_include_both_mirrors():
    def sample():
        generator = torch.Generator().manual_seed(13)
        return tuple(sample_continuous_curve_specs(1, generator=generator)[0] for _ in range(64))
    first, second = sample(), sample()
    assert first == second
    assert {s.horizontal_mirror for s in first} == {s.vertical_mirror for s in first} == {-1, 1}
    build_continuous_curves(first, dtype=torch.float64)


def _query(path, progress, positions=None):
    index = round(progress / 0.25)
    return spatial.query_spatial_path(
        path=path, previous_progress_m=path.points_world_m.new_full((len(path.points_world_m),), progress),
        position_world_m=path.points_world_m[:, index] if positions is None else positions,
        ground_velocity_world_mps=8.3 * path.tangent_world[:, index],
    )


def test_preview_distinguishes_future_geometry_and_is_yaw_equivalent():
    specs = (ContinuousCurveSpec(), replace(ContinuousCurveSpec(), slope_phase_rad=math.pi / 2.0))
    path = spatial.continuous_curves_to_spatial_path(specs, device="cpu", dtype=torch.float64)
    torch.testing.assert_close(path.points_world_m[0, 0], path.points_world_m[1, 0])
    torch.testing.assert_close(path.tangent_world[0, 0], path.tangent_world[1, 0])
    query = _query(path, 0.0)
    assert (query.preview_points_world_m[0] - query.preview_points_world_m[1]).abs().max() / 5.0 > 1e-5
    rotated_specs = tuple(replace(s, initial_heading_rad=math.pi * 0.73) for s in specs)
    rotated = spatial.continuous_curves_to_spatial_path(rotated_specs, device="cpu", dtype=torch.float64)
    for current, recipe in ((path, specs), (rotated, rotated_specs)):
        query = _query(current, 60.0)
        yaw = torch.atan2(query.tangent_world[:, 1], query.tangent_world[:, 0])
        quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), dim=1)
        preview = observation.transform_world_preview_points_to_body(
            preview_points_world_m=query.preview_points_world_m, vehicle_position_world_m=query.reference_position_world_m,
            orientation_world_wxyz=quat,
        ) / 5.0
        canonical = observation.canonicalize_orientation_to_route_heading(
            quat, torch.tensor([s.initial_heading_rad for s in recipe], dtype=torch.float64)
        )
        if current is path:
            expected_preview, expected_canonical = preview, canonical
        else:
            torch.testing.assert_close(preview, expected_preview, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(canonical, expected_canonical, atol=1e-10, rtol=1e-10)


def test_projection_completion_and_partial_row_replacement():
    specs = (ContinuousCurveSpec(), replace(ContinuousCurveSpec(), horizontal_mirror=-1))
    path = spatial.continuous_curves_to_spatial_path(specs, device="cpu", dtype=torch.float64)
    before = path.points_world_m.clone()
    replacement = spatial.continuous_curves_to_spatial_path((replace(specs[0], slope_phase_rad=1.0),), device="cpu", dtype=torch.float64)
    spatial.write_spatial_path_batch_rows_(destination=path, env_ids=torch.tensor([1]), source=replacement)
    assert torch.equal(path.points_world_m[0], before[0])
    assert not torch.equal(path.points_world_m[1], before[1])
    assert not _query(path, 139.75).reached_all_events.any()
    assert _query(path, 140.0).reached_all_events.all()
    assert not path.event_count.any() and (path.task_family_id == 5).all()
    previous = _query(path, 30.0)
    for progress in (30.25, 30.5, 30.75, 31.0):
        index = round(progress / 0.25)
        current = spatial.query_spatial_path(
            path=path, previous_progress_m=previous.progress_m,
            position_world_m=path.points_world_m[:, index],
            ground_velocity_world_mps=8.3 * path.tangent_world[:, index],
        )
        torch.testing.assert_close(current.progress_m - previous.progress_m, torch.full((2,), 0.25, dtype=torch.float64))
        assert (current.preview_points_world_m - previous.preview_points_world_m).norm(dim=-1).max() < 0.3
        previous = current


def test_disabled_adapter_preserves_c3_rng_and_tensors():
    for stage in ("c3a", "c3b", "c3c", "c3joint"):
        args = dict(num_paths=8, stage=stage, device="cpu", dtype=torch.float64)
        a = spatial.sample_spatial_path_batch(**args, generator=torch.Generator().manual_seed(17))
        b = spatial.sample_spatial_path_batch(**args, generator=torch.Generator().manual_seed(17), continuous_curve_specs=None)
        for field in fields(a):
            value = getattr(a, field.name)
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, getattr(b, field.name))


def test_c4_sampler_and_event_override_boundaries():
    kwargs = dict(num_paths=1, device="cpu", dtype=torch.float64)
    batch = spatial.sample_spatial_path_batch(
        **kwargs, stage="c4", generator=torch.Generator().manual_seed(2)
    )
    assert batch.task_family_id.tolist() == [5]
    assert spatial.resolve_spatial_stage("c4").minimum_altitude_m == 4.0
    with pytest.raises(ValueError, match="not C3 event overrides"):
        spatial.sample_spatial_path_batch(**kwargs, stage="c4", evaluation_template_id=torch.tensor([11]))
    with pytest.raises(ValueError, match="require stage c4"):
        spatial.sample_spatial_path_batch(**kwargs, stage="c3a", continuous_curve_specs=(ContinuousCurveSpec(),))
    with pytest.raises(ValueError, match="count"):
        spatial.sample_spatial_path_batch(**kwargs, stage="c4", continuous_curve_specs=())


def test_duplicate_ids_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        save_continuous_curves(tmp_path / "invalid.json", (ContinuousCurveSpec(), ContinuousCurveSpec()))
    assert not (tmp_path / "invalid.json").exists()


def test_screen_rejects_nonlocal_overlap():
    batch = build_continuous_curves((ContinuousCurveSpec(),), dtype=torch.float64)
    points = batch.points_world_m.clone()
    points[:, 160] = points[:, 0]
    report = screen_continuous_curves(replace(batch, points_world_m=points))[0]
    assert not report["screen_passed"]
    assert report["minimum_nonlocal_sample_distance_m"] < 1e-6


def test_standalone_cli_generates_loadable_recipe_and_dense_tables(tmp_path):
    script = ROOT / "scripts/flapping_px4/generate_continuous_curve.py"
    output = tmp_path / "curve.json"
    table = tmp_path / "curve.npz"
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output), "--table-output", str(table), "--count", "1"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(load_continuous_curves(output)) == 1 and table.exists()
    assert "git_commit" in json.loads(output.read_text())["provenance"]
    repeated = subprocess.run(
        [sys.executable, str(script), "--output", str(output)], capture_output=True, text=True,
    )
    assert repeated.returncode != 0 and "overwrite" in repeated.stderr


def test_accepted_sampler_seed_variation_and_exhaustion():
    from flapping_bot.path_tracking.continuous_curve import (
        ContinuousCurveSamplingConfig, sample_accepted_continuous_curve_specs,
    )
    config = ContinuousCurveSamplingConfig()
    generator = torch.Generator().manual_seed(819)
    first = sample_accepted_continuous_curve_specs(12, generator=generator, config=config)
    second = sample_accepted_continuous_curve_specs(12, generator=generator, config=config)
    replay = sample_accepted_continuous_curve_specs(12, generator=torch.Generator().manual_seed(819), config=config)
    assert first == replay and first != second
    assert all(r['screen_passed'] for r in screen_continuous_curves(build_continuous_curves(first)))
    assert len({s.second_harmonic_fraction for s in first}) == 12
    assert len({s.chirp for s in first}) == 12
    with pytest.raises(RuntimeError, match='exhausted 2 attempts'):
        sample_accepted_continuous_curve_specs(1, initial_altitude_m=0,
                                              config=replace(config, max_attempts=2))
    with pytest.raises(ValueError, match='sampling range'):
        sample_accepted_continuous_curve_specs(1, config=replace(config, wavelength_m_range=(30, 80)))


def test_resampling_preserves_already_accepted_rows(monkeypatch):
    import flapping_bot.path_tracking.continuous_curve as module
    original = module.sample_continuous_curve_specs
    calls = []
    def candidates(n, **kwargs):
        rows = original(n, **kwargs)
        calls.append(rows)
        if len(calls) == 1:
            return (replace(rows[0], initial_altitude_m=0), *rows[1:])
        return rows
    monkeypatch.setattr(module, 'sample_continuous_curve_specs', candidates)
    diagnostics = {}
    accepted = module.sample_accepted_continuous_curve_specs(3, generator=torch.Generator().manual_seed(3),
                                                           diagnostics=diagnostics)
    assert accepted[1:] == calls[0][1:]
    assert len(calls) == 2 and len(calls[1]) == 1
    assert len({s.curve_id for s in accepted}) == 3
    assert diagnostics == {'candidate_count': 4, 'rejected_count': 1, 'accepted_count': 3}


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_accepted_single_path_device_dtype(dtype, device):
    from flapping_bot.path_tracking.continuous_curve import sample_accepted_continuous_curve_specs
    specs = sample_accepted_continuous_curve_specs(
        1, device=device, dtype=dtype, generator=torch.Generator(device=device).manual_seed(15))
    batch = build_continuous_curves(specs, device=device, dtype=dtype)
    assert batch.points_world_m.shape == (1, 1201, 3)
    assert batch.points_world_m.dtype == dtype
    assert batch.points_world_m.device.type == device


def test_c4_joint_sampling_has_six_groups_and_preserves_old_distribution():
    joint = spatial.sample_spatial_path_batch(num_paths=384, stage='c4joint', device='cpu', dtype=torch.float32,
                                            generator=torch.Generator().manual_seed(4))
    counts = torch.bincount(joint.task_family_id, minlength=6).float() / 384
    torch.testing.assert_close(counts, torch.tensor(spatial.C4_JOINT_TASK_WEIGHTS), atol=.06, rtol=0)
    c4 = joint.task_family_id == 5
    assert (joint.event_count[c4] == 0).all()
    assert (joint.final_event_progress_m[c4] == 140).all()
    assert (joint.points_world_m[c4, :, 2] >= 4).all()
