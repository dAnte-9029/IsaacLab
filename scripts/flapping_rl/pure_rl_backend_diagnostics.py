"""Pure helpers for comparing CPU-native and GPU-implicit PureRL rollouts."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


FAILURE_LABELS = (
    "runtime_or_nonfinite",
    "wing_drive_tracking",
    "aero_load_shift",
    "policy_feedback_shift",
    "actuator_or_governor_limit",
    "flight_state_instability",
    "path_tracking_only",
    "unattributed",
)

_WING_TRACKING_DIAGNOSTIC_DEG = 1.0
_WRENCH_RELATIVE_RMS_DIAGNOSTIC = 0.25
_FREQUENCY_LIMIT_GATE = 0.25
_TAIL_LIMIT_GATE = 0.10

DRIVE_GATE_THRESHOLDS = {
    "steady_rms_tracking_error_deg": 0.30,
    "steady_p99_abs_tracking_error_deg": 0.60,
    "full_max_abs_tracking_error_deg": 2.0,
    "maximum_amplitude_error_percent": 2.0,
    "maximum_phase_time_error_ms": 1000.0 / 480.0,
    "steady_wing_sync_rms_deg": 0.10,
}

PAIRED_PLANT_GATE_THRESHOLDS = {
    "force_integrated_vector_error_percent": 5.0,
    "moment_integrated_vector_error_percent": 10.0,
    "force_waveform_relative_rms_percent": 10.0,
    "moment_waveform_relative_rms_percent": 10.0,
    "force_waveform_relative_peak_percent": 15.0,
    "moment_waveform_relative_peak_percent": 15.0,
    "short_horizon_max_position_delta_m": 0.25,
    "short_horizon_max_linear_velocity_delta_mps": 0.50,
    "short_horizon_max_attitude_delta_deg": 5.0,
    "short_horizon_max_angular_velocity_delta_rad_s": 1.0,
    "maximum_abs_action_delta": 1.0e-6,
    "maximum_abs_requested_frequency_delta_hz": 1.0e-5,
    "maximum_abs_applied_frequency_delta_hz": 1.0e-5,
}

TAIL_SERVO_GATE_THRESHOLDS = {
    "actual_angle_max_delta_deg": 0.50,
    "actual_angle_rms_delta_deg": 0.25,
    "steady_actual_angle_max_delta_deg": 0.25,
    "rise_time_max_delta_s": 1.0 / 60.0,
    "settling_time_max_delta_s": 1.0 / 60.0,
    "tail_moment_waveform_rms_percent": 10.0,
    "tail_moment_waveform_peak_percent": 15.0,
}

_TAIL_SURFACE_NAMES = ("rudder", "left_elevon", "right_elevon")


def _fit_fundamental_sine(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    frequency_hz: float,
) -> tuple[float, float]:
    """Return amplitude and phase for a least-squares fundamental sine."""

    omega_time = 2.0 * math.pi * float(frequency_hz) * time_s
    design = np.column_stack((np.ones_like(time_s), np.sin(omega_time), np.cos(omega_time)))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    sine_coefficient = float(coefficients[1])
    cosine_coefficient = float(coefficients[2])
    return (
        math.hypot(sine_coefficient, cosine_coefficient),
        math.atan2(cosine_coefficient, sine_coefficient),
    )


def summarize_drive_gate_trace(
    *,
    time_s: np.ndarray,
    target_position_rad: np.ndarray,
    actual_position_rad: np.ndarray,
    frequency_hz: float,
    steady_start_s: float = 0.20,
) -> dict[str, object]:
    """Summarize one 480 Hz, two-wing fixed-root drive trace."""

    time = np.asarray(time_s, dtype=np.float64)
    target = np.asarray(target_position_rad, dtype=np.float64)
    actual = np.asarray(actual_position_rad, dtype=np.float64)
    if time.ndim != 1 or target.shape != time.shape or actual.shape != (time.size, 2):
        raise ValueError("drive trace requires time/target shape (N,) and actual wing shape (N, 2)")
    if time.size == 0 or not np.all(np.diff(time) > 0.0):
        raise ValueError("drive trace time must be non-empty and strictly increasing")
    if frequency_hz <= 0.0:
        raise ValueError("drive frequency must be positive")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(target)) or not np.all(np.isfinite(actual)):
        raise ValueError("drive trace must be finite")
    steady = time >= float(steady_start_s)
    if not np.any(steady):
        raise ValueError("drive trace has no samples in the steady window")

    tracking_error = actual - target[:, None]
    steady_error = tracking_error[steady]
    target_amplitude, target_phase = _fit_fundamental_sine(
        time[steady],
        target[steady],
        frequency_hz=float(frequency_hz),
    )
    if target_amplitude <= np.finfo(np.float64).eps:
        raise ValueError("drive target has zero fitted amplitude")

    amplitude_error_percent: list[float] = []
    phase_time_error_ms: list[float] = []
    for wing_index in range(2):
        actual_amplitude, actual_phase = _fit_fundamental_sine(
            time[steady],
            actual[steady, wing_index],
            frequency_hz=float(frequency_hz),
        )
        amplitude_error_percent.append(
            100.0 * abs(actual_amplitude - target_amplitude) / target_amplitude
        )
        phase_delta = (actual_phase - target_phase + math.pi) % (2.0 * math.pi) - math.pi
        phase_time_error_ms.append(
            1000.0 * abs(phase_delta) / (2.0 * math.pi * float(frequency_hz))
        )

    metrics = {
        "full_max_abs_tracking_error_deg": math.degrees(float(np.max(np.abs(tracking_error)))),
        "steady_rms_tracking_error_deg": math.degrees(
            float(np.sqrt(np.mean(np.square(steady_error))))
        ),
        "steady_p99_abs_tracking_error_deg": math.degrees(
            float(np.quantile(np.abs(steady_error), 0.99))
        ),
        "maximum_amplitude_error_percent": max(amplitude_error_percent),
        "maximum_phase_time_error_ms": max(phase_time_error_ms),
        "steady_wing_sync_rms_deg": math.degrees(
            float(np.sqrt(np.mean(np.square(actual[steady, 0] - actual[steady, 1]))))
        ),
    }
    return {
        "frequency_hz": float(frequency_hz),
        "sample_rate_hz": 480.0,
        "duration_s": float(time[-1]),
        "steady_start_s": float(steady_start_s),
        "sample_count": int(time.size),
        "steady_sample_count": int(np.count_nonzero(steady)),
        "finite": True,
        **metrics,
        "thresholds": dict(DRIVE_GATE_THRESHOLDS),
        "gate_passed": all(
            float(metrics[name]) <= float(limit)
            for name, limit in DRIVE_GATE_THRESHOLDS.items()
        ),
    }


def _integrated_vector_error_percent(
    *,
    time_s: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    delta_time = np.diff(time_s)
    delta = candidate - reference
    integrated_delta = np.sum(
        0.5 * (delta[:-1] + delta[1:]) * delta_time[:, None],
        axis=0,
    )
    reference_norm = np.linalg.norm(reference, axis=1)
    reference_exposure = float(
        np.sum(0.5 * (reference_norm[:-1] + reference_norm[1:]) * delta_time)
    )
    if reference_exposure <= np.finfo(np.float64).eps:
        raise ValueError("paired plant reference has zero integrated magnitude")
    return 100.0 * float(np.linalg.norm(integrated_delta)) / reference_exposure


def _relative_waveform_metrics_percent(
    *,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float]:
    delta_norm = np.linalg.norm(candidate - reference, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    rms_reference = float(np.sqrt(np.mean(np.square(reference_norm))))
    peak_reference = float(np.max(reference_norm))
    if rms_reference <= np.finfo(np.float64).eps or peak_reference <= np.finfo(np.float64).eps:
        raise ValueError("paired plant reference has zero waveform magnitude")
    return (
        100.0 * float(np.sqrt(np.mean(np.square(delta_norm)))) / rms_reference,
        100.0 * float(np.max(delta_norm)) / peak_reference,
    )


def _tail_response_accepted(response: Mapping[str, object]) -> bool:
    transitions = response["transitions"]
    return bool(
        len(transitions) == 4
        and all(
            transition["rise_time_10_90_s"] is not None
            and transition["settling_time_2pct_s"] is not None
            and float(transition["settling_time_2pct_s"]) <= 0.50
            and float(transition["overshoot_percent"]) <= 5.0
            and abs(float(transition["steady_state_error"])) <= math.radians(0.25)
            for transition in transitions
        )
    )


def _maximum_steady_segment_delta_rad(
    *,
    command_rad: np.ndarray,
    cpu_actual_rad: np.ndarray,
    gpu_actual_rad: np.ndarray,
) -> float:
    transitions = np.flatnonzero(np.abs(np.diff(command_rad)) > 1.0e-10) + 1
    boundaries = np.concatenate(([0], transitions, [command_rad.size]))
    steady_deltas: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        count = max(1, int(math.ceil(0.2 * (int(end) - int(start)))))
        window = slice(int(end) - count, int(end))
        steady_deltas.append(
            abs(float(np.mean(gpu_actual_rad[window]) - np.mean(cpu_actual_rad[window])))
        )
    return max(steady_deltas)


def compare_paired_tail_servo_traces(
    cpu_trace: Mapping[str, np.ndarray],
    gpu_trace: Mapping[str, np.ndarray],
    *,
    aerodynamics_enabled: bool,
) -> dict[str, object]:
    """Compare aligned three-surface CPU and GPU tail-servo step responses."""

    from flapping_bot.analysis.pure_rl_action_step_validation import summarize_step_response

    required = (
        "time_s",
        "tail_command_rad",
        "tail_actual_rad",
        "tail_moment_body_nm",
    )
    missing = [name for name in required if name not in cpu_trace or name not in gpu_trace]
    if missing:
        raise ValueError(f"paired tail-servo trace is missing fields: {missing}")

    cpu_time = np.asarray(cpu_trace["time_s"], dtype=np.float64)
    gpu_time = np.asarray(gpu_trace["time_s"], dtype=np.float64)
    expected_count = int(cpu_time.size)
    if (
        cpu_time.ndim != 1
        or expected_count < 2
        or gpu_time.shape != cpu_time.shape
        or not np.all(np.diff(cpu_time) > 0.0)
        or not np.allclose(cpu_time, gpu_time, rtol=0.0, atol=1.0e-9)
    ):
        raise ValueError("paired tail-servo timestamps must be aligned and strictly increasing")

    cpu_command = np.asarray(cpu_trace["tail_command_rad"], dtype=np.float64)
    gpu_command = np.asarray(gpu_trace["tail_command_rad"], dtype=np.float64)
    cpu_actual = np.asarray(cpu_trace["tail_actual_rad"], dtype=np.float64)
    gpu_actual = np.asarray(gpu_trace["tail_actual_rad"], dtype=np.float64)
    cpu_moment = np.asarray(cpu_trace["tail_moment_body_nm"], dtype=np.float64)
    gpu_moment = np.asarray(gpu_trace["tail_moment_body_nm"], dtype=np.float64)
    if any(
        value.shape != (expected_count, 3)
        for value in (cpu_command, gpu_command, cpu_actual, gpu_actual)
    ) or cpu_moment.shape != (expected_count, 3, 3) or gpu_moment.shape != cpu_moment.shape:
        raise ValueError("paired tail-servo traces require angle shape (N, 3) and moment shape (N, 3, 3)")
    if any(
        not np.all(np.isfinite(value))
        for value in (cpu_time, gpu_time, cpu_command, gpu_command, cpu_actual, gpu_actual, cpu_moment, gpu_moment)
    ):
        raise ValueError("paired tail-servo traces must be finite")
    if not np.allclose(cpu_command, gpu_command, rtol=0.0, atol=1.0e-10):
        raise ValueError("paired tail-servo commands must match exactly")

    failures: list[str] = []
    surfaces: list[dict[str, object]] = []
    for surface_index, surface_name in enumerate(_TAIL_SURFACE_NAMES):
        command = cpu_command[:, surface_index]
        cpu_response = summarize_step_response(cpu_time, command, cpu_actual[:, surface_index])
        gpu_response = summarize_step_response(gpu_time, command, gpu_actual[:, surface_index])
        cpu_transitions = cpu_response["transitions"]
        gpu_transitions = gpu_response["transitions"]
        response_complete = bool(
            len(cpu_transitions) == 4
            and len(gpu_transitions) == 4
            and all(
                cpu_transition[field] is not None and gpu_transition[field] is not None
                for cpu_transition, gpu_transition in zip(
                    cpu_transitions,
                    gpu_transitions,
                    strict=True,
                )
                for field in ("rise_time_10_90_s", "settling_time_2pct_s")
            )
        )
        if response_complete:
            rise_delta_s = max(
                abs(
                    float(cpu_transition["rise_time_10_90_s"])
                    - float(gpu_transition["rise_time_10_90_s"])
                )
                for cpu_transition, gpu_transition in zip(
                    cpu_transitions,
                    gpu_transitions,
                    strict=True,
                )
            )
            settling_delta_s = max(
                abs(
                    float(cpu_transition["settling_time_2pct_s"])
                    - float(gpu_transition["settling_time_2pct_s"])
                )
                for cpu_transition, gpu_transition in zip(
                    cpu_transitions,
                    gpu_transitions,
                    strict=True,
                )
            )
        else:
            rise_delta_s = None
            settling_delta_s = None

        angle_delta = gpu_actual[:, surface_index] - cpu_actual[:, surface_index]
        metrics: dict[str, object] = {
            "name": surface_name,
            "transition_count": min(len(cpu_transitions), len(gpu_transitions)),
            "cpu_response_accepted": _tail_response_accepted(cpu_response),
            "gpu_response_accepted": _tail_response_accepted(gpu_response),
            "response_complete": response_complete,
            "actual_angle_max_delta_deg": math.degrees(float(np.max(np.abs(angle_delta)))),
            "actual_angle_rms_delta_deg": math.degrees(
                float(np.sqrt(np.mean(np.square(angle_delta))))
            ),
            "steady_actual_angle_max_delta_deg": math.degrees(
                _maximum_steady_segment_delta_rad(
                    command_rad=command,
                    cpu_actual_rad=cpu_actual[:, surface_index],
                    gpu_actual_rad=gpu_actual[:, surface_index],
                )
            ),
            "rise_time_max_delta_s": rise_delta_s,
            "settling_time_max_delta_s": settling_delta_s,
            "cpu_response": cpu_response,
            "gpu_response": gpu_response,
        }
        if aerodynamics_enabled:
            moment_rms, moment_peak = _relative_waveform_metrics_percent(
                reference=cpu_moment[:, surface_index, :],
                candidate=gpu_moment[:, surface_index, :],
            )
            metrics["tail_moment_waveform_rms_percent"] = moment_rms
            metrics["tail_moment_waveform_peak_percent"] = moment_peak
        else:
            metrics["tail_moment_waveform_rms_percent"] = None
            metrics["tail_moment_waveform_peak_percent"] = None

        if not bool(metrics["cpu_response_accepted"]):
            failures.append(f"{surface_name}.cpu_response_accepted")
        if not bool(metrics["gpu_response_accepted"]):
            failures.append(f"{surface_name}.gpu_response_accepted")
        if not response_complete:
            failures.append(f"{surface_name}.response_complete")
        for metric_name in (
            "actual_angle_max_delta_deg",
            "actual_angle_rms_delta_deg",
            "steady_actual_angle_max_delta_deg",
            "rise_time_max_delta_s",
            "settling_time_max_delta_s",
        ):
            value = metrics[metric_name]
            if value is not None and float(value) > float(TAIL_SERVO_GATE_THRESHOLDS[metric_name]):
                failures.append(f"{surface_name}.{metric_name}")
        if aerodynamics_enabled:
            for metric_name in (
                "tail_moment_waveform_rms_percent",
                "tail_moment_waveform_peak_percent",
            ):
                if float(metrics[metric_name]) > float(TAIL_SERVO_GATE_THRESHOLDS[metric_name]):
                    failures.append(f"{surface_name}.{metric_name}")
        surfaces.append(metrics)

    command_delta_deg = math.degrees(float(np.max(np.abs(gpu_command - cpu_command))))
    return {
        "sample_count": expected_count,
        "duration_s": float(cpu_time[-1] - cpu_time[0]),
        "surface_count": len(_TAIL_SURFACE_NAMES),
        "aerodynamics_enabled": bool(aerodynamics_enabled),
        "maximum_abs_command_delta_deg": command_delta_deg,
        "thresholds": dict(TAIL_SERVO_GATE_THRESHOLDS),
        "surfaces": surfaces,
        "gate_failures": failures,
        "gate_passed": not failures,
    }


def compare_paired_plant_traces(
    cpu_trace: Mapping[str, np.ndarray],
    gpu_trace: Mapping[str, np.ndarray],
    *,
    short_horizon_s: float = 1.0,
) -> dict[str, object]:
    """Compare CPU closed-loop and GPU replay traces over their common interval."""

    required_fields = (
        "time_s",
        "actions",
        "root_position_m",
        "root_quaternion_wxyz",
        "root_linear_velocity_world_mps",
        "root_angular_velocity_body_rad_s",
        "requested_frequency_hz",
        "applied_frequency_hz",
        "actual_tail_angle_rad",
        "wing_actual_position_rad",
        "wing_tracking_error_rad",
        "total_aero_force_body_n",
        "total_aero_moment_body_nm",
    )
    missing = [
        name
        for name in required_fields
        if name not in cpu_trace or name not in gpu_trace
    ]
    if missing:
        raise ValueError(f"paired plant trace is missing fields: {missing}")
    common_count = min(
        int(np.asarray(cpu_trace["time_s"]).shape[0]),
        int(np.asarray(gpu_trace["time_s"]).shape[0]),
    )
    if common_count < 2:
        raise ValueError("paired plant comparison requires at least two common samples")
    cpu_time = np.asarray(cpu_trace["time_s"], dtype=np.float64)[:common_count]
    gpu_time = np.asarray(gpu_trace["time_s"], dtype=np.float64)[:common_count]
    if not np.all(np.isfinite(cpu_time)) or not np.allclose(cpu_time, gpu_time, rtol=0.0, atol=1.0e-9):
        raise ValueError("paired plant timestamps must align over the common interval")
    if not np.all(np.diff(cpu_time) > 0.0):
        raise ValueError("paired plant timestamps must be strictly increasing")

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in required_fields[1:]:
        cpu_values = np.asarray(cpu_trace[name], dtype=np.float64)[:common_count]
        gpu_values = np.asarray(gpu_trace[name], dtype=np.float64)[:common_count]
        if cpu_values.shape != gpu_values.shape or cpu_values.shape[0] != common_count:
            raise ValueError(f"paired plant field {name} is not aligned")
        if not np.all(np.isfinite(cpu_values)) or not np.all(np.isfinite(gpu_values)):
            raise ValueError(f"paired plant field {name} must be finite")
        arrays[name] = (cpu_values, gpu_values)

    cpu_force, gpu_force = arrays["total_aero_force_body_n"]
    cpu_moment, gpu_moment = arrays["total_aero_moment_body_nm"]
    force_rms, force_peak = _relative_waveform_metrics_percent(
        reference=cpu_force,
        candidate=gpu_force,
    )
    moment_rms, moment_peak = _relative_waveform_metrics_percent(
        reference=cpu_moment,
        candidate=gpu_moment,
    )

    relative_time = cpu_time - cpu_time[0]
    short_mask = relative_time <= float(short_horizon_s) + 1.0e-12
    if not np.any(short_mask):
        raise ValueError("paired plant trace has no short-horizon samples")
    cpu_quaternion, gpu_quaternion = arrays["root_quaternion_wxyz"]
    cpu_quaternion = cpu_quaternion[short_mask]
    gpu_quaternion = gpu_quaternion[short_mask]
    cpu_quaternion /= np.linalg.norm(cpu_quaternion, axis=1, keepdims=True)
    gpu_quaternion /= np.linalg.norm(gpu_quaternion, axis=1, keepdims=True)
    quaternion_dot = np.clip(
        np.abs(np.sum(cpu_quaternion * gpu_quaternion, axis=1)),
        0.0,
        1.0,
    )

    cpu_actions, gpu_actions = arrays["actions"]
    cpu_requested, gpu_requested = arrays["requested_frequency_hz"]
    cpu_applied, gpu_applied = arrays["applied_frequency_hz"]
    cpu_tail, gpu_tail = arrays["actual_tail_angle_rad"]
    cpu_wing, gpu_wing = arrays["wing_actual_position_rad"]
    cpu_wing_error, gpu_wing_error = arrays["wing_tracking_error_rad"]
    cpu_position, gpu_position = arrays["root_position_m"]
    cpu_linear_velocity, gpu_linear_velocity = arrays["root_linear_velocity_world_mps"]
    cpu_angular_velocity, gpu_angular_velocity = arrays["root_angular_velocity_body_rad_s"]

    metrics = {
        "force_integrated_vector_error_percent": _integrated_vector_error_percent(
            time_s=cpu_time,
            reference=cpu_force,
            candidate=gpu_force,
        ),
        "moment_integrated_vector_error_percent": _integrated_vector_error_percent(
            time_s=cpu_time,
            reference=cpu_moment,
            candidate=gpu_moment,
        ),
        "force_waveform_relative_rms_percent": force_rms,
        "moment_waveform_relative_rms_percent": moment_rms,
        "force_waveform_relative_peak_percent": force_peak,
        "moment_waveform_relative_peak_percent": moment_peak,
        "short_horizon_max_position_delta_m": float(
            np.max(np.linalg.norm(gpu_position[short_mask] - cpu_position[short_mask], axis=1))
        ),
        "short_horizon_max_linear_velocity_delta_mps": float(
            np.max(
                np.linalg.norm(
                    gpu_linear_velocity[short_mask] - cpu_linear_velocity[short_mask],
                    axis=1,
                )
            )
        ),
        "short_horizon_max_attitude_delta_deg": math.degrees(
            float(np.max(2.0 * np.arccos(quaternion_dot)))
        ),
        "short_horizon_max_angular_velocity_delta_rad_s": float(
            np.max(
                np.linalg.norm(
                    gpu_angular_velocity[short_mask] - cpu_angular_velocity[short_mask],
                    axis=1,
                )
            )
        ),
        "maximum_abs_action_delta": float(np.max(np.abs(gpu_actions - cpu_actions))),
        "maximum_abs_requested_frequency_delta_hz": float(
            np.max(np.abs(gpu_requested - cpu_requested))
        ),
        "maximum_abs_applied_frequency_delta_hz": float(
            np.max(np.abs(gpu_applied - cpu_applied))
        ),
        "maximum_abs_tail_angle_delta_deg": math.degrees(
            float(np.max(np.abs(gpu_tail - cpu_tail)))
        ),
        "maximum_abs_wing_position_delta_deg": math.degrees(
            float(np.max(np.abs(gpu_wing - cpu_wing)))
        ),
        "maximum_abs_wing_tracking_error_delta_deg": math.degrees(
            float(np.max(np.abs(gpu_wing_error - cpu_wing_error)))
        ),
    }
    common_duration_s = float(cpu_time[-1] - cpu_time[0])
    failures = [
        name
        for name, limit in PAIRED_PLANT_GATE_THRESHOLDS.items()
        if float(metrics[name]) > float(limit)
    ]
    if common_duration_s < float(short_horizon_s):
        failures.append("minimum_common_duration_s")
    return {
        "common_sample_count": common_count,
        "common_duration_s": common_duration_s,
        "short_horizon_s": float(short_horizon_s),
        **metrics,
        "thresholds": dict(PAIRED_PLANT_GATE_THRESHOLDS),
        "gate_failures": failures,
        "gate_passed": not failures,
    }


def validate_case_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    backend_id: str,
) -> tuple[dict[str, object], ...]:
    """Validate per-case rows before cross-backend joins."""

    validated: list[dict[str, object]] = []
    case_ids: set[str] = set()
    for source in rows:
        row = dict(source)
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ValueError("case_id must be non-empty")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)
        if row.get("backend_id") != backend_id:
            raise ValueError(
                f"backend_id mismatch for {case_id}: expected {backend_id!r}, got {row.get('backend_id')!r}"
            )
        for name in ("duration_s", "path_error_m"):
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite for {case_id}")
        validated.append(row)
    return tuple(validated)


def validate_trace_arrays(trace: Mapping[str, np.ndarray]) -> None:
    """Validate one selected-case policy-rate trace."""

    required_fields = {
        "time_s",
        "actions",
        "done",
        "terminated",
        "time_out",
        "physical_state_valid",
    }
    missing_fields = sorted(required_fields.difference(trace))
    if missing_fields:
        raise ValueError(f"trace is missing required fields: {missing_fields}")
    time_s = np.asarray(trace["time_s"])
    actions = np.asarray(trace["actions"])
    if time_s.ndim != 1 or time_s.size == 0:
        raise ValueError("time_s must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must be finite")
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError("actions must have exactly four channels")
    count = int(time_s.shape[0])
    for name, values in trace.items():
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != count:
            raise ValueError(f"trace arrays must be aligned; {name} has shape {array.shape}")
    done = np.asarray(trace["done"], dtype=bool)
    terminated = np.asarray(trace["terminated"], dtype=bool)
    time_out = np.asarray(trace["time_out"], dtype=bool)
    physical_state_valid = np.asarray(trace["physical_state_valid"], dtype=bool)
    if np.any(done | terminated | time_out) or not np.all(physical_state_valid):
        raise ValueError("trace must contain only non-terminal physical states from the active episode")


def _path_error(row: Mapping[str, object]) -> float:
    return float(row.get("path_error_m", 0.0))


def _failure_order(row: Mapping[str, object]) -> tuple[float, float, str]:
    return (float(row.get("duration_s", math.inf)), -_path_error(row), str(row["case_id"]))


def _success_order(row: Mapping[str, object]) -> tuple[float, str]:
    return (-_path_error(row), str(row["case_id"]))


def select_diagnostic_cases(
    rows: Sequence[Mapping[str, object]],
    *,
    maximum_cases: int = 8,
    maximum_successes: int = 2,
) -> tuple[str, ...]:
    """Select bounded, deterministic failure traces with task/cause coverage."""

    if maximum_cases <= 0 or maximum_successes < 0:
        raise ValueError("diagnostic case limits must be nonnegative and maximum_cases must be positive")
    failures = sorted((row for row in rows if not bool(row["success"])), key=_failure_order)
    successes = sorted((row for row in rows if bool(row["success"])), key=_success_order)

    selected: list[Mapping[str, object]] = []
    selected_ids: set[str] = set()

    def add(row: Mapping[str, object]) -> None:
        case_id = str(row["case_id"])
        if case_id not in selected_ids and len(selected) < maximum_cases:
            selected.append(row)
            selected_ids.add(case_id)

    if len(failures) <= maximum_cases:
        for row in failures:
            add(row)
    else:
        covered_causes: set[str] = set()
        for row in failures:
            cause = str(row.get("termination_cause", "unknown"))
            if cause not in covered_causes:
                add(row)
                covered_causes.add(cause)
        covered_tasks = {str(row.get("task", "unknown")) for row in selected}
        for row in failures:
            task = str(row.get("task", "unknown"))
            if task not in covered_tasks:
                add(row)
                covered_tasks.add(task)
        for row in failures:
            add(row)

    success_slots = min(maximum_successes, maximum_cases - len(selected))
    for row in successes[:success_slots]:
        add(row)
    return tuple(str(row["case_id"]) for row in selected)


def _evidence_record(
    *,
    label: str,
    score: float,
    case_row: Mapping[str, object],
    comparison: Mapping[str, object],
    raw_evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "label": label,
        "rank_score": float(score),
        "raw_evidence": dict(raw_evidence),
        "first_event_time_s": float(
            comparison.get("first_event_time_s", case_row.get("duration_s", 0.0))
        ),
        "closed_loop_outcome": "passed" if bool(case_row.get("success", False)) else "failed",
        "replay_outcome": str(comparison.get("replay_outcome", "not_run")),
        "causal_claim": False,
    }


def rank_failure_evidence(
    case_row: Mapping[str, object],
    *,
    comparison: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], ...]:
    """Rank non-causal diagnostic evidence for one failed GPU case."""

    compared = {} if comparison is None else dict(comparison)
    evidence: list[dict[str, object]] = []

    def add(label: str, score: float, raw: Mapping[str, object]) -> None:
        evidence.append(
            _evidence_record(
                label=label,
                score=score,
                case_row=case_row,
                comparison=compared,
                raw_evidence=raw,
            )
        )

    if bool(case_row.get("runtime_failure", False)) or not bool(case_row.get("finite_metrics", True)):
        add(
            "runtime_or_nonfinite",
            100.0,
            {
                "runtime_failure": bool(case_row.get("runtime_failure", False)),
                "finite_metrics": bool(case_row.get("finite_metrics", True)),
            },
        )

    replay_outcome = str(compared.get("replay_outcome", "not_run"))
    if not bool(case_row.get("success", False)) and replay_outcome == "passed":
        add("policy_feedback_shift", 90.0, {"replay_outcome": replay_outcome})

    wing_error_deg = float(compared.get("max_abs_wing_tracking_error_deg", 0.0))
    if replay_outcome == "failed" and wing_error_deg > _WING_TRACKING_DIAGNOSTIC_DEG:
        add(
            "wing_drive_tracking",
            80.0 + min(wing_error_deg, 20.0) / 20.0,
            {"max_abs_wing_tracking_error_deg": wing_error_deg},
        )

    wrench_delta = float(compared.get("wrench_relative_rms_delta", 0.0))
    if replay_outcome == "failed" and wrench_delta > _WRENCH_RELATIVE_RMS_DIAGNOSTIC:
        add(
            "aero_load_shift",
            75.0 + min(wrench_delta, 10.0) / 10.0,
            {"wrench_relative_rms_delta": wrench_delta},
        )

    frequency_limit = float(case_row.get("frequency_limit_fraction", 0.0))
    tail_limit = float(case_row.get("tail_limit_fraction", 0.0))
    if frequency_limit > _FREQUENCY_LIMIT_GATE or tail_limit > _TAIL_LIMIT_GATE:
        add(
            "actuator_or_governor_limit",
            70.0,
            {
                "frequency_limit_fraction": frequency_limit,
                "tail_limit_fraction": tail_limit,
                "frequency_governor_limited_fraction": float(
                    case_row.get("frequency_governor_limited_fraction", 0.0)
                ),
            },
        )

    termination_cause = str(case_row.get("termination_cause", "unknown"))
    max_tilt_deg = float(case_row.get("max_tilt_deg", 0.0))
    if termination_cause == "tilt" or max_tilt_deg >= 75.0:
        add(
            "flight_state_instability",
            65.0,
            {"termination_cause": termination_cause, "max_tilt_deg": max_tilt_deg},
        )

    if termination_cause in {"cross_track", "height", "ground"}:
        add(
            "path_tracking_only",
            60.0,
            {"termination_cause": termination_cause, "path_error_m": _path_error(case_row)},
        )

    if not evidence:
        add("unattributed", 0.0, {"termination_cause": termination_cause})

    return tuple(sorted(evidence, key=lambda item: (-float(item["rank_score"]), str(item["label"]))))


__all__ = [
    "DRIVE_GATE_THRESHOLDS",
    "FAILURE_LABELS",
    "PAIRED_PLANT_GATE_THRESHOLDS",
    "TAIL_SERVO_GATE_THRESHOLDS",
    "compare_paired_tail_servo_traces",
    "compare_paired_plant_traces",
    "rank_failure_evidence",
    "select_diagnostic_cases",
    "summarize_drive_gate_trace",
    "validate_case_rows",
    "validate_trace_arrays",
]
