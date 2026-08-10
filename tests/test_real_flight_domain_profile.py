from __future__ import annotations

import json
from pathlib import Path


PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/flapping_bot/flapping_bot/config/real_flight_domain_profile_v1.json"
)


def test_real_flight_domain_profile_is_evidence_only_and_sealed() -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

    assert profile["schema_version"] == "pure_rl_real_flight_domain_candidate_v1"
    assert profile["status"] == "candidate_not_promoted"
    assert profile["parameter_source_partition"] == "train"
    assert profile["validation_role"] == "coverage_report_only"
    assert profile["test_loaded"] is False
    assert profile["sim_real_reference"]["acceptance_thresholds_set"] is False
    assert profile["provenance"]["test_loaded"] is False
    assert profile["provenance"]["canonical_log_counts"] == {"train": 20, "validation": 5}
    assert "not candidate range source" in profile["provenance"]["external_reference_role"]


def test_real_flight_domain_profile_has_explicit_units_and_no_local_paths() -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(profile, sort_keys=True)

    assert profile["units"]["true_airspeed"] == "m/s"
    assert profile["units"]["wind_components"] == "m/s"
    assert profile["units"]["raw_pitot_update_interval"] == "s"
    assert "/home/" not in serialized
    assert "/tmp/" not in serialized
    assert "raw_logs" not in profile["provenance"]
