from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from runtime.multi_station_tabm.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def example() -> dict:
    return yaml.safe_load(
        (ROOT / "runtime" / "config.example.yaml").read_text(encoding="utf-8")
    )


def test_example_declares_configured_station_transfer_protocol():
    config = load_config(example())
    evaluation = config["evaluation"]
    assert evaluation["training_stations"] is None
    assert evaluation["test_stations"] == ["replace_with_test_station"]
    assert evaluation["source_station_time_policy"] == "all_available"
    assert evaluation["validation"]["strategy"] == "target_history_tail"
    assert evaluation["purge_hours"] == 4
    assert evaluation["require_all_training_stations_in_evaluation"] is False
    assert evaluation["primary_metric"] == "mean_monthly_capacity_score"
    assert evaluation["early_stopping_prediction"] == "unclipped_ratio"
    assert config["model"]["label_normalization"] == "none"
    assert config["model"]["prediction_clip"] == [0.0, 1.2]


def test_purge_cannot_be_shorter_than_maximum_horizon():
    config = deepcopy(example())
    config["evaluation"]["purge_hours"] = 3.75
    with pytest.raises(ValueError, match="at least 4"):
        load_config(config)


def test_test_station_evaluation_cannot_require_every_training_station():
    config = deepcopy(example())
    config["evaluation"]["require_all_training_stations_in_evaluation"] = True
    with pytest.raises(ValueError, match="Test-station-only evaluation"):
        load_config(config)


def test_explicit_target_validation_range_is_supported_and_checked():
    config = deepcopy(example())
    config["evaluation"]["validation"] = {
        "strategy": "target_history_range",
        "start": "2024-11-01",
        "end": "2024-11-30",
    }
    loaded = load_config(config)
    assert loaded["evaluation"]["validation"]["strategy"] == "target_history_range"

    config["evaluation"]["validation"] = {
        "strategy": "target_history_range",
        "start": "2024-11-30",
    }
    with pytest.raises(ValueError, match="requires start and end"):
        load_config(config)


def test_explicit_station_roles_allow_overlap_or_held_out_tests():
    config = deepcopy(example())
    config["evaluation"]["training_stations"] = ["source-a", "target"]
    config["evaluation"]["test_stations"] = ["target"]
    loaded = load_config(config)
    assert loaded["evaluation"]["training_stations"] == ["source-a", "target"]

    config["evaluation"]["training_stations"] = ["source-a"]
    with pytest.raises(ValueError, match="set reject_unseen_stations=false"):
        load_config(config)
    config["evaluation"]["reject_unseen_stations"] = False
    assert load_config(config)["evaluation"]["test_stations"] == ["target"]


def test_station_role_lists_reject_duplicates_and_legacy_conflicts():
    config = deepcopy(example())
    config["evaluation"]["test_stations"] = ["target", "target"]
    with pytest.raises(ValueError, match="duplicate"):
        load_config(config)

    config = deepcopy(example())
    config["evaluation"]["target_station"] = "legacy-target"
    with pytest.raises(ValueError, match="must be included"):
        load_config(config)
