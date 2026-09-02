from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from runtime.multi_station_tabm.config import (
    load_config,
    resolve_validation_stations,
)


ROOT = Path(__file__).resolve().parents[2]


def example() -> dict:
    return yaml.safe_load(
        (ROOT / "runtime" / "config.example.yaml").read_text(encoding="utf-8")
    )


def test_example_declares_reference_baseline_split_and_metrics():
    config = load_config(example())
    split = config["split"]
    assert split["train_stations"] is None
    assert split["validation_stations"] == ["replace_with_validation_station"]
    assert split["test_stations"] == ["replace_with_test_station"]
    assert config["evaluation"]["primary_metric"] == "mean_monthly_capacity_score"
    assert config["evaluation"]["early_stopping_prediction"] == "unclipped_ratio"
    assert config["model"]["label_normalization"] == "none"
    assert config["model"]["prediction_clip"] == [0.0, 1.2]


def test_all_three_split_periods_are_required_and_ordered():
    config = deepcopy(example())
    del config["split"]["validation_end"]
    with pytest.raises(ValueError, match="validation_start and validation_end"):
        load_config(config)

    config = deepcopy(example())
    config["split"]["test_start"] = "2026-01-02"
    config["split"]["test_end"] = "2026-01-01"
    with pytest.raises(ValueError, match="test_start must not exceed test_end"):
        load_config(config)


def test_validation_stations_default_to_training_stations_like_reference():
    config = deepcopy(example())
    config["split"]["train_stations"] = ["source-a", "source-b"]
    config["split"]["validation_stations"] = None
    loaded = load_config(config)
    assert resolve_validation_stations(loaded) == ["source-a", "source-b"]


def test_station_lists_allow_overlap_or_held_out_test_stations():
    config = deepcopy(example())
    config["split"]["train_stations"] = ["source-a"]
    config["split"]["validation_stations"] = ["source-a"]
    config["split"]["test_stations"] = ["target"]
    assert load_config(config)["split"]["test_stations"] == ["target"]


def test_station_lists_reject_duplicates():
    config = deepcopy(example())
    config["split"]["test_stations"] = ["target", "target"]
    with pytest.raises(ValueError, match="duplicate"):
        load_config(config)
