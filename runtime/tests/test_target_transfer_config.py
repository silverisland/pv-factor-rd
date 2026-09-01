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


def test_example_declares_target_only_transfer_protocol():
    config = load_config(example())
    evaluation = config["evaluation"]
    assert evaluation["source_station_time_policy"] == "all_available"
    assert evaluation["validation"]["strategy"] == "target_history_tail"
    assert evaluation["purge_hours"] == 4
    assert evaluation["require_all_training_stations_in_evaluation"] is False


def test_purge_cannot_be_shorter_than_maximum_horizon():
    config = deepcopy(example())
    config["evaluation"]["purge_hours"] = 3.75
    with pytest.raises(ValueError, match="at least 4"):
        load_config(config)


def test_target_only_evaluation_cannot_require_every_training_station():
    config = deepcopy(example())
    config["evaluation"]["require_all_training_stations_in_evaluation"] = True
    with pytest.raises(ValueError, match="Target-only evaluation"):
        load_config(config)
