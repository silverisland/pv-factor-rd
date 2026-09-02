from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime.multi_station_tabm.config import resolve_horizons
from runtime.multi_station_tabm.data import load_multi_station_data
from runtime.multi_station_tabm.features import build_horizon_frame
from runtime.multi_station_tabm.splits import (
    select_target_evaluation,
    training_splits,
)


def config(mode: str = "endpoint") -> dict:
    return {
        "data": {
            "path": None,
            "columns": {
                "timestamp": "timestamp_win",
                "power_history": "Power",
                "power_future": "Power_predict",
                "future_weather": [
                    "GHI_SOLARGIS_predict",
                    "TEMP_SOLARGIS_predict",
                    "WS_SOLARGIS_predict",
                    "WD_SOLARGIS_predict",
                ],
            },
        },
        "features": {
            "history_length": 96,
            "n_horizons": 16,
            "minutes_per_point": 15,
            "prediction_mode": mode,
            "endpoint_horizon_step": 16,
            "horizons": "all",
        },
        "model": {
            "label_normalization": "none",
            "label_scale_value": 1.0,
            "prediction_clip": [0.0, 1.2],
        },
        "training": {},
        "evaluation": {},
        "output": {},
    }


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_win": [pd.Timestamp("2024-09-10 20:15:00")],
            "station_id": ["mkv82"],
            "source_file": ["mkv82_v1.parquet"],
            "site_capacity": [100.0],
            "site_longitude": [121.5],
            "site_latitude": [31.2],
            "site_timezone": ["Asia/Shanghai"],
            "Power": [np.arange(100, dtype=np.float32)],
            "Power_predict": [np.arange(16, dtype=np.float32) + 100],
            "GHI_SOLARGIS_predict": [np.arange(16, dtype=np.float32) + 10],
            "TEMP_SOLARGIS_predict": [np.arange(16, dtype=np.float32) + 20],
            "WS_SOLARGIS_predict": [np.arange(16, dtype=np.float32) + 30],
            "WD_SOLARGIS_predict": [np.arange(16, dtype=np.float32) + 40],
        }
    )


def test_endpoint_matches_supplied_script_contract():
    frame, features, target = build_horizon_frame(raw_frame(), config(), 16)
    assert features[:4] == [
        "GHI_SOLARGIS_predict_target",
        "TEMP_SOLARGIS_predict_target",
        "WS_SOLARGIS_predict_target",
        "WD_SOLARGIS_predict_target",
    ]
    assert features[4] == "power_lag_96"
    assert features[-3:] == ["power_lag_1", "predict_hour", "predict_month"]
    assert np.isclose(frame.loc[0, "power_lag_96"], 0.04)
    assert np.isclose(frame.loc[0, "power_lag_1"], 0.99)
    assert frame.loc[0, "GHI_SOLARGIS_predict_target"] == 25
    assert np.isclose(frame.loc[0, target], 1.15)
    assert frame.loc[0, "target_power"] == 115
    assert frame.loc[0, "predict_hour"] == 0
    assert frame.loc[0, "predict_month"] == 9
    assert frame.loc[0, "station_id"] == "mkv82"
    assert frame.loc[0, "row_id"].startswith("mkv82__")


def test_curve_reuses_all_scalar_horizons():
    cfg = config("curve")
    assert resolve_horizons(cfg) == list(range(1, 17))
    first, _, target = build_horizon_frame(raw_frame(), cfg, 1)
    assert first.loc[0, target] == 1.0
    assert first.loc[0, "GHI_SOLARGIS_predict_target"] == 10
    assert first.loc[0, "predict_hour"] == 20


def test_calendar_features_use_target_time_across_month_boundary():
    raw = raw_frame().assign(timestamp_win=pd.Timestamp("2024-09-30 21:00:00"))
    frame, _, _ = build_horizon_frame(raw, config(), 16)
    assert frame.loc[0, "target_timestamp"] == pd.Timestamp("2024-10-01 01:00:00")
    assert frame.loc[0, "predict_hour"] == 1
    assert frame.loc[0, "predict_month"] == 10


def test_station_identity_is_metadata_and_row_ids_are_station_aware():
    first = raw_frame()
    second = raw_frame().assign(
        station_id="mkv83", source_file="mkv83_v1.parquet"
    )
    raw = pd.concat([first, second], ignore_index=True)
    loaded = load_multi_station_data(raw, config(), require_target=True)
    frame, features, _ = build_horizon_frame(loaded, config(), 16)
    assert frame["station_id"].tolist() == ["mkv82", "mkv83"]
    assert frame["row_id"].nunique() == 2
    assert "station_id" not in features


def target_transfer_config() -> dict:
    return {
        "split": {
            "train_start": "2024-12-29 00:00:00",
            "train_end": "2024-12-29 23:59:59",
            "validation_start": "2024-12-30 00:00:00",
            "validation_end": "2024-12-31 23:59:59",
            "test_start": "2025-01-01 00:00:00",
            "test_end": "2025-01-02 23:59:59",
            "train_stations": ["source", "target"],
            "validation_stations": ["target"],
            "test_stations": ["target"],
        },
    }


def transfer_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-03 00:00",
                    "2024-12-29 11:00",
                    "2024-12-29 13:00",
                    "2024-12-30 08:00",
                    "2024-12-31 18:00",
                    "2025-01-01 00:00",
                ]
            ),
            "target_timestamp": pd.to_datetime(
                [
                    "2025-01-03 04:00",
                    "2024-12-29 15:00",
                    "2024-12-29 17:00",
                    "2024-12-30 12:00",
                    "2024-12-31 22:00",
                    "2025-01-01 04:00",
                ]
            ),
            "station_id": ["source", "target", "target", "target", "target", "target"],
            "row_id": [
                "source-future",
                "target-train",
                "target-train-late",
                "target-val",
                "target-val-late",
                "target-eval",
            ],
        }
    )
    return frame


def test_explicit_train_and_validation_use_origin_time_and_station_lists():
    train, validation = training_splits(transfer_frame(), target_transfer_config())
    assert set(train["row_id"]) == {
        "target-train",
        "target-train-late",
    }
    assert validation["row_id"].tolist() == ["target-val", "target-val-late"]
    assert set(validation["station_id"]) == {"target"}


def test_test_partition_filters_station_and_uses_origin_timestamp():
    selected = select_target_evaluation(
        transfer_frame(), target_transfer_config(), "final_test"
    )
    assert selected["row_id"].tolist() == ["target-eval"]
    assert set(selected["station_id"]) == {"target"}


def test_target_timestamp_does_not_change_partition_membership():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-12-29 19:45", "2024-12-29 20:00", "2024-12-31 20:00"]
            ),
            "target_timestamp": pd.to_datetime(
                ["2024-12-29 23:45", "2024-12-30 00:00", "2025-01-01 00:00"]
            ),
            "station_id": ["target", "target", "target"],
            "row_id": ["last-train", "first-validation", "first-evaluation"],
        }
    )
    source = frame.iloc[[0]].assign(
        station_id="source", row_id="source-row"
    )
    frame = pd.concat([source, frame], ignore_index=True)
    train, validation = training_splits(frame, target_transfer_config())
    assert set(train["row_id"]) == {
        "source-row", "last-train", "first-validation"
    }
    assert set(validation["row_id"]) == {"first-evaluation"}
    # The origin is still 2024-12-31, so a +4 h target on 2025-01-01 is not
    # part of the reference baseline's test partition.
    with pytest.raises(ValueError, match="final_test"):
        select_target_evaluation(frame, target_transfer_config(), "final_test")


def test_disjoint_training_and_test_station_roles_are_supported():
    cfg = target_transfer_config()
    cfg["split"]["train_stations"] = ["source"]
    cfg["split"]["train_end"] = "2025-01-03 23:59:59"
    train, validation = training_splits(transfer_frame(), cfg)
    assert train["row_id"].tolist() == ["source-future"]
    assert validation["row_id"].tolist() == ["target-val", "target-val-late"]
    selected = select_target_evaluation(transfer_frame(), cfg, "final_test")
    assert selected["row_id"].tolist() == ["target-eval"]


def test_multiple_test_stations_are_validated_and_evaluated_together():
    cfg = target_transfer_config()
    cfg["split"]["train_stations"] = ["target", "target-2"]
    cfg["split"]["validation_stations"] = ["target", "target-2"]
    cfg["split"]["test_stations"] = ["target", "target-2"]
    second = transfer_frame()
    second = second[second["station_id"].eq("target")].copy()
    second["station_id"] = "target-2"
    second["row_id"] = "target-2__" + second["row_id"]
    frame = pd.concat([transfer_frame(), second], ignore_index=True)
    train, validation = training_splits(frame, cfg)
    assert set(train["station_id"]) == {"target", "target-2"}
    assert set(validation["station_id"]) == {"target", "target-2"}
    selected = select_target_evaluation(frame, cfg, "final_test")
    assert set(selected["station_id"]) == {"target", "target-2"}
