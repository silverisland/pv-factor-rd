from __future__ import annotations

import numpy as np
import pandas as pd

from runtime.multi_station_tabm.config import resolve_horizons
from runtime.multi_station_tabm.data import load_multi_station_data
from runtime.multi_station_tabm.features import build_horizon_frame
from runtime.multi_station_tabm.splits import (
    seasonal_train_validation_split,
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
            "label_scale_value": 500.0,
            "prediction_clip": [0.0, 465.0],
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
    assert features[4] == "Power_lag_96"
    assert features[-3:] == ["Power_lag_1", "predict_hour", "predict_month"]
    assert frame.loc[0, "Power_lag_96"] == 4
    assert frame.loc[0, "Power_lag_1"] == 99
    assert frame.loc[0, "GHI_SOLARGIS_predict_target"] == 25
    assert frame.loc[0, target] == 115
    assert frame.loc[0, "predict_hour"] == 0
    assert frame.loc[0, "predict_month"] == 9
    assert frame.loc[0, "station_id"] == "mkv82"
    assert frame.loc[0, "row_id"].startswith("mkv82__")


def test_curve_reuses_all_scalar_horizons():
    cfg = config("curve")
    assert resolve_horizons(cfg) == list(range(1, 17))
    first, _, target = build_horizon_frame(raw_frame(), cfg, 1)
    assert first.loc[0, target] == 100
    assert first.loc[0, "GHI_SOLARGIS_predict_target"] == 10
    assert first.loc[0, "predict_hour"] == 20


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


def test_last_five_days_per_month_validation_rule():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-09-25", "2024-09-26", "2024-09-30"]),
            "value": [1, 2, 3],
        }
    )
    train, validation = seasonal_train_validation_split(frame, 5)
    assert train["value"].tolist() == [1]
    assert validation["value"].tolist() == [2, 3]


def test_all_stations_share_time_boundary_and_validation_coverage():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-09-20", "2024-09-28", "2024-09-20", "2024-09-28"]
            ),
            "station_id": ["s1", "s1", "s2", "s2"],
        }
    )
    cfg = {
        "evaluation": {
            "periods": {
                "development": {"start": "2024-09-01", "end": "2024-09-30"}
            },
            "validation_last_days_per_month": 5,
            "require_same_stations_in_development_splits": True,
        }
    }
    train, validation = training_splits(frame, cfg)
    assert set(train["station_id"]) == {"s1", "s2"}
    assert set(validation["station_id"]) == {"s1", "s2"}
    incomplete = frame[~((frame["station_id"] == "s2") & (frame["timestamp"].dt.day == 28))]
    try:
        training_splits(incomplete, cfg)
    except ValueError as error:
        assert "station coverage differs" in str(error)
    else:
        raise AssertionError("Expected station coverage mismatch to be rejected")
