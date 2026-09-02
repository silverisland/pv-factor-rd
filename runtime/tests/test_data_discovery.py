from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime.multi_station_tabm.data import load_multi_station_data
from runtime.multi_station_tabm.features import (
    build_horizon_frame,
    build_multi_station_feature_frames,
)
from runtime.multi_station_tabm import features as feature_module


def config(root) -> dict:
    return {
        "data": {
            "parquet_root": str(root),
            "parquet_glob": "station=*.parquet",
            "station_id_column": "station",
            "site_metadata": {"aliases": {}, "overrides": {}},
            "columns": {
                "timestamp": "timestamp_win",
                "power_history": "Power",
                "ghi_history": "GHI_real",
                "power_future": "Power_predict",
                "future_weather": ["GHI_SOLARGIS_predict"],
            },
        },
        "features": {"history_length": 96},
    }


def station_frame(stations: list[str]) -> pd.DataFrame:
    rows = len(stations)
    return pd.DataFrame(
        {
            "timestamp_win": pd.date_range("2024-09-01", periods=rows, freq="15min"),
            "station": stations,
            "Power": [np.arange(96, dtype=np.float32) for _ in stations],
            "GHI_real": [np.arange(96, dtype=np.float32) for _ in stations],
            "Power_predict": [np.arange(16, dtype=np.float32) for _ in stations],
            "GHI_SOLARGIS_predict": [
                np.arange(16, dtype=np.float32) for _ in stations
            ],
        }
    )


def test_discovers_station_glob_and_reads_identity_from_column(tmp_path, monkeypatch):
    matched = tmp_path / "station=filename-a.parquet"
    ignored = tmp_path / "legacy_v1.parquet"
    matched.touch()
    ignored.touch()
    frames = {
        matched: station_frame(["actual-a"]),
        ignored: station_frame(["must-not-load"]),
    }
    calls = []

    def read_parquet(path, *, columns):
        calls.append((path, columns))
        return frames[path].loc[:, columns]

    monkeypatch.setattr(pd, "read_parquet", read_parquet)

    loaded = load_multi_station_data(None, config(tmp_path))

    assert loaded["station_id"].tolist() == ["actual-a"]
    assert loaded["source_file"].tolist() == ["station=filename-a.parquet"]
    assert calls == [(matched, [
        "timestamp_win", "Power", "GHI_SOLARGIS_predict", "station",
        "GHI_real", "Power_predict",
    ])]


def test_rejects_more_than_one_station_inside_a_station_file(tmp_path, monkeypatch):
    mixed = tmp_path / "station=mixed.parquet"
    mixed.touch()
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path, *, columns: station_frame(["a", "b"]).loc[:, columns],
    )

    with pytest.raises(ValueError, match="exactly one station"):
        load_multi_station_data(None, config(tmp_path))


def test_reusing_loaded_frame_preserves_station_file_identity(tmp_path, monkeypatch):
    matched = tmp_path / "station=actual-a.parquet"
    matched.touch()
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path, *, columns: station_frame(["actual-a"]).loc[:, columns],
    )
    first = load_multi_station_data(None, config(tmp_path))
    second = load_multi_station_data(first, config(tmp_path))
    assert second["source_file"].tolist() == ["station=actual-a.parquet"]


def test_parquet_rows_with_missing_required_values_are_dropped(tmp_path, monkeypatch):
    matched = tmp_path / "station=actual-a.parquet"
    matched.touch()
    frame = station_frame(["actual-a", "actual-a"])
    frame.at[1, "Power"] = None
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path, *, columns: frame.loc[:, columns],
    )
    loaded = load_multi_station_data(None, config(tmp_path))
    assert len(loaded) == 1
    assert loaded["timestamp_win"].tolist() == [pd.Timestamp("2024-09-01")]


def test_parquet_file_cannot_be_empty_after_dropna(tmp_path, monkeypatch):
    matched = tmp_path / "station=actual-a.parquet"
    matched.touch()
    frame = station_frame(["actual-a"])
    frame.at[0, "Power"] = None
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path, *, columns: frame.loc[:, columns],
    )
    with pytest.raises(ValueError, match="has no rows after dropna"):
        load_multi_station_data(None, config(tmp_path))


def feature_config(root) -> dict:
    result = config(root)
    result["data"]["site_metadata"] = {
        "timezone": "Asia/Shanghai",
        "aliases": {},
        "overrides": {
            "a": {"capacity": 100.0, "longitude": 121.5, "latitude": 31.2},
            "b": {"capacity": 200.0, "longitude": 103.8, "latitude": 30.6},
        },
    }
    result["features"].update(
        {"n_horizons": 16, "minutes_per_point": 15}
    )
    return result


def test_feature_engineering_runs_before_next_station_file_read(tmp_path, monkeypatch):
    first_path = tmp_path / "station=a.parquet"
    second_path = tmp_path / "station=b.parquet"
    first_path.touch()
    second_path.touch()
    frames = {
        first_path: station_frame(["a"]),
        second_path: station_frame(["b"]),
    }
    events = []

    def read_parquet(path, *, columns):
        events.append(f"read:{path.name}")
        return frames[path].loc[:, columns]

    original_builder = feature_module.build_horizon_frame

    def record_build(raw, *args, **kwargs):
        events.append(f"build:{raw['station_id'].iloc[0]}")
        return original_builder(raw, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    monkeypatch.setattr(feature_module, "build_horizon_frame", record_build)
    built = build_multi_station_feature_frames(
        None,
        feature_config(tmp_path),
        [16],
        require_target=True,
    )
    assert events == [
        "read:station=a.parquet",
        "build:a",
        "read:station=b.parquet",
        "build:b",
    ]
    assert built.file_count == 2


def test_per_file_features_equal_materialized_features(tmp_path, monkeypatch):
    first_path = tmp_path / "station=a.parquet"
    second_path = tmp_path / "station=b.parquet"
    first_path.touch()
    second_path.touch()
    frames = {
        first_path: station_frame(["a"]),
        second_path: station_frame(["b"]),
    }
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path, *, columns: frames[path].loc[:, columns],
    )
    cfg = feature_config(tmp_path)
    materialized_raw = load_multi_station_data(None, cfg)
    expected, expected_names, expected_target = build_horizon_frame(
        materialized_raw, cfg, 16
    )
    streamed = build_multi_station_feature_frames(
        None, cfg, [16], require_target=True
    )
    assert streamed.feature_names[16] == expected_names
    assert streamed.target_names[16] == expected_target
    actual = streamed.frames[16]
    assert actual["row_id"].tolist() == expected["row_id"].tolist()
    np.testing.assert_array_equal(
        actual[[*expected_names, expected_target]].to_numpy(),
        expected[[*expected_names, expected_target]].to_numpy(),
    )


def test_feature_pipeline_keeps_only_configured_station_roles(tmp_path, monkeypatch):
    paths = [tmp_path / f"station={station}.parquet" for station in ("a", "b", "c")]
    for path in paths:
        path.touch()
    frames = {
        path: station_frame([path.stem.removeprefix("station=")]) for path in paths
    }
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path, *, columns: frames[path].loc[:, columns],
    )
    cfg = feature_config(tmp_path)
    cfg["data"]["site_metadata"]["overrides"]["c"] = {
        "capacity": 300.0,
        "longitude": 110.0,
        "latitude": 32.0,
    }
    cfg["evaluation"] = {
        "training_stations": ["a"],
        "test_stations": ["b"],
    }
    built = build_multi_station_feature_frames(
        None, cfg, [16], require_target=True
    )
    assert set(built.frames[16]["station_id"]) == {"a", "b"}
    assert set(built.input_identity["station_id"]) == {"a", "b"}
    assert built.file_count == 2
