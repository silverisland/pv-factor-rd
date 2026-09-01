from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime.multi_station_tabm.data import load_multi_station_data


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
