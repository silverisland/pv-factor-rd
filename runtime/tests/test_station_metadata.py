from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime.multi_station_tabm.data import load_multi_station_data


def config(metadata_path=None) -> dict:
    return {
        "data": {
            "station_id_column": "plantid",
            "site_metadata": {
                "path": metadata_path,
                "station_column": "plantid",
                "capacity_column": "GCCAPACITY",
                "longitude_column": "LONGITUDE",
                "latitude_column": "LATITUDE",
                "timezone": "Asia/Shanghai",
                "aliases": {"station-a-alias": "station-a"},
                "overrides": {},
            },
            "columns": {
                "timestamp": "timestamp_win",
                "power_history": "Power",
                "ghi_history": "GHI_real",
                "power_future": "Power_predict",
                "future_weather": ["GHI_predict"],
            },
        },
        "features": {"history_length": 96},
    }


def raw(ghi_length=96):
    return pd.DataFrame(
        {
            "timestamp_win": ["2024-06-21 12:00"],
            "plantid": ["station-a-alias"],
            "Power": [np.arange(96)],
            "GHI_real": [np.arange(ghi_length)],
            "Power_predict": [np.arange(16)],
            "GHI_predict": [np.arange(16)],
        }
    )


def test_metadata_join_uses_alias_and_attaches_validated_values(tmp_path):
    path = tmp_path / "station_info.csv"
    pd.DataFrame(
        {
            "plantid": ["station-a"],
            "GCCAPACITY": [500.0],
            "LONGITUDE": [121.5],
            "LATITUDE": [31.2],
        }
    ).to_csv(path, index=False)
    frame = load_multi_station_data(raw(), config(str(path)))
    assert frame.loc[0, "station_id"] == "station-a"
    assert frame.loc[0, "site_capacity"] == 500.0
    assert frame.loc[0, "site_timezone"] == "Asia/Shanghai"


def test_misaligned_ghi_history_is_rejected():
    with pytest.raises(ValueError, match="histories must align exactly"):
        load_multi_station_data(raw(95), config())
