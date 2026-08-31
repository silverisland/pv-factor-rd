from __future__ import annotations

import numpy as np
import pandas as pd

from adapters.tabm_factor_adapter import _metric_pair, _pair_predictions


def predictions(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["s1__a__h01", "s2__a__h01"],
            "station_id": ["s1", "s2"],
            "source_file": ["s1.parquet", "s2.parquet"],
            "timestamp": pd.to_datetime(["2024-09-01", "2024-09-01"]),
            "target_timestamp": pd.to_datetime(
                ["2024-09-01 00:15", "2024-09-01 00:15"]
            ),
            "horizon_step": [1, 1],
            "groundtruth": np.asarray([10.0, 20.0], dtype=np.float32),
            "prediction": np.asarray(values, dtype=np.float32),
        }
    )


def test_pairing_and_delta_use_identical_rows_and_targets():
    paired = _pair_predictions(
        predictions([8.0, 22.0]), predictions([9.0, 21.0]), seed=0
    )
    metrics = _metric_pair(paired, capacity=100.0)
    assert len(paired) == 2
    assert metrics["candidate"]["rmse"] < metrics["baseline"]["rmse"]
    assert metrics["delta"]["rmse"] < 0

