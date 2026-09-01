from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime.multi_station_tabm.fingerprints import (
    named_numeric_arrays_sha256,
    numeric_array_sha256,
    split_fingerprint,
)
from runtime.multi_station_tabm.metrics import (
    daily_and_monthly_metrics,
    grouped_horizon_metrics,
    monthly_score_summary,
    station_macro_summary,
    station_metrics,
)
from runtime.multi_station_tabm.model import model_contract
from runtime.multi_station_tabm.preprocessing import prepare_training_data


def config() -> dict:
    return {
        "model": {
            "label_normalization": "scale",
            "label_scale_value": 500.0,
            "prediction_clip": [0.0, 465.0],
        },
        "training": {
            "preprocessing": {
                "noise_seed": 0,
                "noise_std": 1e-5,
                "min_quantiles": 2,
                "max_quantiles": 10,
                "samples_per_quantile": 1,
                "quantile_subsample": 10**9,
            }
        },
    }


def test_preprocessing_is_train_only_and_fingerprinted():
    train = pd.DataFrame(
        {
            "row_id": ["a", "b", "c"],
            "station_id": ["s1", "s1", "s2"],
            "x": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
        }
    )
    validation = pd.DataFrame(
        {
            "row_id": ["d", "e"],
            "station_id": ["s1", "s2"],
            "x": [100.0, 200.0],
            "y": [40.0, 50.0],
        }
    )
    prepared = prepare_training_data(train, validation, ["x"], "y", config())
    assert (
        prepared.manifest["preprocessor"]["fit_partition"]
        == "source_all_plus_target_history_train"
    )
    assert prepared.manifest["train"]["rows"] == 3
    assert prepared.manifest["validation"]["rows"] == 2
    assert prepared.manifest["train_stations"]["station_count"] == 2
    assert prepared.manifest["validation_stations"]["station_ids"] == ["s1", "s2"]
    assert prepared.manifest["x_validation_raw_sha256"] == numeric_array_sha256(
        validation[["x"]].to_numpy(dtype=np.float32)
    )


def test_split_overlap_is_rejected():
    train = pd.DataFrame(
        {"row_id": ["same"], "station_id": ["s1"], "x": [1.0], "y": [1.0]}
    )
    validation = pd.DataFrame(
        {"row_id": ["same"], "station_id": ["s1"], "x": [2.0], "y": [2.0]}
    )
    with pytest.raises(ValueError, match="overlap"):
        prepare_training_data(train, validation, ["x"], "y", config())


def test_fingerprints_are_order_sensitive():
    assert split_fingerprint(["a", "b"])["row_id_sha256"] != split_fingerprint(
        ["b", "a"]
    )["row_id_sha256"]


def test_named_array_fingerprint_binds_names_and_values():
    baseline = named_numeric_arrays_sha256(
        {"weight": np.array([1.0, 2.0], dtype=np.float32)}
    )
    assert baseline != named_numeric_arrays_sha256(
        {"weight": np.array([1.0, 3.0], dtype=np.float32)}
    )
    assert baseline != named_numeric_arrays_sha256(
        {"bias": np.array([1.0, 2.0], dtype=np.float32)}
    )


def test_empty_split_and_duplicate_features_are_rejected():
    train = pd.DataFrame(
        {"row_id": ["a"], "station_id": ["s1"], "x": [1.0], "y": [1.0]}
    )
    empty = train.iloc[0:0].copy()
    with pytest.raises(ValueError, match="non-empty"):
        prepare_training_data(train, empty, ["x"], "y", config())
    with pytest.raises(ValueError, match="unique"):
        prepare_training_data(train, train.assign(row_id="b"), ["x", "x"], "y", config())


def test_metrics_reject_non_finite_values():
    from runtime.multi_station_tabm.metrics import regression_metrics

    with pytest.raises(ValueError, match="finite"):
        regression_metrics(
            np.array([1.0], dtype=np.float32),
            np.array([np.nan], dtype=np.float32),
            465.0,
        )


def test_metrics_use_target_date_and_keep_horizon_groups():
    predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-31 23:45", "2025-02-01 00:00", "2025-02-01 00:15"]
            ),
            "target_timestamp": pd.to_datetime(
                ["2025-02-01 00:00", "2025-02-01 00:15", "2025-02-01 02:15"]
            ),
            "horizon_step": [1, 1, 8],
            "station_id": ["s1", "s2", "s2"],
            "groundtruth": [10.0, 20.0, 30.0],
            "prediction": [11.0, 18.0, 33.0],
        }
    )
    daily, monthly = daily_and_monthly_metrics(predictions, 465.0)
    assert daily["date"].dt.month.unique().tolist() == [2]
    summary = monthly_score_summary(monthly)
    assert set(summary["horizon_step"]) == {1, 8}
    grouped = grouped_horizon_metrics(predictions, 465.0, 15)
    assert set(grouped["horizon_group"]) == {"0_1h", "1_2h"}
    by_station = station_metrics(predictions, 465.0)
    assert set(by_station["station_id"]) == {"s1", "s2"}
    macro = station_macro_summary(by_station)
    assert set(macro["horizon_step"]) == {1, 8}


def test_model_contract_is_scalar_and_has_no_architecture_overrides():
    contract = model_contract(102)
    assert contract["d_out"] == 1
    assert contract["architecture_kwargs"] == {}
    assert contract["n_num_features"] == 102
