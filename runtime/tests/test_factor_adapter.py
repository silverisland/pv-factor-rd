from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adapters.tabm_factor_adapter import (
    _metric_pair,
    _nwp_issue_time_audit,
    _pair_predictions,
    _runtime_pair_audit,
    _station_macro,
)


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


def test_pairing_rejects_target_or_population_changes():
    changed_target = predictions([9.0, 21.0])
    changed_target.loc[0, "groundtruth"] = 11.0
    with pytest.raises(ValueError, match="targets differ"):
        _pair_predictions(predictions([8.0, 22.0]), changed_target, seed=0)

    missing_row = predictions([9.0, 21.0]).iloc[:1].copy()
    with pytest.raises(ValueError, match="row populations differ"):
        _pair_predictions(predictions([8.0, 22.0]), missing_row, seed=0)


def test_station_macro_and_runtime_audit_detect_regression_and_mismatch():
    by_station = [
        {"baseline": {"rmse": 2.0}, "candidate": {"rmse": 1.0}},
        {"baseline": {"rmse": 2.0}, "candidate": {"rmse": 3.0}},
    ]
    macro = _station_macro(by_station)
    assert macro["improved_station_ratio"] == 0.5
    assert macro["worst_station_delta_rmse"] == 1.0

    common = {
        "input_stations": {"station_ids": ["s1"]},
        "input_file_count": 1,
        "raw_array_materialization": "one_station_chunk_at_a_time",
        "training_stations": {"station_ids": ["s1"]},
        "configured_training_stations": None,
        "configured_test_stations": ["s1"],
        "evaluation_object": "configured_test_stations",
        "split_protocol": {"protocol": "configured_station_transfer"},
        "prediction_mode": "endpoint",
        "horizons": [16],
        "seed": 0,
        "fixed_runtime_contract_sha256": "fixed",
        "evaluation_protocol_sha256": "eval",
        "horizon_manifests": [
            {
                "horizon_step": 16,
                "frame_rows": "rows",
                "train_rows": "train",
                "validation_rows": "validation",
                "frame_stations": "stations",
                "train_stations": "stations",
                "validation_stations": "stations",
                "base_feature_names_sha256": "base-names",
                "base_feature_values_sha256": "base-values",
                "target_ratio_sha256": "target-ratio",
                "target_power_sha256": "target-power",
                "capacity_sha256": "capacity",
                "environment_sha256": "environment",
            }
        ],
    }
    candidate = {"manifest": {**common, "seed": 1}}
    audit = _runtime_pair_audit({"manifest": common}, candidate)
    assert not audit["passed"]
    assert "seed" in audit["mismatches"]


def test_nwp_issue_time_is_verified_or_explicitly_assumed():
    raw = predictions([8.0, 22.0]).rename(columns={"timestamp": "origin"})
    runtime_config = {"data": {"columns": {"timestamp": "origin"}}}
    assumed = _nwp_issue_time_audit(raw, runtime_config, None)
    assert assumed["status"] == "contract_assumed"

    raw["issue_time"] = pd.to_datetime(["2024-08-31", "2024-08-31"])
    verified = _nwp_issue_time_audit(raw, runtime_config, "issue_time")
    assert verified["status"] == "verified"
    raw.loc[0, "issue_time"] = pd.Timestamp("2024-09-02")
    with pytest.raises(ValueError, match="NWP leakage detected"):
        _nwp_issue_time_audit(raw, runtime_config, "issue_time")
