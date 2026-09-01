from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from factor_library.implementations.registry import (
    build_factor_frame,
    executable_factor_ids,
    factor_selection_manifest,
    validate_factor_ids,
)
from runtime.multi_station_tabm.features import build_horizon_frame
from factor_library.implementations import registry


def config() -> dict:
    return {
        "data": {
            "columns": {
                "timestamp": "timestamp_win",
                "power_history": "Power",
                "ghi_history": "GHI_real",
                "power_future": "Power_predict",
                "future_weather": [
                    "GHI_SOLARGIS_predict",
                    "TEMP_SOLARGIS_predict",
                    "WS_SOLARGIS_predict",
                    "WD_SOLARGIS_predict",
                ],
            },
            "weather_roles": {
                "ghi": "GHI_SOLARGIS_predict",
                "temperature": "TEMP_SOLARGIS_predict",
                "wind_speed": "WS_SOLARGIS_predict",
                "wind_direction": "WD_SOLARGIS_predict",
            },
            "site_metadata": {"timezone": "Asia/Shanghai"},
        },
        "features": {
            "history_length": 96,
            "n_horizons": 16,
            "minutes_per_point": 15,
        },
    }


def raw_frame() -> pd.DataFrame:
    history = np.arange(96, dtype=np.float32)
    future = np.arange(16, dtype=np.float32)
    return pd.DataFrame(
        {
            "timestamp_win": pd.to_datetime(
                ["2024-09-10 12:00", "2024-09-10 12:15"]
            ),
            "station_id": ["s1", "s2"],
            "source_file": ["s1.parquet", "s2.parquet"],
            "Power": [history, history + 10],
            "GHI_real": [history * 10, (history + 10) * 10],
            "Power_predict": [future + 100, future + 110],
            "GHI_SOLARGIS_predict": [future + 200, future + 210],
            "TEMP_SOLARGIS_predict": [future + 20, future + 21],
            "WS_SOLARGIS_predict": [future + 2, future + 3],
            "WD_SOLARGIS_predict": [future + 100, future + 110],
            "site_capacity": [500.0, 520.0],
            "site_longitude": [121.5, 103.8],
            "site_latitude": [31.2, 30.6],
            "site_timezone": ["Asia/Shanghai", "Asia/Shanghai"],
        }
    )


def test_executable_registry_builds_finite_namespaced_columns():
    factor_ids = [
        "factor.power.multiscale-ramp",
        "factor.power.multiscale-slope",
        "factor.power.variability",
    ]
    frame, names, manifest = build_factor_frame(
        raw_frame(), config(), 16, factor_ids
    )
    assert len(frame) == 2
    assert names
    assert all(name.startswith("factor__") for name in names)
    assert np.isfinite(frame.to_numpy(dtype=np.float32)).all()
    assert manifest["factor_ids"] == factor_ids
    assert manifest["finite_fraction"] == 1.0


def test_candidate_only_appends_features_and_preserves_rows_and_target():
    raw = raw_frame()
    baseline, baseline_features, baseline_target = build_horizon_frame(
        raw, config(), 16
    )
    candidate, candidate_features, candidate_target = build_horizon_frame(
        raw,
        config(),
        16,
        factor_ids=["factor.power.multiscale-ramp"],
    )
    assert candidate_features[: len(baseline_features)] == baseline_features
    assert len(candidate_features) > len(baseline_features)
    assert candidate["row_id"].tolist() == baseline["row_id"].tolist()
    assert candidate_target == baseline_target
    assert np.array_equal(
        candidate[baseline_target].to_numpy(), baseline[baseline_target].to_numpy()
    )
    assert np.array_equal(
        candidate[baseline_features].to_numpy(),
        baseline[baseline_features].to_numpy(),
    )


def test_unbound_or_conditional_factor_is_rejected_before_training():
    assert "factor.power.multiscale-ramp" in executable_factor_ids()
    assert "factor.weather.future-change" in executable_factor_ids()
    assert "factor.weather.clear-sky-index-forecast" in executable_factor_ids()
    with pytest.raises(ValueError, match="not executable"):
        validate_factor_ids(["factor.weather.nwp-current-bias"])


def test_priority_weather_factors_are_finite_and_target_horizon_specific():
    factor_ids = [
        "factor.weather.future-change",
        "factor.weather.clear-sky-index-forecast",
    ]
    frame, names, manifest = build_factor_frame(raw_frame(), config(), 4, factor_ids)
    assert len(names) == 22
    assert np.isfinite(frame.to_numpy(dtype=np.float32)).all()
    assert manifest["factor_ids"] == factor_ids
    assert any(name.endswith("ghi_target_change_from_anchor") for name in names)
    assert any(name.endswith("forecast_clear_sky_index") for name in names)
    assert frame.loc[
        0,
        "factor__weather__future-change__ghi_target_change_from_anchor",
    ] == -747.0
    assert frame.loc[
        0,
        "factor__weather__future-change__ghi_target_step_change",
    ] == 1.0
    assert frame.loc[
        0,
        "factor__weather__future-change__temperature_prefix_slope",
    ] == 1.0


def test_future_change_ignores_forecast_values_after_target_horizon():
    original = raw_frame()
    changed = raw_frame()
    for column in (
        "GHI_SOLARGIS_predict",
        "TEMP_SOLARGIS_predict",
        "WS_SOLARGIS_predict",
    ):
        for row_index in changed.index:
            values = changed.at[row_index, column].copy()
            values[4:] += 10000.0
            changed.at[row_index, column] = values
    first, names, _ = build_factor_frame(
        original, config(), 4, ["factor.weather.future-change"]
    )
    second, second_names, _ = build_factor_frame(
        changed, config(), 4, ["factor.weather.future-change"]
    )
    assert names == second_names
    np.testing.assert_array_equal(first[names].to_numpy(), second[names].to_numpy())


def test_priority_weather_factors_only_append_to_baseline():
    raw = raw_frame()
    baseline, baseline_features, baseline_target = build_horizon_frame(
        raw, config(), 16
    )
    candidate, candidate_features, candidate_target = build_horizon_frame(
        raw,
        config(),
        16,
        factor_ids=[
            "factor.weather.future-change",
            "factor.weather.clear-sky-index-forecast",
        ],
    )
    assert candidate_features[: len(baseline_features)] == baseline_features
    assert candidate_target == baseline_target
    assert candidate["row_id"].tolist() == baseline["row_id"].tolist()
    np.testing.assert_array_equal(
        candidate[baseline_features].to_numpy(),
        baseline[baseline_features].to_numpy(),
    )
    np.testing.assert_array_equal(
        candidate[baseline_target].to_numpy(),
        baseline[baseline_target].to_numpy(),
    )


def test_factor_manifest_binds_catalog_implementation_and_registry():
    manifest = factor_selection_manifest(["factor.power.multiscale-ramp"])
    record = manifest["records"][0]
    assert record["catalog_record_sha256"]
    assert record["implementation_sha256"]
    assert manifest["registry_sha256"]
    assert manifest["selection_sha256"]


def test_confirmed_metadata_and_ghi_factors_are_executable():
    factor_ids = [
        "factor.power.capacity-ratio",
        "factor.power.previous-day-profile",
        "factor.solar.position",
        "factor.solar.daylight-boundary",
        "factor.solar.clear-sky-irradiance",
        "factor.solar.power-clear-sky-index",
        "factor.pv.module-temperature-proxy",
        "factor.pv.temperature-corrected-irradiance",
        "factor.pv.low-irradiance-state",
        "factor.regime.clear-variable-overcast",
        "factor.regime.joint-weather-power-ramp",
        "factor.operation.clipping-score",
        "factor.operation.weather-power-residual",
    ]
    frame, names, manifest = build_factor_frame(raw_frame(), config(), 16, factor_ids)
    assert len(frame) == 2
    assert len(names) >= len(factor_ids)
    assert np.isfinite(frame.to_numpy(dtype=np.float32)).all()
    assert manifest["factor_ids"] == factor_ids


def test_factor_error_contains_factor_station_timestamp_and_horizon():
    broken = raw_frame()
    broken.at[0, "site_capacity"] = 0.0
    with pytest.raises(ValueError) as caught:
        build_factor_frame(
            broken, config(), 16, ["factor.power.capacity-ratio"]
        )
    message = str(caught.value)
    assert "factor.power.capacity-ratio" in message
    assert "station_id='s1'" in message
    assert "timestamp=" in message
    assert "horizon_step=16" in message


def test_non_finite_output_identifies_source_row_and_column():
    broken = raw_frame()
    history = broken.at[0, "Power"].copy()
    history[-1] = np.inf
    broken.at[0, "Power"] = history
    with pytest.raises(ValueError, match="station_id='s1'.*column="):
        build_factor_frame(
            broken, config(), 16, ["factor.power.multiscale-ramp"]
        )


def test_target_solar_position_is_cached_across_factor_family():
    original = registry.solar_position
    with patch.object(registry, "solar_position", wraps=original) as position:
        build_factor_frame(
            raw_frame(),
            config(),
            16,
            [
                "factor.solar.position",
                "factor.solar.daylight-boundary",
                "factor.solar.clear-sky-irradiance",
                "factor.weather.clear-sky-index-forecast",
                "factor.pv.low-irradiance-state",
            ],
        )
    assert position.call_count == len(raw_frame())
