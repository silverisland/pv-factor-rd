from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .data import SOURCE_FILE, STATION_ID
from factor_library.implementations.registry import (
    build_factor_frame,
    factor_selection_manifest,
)


def _array_matrix(series: pd.Series, minimum_length: int, name: str) -> np.ndarray:
    rows = []
    for row_index, value in enumerate(series):
        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} row {row_index} is not a numeric array") from error
        if len(array) < minimum_length:
            raise ValueError(
                f"{name} row {row_index} has {len(array)} points; need at least {minimum_length}"
            )
        rows.append(array)
    if not rows:
        raise ValueError("Multi-station frame is empty")
    return np.stack(rows)


def build_horizon_frame(
    raw: pd.DataFrame,
    config: Config,
    horizon_step: int,
    *,
    require_target: bool = True,
    factor_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Reproduce tabm4pv.py features for one scalar forecast horizon."""
    data_columns = config["data"]["columns"]
    feature_config = config["features"]
    history_length = int(feature_config["history_length"])
    maximum_horizons = int(feature_config["n_horizons"])
    if not 1 <= horizon_step <= maximum_horizons:
        raise ValueError(f"horizon_step must be within 1..{maximum_horizons}")
    horizon_index = horizon_step - 1
    timestamp_column = data_columns["timestamp"]
    power_column = data_columns["power_history"]
    target_column = data_columns["power_future"]
    weather_columns = list(data_columns["future_weather"])

    result = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[timestamp_column]).to_numpy(),
            STATION_ID: raw[STATION_ID].astype(str).to_numpy(),
            SOURCE_FILE: raw[SOURCE_FILE].astype(str).to_numpy(),
        }
    )
    feature_names: list[str] = []

    for column in weather_columns:
        matrix = _array_matrix(raw[column], horizon_step, column)
        output_name = f"{column}_target"
        result[output_name] = matrix[:, horizon_index]
        feature_names.append(output_name)

    history = _array_matrix(raw[power_column], history_length, power_column)[:, -history_length:]
    history_names = [f"{power_column}_lag_{lag}" for lag in range(history_length, 0, -1)]
    history_frame = pd.DataFrame(history, columns=history_names, index=result.index)
    result = pd.concat([result, history_frame], axis=1)
    feature_names.extend(history_names)

    minutes = int(feature_config["minutes_per_point"])
    origin = pd.to_datetime(result["timestamp"])
    target_timestamp = origin + pd.to_timedelta(horizon_step * minutes, unit="m")
    # Integer target hour matches the legacy h16 formula and remains valid when
    # shorter horizons cross an hour boundary.
    result["predict_hour"] = target_timestamp.dt.hour
    # Preserve the legacy origin-month semantics for parity.
    result["predict_month"] = origin.dt.month
    feature_names.extend(["predict_hour", "predict_month"])

    factor_frame, factor_names, _ = build_factor_frame(
        raw, config, horizon_step, factor_ids
    )
    if factor_names:
        factor_frame = factor_frame.reset_index(drop=True)
        result = pd.concat([result, factor_frame], axis=1)
        feature_names.extend(factor_names)

    output_target: str | None = None
    if require_target:
        target_matrix = _array_matrix(raw[target_column], horizon_step, target_column)
        output_target = f"{target_column}_target"
        result[output_target] = target_matrix[:, horizon_index]

    result["horizon_step"] = horizon_step
    result["target_timestamp"] = target_timestamp
    occurrence = result.groupby([STATION_ID, "timestamp"], sort=False).cumcount()
    result["row_id"] = (
        result[STATION_ID].astype(str)
        + "__"
        + origin.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        + "__n"
        + occurrence.astype(str)
        + f"__h{horizon_step:02d}"
    )
    return result, feature_names, output_target


def feature_contract(
    config: Config,
    horizon_step: int,
    factor_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    columns = config["data"]["columns"]
    return {
        "forecast_object": "multi_station_shared_model",
        "horizon_step": horizon_step,
        "history_length": int(config["features"]["history_length"]),
        "power_history_column": columns["power_history"],
        "ghi_history_column": columns.get("ghi_history"),
        "history_alignment": "power_and_ghi_elementwise_same_timestamp",
        "future_weather_columns": list(columns["future_weather"]),
        "weather_roles": dict(config["data"].get("weather_roles", {})),
        "power_target_column": columns["power_future"],
        "station_identity_role": "metadata_only",
        "station_id_as_model_feature": False,
        "training_rows": "pooled_across_stations",
        "station_aggregation": False,
        "capacity_weighted_weather": False,
        "site_metadata_role": "row_local_static_factor_inputs_only",
        "site_timezone": config["data"].get("site_metadata", {}).get(
            "timezone", "Asia/Shanghai"
        ),
        "factor_parameters": dict(config.get("factor_parameters", {})),
        "factor_selection": factor_selection_manifest(factor_ids),
    }
