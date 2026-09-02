from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import Config, resolve_test_stations, resolve_training_stations
from .data import (
    SOURCE_FILE,
    STATION_ID,
    SITE_CAPACITY,
    SITE_LATITUDE,
    SITE_LONGITUDE,
    SITE_TIMEZONE,
    DataInput,
    iter_multi_station_data,
)
from factor_library.implementations.registry import (
    build_factor_frame,
    factor_selection_manifest,
)


@dataclass(frozen=True)
class MultiHorizonFeatureFrames:
    frames: dict[int, pd.DataFrame]
    feature_names: dict[int, list[str]]
    target_names: dict[int, str | None]
    input_identity: pd.DataFrame
    file_count: int
    build_seconds: dict[int, float]


def _array_matrix(
    series: pd.Series,
    minimum_length: int,
    name: str,
    *,
    take_last: int,
) -> np.ndarray:
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
        rows.append(array[-take_last:])
    if not rows:
        raise ValueError("Multi-station frame is empty")
    return np.stack(rows)


def _array_at(
    series: pd.Series, minimum_length: int, name: str, index: int
) -> np.ndarray:
    values = []
    for row_index, value in enumerate(series):
        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} row {row_index} is not a numeric array") from error
        if len(array) < minimum_length:
            raise ValueError(
                f"{name} row {row_index} has {len(array)} points; "
                f"need at least {minimum_length}"
            )
        values.append(float(array[index]))
    if not values:
        raise ValueError("Multi-station frame is empty")
    return np.asarray(values, dtype=np.float32)


def build_horizon_frame(
    raw: pd.DataFrame,
    config: Config,
    horizon_step: int,
    *,
    require_target: bool = True,
    factor_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Build the capacity-normalized reference baseline plus optional factors.

    The protected base feature order is target-index future weather, 96
    capacity-normalized power lags, target hour, and target month.  TabM learns
    the generation coefficient (power / station capacity); physical power is
    restored only in the evaluator.
    """
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
            "capacity": pd.to_numeric(raw[SITE_CAPACITY], errors="coerce").to_numpy(
                dtype=np.float32
            ),
        }
    )
    invalid_capacity = ~np.isfinite(result["capacity"]) | (result["capacity"] <= 0)
    if invalid_capacity.any():
        stations = result.loc[invalid_capacity, STATION_ID].drop_duplicates().tolist()
        raise ValueError(f"Missing or invalid station capacity for: {stations}")
    feature_names: list[str] = []

    for column in weather_columns:
        output_name = f"{column}_target"
        result[output_name] = _array_at(
            raw[column], horizon_step, column, horizon_index
        )
        feature_names.append(output_name)

    history = _array_matrix(
        raw[power_column],
        history_length,
        power_column,
        take_last=history_length,
    )
    history = history / result["capacity"].to_numpy(dtype=np.float32)[:, None]
    history_names = [f"power_lag_{lag}" for lag in range(history_length, 0, -1)]
    history_frame = pd.DataFrame(history, columns=history_names, index=result.index)
    result = pd.concat([result, history_frame], axis=1)
    feature_names.extend(history_names)

    minutes = int(feature_config["minutes_per_point"])
    origin = pd.to_datetime(result["timestamp"])
    target_timestamp = origin + pd.to_timedelta(horizon_step * minutes, unit="m")
    # Both calendar fields describe the predicted instant, including crossings
    # of hour, day, month, and year boundaries.
    result["predict_hour"] = target_timestamp.dt.hour
    result["predict_month"] = target_timestamp.dt.month
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
        target_power = _array_at(
            raw[target_column], horizon_step, target_column, horizon_index
        )
        result["target_power"] = target_power
        output_target = "target_ratio"
        result[output_target] = (
            target_power / result["capacity"].to_numpy(dtype=np.float32)
        )

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
    validity_columns = [*feature_names]
    if output_target is not None:
        validity_columns.append(output_target)
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=validity_columns).reset_index(drop=True)
    if result.empty:
        raise ValueError("No valid samples remain after feature construction")
    return result, feature_names, output_target


def build_multi_station_feature_frames(
    data: DataInput | None,
    config: Config,
    horizons: Sequence[int],
    *,
    require_target: bool,
    factor_ids: Sequence[str] | None = None,
) -> MultiHorizonFeatureFrames:
    """Read one station chunk, engineer all horizons, then release raw arrays."""
    selected_horizons = [int(value) for value in horizons]
    if not selected_horizons or len(selected_horizons) != len(set(selected_horizons)):
        raise ValueError("horizons must be non-empty and unique")
    frame_chunks: dict[int, list[pd.DataFrame]] = {
        horizon: [] for horizon in selected_horizons
    }
    feature_names: dict[int, list[str]] = {}
    target_names: dict[int, str | None] = {}
    identity_chunks: list[pd.DataFrame] = []
    file_count = 0
    build_seconds = {horizon: 0.0 for horizon in selected_horizons}
    configured_training = (
        resolve_training_stations(config) if "evaluation" in config else None
    )
    selected_stations = (
        None
        if configured_training is None
        else set(configured_training) | set(resolve_test_stations(config))
    )

    for raw_chunk in iter_multi_station_data(
        data, config, require_target=require_target
    ):
        if selected_stations is not None:
            raw_chunk = raw_chunk[
                raw_chunk[STATION_ID].astype(str).isin(selected_stations)
            ].copy()
            if raw_chunk.empty:
                continue
        file_count += 1
        identity_columns = [
            STATION_ID,
            SOURCE_FILE,
            SITE_CAPACITY,
            SITE_LONGITUDE,
            SITE_LATITUDE,
            SITE_TIMEZONE,
        ]
        identity_chunks.append(raw_chunk[identity_columns].copy())
        for horizon in selected_horizons:
            started = time.perf_counter()
            current, current_names, current_target = build_horizon_frame(
                raw_chunk,
                config,
                horizon,
                require_target=require_target,
                factor_ids=factor_ids,
            )
            build_seconds[horizon] += time.perf_counter() - started
            if horizon in feature_names and feature_names[horizon] != current_names:
                raise ValueError(
                    f"Feature order differs between station files for horizon {horizon}"
                )
            if horizon in target_names and target_names[horizon] != current_target:
                raise ValueError(
                    f"Target differs between station files for horizon {horizon}"
                )
            feature_names[horizon] = current_names
            target_names[horizon] = current_target
            frame_chunks[horizon].append(current)

    if file_count == 0:
        raise ValueError("No station chunks were processed")
    frames: dict[int, pd.DataFrame] = {}
    for horizon, chunks in frame_chunks.items():
        combined = pd.concat(chunks, ignore_index=True).sort_values(
            ["timestamp", STATION_ID, SOURCE_FILE, "row_id"],
            kind="stable",
            ignore_index=True,
        )
        if combined["row_id"].duplicated().any():
            duplicates = combined.loc[
                combined["row_id"].duplicated(keep=False), "row_id"
            ].head(5).tolist()
            raise ValueError(
                "Duplicate row_id values after per-file feature construction: "
                f"{duplicates}"
            )
        frames[horizon] = combined
    return MultiHorizonFeatureFrames(
        frames=frames,
        feature_names=feature_names,
        target_names=target_names,
        input_identity=pd.concat(identity_chunks, ignore_index=True),
        file_count=file_count,
        build_seconds=build_seconds,
    )


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
        "model_target": "target_ratio",
        "power_normalization": "row_station_capacity",
        "prediction_restoration": "prediction_ratio_times_row_station_capacity",
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
