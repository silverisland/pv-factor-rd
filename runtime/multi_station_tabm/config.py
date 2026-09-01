from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Union

import pandas as pd

Config = dict[str, Any]
ConfigInput = Union[str, Path, Mapping[str, Any]]


def validate_config(config: Config) -> None:
    required = {"data", "features", "model", "training", "evaluation", "output"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config missing sections: {missing}")
    data = config["data"]
    parquet_glob = str(data.get("parquet_glob", "")).strip()
    if not parquet_glob:
        raise ValueError("data.parquet_glob must be non-empty")
    station_id_column = str(data.get("station_id_column", "")).strip()
    if not station_id_column:
        raise ValueError(
            "data.station_id_column must name the station column stored in each parquet"
        )
    if bool(data.get("station_id_as_feature", False)):
        raise ValueError(
            "The fixed pooled baseline keeps station_id as metadata, not a model feature"
        )
    metadata = data.get("site_metadata", {})
    if metadata and metadata.get("timezone", "Asia/Shanghai") != "Asia/Shanghai":
        raise ValueError(
            "The current expert-factor contract requires site_metadata.timezone=Asia/Shanghai"
        )
    roles = data.get("weather_roles", {})
    weather_columns = set(data["columns"]["future_weather"])
    invalid_roles = {
        role: column for role, column in roles.items() if column not in weather_columns
    }
    if invalid_roles:
        raise ValueError(f"data.weather_roles reference unknown columns: {invalid_roles}")
    mode = config["features"].get("prediction_mode", "endpoint")
    if mode not in {"endpoint", "curve"}:
        raise ValueError("features.prediction_mode must be endpoint or curve")
    horizons = resolve_horizons(config)
    maximum = int(config["features"]["n_horizons"])
    if not horizons or horizons[0] < 1 or horizons[-1] > maximum:
        raise ValueError(f"horizons must be within 1..{maximum}: {horizons}")
    if float(config["model"]["label_scale_value"]) <= 0:
        raise ValueError("model.label_scale_value must be positive")
    lower, upper = map(float, config["model"]["prediction_clip"])
    if lower >= upper:
        raise ValueError("model.prediction_clip must be increasing")
    training = config["training"]
    if training.get("sampling_strategy", "pooled_rows") != "pooled_rows":
        raise ValueError(
            "Fixed multi-station baseline requires training.sampling_strategy=pooled_rows"
        )
    for name in ("epochs", "batch_size", "inference_batch_size", "early_stopping_patience"):
        if int(training[name]) <= 0:
            raise ValueError(f"training.{name} must be positive")
    evaluation = config["evaluation"]
    if float(evaluation["score_capacity"]) <= 0:
        raise ValueError("evaluation.score_capacity must be positive")
    target_station = str(evaluation.get("target_station", "")).strip()
    if not target_station:
        raise ValueError("evaluation.target_station must be non-empty")
    if evaluation.get("source_station_time_policy") != "all_available":
        raise ValueError(
            "Target-transfer protocol requires "
            "evaluation.source_station_time_policy=all_available"
        )
    validation = evaluation.get("validation", {})
    if validation.get("strategy") != "target_history_tail":
        raise ValueError(
            "evaluation.validation.strategy must be target_history_tail"
        )
    if int(validation.get("target_history_days", 0)) <= 0:
        raise ValueError(
            "evaluation.validation.target_history_days must be positive"
        )
    required_purge_hours = (
        int(config["features"]["n_horizons"])
        * int(config["features"]["minutes_per_point"])
        / 60.0
    )
    if float(evaluation.get("purge_hours", 0)) < required_purge_hours:
        raise ValueError(
            f"evaluation.purge_hours must be at least {required_purge_hours:g}"
        )
    periods = evaluation.get("periods", {})
    configured_periods = []
    for name in ("confirmation", "final_test"):
        period = periods.get(name)
        if not period:
            continue
        if not period.get("start") or not period.get("end"):
            raise ValueError(f"evaluation.periods.{name} requires start and end")
        start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
        if start > end:
            raise ValueError(
                f"evaluation.periods.{name} start must not exceed end"
            )
        configured_periods.append((start, end, name))
    if not configured_periods:
        raise ValueError(
            "Configure at least one of evaluation.periods.confirmation or final_test"
        )
    configured_periods.sort()
    for (_, previous_end, previous_name), (current_start, _, current_name) in zip(
        configured_periods, configured_periods[1:]
    ):
        if current_start <= previous_end:
            raise ValueError(
                f"evaluation periods overlap: {previous_name} and {current_name}"
            )
    if bool(evaluation.get("require_all_training_stations_in_evaluation", True)):
        raise ValueError(
            "Target-only evaluation requires "
            "evaluation.require_all_training_stations_in_evaluation=false"
        )
    if evaluation.get("primary_metric", "rmse") != "rmse":
        raise ValueError("Fixed baseline requires evaluation.primary_metric=rmse")
    if evaluation.get("early_stopping_prediction", "inverse_scaled_unclipped") != "inverse_scaled_unclipped":
        raise ValueError(
            "Fixed baseline requires un-clipped inverse-scaled early-stopping predictions"
        )


def load_config(value: ConfigInput) -> Config:
    if isinstance(value, Mapping):
        config = deepcopy(dict(value))
    else:
        import yaml

        with Path(value).expanduser().open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    validate_config(config)
    return config


def resolve_horizons(config: Config) -> list[int]:
    features = config["features"]
    if features.get("prediction_mode", "endpoint") == "endpoint":
        return [int(features.get("endpoint_horizon_step", features["n_horizons"]))]
    selected = features.get("horizons", "all")
    if selected == "all":
        return list(range(1, int(features["n_horizons"]) + 1))
    return sorted({int(value) for value in selected})
