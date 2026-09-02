from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Union

import pandas as pd

Config = dict[str, Any]
ConfigInput = Union[str, Path, Mapping[str, Any]]


def _station_list(value: Any, name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"split.{name} must be a YAML list or null")
    stations = [str(item).strip() for item in value]
    if not stations or any(not station for station in stations):
        raise ValueError(f"split.{name} must contain non-empty station IDs")
    if len(stations) != len(set(stations)):
        raise ValueError(f"split.{name} contains duplicate station IDs")
    return stations


def resolve_test_stations(config: Config) -> list[str] | None:
    """Return the reference-baseline test station list."""
    return _station_list(config["split"].get("test_stations"), "test_stations")


def resolve_training_stations(config: Config) -> list[str] | None:
    """Return the explicit training allow-list, or None for all discovered stations."""
    return _station_list(config["split"].get("train_stations"), "train_stations")


def resolve_validation_stations(config: Config) -> list[str] | None:
    if config["split"].get("validation_strategy", "explicit") == "monthly_tail":
        return resolve_training_stations(config)
    configured = _station_list(
        config["split"].get("validation_stations"), "validation_stations"
    )
    return resolve_training_stations(config) if configured is None else configured


def validate_config(config: Config) -> None:
    required = {
        "data", "features", "model", "training", "split", "evaluation", "output"
    }
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
    if config["model"].get("label_normalization") != "none":
        raise ValueError(
            "The reference baseline trains the station-capacity target ratio "
            "directly; model.label_normalization must be none"
        )
    if float(config["model"].get("label_scale_value", 1.0)) != 1.0:
        raise ValueError("Ratio-target parity requires model.label_scale_value=1.0")
    lower, upper = map(float, config["model"]["prediction_clip"])
    if (lower, upper) != (0.0, 1.2):
        raise ValueError("Reference baseline requires prediction_clip=[0.0, 1.2]")
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
    split = config["split"]
    for prefix in ("train", "test"):
        start_name, end_name = f"{prefix}_start", f"{prefix}_end"
        if not split.get(start_name) or not split.get(end_name):
            raise ValueError(f"split requires {start_name} and {end_name}")
        start, end = pd.Timestamp(split[start_name]), pd.Timestamp(split[end_name])
        if start > end:
            raise ValueError(f"split.{start_name} must not exceed {end_name}")
    validation_strategy = str(split.get("validation_strategy", "explicit"))
    if validation_strategy == "explicit":
        if not split.get("validation_start") or not split.get("validation_end"):
            raise ValueError(
                "split.validation_strategy=explicit requires "
                "validation_start and validation_end"
            )
        if pd.Timestamp(split["validation_start"]) > pd.Timestamp(
            split["validation_end"]
        ):
            raise ValueError(
                "split.validation_start must not exceed validation_end"
            )
        if split.get("validation_last_days") is not None:
            raise ValueError(
                "split.validation_last_days must be null for explicit validation"
            )
    elif validation_strategy == "monthly_tail":
        if int(split.get("validation_last_days") or 0) <= 0:
            raise ValueError(
                "split.validation_strategy=monthly_tail requires a positive "
                "validation_last_days"
            )
        if split.get("validation_start") or split.get("validation_end"):
            raise ValueError(
                "Set validation_start and validation_end to null when using "
                "monthly_tail"
            )
        if split.get("validation_stations") is not None:
            raise ValueError(
                "monthly_tail reproduces pv_tabm_baseline and therefore uses "
                "train_stations; set validation_stations to null"
            )
    else:
        raise ValueError(
            "split.validation_strategy must be explicit or monthly_tail"
        )
    confirmation_start = split.get("confirmation_start")
    confirmation_end = split.get("confirmation_end")
    if bool(confirmation_start) != bool(confirmation_end):
        raise ValueError(
            "split.confirmation_start and confirmation_end must be set together"
        )
    if confirmation_start and pd.Timestamp(confirmation_start) > pd.Timestamp(
        confirmation_end
    ):
        raise ValueError(
            "split.confirmation_start must not exceed confirmation_end"
        )
    resolve_training_stations(config)
    resolve_validation_stations(config)
    _station_list(split.get("test_stations"), "test_stations")
    if evaluation.get("primary_metric") != "mean_monthly_capacity_score":
        raise ValueError(
            "Reference baseline requires "
            "evaluation.primary_metric=mean_monthly_capacity_score"
        )
    if evaluation.get("early_stopping_prediction") != "unclipped_ratio":
        raise ValueError(
            "Reference baseline requires unclipped ratio early-stopping predictions"
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
