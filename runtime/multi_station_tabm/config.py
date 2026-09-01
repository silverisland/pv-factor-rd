from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Union

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
    if float(config["evaluation"]["score_capacity"]) <= 0:
        raise ValueError("evaluation.score_capacity must be positive")
    evaluation = config["evaluation"]
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
