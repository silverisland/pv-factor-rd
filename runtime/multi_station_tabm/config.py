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
        raise ValueError(f"evaluation.{name} must be a YAML list or null")
    stations = [str(item).strip() for item in value]
    if not stations or any(not station for station in stations):
        raise ValueError(f"evaluation.{name} must contain non-empty station IDs")
    if len(stations) != len(set(stations)):
        raise ValueError(f"evaluation.{name} contains duplicate station IDs")
    return stations


def resolve_test_stations(config: Config) -> list[str]:
    """Return configured evaluation stations, with target_station compatibility."""
    evaluation = config["evaluation"]
    configured = _station_list(evaluation.get("test_stations"), "test_stations")
    if configured is not None:
        return configured
    legacy = str(evaluation.get("target_station", "")).strip()
    if not legacy:
        raise ValueError(
            "Configure evaluation.test_stations or legacy evaluation.target_station"
        )
    return [legacy]


def resolve_training_stations(config: Config) -> list[str] | None:
    """Return the explicit training allow-list, or None for all discovered stations."""
    return _station_list(
        config["evaluation"].get("training_stations"), "training_stations"
    )


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
    if "purge_hours" in evaluation:
        raise ValueError(
            "evaluation.purge_hours has been removed; split scalar labels "
            "directly by target_timestamp"
        )
    training_stations = resolve_training_stations(config)
    test_stations = resolve_test_stations(config)
    legacy_target = str(evaluation.get("target_station", "")).strip()
    if legacy_target and evaluation.get("test_stations") is not None:
        if legacy_target not in test_stations:
            raise ValueError(
                "Legacy evaluation.target_station must be included in "
                "evaluation.test_stations when both are configured"
            )
    if (
        training_stations is not None
        and set(test_stations) - set(training_stations)
        and bool(evaluation.get("reject_unseen_stations", True))
    ):
        raise ValueError(
            "evaluation.test_stations contains stations excluded from "
            "evaluation.training_stations; set reject_unseen_stations=false "
            "to evaluate held-out stations"
        )
    if evaluation.get("source_station_time_policy") != "all_available":
        raise ValueError(
            "Target-transfer protocol requires "
            "evaluation.source_station_time_policy=all_available"
        )
    validation = evaluation.get("validation", {})
    validation_strategy = validation.get("strategy")
    if validation_strategy not in {"target_history_tail", "target_history_range"}:
        raise ValueError(
            "evaluation.validation.strategy must be target_history_tail "
            "or target_history_range"
        )
    if (
        validation_strategy == "target_history_tail"
        and int(validation.get("target_history_days", 0)) <= 0
    ):
        raise ValueError(
            "evaluation.validation.target_history_days must be positive"
        )
    if validation_strategy == "target_history_range":
        if not validation.get("start") or not validation.get("end"):
            raise ValueError(
                "target_history_range validation requires start and end"
            )
        if pd.Timestamp(validation["start"]) > pd.Timestamp(validation["end"]):
            raise ValueError("evaluation.validation start must not exceed end")
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
            "Test-station-only evaluation requires "
            "evaluation.require_all_training_stations_in_evaluation=false"
        )
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
