from __future__ import annotations

from typing import Any

import pandas as pd

from .config import Config, resolve_test_stations, resolve_training_stations
from .data import STATION_ID


def select_period(
    frame: pd.DataFrame,
    period: dict | None,
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    if not period:
        return frame.iloc[0:0].copy()
    timestamp = pd.to_datetime(frame[timestamp_column])
    mask = pd.Series(True, index=frame.index)
    if period.get("start"):
        mask &= timestamp >= pd.Timestamp(period["start"])
    if period.get("end"):
        end = pd.Timestamp(period["end"])
        if len(str(period["end"])) <= 10:
            end += pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask &= timestamp <= end
    return frame[mask].copy()


def _first_evaluation_start(config: Config) -> pd.Timestamp:
    periods = config["evaluation"]["periods"]
    starts = [
        pd.Timestamp(period["start"])
        for name in ("confirmation", "final_test")
        if (period := periods.get(name)) and period.get("start")
    ]
    if not starts:
        raise ValueError(
            "Configure evaluation.periods.confirmation or final_test with a start"
        )
    return min(starts)


def _exclusive_end(value: Any) -> pd.Timestamp:
    end = pd.Timestamp(value)
    if len(str(value)) <= 10:
        return end + pd.Timedelta(days=1)
    return end + pd.Timedelta(microseconds=1)


def target_transfer_boundaries(config: Config) -> dict[str, Any]:
    evaluation_start = _first_evaluation_start(config)
    validation = config["evaluation"]["validation"]
    purge = pd.Timedelta(hours=float(config["evaluation"]["purge_hours"]))
    strategy = validation["strategy"]
    if strategy == "target_history_tail":
        duration = pd.Timedelta(days=int(validation["target_history_days"]))
        validation_end_exclusive = evaluation_start - purge
        validation_start = validation_end_exclusive - duration
    elif strategy == "target_history_range":
        validation_start = pd.Timestamp(validation["start"])
        validation_end_exclusive = _exclusive_end(validation["end"])
        latest_allowed_end = evaluation_start - purge
        if validation_end_exclusive > latest_allowed_end:
            raise ValueError(
                "Explicit target validation range violates the purge before "
                f"evaluation: validation_end_exclusive={validation_end_exclusive}, "
                f"latest_allowed={latest_allowed_end}"
            )
    else:
        raise ValueError(f"Unknown target validation strategy: {strategy}")
    target_train_end_exclusive = validation_start - purge
    return {
        "evaluation_start": evaluation_start,
        "validation_start": validation_start,
        "validation_end_exclusive": validation_end_exclusive,
        "target_train_end_exclusive": target_train_end_exclusive,
        "purge": purge,
    }


def training_splits(
    frame: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the configured station-role training and historical validation sets."""
    boundaries = target_transfer_boundaries(config)
    stations = frame[STATION_ID].astype(str)
    available_stations = set(stations)
    test_stations = set(resolve_test_stations(config))
    configured_training = resolve_training_stations(config)
    training_stations = (
        available_stations if configured_training is None else set(configured_training)
    )
    missing_training = sorted(training_stations - available_stations)
    missing_test = sorted(test_stations - available_stations)
    if missing_training:
        raise ValueError(f"Configured training stations are absent: {missing_training}")
    if missing_test:
        raise ValueError(f"Configured test stations are absent: {missing_test}")

    target_time = pd.to_datetime(frame["target_timestamp"])
    is_test = stations.isin(test_stations)
    is_training = stations.isin(training_stations)

    source_train = frame[is_training & ~is_test]
    test_history_train = frame[
        is_training
        & is_test
        & (target_time < boundaries["target_train_end_exclusive"])
    ]
    validation = frame[
        is_test
        & (target_time >= boundaries["validation_start"])
        & (target_time < boundaries["validation_end_exclusive"])
    ].copy()
    train = pd.concat(
        [source_train, test_history_train], ignore_index=False
    ).sort_index()

    if train.empty:
        raise ValueError(
            "No rows remain for the configured training stations and historical "
            "cutoff"
        )
    if validation.empty:
        raise ValueError(
            "No test-station rows fall inside the configured historical "
            "validation window"
        )
    present_validation = set(validation[STATION_ID].astype(str))
    missing_validation = sorted(test_stations - present_validation)
    if missing_validation:
        raise ValueError(
            "Configured test stations have no validation rows: "
            f"{missing_validation}"
        )
    overlap = set(train["row_id"]) & set(validation["row_id"])
    if overlap:
        raise ValueError(f"Train/validation row overlap: {len(overlap)}")
    return train.copy(), validation


def select_target_evaluation(
    frame: pd.DataFrame,
    config: Config,
    period_name: str,
) -> pd.DataFrame:
    period = config["evaluation"]["periods"].get(period_name)
    if not period:
        raise ValueError(f"No evaluation period configured for {period_name}")
    test_stations = set(resolve_test_stations(config))
    test_only = frame[frame[STATION_ID].astype(str).isin(test_stations)]
    selected = select_period(
        test_only,
        period,
        timestamp_column="target_timestamp",
    )
    if selected.empty:
        raise ValueError(
            f"No rows for test_stations={sorted(test_stations)!r} in {period_name}"
        )
    present = set(selected[STATION_ID].astype(str))
    missing = sorted(test_stations - present)
    if missing:
        raise ValueError(
            f"Configured test stations have no rows in {period_name}: {missing}"
        )
    return selected


def split_protocol_manifest(config: Config) -> dict[str, Any]:
    boundaries = target_transfer_boundaries(config)
    training_stations = resolve_training_stations(config)
    test_stations = resolve_test_stations(config)
    return {
        "protocol": "configured_station_transfer",
        "training_stations": training_stations,
        "test_stations": test_stations,
        "target_station_compatibility": config["evaluation"].get("target_station"),
        "source_station_time_policy": str(
            config["evaluation"]["source_station_time_policy"]
        ),
        "validation_strategy": config["evaluation"]["validation"]["strategy"],
        "validation_target_history_days": (
            int(config["evaluation"]["validation"]["target_history_days"])
            if config["evaluation"]["validation"]["strategy"]
            == "target_history_tail"
            else None
        ),
        "purge_hours": float(config["evaluation"]["purge_hours"]),
        "evaluation_start": boundaries["evaluation_start"].isoformat(),
        "validation_start": boundaries["validation_start"].isoformat(),
        "validation_end_exclusive": boundaries[
            "validation_end_exclusive"
        ].isoformat(),
        "target_train_end_exclusive": boundaries[
            "target_train_end_exclusive"
        ].isoformat(),
    }
