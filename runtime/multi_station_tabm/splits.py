from __future__ import annotations

from typing import Any

import pandas as pd

from .config import Config
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


def target_transfer_boundaries(config: Config) -> dict[str, Any]:
    evaluation_start = _first_evaluation_start(config)
    validation = config["evaluation"]["validation"]
    duration = pd.Timedelta(days=int(validation["target_history_days"]))
    purge = pd.Timedelta(hours=float(config["evaluation"]["purge_hours"]))
    validation_end_exclusive = evaluation_start - purge
    validation_start = validation_end_exclusive - duration
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
    """Build source-all + target-history training and target-history validation."""
    evaluation = config["evaluation"]
    target_station = str(evaluation["target_station"]).strip()
    boundaries = target_transfer_boundaries(config)
    stations = frame[STATION_ID].astype(str)
    target_time = pd.to_datetime(frame["target_timestamp"])
    is_target = stations.eq(target_station)
    is_source = ~is_target

    source_train = frame[is_source]
    target_train = frame[
        is_target
        & (target_time < boundaries["target_train_end_exclusive"])
    ]
    validation = frame[
        is_target
        & (target_time >= boundaries["validation_start"])
        & (target_time < boundaries["validation_end_exclusive"])
    ].copy()
    train = pd.concat([source_train, target_train], ignore_index=False).sort_index()

    if source_train.empty:
        raise ValueError("At least one non-target source station is required")
    if target_train.empty:
        raise ValueError(
            "No target-station history remains for training before the purged "
            "validation window"
        )
    if validation.empty:
        raise ValueError(
            "No target-station rows fall inside the configured historical "
            "validation window"
        )
    validation_stations = set(validation[STATION_ID].astype(str))
    if validation_stations != {target_station}:
        raise AssertionError(
            f"Validation must contain only target_station={target_station!r}: "
            f"{sorted(validation_stations)}"
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
    target_station = str(config["evaluation"]["target_station"]).strip()
    target_only = frame[frame[STATION_ID].astype(str).eq(target_station)]
    selected = select_period(
        target_only,
        period,
        timestamp_column="target_timestamp",
    )
    if selected.empty:
        raise ValueError(
            f"No target-station rows for station={target_station!r} in {period_name}"
        )
    if set(selected[STATION_ID].astype(str)) != {target_station}:
        raise AssertionError("Target evaluation contains a non-target station")
    return selected


def split_protocol_manifest(config: Config) -> dict[str, Any]:
    boundaries = target_transfer_boundaries(config)
    return {
        "protocol": "target_station_transfer",
        "target_station": str(config["evaluation"]["target_station"]),
        "source_station_time_policy": str(
            config["evaluation"]["source_station_time_policy"]
        ),
        "validation_strategy": "target_history_tail",
        "validation_target_history_days": int(
            config["evaluation"]["validation"]["target_history_days"]
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
