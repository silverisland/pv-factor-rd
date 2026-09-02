from __future__ import annotations

from typing import Any

import pandas as pd

from .config import (
    Config,
    resolve_test_stations,
    resolve_training_stations,
    resolve_validation_stations,
)
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
    mask = timestamp.between(
        pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
    )
    return frame[mask].copy()


def training_splits(
    frame: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce pv_tabm_baseline's explicit origin-time train/validation split."""
    split = config["split"]
    train = select_period(
        frame,
        {"start": split["train_start"], "end": split["train_end"]},
        timestamp_column="timestamp",
    )
    train_stations = resolve_training_stations(config)
    if train_stations is not None:
        train = train[train[STATION_ID].astype(str).isin(train_stations)].copy()
    strategy = split.get("validation_strategy", "explicit")
    if strategy == "monthly_tail":
        timestamps = pd.to_datetime(train["timestamp"])
        days_remaining = timestamps.dt.days_in_month - timestamps.dt.day
        validation_mask = days_remaining < int(split["validation_last_days"])
        validation = train.loc[validation_mask].copy()
        train = train.loc[~validation_mask].copy()
    else:
        validation = select_period(
            frame,
            {
                "start": split["validation_start"],
                "end": split["validation_end"],
            },
            timestamp_column="timestamp",
        )
        validation_stations = resolve_validation_stations(config)
        if validation_stations is not None:
            validation = validation[
                validation[STATION_ID].astype(str).isin(validation_stations)
            ].copy()
    if train.empty:
        raise ValueError("Configured baseline training period/stations are empty")
    if validation.empty:
        raise ValueError("Configured baseline validation period/stations are empty")
    overlap = set(train["row_id"]) & set(validation["row_id"])
    if overlap:
        raise ValueError(f"Train/validation row overlap: {len(overlap)}")
    return train.copy(), validation


def select_target_evaluation(
    frame: pd.DataFrame,
    config: Config,
    period_name: str,
) -> pd.DataFrame:
    split = config["split"]
    prefix = "test" if period_name == "final_test" else "confirmation"
    start_name, end_name = f"{prefix}_start", f"{prefix}_end"
    if not split.get(start_name) or not split.get(end_name):
        raise ValueError(f"No {period_name} period configured in split")
    selected = select_period(
        frame,
        {"start": split[start_name], "end": split[end_name]},
        timestamp_column="timestamp",
    )
    stations = (
        resolve_test_stations(config)
        if prefix == "test"
        else _optional_station_list(split.get("confirmation_stations"))
    )
    if stations is not None:
        selected = selected[selected[STATION_ID].astype(str).isin(stations)].copy()
    if selected.empty:
        raise ValueError(f"Configured {period_name} period/stations are empty")
    return selected


def _optional_station_list(value) -> list[str] | None:
    return None if value is None else [str(item).strip() for item in value]


def split_protocol_manifest(config: Config) -> dict[str, Any]:
    split = config["split"]
    return {
        "protocol": "pv_tabm_baseline_split",
        "timestamp_column": "timestamp",
        "validation_strategy": str(
            split.get("validation_strategy", "explicit")
        ),
        "validation_last_days": split.get("validation_last_days"),
        "train_start": str(split["train_start"]),
        "train_end": str(split["train_end"]),
        "validation_start": _optional_text(split.get("validation_start")),
        "validation_end": _optional_text(split.get("validation_end")),
        "confirmation_start": _optional_text(split.get("confirmation_start")),
        "confirmation_end": _optional_text(split.get("confirmation_end")),
        "test_start": str(split["test_start"]),
        "test_end": str(split["test_end"]),
        "train_stations": resolve_training_stations(config),
        "validation_stations": resolve_validation_stations(config),
        "confirmation_stations": _optional_station_list(
            split.get("confirmation_stations")
        ),
        "test_stations": resolve_test_stations(config),
    }


def _optional_text(value) -> str | None:
    return None if value is None else str(value)
