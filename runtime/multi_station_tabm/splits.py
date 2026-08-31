from __future__ import annotations

import pandas as pd

from .config import Config
from .data import STATION_ID


def select_period(frame: pd.DataFrame, period: dict | None) -> pd.DataFrame:
    if not period:
        return frame.iloc[0:0].copy()
    timestamp = pd.to_datetime(frame["timestamp"])
    mask = pd.Series(True, index=frame.index)
    if period.get("start"):
        mask &= timestamp >= pd.Timestamp(period["start"])
    if period.get("end"):
        end = pd.Timestamp(period["end"])
        if len(str(period["end"])) <= 10:
            end += pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask &= timestamp <= end
    return frame[mask].copy()


def seasonal_train_validation_split(
    frame: pd.DataFrame, validation_last_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp = pd.to_datetime(frame["timestamp"])
    days_remaining = timestamp.dt.days_in_month - timestamp.dt.day
    validation_mask = days_remaining < int(validation_last_days)
    train = frame[~validation_mask].copy()
    validation = frame[validation_mask].copy()
    if train.empty or validation.empty:
        raise ValueError(
            f"Seasonal split produced train={len(train)}, validation={len(validation)}"
        )
    return train, validation


def training_splits(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = config["evaluation"]
    development = select_period(frame, evaluation["periods"]["development"])
    if development.empty:
        raise ValueError("No rows in evaluation.periods.development")
    train, validation = seasonal_train_validation_split(
        development, int(evaluation["validation_last_days_per_month"])
    )
    if bool(evaluation.get("require_same_stations_in_development_splits", True)):
        train_stations = set(train[STATION_ID].astype(str))
        validation_stations = set(validation[STATION_ID].astype(str))
        if train_stations != validation_stations:
            train_only = sorted(train_stations - validation_stations)
            validation_only = sorted(validation_stations - train_stations)
            raise ValueError(
                "Development train/validation station coverage differs: "
                f"train_only={train_only}, validation_only={validation_only}"
            )
    return train, validation
