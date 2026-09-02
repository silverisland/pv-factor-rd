from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import STATION_ID


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs must contain only finite values")
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def regression_metrics(
    target: np.ndarray, prediction: np.ndarray, capacity: float
) -> dict[str, float]:
    if len(target) != len(prediction) or len(target) == 0:
        raise ValueError(
            "Metric arrays must be non-empty and aligned: "
            f"target={len(target)}, prediction={len(prediction)}"
        )
    if capacity <= 0:
        raise ValueError("score capacity must be positive")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs must contain only finite values")
    error = prediction - target
    value = rmse(target, prediction)
    return {
        "rmse": value,
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "nrmse_by_capacity": value / capacity,
        "capacity_score": 1.0 - value / capacity,
    }


def _metrics_for_group(group: pd.DataFrame, capacity: float) -> dict[str, float]:
    return regression_metrics(
        group["groundtruth"].to_numpy(dtype=np.float32),
        group["prediction"].to_numpy(dtype=np.float32),
        capacity,
    )


def station_metrics(predictions: pd.DataFrame, capacity: float) -> pd.DataFrame:
    rows = []
    for (horizon, station), group in predictions.groupby(
        ["horizon_step", STATION_ID], sort=True
    ):
        rows.append(
            {
                "horizon_step": int(horizon),
                STATION_ID: str(station),
                "sample_count": len(group),
                **_metrics_for_group(group, capacity),
            }
        )
    if not rows:
        raise ValueError("No station prediction groups are available")
    return pd.DataFrame(rows)


def station_macro_summary(by_station: pd.DataFrame) -> pd.DataFrame:
    if by_station.empty:
        raise ValueError("Station metrics are empty")
    return (
        by_station.groupby("horizon_step", as_index=False)
        .agg(
            station_count=(STATION_ID, "count"),
            macro_rmse=("rmse", "mean"),
            macro_mae=("mae", "mean"),
            macro_abs_bias=("bias", lambda value: float(np.mean(np.abs(value)))),
            macro_capacity_score=("capacity_score", "mean"),
            worst_station_rmse=("rmse", "max"),
            worst_station_capacity_score=("capacity_score", "min"),
        )
    )


def daily_and_monthly_metrics(
    predictions: pd.DataFrame, capacity: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        STATION_ID,
        "timestamp",
        "target_timestamp",
        "groundtruth",
        "prediction",
        "horizon_step",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table missing metric columns: {missing}")
    current = predictions.copy()
    # Evaluation intervals and reporting both describe the forecast target time.
    current["date"] = pd.to_datetime(current["target_timestamp"]).dt.normalize()
    daily_rows = []
    for (horizon, station, date), group in current.groupby(
        ["horizon_step", STATION_ID, "date"], sort=True
    ):
        daily_rows.append(
            {
                "horizon_step": int(horizon),
                STATION_ID: str(station),
                "date": date,
                "sample_count": len(group),
                **_metrics_for_group(group, capacity),
            }
        )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise ValueError("Daily station metrics are empty")
    daily["month"] = pd.to_datetime(daily["date"]).dt.month
    monthly = (
        daily.groupby(["horizon_step", STATION_ID, "month"], as_index=False)
        .agg(
            days=("date", "count"),
            mean_daily_rmse=("rmse", "mean"),
            mean_daily_mae=("mae", "mean"),
            mean_daily_bias=("bias", "mean"),
        )
    )
    monthly["capacity_score"] = 1.0 - monthly["mean_daily_rmse"] / capacity
    return daily, monthly


def monthly_score_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        raise ValueError("Monthly metrics are empty")
    return (
        monthly.groupby(["horizon_step", STATION_ID], as_index=False)
        .agg(
            months=("month", "count"),
            mean_monthly_capacity_score=("capacity_score", "mean"),
            worst_monthly_capacity_score=("capacity_score", "min"),
        )
    )


def pooled_monthly_score_summary(
    predictions: pd.DataFrame, capacity: float
) -> pd.DataFrame:
    """Reproduce pv_tabm_baseline's score without station-wise averaging.

    For each horizon, all configured test-station rows are pooled. RMSE is
    computed per target date, daily RMSE is averaged within each calendar
    month, and the months present in the test partition receive equal weight.
    """
    if capacity <= 0:
        raise ValueError("score capacity must be positive")
    required = {
        "horizon_step", "target_timestamp", "groundtruth", "prediction"
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table missing metric columns: {missing}")
    current = predictions.copy()
    current["date"] = pd.to_datetime(current["target_timestamp"]).dt.floor("D")
    error = (
        current["prediction"].to_numpy(dtype=np.float64)
        - current["groundtruth"].to_numpy(dtype=np.float64)
    )
    if not np.isfinite(error).all():
        raise ValueError("Metric inputs must contain only finite values")
    current["squared_error"] = np.square(error)
    daily = (
        current.groupby(["horizon_step", "date"], as_index=False)["squared_error"]
        .mean()
    )
    daily["daily_rmse"] = np.sqrt(daily["squared_error"])
    daily["month"] = daily["date"].dt.month
    monthly = (
        daily.groupby(["horizon_step", "month"], as_index=False)["daily_rmse"]
        .mean()
        .rename(columns={"daily_rmse": "mean_daily_rmse"})
    )
    monthly["capacity_score"] = 1.0 - monthly["mean_daily_rmse"] / capacity
    return (
        monthly.groupby("horizon_step", as_index=False)
        .agg(
            months=("month", "count"),
            mean_monthly_capacity_score=("capacity_score", "mean"),
            worst_monthly_capacity_score=("capacity_score", "min"),
        )
    )


def horizon_group(horizon_step: int, minutes_per_point: int) -> str:
    minutes = horizon_step * minutes_per_point
    if minutes <= 60:
        return "0_1h"
    if minutes <= 120:
        return "1_2h"
    return "2_4h"


def grouped_horizon_metrics(
    predictions: pd.DataFrame, capacity: float, minutes_per_point: int
) -> pd.DataFrame:
    current = predictions.copy()
    current["horizon_group"] = current["horizon_step"].map(
        lambda value: horizon_group(int(value), minutes_per_point)
    )
    rows: list[dict[str, Any]] = []
    for name, group in current.groupby("horizon_group", sort=False):
        rows.append(
            {
                "horizon_group": name,
                "sample_count": len(group),
                "station_count": group[STATION_ID].nunique(),
                **_metrics_for_group(group, capacity),
            }
        )
    return pd.DataFrame(rows)
