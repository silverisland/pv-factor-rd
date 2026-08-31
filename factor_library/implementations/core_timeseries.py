"""Small, causal reference implementations for private adapter integration.

Inputs are plain numeric sequences ordered oldest-to-newest. These functions do
not impute values or learn statistics; the private adapter owns partition-aware
preprocessing and deterministic column naming.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from typing import Union


Number = Union[int, float]


def _values(sequence: Sequence[Number]) -> list[float]:
    values = [float(value) for value in sequence]
    if not values:
        raise ValueError("sequence must not be empty")
    return values


def _finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else math.nan


def build_power_lags(power_history: Sequence[Number], lags: Iterable[int] = (1, 2, 4, 8, 16, 32, 96)) -> dict[str, float]:
    values = _values(power_history)
    output: dict[str, float] = {}
    for lag in lags:
        lag = int(lag)
        if lag < 1:
            raise ValueError("lags must be positive")
        output[f"power_lag_{lag}"] = _finite_or_nan(values[-lag]) if len(values) >= lag else math.nan
    return output


def build_power_ramps(power_history: Sequence[Number], windows: Iterable[int] = (1, 2, 4, 8, 16)) -> dict[str, float]:
    values = _values(power_history)
    output: dict[str, float] = {}
    for window in windows:
        window = int(window)
        if window < 1:
            raise ValueError("windows must be positive")
        output[f"power_ramp_{window}"] = values[-1] - values[-1 - window] if len(values) > window else math.nan
    return output


def build_power_slopes(power_history: Sequence[Number], windows: Iterable[int] = (4, 8, 16, 32)) -> dict[str, float]:
    values = _values(power_history)
    output: dict[str, float] = {}
    for window in windows:
        window = int(window)
        if window < 2:
            raise ValueError("slope windows must be at least two")
        if len(values) < window:
            output[f"power_slope_{window}"] = math.nan
            continue
        sample = values[-window:]
        x_mean = (window - 1) / 2.0
        denominator = sum((x - x_mean) ** 2 for x in range(window))
        y_mean = sum(sample) / window
        output[f"power_slope_{window}"] = sum((x - x_mean) * (y - y_mean) for x, y in enumerate(sample)) / denominator
    return output


def build_power_variability(power_history: Sequence[Number], windows: Iterable[int] = (4, 8, 16)) -> dict[str, float]:
    values = _values(power_history)
    output: dict[str, float] = {}
    for window in windows:
        window = int(window)
        if len(values) < window or window < 2:
            output[f"power_std_{window}"] = math.nan
            output[f"power_absdiff_{window}"] = math.nan
            output[f"power_range_{window}"] = math.nan
            continue
        sample = values[-window:]
        output[f"power_std_{window}"] = statistics.pstdev(sample)
        output[f"power_absdiff_{window}"] = sum(abs(right - left) for left, right in zip(sample, sample[1:])) / (window - 1)
        output[f"power_range_{window}"] = max(sample) - min(sample)
    return output


def build_power_acceleration(
    power_history: Sequence[Number], steps: Iterable[int] = (1,)
) -> dict[str, float]:
    values = _values(power_history)
    output: dict[str, float] = {}
    for step in steps:
        step = int(step)
        if step < 1:
            raise ValueError("acceleration steps must be positive")
        output[f"power_acceleration_{step}"] = (
            (values[-1] - values[-1 - step])
            - (values[-1 - step] - values[-1 - 2 * step])
            if len(values) > 2 * step
            else math.nan
        )
    return output


def build_future_changes(future_values: Sequence[Number], last_observed: Number) -> dict[str, float]:
    values = _values(future_values)
    previous = float(last_observed)
    output: dict[str, float] = {}
    for horizon, value in enumerate(values, start=1):
        output[f"future_change_from_origin_h{horizon}"] = value - float(last_observed)
        output[f"future_step_change_h{horizon}"] = value - previous
        previous = value
    return output


def build_stuck_shift(values: Sequence[Number], tolerance: float = 1e-9) -> dict[str, float]:
    sample = _values(values)
    longest = current = 1
    differences: list[float] = []
    for left, right in zip(sample, sample[1:]):
        difference = abs(right - left)
        differences.append(difference)
        current = current + 1 if difference <= tolerance else 1
        longest = max(longest, current)
    return {
        "stuck_longest_fraction": longest / len(sample),
        "mean_absolute_step": sum(differences) / len(differences) if differences else 0.0,
    }


def build_future_weather_quality(sequences: dict[str, Sequence[Number]], expected_steps: int = 16) -> dict[str, float]:
    if expected_steps < 1:
        raise ValueError("expected_steps must be positive")
    output: dict[str, float] = {}
    all_values: list[float] = []
    for name, sequence in sorted(sequences.items()):
        values = [float(value) for value in sequence[:expected_steps]]
        valid = sum(math.isfinite(value) for value in values)
        output[f"{name}_coverage"] = valid / expected_steps
        output[f"{name}_length_ratio"] = min(len(sequence), expected_steps) / expected_steps
        all_values.extend(values)
    output["future_weather_joint_coverage"] = (
        sum(math.isfinite(value) for value in all_values) / (len(sequences) * expected_steps)
        if sequences else 0.0
    )
    return output
