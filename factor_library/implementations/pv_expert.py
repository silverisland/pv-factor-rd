"""Causal PV-domain factors for aligned power, irradiance, and site metadata.

The solar geometry follows the NOAA fractional-year approximation.  Clear-sky
GHI uses the Haurwitz expression, so the implementation is deterministic and
does not require an online ephemeris or an additional runtime package.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from statistics import fmean, pstdev
from typing import Iterable, Sequence


EPSILON = 1e-6


def _floats(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    converter = getattr(value, "to_pydatetime", None)
    if converter is not None:
        return converter()
    return datetime.fromisoformat(str(value))


def build_capacity_ratio(
    power_history: Sequence[float], capacity: float, lags: Sequence[int] = (1, 4, 16, 96)
) -> dict[str, float]:
    values = _floats(power_history)
    scale = float(capacity)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("capacity must be finite and positive")
    result: dict[str, float] = {}
    for lag in lags:
        if lag <= 0 or lag > len(values):
            raise ValueError(f"capacity-ratio lag {lag} exceeds available history")
        result[f"power_capacity_ratio_lag_{lag}"] = values[-lag] / scale
    return result


def build_previous_day_profile(
    power_history: Sequence[float], horizon_step: int, steps_per_day: int = 96
) -> dict[str, float]:
    values = _floats(power_history)
    step = int(horizon_step)
    if not 1 <= step < steps_per_day:
        raise ValueError("horizon_step must be within the previous-day history")
    index = -steps_per_day + step
    if len(values) < steps_per_day:
        raise ValueError(f"previous-day profile requires {steps_per_day} history points")
    return {"previous_day_same_target_power": values[index]}


def solar_position(
    timestamp: object,
    latitude: float,
    longitude: float,
    utc_offset_hours: float = 8.0,
) -> dict[str, float]:
    """Approximate apparent solar position for a local civil timestamp."""
    moment = _timestamp(timestamp)
    latitude_radians = math.radians(float(latitude))
    day = moment.timetuple().tm_yday
    hour = moment.hour + moment.minute / 60.0 + moment.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hour - 12.0) / 24.0)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    time_offset = equation_of_time + 4.0 * float(longitude) - 60.0 * float(utc_offset_hours)
    true_solar_minutes = (hour * 60.0 + time_offset) % 1440.0
    hour_angle_degrees = true_solar_minutes / 4.0 - 180.0
    hour_angle = math.radians(hour_angle_degrees)
    cos_zenith = (
        math.sin(latitude_radians) * math.sin(declination)
        + math.cos(latitude_radians) * math.cos(declination) * math.cos(hour_angle)
    )
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    elevation_degrees = 90.0 - math.degrees(zenith)
    azimuth = math.degrees(
        math.atan2(
            math.sin(hour_angle),
            math.cos(hour_angle) * math.sin(latitude_radians)
            - math.tan(declination) * math.cos(latitude_radians),
        )
    )
    azimuth_degrees = (azimuth + 180.0) % 360.0
    return {
        "solar_elevation_degrees": elevation_degrees,
        "solar_zenith_degrees": math.degrees(zenith),
        "solar_azimuth_degrees": azimuth_degrees,
        "solar_cos_zenith": max(0.0, cos_zenith),
        "equation_of_time_minutes": equation_of_time,
        "solar_declination_radians": declination,
    }


def build_daylight_boundary(
    timestamp: object,
    latitude: float,
    longitude: float,
    utc_offset_hours: float = 8.0,
    position: dict[str, float] | None = None,
) -> dict[str, float]:
    moment = _timestamp(timestamp)
    position = position or solar_position(
        moment, latitude, longitude, utc_offset_hours
    )
    declination = position["solar_declination_radians"]
    latitude_radians = math.radians(float(latitude))
    cosine_hour_angle = -math.tan(latitude_radians) * math.tan(declination)
    if cosine_hour_angle >= 1.0:
        daylight_minutes = 0.0
    elif cosine_hour_angle <= -1.0:
        daylight_minutes = 1440.0
    else:
        daylight_minutes = 8.0 * math.degrees(math.acos(cosine_hour_angle))
    solar_noon = (
        720.0
        - 4.0 * float(longitude)
        - position["equation_of_time_minutes"]
        + 60.0 * float(utc_offset_hours)
    )
    sunrise = solar_noon - daylight_minutes / 2.0
    sunset = solar_noon + daylight_minutes / 2.0
    local_minutes = moment.hour * 60.0 + moment.minute + moment.second / 60.0
    if daylight_minutes <= 0.0:
        progress = 0.0
    else:
        progress = max(0.0, min(1.0, (local_minutes - sunrise) / daylight_minutes))
    return {
        "minutes_since_sunrise": local_minutes - sunrise,
        "minutes_until_sunset": sunset - local_minutes,
        "daylight_duration_minutes": daylight_minutes,
        "daylight_progress": progress,
        "is_daylight": float(position["solar_elevation_degrees"] > 0.0),
    }


def build_clear_sky_irradiance(
    timestamp: object,
    latitude: float,
    longitude: float,
    utc_offset_hours: float = 8.0,
    position: dict[str, float] | None = None,
) -> dict[str, float]:
    position = position or solar_position(
        timestamp, latitude, longitude, utc_offset_hours
    )
    cos_zenith = position["solar_cos_zenith"]
    ghi = 0.0 if cos_zenith <= 0.0 else 1098.0 * cos_zenith * math.exp(-0.059 / cos_zenith)
    return {"clear_sky_ghi": ghi, "clear_sky_ghi_normalized": ghi / 1000.0}


def build_power_clear_sky_index(
    power_history: Sequence[float],
    history_end_timestamp: object,
    capacity: float,
    latitude: float,
    longitude: float,
    minutes_per_point: int = 15,
    utc_offset_hours: float = 8.0,
    lags: Sequence[int] = (1, 4, 16, 96),
) -> dict[str, float]:
    values = _floats(power_history)
    origin = _timestamp(history_end_timestamp)
    result: dict[str, float] = {}
    for lag in lags:
        if lag > len(values):
            raise ValueError(f"clear-sky index lag {lag} exceeds available history")
        moment = origin - timedelta(minutes=lag * int(minutes_per_point))
        clear_ghi = build_clear_sky_irradiance(
            moment, latitude, longitude, utc_offset_hours
        )["clear_sky_ghi"]
        if clear_ghi < 20.0:
            index = 0.0
        else:
            index = max(0.0, min(2.0, (values[-lag] / float(capacity)) / (clear_ghi / 1000.0)))
        result[f"power_clear_sky_index_lag_{lag}"] = index
    return result


def build_module_temperature_proxy(
    irradiance: float,
    air_temperature: float,
    wind_speed: float,
    faiman_u0: float = 25.0,
    faiman_u1: float = 6.84,
) -> dict[str, float]:
    denominator = max(EPSILON, float(faiman_u0) + float(faiman_u1) * max(0.0, float(wind_speed)))
    module_temperature = float(air_temperature) + max(0.0, float(irradiance)) / denominator
    return {
        "module_temperature_proxy": module_temperature,
        "module_air_temperature_delta": module_temperature - float(air_temperature),
    }


def build_temperature_corrected_irradiance(
    irradiance: float,
    module_temperature: float,
    temperature_coefficient_per_c: float = -0.004,
    reference_temperature_c: float = 25.0,
) -> dict[str, float]:
    correction = 1.0 + float(temperature_coefficient_per_c) * (
        float(module_temperature) - float(reference_temperature_c)
    )
    return {
        "temperature_corrected_irradiance": max(0.0, float(irradiance) * correction),
        "temperature_derating_multiplier": correction,
    }


def build_low_irradiance_state(
    irradiance: float, clear_sky_irradiance: float, solar_elevation_degrees: float
) -> dict[str, float]:
    ghi = max(0.0, float(irradiance))
    clear = max(0.0, float(clear_sky_irradiance))
    ratio = 0.0 if clear < 20.0 else max(0.0, min(2.0, ghi / clear))
    return {
        "irradiance_clear_sky_ratio": ratio,
        "low_irradiance_flag": float(0.0 < ghi < 200.0),
        "low_sun_flag": float(0.0 < float(solar_elevation_degrees) < 10.0),
    }


def _recent_clear_sky_ratios(
    ghi_history: Sequence[float],
    origin_timestamp: object,
    latitude: float,
    longitude: float,
    minutes_per_point: int,
    utc_offset_hours: float,
    recent_steps: int = 16,
) -> list[float]:
    if recent_steps < 1:
        raise ValueError("recent_steps must be positive")
    # The regime outputs use only the trailing window. Slicing first avoids 80
    # unnecessary solar-position calculations for the standard 96-point input.
    values = _floats(ghi_history)[-recent_steps:]
    origin = _timestamp(origin_timestamp)
    ratios = []
    for index, ghi in enumerate(reversed(values), start=1):
        moment = origin - timedelta(minutes=index * minutes_per_point)
        clear = build_clear_sky_irradiance(moment, latitude, longitude, utc_offset_hours)["clear_sky_ghi"]
        ratios.append(0.0 if clear < 20.0 else max(0.0, min(2.0, ghi / clear)))
    ratios.reverse()
    return ratios


def build_clear_variable_overcast_regime(
    ghi_history: Sequence[float],
    origin_timestamp: object,
    latitude: float,
    longitude: float,
    minutes_per_point: int = 15,
    utc_offset_hours: float = 8.0,
) -> dict[str, float]:
    ratios = _recent_clear_sky_ratios(
        ghi_history, origin_timestamp, latitude, longitude, minutes_per_point, utc_offset_hours
    )
    recent = ratios
    diffs = [abs(right - left) for left, right in zip(recent, recent[1:])]
    mean_ratio = fmean(recent) if recent else 0.0
    variability = pstdev(recent) if len(recent) > 1 else 0.0
    rampiness = fmean(diffs) if diffs else 0.0
    return {
        "recent_clear_sky_ratio_mean": mean_ratio,
        "recent_clear_sky_ratio_std": variability,
        "recent_clear_sky_ratio_absdiff": rampiness,
        "clear_regime_score": max(0.0, min(1.0, mean_ratio)) * max(0.0, 1.0 - variability),
        "variable_regime_score": max(0.0, min(1.0, variability + rampiness)),
        "overcast_regime_score": max(0.0, min(1.0, 1.0 - mean_ratio)),
    }


def build_joint_weather_power_ramp(
    power_history: Sequence[float],
    ghi_history: Sequence[float],
    future_ghi: float,
    capacity: float,
) -> dict[str, float]:
    power = _floats(power_history)
    ghi = _floats(ghi_history)
    if len(power) < 2 or len(ghi) < 2:
        raise ValueError("joint ramp requires at least two aligned history points")
    return {
        "recent_power_ramp_capacity_ratio": (power[-1] - power[-2]) / float(capacity),
        "recent_ghi_ramp_normalized": (ghi[-1] - ghi[-2]) / 1000.0,
        "future_ghi_change_from_observation": (float(future_ghi) - ghi[-1]) / 1000.0,
    }


def build_clipping_score(
    power_history: Sequence[float], ghi_history: Sequence[float], capacity: float
) -> dict[str, float]:
    power = _floats(power_history)[-8:]
    ghi = _floats(ghi_history)[-8:]
    if len(power) != len(ghi) or len(power) < 2:
        raise ValueError("clipping score requires aligned power and GHI histories")
    normalized = [value / float(capacity) for value in power]
    near_capacity = fmean(float(value >= 0.98) for value in normalized)
    high_ghi = fmean(float(value >= 800.0) for value in ghi)
    plateau = max(0.0, 1.0 - pstdev(normalized) / 0.02)
    return {
        "near_capacity_fraction": near_capacity,
        "high_ghi_fraction": high_ghi,
        "power_plateau_score": plateau,
        "clipping_score": near_capacity * high_ghi * plateau,
    }


def build_weather_power_residual(
    power_history: Sequence[float], ghi_history: Sequence[float], capacity: float
) -> dict[str, float]:
    power = _floats(power_history)[-16:]
    ghi = _floats(ghi_history)[-16:]
    if len(power) != len(ghi) or not power:
        raise ValueError("weather-power residual requires aligned histories")
    residuals = [
        power_value / float(capacity) - max(0.0, min(1.2, ghi_value / 1000.0))
        for power_value, ghi_value in zip(power, ghi)
    ]
    return {
        "weather_power_residual_latest": residuals[-1],
        "weather_power_residual_mean_16": fmean(residuals),
        "weather_power_residual_std_16": pstdev(residuals) if len(residuals) > 1 else 0.0,
    }
