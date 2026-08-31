"""Bind catalog factor IDs to causal, deterministic feature builders."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .core_timeseries import (
    build_future_weather_quality,
    build_power_acceleration,
    build_power_lags,
    build_power_ramps,
    build_power_slopes,
    build_power_variability,
    build_stuck_shift,
)
from .pv_expert import (
    build_capacity_ratio,
    build_clear_sky_irradiance,
    build_clear_variable_overcast_regime,
    build_clipping_score,
    build_daylight_boundary,
    build_joint_weather_power_ramp,
    build_low_irradiance_state,
    build_module_temperature_proxy,
    build_power_clear_sky_index,
    build_previous_day_profile,
    build_temperature_corrected_irradiance,
    build_weather_power_residual,
    solar_position,
)


SKILL_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = SKILL_ROOT / "factor_library" / "factors.json"
Builder = Callable[[dict[str, Any], dict[str, Any], int], dict[str, float]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _catalog() -> dict[str, dict[str, Any]]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)["factors"]
    return {record["id"]: record for record in records}


def _history(row: dict[str, Any], config: dict[str, Any]) -> list[float]:
    column = config["data"]["columns"]["power_history"]
    return np.asarray(row[column], dtype=np.float32).reshape(-1).tolist()


def _weather(row: dict[str, Any], config: dict[str, Any]) -> dict[str, list[float]]:
    columns = config["data"]["columns"]["future_weather"]
    return {
        column: np.asarray(row[column], dtype=np.float32).reshape(-1).tolist()
        for column in columns
    }


def _ghi_history(row: dict[str, Any], config: dict[str, Any]) -> list[float]:
    column = config["data"]["columns"].get("ghi_history")
    if not column:
        raise ValueError("Selected factor requires data.columns.ghi_history")
    return np.asarray(row[column], dtype=np.float32).reshape(-1).tolist()


def _site_value(row: dict[str, Any], name: str) -> float:
    if name not in row or pd.isna(row[name]):
        raise ValueError(
            f"Selected factor requires attached site metadata column {name!r}"
        )
    return float(row[name])


def _utc_offset(config: dict[str, Any]) -> float:
    timezone = str(
        config["data"].get("site_metadata", {}).get("timezone", "Asia/Shanghai")
    )
    if timezone != "Asia/Shanghai":
        raise ValueError(
            "The bundled solar implementation currently requires timezone=Asia/Shanghai"
        )
    return 8.0


def _origin_timestamp(row: dict[str, Any], config: dict[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(row[config["data"]["columns"]["timestamp"]])


def _target_timestamp(
    row: dict[str, Any], config: dict[str, Any], horizon_step: int
) -> pd.Timestamp:
    minutes = int(config["features"]["minutes_per_point"])
    return _origin_timestamp(row, config) + timedelta(minutes=horizon_step * minutes)


def _weather_target(
    row: dict[str, Any], config: dict[str, Any], role: str, horizon_step: int
) -> float:
    roles = config["data"].get("weather_roles", {})
    column = roles.get(role)
    if not column:
        raise ValueError(f"Selected factor requires data.weather_roles.{role}")
    if column not in config["data"]["columns"]["future_weather"]:
        raise ValueError(
            f"Weather role {role!r} references {column!r}, which is not a future_weather column"
        )
    values = np.asarray(row[column], dtype=np.float32).reshape(-1)
    if len(values) < horizon_step:
        raise ValueError(f"{column} has fewer than {horizon_step} forecast steps")
    return float(values[horizon_step - 1])


def _solar_inputs(row: dict[str, Any]) -> tuple[float, float]:
    return _site_value(row, "site_latitude"), _site_value(row, "site_longitude")


def _lags(row: dict[str, Any], config: dict[str, Any], _: int) -> dict[str, float]:
    return build_power_lags(_history(row, config), (1, 2, 4, 8, 16, 96))


def _ramps(row: dict[str, Any], config: dict[str, Any], _: int) -> dict[str, float]:
    return build_power_ramps(_history(row, config), (1, 2, 4, 8, 16))


def _slopes(row: dict[str, Any], config: dict[str, Any], _: int) -> dict[str, float]:
    return build_power_slopes(_history(row, config), (2, 4, 8, 16))


def _variability(
    row: dict[str, Any], config: dict[str, Any], _: int
) -> dict[str, float]:
    return build_power_variability(_history(row, config), (2, 4, 8, 16))


def _acceleration(
    row: dict[str, Any], config: dict[str, Any], _: int
) -> dict[str, float]:
    return build_power_acceleration(_history(row, config), (1, 2, 4))


def _stuck(row: dict[str, Any], config: dict[str, Any], _: int) -> dict[str, float]:
    return build_stuck_shift(_history(row, config))


def _weather_quality(
    row: dict[str, Any], config: dict[str, Any], _: int
) -> dict[str, float]:
    expected = int(config["features"]["n_horizons"])
    return build_future_weather_quality(_weather(row, config), expected)


def _capacity_ratio(
    row: dict[str, Any], config: dict[str, Any], _: int
) -> dict[str, float]:
    return build_capacity_ratio(_history(row, config), _site_value(row, "site_capacity"))


def _previous_day(
    row: dict[str, Any], config: dict[str, Any], horizon_step: int
) -> dict[str, float]:
    steps_per_day = round(24 * 60 / int(config["features"]["minutes_per_point"]))
    return build_previous_day_profile(_history(row, config), horizon_step, steps_per_day)


def _solar_position(
    row: dict[str, Any], config: dict[str, Any], horizon_step: int
) -> dict[str, float]:
    latitude, longitude = _solar_inputs(row)
    return solar_position(
        _target_timestamp(row, config, horizon_step),
        latitude,
        longitude,
        _utc_offset(config),
    )


def _daylight(
    row: dict[str, Any], config: dict[str, Any], horizon_step: int
) -> dict[str, float]:
    latitude, longitude = _solar_inputs(row)
    return build_daylight_boundary(
        _target_timestamp(row, config, horizon_step),
        latitude,
        longitude,
        _utc_offset(config),
    )


def _clear_sky(
    row: dict[str, Any], config: dict[str, Any], horizon_step: int
) -> dict[str, float]:
    latitude, longitude = _solar_inputs(row)
    return build_clear_sky_irradiance(
        _target_timestamp(row, config, horizon_step),
        latitude,
        longitude,
        _utc_offset(config),
    )


def _power_clear_sky(
    row: dict[str, Any], config: dict[str, Any], _: int
) -> dict[str, float]:
    latitude, longitude = _solar_inputs(row)
    return build_power_clear_sky_index(
        _history(row, config),
        _origin_timestamp(row, config),
        _site_value(row, "site_capacity"),
        latitude,
        longitude,
        int(config["features"]["minutes_per_point"]),
        _utc_offset(config),
    )


def _module_temperature(
    row: dict[str, Any], config: dict[str, Any], horizon_step: int
) -> dict[str, float]:
    parameters = config.get("factor_parameters", {})
    return build_module_temperature_proxy(
        _weather_target(row, config, "ghi", horizon_step),
        _weather_target(row, config, "temperature", horizon_step),
        _weather_target(row, config, "wind_speed", horizon_step),
        parameters.get("faiman_u0", 25.0),
        parameters.get("faiman_u1", 6.84),
    )


def _temperature_corrected(
    row: dict[str, Any], config: dict[str, Any], horizon_step: int
) -> dict[str, float]:
    parameters = config.get("factor_parameters", {})
    irradiance = _weather_target(row, config, "ghi", horizon_step)
    module = _module_temperature(row, config, horizon_step)["module_temperature_proxy"]
    return build_temperature_corrected_irradiance(
        irradiance,
        module,
        parameters.get("temperature_coefficient_per_c", -0.004),
        parameters.get("reference_temperature_c", 25.0),
    )


def _low_irradiance(
    row: dict[str, Any], config: dict[str, Any], horizon_step: int
) -> dict[str, float]:
    position = _solar_position(row, config, horizon_step)
    clear = _clear_sky(row, config, horizon_step)
    return build_low_irradiance_state(
        _weather_target(row, config, "ghi", horizon_step),
        clear["clear_sky_ghi"],
        position["solar_elevation_degrees"],
    )


def _weather_regime(
    row: dict[str, Any], config: dict[str, Any], _: int
) -> dict[str, float]:
    latitude, longitude = _solar_inputs(row)
    return build_clear_variable_overcast_regime(
        _ghi_history(row, config),
        _origin_timestamp(row, config),
        latitude,
        longitude,
        int(config["features"]["minutes_per_point"]),
        _utc_offset(config),
    )


def _joint_ramp(
    row: dict[str, Any], config: dict[str, Any], horizon_step: int
) -> dict[str, float]:
    return build_joint_weather_power_ramp(
        _history(row, config),
        _ghi_history(row, config),
        _weather_target(row, config, "ghi", horizon_step),
        _site_value(row, "site_capacity"),
    )


def _clipping(
    row: dict[str, Any], config: dict[str, Any], _: int
) -> dict[str, float]:
    return build_clipping_score(
        _history(row, config), _ghi_history(row, config), _site_value(row, "site_capacity")
    )


def _weather_power_residual(
    row: dict[str, Any], config: dict[str, Any], _: int
) -> dict[str, float]:
    return build_weather_power_residual(
        _history(row, config), _ghi_history(row, config), _site_value(row, "site_capacity")
    )


BUILDERS: dict[str, Builder] = {
    "factor.power.capacity-ratio": _capacity_ratio,
    "factor.power.multiscale-lags": _lags,
    "factor.power.multiscale-ramp": _ramps,
    "factor.power.multiscale-slope": _slopes,
    "factor.power.variability": _variability,
    "factor.power.acceleration": _acceleration,
    "factor.power.previous-day-profile": _previous_day,
    "factor.solar.position": _solar_position,
    "factor.solar.daylight-boundary": _daylight,
    "factor.solar.clear-sky-irradiance": _clear_sky,
    "factor.solar.power-clear-sky-index": _power_clear_sky,
    "factor.pv.module-temperature-proxy": _module_temperature,
    "factor.pv.temperature-corrected-irradiance": _temperature_corrected,
    "factor.pv.low-irradiance-state": _low_irradiance,
    "factor.regime.clear-variable-overcast": _weather_regime,
    "factor.regime.joint-weather-power-ramp": _joint_ramp,
    "factor.operation.clipping-score": _clipping,
    "factor.operation.weather-power-residual": _weather_power_residual,
    "factor.quality.stuck-shift-score": _stuck,
    "factor.quality.future-weather-coverage": _weather_quality,
}


def executable_factor_ids() -> list[str]:
    """Return factors wired into the fixed TabM feature builder."""
    return sorted(BUILDERS)


def validate_factor_ids(factor_ids: Sequence[str] | None) -> list[str]:
    selected = list(factor_ids or [])
    if len(selected) != len(set(selected)):
        raise ValueError(f"Duplicate factor IDs are not allowed: {selected}")
    catalog = _catalog()
    unknown = sorted(set(selected) - set(catalog))
    if unknown:
        raise ValueError(f"Unknown factor IDs: {unknown}")
    unavailable = sorted(set(selected) - set(BUILDERS))
    if unavailable:
        details = {
            factor_id: {
                "status": catalog[factor_id].get("status"),
                "data_availability": catalog[factor_id].get("data_availability"),
                "implementation": catalog[factor_id].get("implementation"),
            }
            for factor_id in unavailable
        }
        raise ValueError(
            "Factors are catalog hypotheses but are not executable with the current "
            f"data contract: {details}"
        )
    return selected


def factor_selection_manifest(factor_ids: Sequence[str] | None) -> dict[str, Any]:
    selected = validate_factor_ids(factor_ids)
    catalog = _catalog()
    implementation_path = Path(__file__).with_name("core_timeseries.py")
    registry_path = Path(__file__)
    records = []
    for factor_id in selected:
        record = catalog[factor_id]
        implementation_path_text, _ = record["implementation"].split(":", 1)
        factor_implementation_path = SKILL_ROOT / implementation_path_text
        records.append(
            {
                "factor_id": factor_id,
                "catalog_record_sha256": _sha256_json(record),
                "implementation": record["implementation"],
                "implementation_sha256": _sha256_file(
                    factor_implementation_path
                ),
            }
        )
    manifest = {
        "factor_ids": selected,
        "records": records,
        "core_implementation_sha256": _sha256_file(implementation_path),
        "registry_sha256": _sha256_file(registry_path),
    }
    manifest["selection_sha256"] = _sha256_json(manifest)
    return manifest


def build_factor_frame(
    raw: pd.DataFrame,
    config: dict[str, Any],
    horizon_step: int,
    factor_ids: Sequence[str] | None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build namespaced factor columns without changing row population or order."""
    selected = validate_factor_ids(factor_ids)
    manifest = factor_selection_manifest(selected)
    if not selected:
        return pd.DataFrame(index=raw.index), [], manifest

    rows: list[dict[str, float]] = []
    for row in raw.to_dict(orient="records"):
        combined: dict[str, float] = {}
        for factor_id in selected:
            outputs = BUILDERS[factor_id](row, config, horizon_step)
            prefix = factor_id.replace("factor.", "factor__").replace(".", "__")
            for name, value in sorted(outputs.items()):
                column = f"{prefix}__{name}"
                if column in combined:
                    raise ValueError(f"Duplicate generated factor column: {column}")
                combined[column] = float(value)
        rows.append(combined)

    frame = pd.DataFrame(rows, index=raw.index)
    names = list(frame.columns)
    values = frame.to_numpy(dtype=np.float32)
    finite = np.isfinite(values)
    manifest["horizon_step"] = int(horizon_step)
    manifest["generated_columns"] = names
    manifest["generated_column_count"] = len(names)
    manifest["row_count"] = len(frame)
    manifest["finite_fraction"] = float(finite.mean()) if values.size else 1.0
    if values.size and not finite.all():
        bad = int((~finite).sum())
        raise ValueError(
            f"Selected factors generated {bad} non-finite values; explicit train-only "
            "imputation must be added as a protected protocol before validation"
        )
    return frame, names, manifest
