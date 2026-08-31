from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import pandas as pd

from .config import Config
from .fingerprints import canonical_json_sha256, ordered_strings_sha256


DataInput = Union[pd.DataFrame, str, Path, Sequence[Union[str, Path]]]
STATION_ID = "station_id"
SOURCE_FILE = "source_file"
SITE_CAPACITY = "site_capacity"
SITE_LONGITUDE = "site_longitude"
SITE_LATITUDE = "site_latitude"
SITE_TIMEZONE = "site_timezone"


def _paths(
    value: Union[str, Path, Sequence[Union[str, Path]]], config: Config
) -> list[Path]:
    values = [value] if isinstance(value, (str, Path)) else list(value)
    suffix = str(config["data"].get("file_suffix", ".parquet"))
    prefix = str(config["data"].get("file_prefix") or "")
    paths: list[Path] = []
    for item in values:
        path = Path(item).expanduser()
        if path.is_dir():
            paths.extend(
                candidate
                for candidate in sorted(path.iterdir())
                if candidate.is_file()
                and candidate.name.endswith(suffix)
                and (not prefix or candidate.name.startswith(prefix))
            )
        elif path.name.endswith(suffix) and (
            not prefix or path.name.startswith(prefix)
        ):
            paths.append(path)
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            f"No station files matched source={value}, prefix={prefix!r}, "
            f"suffix={suffix!r}"
        )
    return paths


def required_columns(config: Config, *, require_target: bool) -> list[str]:
    columns = config["data"]["columns"]
    result = [
        columns["timestamp"],
        columns["power_history"],
        *columns["future_weather"],
    ]
    station_column = config["data"].get("station_id_column")
    if station_column:
        result.append(station_column)
    ghi_history = columns.get("ghi_history")
    if ghi_history:
        result.append(ghi_history)
    if require_target:
        result.append(columns["power_future"])
    return result


def _canonical_station(value: Any, config: Config) -> str:
    station = str(value).strip()
    aliases = config["data"].get("site_metadata", {}).get("aliases", {})
    canonical = str(aliases.get(station, station)).strip()
    if not canonical:
        raise ValueError("station_id must be non-empty after alias normalization")
    return canonical


def _station_from_filename(path: Path, config: Config) -> str:
    suffix = str(config["data"].get("file_suffix", ".parquet"))
    if suffix and path.name.endswith(suffix):
        value = path.name[: -len(suffix)]
    else:
        value = path.stem
    if not value:
        raise ValueError(f"Could not derive station ID from {path.name!r}")
    return value


def _annotate_station(
    frame: pd.DataFrame, config: Config, *, source_file: str | None
) -> pd.DataFrame:
    frame = frame.copy()
    station_column = config["data"].get("station_id_column")
    if station_column:
        if station_column not in frame.columns:
            raise ValueError(
                f"Data is missing configured station ID column: {station_column}"
            )
        station = frame[station_column].astype("string")
    elif STATION_ID in frame.columns:
        station = frame[STATION_ID].astype("string")
    elif source_file is not None:
        station = pd.Series(
            _station_from_filename(Path(source_file), config), index=frame.index
        )
    else:
        raise ValueError(
            "DataFrame input requires data.station_id_column or a station_id column"
        )
    station = station.str.strip()
    if station.isna().any() or station.eq("").any():
        raise ValueError("station_id must be non-empty for every sample")
    frame[STATION_ID] = station.map(lambda value: _canonical_station(value, config))
    frame[SOURCE_FILE] = source_file or "<dataframe>"
    return frame


def _load_site_metadata(config: Config) -> pd.DataFrame | None:
    metadata_config = config["data"].get("site_metadata", {})
    path_value = metadata_config.get("path")
    overrides = metadata_config.get("overrides", {}) or {}
    records: list[dict[str, Any]] = []

    if path_value:
        path = Path(path_value).expanduser()
        metadata = pd.read_csv(path)
        configured = {
            STATION_ID: metadata_config.get("station_column", "plantid"),
            SITE_CAPACITY: metadata_config.get("capacity_column", "GCCAPACITY"),
            SITE_LONGITUDE: metadata_config.get("longitude_column", "LONGITUDE"),
            SITE_LATITUDE: metadata_config.get("latitude_column", "LATITUDE"),
        }
        missing = sorted(set(configured.values()) - set(metadata.columns))
        if missing:
            raise ValueError(f"Site metadata {path} is missing columns: {missing}")
        for source, target in ((value, key) for key, value in configured.items()):
            metadata[target] = metadata[source]
        records.extend(metadata[list(configured)].to_dict(orient="records"))

    for station, values in overrides.items():
        canonical = _canonical_station(station, config)
        records = [
            record
            for record in records
            if _canonical_station(record[STATION_ID], config) != canonical
        ]
        records.append(
            {
                STATION_ID: canonical,
                SITE_CAPACITY: values.get("capacity"),
                SITE_LONGITUDE: values.get("longitude"),
                SITE_LATITUDE: values.get("latitude"),
            }
        )

    if not records:
        return None
    metadata = pd.DataFrame.from_records(records)
    metadata[STATION_ID] = metadata[STATION_ID].map(
        lambda value: _canonical_station(value, config)
    )
    for column in (SITE_CAPACITY, SITE_LONGITUDE, SITE_LATITUDE):
        metadata[column] = pd.to_numeric(metadata[column], errors="raise")
    if (
        not metadata[SITE_CAPACITY].map(math.isfinite).all()
        or (metadata[SITE_CAPACITY] <= 0).any()
    ):
        raise ValueError("Every station capacity must be finite and positive")
    if metadata[SITE_LONGITUDE].isna().any() or not metadata[SITE_LONGITUDE].between(-180, 180).all():
        raise ValueError("Every station longitude must be within [-180, 180]")
    if metadata[SITE_LATITUDE].isna().any() or not metadata[SITE_LATITUDE].between(-90, 90).all():
        raise ValueError("Every station latitude must be within [-90, 90]")

    value_columns = [SITE_CAPACITY, SITE_LONGITUDE, SITE_LATITUDE]
    conflicts = (
        metadata.groupby(STATION_ID)[value_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        stations = conflicts[conflicts].index.astype(str).tolist()
        raise ValueError(f"Conflicting metadata records for stations: {stations}")
    metadata = metadata.drop_duplicates(STATION_ID, keep="last")
    metadata[SITE_TIMEZONE] = str(metadata_config.get("timezone", "Asia/Shanghai"))
    return metadata[[STATION_ID, *value_columns, SITE_TIMEZONE]]


def _attach_site_metadata(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    metadata = _load_site_metadata(config)
    if metadata is None:
        present = {
            name
            for name in (SITE_CAPACITY, SITE_LONGITUDE, SITE_LATITUDE, SITE_TIMEZONE)
            if name in frame.columns
        }
        if present and present != {
            SITE_CAPACITY,
            SITE_LONGITUDE,
            SITE_LATITUDE,
            SITE_TIMEZONE,
        }:
            raise ValueError(
                "Inline site metadata must provide capacity, longitude, latitude, and timezone together"
            )
        if present:
            frame = frame.copy()
            for column in (SITE_CAPACITY, SITE_LONGITUDE, SITE_LATITUDE):
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            if (
                not frame[SITE_CAPACITY].map(math.isfinite).all()
                or (frame[SITE_CAPACITY] <= 0).any()
            ):
                raise ValueError("Every station capacity must be finite and positive")
            if not frame[SITE_LONGITUDE].between(-180, 180).all():
                raise ValueError("Every station longitude must be within [-180, 180]")
            if not frame[SITE_LATITUDE].between(-90, 90).all():
                raise ValueError("Every station latitude must be within [-90, 90]")
            if not frame[SITE_TIMEZONE].astype(str).eq("Asia/Shanghai").all():
                raise ValueError("Every station timezone must be Asia/Shanghai")
        return frame
    existing = [SITE_CAPACITY, SITE_LONGITUDE, SITE_LATITUDE, SITE_TIMEZONE]
    frame = frame.drop(columns=[name for name in existing if name in frame], errors="ignore")
    frame = frame.merge(metadata, on=STATION_ID, how="left", validate="many_to_one", sort=False)
    missing = frame.loc[frame[SITE_CAPACITY].isna(), STATION_ID].astype(str).unique().tolist()
    if missing:
        raise ValueError(f"No site metadata found for stations: {sorted(missing)}")
    return frame


def _validate_aligned_histories(frame: pd.DataFrame, config: Config) -> None:
    columns = config["data"]["columns"]
    ghi_column = columns.get("ghi_history")
    if not ghi_column:
        return
    power_column = columns["power_history"]
    minimum = int(config["features"]["history_length"])
    for row_index, (power, ghi) in enumerate(zip(frame[power_column], frame[ghi_column])):
        try:
            power_length = len(power)
            ghi_length = len(ghi)
        except TypeError as error:
            raise ValueError(
                f"{power_column} and {ghi_column} row {row_index} must be arrays"
            ) from error
        if power_length != ghi_length:
            raise ValueError(
                f"{ghi_column} row {row_index} has {ghi_length} points but "
                f"{power_column} has {power_length}; histories must align exactly"
            )
        if power_length < minimum:
            raise ValueError(
                f"Aligned histories row {row_index} has {power_length} points; "
                f"need at least {minimum}"
            )


def load_multi_station_data(
    data: DataInput | None,
    config: Config,
    *,
    require_target: bool = True,
) -> pd.DataFrame:
    source = data if data is not None else config["data"].get("path")
    if source is None:
        raise ValueError("Provide station data through data or config data.path")
    if isinstance(source, pd.DataFrame):
        frames = [_annotate_station(source, config, source_file=None)]
    else:
        frames = []
        for path in _paths(source, config):
            current = pd.read_parquet(path)
            missing = sorted(
                set(required_columns(config, require_target=require_target))
                - set(current.columns)
            )
            if missing:
                raise ValueError(f"{path.name} is missing columns: {missing}")
            frames.append(
                _annotate_station(current, config, source_file=path.name)
            )
    frame = pd.concat(frames, ignore_index=True)
    missing = sorted(
        set(required_columns(config, require_target=require_target))
        - set(frame.columns)
    )
    if missing:
        raise ValueError(f"Multi-station data is missing columns: {missing}")
    timestamp = config["data"]["columns"]["timestamp"]
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="raise")
    frame = _attach_site_metadata(frame, config)
    _validate_aligned_histories(frame, config)
    return frame.sort_values(
        [timestamp, STATION_ID, SOURCE_FILE], kind="stable", ignore_index=True
    )


def station_manifest(frame: pd.DataFrame) -> dict[str, object]:
    stations = sorted(frame[STATION_ID].astype(str).unique().tolist())
    counts = (
        frame.groupby(STATION_ID, sort=True).size().astype(int).to_dict()
    )
    manifest: dict[str, object] = {
        "station_count": len(stations),
        "station_ids": stations,
        "station_ids_sha256": ordered_strings_sha256(stations),
        "rows_by_station": counts,
        "rows_by_station_sha256": canonical_json_sha256(counts),
    }
    metadata_columns = [
        name
        for name in (SITE_CAPACITY, SITE_LONGITUDE, SITE_LATITUDE, SITE_TIMEZONE)
        if name in frame.columns
    ]
    if metadata_columns:
        metadata = (
            frame[[STATION_ID, *metadata_columns]]
            .drop_duplicates()
            .sort_values(STATION_ID)
            .to_dict(orient="records")
        )
        manifest["site_metadata_sha256"] = canonical_json_sha256(metadata)
        manifest["site_metadata_columns"] = metadata_columns
    return manifest
