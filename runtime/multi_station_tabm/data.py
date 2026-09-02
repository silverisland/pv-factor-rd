from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Iterator, Union

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
    pattern = str(config["data"].get("parquet_glob", "station=*.parquet")).strip()
    if not pattern:
        raise ValueError("data.parquet_glob must be non-empty")
    paths: list[Path] = []
    for item in values:
        path = Path(item).expanduser()
        if path.is_dir():
            paths.extend(
                candidate
                for candidate in sorted(path.glob(pattern))
                if candidate.is_file()
            )
        elif path.is_file():
            paths.append(path)
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            f"No station parquet files matched parquet_glob={pattern!r} "
            f"under source={value}"
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
    issue_time_column = config["data"].get("nwp_issue_time_column")
    if issue_time_column:
        result.append(issue_time_column)
    return list(dict.fromkeys(result))


def _canonical_station(value: Any, config: Config) -> str:
    station = str(value).strip()
    aliases = config["data"].get("site_metadata", {}).get("aliases", {})
    canonical = str(aliases.get(station, station)).strip()
    if not canonical:
        raise ValueError("station_id must be non-empty after alias normalization")
    return canonical


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
    else:
        raise ValueError(
            "Data requires the configured station ID column or an internal station_id column"
        )
    station = station.str.strip()
    if station.isna().any() or station.eq("").any():
        raise ValueError("station_id must be non-empty for every sample")
    if source_file is not None and station.nunique() != 1:
        values = sorted(station.unique().astype(str).tolist())
        raise ValueError(
            f"{source_file}: configured station column must contain exactly one "
            f"station, found {values}"
        )
    frame[STATION_ID] = station.map(lambda value: _canonical_station(value, config))
    if source_file is not None:
        frame[SOURCE_FILE] = source_file
    elif SOURCE_FILE not in frame.columns:
        frame[SOURCE_FILE] = "<dataframe>"
    else:
        frame[SOURCE_FILE] = frame[SOURCE_FILE].astype(str)
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


def _attach_site_metadata(
    frame: pd.DataFrame,
    config: Config,
    *,
    metadata: pd.DataFrame | None = None,
    metadata_loaded: bool = False,
) -> pd.DataFrame:
    if not metadata_loaded:
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
    timestamp_column = columns["timestamp"]
    minimum = int(config["features"]["history_length"])
    for row_index, (power, ghi) in enumerate(zip(frame[power_column], frame[ghi_column])):
        station = frame.iloc[row_index][STATION_ID]
        timestamp = frame.iloc[row_index][timestamp_column]
        context = (
            f"row={row_index}, station_id={station!r}, timestamp={timestamp!r}"
        )
        try:
            power_length = len(power)
            ghi_length = len(ghi)
        except TypeError as error:
            raise ValueError(
                f"{power_column} and {ghi_column} must be arrays at {context}"
            ) from error
        if power_length != ghi_length:
            raise ValueError(
                f"{ghi_column} has {ghi_length} points but "
                f"{power_column} has {power_length}; histories must align exactly"
                f" at {context}"
            )
        if power_length < minimum:
            raise ValueError(
                f"Aligned histories have {power_length} points; need at least "
                f"{minimum} at {context}"
            )


def iter_multi_station_data(
    data: DataInput | None,
    config: Config,
    *,
    require_target: bool = True,
) -> Iterator[pd.DataFrame]:
    """Yield validated station chunks without retaining raw arrays globally.

    For a parquet source, each yielded chunk corresponds to one station file.
    This lets callers construct numerical features immediately and release the
    object-array columns before reading the next file.
    """
    source = data if data is not None else config["data"].get("parquet_root")
    if source is None:
        raise ValueError(
            "Provide a parquet root through data or config data.parquet_root"
        )
    metadata = _load_site_metadata(config)
    timestamp = config["data"]["columns"]["timestamp"]
    required = set(required_columns(config, require_target=require_target))

    def prepare(frame: pd.DataFrame, source_name: str | None) -> pd.DataFrame:
        missing = sorted(required - set(frame.columns))
        if missing:
            location = source_name or "DataFrame"
            raise ValueError(f"{location} is missing columns: {missing}")
        frame = frame.copy()
        frame[timestamp] = pd.to_datetime(frame[timestamp], errors="raise")
        frame = _attach_site_metadata(
            frame,
            config,
            metadata=metadata,
            metadata_loaded=True,
        )
        _validate_aligned_histories(frame, config)
        return frame.sort_values(
            [timestamp, STATION_ID, SOURCE_FILE], kind="stable", ignore_index=True
        )

    if isinstance(source, pd.DataFrame):
        yield prepare(
            _annotate_station(source, config, source_file=None), None
        )
        return

    columns_to_read = required_columns(config, require_target=require_target)
    for path in _paths(source, config):
        # Read one station, validate it, and yield it immediately.  The caller
        # can replace object-array columns with numerical features before this
        # loop advances to the next parquet file.
        current = (
            pd.read_parquet(path, columns=columns_to_read)
            .dropna()
            .reset_index(drop=True)
        )
        if current.empty:
            raise ValueError(f"{path.name} has no rows after dropna")
        yield prepare(
            _annotate_station(current, config, source_file=path.name),
            path.name,
        )


def load_multi_station_data(
    data: DataInput | None,
    config: Config,
    *,
    require_target: bool = True,
) -> pd.DataFrame:
    """Compatibility loader that materializes all validated station chunks."""
    frames = list(
        iter_multi_station_data(
            data, config, require_target=require_target
        )
    )
    if not frames:
        raise ValueError("Multi-station data is empty")
    timestamp = config["data"]["columns"]["timestamp"]
    return pd.concat(frames, ignore_index=True).sort_values(
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
