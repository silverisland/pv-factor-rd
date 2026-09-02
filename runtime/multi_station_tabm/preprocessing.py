from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import sklearn.preprocessing

from .config import Config
from .data import STATION_ID, station_manifest
from .fingerprints import (
    canonical_json_sha256,
    numeric_array_sha256,
    ordered_strings_sha256,
    split_fingerprint,
)


@dataclass(frozen=True)
class LabelStats:
    mean: float
    std: float


@dataclass
class PreparedTrainingData:
    x_train: np.ndarray
    y_train_normalized: np.ndarray
    x_validation: np.ndarray
    y_validation_raw: np.ndarray
    feature_preprocessor: object
    label_stats: Optional[LabelStats]
    manifest: dict[str, Any]


def _require_finite(name: str, values: np.ndarray) -> None:
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(
            f"{name} contains {int((~finite).sum())} non-finite values; "
            "baseline parity mode does not silently impute"
        )


def preprocessor_state_manifest(preprocessor: object) -> dict[str, Any]:
    """Describe learned QuantileTransformer state for artifact identity checks."""
    required = ("quantiles_", "references_", "n_quantiles_", "n_features_in_")
    missing = [name for name in required if not hasattr(preprocessor, name)]
    if missing:
        raise ValueError(f"Preprocessor is missing learned attributes: {missing}")
    state = {
        "class": f"{type(preprocessor).__module__}.{type(preprocessor).__name__}",
        "n_quantiles": int(getattr(preprocessor, "n_quantiles_")),
        "n_features": int(getattr(preprocessor, "n_features_in_")),
        "quantiles_sha256": numeric_array_sha256(getattr(preprocessor, "quantiles_")),
        "references_sha256": numeric_array_sha256(getattr(preprocessor, "references_")),
    }
    state["state_sha256"] = canonical_json_sha256(state)
    return state


def fit_feature_preprocessor(
    x_train: np.ndarray, config: Config
) -> tuple[object, np.ndarray, dict[str, Any]]:
    _require_finite("x_train", x_train)
    settings = dict(config["training"]["preprocessing"])
    # The reference implementation ties preprocessing noise to the model seed,
    # so paired runs share it while multi-seed experiments genuinely vary it.
    noise_seed = int(config["training"]["seed"])
    noise = np.random.default_rng(noise_seed).normal(
        0.0, float(settings.get("noise_std", 1e-5)), x_train.shape
    ).astype(x_train.dtype)
    n_quantiles = max(
        min(
            len(x_train) // int(settings.get("samples_per_quantile", 30)),
            int(settings.get("max_quantiles", 1000)),
        ),
        int(settings.get("min_quantiles", 10)),
    )
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=min(len(x_train), n_quantiles),
        output_distribution="normal",
        subsample=int(settings.get("quantile_subsample", 10**9)),
    ).fit(x_train + noise)
    transformed = transformer.transform(x_train).astype(np.float32)
    manifest = {
        "class": "sklearn.preprocessing.QuantileTransformer",
        "fit_partition": "configured_training_period_and_stations_only",
        "settings": settings,
        "effective_noise_seed": noise_seed,
        "effective_n_quantiles": int(getattr(transformer, "n_quantiles_", n_quantiles)),
        "input_sha256": numeric_array_sha256(x_train),
        "output_sha256": numeric_array_sha256(transformed),
        "learned_state": preprocessor_state_manifest(transformer),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return transformer, transformed, manifest


def transform_features(preprocessor: object, values: np.ndarray, name: str) -> np.ndarray:
    _require_finite(name, values)
    return preprocessor.transform(values).astype(np.float32)


def normalize_labels(
    y_train: np.ndarray, config: Config
) -> tuple[np.ndarray, Optional[LabelStats], dict[str, Any]]:
    _require_finite("y_train", y_train)
    model = config["model"]
    mode = str(model.get("label_normalization", "scale"))
    stats: Optional[LabelStats] = None
    if mode == "standard":
        stats = LabelStats(float(y_train.mean()), float(y_train.std()))
        if stats.std == 0:
            raise ValueError("Training target has zero standard deviation")
        result = (y_train - stats.mean) / stats.std
    elif mode == "scale":
        result = y_train / float(model["label_scale_value"])
    elif mode == "none":
        result = y_train.copy()
    else:
        raise ValueError(f"Unknown label_normalization: {mode}")
    manifest = {
        "mode": mode,
        "label_scale_value": float(model["label_scale_value"]),
        "stats": asdict(stats) if stats else None,
        "input_sha256": numeric_array_sha256(y_train),
        "output_sha256": numeric_array_sha256(result),
    }
    return result.astype(np.float32), stats, manifest


def inverse_labels(
    values: np.ndarray, config: Config, stats: Optional[LabelStats]
) -> np.ndarray:
    model = config["model"]
    mode = str(model.get("label_normalization", "scale"))
    if mode == "standard":
        if stats is None:
            raise ValueError("Missing standard label statistics")
        return values * stats.std + stats.mean
    if mode == "scale":
        return values * float(model["label_scale_value"])
    if mode == "none":
        return values
    raise ValueError(f"Unknown label_normalization: {mode}")


def prepare_training_data(
    train_frame,
    validation_frame,
    feature_names: list[str],
    target_name: str,
    config: Config,
) -> PreparedTrainingData:
    if train_frame.empty or validation_frame.empty:
        raise ValueError(
            f"Training and validation splits must be non-empty: "
            f"train={len(train_frame)}, validation={len(validation_frame)}"
        )
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names must be non-empty and unique")
    if STATION_ID in feature_names:
        raise ValueError(
            "station_id is metadata and must not enter the fixed numerical TabM input"
        )
    if list(train_frame.columns).count("row_id") != 1:
        raise ValueError("Training frame requires one row_id column")
    if train_frame["row_id"].duplicated().any() or validation_frame["row_id"].duplicated().any():
        raise ValueError("row_id must be unique inside each split")
    overlap = set(train_frame["row_id"]) & set(validation_frame["row_id"])
    if overlap:
        raise ValueError(f"Train/validation row overlap: {len(overlap)}")

    x_train_raw = train_frame[feature_names].to_numpy(dtype=np.float32)
    x_validation_raw = validation_frame[feature_names].to_numpy(dtype=np.float32)
    y_train_raw = train_frame[target_name].to_numpy(dtype=np.float32)
    y_validation_raw = validation_frame[target_name].to_numpy(dtype=np.float32)
    _require_finite("x_validation", x_validation_raw)
    _require_finite("y_validation", y_validation_raw)

    preprocessor, x_train, preprocessor_manifest = fit_feature_preprocessor(
        x_train_raw, config
    )
    x_validation = transform_features(preprocessor, x_validation_raw, "x_validation")
    y_train, label_stats, label_manifest = normalize_labels(y_train_raw, config)
    manifest = {
        "feature_names": list(feature_names),
        "feature_order_sha256": ordered_strings_sha256(feature_names),
        "target_name": target_name,
        "train": split_fingerprint(train_frame["row_id"].tolist()),
        "validation": split_fingerprint(validation_frame["row_id"].tolist()),
        "train_stations": station_manifest(train_frame),
        "validation_stations": station_manifest(validation_frame),
        "x_train_raw_sha256": numeric_array_sha256(x_train_raw),
        "x_validation_raw_sha256": numeric_array_sha256(x_validation_raw),
        "y_train_raw_sha256": numeric_array_sha256(y_train_raw),
        "y_validation_raw_sha256": numeric_array_sha256(y_validation_raw),
        "preprocessor": preprocessor_manifest,
        "label_transform": label_manifest,
    }
    manifest["prepared_data_sha256"] = canonical_json_sha256(manifest)
    return PreparedTrainingData(
        x_train=x_train,
        y_train_normalized=y_train,
        x_validation=x_validation,
        y_validation_raw=y_validation_raw,
        feature_preprocessor=preprocessor,
        label_stats=label_stats,
        manifest=manifest,
    )
