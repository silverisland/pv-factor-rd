from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .data import SOURCE_FILE, STATION_ID, station_manifest
from .fingerprints import canonical_json_sha256, split_fingerprint
from .metrics import regression_metrics
from .model import predict_normalized
from .preprocessing import inverse_labels, transform_features
from .trainer import load_training_artifacts


def predict_horizon(
    frame: pd.DataFrame,
    *,
    checkpoint_dir: Path,
    horizon_step: int,
    config: Config,
    clip: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if frame.empty:
        raise ValueError(f"No rows to predict for horizon {horizon_step}")
    model, preprocessor, payload, device, label_stats = load_training_artifacts(
        checkpoint_dir, horizon_step, config
    )
    current_contract = canonical_json_sha256(
        {"model": config["model"], "training": config["training"]}
    )
    if current_contract != payload["fixed_runtime_contract_sha256"]:
        raise ValueError(
            "Current model/training config differs from checkpoint fixed runtime contract"
        )
    trained_stations = set(map(str, payload["training_station_ids"]))
    evaluation_stations = set(frame[STATION_ID].astype(str))
    unseen_stations = sorted(evaluation_stations - trained_stations)
    if unseen_stations and bool(
        config["evaluation"].get("reject_unseen_stations", True)
    ):
        raise ValueError(
            f"Evaluation contains stations absent from training: {unseen_stations}"
        )
    if bool(
        config["evaluation"].get(
            "require_all_training_stations_in_evaluation", False
        )
    ):
        missing_stations = sorted(trained_stations - evaluation_stations)
        if missing_stations:
            raise ValueError(
                "Evaluation is missing trained stations: "
                f"{missing_stations}"
            )
    feature_names = list(payload["feature_names"])
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"Inference frame missing trained features: {missing}")
    values = transform_features(
        preprocessor,
        frame[feature_names].to_numpy(dtype=np.float32),
        "inference_features",
    )
    normalized = predict_normalized(
        model,
        values,
        device=device,
        batch_size=int(config["training"]["inference_batch_size"]),
    )
    prediction = inverse_labels(normalized, config, label_stats).astype(np.float32)
    if clip:
        lower, upper = map(float, config["model"]["prediction_clip"])
        prediction = np.clip(prediction, lower, upper).astype(np.float32)
    audit = {
        "horizon_step": horizon_step,
        "rows": split_fingerprint(frame["row_id"].tolist()),
        "stations": station_manifest(frame),
        "unseen_stations": unseen_stations,
        "feature_names": feature_names,
        "artifact_manifest_sha256": payload["artifact_manifest_sha256"],
        "model_state_sha256": payload["model_state_sha256"],
        "preprocessor_state_sha256": payload["preprocessor_state_sha256"],
        "fixed_runtime_contract_sha256": payload["fixed_runtime_contract_sha256"],
        "clipped": clip,
    }
    return prediction, audit


def evaluate_horizon(
    frame: pd.DataFrame,
    target_name: str,
    *,
    checkpoint_dir: Path,
    horizon_step: int,
    config: Config,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    prediction, audit = predict_horizon(
        frame,
        checkpoint_dir=checkpoint_dir,
        horizon_step=horizon_step,
        config=config,
        clip=True,
    )
    target = frame[target_name].to_numpy(dtype=np.float32)
    metrics = {
        "horizon_step": horizon_step,
        "sample_count": len(frame),
        "station_count": frame[STATION_ID].nunique(),
        **regression_metrics(
            target, prediction, float(config["evaluation"]["score_capacity"])
        ),
    }
    output = frame[
        [
            "row_id",
            STATION_ID,
            SOURCE_FILE,
            "timestamp",
            "target_timestamp",
            "horizon_step",
        ]
    ].copy()
    output["groundtruth"] = target
    output["prediction"] = prediction
    return metrics, output, audit
