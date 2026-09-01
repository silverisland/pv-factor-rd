from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib import metadata
import json
import platform
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import Config
from .fingerprints import canonical_json_sha256, named_numeric_arrays_sha256
from .metrics import rmse
from .model import (
    configure_reproducibility,
    make_model,
    model_contract,
    predict_normalized,
    resolve_device,
)
from .preprocessing import (
    LabelStats,
    PreparedTrainingData,
    inverse_labels,
    preprocessor_state_manifest,
)


@dataclass
class TrainingResult:
    summary: dict[str, Any]
    history: pd.DataFrame
    artifact_manifest: dict[str, Any]


def _runtime_environment(device: torch.device) -> dict[str, Any]:
    packages = {}
    for distribution in (
        "joblib",
        "numpy",
        "pandas",
        "pyarrow",
        "rtdl_num_embeddings",
        "scikit-learn",
        "tabm",
        "torch",
    ):
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": str(device),
        "device_type": device.type,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version()
        if hasattr(torch.backends, "cudnn")
        else None,
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
    }
    environment["environment_sha256"] = canonical_json_sha256(environment)
    return environment


def _unclipped_validation_prediction(
    model: torch.nn.Module,
    prepared: PreparedTrainingData,
    device: torch.device,
    config: Config,
) -> np.ndarray:
    normalized = predict_normalized(
        model,
        prepared.x_validation,
        device=device,
        batch_size=int(config["training"]["inference_batch_size"]),
    )
    # Reference early stopping uses the direct, UNCLIPPED ratio prediction.
    return inverse_labels(normalized, config, prepared.label_stats).astype(np.float32)


def train_prepared(
    prepared: PreparedTrainingData,
    *,
    horizon_step: int,
    config: Config,
    checkpoint_dir: Path,
    seed: Optional[int] = None,
) -> TrainingResult:
    training = config["training"]
    effective_seed = int(training["seed"] if seed is None else seed)
    reproducibility = configure_reproducibility(effective_seed, config)
    device = resolve_device(str(config["model"].get("device", "auto")))
    x_train = torch.as_tensor(prepared.x_train, device=device)
    y_train = torch.as_tensor(prepared.y_train_normalized, device=device).float()
    model = make_model(prepared.x_train.shape[1], device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    best_model_state = deepcopy(model.state_dict())
    best_optimizer_state = deepcopy(optimizer.state_dict())
    best_epoch = -1
    best_rmse = float("inf")
    patience = int(training["early_stopping_patience"])
    remaining_patience = patience
    batch_size = int(training["batch_size"])
    history_rows = []

    for epoch in range(int(training["epochs"])):
        batch_losses = []
        for batch in torch.randperm(len(x_train), device=device).split(batch_size):
            model.train()
            optimizer.zero_grad()
            prediction = model(x_train[batch], None).squeeze(-1).float()
            target = y_train[batch].repeat_interleave(model.backbone.k)
            loss = nn.functional.mse_loss(prediction.flatten(0, 1), target)
            loss.backward()
            clipping = training.get("gradient_clipping_norm", 1.0)
            if clipping is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), float(clipping))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        validation_prediction = _unclipped_validation_prediction(
            model, prepared, device, config
        )
        validation_rmse = rmse(prepared.y_validation_raw, validation_prediction)
        improved = validation_rmse < best_rmse
        if improved:
            best_rmse = validation_rmse
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            best_optimizer_state = deepcopy(optimizer.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1
        history_rows.append(
            {
                "epoch": epoch,
                "train_mse_normalized": float(np.mean(batch_losses)),
                "validation_ratio_rmse_unclipped": validation_rmse,
                "improved": improved,
                "remaining_patience": remaining_patience,
            }
        )
        if epoch == 0 or (epoch + 1) % int(training.get("log_every_n_epochs", 10)) == 0:
            print(
                f"horizon={horizon_step:02d} seed={effective_seed} epoch={epoch:03d} "
                f"validation_rmse={validation_rmse:.6f} best_rmse={best_rmse:.6f}"
            )
        if remaining_patience < 0:
            break

    models_dir = checkpoint_dir / "models"
    preprocessors_dir = checkpoint_dir / "preprocessors"
    manifests_dir = checkpoint_dir / "manifests"
    histories_dir = checkpoint_dir / "histories"
    for directory in (models_dir, preprocessors_dir, manifests_dir, histories_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"model_h{horizon_step:02d}.pt"
    preprocessor_path = preprocessors_dir / f"preprocessor_h{horizon_step:02d}.joblib"
    history_path = histories_dir / f"history_h{horizon_step:02d}.csv"
    contract = model_contract(prepared.x_train.shape[1])
    model_state_sha256 = named_numeric_arrays_sha256(best_model_state)
    learned_preprocessor_state = preprocessor_state_manifest(
        prepared.feature_preprocessor
    )
    runtime_environment = _runtime_environment(device)
    training_contract = {
        "optimizer": "torch.optim.AdamW",
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "loss": "torch.nn.functional.mse_loss",
        "gradient_clipping_norm": training.get("gradient_clipping_norm"),
        "batch_size": batch_size,
        "early_stopping_metric": "validation_ratio_rmse_unclipped",
        "early_stopping_patience": patience,
    }
    fixed_runtime_contract = {
        "model": config["model"],
        "training": config["training"],
    }
    artifact_manifest = {
        "horizon_step": horizon_step,
        "seed": effective_seed,
        "best_epoch": best_epoch,
        "best_validation_ratio_rmse_unclipped": best_rmse,
        "prepared_data": prepared.manifest,
        "model_contract": contract,
        "model_contract_sha256": canonical_json_sha256(contract),
        "training_contract": training_contract,
        "training_contract_sha256": canonical_json_sha256(training_contract),
        "fixed_runtime_contract": fixed_runtime_contract,
        "fixed_runtime_contract_sha256": canonical_json_sha256(fixed_runtime_contract),
        "reproducibility": reproducibility,
        "runtime_environment": runtime_environment,
        "model_state_sha256": model_state_sha256,
        "preprocessor_state_sha256": learned_preprocessor_state["state_sha256"],
        "label_stats": asdict(prepared.label_stats) if prepared.label_stats else None,
    }
    artifact_manifest["artifact_manifest_sha256"] = canonical_json_sha256(
        artifact_manifest
    )
    torch.save(
        {
            "model_state_dict": best_model_state,
            "optimizer_state_dict": best_optimizer_state,
            "n_num_features": prepared.x_train.shape[1],
            "feature_names": prepared.manifest["feature_names"],
            "training_station_ids": prepared.manifest["train_stations"][
                "station_ids"
            ],
            "horizon_step": horizon_step,
            "best_epoch": best_epoch,
            "label_stats": artifact_manifest["label_stats"],
            "artifact_manifest_sha256": artifact_manifest["artifact_manifest_sha256"],
            "fixed_runtime_contract_sha256": artifact_manifest[
                "fixed_runtime_contract_sha256"
            ],
            "model_state_sha256": model_state_sha256,
            "preprocessor_state_sha256": artifact_manifest[
                "preprocessor_state_sha256"
            ],
        },
        model_path,
    )
    joblib.dump(prepared.feature_preprocessor, preprocessor_path)
    history = pd.DataFrame(history_rows)
    history.to_csv(history_path, index=False)
    manifest_path = manifests_dir / f"manifest_h{horizon_step:02d}.json"
    manifest_path.write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "horizon_step": horizon_step,
        "seed": effective_seed,
        "best_epoch": best_epoch,
        "validation_ratio_rmse_unclipped": best_rmse,
        "epochs_ran": len(history),
        "feature_count": prepared.x_train.shape[1],
        "train_rows": len(prepared.x_train),
        "validation_rows": len(prepared.x_validation),
        "prepared_data_sha256": prepared.manifest["prepared_data_sha256"],
        "artifact_manifest_sha256": artifact_manifest["artifact_manifest_sha256"],
    }
    return TrainingResult(summary=summary, history=history, artifact_manifest=artifact_manifest)


def load_training_artifacts(
    checkpoint_dir: Path, horizon_step: int, config: Config
) -> tuple[torch.nn.Module, object, dict[str, Any], torch.device, Optional[LabelStats]]:
    device = resolve_device(str(config["model"].get("device", "auto")))
    payload = torch.load(
        checkpoint_dir / "models" / f"model_h{horizon_step:02d}.pt",
        map_location=device,
        weights_only=True,
    )
    if int(payload["n_num_features"]) != len(payload["feature_names"]):
        raise ValueError("Checkpoint feature count and ordered feature names disagree")
    manifest_path = (
        checkpoint_dir / "manifests" / f"manifest_h{horizon_step:02d}.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_hash = manifest.pop("artifact_manifest_sha256")
    if canonical_json_sha256(manifest) != stored_hash:
        raise ValueError("Artifact manifest content hash mismatch")
    if stored_hash != payload["artifact_manifest_sha256"]:
        raise ValueError("Checkpoint and artifact manifest identity mismatch")
    if int(payload["horizon_step"]) != horizon_step or int(
        manifest["horizon_step"]
    ) != horizon_step:
        raise ValueError("Requested horizon and stored artifact horizon disagree")
    if list(payload["feature_names"]) != list(
        manifest["prepared_data"]["feature_names"]
    ):
        raise ValueError("Checkpoint and manifest feature order disagree")
    if list(payload["training_station_ids"]) != list(
        manifest["prepared_data"]["train_stations"]["station_ids"]
    ):
        raise ValueError("Checkpoint and manifest training-station sets disagree")
    if payload["fixed_runtime_contract_sha256"] != manifest[
        "fixed_runtime_contract_sha256"
    ]:
        raise ValueError("Checkpoint and manifest runtime contracts disagree")
    if payload.get("label_stats") != manifest.get("label_stats"):
        raise ValueError("Checkpoint and manifest label statistics disagree")
    actual_model_state_sha256 = named_numeric_arrays_sha256(
        payload["model_state_dict"]
    )
    if actual_model_state_sha256 != manifest["model_state_sha256"]:
        raise ValueError("Checkpoint model-state fingerprint mismatch")
    if actual_model_state_sha256 != payload["model_state_sha256"]:
        raise ValueError("Checkpoint model-state identity metadata mismatch")
    model = make_model(int(payload["n_num_features"]), device)
    model.load_state_dict(payload["model_state_dict"])
    preprocessor = joblib.load(
        checkpoint_dir / "preprocessors" / f"preprocessor_h{horizon_step:02d}.joblib"
    )
    if int(getattr(preprocessor, "n_features_in_", -1)) != int(
        payload["n_num_features"]
    ):
        raise ValueError("Preprocessor and checkpoint feature counts disagree")
    actual_preprocessor_state_sha256 = preprocessor_state_manifest(preprocessor)[
        "state_sha256"
    ]
    if actual_preprocessor_state_sha256 != manifest["preprocessor_state_sha256"]:
        raise ValueError("Preprocessor learned-state fingerprint mismatch")
    if actual_preprocessor_state_sha256 != manifest["prepared_data"][
        "preprocessor"
    ]["learned_state"]["state_sha256"]:
        raise ValueError("Prepared-data and artifact preprocessor states disagree")
    if actual_preprocessor_state_sha256 != payload["preprocessor_state_sha256"]:
        raise ValueError("Preprocessor identity metadata mismatch")
    raw_stats = payload.get("label_stats")
    label_stats = LabelStats(**raw_stats) if raw_stats else None
    return model, preprocessor, payload, device, label_stats
