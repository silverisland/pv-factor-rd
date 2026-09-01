from __future__ import annotations

import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from .config import Config, ConfigInput, load_config, resolve_horizons
from .data import (
    SOURCE_FILE,
    STATION_ID,
    DataInput,
    load_multi_station_data,
    station_manifest,
)
from .evaluator import evaluate_horizon, predict_horizon
from .features import build_horizon_frame, feature_contract
from factor_library.implementations.registry import factor_selection_manifest
from .fingerprints import (
    canonical_json_sha256,
    numeric_array_sha256,
    ordered_strings_sha256,
    split_fingerprint,
)
from .metrics import (
    daily_and_monthly_metrics,
    grouped_horizon_metrics,
    monthly_score_summary,
    station_macro_summary,
    station_metrics,
)
from .preprocessing import prepare_training_data
from .splits import (
    select_target_evaluation,
    split_protocol_manifest,
    training_splits,
)
from .trainer import train_prepared


def _effective_config(config: ConfigInput, seed: Optional[int]) -> Config:
    result = deepcopy(load_config(config))
    if seed is not None:
        result["training"]["seed"] = int(seed)
    return result


def _run_id(config: Config) -> str:
    contract = {
        "features": config["features"],
        "model": config["model"],
        "training": config["training"],
        "evaluation": config["evaluation"],
    }
    # PyYAML parses unquoted YYYY-MM-DD values as datetime.date.  Reuse the
    # canonical protocol serializer, which normalizes date-like values through
    # their stable string representation.
    suffix = canonical_json_sha256(contract)[:8]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + suffix


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def train(
    config: ConfigInput,
    data: DataInput | None = None,
    *,
    seed: Optional[int] = None,
    factor_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Orchestrate data -> features/splits -> preprocessing -> training -> validation."""
    cfg = _effective_config(config, seed)
    raw = load_multi_station_data(data, cfg, require_target=True)
    factor_manifest = factor_selection_manifest(factor_ids)
    selected_factors = list(factor_manifest["factor_ids"])
    horizons = resolve_horizons(cfg)
    output_root = Path(cfg["output"]["checkpoint_dir"]).expanduser().resolve()
    output_dir = output_root / _run_id(cfg)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    training_rows = []
    validation_metric_rows = []
    validation_predictions = []
    horizon_manifests = []
    for horizon_step in horizons:
        feature_build_started = time.perf_counter()
        frame, feature_names, target_name = build_horizon_frame(
            raw,
            cfg,
            horizon_step,
            require_target=True,
            factor_ids=selected_factors,
        )
        feature_build_seconds = time.perf_counter() - feature_build_started
        assert target_name is not None
        train_frame, validation_frame = training_splits(frame, cfg)
        prepared = prepare_training_data(
            train_frame, validation_frame, feature_names, target_name, cfg
        )
        training_result = train_prepared(
            prepared,
            horizon_step=horizon_step,
            config=cfg,
            checkpoint_dir=output_dir,
        )
        validation_metrics, predictions, evaluation_audit = evaluate_horizon(
            validation_frame,
            target_name,
            checkpoint_dir=output_dir,
            horizon_step=horizon_step,
            config=cfg,
        )
        training_rows.append(
            {
                "minutes_ahead": horizon_step
                * int(cfg["features"]["minutes_per_point"]),
                "feature_build_seconds": feature_build_seconds,
                **training_result.summary,
            }
        )
        validation_metric_rows.append(validation_metrics)
        validation_predictions.append(predictions)
        horizon_manifests.append(
            {
                "horizon_step": horizon_step,
                "feature_build_seconds": feature_build_seconds,
                "feature_contract": feature_contract(
                    cfg, horizon_step, selected_factors
                ),
                "frame_rows": split_fingerprint(frame["row_id"].tolist()),
                "train_rows": split_fingerprint(train_frame["row_id"].tolist()),
                "validation_rows": split_fingerprint(
                    validation_frame["row_id"].tolist()
                ),
                "frame_stations": station_manifest(frame),
                "train_stations": station_manifest(train_frame),
                "validation_stations": station_manifest(validation_frame),
                "base_feature_names_sha256": ordered_strings_sha256(
                    [name for name in feature_names if not name.startswith("factor__")]
                ),
                "base_feature_values_sha256": numeric_array_sha256(
                    frame[
                        [
                            name
                            for name in feature_names
                            if not name.startswith("factor__")
                        ]
                    ].to_numpy(dtype=np.float32)
                ),
                "target_ratio_sha256": numeric_array_sha256(
                    frame[target_name].to_numpy(dtype=np.float32)
                ),
                "target_power_sha256": numeric_array_sha256(
                    frame["target_power"].to_numpy(dtype=np.float32)
                ),
                "capacity_sha256": numeric_array_sha256(
                    frame["capacity"].to_numpy(dtype=np.float32)
                ),
                "prepared_data_sha256": prepared.manifest["prepared_data_sha256"],
                "artifact_manifest_sha256": training_result.artifact_manifest[
                    "artifact_manifest_sha256"
                ],
                "environment_sha256": training_result.artifact_manifest[
                    "runtime_environment"
                ]["environment_sha256"],
                "model_state_sha256": training_result.artifact_manifest[
                    "model_state_sha256"
                ],
                "preprocessor_state_sha256": training_result.artifact_manifest[
                    "preprocessor_state_sha256"
                ],
                "validation_evaluation_audit": evaluation_audit,
            }
        )

    training_summary = pd.DataFrame(training_rows).sort_values("horizon_step")
    validation_metrics = pd.DataFrame(validation_metric_rows).sort_values(
        "horizon_step"
    )
    predictions = pd.concat(validation_predictions, ignore_index=True)
    capacity = float(cfg["evaluation"]["score_capacity"])
    by_station = station_metrics(predictions, capacity)
    station_macro = station_macro_summary(by_station)
    daily, monthly = daily_and_monthly_metrics(predictions, capacity)
    monthly_summary = monthly_score_summary(monthly)
    validation_metrics = validation_metrics.merge(
        monthly_summary[["horizon_step", "mean_monthly_capacity_score"]],
        on="horizon_step",
        how="left",
        validate="one_to_one",
    )
    grouped = grouped_horizon_metrics(
        predictions, capacity, int(cfg["features"]["minutes_per_point"])
    )

    training_summary.to_csv(output_dir / "training_summary_by_horizon.csv", index=False)
    validation_metrics.to_csv(
        output_dir / "validation_metrics_by_horizon.csv", index=False
    )
    by_station.to_csv(
        output_dir / "validation_metrics_by_station.csv", index=False
    )
    station_macro.to_csv(
        output_dir / "validation_station_macro_summary.csv", index=False
    )
    daily.to_csv(output_dir / "validation_metrics_daily.csv", index=False)
    monthly.to_csv(output_dir / "validation_metrics_monthly.csv", index=False)
    monthly_summary.to_csv(
        output_dir / "validation_monthly_score_summary.csv", index=False
    )
    grouped.to_csv(output_dir / "validation_metrics_horizon_groups.csv", index=False)
    predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)

    run_contract = {
        "forecast_object": "multi_station_shared_model",
        "evaluation_object": "single_target_station",
        "station_identity_role": "metadata_only",
        "input_stations": station_manifest(raw),
        "training_stations": horizon_manifests[0]["train_stations"],
        "prediction_mode": cfg["features"]["prediction_mode"],
        "horizons": horizons,
        "seed": int(cfg["training"]["seed"]),
        "split_protocol": split_protocol_manifest(cfg),
        "factor_selection": factor_manifest,
        "fixed_runtime_contract_sha256": canonical_json_sha256(
            {"model": cfg["model"], "training": cfg["training"]}
        ),
        "feature_protocol_sha256": canonical_json_sha256(
            {
                "features": cfg["features"],
                "factor_parameters": cfg.get("factor_parameters", {}),
                "weather_roles": cfg["data"].get("weather_roles", {}),
                "site_timezone": cfg["data"].get("site_metadata", {}).get(
                    "timezone", "Asia/Shanghai"
                ),
                "factor_selection": factor_manifest,
            }
        ),
        "evaluation_protocol_sha256": canonical_json_sha256(cfg["evaluation"]),
        "horizon_manifests": horizon_manifests,
        "final_test_read": False,
    }
    run_contract["run_contract_sha256"] = canonical_json_sha256(run_contract)
    _write_json(output_dir / "run_manifest.json", run_contract)
    return {
        "checkpoint_dir": output_dir,
        "training_summary": training_summary,
        "validation_metrics": validation_metrics,
        "station_metrics": by_station,
        "station_macro_summary": station_macro,
        "daily_metrics": daily,
        "monthly_metrics": monthly,
        "monthly_score_summary": monthly_summary,
        "horizon_group_metrics": grouped,
        "validation_predictions": predictions,
        "manifest": run_contract,
    }


def evaluate(
    checkpoint: str | Path,
    config: ConfigInput,
    *,
    period_name: str,
    data: DataInput | None = None,
    seed: Optional[int] = None,
    factor_ids: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = _effective_config(config, seed)
    if period_name not in {"confirmation", "final_test"}:
        raise ValueError("period_name must be confirmation or final_test")
    period = cfg["evaluation"]["periods"].get(period_name)
    if not period:
        raise ValueError(f"No evaluation period configured for {period_name}")
    raw = load_multi_station_data(data, cfg, require_target=True)
    checkpoint_dir = Path(checkpoint).expanduser().resolve()
    run_manifest = json.loads(
        (checkpoint_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    current_evaluation_sha256 = canonical_json_sha256(cfg["evaluation"])
    if (
        run_manifest.get("evaluation_protocol_sha256")
        != current_evaluation_sha256
    ):
        raise ValueError(
            "Current target station, split, or evaluation periods differ from "
            "the checkpoint evaluation protocol"
        )
    if factor_ids is None:
        factor_ids = run_manifest.get("factor_selection", {}).get("factor_ids", [])
    metrics_rows, prediction_frames, audits = [], [], []
    for horizon_step in resolve_horizons(cfg):
        frame, _, target_name = build_horizon_frame(
            raw,
            cfg,
            horizon_step,
            require_target=True,
            factor_ids=factor_ids,
        )
        assert target_name is not None
        current = select_target_evaluation(frame, cfg, period_name)
        metrics, predictions, audit = evaluate_horizon(
            current,
            target_name,
            checkpoint_dir=checkpoint_dir,
            horizon_step=horizon_step,
            config=cfg,
        )
        metrics_rows.append(metrics)
        prediction_frames.append(predictions)
        audits.append(audit)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    capacity = float(cfg["evaluation"]["score_capacity"])
    by_station = station_metrics(predictions, capacity)
    station_macro = station_macro_summary(by_station)
    daily, monthly = daily_and_monthly_metrics(predictions, capacity)
    monthly_summary = monthly_score_summary(monthly)
    by_horizon = pd.DataFrame(metrics_rows).sort_values("horizon_step").merge(
        monthly_summary[["horizon_step", "mean_monthly_capacity_score"]],
        on="horizon_step",
        how="left",
        validate="one_to_one",
    )
    grouped = grouped_horizon_metrics(
        predictions, capacity, int(cfg["features"]["minutes_per_point"])
    )
    audit_frame = pd.DataFrame(
        [
            {
                "horizon_step": item["horizon_step"],
                "rows": item["rows"]["rows"],
                "row_id_sha256": item["rows"]["row_id_sha256"],
                "station_count": item["stations"]["station_count"],
                "station_ids_sha256": item["stations"]["station_ids_sha256"],
                "artifact_manifest_sha256": item["artifact_manifest_sha256"],
                "model_state_sha256": item["model_state_sha256"],
                "preprocessor_state_sha256": item["preprocessor_state_sha256"],
                "fixed_runtime_contract_sha256": item[
                    "fixed_runtime_contract_sha256"
                ],
                "clipped": item["clipped"],
            }
            for item in audits
        ]
    )
    return {
        "by_horizon": by_horizon,
        "by_station": by_station,
        "station_macro_summary": station_macro,
        "by_day": daily,
        "by_month": monthly,
        "monthly_score_summary": monthly_summary,
        "by_horizon_group": grouped,
        "predictions": predictions,
        "audit": audit_frame,
    }


def predict(
    checkpoint: str | Path,
    config: ConfigInput,
    data: DataInput,
    *,
    seed: Optional[int] = None,
    factor_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    cfg = _effective_config(config, seed)
    raw = load_multi_station_data(data, cfg, require_target=False)
    checkpoint_dir = Path(checkpoint).expanduser().resolve()
    if factor_ids is None:
        run_manifest = json.loads(
            (checkpoint_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        factor_ids = run_manifest.get("factor_selection", {}).get("factor_ids", [])
    by_horizon = []
    for horizon_step in resolve_horizons(cfg):
        frame, _, _ = build_horizon_frame(
            raw,
            cfg,
            horizon_step,
            require_target=False,
            factor_ids=factor_ids,
        )
        prediction_ratio, _ = predict_horizon(
            frame,
            checkpoint_dir=checkpoint_dir,
            horizon_step=horizon_step,
            config=cfg,
            clip=True,
        )
        frame = frame[
            [STATION_ID, SOURCE_FILE, "timestamp", "target_timestamp", "horizon_step", "capacity"]
        ].copy()
        frame["prediction_ratio"] = prediction_ratio
        frame["prediction"] = (
            prediction_ratio * frame["capacity"].to_numpy(dtype=np.float32)
        )
        by_horizon.append(frame)
    long = pd.concat(by_horizon, ignore_index=True).sort_values(
        [STATION_ID, "timestamp", "horizon_step"]
    )
    return (
        long.groupby([STATION_ID, SOURCE_FILE, "timestamp"], sort=True)["prediction"]
        .apply(lambda value: value.to_numpy(dtype=np.float32))
        .rename("power_prediction")
        .reset_index()
    )
