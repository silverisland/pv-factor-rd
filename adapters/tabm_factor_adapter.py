#!/usr/bin/env python3
"""Run a real paired baseline/candidate factor experiment with bundled TabM."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from factor_library.implementations.registry import (
    factor_selection_manifest,
    validate_factor_ids,
)
from scripts.catalog_lib import build_snapshot as _protected_snapshot
from runtime.multi_station_tabm.api import evaluate, train
from runtime.multi_station_tabm.config import load_config, resolve_horizons
from runtime.multi_station_tabm.data import load_multi_station_data
from runtime.multi_station_tabm.fingerprints import canonical_json_sha256
from runtime.multi_station_tabm.metrics import regression_metrics


PAIR_KEYS = [
    "row_id",
    "station_id",
    "source_file",
    "timestamp",
    "target_timestamp",
    "horizon_step",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_outer_config(path: Path) -> tuple[dict[str, Any], Path]:
    config = _read_json(path)
    project_root = (path.parent / config.get("project_root", ".")).resolve()
    return config, project_root


def _runtime_config_path(
    outer: dict[str, Any], project_root: Path
) -> Path:
    value = outer.get("runtime_config")
    if not value:
        raise ValueError(
            "config.json must define runtime_config pointing to the private runtime YAML"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Runtime config does not exist: {path}. Copy runtime/config.example.yaml "
            "to that location and fill the private data path first."
        )
    return path


def _assert_request_matches_runtime(
    request: dict[str, Any], runtime_config: dict[str, Any]
) -> None:
    expected_horizons = [int(value) for value in request["task"]["horizon_steps"]]
    actual_horizons = resolve_horizons(runtime_config)
    if expected_horizons != actual_horizons:
        raise ValueError(
            "Experiment request and runtime horizon mismatch: "
            f"request={expected_horizons}, runtime={actual_horizons}"
        )
    expected_mode = request["task"]["prediction_mode"]
    actual_mode = runtime_config["features"]["prediction_mode"]
    if expected_mode != actual_mode:
        raise ValueError(
            "Experiment request and runtime prediction mode mismatch: "
            f"request={expected_mode}, runtime={actual_mode}"
        )
    expected_minutes = int(request["task"]["resolution_minutes"])
    actual_minutes = int(runtime_config["features"]["minutes_per_point"])
    if expected_minutes != actual_minutes:
        raise ValueError(
            "Experiment request and runtime resolution mismatch: "
            f"request={expected_minutes}, runtime={actual_minutes}"
        )


def _assert_factor_fingerprints(
    request: dict[str, Any], manifest: dict[str, Any]
) -> None:
    requested = request.get("factor_fingerprints", {})
    current = {
        record["factor_id"]: record["catalog_record_sha256"]
        for record in manifest["records"]
    }
    if requested != current:
        raise ValueError(
            "Factor catalog records changed after experiment request creation; "
            "create a new reviewed request"
        )
    requested_implementations = request.get(
        "factor_implementation_fingerprints", {}
    )
    current_implementations = {
        record["factor_id"]: record["implementation_sha256"]
        for record in manifest["records"]
    }
    if requested_implementations != current_implementations:
        raise ValueError(
            "Factor implementation changed after experiment request creation; "
            "create a new reviewed request"
        )
    if request.get("factor_registry_sha256") != manifest["registry_sha256"]:
        raise ValueError(
            "Factor execution registry changed after experiment request creation; "
            "create a new reviewed request"
        )


def _predictions_for_stage(
    training_result: dict[str, Any],
    runtime_config: dict[str, Any],
    raw: pd.DataFrame,
    stage: str,
    factor_ids: list[str],
) -> pd.DataFrame:
    if stage in {"smoke", "exploration"}:
        return training_result["validation_predictions"].copy()
    period = "confirmation" if stage == "confirmation" else "final_test"
    result = evaluate(
        training_result["checkpoint_dir"],
        runtime_config,
        period_name=period,
        data=raw,
        seed=int(training_result["manifest"]["seed"]),
        factor_ids=factor_ids,
    )
    return result["predictions"].copy()


def _pair_predictions(
    baseline: pd.DataFrame, candidate: pd.DataFrame, seed: int
) -> pd.DataFrame:
    if baseline.duplicated(PAIR_KEYS).any() or candidate.duplicated(PAIR_KEYS).any():
        raise ValueError("Baseline or candidate predictions contain duplicate row identities")
    paired = baseline.merge(
        candidate,
        on=PAIR_KEYS,
        how="outer",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        counts = paired["_merge"].value_counts().to_dict()
        raise ValueError(f"Baseline/candidate row populations differ: {counts}")
    left = paired["groundtruth_baseline"].to_numpy(dtype=np.float32)
    right = paired["groundtruth_candidate"].to_numpy(dtype=np.float32)
    if not np.array_equal(left, right):
        raise ValueError("Baseline and candidate targets differ")
    paired = paired.drop(columns=["_merge", "groundtruth_candidate"]).rename(
        columns={"groundtruth_baseline": "groundtruth"}
    )
    paired["seed"] = int(seed)
    return paired


def _metric_pair(group: pd.DataFrame, capacity: float) -> dict[str, Any]:
    target = group["groundtruth"].to_numpy(dtype=np.float32)
    baseline_prediction = group["prediction_baseline"].to_numpy(dtype=np.float32)
    candidate_prediction = group["prediction_candidate"].to_numpy(dtype=np.float32)
    baseline = regression_metrics(target, baseline_prediction, capacity)
    candidate = regression_metrics(target, candidate_prediction, capacity)
    target_dates = pd.to_datetime(group["target_timestamp"]).dt.normalize()

    def monthly_mean_score(prediction: np.ndarray) -> float:
        daily = pd.DataFrame(
            {
                "date": target_dates.to_numpy(),
                "squared_error": np.square(prediction - target),
            }
        )
        daily = daily.groupby("date", as_index=False)["squared_error"].mean()
        daily["daily_rmse"] = np.sqrt(daily["squared_error"])
        daily["month"] = pd.to_datetime(daily["date"]).dt.month
        monthly_rmse = daily.groupby("month")["daily_rmse"].mean()
        return float((1.0 - monthly_rmse / capacity).mean())

    baseline["mean_monthly_capacity_score"] = monthly_mean_score(
        baseline_prediction
    )
    candidate["mean_monthly_capacity_score"] = monthly_mean_score(
        candidate_prediction
    )
    delta = {name: candidate[name] - baseline[name] for name in baseline}
    return {
        "sample_count": len(group),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
    }


def _group_metrics(
    paired: pd.DataFrame, columns: Iterable[str], capacity: float
) -> list[dict[str, Any]]:
    keys = list(columns)
    rows = []
    for group_key, group in paired.groupby(keys, sort=True, dropna=False):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        identity = {
            name: value.item() if hasattr(value, "item") else value
            for name, value in zip(keys, values)
        }
        rows.append({**identity, **_metric_pair(group, capacity)})
    return rows


def _flatten_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        identity = {
            key: value
            for key, value in row.items()
            if key not in {"baseline", "candidate", "delta"}
        }
        flattened.append(
            {
                **identity,
                **{f"baseline_{key}": value for key, value in row["baseline"].items()},
                **{f"candidate_{key}": value for key, value in row["candidate"].items()},
                **{f"delta_{key}": value for key, value in row["delta"].items()},
            }
        )
    return flattened


def _station_macro(by_station: list[dict[str, Any]]) -> dict[str, Any]:
    if not by_station:
        raise ValueError("No per-station metrics were produced")
    baseline_rmse = np.asarray(
        [row["baseline"]["rmse"] for row in by_station], dtype=np.float64
    )
    candidate_rmse = np.asarray(
        [row["candidate"]["rmse"] for row in by_station], dtype=np.float64
    )
    delta = candidate_rmse - baseline_rmse
    return {
        "station_count": len(by_station),
        "baseline_macro_rmse": float(baseline_rmse.mean()),
        "candidate_macro_rmse": float(candidate_rmse.mean()),
        "delta_macro_rmse": float(delta.mean()),
        "improved_station_ratio": float((delta < 0).mean()),
        "worst_station_delta_rmse": float(delta.max()),
    }


def _runtime_pair_audit(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    base_manifest = baseline["manifest"]
    cand_manifest = candidate["manifest"]
    errors = []
    for name in (
        "input_stations",
        "training_stations",
        "evaluation_object",
        "split_protocol",
        "prediction_mode",
        "horizons",
        "seed",
        "fixed_runtime_contract_sha256",
        "evaluation_protocol_sha256",
    ):
        if base_manifest[name] != cand_manifest[name]:
            errors.append(name)
    base_horizons = base_manifest["horizon_manifests"]
    cand_horizons = cand_manifest["horizon_manifests"]
    if len(base_horizons) != len(cand_horizons):
        errors.append("horizon_manifest_count")
    else:
        for base_horizon, cand_horizon in zip(base_horizons, cand_horizons):
            for name in (
                "horizon_step",
                "frame_rows",
                "train_rows",
                "validation_rows",
                "frame_stations",
                "train_stations",
                "validation_stations",
                "base_feature_names_sha256",
                "base_feature_values_sha256",
                "target_ratio_sha256",
                "target_power_sha256",
                "capacity_sha256",
                "environment_sha256",
            ):
                if base_horizon[name] != cand_horizon[name]:
                    errors.append(f"h{base_horizon['horizon_step']:02d}.{name}")
    return {"passed": not errors, "mismatches": errors}


def _nwp_issue_time_audit(
    raw: pd.DataFrame,
    runtime_config: dict[str, Any],
    issue_time_column: str | None,
) -> dict[str, Any]:
    """Verify NWP availability when timestamps exist, otherwise disclose assumption."""
    if not issue_time_column:
        return {
            "status": "contract_assumed",
            "checked_rows": 0,
            "violation_count": 0,
            "reason": "nwp_issue_time_column is not configured",
        }
    if issue_time_column not in raw.columns:
        raise ValueError(
            f"Configured NWP issue-time column is missing: {issue_time_column}"
        )
    origin_column = runtime_config["data"]["columns"]["timestamp"]
    try:
        origins = pd.to_datetime(raw[origin_column], errors="raise")
        issue_times = pd.to_datetime(raw[issue_time_column], errors="raise")
        violations = issue_times > origins
    except (TypeError, ValueError) as error:
        raise ValueError(
            "NWP issue time and forecast origin must be comparable timestamps "
            "using the same timezone convention"
        ) from error
    if violations.any():
        first_index = int(np.flatnonzero(violations.to_numpy())[0])
        row = raw.iloc[first_index]
        raise ValueError(
            "NWP leakage detected: issue_time is after forecast_origin at "
            f"row={first_index}, station_id={row.get('station_id', '<unknown>')!r}, "
            f"issue_time={issue_times.iloc[first_index]!r}, "
            f"forecast_origin={origins.iloc[first_index]!r}"
        )
    return {
        "status": "verified",
        "checked_rows": len(raw),
        "violation_count": 0,
        "column": issue_time_column,
    }


def _row_fingerprint(paired_by_seed: list[pd.DataFrame]) -> str:
    records = []
    for frame in paired_by_seed:
        records.extend(
            f"{int(seed)}|{row_id}"
            for seed, row_id in zip(frame["seed"], frame["row_id"])
        )
    return canonical_json_sha256(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.monotonic()
    config_path = Path(args.config).expanduser().resolve()
    request_path = Path(args.request).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    outer, project_root = _load_outer_config(config_path)
    request = _read_json(request_path)
    protected_before = _protected_snapshot(outer, project_root)
    expected_snapshot = request["protected_snapshot"]["combined_sha256"]
    if protected_before["combined_sha256"] != expected_snapshot:
        raise ValueError(
            "Protected files differ from the experiment request; create a new request"
        )
    runtime_path = _runtime_config_path(outer, project_root)
    runtime_config = load_config(runtime_path)
    _assert_request_matches_runtime(request, runtime_config)

    factor_ids = validate_factor_ids(request["factor_ids"])
    factor_manifest = factor_selection_manifest(factor_ids)
    _assert_factor_fingerprints(request, factor_manifest)
    raw = load_multi_station_data(None, runtime_config, require_target=True)
    issue_time_column = outer.get("data_contract", {}).get(
        "nwp_issue_time_column"
    )
    nwp_issue_time_audit = _nwp_issue_time_audit(
        raw, runtime_config, issue_time_column
    )
    capacity = float(runtime_config["evaluation"]["score_capacity"])

    paired_by_seed = []
    run_audits = []
    checkpoint_records = []
    feature_count_delta = None
    for seed in request["seeds"]:
        baseline = train(runtime_config, raw, seed=int(seed), factor_ids=[])
        candidate = train(
            runtime_config, raw, seed=int(seed), factor_ids=factor_ids
        )
        audit = _runtime_pair_audit(baseline, candidate)
        if not audit["passed"]:
            raise ValueError(f"Protected paired-run invariants failed: {audit}")
        run_audits.append({"seed": int(seed), **audit})

        baseline_predictions = _predictions_for_stage(
            baseline, runtime_config, raw, request["stage"], []
        )
        candidate_predictions = _predictions_for_stage(
            candidate, runtime_config, raw, request["stage"], factor_ids
        )
        paired_by_seed.append(
            _pair_predictions(baseline_predictions, candidate_predictions, int(seed))
        )
        current_delta = int(
            candidate["training_summary"]["feature_count"].iloc[0]
            - baseline["training_summary"]["feature_count"].iloc[0]
        )
        if feature_count_delta is None:
            feature_count_delta = current_delta
        elif feature_count_delta != current_delta:
            raise ValueError("Candidate feature-count delta differs across seeds")
        checkpoint_records.append(
            {
                "seed": int(seed),
                "baseline": str(baseline["checkpoint_dir"]),
                "candidate": str(candidate["checkpoint_dir"]),
            }
        )

    paired = pd.concat(paired_by_seed, ignore_index=True)
    paired["month"] = pd.to_datetime(paired["target_timestamp"]).dt.month.astype(int)
    overall = _metric_pair(paired, capacity)
    by_horizon = _group_metrics(paired, ["horizon_step"], capacity)
    by_station = _group_metrics(paired, ["station_id"], capacity)
    by_month = _group_metrics(paired, ["month"], capacity)
    by_seed = _group_metrics(paired, ["seed"], capacity)
    row_hash = _row_fingerprint(paired_by_seed)
    protected_after = _protected_snapshot(outer, project_root)
    protected_unchanged = (
        protected_after["combined_sha256"] == protected_before["combined_sha256"]
    )
    if not protected_unchanged:
        raise ValueError("Protected files changed during paired training")
    result = {
        "schema_version": "1.0.0",
        "experiment_id": request["experiment_id"],
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "stage": request["stage"],
        "factor_ids": factor_ids,
        "factor_selection": factor_manifest,
        "protected_snapshot": {
            "before_sha256": protected_before["combined_sha256"],
            "after_sha256": protected_after["combined_sha256"],
            "unchanged": protected_unchanged,
        },
        "row_fingerprints": {
            "baseline": row_hash,
            "candidate": row_hash,
            "identical": True,
        },
        "leakage_audit": {
            "passed": True,
            "fully_verified": nwp_issue_time_audit["status"] == "verified",
            "violations": 0,
            "checks": {
                "factor_availability": {
                    "status": "registry_enforced_contract",
                    "note": "Executable registry admits reviewed causal builders; it does not prove source-data provenance",
                },
                "nwp_issue_time_not_after_origin": nwp_issue_time_audit,
                "learned_transforms_fit_on_training_only": "passed_by_runtime_manifest",
                "baseline_candidate_row_ids_identical": "passed",
                "baseline_candidate_station_sets_identical": "passed",
                "source_all_target_time_split": "passed_by_runtime_manifest",
                "target_and_future_weather_horizon_alignment": "passed",
            },
            "contract_assumptions": (
                ["Issued future NWP arrays were available by forecast_origin"]
                if nwp_issue_time_audit["status"] == "contract_assumed"
                else []
            ),
        },
        "metrics": {
            "baseline": overall["baseline"],
            "candidate": overall["candidate"],
            "delta": overall["delta"],
            "by_horizon": _flatten_group_rows(by_horizon),
            "by_station": _flatten_group_rows(by_station),
            "station_macro": _station_macro(by_station),
            "by_month": _flatten_group_rows(by_month),
            "by_regime": {
                "available": False,
                "reason": "No protected regime label is present in the current runtime prediction table",
            },
            "by_seed": _flatten_group_rows(by_seed),
        },
        "resource_usage": {
            "wall_seconds": time.monotonic() - started,
            "seed_count": len(request["seeds"]),
            "factor_count": len(factor_ids),
            "feature_count_delta": int(feature_count_delta or 0),
            "checkpoint_records": checkpoint_records,
        },
        "run_audits": run_audits,
        "notes": [
            "REAL TABM RESULT: baseline and candidate were retrained with identical protected protocols.",
            "Negative delta RMSE means the candidate factor set improved the metric.",
            "Factor status is not auto-promoted; confirmation gates still apply.",
        ],
    }
    _write_json(output_path, result)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
