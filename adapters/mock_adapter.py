#!/usr/bin/env python3
"""Deterministic synthetic adapter for protocol tests only—never real evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.monotonic()
    request = read_json(Path(args.request).resolve())
    signature = hashlib.sha256("|".join(request["factor_ids"]).encode("utf-8")).hexdigest()
    synthetic_gain = (int(signature[:8], 16) % 21 - 15) / 10000.0
    baseline_rmse = 0.125
    candidate_rmse = baseline_rmse + synthetic_gain
    horizons = request["task"]["horizon_steps"]
    by_horizon = []
    for step in horizons:
        base = baseline_rmse + 0.0015 * (int(step) - 1)
        candidate = base + synthetic_gain * (0.6 + 0.4 * int(step) / max(horizons))
        by_horizon.append({
            "horizon_step": step,
            "baseline_rmse": round(base, 6),
            "candidate_rmse": round(candidate, 6),
            "delta_rmse": round(candidate - base, 6),
        })

    snapshot_hash = request["protected_snapshot"]["combined_sha256"]
    row_hash = hashlib.sha256((request["experiment_id"] + "synthetic-rows").encode("utf-8")).hexdigest()
    result = {
        "schema_version": "1.0.0",
        "experiment_id": request["experiment_id"],
        "completed_at": request["created_at"],
        "status": "completed",
        "stage": request["stage"],
        "factor_ids": request["factor_ids"],
        "protected_snapshot": {
            "before_sha256": snapshot_hash,
            "after_sha256": snapshot_hash,
            "unchanged": True,
        },
        "row_fingerprints": {"baseline": row_hash, "candidate": row_hash, "identical": True},
        "leakage_audit": {
            "passed": True,
            "violations": 0,
            "checks": {check: "synthetic_pass" for check in request["leakage_checks"]},
        },
        "metrics": {
            "baseline": {"rmse": baseline_rmse, "nrmse": baseline_rmse},
            "candidate": {"rmse": round(candidate_rmse, 6), "nrmse": round(candidate_rmse, 6)},
            "delta": {"rmse": round(synthetic_gain, 6), "nrmse": round(synthetic_gain, 6)},
            "by_horizon": by_horizon,
            "by_station": [
                {
                    "station_id": "synthetic_station",
                    "baseline_rmse": baseline_rmse,
                    "candidate_rmse": round(candidate_rmse, 6),
                    "delta_rmse": round(synthetic_gain, 6),
                }
            ],
            "station_macro": {
                "station_count": 1,
                "delta_rmse": round(synthetic_gain, 6),
                "worst_station_delta_rmse": round(synthetic_gain, 6),
            },
            "by_month": {
                "month_count": 4,
                "positive_month_ratio": 0.75 if synthetic_gain < 0 else 0.25,
                "worst_month_delta_rmse": round(abs(synthetic_gain) + 0.001, 6),
            },
            "by_regime": {
                "stable": {"delta_rmse": round(synthetic_gain * 0.6, 6)},
                "ramp": {"delta_rmse": round(synthetic_gain * 1.4, 6)},
            },
        },
        "resource_usage": {"wall_seconds": round(time.monotonic() - started, 6), "feature_count_delta": len(request["factor_ids"])},
        "notes": [
            "SYNTHETIC MOCK RESULT: validates orchestration only.",
            "Do not use this output to promote, reject, or rank any factor.",
        ],
    }
    write_json(Path(args.output).resolve(), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
