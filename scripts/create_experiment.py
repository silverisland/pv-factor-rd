#!/usr/bin/env python3
"""Create a reviewable fixed-model factor experiment request."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from catalog_lib import (
    SKILL_ROOT,
    factor_index,
    load_config,
    read_json,
    sha256_file,
    sha256_json,
    utc_now,
    write_json,
)


FINAL_CONFIRMATION = "RUN_SEALED_FINAL_TEST"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--factor", action="append", required=True, dest="factor_ids")
    parser.add_argument("--hypothesis")
    parser.add_argument("--stage", choices=["smoke", "exploration", "confirmation", "final_test"], default="exploration")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--output")
    parser.add_argument("--confirm", help="Required exact phrase for final_test")
    args = parser.parse_args()

    config, _, project_root = load_config(args.config)
    factors = factor_index()
    unknown = sorted(set(args.factor_ids) - set(factors))
    if unknown:
        raise SystemExit(f"Unknown factor IDs: {unknown}")
    if args.stage == "final_test" and args.confirm != FINAL_CONFIRMATION:
        raise SystemExit(f"Final test is sealed; pass --confirm {FINAL_CONFIRMATION} only after human approval")

    snapshot_path = project_root / config.get("state_dir", "state") / "project_snapshot.json"
    if not snapshot_path.is_file():
        raise SystemExit(f"Missing {snapshot_path}; run inspect_project.py first")
    snapshot = read_json(snapshot_path)
    seed_values = args.seeds or config["experiment"].get("default_seeds", [0])
    selected = [factors[factor_id] for factor_id in args.factor_ids]
    implementation_fingerprints = {}
    for factor in selected:
        implementation = factor.get("implementation")
        if implementation:
            relative_path, _ = implementation.split(":", 1)
            implementation_fingerprints[factor["id"]] = sha256_file(
                SKILL_ROOT / relative_path
            )
    hypothesis = args.hypothesis or (
        f"Adding {', '.join(factor['name'] for factor in selected)} improves at least one predeclared "
        "0-4h validation slice without material regression in predeclared temporal regimes."
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    signature = sha256_json({"factor_ids": args.factor_ids, "stage": args.stage, "snapshot": snapshot["combined_sha256"], "time": timestamp})[:8]
    experiment_id = f"exp-{timestamp}-{signature}"
    request = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "stage": args.stage,
        "baseline_id": config["experiment"]["baseline_id"],
        "factor_ids": args.factor_ids,
        "factor_fingerprints": {factor["id"]: sha256_json(factor) for factor in selected},
        "factor_implementation_fingerprints": implementation_fingerprints,
        "factor_registry_sha256": sha256_file(
            SKILL_ROOT / "factor_library" / "implementations" / "registry.py"
        ),
        "hypothesis": hypothesis,
        "seeds": seed_values,
        "task": config["task"],
        "protected_snapshot": snapshot,
        "leakage_checks": [
            "all_features_available_by_forecast_origin",
            "nwp_issue_time_not_after_origin",
            "learned_transforms_fit_on_training_only",
            "baseline_candidate_row_ids_identical",
            "baseline_candidate_station_sets_identical",
            "source_all_target_history_split_by_target_time",
            "test_train_validation_evaluation_target_times_disjoint",
            "target_and_future_weather_horizon_alignment",
        ],
        "requested_slices": [
            "overall", "by_station", "station_macro", "worst_station",
            "each_horizon", "0_1h", "1_2h", "2_4h", "by_month",
            "month", "daylight_low_sun", "stable_ramp", "factor_coverage",
        ],
    }
    output = Path(args.output).expanduser().resolve() if args.output else project_root / config.get("state_dir", "state") / "requests" / f"{experiment_id}.json"
    write_json(output, request)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
