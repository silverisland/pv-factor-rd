#!/usr/bin/env python3
"""Run and report the bundled multi-station TabM baseline.

With no arguments this script trains the real TabM runtime on a small,
deterministic synthetic two-station dataset.  This is an execution smoke test;
its metric is not forecasting evidence.

Pass ``--config`` to train and report the baseline on private station data.
The configured data split, model, preprocessing, clipping, and metrics are used
without being reimplemented here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUNTIME_REQUIREMENTS = ROOT / "runtime" / "requirements.txt"
REQUIRED_MODULES = {
    "joblib": "joblib",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "yaml": "PyYAML",
    "rtdl_num_embeddings": "rtdl_num_embeddings",
    "sklearn": "scikit-learn",
    "tabm": "tabm",
    "torch": "torch",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the bundled multi-station TabM baseline and print its metrics."
    )
    parser.add_argument(
        "--config",
        help=(
            "Private runtime YAML. If omitted, run a small deterministic synthetic "
            "smoke test through the real TabM pipeline."
        ),
    )
    parser.add_argument(
        "--data",
        help=(
            "Optional station parquet file or parquet root overriding "
            "data.parquet_root in --config. Directories use data.parquet_glob."
        ),
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Training seed; repeat for multiple seeds. Defaults to the YAML seed.",
    )
    parser.add_argument(
        "--mode",
        choices=("endpoint", "curve"),
        help="Optional baseline mode override. curve trains horizons 1 through 16.",
    )
    parser.add_argument(
        "--factor",
        action="append",
        dest="factor_ids",
        help=(
            "Executable factor ID to test; repeat for a factor set. When given, "
            "the demo retrains paired empty-factor baseline and candidate runs."
        ),
    )
    parser.add_argument(
        "--report-dir",
        help="Directory for combined CSV/JSON reports. Defaults to state/demo_baseline_reports.",
    )
    return parser


def _require_dependencies() -> None:
    missing = [
        package
        for module, package in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        command = f'python3 -m pip install -r "{RUNTIME_REQUIREMENTS}"'
        raise SystemExit(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + "\nInstall them with:\n  "
            + command
        )


def _synthetic_config() -> dict[str, Any]:
    import yaml

    with (ROOT / "runtime" / "config.example.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    config["data"]["parquet_root"] = None
    config["features"]["prediction_mode"] = "endpoint"
    config["features"]["endpoint_horizon_step"] = 16
    config["training"].update(
        {
            "epochs": 3,
            "batch_size": 128,
            "inference_batch_size": 128,
            "early_stopping_patience": 2,
            "log_every_n_epochs": 1,
        }
    )
    config["split"] = {
        "train_start": "2024-09-01 00:00:00",
        "train_end": "2024-09-20 23:59:59",
        "validation_start": "2024-09-21 00:00:00",
        "validation_end": "2024-09-25 23:59:59",
        "test_start": "2024-09-26 00:00:00",
        "test_end": "2024-09-30 23:59:59",
        "train_stations": None,
        "validation_stations": ["demo_station_b"],
        "test_stations": ["demo_station_b"],
    }
    config["output"]["checkpoint_dir"] = str(
        ROOT / "state" / "checkpoints" / "demo_synthetic_tabm"
    )
    return config


def _synthetic_data():
    """Create finite causal arrays matching the private parquet column contract."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20260831)
    origins = pd.date_range(
        "2024-09-01 00:00:00", "2024-09-30 21:00:00", freq="3h"
    )
    rows: list[dict[str, Any]] = []

    def solar_profile(timestamp: pd.Timestamp) -> float:
        hour = timestamp.hour + timestamp.minute / 60.0
        angle = np.pi * (hour - 6.0) / 12.0
        return float(max(0.0, np.sin(angle)))

    for station_index, station_id in enumerate(("demo_station_a", "demo_station_b")):
        capacity = 420.0 + 30.0 * station_index
        site_scale = 0.92 + 0.06 * station_index
        for row_index, origin in enumerate(origins):
            history_times = pd.date_range(
                end=origin - pd.Timedelta(minutes=15), periods=96, freq="15min"
            )
            future_times = pd.date_range(
                start=origin + pd.Timedelta(minutes=15), periods=16, freq="15min"
            )
            phase = 0.17 * row_index + 0.8 * station_index
            history = []
            ghi_real = []
            for step, timestamp in enumerate(history_times):
                cloud = 0.78 + 0.16 * np.sin(phase + step / 13.0)
                noise = rng.normal(0.0, 2.0)
                observed_ghi = 1000.0 * solar_profile(timestamp) * cloud
                ghi_real.append(observed_ghi)
                history.append(
                    np.clip(
                        capacity * site_scale * observed_ghi / 1000.0
                        + noise,
                        0.0,
                        465.0,
                    )
                )

            ghi, temperature, wind_speed, wind_direction, target = [], [], [], [], []
            for step, timestamp in enumerate(future_times, start=1):
                solar = solar_profile(timestamp)
                cloud = 0.78 + 0.16 * np.sin(phase + (96 + step) / 13.0)
                current_ghi = 1000.0 * solar * cloud
                ghi.append(current_ghi)
                temperature.append(16.0 + 12.0 * solar + 0.5 * station_index)
                wind_speed.append(2.5 + 1.0 * np.sin(phase + step / 5.0))
                wind_direction.append(
                    (160.0 + 35.0 * np.sin(phase + step / 7.0)) % 360.0
                )
                target.append(
                    np.clip(
                        capacity * site_scale * current_ghi / 1000.0
                        + rng.normal(0.0, 2.0),
                        0.0,
                        465.0,
                    )
                )

            rows.append(
                {
                    "timestamp_win": origin,
                    "station": station_id,
                    "cap_power_on": capacity,
                    "site_capacity": capacity,
                    "site_longitude": 102.0 + station_index,
                    "site_latitude": 30.0 + 0.5 * station_index,
                    "site_timezone": "Asia/Shanghai",
                    "Power": np.asarray(history, dtype=np.float32),
                    "GHI_real": np.asarray(ghi_real, dtype=np.float32),
                    "Power_predict": np.asarray(target, dtype=np.float32),
                    "GHI_SOLARGIS_predict": np.asarray(ghi, dtype=np.float32),
                    "TEMP_SOLARGIS_predict": np.asarray(temperature, dtype=np.float32),
                    "WS_SOLARGIS_predict": np.asarray(wind_speed, dtype=np.float32),
                    "WD_SOLARGIS_predict": np.asarray(wind_direction, dtype=np.float32),
                }
            )
    return pd.DataFrame(rows)


def _apply_mode(config: dict[str, Any], mode: str | None) -> dict[str, Any]:
    result = deepcopy(config)
    if mode is None:
        return result
    result["features"]["prediction_mode"] = mode
    if mode == "curve":
        result["features"]["horizons"] = "all"
    return result


def _population_std(series) -> float:
    return float(series.std(ddof=0))


def _summarize(frames, group_columns: list[str]):
    import pandas as pd

    combined = pd.concat(frames, ignore_index=True)
    numeric_metrics = [
        name
        for name in (
            "rmse",
            "mae",
            "bias",
            "nrmse_by_capacity",
            "capacity_score",
            "mean_monthly_capacity_score",
            "macro_rmse",
            "macro_mae",
            "macro_capacity_score",
            "worst_station_rmse",
            "worst_station_capacity_score",
        )
        if name in combined.columns
    ]
    aggregations: dict[str, tuple[str, Any]] = {}
    for metric in numeric_metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, _population_std)
    summary = (
        combined.groupby(group_columns, as_index=False).agg(**aggregations)
        if group_columns
        else pd.DataFrame()
    )
    return combined, summary


def _write_reports(
    results: list[dict[str, Any]], report_dir: Path, *, synthetic: bool
) -> None:
    report_dir.mkdir(parents=True, exist_ok=False)
    horizon_frames = []
    group_frames = []
    station_macro_frames = []
    run_index = []
    for item in results:
        seed = int(item["manifest"]["seed"])
        evaluation = item["test_evaluation"]
        horizon_frames.append(evaluation["by_horizon"].assign(seed=seed))
        group_frames.append(evaluation["by_horizon_group"].assign(seed=seed))
        station_macro_frames.append(
            evaluation["station_macro_summary"].assign(seed=seed)
        )
        run_index.append(
            {
                "seed": seed,
                "checkpoint_dir": str(item["checkpoint_dir"]),
                "run_contract_sha256": item["manifest"]["run_contract_sha256"],
            }
        )

    by_seed_horizon, summary_horizon = _summarize(
        horizon_frames, ["horizon_step"]
    )
    by_seed_group, summary_group = _summarize(
        group_frames, ["horizon_group"]
    )
    by_seed_macro, summary_macro = _summarize(
        station_macro_frames, ["horizon_step"]
    )

    by_seed_horizon.to_csv(report_dir / "metrics_by_seed_horizon.csv", index=False)
    summary_horizon.to_csv(report_dir / "metrics_summary_by_horizon.csv", index=False)
    by_seed_group.to_csv(report_dir / "metrics_by_seed_horizon_group.csv", index=False)
    summary_group.to_csv(
        report_dir / "metrics_summary_by_horizon_group.csv", index=False
    )
    by_seed_macro.to_csv(report_dir / "station_macro_by_seed_horizon.csv", index=False)
    summary_macro.to_csv(
        report_dir / "station_macro_summary_by_horizon.csv", index=False
    )
    (report_dir / "runs.json").write_text(
        json.dumps(
            {
                "synthetic_smoke_test": synthetic,
                "forecasting_evidence": not synthetic,
                "runs": run_index,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Test metrics: seed × horizon ===")
    print(by_seed_horizon.to_string(index=False))
    print("\n=== Test metrics: aggregate across seeds ===")
    print(summary_horizon.to_string(index=False))
    print("\n=== Station-macro metrics: aggregate across seeds ===")
    print(summary_macro.to_string(index=False))
    if not summary_group.empty:
        print("\n=== Horizon-group metrics: aggregate across seeds ===")
        print(summary_group.to_string(index=False))
    print(f"\nReports: {report_dir}")
    for run in run_index:
        print(f"Checkpoint (seed={run['seed']}): {run['checkpoint_dir']}")
    if synthetic:
        print(
            "\nWARNING: synthetic smoke-test metrics only verify execution; "
            "they are not forecasting evidence."
        )


def _write_factor_reports(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    report_dir: Path,
    factor_ids: list[str],
    *,
    synthetic: bool,
) -> None:
    import numpy as np
    import pandas as pd

    report_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    audits = []
    for baseline, candidate in pairs:
        seed = int(baseline["manifest"]["seed"])
        left = baseline["validation_predictions"]
        right = candidate["validation_predictions"]
        keys = ["row_id", "station_id", "timestamp", "target_timestamp", "horizon_step"]
        paired = left.merge(
            right,
            on=keys,
            how="outer",
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
            indicator=True,
        )
        rows_identical = bool(paired["_merge"].eq("both").all())
        targets_identical = rows_identical and np.array_equal(
            paired["groundtruth_baseline"].to_numpy(),
            paired["groundtruth_candidate"].to_numpy(),
        )
        if not rows_identical or not targets_identical:
            raise ValueError(
                f"Paired factor audit failed for seed={seed}: "
                f"rows_identical={rows_identical}, targets_identical={targets_identical}"
            )
        for horizon, group in paired.groupby("horizon_step", sort=True):
            target = group["groundtruth_baseline"].to_numpy(dtype=np.float64)
            base = group["prediction_baseline"].to_numpy(dtype=np.float64)
            cand = group["prediction_candidate"].to_numpy(dtype=np.float64)
            baseline_rmse = float(np.sqrt(np.mean(np.square(base - target))))
            candidate_rmse = float(np.sqrt(np.mean(np.square(cand - target))))
            rows.append(
                {
                    "seed": seed,
                    "horizon_step": int(horizon),
                    "sample_count": len(group),
                    "baseline_rmse": baseline_rmse,
                    "candidate_rmse": candidate_rmse,
                    "delta_rmse": candidate_rmse - baseline_rmse,
                }
            )
        audits.append(
            {
                "seed": seed,
                "rows_identical": rows_identical,
                "targets_identical": targets_identical,
                "baseline_checkpoint": str(baseline["checkpoint_dir"]),
                "candidate_checkpoint": str(candidate["checkpoint_dir"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(report_dir / "paired_factor_metrics.csv", index=False)
    (report_dir / "paired_factor_audit.json").write_text(
        json.dumps(
            {
                "factor_ids": factor_ids,
                "synthetic_smoke_test": synthetic,
                "forecasting_evidence": not synthetic,
                "audits": audits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n=== Paired factor validation (negative delta_rmse is better) ===")
    print(table.to_string(index=False))
    print(f"\nReports: {report_dir}")
    if synthetic:
        print("\nWARNING: synthetic results verify execution only, not factor value.")



def main() -> int:
    args = _parser().parse_args()
    _require_dependencies()
    # Keep relative data/output paths consistent with the documented Skill-root CLI.
    os.chdir(ROOT)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from runtime.multi_station_tabm.api import evaluate, train
    from runtime.multi_station_tabm.config import load_config
    from factor_library.implementations.registry import validate_factor_ids

    synthetic = args.config is None
    if synthetic:
        if args.data:
            raise SystemExit(
                "--data requires --config; omit both for the synthetic smoke test"
            )
        config = _synthetic_config()
        data = _synthetic_data()
    else:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.is_file():
            raise SystemExit(f"Config does not exist: {config_path}")
        config = load_config(config_path)
        # Keep the path as the data source. The runtime reads one station file,
        # constructs numerical features immediately, and only then advances to
        # the next file.
        data = args.data

    config = _apply_mode(config, args.mode)
    # Validate the effective mapping after any mode override.
    config = load_config(config)
    seeds = args.seed or [int(config["training"]["seed"])]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"Duplicate seeds are not allowed: {seeds}")

    mode = config["features"]["prediction_mode"]
    print(
        f"Running {'synthetic smoke' if synthetic else 'private-data'} baseline: "
        f"mode={mode}, seeds={seeds}"
    )
    factor_ids = validate_factor_ids(args.factor_ids)
    if factor_ids:
        pairs = [
            (
                train(config, data, seed=seed, factor_ids=[]),
                train(config, data, seed=seed, factor_ids=factor_ids),
            )
            for seed in seeds
        ]
        results = []
    else:
        pairs = []
        results = []
        for seed in seeds:
            trained = train(config, data, seed=seed, factor_ids=[])
            trained["test_evaluation"] = evaluate(
                trained["checkpoint_dir"],
                config,
                period_name="final_test",
                data=data,
                seed=seed,
                factor_ids=[],
            )
            results.append(trained)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_root = (
        Path(args.report_dir).expanduser().resolve()
        if args.report_dir
        else ROOT / "state" / "demo_baseline_reports"
    )
    if factor_ids:
        _write_factor_reports(
            pairs,
            report_root / f"factor-{timestamp}",
            factor_ids,
            synthetic=synthetic,
        )
    else:
        _write_reports(
            results,
            report_root / f"baseline-{timestamp}",
            synthetic=synthetic,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
