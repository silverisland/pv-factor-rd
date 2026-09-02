from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "runtime" / "multi_station_tabm"


class MultiStationRuntimeTests(unittest.TestCase):
    def test_baseline_demo_is_present_and_help_has_no_training_dependency(self):
        demo = ROOT / "test_demo.py"
        self.assertTrue(demo.is_file())
        compile(demo.read_text(encoding="utf-8"), str(demo), "exec")
        completed = subprocess.run(
            [sys.executable, str(demo), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--config", completed.stdout)
        self.assertIn("--mode", completed.stdout)
        self.assertIn("--factor", completed.stdout)

    def test_runtime_is_present_and_syntax_valid(self):
        names = {
            "__init__.py", "api.py", "cli.py", "config.py", "data.py",
            "evaluator.py", "features.py", "fingerprints.py", "metrics.py",
            "model.py", "preprocessing.py", "splits.py", "trainer.py",
        }
        self.assertEqual(names, {path.name for path in PACKAGE.glob("*.py")})
        for path in PACKAGE.glob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        self.assertFalse((ROOT / "runtime" / "single_station_tabm").exists())
        self.assertFalse((ROOT / "runtime" / "province_tabm_engineered").exists())

    def test_training_layers_are_separated_and_preserve_contract(self):
        model = (PACKAGE / "model.py").read_text(encoding="utf-8")
        preprocessing = (PACKAGE / "preprocessing.py").read_text(encoding="utf-8")
        trainer = (PACKAGE / "trainer.py").read_text(encoding="utf-8")
        evaluator = (PACKAGE / "evaluator.py").read_text(encoding="utf-8")
        metrics = (PACKAGE / "metrics.py").read_text(encoding="utf-8")
        for marker in ("LinearReLUEmbeddings", "tabm.TabM.make", "d_out=1"):
            self.assertIn(marker, model)
        for forbidden in ("QuantileTransformer", "AdamW", "mse_loss"):
            self.assertNotIn(forbidden, model)
        for marker in ("QuantileTransformer", "prepare_training_data", "fit_partition"):
            self.assertIn(marker, preprocessing)
        for marker in (
            "torch.optim.AdamW", "mse_loss", "repeat_interleave",
            "loss.backward()", "optimizer.step()", "validation_ratio_rmse_unclipped",
        ):
            self.assertIn(marker, trainer)
        self.assertIn("regression_metrics", evaluator)
        self.assertIn("daily_and_monthly_metrics", metrics)

    def test_features_pool_station_rows_without_station_aggregation(self):
        source = (PACKAGE / "features.py").read_text(encoding="utf-8")
        for marker in ("power_history", "future_weather", "power_future", "history_length"):
            self.assertIn(marker, source)
        for marker in (
            '"forecast_object": "multi_station_shared_model"',
            '"station_id_as_model_feature": False',
            '"training_rows": "pooled_across_stations"',
            '"station_aggregation": False',
            '"capacity_weighted_weather": False',
        ):
            self.assertIn(marker, source)
        for forbidden in ("province", "plant_station", ".merge(", ".pivot("):
            self.assertNotIn(forbidden, source)

    def test_portable_config_protects_only_local_multi_station_code(self):
        config = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
        protected = config["model_paths"] + config["protocol_paths"]
        self.assertTrue(protected)
        for value in protected:
            path = Path(value)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertIn("multi_station_tabm", path.parts)
            self.assertTrue((ROOT / path).is_file(), value)

    def test_runtime_config_reproduces_endpoint_contract(self):
        text = (ROOT / "runtime" / "config.example.yaml").read_text(encoding="utf-8")
        for marker in (
            "parquet_glob: station=*.parquet", "station_id_column: station",
            "sampling_strategy: pooled_rows", "history_length: 96", "n_horizons: 16",
            "prediction_mode: endpoint", "endpoint_horizon_step: 16",
            "label_normalization: none", "label_scale_value: 1.0",
            "prediction_clip: [0.0, 1.2]",
            "training_stations: null", "test_stations:",
            "- replace_with_test_station",
            "source_station_time_policy: all_available",
            "strategy: target_history_tail", "purge_hours: 4",
            "require_all_training_stations_in_evaluation: false",
            "primary_metric: mean_monthly_capacity_score",
            "early_stopping_prediction: unclipped_ratio",
        ):
            self.assertIn(marker, text)
        for obsolete in (
            "file_prefix:", "file_suffix:", "_v1.parquet",
            "validation_last_days_per_month:",
            "require_same_stations_in_development_splits:",
        ):
            self.assertNotIn(obsolete, text)
        for forbidden in ("/Users/", "/home/", "/jtdata/", "/data/"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
