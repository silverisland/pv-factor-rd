from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factor_library.implementations.core_timeseries import (  # noqa: E402
    build_future_changes,
    build_future_weather_quality,
    build_power_acceleration,
    build_power_lags,
    build_power_ramps,
    build_power_slopes,
    build_power_variability,
    build_stuck_shift,
)


class CoreTimeSeriesTests(unittest.TestCase):
    def test_lags_use_only_past_values(self):
        self.assertEqual(build_power_lags([10, 20, 30], (1, 2)), {"power_lag_1": 30.0, "power_lag_2": 20.0})
        self.assertTrue(math.isnan(build_power_lags([10], (2,))["power_lag_2"]))

    def test_ramp_slope_variability_and_acceleration(self):
        history = [0, 1, 2, 3, 4]
        self.assertEqual(build_power_ramps(history, (1, 4))["power_ramp_4"], 4.0)
        self.assertAlmostEqual(build_power_slopes(history, (4,))["power_slope_4"], 1.0)
        self.assertAlmostEqual(build_power_variability(history, (4,))["power_absdiff_4"], 1.0)
        self.assertAlmostEqual(build_power_acceleration(history)["power_acceleration_1"], 0.0)

    def test_future_changes_preserve_horizon_order(self):
        result = build_future_changes([12, 15], 10)
        self.assertEqual(result["future_change_from_origin_h2"], 5.0)
        self.assertEqual(result["future_step_change_h2"], 3.0)

    def test_quality_features(self):
        stuck = build_stuck_shift([1, 1, 1, 2])
        self.assertEqual(stuck["stuck_longest_fraction"], 0.75)
        quality = build_future_weather_quality({"ghi": [1, float("nan"), 3]}, expected_steps=4)
        self.assertEqual(quality["ghi_coverage"], 0.5)
        self.assertEqual(quality["ghi_length_ratio"], 0.75)


if __name__ == "__main__":
    unittest.main()

