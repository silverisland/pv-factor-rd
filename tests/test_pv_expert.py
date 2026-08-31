from __future__ import annotations

import math
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factor_library.implementations.pv_expert import (  # noqa: E402
    build_capacity_ratio,
    build_clear_sky_irradiance,
    build_clear_variable_overcast_regime,
    build_clipping_score,
    build_daylight_boundary,
    build_joint_weather_power_ramp,
    build_low_irradiance_state,
    build_module_temperature_proxy,
    build_previous_day_profile,
    build_temperature_corrected_irradiance,
    build_weather_power_residual,
    solar_position,
)
from factor_library.implementations import pv_expert  # noqa: E402


class PvExpertFactorTests(unittest.TestCase):
    def test_capacity_ratio_and_previous_day_are_causal(self):
        history = list(range(96))
        self.assertEqual(
            build_capacity_ratio(history, 100.0, (1, 4)),
            {
                "power_capacity_ratio_lag_1": 0.95,
                "power_capacity_ratio_lag_4": 0.92,
            },
        )
        self.assertEqual(
            build_previous_day_profile(history, 16),
            {"previous_day_same_target_power": 16.0},
        )

    def test_solar_geometry_has_day_night_consistency(self):
        noon = datetime(2024, 6, 21, 12, 0)
        midnight = datetime(2024, 6, 21, 0, 0)
        position = solar_position(noon, 31.2, 121.5)
        self.assertGreater(position["solar_elevation_degrees"], 60.0)
        self.assertGreater(
            build_clear_sky_irradiance(noon, 31.2, 121.5)["clear_sky_ghi"],
            700.0,
        )
        self.assertEqual(
            build_clear_sky_irradiance(midnight, 31.2, 121.5)["clear_sky_ghi"],
            0.0,
        )

    def test_regime_and_clipping_outputs_are_finite(self):
        ghi = [max(0.0, 900.0 * math.sin(math.pi * i / 95.0)) for i in range(96)]
        regime = build_clear_variable_overcast_regime(
            ghi, datetime(2024, 6, 21, 18, 0), 31.2, 121.5
        )
        clipping = build_clipping_score([99.0] * 8, [900.0] * 8, 100.0)
        self.assertTrue(all(math.isfinite(value) for value in regime.values()))
        self.assertGreater(clipping["clipping_score"], 0.9)

    def test_regime_only_computes_the_trailing_sixteen_solar_points(self):
        original = pv_expert.build_clear_sky_irradiance
        with patch.object(
            pv_expert, "build_clear_sky_irradiance", wraps=original
        ) as clear_sky:
            build_clear_variable_overcast_regime(
                [500.0] * 96, datetime(2024, 6, 21, 12, 0), 31.2, 121.5
            )
        self.assertEqual(clear_sky.call_count, 16)

    def test_thermal_low_resource_and_daylight_features(self):
        module = build_module_temperature_proxy(800.0, 25.0, 2.0)
        corrected = build_temperature_corrected_irradiance(
            800.0, module["module_temperature_proxy"]
        )
        low = build_low_irradiance_state(100.0, 500.0, 5.0)
        daylight = build_daylight_boundary(
            datetime(2024, 6, 21, 12, 0), 31.2, 121.5
        )
        self.assertGreater(module["module_temperature_proxy"], 25.0)
        self.assertLess(corrected["temperature_derating_multiplier"], 1.0)
        self.assertEqual(low["low_irradiance_flag"], 1.0)
        self.assertEqual(low["low_sun_flag"], 1.0)
        self.assertEqual(daylight["is_daylight"], 1.0)

    def test_joint_ramp_and_weather_power_residual(self):
        ramp = build_joint_weather_power_ramp(
            [10.0, 20.0], [100.0, 200.0], 350.0, 500.0
        )
        residual = build_weather_power_residual(
            [50.0, 100.0], [100.0, 300.0], 500.0
        )
        self.assertAlmostEqual(ramp["recent_power_ramp_capacity_ratio"], 0.02)
        self.assertAlmostEqual(ramp["future_ghi_change_from_observation"], 0.15)
        self.assertTrue(all(math.isfinite(value) for value in residual.values()))

    def test_invalid_capacity_and_empty_joint_history_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            build_capacity_ratio([1.0], 0.0, (1,))
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_joint_weather_power_ramp([1.0], [2.0], 3.0, 10.0)


if __name__ == "__main__":
    unittest.main()
