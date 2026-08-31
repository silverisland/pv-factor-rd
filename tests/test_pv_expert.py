from __future__ import annotations

import math
import sys
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factor_library.implementations.pv_expert import (  # noqa: E402
    build_capacity_ratio,
    build_clear_sky_irradiance,
    build_clear_variable_overcast_regime,
    build_clipping_score,
    build_previous_day_profile,
    solar_position,
)


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


if __name__ == "__main__":
    unittest.main()
