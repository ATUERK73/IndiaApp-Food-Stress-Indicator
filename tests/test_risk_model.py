import unittest

from src.risk_model import (
    enso_label,
    enso_stress_from_oni,
    fertilizer_stress,
    food_price_stress_from_change,
    monsoon_stress_from_anomaly,
    risk_label,
    risk_score,
    vegetation_soil_stress,
)


class RiskModelTests(unittest.TestCase):
    def test_monsoon_stress_bounds_and_direction(self):
        self.assertEqual(monsoon_stress_from_anomaly(10), 10)
        self.assertEqual(monsoon_stress_from_anomaly(-35), 100)
        self.assertLess(monsoon_stress_from_anomaly(-10), monsoon_stress_from_anomaly(-20))

    def test_enso_classification(self):
        self.assertEqual(enso_label(-1.6), "strong La Nina")
        self.assertEqual(enso_label(0.0), "neutral")
        self.assertEqual(enso_label(0.8), "El Nino")
        self.assertEqual(enso_label(1.6), "strong El Nino")
        self.assertLess(enso_stress_from_oni(0.0), enso_stress_from_oni(1.0))

    def test_component_and_total_scores_are_clamped(self):
        self.assertEqual(fertilizer_stress(5, 5, 5), 100)
        self.assertEqual(fertilizer_stress(-1, -1, -1), 0)
        self.assertEqual(risk_score(1000, 1000, 1000, 1000), 100)

    def test_food_price_stress(self):
        self.assertEqual(food_price_stress_from_change(-5), 10)
        self.assertEqual(food_price_stress_from_change(30), 100)
        self.assertLess(food_price_stress_from_change(5), food_price_stress_from_change(20))

    def test_crop_condition_stress_prioritizes_soil_moisture(self):
        soil_drought = vegetation_soil_stress(-30, 0)
        rain_only_drought = vegetation_soil_stress(0, -30)
        self.assertGreater(soil_drought, rain_only_drought)

    def test_extended_risk_score_accepts_crop_condition(self):
        score = risk_score(20, 20, 20, 20, 100)
        self.assertAlmostEqual(score, 32.0)

    def test_risk_labels(self):
        self.assertEqual(risk_label(30), "niedrig")
        self.assertEqual(risk_label(31), "elevated")
        self.assertEqual(risk_label(56), "hoch")
        self.assertEqual(risk_label(76), "kritisch")


if __name__ == "__main__":
    unittest.main()
