import unittest

import pandas as pd

from src.forecast_model import simulate_three_month_forecast


class ForecastModelTests(unittest.TestCase):
    def test_forecast_has_three_bounded_months(self):
        forecast = simulate_three_month_forecast(
            reference_date=pd.Timestamp("2026-06-22"),
            rainfall_anomaly_pct=-12,
            soil_moisture_anomaly_pct=-8,
            oni=0.7,
            import_exposure=0.65,
            fertilizer_price_ratio=1.2,
            food_price_stress=35,
            simulations=500,
        )
        self.assertEqual(len(forecast), 3)
        self.assertEqual(forecast.iloc[0]["date"], pd.Timestamp("2026-07-01"))
        self.assertTrue((forecast["p10"] <= forecast["median"]).all())
        self.assertTrue((forecast["median"] <= forecast["p90"]).all())
        self.assertTrue(forecast[["p10", "median", "p90"]].ge(0).all().all())
        self.assertTrue(forecast[["p10", "median", "p90"]].le(100).all().all())

    def test_class_probabilities_sum_to_one(self):
        forecast = simulate_three_month_forecast(
            reference_date=pd.Timestamp("2026-06-22"),
            rainfall_anomaly_pct=0,
            soil_moisture_anomaly_pct=0,
            oni=0,
            import_exposure=0.65,
            fertilizer_price_ratio=1,
            food_price_stress=25,
            simulations=300,
        )
        total = forecast[["prob_low", "prob_elevated", "prob_high", "prob_critical"]].sum(axis=1)
        self.assertTrue((total.sub(1.0).abs() < 1e-12).all())

    def test_too_few_simulations_are_rejected(self):
        with self.assertRaises(ValueError):
            simulate_three_month_forecast(
                pd.Timestamp("2026-06-22"), 0, 0, 0, 0.65, 1, 25, simulations=10
            )

    def test_dry_and_favorable_scenarios_bracket_baseline(self):
        arguments = dict(
            reference_date=pd.Timestamp("2026-06-22"),
            rainfall_anomaly_pct=-12,
            soil_moisture_anomaly_pct=-8,
            oni=0.7,
            import_exposure=0.65,
            fertilizer_price_ratio=1.2,
            food_price_stress=35,
            simulations=1000,
            seed=7,
        )
        baseline = simulate_three_month_forecast(**arguments, scenario="baseline")
        dry = simulate_three_month_forecast(**arguments, scenario="dry")
        favorable = simulate_three_month_forecast(**arguments, scenario="favorable")
        self.assertTrue((dry["median"] >= baseline["median"]).all())
        self.assertTrue((baseline["median"] >= favorable["median"]).all())

    def test_unknown_scenario_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate_three_month_forecast(
                pd.Timestamp("2026-06-22"), 0, 0, 0, 0.65, 1, 25, scenario="unknown"
            )

    def test_official_strengthening_signal_raises_oni_projection(self):
        arguments = dict(
            reference_date=pd.Timestamp("2026-06-22"),
            rainfall_anomaly_pct=0,
            soil_moisture_anomaly_pct=0,
            oni=0.5,
            import_exposure=0.65,
            fertilizer_price_ratio=1,
            food_price_stress=25,
            wet_bulb_temperature_c=25,
            simulations=1000,
            seed=9,
        )
        history_only = simulate_three_month_forecast(**arguments)
        official = simulate_three_month_forecast(
            **arguments,
            enso_weekly_nino34_c=0.7,
            enso_expected_to_strengthen=True,
        )
        self.assertGreater(
            official.iloc[-1]["oni_value_median"],
            history_only.iloc[-1]["oni_value_median"],
        )


if __name__ == "__main__":
    unittest.main()
