import unittest

import pandas as pd

from src.data_sources import (
    build_fertilizer_price_index,
    build_monsoon_anomaly,
    fertilizer_price_shock_factor,
    load_india_fertilizer_prices,
    oni_table_to_series,
    parse_noaa_enso_outlook,
    wet_bulb_temperature_c,
)


class RainfallAnomalyTests(unittest.TestCase):
    def test_builds_sorted_cumulative_anomaly(self):
        current = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-05-02", "2026-05-01"]),
                "precipitation_mm": [20.0, 5.0],
            }
        )
        baseline = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-05-01", "2025-05-02"]),
                "precipitation_mm": [10.0, 10.0],
            }
        )

        result = build_monsoon_anomaly(current, baseline)

        self.assertEqual(result["date"].tolist(), sorted(result["date"].tolist()))
        self.assertAlmostEqual(result["rainfall_anomaly_pct"].iloc[-1], 25.0)

    def test_rejects_empty_inputs(self):
        empty = pd.DataFrame(columns=["date", "precipitation_mm"])
        with self.assertRaises(ValueError):
            build_monsoon_anomaly(empty, empty)


class WetBulbTemperatureTests(unittest.TestCase):
    def test_higher_humidity_raises_wet_bulb_temperature(self):
        dry = wet_bulb_temperature_c(30.0, 30.0)
        humid = wet_bulb_temperature_c(30.0, 80.0)
        self.assertLess(dry, humid)
        self.assertLess(humid, 30.5)

    def test_typical_hot_humid_value_is_plausible(self):
        result = wet_bulb_temperature_c(30.0, 70.0)
        self.assertGreater(result, 24.0)
        self.assertLess(result, 28.0)


class OniSeriesTests(unittest.TestCase):
    def test_converts_seasons_to_chronological_dates(self):
        table = pd.DataFrame({
            "Year": [2025],
            "DJF": [-0.5], "JFM": [-0.3], "FMA": [-0.1], "MAM": [0.0],
            "AMJ": [0.1], "MJJ": [0.2], "JJA": [0.3], "JAS": [0.4],
            "ASO": [0.5], "SON": [0.6], "OND": [0.7], "NDJ": [0.8],
        })

        result = oni_table_to_series(table)

        self.assertEqual(result.iloc[0]["date"], pd.Timestamp("2025-02-01"))
        self.assertEqual(result.iloc[-1]["date"], pd.Timestamp("2026-01-01"))
        self.assertEqual(result.iloc[-1]["season"], "NDJ")
        self.assertAlmostEqual(result.iloc[-1]["oni"], 0.8)

    def test_parses_forward_noaa_enso_outlook(self):
        html = """
        <html><body>
        issued by CLIMATE PREDICTION CENTER/NCEP/NWS 11 June 2026
        ENSO Alert System Status: El Niño Advisory
        Synopsis: El Niño conditions are present and expected to strengthen into winter 2026-27.
        The latest weekly Niño-3.4 index value was +0.7°C.
        There is a 63% chance of a very strong El Niño during November-January [Fig. 8]
        </body></html>
        """
        result = parse_noaa_enso_outlook(html)
        self.assertEqual(result["alert_status"], "El Niño Advisory")
        self.assertAlmostEqual(result["weekly_nino34_c"], 0.7)
        self.assertTrue(result["expected_to_strengthen"])
        self.assertEqual(result["very_strong_probability_pct"], 63)
        self.assertEqual(result["very_strong_period"], "November-January")


class FertilizerPriceTests(unittest.TestCase):
    def test_builds_weighted_price_index_and_shock_factor(self):
        dates = pd.date_range("2024-01-01", periods=62, freq="MS")
        df = pd.DataFrame({
            "date": dates,
            "urea_usd_per_ton": [100.0] * 61 + [150.0],
            "dap_usd_per_ton": [200.0] * 62,
            "potassium_chloride_usd_per_ton": [300.0] * 62,
        })

        result = build_fertilizer_price_index(df)
        shock, ratio = fertilizer_price_shock_factor(result)

        self.assertGreater(result["fertilizer_price_index"].iloc[-1], 100)
        self.assertAlmostEqual(ratio, 1.25)
        self.assertGreater(shock, 0)

    def test_loads_indian_fertilizer_prices_from_csv(self):
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("fertilizer,bag_size_kg,price_inr_per_bag\nUrea,45,266.5\n")
            path = handle.name

        result, source = load_india_fertilizer_prices(path)

        self.assertEqual(source, path)
        self.assertAlmostEqual(result.iloc[0]["price_inr_per_kg"], 266.5 / 45)


if __name__ == "__main__":
    unittest.main()
