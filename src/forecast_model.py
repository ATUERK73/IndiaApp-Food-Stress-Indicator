from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_model import (
    enso_stress_from_oni,
    fertilizer_stress,
    food_price_stress_from_change,
    monsoon_stress_from_anomaly,
    risk_score,
    vegetation_soil_stress,
    wet_bulb_stress_from_temperature,
)


def _recent_log_return_parameters(
    history: pd.DataFrame | None,
    value_column: str,
    default_volatility: float,
) -> tuple[float, float]:
    """Estimate a conservative monthly drift and volatility from recent data."""
    if history is None or value_column not in history or len(history) < 4:
        return 0.0, default_volatility
    values = pd.to_numeric(history[value_column], errors="coerce").dropna().tail(13)
    values = values[values > 0]
    if len(values) < 4:
        return 0.0, default_volatility
    returns = np.log(values).diff().dropna()
    drift = float(np.clip(returns.median(), -0.04, 0.04))
    volatility = float(np.clip(returns.std(ddof=1), 0.02, 0.20))
    return drift, volatility


def _oni_parameters(history: pd.DataFrame | None) -> tuple[float, float]:
    if history is None or "oni" not in history or len(history) < 6:
        return 0.0, 0.16
    values = pd.to_numeric(history["oni"], errors="coerce").dropna().tail(24)
    changes = values.diff().dropna()
    if changes.empty:
        return 0.0, 0.16
    drift = float(np.clip(changes.median(), -0.15, 0.15))
    volatility = float(np.clip(changes.std(ddof=1), 0.08, 0.35))
    return drift, volatility


def simulate_three_month_forecast(
    reference_date: pd.Timestamp,
    rainfall_anomaly_pct: float,
    soil_moisture_anomaly_pct: float,
    oni: float,
    import_exposure: float,
    fertilizer_price_ratio: float,
    food_price_stress: float,
    wet_bulb_temperature_c: float = 24.0,
    enso_weekly_nino34_c: float | None = None,
    enso_expected_to_strengthen: bool = False,
    oni_history: pd.DataFrame | None = None,
    fertilizer_history: pd.DataFrame | None = None,
    food_price_history: pd.DataFrame | None = None,
    scenario: str = "baseline",
    simulations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate three monthly composite-stress outcomes.

    The model is deliberately small and transparent: anomalies mean-revert, ONI
    follows its recent monthly change, and price indices follow recent log returns.
    The returned intervals describe model uncertainty, not crisis probabilities.
    """
    scenario_parameters = {
        "baseline": {
            "rain_persistence": 0.78,
            "rain_shift": 0.0,
            "soil_persistence": 0.75,
            "food_shift": 0.0,
            "fertilizer_shift": 0.0,
            "wet_bulb_shift": 0.0,
        },
        "dry": {
            "rain_persistence": 0.92,
            "rain_shift": -6.0,
            "soil_persistence": 0.85,
            "food_shift": 0.012,
            "fertilizer_shift": 0.008,
            "wet_bulb_shift": 0.5,
        },
        "favorable": {
            "rain_persistence": 0.58,
            "rain_shift": 7.0,
            "soil_persistence": 0.62,
            "food_shift": -0.008,
            "fertilizer_shift": -0.005,
            "wet_bulb_shift": -0.4,
        },
    }
    if scenario not in scenario_parameters:
        raise ValueError(f"Unknown forecast scenario: {scenario}")
    params = scenario_parameters[scenario]
    if simulations < 100:
        raise ValueError("At least 100 simulations are required.")

    rng = np.random.default_rng(seed)
    months = pd.date_range(
        pd.Timestamp(reference_date).to_period("M").to_timestamp() + pd.offsets.MonthBegin(1),
        periods=3,
        freq="MS",
    )
    oni_drift, oni_volatility = _oni_parameters(oni_history)
    if enso_expected_to_strengthen:
        # Explicit model translation of NOAA's qualitative forward signal;
        # this is not an official NOAA monthly ONI forecast.
        oni_drift = max(oni_drift, 0.15)
    fert_drift, fert_volatility = _recent_log_return_parameters(
        fertilizer_history, "fertilizer_price_index", 0.06
    )
    food_drift, food_volatility = _recent_log_return_parameters(
        food_price_history, "food_price_index", 0.05
    )

    rain = np.full(simulations, float(rainfall_anomaly_pct))
    soil = np.full(simulations, float(soil_moisture_anomaly_pct))
    forecast_oni_start = float(oni)
    if enso_weekly_nino34_c is not None and np.isfinite(enso_weekly_nino34_c):
        forecast_oni_start = max(forecast_oni_start, float(enso_weekly_nino34_c))
    oni_paths = np.full(simulations, forecast_oni_start)
    fert_ratio = np.full(simulations, max(0.05, float(fertilizer_price_ratio)))
    wet_bulb = np.full(simulations, float(wet_bulb_temperature_c))

    food_values: list[np.ndarray] = []
    if food_price_history is not None and "food_price_index" in food_price_history:
        recent_food = pd.to_numeric(
            food_price_history["food_price_index"], errors="coerce"
        ).dropna().tail(3)
    else:
        recent_food = pd.Series(dtype=float)
    if len(recent_food) == 3 and (recent_food > 0).all():
        food_values = [np.full(simulations, float(value)) for value in recent_food]
    else:
        # Reconstruct a neutral index path whose latest three-month change matches
        # the current stress approximately.
        implied_change = max(0.0, (float(food_price_stress) - 10.0) / 3.0)
        food_values = [
            np.full(simulations, 100.0 / (1.0 + implied_change / 100.0)),
            np.full(simulations, 100.0),
            np.full(simulations, 100.0),
        ]

    records: list[dict] = []
    for horizon, month in enumerate(months, start=1):
        # Positive ONI adds a modest dry-side tilt; weather uncertainty expands
        # with the horizon and dominates the deterministic tilt.
        enso_rain_tilt = -3.0 * np.maximum(oni_paths, 0) + 1.5 * np.maximum(-oni_paths, 0)
        rain = (
            params["rain_persistence"] * rain
            + enso_rain_tilt
            + params["rain_shift"]
            + rng.normal(0, 8.0 + 3.0 * horizon, simulations)
        )
        rain = np.clip(rain, -80, 80)
        soil_persistence = params["soil_persistence"]
        soil = (
            soil_persistence * soil
            + (1.0 - soil_persistence) * rain
            + rng.normal(0, 5.0 + horizon, simulations)
        )
        soil = np.clip(soil, -80, 80)
        wet_bulb = (
            0.85 * wet_bulb
            + 0.15 * float(wet_bulb_temperature_c)
            + params["wet_bulb_shift"]
            + rng.normal(0, 0.7 + 0.2 * horizon, simulations)
        )
        wet_bulb = np.clip(wet_bulb, 15, 38)

        oni_paths = 0.92 * oni_paths + oni_drift + rng.normal(
            0, oni_volatility, simulations
        )
        oni_paths = np.clip(oni_paths, -3, 3)

        fert_ratio *= np.exp(
            fert_drift
            + params["fertilizer_shift"]
            + rng.normal(0, fert_volatility, simulations)
        )
        fert_ratio = np.clip(fert_ratio, 0.25, 4.0)
        fert_shock = np.clip((fert_ratio - 1.0) / 0.75, 0, 1)

        next_food = food_values[-1] * np.exp(
            food_drift
            + params["food_shift"]
            + rng.normal(0, food_volatility, simulations)
        )
        food_values.append(next_food)
        food_change = 100 * (food_values[-1] / food_values[-4] - 1)

        monsoon = np.array([monsoon_stress_from_anomaly(value) for value in rain])
        enso = np.array([enso_stress_from_oni(value) for value in oni_paths])
        fert = np.array(
            [fertilizer_stress(0.0, import_exposure, value) for value in fert_shock]
        )
        food = np.array([food_price_stress_from_change(value) for value in food_change])
        crop = np.array(
            [vegetation_soil_stress(s, r) for s, r in zip(soil, rain)]
        )
        wet_bulb_stress = np.array(
            [wet_bulb_stress_from_temperature(value) for value in wet_bulb]
        )
        scores = np.array(
            [
                risk_score(m, e, f, p, c, w)
                for m, e, f, p, c, w in zip(
                    monsoon, enso, fert, food, crop, wet_bulb_stress
                )
            ]
        )

        records.append(
            {
                "date": month,
                "scenario": scenario,
                "p10": float(np.quantile(scores, 0.10)),
                "median": float(np.quantile(scores, 0.50)),
                "p90": float(np.quantile(scores, 0.90)),
                "prob_low": float(np.mean(scores <= 30)),
                "prob_elevated": float(np.mean((scores > 30) & (scores <= 55))),
                "prob_high": float(np.mean((scores > 55) & (scores <= 75))),
                "prob_critical": float(np.mean(scores > 75)),
                "monsoon_median": float(np.median(monsoon)),
                "enso_median": float(np.median(enso)),
                "oni_value_median": float(np.median(oni_paths)),
                "fertilizer_median": float(np.median(fert)),
                "food_price_median": float(np.median(food)),
                "crop_condition_median": float(np.median(crop)),
                "wet_bulb_median": float(np.median(wet_bulb_stress)),
            }
        )
    return pd.DataFrame(records)
