from __future__ import annotations

def clamp_0_100(x: float) -> float:
    return float(max(0, min(100, x)))


def monsoon_stress_from_anomaly(anomaly_pct: float) -> float:
    """Map rainfall deficit to 0-100 stress. Positive anomaly = low stress."""
    if anomaly_pct >= -5:
        return 10
    if anomaly_pct <= -35:
        return 100
    return clamp_0_100(10 + ((-5 - anomaly_pct) / 30) * 90)


def enso_stress_from_oni(oni: float) -> float:
    """Map ONI to stress; positive values indicate El Nino conditions."""
    if oni <= 0.3:
        return 10
    if oni >= 1.8:
        return 100
    return clamp_0_100(10 + ((oni - 0.3) / 1.5) * 90)


def enso_label(oni: float) -> str:
    """Classify ONI without implying a deterministic regional impact."""
    if oni <= -1.5:
        return "strong La Nina"
    if oni <= -0.5:
        return "La Nina"
    if oni < 0.5:
        return "neutral"
    if oni < 1.5:
        return "El Nino"
    return "strong El Nino"


def fertilizer_stress(hormus_blockade_factor: float, import_dependency_factor: float = 0.65, price_shock_factor: float = 0.5) -> float:
    """Heuristic fertilizer stress.

    hormus_blockade_factor: legacy geopolitical scenario input; 0 means no explicit route shock.
    import_dependency_factor: 0-1, degree to which critical fertilizer supply is import-exposed.
    price_shock_factor: 0-1, current global benchmark price stress.
    """
    h = max(0, min(1, hormus_blockade_factor))
    i = max(0, min(1, import_dependency_factor))
    p = max(0, min(1, price_shock_factor))
    return clamp_0_100(100 * (0.20 * h + 0.45 * i + 0.35 * p))


def food_price_stress_from_change(change_pct: float) -> float:
    """Map a three-month staple-price change to a 0-100 stress score."""
    if change_pct <= 0:
        return 10
    if change_pct >= 30:
        return 100
    return clamp_0_100(10 + (change_pct / 30) * 90)


def vegetation_soil_stress(soil_moisture_anomaly_pct: float, rainfall_anomaly_pct: float) -> float:
    """Estimate crop-condition stress from soil moisture and rainfall anomalies."""
    soil_stress = monsoon_stress_from_anomaly(soil_moisture_anomaly_pct)
    rain_stress = monsoon_stress_from_anomaly(rainfall_anomaly_pct)
    return clamp_0_100(0.7 * soil_stress + 0.3 * rain_stress)


def risk_score(
    monsoon_stress: float,
    enso_stress: float,
    fert_stress: float,
    food_price_stress: float = 25,
    crop_condition_stress: float | None = None,
) -> float:
    if crop_condition_stress is None:
        return clamp_0_100(
            0.40 * monsoon_stress
            + 0.25 * enso_stress
            + 0.25 * fert_stress
            + 0.10 * food_price_stress
        )
    return clamp_0_100(
        0.30 * monsoon_stress
        + 0.20 * enso_stress
        + 0.20 * fert_stress
        + 0.15 * food_price_stress
        + 0.15 * crop_condition_stress
    )


def risk_label(score: float) -> str:
    if score <= 30:
        return "niedrig"
    if score <= 55:
        return "elevated"
    if score <= 75:
        return "hoch"
    return "kritisch"
