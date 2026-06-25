from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_sources import (
    IMD_RAINFALL_FILE,
    build_monsoon_anomaly,
    build_regional_agroclimate_snapshot,
    fetch_kerala_rainfall,
    fetch_noaa_enso_outlook,
    fetch_noaa_oni_series,
    latest_oni_value,
    load_imd_rainfall,
)
from src.risk_model import (
    enso_stress_from_oni,
    monsoon_stress_from_anomaly,
    risk_label,
    vegetation_soil_stress,
    wet_bulb_stress_from_temperature,
)

st.set_page_config(page_title="India Composite Food Stress Indicator", layout="wide")
st.title("India Composite Food Stress Indicator")
st.caption(
    "Exploratory indicator combining regional rainfall, soil moisture, ENSO, and wet-bulb heat. "
    "It is not a forecast or a probability of a food crisis."
)

today = pd.Timestamp.now().normalize()


def cap_end_date(end_value: str) -> str:
    requested_end = pd.to_datetime(end_value)
    return min(requested_end, today).strftime("%Y-%m-%d")


def stress_text(value: float) -> str:
    return f"{value:.0f}/100" if pd.notna(value) and np.isfinite(value) else "Unavailable"


def composite_weather_score(
    rainfall_anomaly_pct: float,
    soil_moisture_anomaly_pct: float,
    oni_value: float,
    wet_bulb_temperature_c: float,
) -> float:
    monsoon = monsoon_stress_from_anomaly(rainfall_anomaly_pct)
    enso = enso_stress_from_oni(oni_value)
    crop = vegetation_soil_stress(soil_moisture_anomaly_pct, rainfall_anomaly_pct)
    wet_bulb = wet_bulb_stress_from_temperature(wet_bulb_temperature_c)
    return (
        0.35 * monsoon
        + 0.25 * enso
        + 0.25 * crop
        + 0.15 * wet_bulb
    )


def simulate_six_month_weather_outlook(
    reference_date: pd.Timestamp,
    rainfall_anomaly_pct: float,
    soil_moisture_anomaly_pct: float,
    oni_value: float,
    wet_bulb_temperature_c: float,
    simulations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate favorable and adverse six-month paths for visible components only."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(
        pd.Timestamp(reference_date).to_period("M").to_timestamp() + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS",
    )
    scenarios = {
        "Worst case": {
            "rain_shift": -7.0,
            "soil_shift": -3.0,
            "oni_shift": 0.18,
            "wet_bulb_shift": 0.35,
            "quantile": 0.90,
        },
        "Best case": {
            "rain_shift": 7.0,
            "soil_shift": 3.0,
            "oni_shift": -0.12,
            "wet_bulb_shift": -0.25,
            "quantile": 0.10,
        },
    }
    records = []
    for scenario, params in scenarios.items():
        rain = np.full(simulations, float(rainfall_anomaly_pct))
        soil = np.full(simulations, float(soil_moisture_anomaly_pct))
        oni_path = np.full(simulations, float(oni_value))
        wet_bulb = np.full(simulations, float(wet_bulb_temperature_c))

        for horizon, month in enumerate(months, start=1):
            ramp = min(1.0, horizon / 4.0)
            rain = (
                0.82 * rain
                + ramp * params["rain_shift"]
                + rng.normal(0, 8.0 + horizon, simulations)
            )
            rain = np.clip(rain, -80, 80)
            soil = (
                0.72 * soil
                + 0.28 * rain
                + ramp * params["soil_shift"]
                + rng.normal(0, 5.0 + 0.5 * horizon, simulations)
            )
            soil = np.clip(soil, -80, 80)
            oni_path = np.clip(
                0.90 * oni_path + ramp * params["oni_shift"] + rng.normal(0, 0.16, simulations),
                -3,
                3,
            )
            wet_bulb = np.clip(
                0.88 * wet_bulb
                + 0.12 * float(wet_bulb_temperature_c)
                + ramp * params["wet_bulb_shift"]
                + rng.normal(0, 0.45 + 0.08 * horizon, simulations),
                15,
                38,
            )

            scores = np.array(
                [
                    composite_weather_score(r, s, o, w)
                    for r, s, o, w in zip(rain, soil, oni_path, wet_bulb)
                ]
            )
            q = params["quantile"]
            records.append(
                {
                    "Month": month,
                    "Scenario": scenario,
                    "Stress score": float(np.quantile(scores, q)),
                    "P10": float(np.quantile(scores, 0.10)),
                    "Median": float(np.quantile(scores, 0.50)),
                    "P90": float(np.quantile(scores, 0.90)),
                    "Low %": float(100 * np.mean(scores <= 30)),
                    "Elevated %": float(100 * np.mean((scores > 30) & (scores <= 55))),
                    "High %": float(100 * np.mean((scores > 55) & (scores <= 75))),
                    "Critical %": float(100 * np.mean(scores > 75)),
                    "Rainfall anomaly": float(np.quantile(rain, q)),
                    "Soil moisture anomaly": float(np.quantile(soil, q)),
                    "ONI": float(np.quantile(oni_path, q)),
                    "Wet-bulb": float(np.quantile(wet_bulb, q)),
                    "Simulation runs": simulations,
                }
            )
    return pd.DataFrame(records)


def simulate_enso_probability_impact(
    reference_date: pd.Timestamp,
    rainfall_anomaly_pct: float,
    soil_moisture_anomaly_pct: float,
    oni_value: float,
    wet_bulb_temperature_c: float,
    strengthening_probability_pct: float,
    weekly_nino34_c: float | None = None,
    simulations: int = 2000,
    seed: int = 84,
) -> pd.DataFrame:
    """Estimate impact if NOAA's strengthening ENSO branch materializes."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(
        pd.Timestamp(reference_date).to_period("M").to_timestamp() + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS",
    )
    probability = float(np.clip(strengthening_probability_pct, 0, 100) / 100)
    branch_is_strengthening = rng.random(simulations) < probability

    start_oni = float(oni_value)
    if weekly_nino34_c is not None and np.isfinite(weekly_nino34_c):
        start_oni = max(start_oni, float(weekly_nino34_c))

    states = {
        "Probability weighted": {
            "rain": np.full(simulations, float(rainfall_anomaly_pct)),
            "soil": np.full(simulations, float(soil_moisture_anomaly_pct)),
            "oni": np.full(simulations, start_oni),
            "wet_bulb": np.full(simulations, float(wet_bulb_temperature_c)),
            "strengthening_mask": branch_is_strengthening,
        },
        "If strengthening occurs": {
            "rain": np.full(simulations, float(rainfall_anomaly_pct)),
            "soil": np.full(simulations, float(soil_moisture_anomaly_pct)),
            "oni": np.full(simulations, start_oni),
            "wet_bulb": np.full(simulations, float(wet_bulb_temperature_c)),
            "strengthening_mask": np.full(simulations, True),
        },
        "If strengthening does not occur": {
            "rain": np.full(simulations, float(rainfall_anomaly_pct)),
            "soil": np.full(simulations, float(soil_moisture_anomaly_pct)),
            "oni": np.full(simulations, float(oni_value)),
            "wet_bulb": np.full(simulations, float(wet_bulb_temperature_c)),
            "strengthening_mask": np.full(simulations, False),
        },
    }

    records = []
    for horizon, month in enumerate(months, start=1):
        ramp = min(1.0, horizon / 4.0)
        for scenario, state in states.items():
            strengthening = state["strengthening_mask"]
            enso_rain_tilt = ramp * (
                np.where(strengthening, -3.8, -0.6) * np.maximum(state["oni"], 0)
            )
            rain_shift = ramp * np.where(strengthening, -3.0, 1.5)
            soil_shift = ramp * np.where(strengthening, -1.5, 0.8)
            oni_shift = ramp * np.where(strengthening, 0.16, -0.04)
            wet_bulb_shift = ramp * np.where(strengthening, 0.18, -0.05)

            state["rain"] = np.clip(
                0.80 * state["rain"]
                + enso_rain_tilt
                + rain_shift
                + rng.normal(0, 7.5, simulations),
                -80,
                80,
            )
            state["soil"] = np.clip(
                0.74 * state["soil"]
                + 0.26 * state["rain"]
                + soil_shift
                + rng.normal(0, 4.8, simulations),
                -80,
                80,
            )
            state["oni"] = np.clip(
                0.91 * state["oni"] + oni_shift + rng.normal(0, 0.13, simulations),
                -3,
                3,
            )
            state["wet_bulb"] = np.clip(
                0.88 * state["wet_bulb"]
                + 0.12 * float(wet_bulb_temperature_c)
                + wet_bulb_shift
                + rng.normal(0, 0.40, simulations),
                15,
                38,
            )
            scores = np.array(
                [
                    composite_weather_score(r, s, o, w)
                    for r, s, o, w in zip(
                        state["rain"], state["soil"], state["oni"], state["wet_bulb"]
                    )
                ]
            )
            records.append(
                {
                    "Month": month,
                    "Scenario": scenario,
                    "P10": float(np.quantile(scores, 0.10)),
                    "Median": float(np.quantile(scores, 0.50)),
                    "P90": float(np.quantile(scores, 0.90)),
                    "High-or-critical %": float(100 * np.mean(scores > 55)),
                    "ONI median": float(np.median(state["oni"])),
                    "Rainfall anomaly median": float(np.median(state["rain"])),
                    "Soil anomaly median": float(np.median(state["soil"])),
                    "Simulation runs": simulations,
                }
            )
    return pd.DataFrame(records)


with st.sidebar:
    st.header("Scenario")
    current_year = st.number_input("Analysis year", min_value=2001, max_value=2035, value=2026)
    start = st.text_input("Monsoon start", f"{current_year}-05-01")
    end = st.text_input("End date", f"{current_year}-09-30")
    baseline_start = st.text_input("Baseline start", "2001-05-01")
    baseline_end = st.text_input("Baseline end", "2025-09-30")

    manual_oni = st.toggle("Set ONI manually", value=False)
    oni_input = st.slider("ONI", -2.5, 2.5, 1.0, 0.1)


@st.cache_data(ttl=3600)
def load_rain(start_date: str, end_date: str, climate_start: str, climate_end: str):
    effective_end = cap_end_date(end_date)
    if effective_end != end_date:
        st.info(f"Rainfall data is capped at the latest observed date: {effective_end}.")

    if os.path.exists(IMD_RAINFALL_FILE):
        st.info("Using local IMD rainfall data from data/imd_kerala_daily.csv")
        current = load_imd_rainfall(start_date, effective_end)
        baseline = load_imd_rainfall(climate_start, climate_end)
        if current.empty or baseline.empty:
            st.warning(
                "Local IMD data does not cover the requested period; falling back to NASA POWER rainfall."
            )
            current = fetch_kerala_rainfall(start_date, effective_end)
            baseline = fetch_kerala_rainfall(climate_start, climate_end)
            source = "NASA POWER (local IMD coverage incomplete)"
        else:
            source = "Local IMD file"
    else:
        current = fetch_kerala_rainfall(start_date, effective_end)
        baseline = fetch_kerala_rainfall(climate_start, climate_end)
        source = "NASA POWER"
    return current, baseline, source


@st.cache_data(ttl=3600)
def load_oni():
    try:
        history = fetch_noaa_oni_series()
        if history.empty:
            raise ValueError("NOAA ONI history is empty.")
        return float(history["oni"].iloc[-1]), history, "NOAA CPC"
    except Exception:
        value = latest_oni_value(default=np.nan)
        if pd.isna(value):
            return 1.0, pd.DataFrame(), "configured fallback value"
        return value, pd.DataFrame(), "NOAA CPC"


@st.cache_data(ttl=3600)
def load_enso_outlook():
    return fetch_noaa_enso_outlook()


@st.cache_data(ttl=21600)
def load_regional_snapshot(current_start: str, current_end: str, climate_start: str, climate_end: str):
    return build_regional_agroclimate_snapshot(
        current_start,
        cap_end_date(current_end),
        climate_start,
        climate_end,
    )


def show_oni_history(history: pd.DataFrame) -> None:
    years = st.slider("ONI history shown (years)", 5, 50, 20, 5)
    cutoff = history["date"].max() - pd.DateOffset(years=years)
    plot_data = history.loc[history["date"] >= cutoff]
    fig = px.line(
        plot_data,
        x="date",
        y="oni",
        markers=True,
        hover_data={"season": True, "date": "|%Y-%m", "oni": ":.1f"},
        title="Oceanic Nino Index development",
        labels={"date": "Season ending", "oni": "ONI (deg C)", "season": "Season"},
    )
    fig.add_hrect(y0=0.5, y1=3, fillcolor="firebrick", opacity=0.08, line_width=0)
    fig.add_hrect(y0=-3, y1=-0.5, fillcolor="royalblue", opacity=0.08, line_width=0)
    fig.add_hline(y=0.5, line_dash="dot", line_color="firebrick")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="royalblue")
    fig.update_yaxes(range=[-3, 3], zeroline=True, zerolinecolor="gray")
    st.plotly_chart(fig, use_container_width=True)


try:
    current, baseline, rainfall_source = load_rain(start, end, baseline_start, baseline_end)
    if current.empty or baseline.empty:
        raise ValueError("No valid rainfall data available.")

    rainfall_df = build_monsoon_anomaly(current, baseline).sort_values("date").reset_index(drop=True)
    rainfall_is_simulated = False
    rainfall_df["data_type"] = "Observed"
    latest_obs = rainfall_df["date"].max()
    if latest_obs < today:
        st.warning(
            f"Latest observed rainfall is available through {latest_obs.date()}. "
            "NASA POWER may not have published newer daily rainfall values yet."
        )
except Exception:
    st.warning(
        "NASA POWER could not be loaded or valid rainfall data is missing. Using simulated rainfall fallback data."
    )
    st.error("Simulated rainfall fallback data must not be interpreted as observations.")
    dates = pd.date_range(start=start, end=cap_end_date(end), freq="D")
    rainfall_df = pd.DataFrame({"date": dates})
    rainfall_df["precipitation_mm"] = [1.5 + i * 0.05 for i in range(len(rainfall_df))]
    rainfall_df["baseline_mm"] = [1.75 + i * 0.04 for i in range(len(rainfall_df))]
    rainfall_df["actual_cum_mm"] = rainfall_df["precipitation_mm"].cumsum()
    rainfall_df["baseline_cum_mm"] = rainfall_df["baseline_mm"].cumsum()
    rainfall_df["rainfall_anomaly_pct"] = 100 * (
        rainfall_df["actual_cum_mm"] / rainfall_df["baseline_cum_mm"] - 1
    )
    rainfall_df["data_type"] = "Simulated"
    rainfall_source = "Simulated rainfall fallback"
    rainfall_is_simulated = True

if manual_oni:
    oni = oni_input
    oni_history = pd.DataFrame()
    oni_source = "manual scenario input"
else:
    oni, oni_history, oni_source = load_oni()
    st.sidebar.info(f"ONI: {oni:.2f} ({oni_source})")
    if oni_source != "NOAA CPC":
        st.sidebar.warning("NOAA ONI could not be loaded; the displayed ONI is a fallback assumption.")

try:
    enso_outlook = load_enso_outlook() if not manual_oni else {}
except Exception:
    enso_outlook = {}

rainfall_values = rainfall_df.get("rainfall_anomaly_pct", pd.Series(dtype=float)).dropna()
latest_anomaly = float(rainfall_values.iloc[-1]) if not rainfall_values.empty else 0.0

try:
    regional_df = load_regional_snapshot(start, end, baseline_start, baseline_end)
    if regional_df.empty:
        raise ValueError("No regional agroclimate rows were returned.")
    regional_rainfall_anomaly = float(regional_df["rainfall_anomaly_pct"].median())
    regional_soil_anomaly = float(regional_df["soil_moisture_anomaly_pct"].median())
    regional_wet_bulb_temperature = float(regional_df["wet_bulb_max_c"].median())
    regional_source = "NASA POWER regional monitoring points"
except Exception:
    regional_df = pd.DataFrame()
    regional_rainfall_anomaly = latest_anomaly
    regional_soil_anomaly = 0.0
    regional_wet_bulb_temperature = 24.0
    regional_source = "Unavailable; Kerala rainfall fallback used"

monsoon_stress = monsoon_stress_from_anomaly(regional_rainfall_anomaly)
enso_stress = enso_stress_from_oni(oni)
crop_condition_stress = vegetation_soil_stress(regional_soil_anomaly, regional_rainfall_anomaly)
wet_bulb_stress = wet_bulb_stress_from_temperature(regional_wet_bulb_temperature)

component_weights = {
    "Monsoon": 0.35,
    "ENSO": 0.25,
    "Crop Condition": 0.25,
    "Wet-bulb Heat": 0.15,
}
component_scores = {
    "Monsoon": monsoon_stress,
    "ENSO": enso_stress,
    "Crop Condition": crop_condition_stress,
    "Wet-bulb Heat": wet_bulb_stress,
}
score = sum(component_weights[name] * value for name, value in component_scores.items())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Composite Stress Indicator", stress_text(score), risk_label(score), delta_color="off")
c2.metric(
    "Monsoon Stress",
    f"{monsoon_stress:.0f}/100",
    f"{regional_rainfall_anomaly:.1f}% regional median",
    delta_color="off",
)
c3.metric("ENSO Stress", f"{enso_stress:.0f}/100", f"ONI {oni:.1f}", delta_color="off")
c4.metric(
    "Crop-condition Stress",
    f"{crop_condition_stress:.0f}/100",
    f"Soil {regional_soil_anomaly:.1f}%",
    delta_color="off",
)
c5.metric(
    "Wet-bulb Stress",
    f"{wet_bulb_stress:.0f}/100",
    f"{regional_wet_bulb_temperature:.1f} deg C regional median max",
    delta_color="off",
)

st.caption(
    f"Rainfall source: {rainfall_source}. ONI source: {oni_source}. Regional source: {regional_source}."
)

rainfall_phrase = "above baseline" if regional_rainfall_anomaly >= 0 else "below baseline"
soil_phrase = "wetter than baseline" if regional_soil_anomaly >= 0 else "drier than baseline"
summary_parts = [
    f"Composite stress is **{score:.0f}/100** ({risk_label(score)}).",
    f"Regional rainfall is **{abs(regional_rainfall_anomaly):.1f}% {rainfall_phrase}**.",
    f"Root-zone soil moisture is **{abs(regional_soil_anomaly):.1f}% {soil_phrase}**.",
    f"ONI is **{oni:.1f}** from {oni_source}.",
    f"Regional median maximum wet-bulb temperature is **{regional_wet_bulb_temperature:.1f} deg C**.",
]
if enso_outlook:
    weekly_nino34 = enso_outlook.get("weekly_nino34_c")
    if weekly_nino34 is not None:
        summary_parts.append(f"NOAA weekly Nino-3.4 is **{weekly_nino34:+.1f} deg C**.")
    probability = enso_outlook.get("very_strong_probability_pct")
    period = enso_outlook.get("very_strong_period") or "forecast period"
    if probability is not None:
        summary_parts.append(f"NOAA very-strong probability is **{probability}%** for {period}.")
if rainfall_is_simulated:
    summary_parts.append("Rainfall currently uses simulated fallback data.")

st.subheader("Current values summary")
st.info(" ".join(summary_parts))

st.subheader("Regional rainfall and crop-condition monitoring")
if regional_df.empty:
    st.warning("Regional NASA POWER data could not be loaded; regional indicators use the Kerala fallback.")
else:
    map_df = regional_df.copy()
    map_df["crop_condition_stress"] = map_df.apply(
        lambda row: vegetation_soil_stress(
            row["soil_moisture_anomaly_pct"], row["rainfall_anomaly_pct"]
        ),
        axis=1,
    )
    regional_map = px.scatter_geo(
        map_df,
        lat="latitude",
        lon="longitude",
        color="rainfall_anomaly_pct",
        size="crop_condition_stress",
        hover_name="region",
        hover_data={
            "latitude": False,
            "longitude": False,
            "rainfall_anomaly_pct": ":.1f",
            "soil_moisture_anomaly_pct": ":.1f",
            "wet_bulb_mean_c": ":.1f",
            "wet_bulb_max_c": ":.1f",
            "wet_bulb_anomaly_c": ":+.1f",
            "crop_condition_stress": ":.0f",
        },
        color_continuous_scale="RdYlBu",
        color_continuous_midpoint=0,
        title="Major agricultural regions: rainfall anomaly and crop-condition stress",
    )
    regional_map.update_geos(
        center={"lat": 22.5, "lon": 79.0},
        projection_scale=3.7,
        showcountries=True,
        countrycolor="gray",
    )
    regional_map.update_layout(height=520, margin={"l": 0, "r": 0, "t": 50, "b": 0})
    st.plotly_chart(regional_map, use_container_width=True)
    st.dataframe(
        map_df[
            [
                "region",
                "rainfall_anomaly_pct",
                "soil_moisture_anomaly_pct",
                "wet_bulb_mean_c",
                "wet_bulb_max_c",
                "wet_bulb_anomaly_c",
                "wet_bulb_days_ge_28c",
                "crop_condition_stress",
                "latest_date",
            ]
        ].sort_values("crop_condition_stress", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

crop_c1, crop_c2, crop_c3 = st.columns(3)
crop_c1.metric("Crop-condition stress", f"{crop_condition_stress:.0f}/100")
crop_c2.metric("Regional rainfall anomaly", f"{regional_rainfall_anomaly:.1f}%")
crop_c3.metric("Root-zone soil moisture anomaly", f"{regional_soil_anomaly:.1f}%")
st.caption(
    "Crop-condition stress is a proxy based on NASA POWER root-zone wetness and rainfall. "
    "It is not a satellite NDVI measurement."
)

st.subheader("Wet-bulb temperature monitoring")
if regional_df.empty:
    st.warning("Regional wet-bulb temperature is unavailable because the NASA POWER snapshot could not be loaded.")
else:
    wet_bulb_mean = float(regional_df["wet_bulb_mean_c"].median())
    wet_bulb_max = float(regional_df["wet_bulb_max_c"].max())
    wet_bulb_anomaly = float(regional_df["wet_bulb_anomaly_c"].median())
    wb1, wb2, wb3, wb4 = st.columns(4)
    wb1.metric("Regional median wet-bulb", f"{wet_bulb_mean:.1f} deg C")
    wb2.metric("Highest regional daily value", f"{wet_bulb_max:.1f} deg C")
    wb3.metric("Median anomaly vs baseline", f"{wet_bulb_anomaly:+.1f} deg C")
    wb4.metric("Composite component stress", f"{wet_bulb_stress:.0f}/100")

    wet_bulb_chart_df = regional_df.sort_values("wet_bulb_max_c", ascending=False)
    st.plotly_chart(
        px.bar(
            wet_bulb_chart_df,
            x="region",
            y="wet_bulb_max_c",
            color="wet_bulb_anomaly_c",
            color_continuous_scale="RdYlBu_r",
            labels={
                "region": "Region",
                "wet_bulb_max_c": "Maximum wet-bulb temperature (deg C)",
                "wet_bulb_anomaly_c": "Anomaly (deg C)",
            },
            title="Maximum observed wet-bulb temperature by agricultural region",
        ),
        use_container_width=True,
    )
st.caption(
    "Wet-bulb temperature is estimated from NASA POWER 2 m air temperature and relative humidity "
    "using the Stull approximation. It is not WBGT or a weather-station measurement."
)

with st.expander("What is ENSO and where does the ONI come from?", expanded=False):
    st.markdown(
        "ENSO (El Nino-Southern Oscillation) describes coupled variations in oceanic and atmospheric "
        "conditions in the tropical Pacific. Its relationship with India's monsoon is statistical, "
        "not deterministic, and varies by season and region."
    )
    st.markdown(
        "The ONI is NOAA's three-month running mean sea-surface-temperature anomaly in the "
        "Nino 3.4 region. Values beyond +/-0.5 deg C indicate El Nino- or La Nina-like ocean "
        "conditions; an official episode also requires persistence and atmospheric coupling. "
        "The data come from NOAA CPC: "
        "[NOAA CPC ONI](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php)."
    )
    if manual_oni:
        st.info("The composite indicator currently uses the manual ONI scenario. Switch it off to load NOAA history.")
    elif oni_history.empty:
        st.warning("The NOAA ONI history could not be loaded.")
    else:
        show_oni_history(oni_history)

st.subheader("NOAA forward ENSO outlook")
if enso_outlook:
    outlook_c1, outlook_c2, outlook_c3 = st.columns(3)
    outlook_c1.metric("Alert status", enso_outlook.get("alert_status") or "Not stated")
    weekly_nino34 = enso_outlook.get("weekly_nino34_c")
    outlook_c2.metric(
        "Latest weekly Nino-3.4",
        f"{weekly_nino34:+.1f} deg C" if weekly_nino34 is not None else "Not stated",
        delta_color="off",
    )
    very_strong_probability = enso_outlook.get("very_strong_probability_pct")
    very_strong_period = enso_outlook.get("very_strong_period") or "forecast period"
    outlook_c3.metric(
        f"Very strong chance ({very_strong_period})",
        f"{very_strong_probability}%" if very_strong_probability is not None else "Not stated",
        delta_color="off",
    )
    st.info(enso_outlook.get("synopsis") or "NOAA forward outlook loaded.")
    st.caption(
        f"Issued {enso_outlook.get('issued') or 'date not parsed'}; refreshed at most hourly. "
        f"[Source: NOAA CPC]({enso_outlook.get('source_url')})"
    )
else:
    st.warning("The live NOAA forward ENSO outlook could not be loaded.")

st.subheader("ENSO probability impact")
enso_probability = (
    float(enso_outlook.get("very_strong_probability_pct"))
    if enso_outlook and enso_outlook.get("very_strong_probability_pct") is not None
    else np.nan
)
weekly_nino34 = enso_outlook.get("weekly_nino34_c") if enso_outlook else None
if pd.notna(enso_probability):
    impact_df = simulate_enso_probability_impact(
        today,
        regional_rainfall_anomaly,
        regional_soil_anomaly,
        oni,
        regional_wet_bulb_temperature,
        enso_probability,
        weekly_nino34_c=weekly_nino34,
    )
    final_month = impact_df["Month"].max()
    final_rows = impact_df[impact_df["Month"] == final_month].set_index("Scenario")
    strengthening_impact = (
        final_rows.loc["If strengthening occurs", "Median"]
        - final_rows.loc["If strengthening does not occur", "Median"]
    )
    impact_c1, impact_c2, impact_c3 = st.columns(3)
    impact_c1.metric("NOAA probability used", f"{enso_probability:.0f}%")
    impact_c2.metric("Six-month impact if it occurs", f"{strengthening_impact:+.1f} points")
    impact_c3.metric(
        "Weighted high/critical chance",
        f"{final_rows.loc['Probability weighted', 'High-or-critical %']:.0f}%",
    )

    impact_chart = go.Figure()
    impact_colors = {
        "Probability weighted": "#d65f0e",
        "If strengthening occurs": "#b2182b",
        "If strengthening does not occur": "#1a9850",
    }
    for scenario, color in impact_colors.items():
        scenario_df = impact_df[impact_df["Scenario"] == scenario]
        impact_chart.add_trace(
            go.Scatter(
                x=scenario_df["Month"],
                y=scenario_df["Median"],
                mode="lines+markers",
                line={"color": color, "width": 3},
                marker={"size": 8},
                name=scenario,
                hovertemplate="%{x|%b %Y}: %{y:.1f}/100<extra></extra>",
            )
        )
    impact_chart.update_layout(
        title="Conditional ENSO-strengthening impact on composite stress",
        yaxis={"title": "Median composite stress score", "range": [0, 100]},
        xaxis={"title": None},
        hovermode="x unified",
    )
    st.plotly_chart(impact_chart, use_container_width=True)

    impact_display = impact_df.copy()
    impact_display["Month"] = impact_display["Month"].dt.strftime("%b %Y")
    for column in [
        "P10",
        "Median",
        "P90",
        "High-or-critical %",
        "ONI median",
        "Rainfall anomaly median",
        "Soil anomaly median",
    ]:
        impact_display[column] = impact_display[column].round(1)
    st.dataframe(impact_display, use_container_width=True, hide_index=True)
    st.caption(
        "This uses NOAA's stated probability as a branch weight. The strengthening branch applies "
        "additional positive ONI drift and dry-side rainfall/soil pressure; the non-strengthening "
        "branch relaxes those assumptions. It recalculates whenever the live NOAA and NASA inputs refresh."
    )
else:
    st.warning(
        "NOAA did not provide a parsed strengthening probability, so the probability-weighted ENSO impact "
        "simulation is unavailable."
    )

st.subheader("Six-month scenario outlook")
outlook_df = simulate_six_month_weather_outlook(
    today,
    regional_rainfall_anomaly,
    regional_soil_anomaly,
    oni,
    regional_wet_bulb_temperature,
)
sim_c1, sim_c2 = st.columns(2)
sim_c1.metric("Simulation runs", "2,000 per scenario")
sim_c2.metric("Horizon", "6 months")

scenario_colors = {"Worst case": "#b2182b", "Best case": "#1a9850"}
band_colors = {
    "Worst case": "rgba(178, 24, 43, 0.16)",
    "Best case": "rgba(26, 152, 80, 0.16)",
}
current_rows = pd.DataFrame(
    [
        {
            "Month": today.to_period("M").to_timestamp(),
            "Scenario": scenario,
            "Stress score": score,
            "P10": score,
            "Median": score,
            "P90": score,
        }
        for scenario in ["Worst case", "Best case"]
    ]
)
chart_outlook_df = pd.concat([current_rows, outlook_df], ignore_index=True)
outlook_chart = go.Figure()
for scenario in ["Worst case", "Best case"]:
    scenario_df = chart_outlook_df[chart_outlook_df["Scenario"] == scenario]
    outlook_chart.add_trace(
        go.Scatter(
            x=scenario_df["Month"],
            y=scenario_df["P90"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    outlook_chart.add_trace(
        go.Scatter(
            x=scenario_df["Month"],
            y=scenario_df["P10"],
            mode="lines",
            fill="tonexty",
            fillcolor=band_colors[scenario],
            line={"width": 0},
            name=f"{scenario} P10-P90 band",
            hovertemplate="%{x|%b %Y}: P10 %{y:.1f}<extra></extra>",
        )
    )
    outlook_chart.add_trace(
        go.Scatter(
            x=scenario_df["Month"],
            y=scenario_df["Stress score"],
            mode="lines+markers",
            line={"color": scenario_colors[scenario], "width": 3},
            marker={"size": 8},
            name=scenario,
            hovertemplate="%{x|%b %Y}: %{y:.1f}/100<extra></extra>",
        )
    )
outlook_chart.update_layout(
    title="Best-case and worst-case six-month stress scenarios",
    yaxis={"title": "Composite stress score", "range": [0, 100]},
    xaxis={"title": None},
    hovermode="x unified",
)
st.plotly_chart(outlook_chart, use_container_width=True)
display_outlook = outlook_df.copy()
display_outlook["Month"] = display_outlook["Month"].dt.strftime("%b %Y")
for column in [
    "Stress score",
    "P10",
    "Median",
    "P90",
    "Low %",
    "Elevated %",
    "High %",
    "Critical %",
    "Rainfall anomaly",
    "Soil moisture anomaly",
    "ONI",
    "Wet-bulb",
]:
    display_outlook[column] = display_outlook[column].round(1)
st.dataframe(
    display_outlook,
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "Both scenario lines start from the current composite score and then fan out over the six-month horizon. "
    "The worst-case line shows the adverse scenario's 90th percentile stress path. "
    "The best-case line shows the favorable scenario's 10th percentile stress path. "
    "Each scenario runs 2,000 Monte Carlo paths and displays the P10-P90 spread. "
    "Scenario shocks ramp in gradually from month one to month four, so the first month remains closer "
    "to current observed conditions. "
    "The paths simulate only the visible weather, ENSO, crop-condition, and wet-bulb components; "
    "they are scenario stress tests, not an official forecast or crisis probability."
)

components = pd.DataFrame(
    {
        "Component": list(component_scores.keys()),
        "Stress": list(component_scores.values()),
        "Weight": [component_weights[name] for name in component_scores],
    }
)
st.plotly_chart(
    px.bar(
        components,
        x="Component",
        y="Stress",
        text="Stress",
        title="Indicator components",
        color_discrete_sequence=["#c0392b"],
    ),
    use_container_width=True,
)

if rainfall_is_simulated:
    st.warning("At least one input series is simulated; this assessment is illustrative only.")

st.info(
    "Interpretation: This is a heuristic monitoring indicator. For operational use, augment with "
    "official food-security, stock, market-price, crop-yield, and household data."
)
