from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.forecast_model import simulate_three_month_forecast

from src.data_sources import (
    build_regional_agroclimate_snapshot,
    fetch_staple_food_price_index,
    fetch_kerala_rainfall,
    fetch_indexmundi_urea_prices,
    fetch_world_bank_fertilizer_prices,
    fertilizer_price_shock_factor,
    load_india_fertilizer_prices,
    build_monsoon_anomaly,
    fetch_noaa_oni_series,
    latest_oni_value,
    load_imd_rainfall,
    IMD_RAINFALL_FILE,
)
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

st.set_page_config(page_title="India Composite Food Stress Indicator", layout="wide")
st.title("India Composite Food Stress Indicator")
st.caption(
    "Exploratory indicator combining regional weather, soil moisture, ENSO, food and fertilizer prices. "
    "It is not a forecast or a probability of a food crisis."
)
today = pd.Timestamp.now().normalize()


def cap_end_date(end_value: str) -> str:
    requested_end = pd.to_datetime(end_value)
    return min(requested_end, today).strftime("%Y-%m-%d")

with st.sidebar:
    st.header("Scenario")
    current_year = st.number_input("Analysis year", min_value=2001, max_value=2035, value=2026)
    start = st.text_input("Monsoon start", f"{current_year}-05-01")
    end = st.text_input("End date", f"{current_year}-09-30")
    baseline_start = st.text_input("Baseline start", "2001-05-01")
    baseline_end = st.text_input("Baseline end", "2025-09-30")

    import_dep = st.slider("Fertilizer import exposure", 0.0, 1.0, 0.65, 0.05)
    manual_food_price = st.toggle("Set food-price stress manually", value=False)
    food_price_input = st.slider(
        "Food price stress", 0, 100, 25, 5, disabled=not manual_food_price
    )

    manual_oni = st.toggle("Set ONI manually", value=False)
    oni_input = st.slider("ONI", -2.5, 2.5, 1.0, 0.1)

@st.cache_data(ttl=3600)
def load_rain(start, end, baseline_start, baseline_end):
    effective_end = cap_end_date(end)
    if effective_end != end:
        st.info(f"Rainfall data is capped at the latest observed date: {effective_end}.")

    if os.path.exists(IMD_RAINFALL_FILE):
        st.info("Using local IMD rainfall data from data/imd_kerala_daily.csv")
        current = load_imd_rainfall(start, effective_end)
        baseline = load_imd_rainfall(baseline_start, baseline_end)
        if current.empty or baseline.empty:
            st.warning(
                "Local IMD data does not cover the requested period; falling back to NASA POWER rainfall."
            )
            current = fetch_kerala_rainfall(start, effective_end)
            baseline = fetch_kerala_rainfall(baseline_start, baseline_end)
            source = "NASA POWER (local IMD coverage incomplete)"
        else:
            source = "Local IMD file"
    else:
        current = fetch_kerala_rainfall(start, effective_end)
        baseline = fetch_kerala_rainfall(baseline_start, baseline_end)
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


def show_oni_history(history: pd.DataFrame) -> None:
    years = st.slider("ONI history shown (years)", 5, 50, 20, 5)
    cutoff = history["date"].max() - pd.DateOffset(years=years)
    plot_data = history.loc[history["date"] >= cutoff]
    fig = px.line(
        plot_data, x="date", y="oni", markers=True,
        hover_data={"season": True, "date": "|%Y-%m", "oni": ":.1f"},
        title="Oceanic Niño Index development",
        labels={"date": "Season ending", "oni": "ONI (°C)", "season": "Season"},
    )
    fig.add_hrect(y0=0.5, y1=3, fillcolor="firebrick", opacity=0.08, line_width=0)
    fig.add_hrect(y0=-3, y1=-0.5, fillcolor="royalblue", opacity=0.08, line_width=0)
    fig.add_hline(y=0.5, line_dash="dot", line_color="firebrick")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="royalblue")
    fig.update_yaxes(range=[-3, 3], zeroline=True, zerolinecolor="gray")
    st.plotly_chart(fig, use_container_width=True)

def load_urea_price():
    try:
        df = fetch_indexmundi_urea_prices(12)
        if df.empty:
            raise ValueError("No urea price rows were parsed from IndexMundi.")
        return df, "IndexMundi (live retrieval)"
    except Exception:
        months = pd.date_range(end=today, periods=12, freq="MS")
        values = np.linspace(360, 520, len(months))
        return (
            pd.DataFrame({"date": months, "price_usd_per_ton": values}),
            "SIMULATED fallback data",
        )


@st.cache_data(ttl=21600)
def load_fertilizer_prices():
    try:
        df = fetch_world_bank_fertilizer_prices()
        shock_factor, price_ratio = fertilizer_price_shock_factor(df)
        return df, shock_factor, price_ratio, "World Bank Pink Sheet"
    except Exception:
        df, source = load_urea_price()
        if df.empty:
            return df, 0.5, 1.0, source
        fallback = df.rename(columns={"price_usd_per_ton": "urea_usd_per_ton"}).copy()
        fallback["fertilizer_price_index"] = 100 * fallback["urea_usd_per_ton"] / fallback["urea_usd_per_ton"].iloc[0]
        shock_factor, price_ratio = fertilizer_price_shock_factor(fallback)
        return fallback, shock_factor, price_ratio, source


@st.cache_data(ttl=21600)
def load_indian_fertilizer_prices():
    return load_india_fertilizer_prices()


@st.cache_data(ttl=21600)
def load_regional_snapshot(current_start, current_end, climate_start, climate_end):
    return build_regional_agroclimate_snapshot(
        current_start,
        cap_end_date(current_end),
        climate_start,
        climate_end,
    )


@st.cache_data(ttl=21600)
def load_food_price_index():
    return fetch_staple_food_price_index(12)

try:
    current, baseline, rainfall_source = load_rain(start, end, baseline_start, baseline_end)

    if current.empty or baseline.empty:
        raise ValueError(
            "No valid rainfall data available. For future periods NASA POWER currently does not provide daily data."
        )

    df = build_monsoon_anomaly(current, baseline)
    rainfall_is_simulated = False
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].max() > today:
        df["data_type"] = np.where(df["date"] <= today, "Observed", "Expected")
        st.info(f"Actual rainfall is shown through {today.date()}. Values after that date are projected/expected.")
    else:
        df["data_type"] = "Observed"
    latest_obs = df["date"].max()
    if latest_obs < today:
        st.warning(
            f"Latest observed rainfall is available through {latest_obs.date()}. "
            "NASA POWER may not have published newer daily rainfall values yet."
        )
except Exception as e:
    st.warning(
        "NASA POWER could not be loaded or valid rainfall data is missing. Using simulated fallback data."
    )
    st.info("Note: For future dates NASA POWER does not provide daily precipitation values.")
    st.error(
        "Rainfall charts below use SIMULATED fallback data and must not be interpreted as observations."
    )
    dates = pd.date_range(start=start, end=cap_end_date(end), freq="D")
    df = pd.DataFrame({"date": dates})
    df["precipitation_mm"] = [1.5 + i * 0.05 for i in range(len(df))]
    df["baseline_mm"] = [1.75 + i * 0.04 for i in range(len(df))]
    df["actual_cum_mm"] = df["precipitation_mm"].cumsum()
    df["baseline_cum_mm"] = df["baseline_mm"].cumsum()
    df["rainfall_anomaly_pct"] = 100 * (df["actual_cum_mm"] / df["baseline_cum_mm"] - 1)
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].max() > today:
        df["data_type"] = np.where(df["date"] <= today, "Observed", "Expected")
        st.info(f"Actual rainfall is shown through {today.date()}. Values after that date are projected/expected.")
    else:
        df["data_type"] = "Simulated"
    rainfall_source = "SIMULATED fallback data"
    rainfall_is_simulated = True

if manual_oni:
    oni = oni_input
    oni_history = pd.DataFrame()
    oni_source = "manual scenario input"
else:
    oni, oni_history, oni_source = load_oni()
if not manual_oni:
    st.sidebar.info(f"ONI: {oni:.2f} ({oni_source})")
    if oni_source != "NOAA CPC":
        st.sidebar.warning("NOAA ONI could not be loaded; the displayed ONI is a fallback assumption.")

rainfall_values = df.get("rainfall_anomaly_pct", pd.Series(dtype=float)).dropna()
if rainfall_values.empty:
    st.warning("No valid Kerala rainfall anomaly was available; the local fallback is 0%.")
    latest_anomaly = 0.0
else:
    latest_anomaly = float(rainfall_values.iloc[-1])

try:
    regional_df = load_regional_snapshot(start, end, baseline_start, baseline_end)
    if regional_df.empty:
        raise ValueError("No regional agroclimate rows were returned.")
    regional_rainfall_anomaly = float(regional_df["rainfall_anomaly_pct"].median())
    regional_soil_anomaly = float(regional_df["soil_moisture_anomaly_pct"].median())
    regional_source = "NASA POWER regional monitoring points"
except Exception:
    regional_df = pd.DataFrame()
    regional_rainfall_anomaly = latest_anomaly if "latest_anomaly" in locals() else 0.0
    regional_soil_anomaly = 0.0
    regional_source = "Unavailable; Kerala rainfall fallback used"

try:
    food_price_df = load_food_price_index()
    if len(food_price_df) < 4:
        raise ValueError("Insufficient food-price history.")
    latest_food_index = float(food_price_df["food_price_index"].iloc[-1])
    prior_food_index = float(food_price_df["food_price_index"].iloc[-4])
    food_price_change_pct = 100 * (latest_food_index / prior_food_index - 1)
    automatic_food_price_stress = food_price_stress_from_change(food_price_change_pct)
    food_price_source = "Rice, wheat, maize and soybean-oil global benchmarks"
except Exception:
    food_price_df = pd.DataFrame()
    food_price_change_pct = 0.0
    automatic_food_price_stress = 25.0
    food_price_source = "Configured fallback value"

fertilizer_price_df, price_shock, fertilizer_price_ratio, fertilizer_price_source = load_fertilizer_prices()
india_fertilizer_price_df, india_fertilizer_price_source = load_indian_fertilizer_prices()
food_price = float(food_price_input) if manual_food_price else automatic_food_price_stress
monsoon_stress = monsoon_stress_from_anomaly(regional_rainfall_anomaly)
enso_stress = enso_stress_from_oni(oni)
fert_stress = fertilizer_stress(0.0, import_dep, price_shock)
crop_condition_stress = vegetation_soil_stress(
    regional_soil_anomaly, regional_rainfall_anomaly
)
score = risk_score(
    monsoon_stress,
    enso_stress,
    fert_stress,
    food_price,
    crop_condition_stress,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Composite Stress Indicator", f"{score:.0f}/100", risk_label(score))
c2.metric("Monsoon Stress", f"{monsoon_stress:.0f}/100", f"{regional_rainfall_anomaly:.1f}% regional median")
c3.metric("ENSO Stress", f"{enso_stress:.0f}/100", f"ONI {oni:.1f}")
c4.metric("Fertilizer Stress", f"{fert_stress:.0f}/100", f"Import exposure {import_dep:.2f}")
st.caption(
    f"Rainfall source: {rainfall_source}. ONI source: {oni_source}. "
    f"Regional source: {regional_source}. Food-price source: {food_price_source}. "
    "Fertilizer stress remains a scenario input."
)

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
    wb1, wb2, wb3 = st.columns(3)
    wb1.metric("Regional median wet-bulb", f"{wet_bulb_mean:.1f} °C")
    wb2.metric("Highest regional daily value", f"{wet_bulb_max:.1f} °C")
    wb3.metric("Median anomaly vs baseline", f"{wet_bulb_anomaly:+.1f} °C")

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
                "wet_bulb_max_c": "Maximum wet-bulb temperature (°C)",
                "wet_bulb_anomaly_c": "Anomaly (°C)",
            },
            title="Maximum observed wet-bulb temperature by agricultural region",
        ),
        use_container_width=True,
    )
st.caption(
    "Wet-bulb temperature is estimated from NASA POWER 2 m air temperature and relative humidity "
    "using the Stull approximation. It is not WBGT, a weather-station measurement, or currently a "
    "component of the Composite Stress Indicator. The ≥28 °C count is descriptive, not a universal "
    "health or crop-loss threshold."
)

with st.expander("What is ENSO and where does the ONI come from?", expanded=False):
    st.markdown(
        "ENSO (El Niño-Southern Oscillation) describes coupled variations in oceanic and atmospheric "
        "conditions in the tropical Pacific. Its relationship with India's monsoon is statistical, "
        "not deterministic, and varies by season and region."
    )
    st.markdown(
        "The ONI is NOAA's three-month running mean sea-surface-temperature anomaly in the "
        "Niño 3.4 region. Values beyond ±0.5 °C indicate El Niño- or La Niña-like ocean "
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
        latest_oni_row = oni_history.iloc[-1]
        st.caption(
            f"Latest loaded NOAA value: {latest_oni_row['season']} "
            f"({latest_oni_row['date']:%Y-%m}), ONI {latest_oni_row['oni']:.1f} °C. "
            "Shading marks the ±0.5 °C oceanic thresholds."
        )

components = pd.DataFrame({
    "Component": ["Monsoon", "ENSO", "Fertilizer", "Food Price", "Crop Condition"],
    "Stress": [monsoon_stress, enso_stress, fert_stress, food_price, crop_condition_stress],
    "Weight": [0.30, 0.20, 0.20, 0.15, 0.15],
})
st.plotly_chart(
    px.bar(components, x="Component", y="Stress", text="Stress", title="Indicator components"),
    use_container_width=True,
)

st.subheader("Three-month stress outlook")
forecast_inputs = {
    "reference_date": today,
    "rainfall_anomaly_pct": regional_rainfall_anomaly,
    "soil_moisture_anomaly_pct": regional_soil_anomaly,
    "oni": oni,
    "import_exposure": import_dep,
    "fertilizer_price_ratio": fertilizer_price_ratio,
    "food_price_stress": food_price,
    "oni_history": oni_history,
    "fertilizer_history": fertilizer_price_df,
    "food_price_history": food_price_df,
}
forecast_df = simulate_three_month_forecast(**forecast_inputs, scenario="baseline")
dry_forecast_df = simulate_three_month_forecast(**forecast_inputs, scenario="dry")
favorable_forecast_df = simulate_three_month_forecast(**forecast_inputs, scenario="favorable")
forecast_chart = go.Figure()
forecast_chart.add_trace(
    go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["p90"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    )
)
forecast_chart.add_trace(
    go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["p10"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(255, 127, 14, 0.20)",
        line={"width": 0},
        name="80% model interval",
        hovertemplate="Lower bound %{y:.1f}<extra></extra>",
    )
)
forecast_chart.add_trace(
    go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["median"],
        mode="lines+markers",
        line={"color": "#d65f0e", "width": 3},
        marker={"size": 9},
        name="Median forecast",
        hovertemplate="%{x|%B %Y}: %{y:.1f}/100<extra></extra>",
    )
)
forecast_chart.add_trace(
    go.Scatter(
        x=dry_forecast_df["date"],
        y=dry_forecast_df["median"],
        mode="lines+markers",
        line={"color": "#b2182b", "width": 2, "dash": "dash"},
        marker={"size": 8},
        name="Persistent dryness / price pressure",
        hovertemplate="%{x|%B %Y}: %{y:.1f}/100<extra></extra>",
    )
)
forecast_chart.add_trace(
    go.Scatter(
        x=favorable_forecast_df["date"],
        y=favorable_forecast_df["median"],
        mode="lines+markers",
        line={"color": "#1a9850", "width": 2, "dash": "dash"},
        marker={"size": 8},
        name="Favorable monsoon",
        hovertemplate="%{x|%B %Y}: %{y:.1f}/100<extra></extra>",
    )
)
forecast_chart.update_layout(
    title="Composite stress scenarios (baseline interval shown as shading)",
    yaxis={"title": "Stress score", "range": [0, 100]},
    xaxis={
        "title": None,
        "tickmode": "array",
        "tickvals": forecast_df["date"],
        "ticktext": forecast_df["date"].dt.strftime("%b %Y"),
    },
    hovermode="x unified",
)
st.plotly_chart(forecast_chart, use_container_width=True)

forecast_table = forecast_df[[
    "date", "median", "p10", "p90", "prob_low", "prob_elevated", "prob_high", "prob_critical"
]].copy()
forecast_table["month"] = forecast_table["date"].dt.strftime("%b %Y")
for probability_column in ["prob_low", "prob_elevated", "prob_high", "prob_critical"]:
    forecast_table[probability_column] = (100 * forecast_table[probability_column]).round(0)
st.dataframe(
    forecast_table[[
        "month", "median", "p10", "p90", "prob_low", "prob_elevated", "prob_high", "prob_critical"
    ]].rename(columns={
        "month": "Month",
        "median": "Median",
        "p10": "P10",
        "p90": "P90",
        "prob_low": "Low %",
        "prob_elevated": "Elevated %",
        "prob_high": "High %",
        "prob_critical": "Critical %",
    }),
    use_container_width=True,
    hide_index=True,
)
scenario_table = pd.DataFrame({
    "Month": forecast_df["date"].dt.strftime("%b %Y"),
    "Favorable monsoon": favorable_forecast_df["median"].round(1),
    "Baseline": forecast_df["median"].round(1),
    "Persistent dryness / price pressure": dry_forecast_df["median"].round(1),
})
st.dataframe(scenario_table, use_container_width=True, hide_index=True)
st.caption(
    "Each path uses 2,000 Monte Carlo simulations. The baseline now preserves more of the current "
    "weather anomaly; the dry scenario assumes persistent rainfall deficits and additional price pressure, "
    "while the favorable scenario assumes improving monsoon conditions. The scenarios are alternatives, "
    "not assigned probabilities. P10-P90 shading shows baseline model uncertainty and is not a confidence "
    "interval for a food crisis. The model has not yet been operationally validated."
)
if rainfall_is_simulated or fertilizer_price_source.startswith("SIMULATED"):
    st.warning("At least one forecast input is simulated; the outlook is illustrative only.")

st.subheader("Automatic staple-food price indicator")
price_c1, price_c2 = st.columns(2)
price_c1.metric("Food-price stress", f"{food_price:.0f}/100")
price_c2.metric("Three-month benchmark change", f"{food_price_change_pct:.1f}%")
st.caption(
    f"Source: {food_price_source}. The equal-weight index uses international benchmarks and "
    "does not replace Indian retail or wholesale market prices."
)
if manual_food_price:
    st.info("The manually selected food-price stress currently overrides the automatic index.")
elif food_price_source == "Configured fallback value":
    st.error("Food-price benchmarks could not be loaded; the score uses a configured fallback value.")
if not food_price_df.empty:
    st.plotly_chart(
        px.line(
            food_price_df,
            x="date",
            y="food_price_index",
            markers=True,
            title="Staple-food benchmark index (first available month = 100)",
        ),
        use_container_width=True,
    )

summary = None
st.subheader("Indian fertilizer farmer prices")
if india_fertilizer_price_df.empty:
    st.warning("No Indian fertilizer price data is available.")
else:
    st.caption(
        f"Source: {india_fertilizer_price_source}. These are farmer-facing Indian bag prices/MRPs where available; "
        "they are distinct from global import benchmark prices."
    )
    if india_fertilizer_price_source == "configured fallback values":
        st.warning(
            "Only configured fallback Indian prices are available. Add data/india_fertilizer_prices.csv "
            "to track DAP, MOP, NPK or state/company-specific MRPs."
        )
    display_columns = ["fertilizer", "bag_size_kg", "price_inr_per_bag", "price_inr_per_kg"]
    if "as_of" in india_fertilizer_price_df.columns:
        display_columns.append("as_of")
    if "source_note" in india_fertilizer_price_df.columns:
        display_columns.append("source_note")
    st.dataframe(
        india_fertilizer_price_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

price_df = fertilizer_price_df
price_source = fertilizer_price_source
if not price_df.empty:
    latest_price = float(price_df["fertilizer_price_index"].iloc[-1])
    previous_price = float(price_df["fertilizer_price_index"].iloc[-2]) if len(price_df) > 1 else latest_price
    price_change_pct = 0.0 if previous_price == 0 else 100 * (latest_price / previous_price - 1)
    st.subheader("Global fertilizer import cost pressure")
    st.caption(
        f"Source: {price_source}. The indicator combines global Urea, DAP and Potassium chloride benchmarks "
        "where available and is only a proxy for Indian fertilizer cost trends."
    )
    if price_source.startswith("SIMULATED"):
        st.error("The fertilizer price chart uses SIMULATED fallback data, not current market observations.")
    st.metric("Global fertilizer benchmark index", f"{latest_price:.1f}", f"{fertilizer_price_ratio:.2f}x 5-year median")

    if score <= 30:
        risk_phrase = "low"
        risk_detail = "a low composite stress signal"
    elif score <= 55:
        risk_phrase = "moderate"
        risk_detail = "an elevated composite stress signal"
    elif score <= 75:
        risk_phrase = "high"
        risk_detail = "a high composite stress signal"
    else:
        risk_phrase = "critical"
        risk_detail = "a critical composite stress signal"

    rain_phrase = "above baseline" if regional_rainfall_anomaly >= 0 else "below baseline"
    rain_detail = "supportive for crops" if regional_rainfall_anomaly >= 0 else "a crop-stress warning"
    enso_phrase = enso_label(oni)
    fert_phrase = "elevated" if fertilizer_price_ratio >= 1.25 else "near its recent baseline"

    summary = (
        f"**Current indicator reading:** the composite stress signal is **{risk_phrase}**. "
        f"Regional rainfall is {rain_phrase} at {regional_rainfall_anomaly:.1f}% versus baseline, which is {rain_detail}; "
        f"ENSO is at ONI {oni:.1f} ({enso_phrase}); and fertilizer prices are {fert_phrase} "
        f"at {fertilizer_price_ratio:.2f}x their five-year median. Staple-food benchmarks changed "
        f"{food_price_change_pct:.1f}% over three months and crop-condition stress is {crop_condition_stress:.0f}/100. "
        f"Overall, this is {risk_detail}. "
        "This indicator is not a probability, forecast or substitute for regional food-security data."
    )

    if rainfall_is_simulated or price_source.startswith("SIMULATED"):
        summary += " At least one input series is simulated, so this assessment is illustrative only."

    st.plotly_chart(
        px.line(
            price_df,
            x="date",
            y="fertilizer_price_index",
            markers=True,
            title="Fertilizer benchmark price index",
        ),
        use_container_width=True,
    )

st.info(
    "Interpretation: This is a heuristic composite stress indicator. For operational use, augment with FAO/WFP/FEWS NET, Indian grain reserves, domestic retail prices and crop-yield data."
)

if summary is not None:
    st.subheader("Current indicator summary")
    st.info(summary)
