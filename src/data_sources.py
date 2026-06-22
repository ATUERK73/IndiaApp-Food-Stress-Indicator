from __future__ import annotations

from io import BytesIO, StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re

import numpy as np
import pandas as pd
import requests

IMD_RAINFALL_FILE = os.path.join("data", "imd_kerala_daily.csv")
INDIA_FERTILIZER_PRICE_FILE = os.path.join("data", "india_fertilizer_prices.csv")


NASA_POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"
WORLD_BANK_COMMODITY_MONTHLY_XLSX_URL = "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx"

ONI_SEASON_END_MONTH = {
    "DJF": 2, "JFM": 3, "FMA": 4, "MAM": 5, "AMJ": 6, "MJJ": 7,
    "JJA": 8, "JAS": 9, "ASO": 10, "SON": 11, "OND": 12, "NDJ": 1,
}

KERALA_LOCATIONS = {
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kochi": (9.9312, 76.2673),
    "Kozhikode": (11.2588, 75.7804),
}

INDIA_AGRICULTURAL_REGIONS = {
    "Punjab": (30.90, 75.85),
    "Haryana": (29.06, 76.09),
    "Uttar Pradesh": (26.85, 80.95),
    "Rajasthan": (26.91, 75.79),
    "Madhya Pradesh": (23.26, 77.41),
    "Gujarat": (23.02, 72.57),
    "Maharashtra": (19.75, 75.71),
    "West Bengal": (22.57, 88.36),
    "Odisha": (20.30, 85.82),
    "Telangana": (17.39, 78.49),
    "Karnataka": (15.32, 75.71),
    "Kerala": (10.85, 76.27),
}

STAPLE_COMMODITIES = {
    "Rice": "rice",
    "Wheat": "wheat",
    "Maize": "corn",
    "Soybean oil": "soybean-oil",
}

FERTILIZER_PRICE_COMPONENTS = {
    "urea_usd_per_ton": 0.50,
    "dap_usd_per_ton": 0.30,
    "potassium_chloride_usd_per_ton": 0.20,
}

INDIA_FERTILIZER_PRICE_FALLBACK = [
    {
        "fertilizer": "Urea",
        "bag_size_kg": 45.0,
        "price_inr_per_bag": 266.50,
        "source_note": "configured fallback; regulated Indian farmer MRP commonly cited for a 45 kg bag",
    },
]


def wet_bulb_temperature_c(
    air_temperature_c: float | np.ndarray | pd.Series,
    relative_humidity_pct: float | np.ndarray | pd.Series,
) -> float | np.ndarray:
    """Approximate psychrometric wet-bulb temperature in degrees Celsius.

    Uses the Stull (2011) approximation for near-surface air temperature and
    relative humidity. It is a monitoring proxy and is not WBGT.
    """
    temperature = np.asarray(air_temperature_c, dtype=float)
    humidity = np.clip(np.asarray(relative_humidity_pct, dtype=float), 0.0, 100.0)
    result = (
        temperature * np.arctan(0.151977 * np.sqrt(humidity + 8.313659))
        + np.arctan(temperature + humidity)
        - np.arctan(humidity - 1.676331)
        + 0.00391838 * humidity ** 1.5 * np.arctan(0.023101 * humidity)
        - 4.686035
    )
    return float(result) if result.ndim == 0 else result


def load_india_fertilizer_prices(file_path: str = INDIA_FERTILIZER_PRICE_FILE) -> tuple[pd.DataFrame, str]:
    """Load Indian farmer fertilizer prices from a local CSV, with a clearly marked fallback."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        source = file_path
    else:
        df = pd.DataFrame(INDIA_FERTILIZER_PRICE_FALLBACK)
        source = "configured fallback values"

    required_columns = {"fertilizer", "bag_size_kg", "price_inr_per_bag"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Indian fertilizer price data is missing columns: {sorted(missing_columns)}")

    df = df.copy()
    df["bag_size_kg"] = pd.to_numeric(df["bag_size_kg"], errors="coerce")
    df["price_inr_per_bag"] = pd.to_numeric(df["price_inr_per_bag"], errors="coerce")
    df = df.dropna(subset=["bag_size_kg", "price_inr_per_bag"])
    df = df[df["bag_size_kg"] > 0]
    df["price_inr_per_kg"] = df["price_inr_per_bag"] / df["bag_size_kg"]
    if "as_of" in df.columns:
        df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce").dt.date
    return df.reset_index(drop=True), source


def fetch_nasa_power_precipitation(
    lat: float,
    lon: float,
    start: str,
    end: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch daily precipitation from NASA POWER. Returns DATE + precipitation_mm.

    PRECTOTCORR is corrected precipitation in mm/day.
    """
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "JSON",
    }
    r = requests.get(NASA_POWER_DAILY_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]["PRECTOTCORR"]
    df = pd.DataFrame({"date": pd.to_datetime(list(data.keys())), "precipitation_mm": list(data.values())})
    df["precipitation_mm"] = pd.to_numeric(df["precipitation_mm"], errors="coerce")
    df = df.replace(-999, np.nan).dropna()
    return df


def fetch_nasa_power_agroclimate(
    lat: float,
    lon: float,
    start: str,
    end: str,
    timeout: int = 45,
) -> pd.DataFrame:
    """Fetch precipitation, soil wetness, temperature and humidity from NASA POWER."""
    params = {
        "parameters": "PRECTOTCORR,GWETROOT,T2M,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "JSON",
    }
    response = requests.get(NASA_POWER_DAILY_URL, params=params, timeout=timeout)
    response.raise_for_status()
    parameters = response.json()["properties"]["parameter"]
    dates = list(parameters["PRECTOTCORR"].keys())
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "precipitation_mm": [parameters["PRECTOTCORR"].get(day) for day in dates],
            "root_zone_wetness": [parameters["GWETROOT"].get(day) for day in dates],
            "air_temperature_c": [parameters["T2M"].get(day) for day in dates],
            "relative_humidity_pct": [parameters["RH2M"].get(day) for day in dates],
        }
    )
    for column in [
        "precipitation_mm",
        "root_zone_wetness",
        "air_temperature_c",
        "relative_humidity_pct",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace(-999, np.nan).dropna()
    df["wet_bulb_temperature_c"] = wet_bulb_temperature_c(
        df["air_temperature_c"], df["relative_humidity_pct"]
    )
    return df


def build_regional_agroclimate_snapshot(
    current_start: str,
    current_end: str,
    baseline_start: str,
    baseline_end: str,
    max_workers: int = 6,
) -> pd.DataFrame:
    """Build rainfall and soil-moisture anomalies for major agricultural regions."""
    def fetch_region(name: str, lat: float, lon: float) -> dict:
        current = fetch_nasa_power_agroclimate(lat, lon, current_start, current_end)
        baseline = fetch_nasa_power_agroclimate(lat, lon, baseline_start, baseline_end)
        if current.empty or baseline.empty:
            raise ValueError(f"No agroclimate data for {name}.")

        current_total = current["precipitation_mm"].sum()
        current_days = current["date"].dt.dayofyear
        baseline_match = baseline[baseline["date"].dt.dayofyear.isin(current_days)]
        baseline_years = max(1, baseline_match["date"].dt.year.nunique())
        baseline_total = baseline_match["precipitation_mm"].sum() / baseline_years
        rainfall_anomaly = 100 * (current_total / baseline_total - 1) if baseline_total else np.nan

        current_soil = current["root_zone_wetness"].mean()
        baseline_soil = baseline_match["root_zone_wetness"].mean()
        soil_anomaly = 100 * (current_soil / baseline_soil - 1) if baseline_soil else np.nan
        current_wet_bulb = current["wet_bulb_temperature_c"]
        baseline_wet_bulb = baseline_match["wet_bulb_temperature_c"]
        return {
            "region": name,
            "latitude": lat,
            "longitude": lon,
            "rainfall_anomaly_pct": rainfall_anomaly,
            "soil_moisture_anomaly_pct": soil_anomaly,
            "wet_bulb_mean_c": current_wet_bulb.mean(),
            "wet_bulb_max_c": current_wet_bulb.max(),
            "wet_bulb_anomaly_c": current_wet_bulb.mean() - baseline_wet_bulb.mean(),
            "wet_bulb_days_ge_28c": int((current_wet_bulb >= 28.0).sum()),
            "latest_date": current["date"].max(),
        }

    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_region, name, lat, lon): name
            for name, (lat, lon) in INDIA_AGRICULTURAL_REGIONS.items()
        }
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception:
                continue
    return pd.DataFrame(records)


def load_imd_rainfall(start: str, end: str, file_path: str = IMD_RAINFALL_FILE) -> pd.DataFrame:
    """Load daily Kerala rainfall from a local IMD CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"IMD data file not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["date"])
    if "precipitation_mm" not in df.columns:
        raise ValueError(
            f"IMD data file {file_path} must contain a 'date' and 'precipitation_mm' column."
        )

    df["precipitation_mm"] = pd.to_numeric(df["precipitation_mm"], errors="coerce")
    df = df.dropna(subset=["precipitation_mm"])
    df = df.sort_values("date")

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)
    return df.loc[mask, ["date", "precipitation_mm"]].reset_index(drop=True)


def fetch_kerala_rainfall(start: str, end: str) -> pd.DataFrame:
    """Average 3 Kerala city/grid points as proxy for Kerala rainfall."""
    frames = []
    for name, (lat, lon) in KERALA_LOCATIONS.items():
        df = fetch_nasa_power_precipitation(lat, lon, start, end)
        df["location"] = name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return raw.groupby("date", as_index=False)["precipitation_mm"].mean()


def build_monsoon_anomaly(current: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Compare current cumulative rainfall with baseline day-of-year climatology."""
    if current.empty or baseline.empty:
        raise ValueError("Current and baseline rainfall data must not be empty.")

    cur = current.sort_values("date").copy()
    base = baseline.sort_values("date").copy()
    cur["doy"] = cur["date"].dt.dayofyear
    base["doy"] = base["date"].dt.dayofyear
    clim = base.groupby("doy", as_index=False)["precipitation_mm"].mean().rename(columns={"precipitation_mm": "baseline_mm"})
    out = cur.merge(clim, on="doy", how="left")
    if out["baseline_mm"].isna().any():
        raise ValueError("Baseline rainfall does not cover all requested calendar days.")
    out["actual_cum_mm"] = out["precipitation_mm"].cumsum()
    out["baseline_cum_mm"] = out["baseline_mm"].cumsum()
    out["rainfall_anomaly_pct"] = 100 * (out["actual_cum_mm"] / out["baseline_cum_mm"] - 1)
    return out


def fetch_noaa_oni() -> pd.DataFrame:
    """Parse NOAA CPC ONI table from the public HTML page."""
    tables = pd.read_html(NOAA_ONI_URL)

    oni_table = None
    for table in tables:
        if table.shape[1] == 13 and str(table.iloc[0, 0]).strip() == "Year":
            oni_table = table.copy()
            oni_table.columns = oni_table.iloc[0].astype(str).str.strip()
            oni_table = oni_table.iloc[1:].reset_index(drop=True)
            break

    if oni_table is None:
        raise ValueError("Could not find ONI table on NOAA ONI page.")

    season_cols = [col for col in oni_table.columns if col != "Year"]
    oni_table[season_cols] = oni_table[season_cols].apply(pd.to_numeric, errors="coerce")
    oni_table["Year"] = pd.to_numeric(oni_table["Year"], errors="coerce")
    return oni_table


def latest_oni_value(default: float = 0.0) -> float:
    try:
        df = fetch_noaa_oni()
        last_value = df.iloc[:, 1:].stack().dropna().iloc[-1]
        return float(last_value)
    except Exception:
        return default


def oni_table_to_series(oni_table: pd.DataFrame) -> pd.DataFrame:
    """Convert NOAA's year-by-season ONI table to a chronological series."""
    required_columns = {"Year", *ONI_SEASON_END_MONTH}
    missing_columns = required_columns.difference(oni_table.columns)
    if missing_columns:
        raise ValueError(f"ONI table is missing columns: {sorted(missing_columns)}")

    records = []
    for _, row in oni_table.iterrows():
        year = pd.to_numeric(row["Year"], errors="coerce")
        if pd.isna(year):
            continue
        year = int(year)
        for season, end_month in ONI_SEASON_END_MONTH.items():
            value = pd.to_numeric(row[season], errors="coerce")
            if pd.isna(value):
                continue
            records.append({
                "date": pd.Timestamp(year + (season == "NDJ"), end_month, 1),
                "season": season,
                "oni": float(value),
            })
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def fetch_noaa_oni_series() -> pd.DataFrame:
    """Fetch NOAA CPC ONI values as a chronological three-month-season series."""
    return oni_table_to_series(fetch_noaa_oni())


def build_fertilizer_price_index(price_df: pd.DataFrame) -> pd.DataFrame:
    """Build a weighted fertilizer price index where 100 is the trailing 5-year median."""
    required_columns = {"date", *FERTILIZER_PRICE_COMPONENTS}
    missing_columns = required_columns.difference(price_df.columns)
    if missing_columns:
        raise ValueError(f"Fertilizer price data is missing columns: {sorted(missing_columns)}")

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for column in FERTILIZER_PRICE_COMPONENTS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    index_values = []
    for idx, row in df.iterrows():
        baseline = df.iloc[max(0, idx - 60):idx]
        ratios = []
        weights = []
        for column, weight in FERTILIZER_PRICE_COMPONENTS.items():
            latest_value = row[column]
            median_value = baseline[column].median() if not baseline.empty else np.nan
            if pd.isna(latest_value) or pd.isna(median_value) or median_value <= 0:
                continue
            ratios.append(weight * float(latest_value / median_value))
            weights.append(weight)
        index_values.append(100 * sum(ratios) / sum(weights) if ratios else np.nan)

    df["fertilizer_price_index"] = index_values
    return df.dropna(subset=["fertilizer_price_index"]).reset_index(drop=True)


def fertilizer_price_shock_factor(price_df: pd.DataFrame, baseline_months: int = 60) -> tuple[float, float]:
    """Return a 0-1 shock factor and latest/5-year-median price ratio."""
    if len(price_df) < 2:
        return 0.0, 1.0

    history = price_df.tail(baseline_months + 1)
    latest = history.iloc[-1]
    baseline = history.iloc[:-1]
    ratios = []
    weights = []
    for column, weight in FERTILIZER_PRICE_COMPONENTS.items():
        if column not in price_df.columns:
            continue
        latest_value = pd.to_numeric(latest.get(column), errors="coerce")
        median_value = pd.to_numeric(baseline[column], errors="coerce").median()
        if pd.isna(latest_value) or pd.isna(median_value) or median_value <= 0:
            continue
        ratios.append(weight * float(latest_value / median_value))
        weights.append(weight)
    if not ratios:
        return 0.0, 1.0

    ratio = sum(ratios) / sum(weights)
    shock_factor = max(0.0, min(1.0, (ratio - 1.0) / 0.75))
    return shock_factor, ratio


def fetch_world_bank_fertilizer_prices(timeout: int = 30) -> pd.DataFrame:
    """Fetch monthly fertilizer prices from the World Bank Pink Sheet."""
    response = requests.get(WORLD_BANK_COMMODITY_MONTHLY_XLSX_URL, timeout=timeout)
    response.raise_for_status()
    raw = pd.read_excel(BytesIO(response.content), sheet_name="Monthly Prices", header=4)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.iloc[:, 0].astype(str), format="%YM%m", errors="coerce"),
            "urea_usd_per_ton": pd.to_numeric(raw.get("Urea "), errors="coerce"),
            "dap_usd_per_ton": pd.to_numeric(raw.get("DAP"), errors="coerce"),
            "potassium_chloride_usd_per_ton": pd.to_numeric(
                raw.get("Potassium chloride **"), errors="coerce"
            ),
        }
    )
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    return build_fertilizer_price_index(df)


def fetch_indexmundi_urea_prices(months: int = 12, timeout: int = 30) -> pd.DataFrame:
    """Scrape IndexMundi monthly urea prices as a proxy for fertilizer cost."""
    url = f"https://www.indexmundi.com/commodities/?commodity=urea&months={months}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text

    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        tables = []

    for table in tables:
        if table.shape[1] < 2:
            continue

        first_col = str(table.columns[0]).strip().lower()
        second_col = str(table.columns[1]).strip().lower()
        if "month" not in first_col or "price" not in second_col:
            continue

        df = table.iloc[:, :2].copy()
        df.columns = ["month", "price_usd_per_ton"]
        df["month"] = df["month"].astype(str).str.strip()
        df = df[~df["month"].str.contains(r"Month|Price|Change", case=False, regex=True, na=False)]
        df["price_usd_per_ton"] = pd.to_numeric(df["price_usd_per_ton"].astype(str).str.replace(",", "", regex=False), errors="coerce")
        df["date"] = pd.to_datetime(df["month"], format="%b %Y", errors="coerce")
        df = df.dropna(subset=["date", "price_usd_per_ton"])
        if not df.empty:
            return df[["date", "price_usd_per_ton"]].sort_values("date").reset_index(drop=True)

    table_match = re.search(r"<table[^>]*>.*?<th[^>]*>Month.*?</table>", html, flags=re.S | re.I)
    if table_match is None:
        raise ValueError("Could not find the IndexMundi urea price table in the HTML response.")

    table_html = table_match.group(0)
    rows = re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.S | re.I)
    records = []
    for row in rows:
        cols = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)
        if len(cols) < 2:
            continue
        month_text = re.sub(r"<.*?>", "", cols[0]).strip()
        price_text = re.sub(r"<.*?>", "", cols[1]).strip()
        if not month_text or not price_text or re.search(r"Month|Price|Change", month_text, re.I):
            continue
        price_num = pd.to_numeric(price_text.replace(",", ""), errors="coerce")
        if pd.isna(price_num):
            continue
        date_val = pd.to_datetime(month_text, format="%b %Y", errors="coerce")
        if pd.isna(date_val):
            continue
        records.append({"date": date_val, "price_usd_per_ton": price_num})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("IndexMundi urea price parsing returned no rows.")
    return df.sort_values("date").reset_index(drop=True)


def fetch_indexmundi_commodity_prices(
    commodity: str,
    months: int = 12,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch a generic monthly commodity benchmark from IndexMundi."""
    url = f"https://www.indexmundi.com/commodities/?commodity={commodity}&months={months}"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    for table in tables:
        if table.shape[1] < 2:
            continue
        columns = [str(column).lower() for column in table.columns[:2]]
        if "month" not in columns[0] or "price" not in columns[1]:
            continue
        result = table.iloc[:, :2].copy()
        result.columns = ["month", "price"]
        result["date"] = pd.to_datetime(result["month"], format="%b %Y", errors="coerce")
        result["price"] = pd.to_numeric(
            result["price"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        result = result.dropna(subset=["date", "price"])
        if not result.empty:
            return result[["date", "price"]].sort_values("date").reset_index(drop=True)
    raise ValueError(f"No price table found for {commodity}.")


def fetch_staple_food_price_index(months: int = 12) -> pd.DataFrame:
    """Create an equal-weight index from global staple commodity benchmarks."""
    frames = []
    for label, commodity in STAPLE_COMMODITIES.items():
        try:
            frame = fetch_indexmundi_commodity_prices(commodity, months)
        except Exception:
            continue
        first_price = frame["price"].iloc[0]
        if first_price <= 0:
            continue
        frame["commodity"] = label
        frame["index"] = 100 * frame["price"] / first_price
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "food_price_index"])
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby("date", as_index=False)["index"]
        .mean()
        .rename(columns={"index": "food_price_index"})
        .sort_values("date")
        .reset_index(drop=True)
    )
