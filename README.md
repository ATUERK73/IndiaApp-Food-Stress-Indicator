# India Composite Food Stress Indicator

An exploratory Streamlit prototype for monitoring stress indicators related to
India's food supply:

- rainfall anomalies in Kerala as a regional early indicator
- regional rainfall and soil-moisture anomalies across major agricultural regions
- regional wet-bulb temperature derived from air temperature and relative humidity
- ENSO / El Nino conditions based on NOAA's Oceanic Nino Index (ONI)
- an hourly refreshed NOAA ENSO outlook with the weekly index, alert status, and strengthening signal
- fertilizer imports, prices, and a Strait of Hormuz scenario
- an automatic staple-food price index with a manual override

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data sources and labeling

- NASA POWER Daily API for rainfall and regional agroclimate data
- NOAA CPC ONI for ENSO conditions
- World Bank Pink Sheet for global fertilizer price benchmarks
- Department of Fertilizers and official Indian publications for import data
- IndexMundi as a fallback global urea price proxy
- international rice, wheat, maize, and soybean-oil benchmarks as food-price proxies

The interface labels data as live/local, manual scenario input, or simulated
fallback data. Simulated data must not be interpreted as current observations.

### Optional Indian fertilizer price dataset

Create `data/india_fertilizer_prices.csv` to display farmer-facing Indian prices
and maximum retail prices (MRPs). The file requires:

- `fertilizer`, for example `Urea`, `DAP`, `MOP`, or `NPK 10-26-26`
- `bag_size_kg`
- `price_inr_per_bag`

Optional columns: `as_of` and `source_note`.

### Optional IMD dataset

Create `data/imd_kerala_daily.csv`. The file requires:

- `date` in ISO format (`YYYY-MM-DD`)
- `precipitation_mm` for daily rainfall in millimeters

## Methodology

The prototype calculates neither a crisis probability nor an official forecast.
It produces a heuristic Composite Stress Indicator ranging from 0 to 100. Kerala
is only a regional indicator and is not representative of India as a whole.

The interface also presents an experimental three-month outlook with three
alternatives: a baseline scenario, persistent dryness and price pressure, and a
favorable monsoon. Each path simulates 2,000 possible developments in weather,
ENSO, and prices. The P10-P90 band applies to the baseline scenario and represents
model uncertainty. The three scenarios have no assigned probabilities and do not
represent the probability of a food crisis. The forecast model has not yet been
operationally backtested or validated.

When NOAA expects strengthening, the ENSO forecast starts from the more recent
weekly Nino-3.4 value, provided it is higher than the ONI, and applies at least
+0.15 degrees C of model drift per month. This translation of NOAA's qualitative
outlook is an explicit model assumption, not an official NOAA ONI trajectory.

```text
Composite Stress Indicator =
0.27 * MonsoonStress
+ 0.18 * ENSOStress
+ 0.18 * FertilizerStress
+ 0.135 * FoodPriceStress
+ 0.135 * CropConditionStress
+ 0.10 * WetBulbStress
```

`CropConditionStress` combines the root-zone soil-moisture anomaly with the
regional rainfall anomaly. It is a proxy, not an NDVI measurement.

Wet-bulb temperature is calculated using the Stull approximation from NASA POWER
daily values for air temperature at 2 meters and relative humidity. It is not the
same as WBGT. `WetBulbStress` uses the median of the regional daily maximum values:
temperatures up to 24 degrees C receive a stress score of 10, temperatures of
32 degrees C or higher receive a score of 100, and values in between are linearly
interpolated. These thresholds are heuristic model assumptions, not universal
health or crop-loss limits.

The component weights are model assumptions and have not been empirically
validated. Operational use would additionally require regional crop yields,
food stocks, market prices, household data, and indicators from organizations
such as FAO, WFP, or FEWS NET.
