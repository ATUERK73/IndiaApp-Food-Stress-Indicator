# India Composite Food Stress Indicator

An exploratory Streamlit prototype for monitoring stress indicators related to
India's food supply:

- rainfall anomalies in Kerala as a regional early indicator
- regional rainfall and soil-moisture anomalies across major agricultural regions
- regional wet-bulb temperature derived from air temperature and relative humidity
- ENSO / El Nino conditions based on NOAA's Oceanic Nino Index (ONI)
- an hourly refreshed NOAA ENSO outlook with the weekly index, alert status, and strengthening signal
- a six-month best-case / worst-case scenario outlook using 2,000 Monte Carlo paths per scenario
- a probability-weighted ENSO impact simulation using NOAA's stated strengthening probability when available

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data sources and labeling

- NASA POWER Daily API for rainfall and regional agroclimate data
- NOAA CPC ONI for ENSO conditions

The interface labels data as live/local, manual scenario input, or simulated
fallback data. Simulated data must not be interpreted as current observations.

### Optional IMD dataset

Create `data/imd_kerala_daily.csv`. The file requires:

- `date` in ISO format (`YYYY-MM-DD`)
- `precipitation_mm` for daily rainfall in millimeters

## Methodology

The prototype calculates neither a crisis probability nor an official forecast.
It produces a heuristic Composite Stress Indicator ranging from 0 to 100. Kerala
is only a regional indicator and is not representative of India as a whole.
The six-month outlook is a scenario stress test, not a prediction.
The ENSO probability panel is a conditional impact calculation, not an official
NOAA forecast translation.

```text
Composite Stress Indicator =
0.35 * MonsoonStress
+ 0.25 * ENSOStress
+ 0.25 * CropConditionStress
+ 0.15 * WetBulbStress
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
