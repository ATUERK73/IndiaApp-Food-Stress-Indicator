# India Composite Food Stress Indicator

Ein explorativer Streamlit-Prototyp zur Beobachtung von Stressindikatoren fuer die
Lebensmittelversorgung in Indien:

- Niederschlagsabweichung in Kerala als regionaler Fruehindikator
- regionale Regen- und Bodenfeuchte-Anomalien fuer wichtige Agrarregionen
- regionale Wet-Bulb-Temperatur aus Lufttemperatur und relativer Feuchte
- ENSO / El Nino ueber den NOAA Oceanic Nino Index (ONI)
- Duengemittelimporte, Preise und ein Hormus-Szenario
- automatischer Grundnahrungsmittel-Preisindex mit manueller Uebersteuerung

## Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Datenquellen und Kennzeichnung

- NASA POWER Daily API fuer Niederschlag
- NOAA CPC ONI fuer ENSO
- World Bank Pink Sheet fuer globale Duengerpreis-Benchmarks
- Department of Fertilizers / offizielle indische Publikationen fuer Importdaten
- IndexMundi als Fallback fuer einen globalen Harnstoffpreis-Proxy
- internationale Reis-, Weizen-, Mais- und Sojaoel-Benchmarks als Preis-Proxy

Die Oberflaeche kennzeichnet Daten als live/lokal, manuelle Szenarioeingabe oder
simulierte Ersatzdaten. Simulierte Daten duerfen nicht als aktuelle Beobachtungen
interpretiert werden.

### Optionaler indischer Duengerpreis-Datensatz

Lege `data/india_fertilizer_prices.csv` an, um farmer-facing indische Preise/MRPs
anzuzeigen. Die Datei benoetigt:

- `fertilizer`, z.B. `Urea`, `DAP`, `MOP`, `NPK 10-26-26`
- `bag_size_kg`
- `price_inr_per_bag`

Optional: `as_of` und `source_note`.

### Optionaler IMD-Datensatz

Lege `data/imd_kerala_daily.csv` an. Die Datei benoetigt:

- `date` im ISO-Format (`YYYY-MM-DD`)
- `precipitation_mm` als taeglichen Niederschlag in Millimetern

## Methodik

Der Prototyp berechnet weder eine Krisenwahrscheinlichkeit noch einen offiziellen
Forecast. Er erzeugt einen heuristischen Composite Stress Indicator von 0 bis 100.
Kerala ist dabei nur ein regionaler Indikator und nicht repraesentativ fuer ganz Indien.

Die Oberflaeche zeigt zusaetzlich einen experimentellen Drei-Monats-Ausblick mit
drei Alternativen: Basisszenario, anhaltende Trockenheit/Preisdruck und guenstiger
Monsun. Jeder Pfad simuliert 2.000 moegliche Entwicklungen fuer Wetter, ENSO und
Preise. Das P10-P90-Band gilt fuer das Basisszenario und beschreibt
Modellunsicherheit. Die drei Szenarien haben keine zugewiesenen
Eintrittswahrscheinlichkeiten und sind keine Wahrscheinlichkeit einer
Lebensmittelkrise. Das Forecast-Modell ist noch nicht operativ backgetestet oder
validiert.

```text
Composite Stress Indicator =
0.30 * MonsoonStress
+ 0.20 * ENSOStress
+ 0.20 * FertilizerStress
+ 0.15 * FoodPriceStress
+ 0.15 * CropConditionStress
```

`CropConditionStress` kombiniert die Abweichung der Wurzelzonen-Bodenfeuchte mit
der regionalen Regenabweichung. Der Wert ist ein Proxy und keine NDVI-Messung.

Die Wet-Bulb-Temperatur wird mit der Stull-Naeherung aus NASA-POWER-Tageswerten
fuer Temperatur in 2 m Hoehe und relative Luftfeuchte berechnet. Sie ist nicht
gleichbedeutend mit WBGT und beeinflusst den Composite Score derzeit nicht.

Die Gewichtungen sind Modellannahmen und derzeit nicht empirisch validiert. Fuer
operative Nutzung sind unter anderem regionale Ernteertraege, Lagerbestaende,
Marktpreise, Haushaltsdaten sowie FAO-, WFP- oder FEWS-NET-Indikatoren erforderlich.
