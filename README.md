# nordic-trading-pipeline
A Python project that fetches day-ahead power prices and stores them in a local database for later use.

This project uses Python libraries, including Pandas, Polars, and SQLite3, to extract electricity data from ENTSO-E Transparency Platform using the API. Then the data is written to a local database for analysis, visualization, and developing the XGBoost machine learning model for price prediction.

The project focuses on day-ahead prices in the Nordic bidding zones during the heatwave period in June 2026.

## Source code

The pipeline is available as standard Python source files:

- `pipeline.py` contains database ingestion, feature engineering, model training, evaluation, and plotting functions.
- `main.py` provides the command-line entry point.
- `notebook/` contains the original exploratory notebooks and a downloadable [database](https://github.com/boonsita16/nordic-trading-pipeline/blob/main/notebook/nordic_energy_market.db).

Install the dependencies and provide an ENTSO-E API key either in the `ENTSOE_API_KEY` environment variable or in `entsoe-api.txt` at the project root:

```powershell
python -m pip install -r requirements.txt
$env:ENTSOE_API_KEY = "your-api-key"
```

Run the complete pipeline:

```powershell
python main.py
```

Useful modes:

- `python main.py --mode fetch` runs ingestion only.
- `python main.py --mode analyze --no-plots` analyzes the existing database without opening Plotly windows.
- `python main.py --mode analyze` prints all prediction rows, lists the available bidding zones, and prompts for a zone to compare in the Plotly chart.

Dates, database location, token file, and the prediction plot zone can be changed with `--start`, `--end`, `--database`, `--api-token-file`, and `--selected-zone`. Supplying `--selected-zone` skips the prompt.

## Notebooks

The original notebooks include:

`01-pull-entsoe-to-database-polars-sqlite3.ipynb`

Fetches ENTSO-E data into a Pandas DataFrame, converts it to Polars, and writes it into SQLite.

`02-analyze-database.ipynb`

Loads the database, creates time-based lag and calendar features, performs an 80/20 chronological split, trains an XGBRegressor model, and compares predicted and actual prices.

The interactive plots in notebook 02 are available through [nbviewer](https://nbviewer.org/github/boonsita16/nordic-trading-pipeline/blob/main/notebook/02-analyze-database.ipynb).

## Results and Discussion

The visualization shows that Danish bidding zones had day-ahead price spikes in the evenings of June 23 and 24, diverging from other Nordic zones. The high day-ahead prices were likely due to low offshore and onshore wind generation combined with zero solar energy at night. Therefore, according to the Merit Order, expensive oil and gas plants set the clearing prices of those periods. This can be seen in the [Actual Electricity generation](https://transparency.entsoe.eu/generation/actual/perType/generation?appState=%7B%22sa%22%3A%5B%22BZN%7C10YDK-2--------M%22%5D%2C%22st%22%3A%22BZN%22%2C%22mm%22%3Atrue%2C%22ma%22%3Afalse%2C%22sp%22%3A%22HALF%22%2C%22dt%22%3A%22CHART%22%2C%22df%22%3A%5B%222026-06-20%22%2C%222026-06-26%22%5D%2C%22tz%22%3A%22CET%22%2C%22ii%22%3Anull%2C%22ps%22%3Anull%7D) where it had higher share of fossil oil and gas, compared to other periods of time.

The trained ML model includes these high-price periods in the Danish bidding zones and therefore produces large maximum overestimations in the evenings in both zones. The model statistics also show a high mean-absolut-error in the Swedish SE4 zone, meaning that it routinely overestimates/underestimates the day-ahead prices. This can come from a high volatility (though below the Danish) of the train dataset and needs further calibration.  

## Future Development

1. Improve the accuracy of the model by
   -  Increasing the historical data range to at least 2 years to capture the seasonal trends.
   -  Including weather forecast data to capture changes in renewable energy production and changes in energy consumption.
2. Turn the price prediction into day-ahead and intraday power trading decisions by
   -  Calculating fair value with the most recent weather data
   -  Considering potential power production in nearby bidding zones and the grid availability 
