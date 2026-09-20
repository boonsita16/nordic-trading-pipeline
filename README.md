# nordic-trading-pipeline
A Python project that fetches the day-ahead power prices into a local database for later use.

This project uses Python libraries, including Pandas, Polars, and SQLite3, to extract electricity data from ENTSO-E Transparency Platform using API. Then the data is written to a local database for analysis, visualization, and developing the XGBoost machine learning model for price prediction.

The scope of this project is the Day-ahead prices in the Nordic Bidding Zones during the heatwave period in June 2026.

## Current status
Notebooks include:

#01-pull-entsoe-to-database-polars-sqlite3.ipynb 

Fetch the data from ENTSO-E using API into Pandas dataframe. The Pandas dataframe is converted into a Polars dataframe for faster data handling.  The data is written into a local database using Python SQLite3.

#02-analyze-database.ipynb 

Extract the data from the local database into Polars dataframe to analyze the prices of the selected bidding zones and period. Lad features are added for comparing the price in the previous hour, day, and week. Add the columns that indicate hour of the day, weekend and weekdays, and month to capture market cyclicality. Then, the data is split into two datasets: 80% is train dataset and last 20% is test dataset, by Chronological Time Split since it is a time-series data. The XGBRegressor model is trained, and the price of the last 20% period is predicted by the model. The predicted and actual data are compared.

The interactive plots on notebook #02-analyze-database can be access via [nbviewer](https://nbviewer.org/github/boonsita16/nordic-trading-pipeline/blob/main/notebook/02-analyze-database.ipynb)

## Discuss result and propose development
The visualization shows that Denmark bidding zones had price spikes between June 23-24, diverging from other zones which indicates the transmission bottleneck due to low wind power production.

The train data, however, includes those heatwave period, and it results in the highest maximum overestimation of 112.6 EUR/MWh in DK2 Zone. 

## Future development
1. Develop the codes into End-to-End Production Pipeline by refactoring the notebooks into standard source code and a main.py

2. Improve the accuracy of the model by including weather data to capture possibilities of 1) high demand due to heatwave and 2) low supply due to low wind.
