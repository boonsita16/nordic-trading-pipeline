"""Reusable ingestion, feature engineering, and modelling steps."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from entsoe import EntsoePandasClient
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


DEFAULT_ZONES = (
    "DK_1", "DK_2", "SE_1", "SE_2", "SE_3", "SE_4",
    "NO_1", "NO_2", "NO_3", "NO_4", "NO_5",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for the Nordic electricity price pipeline."""

    database_path: Path = Path("nordic_energy_market.db")
    api_token_path: Path = Path("entsoe-api.txt")
    start: str = "2026-06-14"
    end: str = "2026-06-29"
    zones: tuple[str, ...] = DEFAULT_ZONES
    train_fraction: float = 0.8
    selected_zone: str = "SE_4"


def _connection(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(database_path)


def _get_api_token(token_path: Path) -> str:
    token = os.getenv("ENTSOE_API_KEY")
    if token:
        return token.strip()
    if not token_path.exists():
        raise FileNotFoundError(
            f"ENTSOE API token not found at {token_path}. "
            "Set ENTSOE_API_KEY or provide --api-token-file."
        )
    return token_path.read_text(encoding="utf-8").strip()


def initialise_database(database_path: Path) -> None:
    """Create the price table when it does not exist."""
    with _connection(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS master_day_ahead_prices (
                timestamp TEXT NOT NULL,
                bidding_zone TEXT NOT NULL,
                price_eur_mwh REAL NOT NULL,
                PRIMARY KEY (timestamp, bidding_zone)
            )
            """
        )


def fetch_prices(config: PipelineConfig) -> int:
    """Fetch day-ahead prices and upsert them into SQLite."""
    client = EntsoePandasClient(api_key=_get_api_token(config.api_token_path))
    start = pd.Timestamp(config.start, tz="Europe/Berlin")
    end = pd.Timestamp(config.end, tz="Europe/Berlin")
    initialise_database(config.database_path)
    rows_written = 0

    with _connection(config.database_path) as connection:
        for zone in config.zones:
            print(f"Fetching data for {zone}...")
            prices = client.query_day_ahead_prices(zone, start=start, end=end)
            prices = prices.reset_index()
            prices.columns = ["timestamp", "price_eur_mwh"]
            prices["timestamp"] = pd.to_datetime(prices["timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            rows = [
                (timestamp, zone, float(price))
                for timestamp, price in prices.itertuples(index=False, name=None)
            ]
            connection.executemany(
                """
                INSERT OR REPLACE INTO master_day_ahead_prices
                    (timestamp, bidding_zone, price_eur_mwh)
                VALUES (?, ?, ?)
                """,
                rows,
            )
            rows_written += len(rows)
            print(f"Wrote {len(rows)} rows for {zone}.")

    return rows_written


def load_prices(database_path: Path) -> pl.DataFrame:
    """Load raw prices from SQLite and parse timestamps."""
    query = """
        SELECT timestamp, bidding_zone, price_eur_mwh AS target_price
        FROM master_day_ahead_prices
        ORDER BY bidding_zone, timestamp
    """
    with _connection(database_path) as connection:
        raw = pd.read_sql_query(query, connection)
    return (
        pl.from_pandas(raw)
        .with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S"))
        .sort(["bidding_zone", "timestamp"])
    )


def create_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add time-based lag and calendar features."""
    result = df
    for suffix, hours in (("1h", 1), ("24h", 24), ("1w", 168)):
        lagged = (
            df.with_columns(
                (pl.col("timestamp") + pl.duration(hours=hours)).alias("timestamp")
            )
            .select(
                [
                    "timestamp",
                    "bidding_zone",
                    pl.col("target_price").alias(f"price_lag_{suffix}"),
                ]
            )
        )
        result = result.join(lagged, on=["timestamp", "bidding_zone"], how="left")

    result = result.with_columns(
        [
            pl.col("price_lag_1h").forward_fill().over("bidding_zone"),
            pl.col("price_lag_24h").forward_fill().over("bidding_zone"),
            pl.col("price_lag_1w").forward_fill().over("bidding_zone"),
        ]
    ).filter(pl.col("price_lag_1w").is_not_null())

    return result.with_columns(
        [
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp")
            .dt.weekday()
            .is_in([6, 7])
            .cast(pl.Int8)
            .alias("is_weekend"),
        ]
    )


def market_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Summarise average price and volatility by bidding zone."""
    return df.group_by("bidding_zone").agg(
        [
            pl.col("target_price").mean().alias("avg_price"),
            pl.col("target_price").max().alias("max_price"),
            pl.col("target_price").std().alias("price_volatility"),
        ]
    ).sort("avg_price", descending=True)


@dataclass
class ModelResult:
    results: pl.DataFrame
    statistics: pl.DataFrame
    mae: float
    rmse: float


def train_model(df: pl.DataFrame, train_fraction: float = 0.8) -> ModelResult:
    """Train an XGBoost model using a chronological split."""
    encoded = df.to_dummies(columns=["bidding_zone"]).sort("timestamp")
    feature_columns = [
        column for column in encoded.columns if column not in {"timestamp", "target_price"}
    ]
    split_index = int(len(encoded) * train_fraction)
    train = encoded.slice(0, split_index)
    test = encoded.slice(split_index, len(encoded) - split_index)

    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train.select(feature_columns).to_numpy(),
        train["target_price"].to_numpy(),
        eval_set=[
            (test.select(feature_columns).to_numpy(), test["target_price"].to_numpy())
        ],
        verbose=False,
    )

    actual = test["target_price"].to_numpy()
    predicted = model.predict(test.select(feature_columns).to_numpy())
    results = test.with_columns(
        [
            pl.Series("actual_price", actual),
            pl.Series("predicted_price", predicted),
        ]
    )
    zone_columns = [column for column in results.columns if column.startswith("bidding_zone_")]
    zone_mapping = {
        index: column.removeprefix("bidding_zone_")
        for index, column in enumerate(zone_columns)
    }
    results = results.with_columns(
        pl.concat_list(zone_columns)
        .list.arg_max()
        .replace_strict(zone_mapping, default=None)
        .alias("bidding_zone")
    )
    statistics = (
        results.group_by("bidding_zone")
        .agg(
            [
                pl.col("actual_price").mean().alias("avg_actual_price"),
                pl.col("predicted_price").mean().alias("avg_predicted_price"),
                (pl.col("predicted_price") - pl.col("actual_price"))
                .mean()
                .alias("mean_bias_error"),
                (pl.col("predicted_price") - pl.col("actual_price"))
                .abs()
                .mean()
                .alias("mean_absolute_error"),
                (pl.col("predicted_price") - pl.col("actual_price"))
                .max()
                .alias("max_overestimation"),
            ]
        )
        .sort("mean_absolute_error", descending=True)
    )
    return ModelResult(
        results=results,
        statistics=statistics,
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
    )


def create_price_plot(df: pl.DataFrame):
    """Create an interactive price-by-zone Plotly figure."""
    import plotly.express as px

    return px.line(
        df.to_pandas(),
        x="timestamp",
        y="target_price",
        color="bidding_zone",
        title="Nordic Electricity Prices",
        labels={"timestamp": "Date & Time", "target_price": "Price (EUR/MWh)"},
    ).update_layout(template="plotly_white", hovermode="closest")


def create_prediction_plot(results: pl.DataFrame, zone: str):
    """Create an actual-versus-predicted plot for one zone."""
    import plotly.graph_objects as go

    zone_results = results.filter(pl.col("bidding_zone") == zone).sort("timestamp").to_pandas()
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=zone_results["timestamp"], y=zone_results["actual_price"], name="Actual"))
    figure.add_trace(
        go.Scatter(
            x=zone_results["timestamp"],
            y=zone_results["predicted_price"],
            name="Predicted",
            line={"dash": "dash"},
        )
    )
    return figure.update_layout(
        title=f"Actual vs predicted price: {zone}",
        xaxis_title="Timeline",
        yaxis_title="Price (EUR/MWh)",
        template="plotly_white",
        hovermode="x unified",
    )