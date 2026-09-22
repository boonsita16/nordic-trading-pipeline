"""Command-line entry point for the Nordic trading pipeline."""

import argparse
from pathlib import Path

import polars as pl

from pipeline import (
    PipelineConfig,
    create_features,
    create_prediction_plot,
    create_price_plot,
    fetch_prices,
    load_prices,
    market_summary,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "fetch", "analyze"), default="all")
    parser.add_argument("--database", type=Path, default=Path("nordic_energy_market.db"))
    parser.add_argument("--api-token-file", type=Path, default=Path("entsoe-api.txt"))
    parser.add_argument("--start", default="2026-06-14")
    parser.add_argument("--end", default="2026-06-29")
    parser.add_argument("--selected-zone", help="Zone to plot; prompts when omitted")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        database_path=args.database,
        api_token_path=args.api_token_file,
        start=args.start,
        end=args.end,
        selected_zone=args.selected_zone,
    )

    if args.mode in {"all", "fetch"}:
        print(f"Stored {fetch_prices(config)} rows.")
    if args.mode in {"all", "analyze"}:
        print("Loading prices...")
        features = create_features(load_prices(config.database_path))
        print("Market summary:")
        print(market_summary(features))
        model_result = train_model(features, config.train_fraction)
        print(f"MAE: {model_result.mae:.2f} EUR/MWh")
        print(f"RMSE: {model_result.rmse:.2f} EUR/MWh")
        with pl.Config(tbl_rows=-1):
            print("Model statistics:")
            print(model_result.statistics)
        if not args.no_plots:
            available_zones = sorted(model_result.results["bidding_zone"].unique().to_list())
            selected_zone = args.selected_zone
            if selected_zone is None:
                print(f"Available bidding zones: {', '.join(available_zones)}")
                selected_zone = input("Enter the bidding zone to plot: ").strip().upper()
            if selected_zone not in available_zones:
                raise ValueError(
                    f"Unknown bidding zone '{selected_zone}'. "
                    f"Choose one of: {', '.join(available_zones)}"
                )
            create_price_plot(features).show()
            create_prediction_plot(model_result.results, selected_zone).show()


if __name__ == "__main__":
    main()