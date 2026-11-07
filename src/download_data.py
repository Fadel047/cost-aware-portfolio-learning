
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


# Configuration
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "IEF", "GLD"]
START_DATE = "2010-01-01"
END_DATE = datetime.now(ZoneInfo("America/New_York")).date().isoformat()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download_data():

    data = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        multi_level_index=True,
        progress=False,
        threads=True,
    )

    if data is None or data.empty:
        raise ValueError("No data downloaded.")

    prices = pd.DataFrame()
    volumes = pd.DataFrame()

    for ticker in TICKERS:
        if ticker not in data.columns.get_level_values(0):
            raise ValueError(f"Missing ticker: {ticker}")

        prices[ticker] = data[ticker]["Close"]
        volumes[ticker] = data[ticker]["Volume"]

    prices = prices.sort_index()
    volumes = volumes.sort_index()

    prices.index.name = "Date"
    volumes.index.name = "Date"

    return prices, volumes


def validate_data(prices, volumes):
    if prices.empty or volumes.empty:
        raise ValueError("Empty dataset.")

    if not prices.index.equals(volumes.index):
        raise ValueError("Price and volume dates do not match.")

    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates detected.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted.")

    if prices.isna().any().any():
        missing = prices.isna().sum()
        raise ValueError(f"Missing prices:\n{missing[missing > 0]}")

    if volumes.isna().any().any():
        missing = volumes.isna().sum()
        raise ValueError(f"Missing volumes:\n{missing[missing > 0]}")

    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Non-finite prices detected.")

    if (prices <= 0).any().any():
        raise ValueError("Non-positive prices detected.")

    if not np.isfinite(volumes.to_numpy()).all():
        raise ValueError("Non-finite volumes detected.")

    if (volumes < 0).any().any():
        raise ValueError("Negative volumes detected.")

    report = pd.DataFrame({
        "observations": prices.count(),
        "missing_prices": prices.isna().sum(),
        "missing_volumes": volumes.isna().sum(),
        "zero_volume_days": (volumes == 0).sum(),
        "first_date": [prices.index.min().date()] * len(TICKERS),
        "last_date": [prices.index.max().date()] * len(TICKERS),
    })

    report.index.name = "Ticker"

    return report


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    prices, volumes = download_data()
    report = validate_data(prices, volumes)

    prices.to_csv(DATA_DIR / "prices_adjusted.csv")
    volumes.to_csv(DATA_DIR / "volumes.csv")
    report.to_csv(DATA_DIR / "data_quality.csv")

    print("\nData quality report:")
    print(report.to_string())

    print(f"\nDataset shape: {prices.shape}")
    print(f"Period: {prices.index.min().date()} to {prices.index.max().date()}")

if __name__ == "__main__":
    main()
