
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

TRADING_DAYS = 252


def load_prices():
    prices = pd.read_csv(
        DATA_DIR / "prices_adjusted.csv",
        index_col="Date",
        parse_dates=True,
    )

    if prices.empty or prices.isna().any().any():
        raise ValueError("Missing or empty price data.")

    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Invalid date index.")

    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Non-finite prices.")

    if (prices <= 0).any().any():
        raise ValueError("Non-positive prices.")

    return prices


def compute_returns(prices):
    returns = prices.pct_change(fill_method=None).iloc[1:]

    if returns.isna().any().any():
        raise ValueError("Missing returns.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Non-finite returns.")

    return returns


def compute_statistics(prices, returns):
    years = (prices.index[-1] - prices.index[0]).days / 365.25

    annual_return = (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS)

    drawdowns = prices.div(prices.cummax()) - 1

    stats = pd.DataFrame({
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "max_drawdown": drawdowns.min(),
        "worst_daily_return": returns.min(),
        "best_daily_return": returns.max(),
        "extreme_days_10pct": (returns.abs() > 0.10).sum(),
    })

    stats.index.name = "Ticker"
    return stats


def plot_prices(prices):
    normalized = prices.div(prices.iloc[0])

    ax = normalized.plot(figsize=(11, 6), linewidth=1.5)
    ax.set_yscale("log")
    ax.set_title("ETF Adjusted Prices - Growth of $1")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth (Log Scale)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "normalized_prices.png", dpi=150)
    plt.close()


def plot_correlations(corr):
    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(range(len(corr.columns)), corr.columns)
    ax.set_yticks(range(len(corr.index)), corr.index)

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax)
    ax.set_title("Daily Return Correlations")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlations.png", dpi=150)
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_prices()
    returns = compute_returns(prices)

    stats = compute_statistics(prices, returns)
    corr = returns.corr()

    returns.to_csv(DATA_DIR / "daily_returns.csv")
    stats.to_csv(RESULTS_DIR / "summary_statistics.csv")
    corr.to_csv(RESULTS_DIR / "correlations.csv")

    plot_prices(prices)
    plot_correlations(corr)

    print("\nSummary Statistics:")
    print(stats.round(4).to_string())

    print("\nCorrelation Matrix:")
    print(corr.round(2).to_string())

    print(f"\nDaily returns shape: {returns.shape}")


if __name__ == "__main__":
    main()