
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

LOOKBACK = 756
COST_BPS = 10
COST_RATE = COST_BPS / 10000
TRADING_DAYS = 252

STRATEGIES = [
    "equal_weight",
    "inverse_volatility",
    "minimum_variance",
]


def load_returns():
    returns = pd.read_csv(
        DATA_DIR / "daily_returns.csv",
        index_col="Date",
        parse_dates=True,
    )

    if len(returns) <= LOOKBACK:
        raise ValueError("Insufficient historical data.")

    if returns.index.has_duplicates:
        raise ValueError("Duplicate dates.")

    if not returns.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Invalid returns.")

    if (returns <= -1).any().any():
        raise ValueError("Returns below or equal to -100%.")

    return returns


def compute_weights(history, strategy):
    n_assets = history.shape[1]

    if strategy == "equal_weight":
        return np.ones(n_assets) / n_assets

    if strategy == "inverse_volatility":
        volatility = history.std().to_numpy()

        if np.any(volatility <= 0):
            raise ValueError("Non-positive volatility.")

        inverse_vol = 1 / volatility
        return inverse_vol / inverse_vol.sum()

    if strategy == "minimum_variance":
        covariance = LedoitWolf().fit(
            history.to_numpy()
        ).covariance_

        def objective(w):
            return float(w @ covariance @ w) * 10000

        result = minimize(
            objective,
            x0=np.ones(n_assets) / n_assets,
            method="SLSQP",
            bounds=[(0, 1)] * n_assets,
            constraints=[{
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1,
            }],
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        if not result.success:
            raise RuntimeError(result.message)

        if np.min(result.x) < -1e-6:
            raise ValueError("Negative optimized weight.")

        weights = np.clip(result.x, 0, 1)
        return weights / weights.sum()

    raise ValueError(f"Unknown strategy: {strategy}")


def run_backtest(returns, strategy):
    dates = returns.index
    month_ends = set(
        returns.groupby(returns.index.to_period("M"))
        .tail(1)
        .index
    )

    first_position = LOOKBACK - 1
    initial_date = dates[first_position]

    weights = compute_weights(
        returns.iloc[:LOOKBACK], strategy
    )

    # Investment at the first backtest close
    nav = 1 - COST_RATE

    records = [{
        "Date": initial_date,
        "net_return": -COST_RATE,
        "nav": nav,
        "turnover": 1.0,
        "cost_rate": COST_RATE,
    }]

    allocations = [{
        "Date": initial_date,
        "Strategy": strategy,
        **dict(zip(returns.columns, weights)),
    }]

    for i in range(LOOKBACK, len(returns)):
        date = dates[i]
        asset_returns = returns.iloc[i].to_numpy()

        # Today's return uses yesterday's post-trade weights
        gross_return = float(weights @ asset_returns)

        if gross_return <= -1:
            raise ValueError("Portfolio lost all its value.")

        # Drift in portfolio weights before rebalancing
        pre_trade = (
            weights * (1 + asset_returns)
            / (1 + gross_return)
        )

        turnover = 0.0
        cost_rate = 0.0
        weights = pre_trade

        # Trade at month-end close, using information
        # available up to and including this close
        if date in month_ends and i < len(returns) - 1:
            history = returns.iloc[i - LOOKBACK + 1:i + 1]

            target = compute_weights(history, strategy)

            turnover = float(np.abs(target - pre_trade).sum())
            cost_rate = COST_RATE * turnover
            weights = target

            allocations.append({
                "Date": date,
                "Strategy": strategy,
                **dict(zip(returns.columns, weights)),
            })

        net_return = (
            (1 + gross_return) * (1 - cost_rate) - 1
        )

        nav *= 1 + net_return

        records.append({
            "Date": date,
            "net_return": net_return,
            "nav": nav,
            "turnover": turnover,
            "cost_rate": cost_rate,
        })

    performance = pd.DataFrame(records).set_index("Date")
    allocations = pd.DataFrame(allocations)

    return performance, allocations


def compute_metrics(performance):
    returns = performance["net_return"]
    nav = performance["nav"]

    annual_return = nav.iloc[-1] ** (
        TRADING_DAYS / len(returns)
    ) - 1

    annual_vol = returns.std() * np.sqrt(TRADING_DAYS)

    sharpe = (
        returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)
        if returns.std() > 0 else np.nan
    )

    wealth = np.r_[1.0, nav.to_numpy()]
    max_drawdown = (
        wealth / np.maximum.accumulate(wealth) - 1
    ).min()

    annual_turnover = (
        performance["turnover"].sum()
        * TRADING_DAYS / len(returns)
    )

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_drawdown,
        "annual_traded_notional": annual_turnover,
        "total_cost_rate_sum": performance["cost_rate"].sum(),
        "final_nav": nav.iloc[-1],
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_returns()

    nav_results = {}
    return_results = {}
    metrics = {}
    all_allocations = []

    for strategy in STRATEGIES:
        performance, allocations = run_backtest(
            returns, strategy
        )

        nav_results[strategy] = performance["nav"]
        return_results[strategy] = performance["net_return"]
        metrics[strategy] = compute_metrics(performance)
        all_allocations.append(allocations)

    nav_df = pd.DataFrame(nav_results)
    returns_df = pd.DataFrame(return_results)
    summary = pd.DataFrame(metrics).T

    nav_df.to_csv(RESULTS_DIR / "baseline_nav.csv")
    returns_df.to_csv(RESULTS_DIR / "baseline_returns.csv")
    summary.to_csv(RESULTS_DIR / "baseline_summary.csv")

    pd.concat(
        all_allocations, ignore_index=True
    ).to_csv(
        RESULTS_DIR / "baseline_weights.csv",
        index=False,
    )

    ax = nav_df.plot(figsize=(11, 6), linewidth=1.5)
    ax.set_title("Portfolio Baselines - Net of Transaction Costs")
    ax.set_ylabel("Portfolio NAV")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "baseline_nav.png", dpi=150
    )
    plt.close()

    print("\nBacktest period:")
    print(f"{nav_df.index.min().date()} to {nav_df.index.max().date()}")

    print("\nBaseline Performance:")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()