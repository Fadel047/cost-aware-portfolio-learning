
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_baselines import compute_metrics, compute_weights


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

TRAIN_MONTHS = 60
LOOKBACK = 756
RISK_AVERSION = 10.0
COST_RATE = 0.001

STRATEGIES = [
    "equal_weight",
    "inverse_volatility",
    "minimum_variance",
    "ridge_mvo",
]


def load_returns():
    returns = pd.read_csv(
        ROOT / "data" / "daily_returns.csv",
        index_col="Date",
        parse_dates=True,
    )

    if (
        returns.empty
        or returns.index.has_duplicates
        or not returns.index.is_monotonic_increasing
        or not np.isfinite(returns.to_numpy()).all()
        or (returns <= -1).any().any()
    ):
        raise ValueError("Invalid daily returns.")

    return returns


def build_features(returns):
    periods = returns.index.to_period("M")
    monthly = (1 + returns).groupby(periods).prod() - 1
    month_ends = returns.groupby(periods).tail(1).index

    log_monthly = np.log1p(monthly)

    factors = [
        np.expm1(log_monthly.rolling(k).sum()).to_numpy()
        for k in [1, 3, 6, 12]
    ]

    factors.append(
        returns.rolling(63).std().loc[month_ends].to_numpy()
    )

    features = np.stack(factors, axis=-1)

    asset_ids = np.broadcast_to(
        np.eye(returns.shape[1]),
        (*features.shape[:2], returns.shape[1]),
    )

    features = np.concatenate([features, asset_ids], axis=-1)
    targets = monthly.shift(-1).to_numpy()

    valid = np.isfinite(features).all(axis=(1, 2))

    if valid.sum() < TRAIN_MONTHS + 2:
        raise ValueError("Insufficient feature history.")

    start = np.flatnonzero(valid)[0]

    if not valid[start:].all():
        raise ValueError("Missing feature observations.")

    return month_ends[start:], features[start:], targets[start:]


def optimize_portfolio(expected_returns, history):
    covariance = LedoitWolf().fit(history.to_numpy()).covariance_ * 21
    n = len(expected_returns)

    def objective(w):
        return (
            0.5 * RISK_AVERSION * (w @ covariance @ w)
            - expected_returns @ w
        )

    def gradient(w):
        return RISK_AVERSION * covariance @ w - expected_returns

    result = minimize(
        objective,
        np.ones(n) / n,
        jac=gradient,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{
            "type": "eq",
            "fun": lambda w: w.sum() - 1,
            "jac": lambda w: np.ones(n),
        }],
        options={"ftol": 1e-10, "maxiter": 500},
    )

    if not result.success:
        raise RuntimeError(result.message)

    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def build_targets(returns):
    month_ends, features, targets = build_features(returns)

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )

    orders = {}
    forecast_rows = []
    weight_rows = []

    for i in range(TRAIN_MONTHS, len(month_ends) - 1):
        date = month_ends[i]

        x_train = features[i - TRAIN_MONTHS:i]
        y_train = targets[i - TRAIN_MONTHS:i]

        if not np.isfinite(y_train).all():
            raise ValueError("Missing training targets.")

        model.fit(
            x_train.reshape(-1, x_train.shape[-1]),
            y_train.reshape(-1),
        )

        prediction = model.predict(features[i])

        history = returns.loc[:date].tail(LOOKBACK)

        if len(history) != LOOKBACK:
            raise ValueError("Insufficient covariance history.")

        weights = {
            strategy: compute_weights(history, strategy)
            for strategy in STRATEGIES[:-1]
        }

        weights["ridge_mvo"] = optimize_portfolio(
            prediction, history
        )

        next_position = returns.index.searchsorted(
            date, side="right"
        )

        trade_date = returns.index[next_position]
        orders[trade_date] = weights

        for strategy, allocation in weights.items():
            weight_rows.append({
                "Date": trade_date,
                "strategy": strategy,
                **dict(zip(returns.columns, allocation)),
            })

        for j, ticker in enumerate(returns.columns):
            forecast_rows.append({
                "signal_date": date,
                "ticker": ticker,
                "prediction": prediction[j],
                "realized": (
                    targets[i, j]
                    if i < len(month_ends) - 2
                    else np.nan
                ),
            })

    return orders, pd.DataFrame(forecast_rows), pd.DataFrame(weight_rows)


def run_backtest(returns, orders):
    positions = {name: None for name in STRATEGIES}
    navs = {name: 1.0 for name in STRATEGIES}
    records = {name: [] for name in STRATEGIES}

    first_date = min(orders)

    for date, row in returns.loc[first_date:].iterrows():
        asset_returns = row.to_numpy()
        today_orders = orders.get(date)

        for name in STRATEGIES:
            previous = positions[name]

            if previous is None:
                gross_return = 0.0
                pre_trade = None
            else:
                gross_return = float(previous @ asset_returns)
                pre_trade = (
                    previous * (1 + asset_returns)
                    / (1 + gross_return)
                )

            turnover = 0.0

            if today_orders is not None:
                target = today_orders[name]

                turnover = (
                    1.0 if pre_trade is None
                    else float(np.abs(target - pre_trade).sum())
                )

                positions[name] = target
            else:
                positions[name] = pre_trade

            cost = COST_RATE * turnover
            net_return = (1 + gross_return) * (1 - cost) - 1

            navs[name] *= 1 + net_return

            records[name].append((
                date,
                net_return,
                navs[name],
                turnover,
                cost,
            ))

    columns = ["Date", "net_return", "nav", "turnover", "cost_rate"]

    return {
        name: pd.DataFrame(rows, columns=columns).set_index("Date")
        for name, rows in records.items()
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_returns()
    orders, forecasts, weights = build_targets(returns)
    performance = run_backtest(returns, orders)

    summary = pd.DataFrame({
        name: compute_metrics(df)
        for name, df in performance.items()
    }).T

    nav = pd.DataFrame({
        name: df["nav"]
        for name, df in performance.items()
    })

    net_returns = pd.DataFrame({
        name: df["net_return"]
        for name, df in performance.items()
    })

    summary.to_csv(RESULTS_DIR / "pto_summary.csv")
    nav.to_csv(RESULTS_DIR / "pto_nav.csv")
    net_returns.to_csv(RESULTS_DIR / "pto_returns.csv")
    forecasts.to_csv(RESULTS_DIR / "pto_forecasts.csv", index=False)
    weights.to_csv(RESULTS_DIR / "pto_weights.csv", index=False)

    ax = nav.plot(figsize=(11, 6), linewidth=1.5)
    ax.set_title("Predict-then-Optimize vs Baselines")
    ax.set_ylabel("Net Portfolio NAV")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pto_nav.png", dpi=150)
    plt.close()

    print(f"Out-of-sample: {nav.index[0].date()} to {nav.index[-1].date()}")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
