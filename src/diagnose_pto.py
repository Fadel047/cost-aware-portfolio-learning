import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_baselines import compute_metrics, compute_weights
from tune_pto import COST_RATE, LOOKBACK, load_data, optimize


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEST_START = pd.Timestamp("2022-01-01")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config" / "pto_selected.json").read_text())
    alpha, window = float(config["alpha"]), int(config["window"])
    gamma, penalty = float(config["gamma"]), float(config["penalty"])

    returns, features, targets, signal_pos, exec_pos, signals, covariance = load_data()
    valid = [i for i in signals if returns.index[exec_pos[i]] >= TEST_START]
    if not valid:
        raise ValueError("No test observations.")

    predictions = {}
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    for i in valid:
        start, stop = i - window - 1, i - 1
        x, y = features[start:stop], targets[start:stop]
        if start < 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Invalid training window.")
        model.fit(x.reshape(-1, x.shape[-1]), y.ravel())
        predictions[i] = model.predict(features[i])

    forecast_rows = []
    for i in valid:
        if np.isfinite(targets[i]).all():
            for ticker, predicted, realized in zip(
                returns.columns, predictions[i], targets[i]
            ):
                forecast_rows.append((returns.index[signal_pos[i]], ticker, predicted, realized))

    forecasts = pd.DataFrame(
        forecast_rows, columns=["signal_date", "ticker", "prediction", "realized"]
    )
    rank_ics = forecasts.groupby("signal_date")[["prediction", "realized"]].apply(
        lambda g: g["prediction"].corr(g["realized"], method="spearman")
    )
    mse = np.mean((forecasts["prediction"] - forecasts["realized"]) ** 2)
    zero_mse = np.mean(forecasts["realized"] ** 2)

    strategies = {
        "equal_weight": ("equal_weight", 0.0),
        "inverse_volatility": ("inverse_volatility", 0.0),
        "minimum_variance": ("minimum_variance", 0.0),
        "min_var_cost": ("minimum_variance", COST_RATE),
        "ridge_no_penalty": ("ridge", 0.0),
        "ridge_cost": ("ridge", COST_RATE),
        "ridge_tuned": ("ridge", penalty),
    }

    def target(i, previous, strategy):
        mode, lam = strategies[strategy]
        if mode == "ridge":
            return optimize(predictions[i], covariance[i], gamma, lam, previous)
        if mode == "minimum_variance" and lam > 0:
            return optimize(np.zeros(len(returns.columns)), covariance[i], gamma, lam, previous)
        history = returns.iloc[signal_pos[i] - LOOKBACK + 1:signal_pos[i] + 1]
        return compute_weights(history, mode)

    trade_map = {exec_pos[i]: i for i in valid}
    signal_map = {signal_pos[i]: i for i in valid[1:]}
    first = exec_pos[valid[0]]
    dates = returns.index

    metrics, navs, allocation_rows = {}, {}, []
    for strategy in strategies:
        weights = None
        pending = target(valid[0], None, strategy)
        nav = 1.0
        records = []

        for t in range(first, len(returns)):
            daily = returns.iloc[t].to_numpy()
            gross = 0.0 if weights is None else float(weights @ daily)
            pre_trade = None if weights is None else weights * (1 + daily) / (1 + gross)
            turnover = 0.0

            if t in trade_map:
                turnover = 1.0 if pre_trade is None else float(np.abs(pending - pre_trade).sum())
                weights = pending
                allocation_rows.append({
                    "Date": dates[t], "strategy": strategy,
                    **dict(zip(returns.columns, weights)),
                })
            else:
                weights = pre_trade

            cost = COST_RATE * turnover
            net_return = (1 + gross) * (1 - cost) - 1
            nav *= 1 + net_return
            records.append((dates[t], net_return, nav, turnover, cost))

            if t in signal_map:
                pending = target(signal_map[t], weights, strategy)

        performance = pd.DataFrame(
            records, columns=["Date", "net_return", "nav", "turnover", "cost_rate"]
        ).set_index("Date")
        metrics[strategy] = compute_metrics(performance)
        navs[strategy] = performance["nav"]

    allocations = pd.DataFrame(allocation_rows)
    summary = pd.DataFrame(metrics).T
    for strategy, group in allocations.groupby("strategy"):
        weights = group[list(returns.columns)]
        summary.loc[strategy, "avg_max_weight"] = weights.max(axis=1).mean()
        summary.loc[strategy, "avg_effective_assets"] = (1 / weights.pow(2).sum(axis=1)).mean()

    summary.to_csv(RESULTS / "pto_diagnostics_summary.csv")
    pd.DataFrame(navs).to_csv(RESULTS / "pto_diagnostics_nav.csv")
    forecasts.to_csv(RESULTS / "pto_diagnostics_forecasts.csv", index=False)
    allocations.to_csv(RESULTS / "pto_diagnostics_weights.csv", index=False)

    print(f"Selected: alpha={alpha:g}, window={window}, gamma={gamma:g}, lambda={penalty:g}")
    print(f"Test: {dates[first].date()} to {dates[-1].date()}")
    print(
        f"Forecasts: Pearson={forecasts['prediction'].corr(forecasts['realized']):.4f}, "
        f"Rank IC={rank_ics.mean():.4f}, "
        f"MSE/zero-forecast MSE={mse / zero_mse:.4f}"
    )
    print("\nStrategy diagnostics:")
    print(summary[[
        "annual_return", "annual_volatility", "sharpe_rf0", "max_drawdown",
        "annual_traded_notional", "avg_max_weight", "avg_effective_assets",
    ]].round(4).to_string())


if __name__ == "__main__":
    main()