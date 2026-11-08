import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_baselines import compute_metrics, compute_weights
from portfolio import COST_RATE, LOOKBACK, load_data, optimize, ridge_forecasts, run_backtest


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

    predictions = ridge_forecasts(features, targets, valid, alpha, window)

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

    first = exec_pos[valid[0]]
    dates = returns.index
    metrics, navs, allocation_rows = {}, {}, []
    for strategy in strategies:
        performance, allocation = run_backtest(
            returns, signal_pos, exec_pos, signals,
            lambda i, previous: target(i, previous, strategy), strategy, TEST_START,
        )
        allocation_rows.append(allocation)
        metrics[strategy] = compute_metrics(performance)
        navs[strategy] = performance["nav"]

    allocations = pd.concat(allocation_rows, ignore_index=True)
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