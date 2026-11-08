import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_baselines import compute_metrics, compute_weights
from portfolio import LOOKBACK, load_data, optimize, ridge_forecasts, run_backtest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]
WINDOWS = [24, 36, 48, 60, 72]
GAMMAS = [2, 5, 10, 20, 40, 80]
LAMBDAS = [0, 0.0005, 0.001, 0.003, 0.01, 0.03, 0.1]
N_TRIALS = 1260
SEED = 42



def main():
    RESULTS.mkdir(exist_ok=True)
    returns, features, targets, signal_pos, exec_pos, signals, cov = load_data()
    forecast_cache = {}

    def forecasts(alpha, window, required_signals):
        key = (alpha, window)
        predictions = forecast_cache.setdefault(key, {})
        missing = [i for i in required_signals if i not in predictions]
        predictions.update(ridge_forecasts(features, targets, missing, alpha, window))
        return predictions

    baseline_cache = {}

    def target(i, current, config, strategy, preds):
        if strategy == "ridge_mvo":
            alpha, window, gamma, penalty = config
            return optimize(preds[i], cov[i], gamma, penalty, current)
        key = (strategy, i)
        if key not in baseline_cache:
            history = returns.iloc[signal_pos[i] - LOOKBACK + 1:signal_pos[i] + 1]
            baseline_cache[key] = compute_weights(history, strategy)
        return baseline_cache[key]

    def backtest(start, end, config=None, strategy="ridge_mvo"):
        valid = [i for i in signals if start <= returns.index[exec_pos[i]] <= end]
        preds = forecasts(config[0], config[1], valid) if config is not None else None
        return run_backtest(
            returns, signal_pos, exec_pos, signals,
            lambda i, current: target(i, current, config, strategy, preds),
            strategy, start, end,
        )

    grid = list(product(ALPHAS, WINDOWS, GAMMAS, LAMBDAS))
    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(grid), size=min(N_TRIALS, len(grid)), replace=False)
    configs = [grid[int(j)] for j in picks]
    default = (10, 60, 10, 0)
    if default not in configs:
        configs.append(default)

    results = []
    for config in configs:
        performance, _ = backtest(pd.Timestamp("2018-01-01"),
                                  pd.Timestamp("2021-12-31"), config)
        metrics = compute_metrics(performance)
        results.append({
            "alpha": config[0], "window": config[1],
            "gamma": config[2], "lambda": config[3],
            **metrics,
        })

    ranking = pd.DataFrame(results).sort_values("sharpe_rf0", ascending=False)
    ranking.to_csv(RESULTS / "pto_tuning_validation.csv", index=False)
    best = ranking.iloc[0]
    config = (float(best["alpha"]), int(best["window"]),
              float(best["gamma"]), float(best["lambda"]))

    config_dir = ROOT / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "pto_selected.json").write_text(json.dumps({
        "alpha": config[0], "window": config[1],
        "gamma": config[2], "penalty": config[3],
    }, indent=2) + "\n")

    performances, weights = {}, []
    for strategy in ("equal_weight", "inverse_volatility", "minimum_variance", "ridge_mvo"):
        perf, w = backtest(pd.Timestamp("2022-01-01"), returns.index[-1],
                           config if strategy == "ridge_mvo" else None, strategy)
        performances[strategy] = perf
        weights.append(w)

    summary = pd.DataFrame({
        name: compute_metrics(perf) for name, perf in performances.items()
    }).T
    nav = pd.DataFrame({name: perf["nav"] for name, perf in performances.items()})
    summary.to_csv(RESULTS / "pto_tuned_test_summary.csv")
    nav.to_csv(RESULTS / "pto_tuned_test_nav.csv")
    pd.concat(weights, ignore_index=True).to_csv(RESULTS / "pto_tuned_test_weights.csv", index=False)

    print(f"Validated {len(configs)} configurations on 2018-2021")
    print("\nTop 5 validation configurations:")
    print(ranking[["alpha", "window", "gamma", "lambda", "sharpe_rf0", "annual_return",
                   "annual_traded_notional"]].head().round(4).to_string(index=False))
    print(f"\nSelected: alpha={config[0]}, window={config[1]}, "
          f"gamma={config[2]}, lambda={config[3]}")
    print(f"\nTest: {nav.index[0].date()} to {nav.index[-1].date()}")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
