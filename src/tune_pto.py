import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_baselines import compute_metrics, compute_weights


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]
WINDOWS = [24, 36, 48, 60, 72]
GAMMAS = [2, 5, 10, 20, 40, 80]
LAMBDAS = [0, 0.0005, 0.001, 0.003, 0.01, 0.03, 0.1]
N_TRIALS = 1260
SEED = 42
COST_RATE = 0.001
LOOKBACK = 756


def load_data():
    returns = pd.read_csv(DATA / "daily_returns.csv", index_col="Date", parse_dates=True)
    if (returns.empty or returns.isna().any().any()
            or not returns.index.is_monotonic_increasing
            or returns.index.has_duplicates
            or not np.isfinite(returns.to_numpy()).all()
            or (returns <= -1).any().any()):
        raise ValueError("Invalid returns")

    periods = returns.index.to_period("M")
    monthly = (1 + returns).groupby(periods).prod() - 1
    month_ends = returns.groupby(periods).tail(1).index
    log_monthly = np.log1p(monthly)

    features = np.stack([
        *[np.expm1(log_monthly.rolling(k).sum()).to_numpy() for k in (1, 3, 6, 12)],
        returns.rolling(63).std().loc[month_ends].to_numpy(),
    ], axis=-1)
    ids = np.broadcast_to(np.eye(returns.shape[1]),
                          (*features.shape[:2], returns.shape[1]))
    features = np.concatenate([features, ids], axis=-1)

    exec_pos = returns.index.searchsorted(month_ends, side="right")
    growth = (1 + returns).cumprod().to_numpy()
    targets = np.full(features.shape[:2], np.nan)
    for i in range(len(month_ends) - 2):
        targets[i] = growth[exec_pos[i + 1]] / growth[exec_pos[i]] - 1

    signal_pos = returns.index.get_indexer(month_ends)
    signals = [i for i in range(len(month_ends) - 1)
               if exec_pos[i] < len(returns)
               and returns.index[exec_pos[i]] >= pd.Timestamp("2018-01-01")]

    if not signals or not np.isfinite(features[signals]).all():
        raise ValueError("Missing features or insufficient history")

    covariance = {
        i: LedoitWolf().fit(returns.iloc[signal_pos[i] - LOOKBACK + 1:
                                         signal_pos[i] + 1].to_numpy()).covariance_ * 21
        for i in signals
    }
    if any(signal_pos[i] + 1 < LOOKBACK for i in signals):
        raise ValueError("Insufficient covariance history")

    return returns, features, targets, signal_pos, exec_pos, signals, covariance


def optimize(mu, cov, gamma, penalty, previous):
    n = len(mu)
    x0 = np.full(n, 1 / n) if previous is None else previous

    def objective(w):
        diff = w - x0
        value = 0.5 * gamma * (w @ cov @ w) - mu @ w
        if previous is not None:
            value += penalty * np.sqrt(diff * diff + 1e-6).sum()
        return value

    def gradient(w):
        result = gamma * cov @ w - mu
        if previous is not None:
            diff = w - x0
            result += penalty * diff / np.sqrt(diff * diff + 1e-6)
        return result

    result = minimize(
        objective, x0, jac=gradient, method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                      "jac": lambda w: np.ones(n)}],
        options={"ftol": 1e-9, "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    w = np.clip(result.x, 0, 1)
    return w / w.sum()


def main():
    RESULTS.mkdir(exist_ok=True)
    returns, features, targets, signal_pos, exec_pos, signals, cov = load_data()
    forecast_cache = {}

    def forecasts(alpha, window, required_signals):
        key = (alpha, window)
        predictions = forecast_cache.setdefault(key, {})
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        for i in required_signals:
            if i in predictions:
                continue
            start, stop = i - 1 - window, i - 1
            if start < 0:
                raise ValueError("Insufficient training history")
            x = features[start:stop]
            y = targets[start:stop]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Invalid training sample")
            model.fit(x.reshape(-1, x.shape[-1]), y.ravel())
            predictions[i] = model.predict(features[i])
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
        if not valid:
            raise ValueError("No trades in evaluation period")
        trade_map = {exec_pos[i]: i for i in valid}
        signal_map = {signal_pos[i]: i for i in valid[1:]}
        preds = forecasts(config[0], config[1], valid) if config is not None else None

        first, last = exec_pos[valid[0]], returns.index.searchsorted(end, side="right")
        w = None
        pending = target(valid[0], None, config, strategy, preds)
        nav = 1.0
        rows, weights = [], []

        for t in range(first, last):
            r = returns.iloc[t].to_numpy()
            gross = 0.0 if w is None else float(w @ r)
            pre = None if w is None else w * (1 + r) / (1 + gross)
            turnover = 0.0
            if t in trade_map:
                turnover = 1.0 if pre is None else float(np.abs(pending - pre).sum())
                w = pending
                weights.append({"Date": returns.index[t], "strategy": strategy,
                                **dict(zip(returns.columns, w))})
            else:
                w = pre
            cost = COST_RATE * turnover
            net_return = (1 + gross) * (1 - cost) - 1
            nav *= 1 + net_return
            rows.append((returns.index[t], net_return, nav, turnover, cost))
            if t in signal_map:
                pending = target(signal_map[t], w, config, strategy, preds)

        performance = pd.DataFrame(rows, columns=[
            "Date", "net_return", "nav", "turnover", "cost_rate"
        ]).set_index("Date")
        return performance, pd.DataFrame(weights)

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
