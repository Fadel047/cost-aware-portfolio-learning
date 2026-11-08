from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COST_RATE = 0.001
LOOKBACK = 756


def load_data(data_dir=DATA):
    returns = pd.read_csv(data_dir / "daily_returns.csv", index_col="Date", parse_dates=True)
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


def optimize(mu, cov, gamma, penalty, previous, tolerance=1e-9):
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
        options={"ftol": tolerance, "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    w = np.clip(result.x, 0, 1)
    return w / w.sum()



def training_window(features, targets, i, window):
    start, stop = i - window - 1, i - 1
    x, y = features[start:stop], targets[start:stop]
    if start < 0 or len(x) != window or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid training window.")
    return x, y


def ridge_forecasts(features, targets, signals, alpha, window):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    predictions = {}
    for i in signals:
        x, y = training_window(features, targets, i, window)
        model.fit(x.reshape(-1, x.shape[-1]), y.ravel())
        predictions[i] = model.predict(features[i])
    return predictions


def run_backtest(returns, signal_pos, exec_pos, signals, target, strategy,
                 start, end=None, cost_rate=COST_RATE):
    end = returns.index[-1] if end is None else pd.Timestamp(end)
    start = pd.Timestamp(start)
    valid = [i for i in signals if start <= returns.index[exec_pos[i]] <= end]
    if not valid:
        raise ValueError("No trades in evaluation period.")
    trade_map = {exec_pos[i]: i for i in valid}
    signal_map = {signal_pos[i]: i for i in valid[1:]}
    first, last = exec_pos[valid[0]], returns.index.searchsorted(end, side="right")
    weights = None
    pending = target(valid[0], None)
    nav = 1.0
    records, allocations = [], []
    for t in range(first, last):
        daily = returns.iloc[t].to_numpy()
        gross = 0.0 if weights is None else float(weights @ daily)
        if gross <= -1:
            raise ValueError("Portfolio lost all its value.")
        pre_trade = None if weights is None else weights * (1 + daily) / (1 + gross)
        turnover = 0.0
        if t in trade_map:
            pending = np.asarray(pending)
            if (not np.isfinite(pending).all() or (pending < -1e-8).any()
                    or not np.isclose(pending.sum(), 1.0, atol=1e-8)):
                raise ValueError("Invalid target weights.")
            turnover = 1.0 if pre_trade is None else float(np.abs(pending - pre_trade).sum())
            weights = pending
            allocations.append({"Date": returns.index[t], "strategy": strategy,
                                **dict(zip(returns.columns, weights))})
        else:
            weights = pre_trade
        cost = cost_rate * turnover
        net_return = (1 + gross) * (1 - cost) - 1
        nav *= 1 + net_return
        records.append((returns.index[t], net_return, nav, turnover, cost))
        if t in signal_map:
            pending = target(signal_map[t], weights)
    performance = pd.DataFrame(records, columns=[
        "Date", "net_return", "nav", "turnover", "cost_rate"
    ]).set_index("Date")
    return performance, pd.DataFrame(allocations)
