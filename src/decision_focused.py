import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from backtest_baselines import compute_metrics, compute_weights
from portfolio import (
    COST_RATE,
    LOOKBACK,
    load_data,
    optimize,
    ridge_forecasts,
    run_backtest,
    training_window,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SCALE = 0.01
SMOOTH = 1e-6


def drift(weights, jacobian, growth):
    value = weights @ growth
    drifted = weights * growth / value
    derivative = (np.diag(growth) - np.outer(drifted, growth)) / value
    return drifted, derivative @ jacobian


def decision_jacobian(
    weights, covariance, gamma, penalty,
    previous, design, previous_jacobian,
):
    # Différentiation des conditions KKT à ensemble actif fixé.
    free = np.flatnonzero(
        (weights > 1e-6) & (weights < 1 - 1e-6)
    )
    jacobian = np.zeros_like(design)

    if len(free) <= 1:
        return jacobian

    curvature = np.zeros(len(weights))

    if previous is not None:
        curvature = (
            penalty * SMOOTH
            / ((weights - previous) ** 2 + SMOOTH) ** 1.5
        )

    hessian = gamma * covariance + np.diag(curvature)

    kkt = np.zeros((len(free) + 1, len(free) + 1))
    kkt[:-1, :-1] = hessian[np.ix_(free, free)]
    kkt[:-1, -1] = 1
    kkt[-1, :-1] = 1

    rhs = np.zeros((len(free) + 1, design.shape[1]))
    rhs[:-1] = (
        design + curvature[:, None] * previous_jacobian
    )[free]

    jacobian[free] = np.linalg.solve(kkt, rhs)[:-1]
    return jacobian


def sequence_loss(
    theta, x, y, covariance, signal_growth, execution_growth,
    gamma, penalty, anchor, regularization, return_details=False,
):
    previous = None
    previous_jacobian = np.zeros_like(x[0])

    value = 0.0
    gradient = np.zeros_like(theta)
    net_sum = cost_sum = risk_sum = 0.0

    for j in range(len(x)):
        if previous is None:
            reference = None
            reference_jacobian = previous_jacobian
            pre_trade = None
            pre_jacobian = previous_jacobian
        else:
            reference, reference_jacobian = drift(
                previous, previous_jacobian, signal_growth[j]
            )
            pre_trade, pre_jacobian = drift(
                previous, previous_jacobian, execution_growth[j]
            )

        weights = optimize(
            SCALE * x[j] @ theta,
            covariance[j],
            gamma,
            penalty,
            reference,
            tolerance=1e-12,
        )

        jacobian = decision_jacobian(
            weights,
            covariance[j],
            gamma,
            penalty,
            reference,
            SCALE * x[j],
            reference_jacobian,
        )

        if pre_trade is None:
            turnover = 1.0
            turnover_gradient = np.zeros_like(theta)
        else:
            diff = weights - pre_trade
            smooth = np.sqrt(diff**2 + SMOOTH)
            turnover = (smooth - np.sqrt(SMOOTH)).sum()
            turnover_gradient = (
                (diff / smooth) @ (jacobian - pre_jacobian)
            )

        gross = 1 + weights @ y[j]
        net = gross * (1 - COST_RATE * turnover) - 1
        risk = 0.5 * gamma * (weights @ covariance[j] @ weights)

        net_sum += net
        cost_sum += gross * COST_RATE * turnover
        risk_sum += risk
        value += -net + risk

        gradient += (
            -(1 - COST_RATE * turnover) * y[j]
            + gamma * covariance[j] @ weights
        ) @ jacobian

        gradient += COST_RATE * gross * turnover_gradient
        previous, previous_jacobian = weights, jacobian

    difference = theta - anchor
    anchor_penalty = (
        0.5 * regularization * (difference @ difference)
    )
    loss = value / len(x) + anchor_penalty

    if return_details:
        return {
            "net": net_sum / len(x),
            "cost": cost_sum / len(x),
            "risk": risk_sum / len(x),
            "anchor": anchor_penalty,
            "loss": loss,
        }

    return loss, gradient / len(x) + regularization * difference


def fit_forecast(
    features, targets, i, window, alpha, covariance,
    signal_growth, execution_growth, gamma, penalty,
    regularization, max_iterations,
):
    x, y = training_window(features, targets, i, window)

    scaler = StandardScaler().fit(
        x.reshape(-1, x.shape[-1])
    )
    standardized = scaler.transform(
        x.reshape(-1, x.shape[-1])
    ).reshape(x.shape)

    ridge = Ridge(alpha=alpha).fit(
        standardized.reshape(-1, x.shape[-1]),
        y.ravel(),
    )
    anchor = ridge.coef_ / SCALE

    start, stop = i - window - 1, i - 1
    args = (
        standardized,
        y,
        covariance[start:stop],
        signal_growth[start:stop],
        execution_growth[start:stop],
        gamma,
        penalty,
        anchor,
        regularization,
    )

    initial_loss, _ = sequence_loss(anchor, *args)

    result = minimize(
        sequence_loss,
        anchor,
        args=args,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": max_iterations,
            "ftol": 1e-9,
            "gtol": 1e-6,
            "maxls": 30,
        },
    )

    # Une amélioration à la limite d'itérations peut être acceptée.
    accepted = bool(
        result.status in (0, 1)
        and np.isfinite(result.fun)
        and np.isfinite(result.x).all()
        and result.fun <= initial_loss
    )
    theta = result.x if accepted else anchor

    prediction = (
        SCALE * scaler.transform(features[i]) @ theta
        + ridge.intercept_
    )

    diagnostics = {
        "initial_loss": initial_loss,
        "final_loss": result.fun,
        "used_loss": result.fun if accepted else initial_loss,
        "iterations": result.nit,
        "converged": result.success,
        "accepted": accepted,
        "solver_message": str(result.message),
        "gradient_norm": float(np.linalg.norm(result.jac)),
    }

    for name, parameters in (("ridge", anchor), ("dfl", theta)):
        details = sequence_loss(
            parameters, *args, return_details=True
        )
        diagnostics.update({
            f"{name}_{key}": value
            for key, value in details.items()
        })

    return prediction, diagnostics, theta


def main():
    RESULTS.mkdir(exist_ok=True)

    config = json.loads(
        (ROOT / "config/dfl.json").read_text()
    )
    pto = json.loads(
        (ROOT / "config/pto_selected.json").read_text()
    )

    (
        returns, features, targets,
        signal_pos, exec_pos, signals, _,
    ) = load_data()

    start_date = pd.Timestamp(config["start_date"])
    valid = [
        i for i in signals
        if returns.index[exec_pos[i]] >= start_date
    ]

    if not valid:
        raise ValueError("No evaluation observations.")

    alpha, window = pto["alpha"], pto["window"]
    gamma, penalty = pto["gamma"], pto["penalty"]

    first_training = valid[0] - window - 1

    if (
        first_training < 0
        or signal_pos[first_training] + 1 < LOOKBACK
    ):
        raise ValueError("Insufficient training covariance history.")

    covariance = np.full(
        (len(features), returns.shape[1], returns.shape[1]),
        np.nan,
    )

    for i in range(first_training, valid[-1] + 1):
        history = returns.iloc[
            signal_pos[i] - LOOKBACK + 1:signal_pos[i] + 1
        ]
        covariance[i] = (
            LedoitWolf().fit(history.to_numpy()).covariance_ * 21
        )

    growth = (1 + returns).cumprod().to_numpy()
    signal_growth = np.ones_like(targets)
    execution_growth = np.ones_like(targets)

    for i in range(1, len(features) - 1):
        signal_growth[i] = (
            growth[signal_pos[i]] / growth[exec_pos[i - 1]]
        )
        execution_growth[i] = (
            growth[exec_pos[i]] / growth[exec_pos[i - 1]]
        )

    predictions = {}
    fit_rows = []
    coefficient_rows = []

    for i in valid:
        predictions[i], diagnostic, theta = fit_forecast(
            features,
            targets,
            i,
            window,
            alpha,
            covariance,
            signal_growth,
            execution_growth,
            gamma,
            penalty,
            config["regularization"],
            config["max_iterations"],
        )

        fit_rows.append({
            "signal_date": returns.index[signal_pos[i]],
            "last_training_target_end": returns.index[exec_pos[i - 1]],
            **diagnostic,
        })
        coefficient_rows.append({
            "signal_date": returns.index[signal_pos[i]],
            **dict(enumerate(theta)),
        })

    ridge = ridge_forecasts(
        features, targets, valid, alpha, window
    )
    strategies = [
        "equal_weight",
        "inverse_volatility",
        "minimum_variance",
        "ridge_tuned",
        "dfl",
    ]

    def target(i, previous, strategy):
        if strategy in ("ridge_tuned", "dfl"):
            mu = (
                ridge[i] if strategy == "ridge_tuned"
                else predictions[i]
            )
            return optimize(
                mu, covariance[i], gamma, penalty, previous
            )

        history = returns.iloc[
            signal_pos[i] - LOOKBACK + 1:signal_pos[i] + 1
        ]
        return compute_weights(history, strategy)

    metrics, navs, daily = {}, {}, {}
    allocations = []

    for strategy in strategies:
        performance, weights = run_backtest(
            returns,
            signal_pos,
            exec_pos,
            signals,
            lambda i, previous: target(i, previous, strategy),
            strategy,
            start_date,
        )

        metrics[strategy] = compute_metrics(performance)
        navs[strategy] = performance["nav"]
        daily[strategy] = performance["net_return"]
        allocations.append(weights)

    summary = pd.DataFrame(metrics).T
    fits = pd.DataFrame(fit_rows)

    summary.to_csv(RESULTS / "dfl_summary.csv")
    pd.DataFrame(navs).to_csv(RESULTS / "dfl_nav.csv")
    pd.DataFrame(daily).to_csv(RESULTS / "dfl_returns.csv")
    fits.to_csv(RESULTS / "dfl_training.csv", index=False)

    pd.concat(allocations, ignore_index=True).to_csv(
        RESULTS / "dfl_weights.csv", index=False
    )
    pd.DataFrame(coefficient_rows).to_csv(
        RESULTS / "dfl_coefficients.csv", index=False
    )

    pd.DataFrame([
        {
            "signal_date": returns.index[signal_pos[i]],
            "ticker": ticker,
            "prediction": predictions[i][j],
            "realized": targets[i, j],
        }
        for i in valid
        for j, ticker in enumerate(returns.columns)
    ]).to_csv(RESULTS / "dfl_forecasts.csv", index=False)

    print(
        f"Exploratory comparison: "
        f"{returns.index[exec_pos[valid[0]]].date()} "
        f"to {returns.index[-1].date()}"
    )
    print(summary[[
        "annual_return",
        "sharpe_rf0",
        "max_drawdown",
        "annual_traded_notional",
    ]].round(4).to_string())
    print(
        f"Fits: {len(fits)}, "
        f"converged: {fits.converged.sum()}, "
        f"accepted: {fits.accepted.sum()}"
    )


if __name__ == "__main__":
    main()