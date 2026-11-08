import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from decision_focused import SCALE, fit_forecast, sequence_loss
from portfolio import LOOKBACK, load_data, training_window


ROOT = Path(__file__).resolve().parents[1]
TRAIN_MONTHS = 35
VALIDATION_MONTHS = 12
YEARS = [2018, 2019, 2020, 2021]


def main():
    pto = json.loads(
        (ROOT / "config/pto_selected.json").read_text()
    )
    dfl = json.loads(
        (ROOT / "config/dfl.json").read_text()
    )

    (
        returns, features, targets,
        signal_pos, exec_pos, _, _,
    ) = load_data()

    signal_dates = returns.index[signal_pos]
    growth = (1 + returns).cumprod().to_numpy()

    covariance = np.full(
        (len(features), returns.shape[1], returns.shape[1]),
        np.nan,
    )
    signal_growth = np.ones_like(targets)
    execution_growth = np.ones_like(targets)

    for j in range(1, len(features) - 1):
        signal_growth[j] = (
            growth[signal_pos[j]] / growth[exec_pos[j - 1]]
        )
        execution_growth[j] = (
            growth[exec_pos[j]] / growth[exec_pos[j - 1]]
        )

    rows = []

    for year in YEARS:
        matches = np.flatnonzero(
            (signal_dates.year == year)
            & (signal_dates.month == 12)
        )
        if len(matches) != 1:
            raise ValueError(f"Missing December signal for {year}.")

        i = int(matches[0])
        stop = i - 1
        cutoff = stop - VALIDATION_MONTHS
        start = cutoff - TRAIN_MONTHS - 1

        if start < 0 or signal_pos[start] + 1 < LOOKBACK:
            raise ValueError(f"Insufficient history for {year}.")

        validation_start = returns.index[exec_pos[cutoff]]
        validation_end = returns.index[exec_pos[stop]]

        if validation_end >= pd.Timestamp("2022-01-01"):
            raise ValueError("Validation extends into the test period.")

        for j in range(start, stop):
            if np.isfinite(covariance[j]).all():
                continue

            history = returns.iloc[
                signal_pos[j] - LOOKBACK + 1:signal_pos[j] + 1
            ]
            covariance[j] = (
                LedoitWolf().fit(history.to_numpy()).covariance_ * 21
            )

        _, diagnostics, theta = fit_forecast(
            features,
            targets,
            cutoff,
            TRAIN_MONTHS,
            pto["alpha"],
            covariance,
            signal_growth,
            execution_growth,
            pto["gamma"],
            pto["penalty"],
            dfl["regularization"],
            max_iterations=100,
        )

        x_train, y_train = training_window(
            features, targets, cutoff, TRAIN_MONTHS
        )
        flat_train = x_train.reshape(-1, x_train.shape[-1])
        scaler = StandardScaler().fit(flat_train)

        ridge = Ridge(alpha=pto["alpha"]).fit(
            scaler.transform(flat_train), y_train.ravel()
        )
        anchor = ridge.coef_ / SCALE

        x_valid = features[cutoff:stop]
        y_valid = targets[cutoff:stop]

        if not np.isfinite(y_valid).all():
            raise ValueError(f"Missing validation targets for {year}.")

        standardized = scaler.transform(
            x_valid.reshape(-1, x_valid.shape[-1])
        ).reshape(x_valid.shape)

        validation = {}

        for model, parameters in (("ridge", anchor), ("dfl", theta)):
            validation[model] = sequence_loss(
                parameters,
                standardized,
                y_valid,
                covariance[cutoff:stop],
                signal_growth[cutoff:stop],
                execution_growth[cutoff:stop],
                pto["gamma"],
                pto["penalty"],
                anchor,
                dfl["regularization"],
                return_details=True,
            )

        ridge_utility = (
            validation["ridge"]["net"] - validation["ridge"]["risk"]
        )
        dfl_utility = (
            validation["dfl"]["net"] - validation["dfl"]["risk"]
        )
        training_gain = (
            diagnostics["dfl_net"] - diagnostics["dfl_risk"]
            - diagnostics["ridge_net"] + diagnostics["ridge_risk"]
        )

        rows.append({
            "start": validation_start.date(),
            "end": validation_end.date(),
            "train_gain_bps": 10000 * training_gain,
            "ridge_net_bps": 10000 * validation["ridge"]["net"],
            "dfl_net_bps": 10000 * validation["dfl"]["net"],
            "ridge_utility_bps": 10000 * ridge_utility,
            "dfl_utility_bps": 10000 * dfl_utility,
            "validation_gain_bps": 10000 * (
                dfl_utility - ridge_utility
            ),
            "converged": diagnostics["converged"],
            "accepted": diagnostics["accepted"],
        })

        print(f"{year}: {diagnostics['solver_message']}")

    results = pd.DataFrame(rows)
    gains = results["validation_gain_bps"]

    print(results.round(3).to_string(index=False))
    print(
        f"\nDFL supérieur au Ridge : {(gains > 0).sum()}/{len(gains)}"
        f"\nGain moyen : {gains.mean():.3f} bps"
        f"\nGain médian : {gains.median():.3f} bps"
        f"\nPire bloc : {gains.min():.3f} bps"
    )

    output = ROOT / "results/dfl_validation_blocks.csv"
    output.parent.mkdir(exist_ok=True)
    results.to_csv(output, index=False)


if __name__ == "__main__":
    main()