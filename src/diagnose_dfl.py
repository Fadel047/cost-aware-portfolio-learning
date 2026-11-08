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
DATES = ["2022-02-28", "2023-03-31", "2026-10-30"]
TRAIN_MONTHS = 35
VALIDATION_MONTHS = 12


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

    for date in DATES:
        i = signal_dates.get_loc(pd.Timestamp(date))
        validation_stop = i - 1
        cutoff = validation_stop - VALIDATION_MONTHS
        training_start = cutoff - TRAIN_MONTHS - 1

        if (
            training_start < 0
            or signal_pos[training_start] + 1 < LOOKBACK
        ):
            raise ValueError(f"Insufficient history for {date}.")

        for j in range(training_start, validation_stop):
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

        x_valid = features[cutoff:validation_stop]
        y_valid = targets[cutoff:validation_stop]

        if not np.isfinite(y_valid).all():
            raise ValueError(f"Missing validation targets for {date}.")

        standardized = scaler.transform(
            x_valid.reshape(-1, x_valid.shape[-1])
        ).reshape(x_valid.shape)

        for model, parameters in (("ridge", anchor), ("dfl", theta)):
            train = {
                key: diagnostics[f"{model}_{key}"]
                for key in ("net", "cost", "risk")
            }

            validation = sequence_loss(
                parameters,
                standardized,
                y_valid,
                covariance[cutoff:validation_stop],
                signal_growth[cutoff:validation_stop],
                execution_growth[cutoff:validation_stop],
                pto["gamma"],
                pto["penalty"],
                anchor,
                dfl["regularization"],
                return_details=True,
            )

            for phase, values in (
                ("train", train),
                ("validation", validation),
            ):
                rows.append({
                    "date": date,
                    "phase": phase,
                    "model": model,
                    "net_bps": 10000 * values["net"],
                    "cost_bps": 10000 * values["cost"],
                    "risk_bps": 10000 * values["risk"],
                    "utility_bps": 10000 * (
                        values["net"] - values["risk"]
                    ),
                })

        print(
            f"{date}: {diagnostics['solver_message']} "
            f"| accepted={diagnostics['accepted']}"
        )

    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()