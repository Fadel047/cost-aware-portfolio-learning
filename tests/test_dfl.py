import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from decision_focused import sequence_loss
from portfolio import run_backtest, training_window


class DFLTests(unittest.TestCase):
    def test_recurrent_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(18)
        x = rng.normal(size=(4, 3, 2))
        y = rng.normal(0.005, 0.02, size=(4, 3))
        cov = np.broadcast_to(np.eye(3)*0.02, (4, 3, 3))
        signal_growth = 1 + rng.normal(0, 0.01, (4, 3))
        execution_growth = signal_growth * (1 + rng.normal(0, 0.005, (4, 3)))
        theta = np.array([0.1, -0.2])
        args = (x, y, cov, signal_growth, execution_growth, 5.0, 0.003,
                np.zeros(2), 0.001)
        _, analytic = sequence_loss(theta, *args)
        epsilon = 1e-4
        numeric = np.empty_like(theta)
        for k in range(len(theta)):
            direction = np.eye(len(theta))[k]*epsilon
            numeric[k] = (sequence_loss(theta+direction, *args)[0]
                          - sequence_loss(theta-direction, *args)[0])/(2*epsilon)
        np.testing.assert_allclose(analytic, numeric, rtol=0.02, atol=2e-6)

    def test_execution_day_uses_old_weights_and_charges_drifted_turnover(self):
        dates = pd.bdate_range('2020-01-01', periods=6)
        returns = pd.DataFrame([[0, 0], [0.5, 0], [0.1, 0], [0, 0.2], [0, 0.1], [0, 0]],
                               index=dates, columns=['a', 'b'])
        observed = []
        def target(i, previous):
            observed.append(previous)
            return np.array([0.5, 0.5]) if i == 0 else np.array([1.0, 0.0])
        perf, weights = run_backtest(
            returns, np.array([0, 2]), np.array([1, 3]), [0, 1],
            target, 'test', dates[0], cost_rate=0.001,
        )
        self.assertAlmostEqual(perf.iloc[0].nav, 0.999)
        self.assertAlmostEqual(perf.iloc[1].net_return, 0.05)
        pre = np.array([0.55, 0.60])/1.15
        turnover = np.abs(np.array([1, 0])-pre).sum()
        expected = (1 + (0.5/1.05)*0.2)*(1 - 0.001*turnover)-1
        self.assertAlmostEqual(perf.iloc[2].net_return, expected)
        self.assertAlmostEqual(perf.iloc[2].turnover, turnover)
        np.testing.assert_allclose(observed[1], np.array([0.55, 0.5])/1.05)
        self.assertAlmostEqual(perf.iloc[3].net_return, 0)
        self.assertEqual(list(weights.Date), [dates[1], dates[3]])

    def test_training_window_excludes_unavailable_labels(self):
        x = np.arange(120, dtype=float).reshape(20, 3, 2)
        y = np.arange(60, dtype=float).reshape(20, 3)
        a, b = training_window(x, y, 12, 5)
        y[11:] = np.nan
        c, d = training_window(x, y, 12, 5)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        self.assertEqual(len(a), 5)
        np.testing.assert_array_equal(b, y[6:11])


if __name__ == '__main__':
    unittest.main()
