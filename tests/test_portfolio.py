import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import portfolio
from backtest_baselines import compute_metrics


class PortfolioTests(unittest.TestCase):
    def test_target_timing_and_future_data_isolation(self):
        rng = np.random.default_rng(7)
        returns = pd.DataFrame(
            rng.normal(0.0002, 0.01, (2700, 3)),
            index=pd.bdate_range('2010-01-01', periods=2700, name='Date'),
            columns=['a', 'b', 'c'],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            returns.to_csv(path / 'daily_returns.csv')
            _, features, targets, signal_pos, exec_pos, signals, cov = portfolio.load_data(path)
            i = signals[0]
            expected = (1 + returns.iloc[exec_pos[i]+1:exec_pos[i+1]+1]).prod() - 1
            np.testing.assert_allclose(targets[i], expected)
            self.assertLessEqual(exec_pos[i-1], signal_pos[i])
            self.assertGreater(exec_pos[i], signal_pos[i])
            changed = returns.copy()
            changed.iloc[signal_pos[i]+1:] += 0.005
            changed.to_csv(path / 'daily_returns.csv')
            _, new_features, new_targets, _, _, _, new_cov = portfolio.load_data(path)
            np.testing.assert_allclose(features[i], new_features[i])
            np.testing.assert_allclose(cov[i], new_cov[i])
            np.testing.assert_allclose(targets[:i-1], new_targets[:i-1], equal_nan=True)

    def test_optimizer_constraints_and_penalty(self):
        previous = np.array([0.5, 0.5])
        mu, cov = np.array([0.02, 0.0]), np.eye(2) * 0.01
        plain = portfolio.optimize(mu, cov, 1, 0, previous)
        penalized = portfolio.optimize(mu, cov, 1, 0.1, previous)
        for weights in [plain, penalized]:
            self.assertAlmostEqual(weights.sum(), 1)
            self.assertTrue((weights >= 0).all())
        self.assertLess(np.abs(penalized-previous).sum(), np.abs(plain-previous).sum())

    def test_initial_cost_is_in_drawdown(self):
        performance = pd.DataFrame({
            'net_return': [-0.001, 0.0], 'nav': [0.999, 0.999],
            'turnover': [1.0, 0.0], 'cost_rate': [0.001, 0.0],
        })
        metrics = compute_metrics(performance)
        self.assertAlmostEqual(metrics['max_drawdown'], -0.001)
        self.assertAlmostEqual(metrics['final_nav'], 0.999)


if __name__ == '__main__':
    unittest.main()
