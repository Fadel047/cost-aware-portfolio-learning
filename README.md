# Cost-aware portfolio learning

Research project comparing predict-then-optimize (PTO), cost-aware allocation,
and a first linear decision-focused learning (DFL) prototype.

## Setup and execution

Python 3.11+ is required. Run commands from the project root:

```bash
python -m pip install -r requirements.txt
python src/download_data.py
python src/analyze_data.py
python src/diagnose_pto.py
python src/decision_focused.py
python -m unittest discover -s tests
```

The supplied ZIP includes the data snapshot and generated results. With that
snapshot, skip the download and analysis steps to reproduce the diagnostics.
Data and generated results are excluded from Git; the selected configuration
and the research snapshot below are tracked. The downloader requires network
access. Its yfinance dependency is not pinned or verified in this review.
Fresh downloads may revise historical adjusted prices and extend the sample.

Optional research scripts:

```bash
python src/backtest_baselines.py
python src/predict_then_optimize.py
python src/tune_pto.py
```

`tune_pto.py` runs all 1,260 configurations and overwrites
`config/pto_selected.json` using validation Sharpe (2018–2021). It then evaluates
the selection from 2022 onward. Do not rerun tuning to improve the observed test.
`diagnose_pto.py` uses the saved configuration without repeating the search.

## Reference experiment

- Assets: SPY, QQQ, IWM, EFA, EEM, IEF, GLD; adjusted daily closes from Yahoo Finance.
- Features: 1/3/6/12-month returns, 63-day daily volatility, asset indicators.
- Pooled Ridge with training-only standardization; rolling training windows.
- Long-only, fully invested portfolios; Ledoit–Wolf covariance over 756 days,
  scaled by 21 for the monthly optimization objective.
- Signals at month-end close; execution at the next observed trading-day close.
  Existing weights earn that day's return before the trade. New weights earn
  returns starting the following day.
- Targets run from execution close to the following execution close. Training
  excludes the most recent unfinished target (one-month label gap).
- Realized costs: 10 bps times sum of absolute weight changes; initial investment
  turnover is 1. Weights drift between trades. The optimizer uses a smoothed L1
  penalty relative to signal-date weights; charged costs use execution-date weights.
- Costs are a proportional NAV haircut, an approximation rather than an exact
  self-financing share-level execution model. No market impact or taxes.
- Annualized metrics use 252 observations per year; Sharpe assumes zero risk-free
  rate. Traded notional is the full sum of buys and sells, not half-turnover.

`backtest_baselines.py` is an earlier standalone experiment using month-end-close
execution. `predict_then_optimize.py` is an exploratory calendar-month-target
experiment. Their dates and conventions differ from the reference experiment;
do not compare their summary tables directly with the tuned diagnostics.
`portfolio.py` supplies data preparation, Ridge forecasts, constrained allocation
and the common backtest used by `tune_pto.py`, `diagnose_pto.py` and
`decision_focused.py`. The earlier exploratory scripts remain separate.

## Frozen research snapshot

Validation: 2018–2021. Selected alpha=100, window=48 months, gamma=20,
penalty=0.03. Test: 2022-01-03 through 2026-11-06.

| Strategy | Annual return | Sharpe (rf=0) | Annual traded notional |
|---|---:|---:|---:|
| Equal weight | 8.82% | 0.669 | 0.500 |
| Inverse volatility | 7.85% | 0.716 | 0.525 |
| Minimum variance | 1.67% | 0.260 | 0.552 |
| Ridge, no optimization penalty | 2.49% | 0.281 | 5.775 |
| Ridge, penalty=0.001 | 3.28% | 0.355 | 4.171 |
| Ridge, selected penalty | 2.37% | 0.311 | 0.261 |

All rows include realized transaction costs, including the no-penalty variant.
Pooled forecast Pearson correlation: -0.1331; mean monthly cross-sectional rank
IC: -0.0320; forecast MSE / zero-forecast MSE: 1.0617. The final incomplete
holding-period target is excluded from forecast diagnostics.

The penalty reduces trading, but this Ridge specification does not outperform
simple allocations on the observed test. Searching 1,260 configurations over
four validation years also creates selection risk. The test has already been
inspected; further model development must disclose this reuse.

## Decision-focused prototype

`decision_focused.py` uses the same linear model features, 48-month training
window, training-only scaler, label gap, covariance estimator and portfolio
constraints as the selected PTO. Each month starts from a newly fitted Ridge.
The coefficients are then optimized for mean historical decision loss:

- negative holding-period net return;
- plus one half of gamma times portfolio variance;
- plus an L2 penalty anchoring coefficients to that month's Ridge fit.

The training simulation is sequential: weights drift between execution and
signal dates; the next decision depends on existing positions. Gradients pass
through this recurrence and the constrained optimizer using the KKT system
on the locally fixed active set. They are not globally smooth when bounds become
active or inactive. The implementation uses NumPy/SciPy, with no new dependency.
The implicit-differentiation approach follows the general method described in
[Amos and Kolter (2017), OptNet](https://proceedings.mlr.press/v70/amos17a.html);
this is a small custom implementation, not their solver or a replication.

Training approximates absolute turnover by `sqrt(delta**2 + 1e-6) - 0.001`.
Evaluation always charges exact absolute turnover in the shared daily backtest.
The allocation optimizer retains the PTO's smooth turnover penalty, with the
same gamma=20 and penalty=0.03. A tighter solver tolerance is used only during
training to compute meaningful gradients. All strategies use the same default
solver tolerance during evaluation. Every historical training window restarts
from cash; its sequence is an empirical training loss, not a reportable backtest.

`config/dfl.json` fixes a coefficient scale-adjusted L2 regularization of 0.001
and a maximum of 30 optimizer iterations per rolling fit. These are initial
engineering settings, not selected using the reported comparison. No DFL
hyperparameter search has been performed. Coefficients are represented in units
of 0.01, so the regularization applies to these scaled coefficients. A common
Ridge intercept is retained in forecast outputs but has no effect on a fully
invested allocation.

A fit is accepted if it converges or reaches its iteration budget and improves
(or matches) the initial training objective with finite coefficients. Other
outcomes explicitly fall back to the Ridge initialization. Each fit logs
convergence, acceptance, losses and the solver message in `dfl_training.csv`.
An accepted iteration-limited fit is not a claim of convergence.

Outputs in `results/`: `dfl_summary.csv`, `dfl_nav.csv`, `dfl_returns.csv`,
`dfl_weights.csv`, `dfl_forecasts.csv`, `dfl_coefficients.csv`, and
`dfl_training.csv`. The coefficient file stores standardized coefficients in
0.01 units for auditing; it is not a standalone deployable model artifact.

The 2022–2026 comparison is exploratory: this period was inspected during PTO
development. No fresh, untouched test is claimed. The next research step is to
inspect fit convergence and stability, then specify a validation protocol before
changing DFL hyperparameters. Performance improvement is not assumed.

## First DFL run: exploratory results

Same 2022-01-03 through 2026-11-06 snapshot; all returns are net of costs.

| Strategy | Annual return | Sharpe (rf=0) | Max drawdown | Annual traded notional |
|---|---:|---:|---:|---:|
| Equal weight | 8.82% | 0.669 | -23.96% | 0.500 |
| Inverse volatility | 7.85% | 0.716 | -21.11% | 0.525 |
| Minimum variance | 1.67% | 0.260 | -18.77% | 0.552 |
| Selected Ridge PTO | 2.37% | 0.311 | -20.78% | 0.261 |
| Linear DFL prototype | 1.33% | 0.202 | -20.10% | 0.259 |

Of 59 rolling fits, 36 report convergence, 22 reach the iteration limit with
an accepted training improvement, and one reports abnormal line-search
termination and falls back to Ridge. These diagnostics limit conclusions about
the trained method. The run does not establish a DFL advantage: its return and
Sharpe are lower than PTO, with similar turnover. No settings were changed after
viewing these results. Diagnose numerical convergence and objective alignment
before considering any validation-based parameter changes.

## Verification

```bash
python -m unittest discover -s tests
```

Tests cover target timing and future-data isolation, optimizer constraints,
initial-cost drawdown, old-weight execution-day returns and drifted turnover,
unavailable-label exclusion, and recurrent gradients against finite differences.
The refactor was also checked against all four original PTO diagnostic CSVs
at absolute/relative tolerance 1e-12 using the supplied data snapshot.

The full DFL comparison reproduced all four reference strategies within
rtol=1e-7 / atol=1e-8 across BLAS thread settings. DFL weight constraints,
label availability and non-increasing accepted training losses were checked.
