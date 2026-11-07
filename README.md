# Cost-aware portfolio learning

Research project comparing predict-then-optimize (PTO), cost-aware allocation,
and eventually decision-focused learning (DFL). DFL is not implemented yet.

## Setup and execution

Python 3.11+ is required. Run commands from the project root:

```bash
python -m pip install -r requirements.txt
python src/download_data.py
python src/analyze_data.py
python src/diagnose_pto.py
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
`tune_pto.py` and `diagnose_pto.py` define the reference comparison for the next stage.

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

## Next stage

Add a minimal DFL model using the same inputs, execution conventions and costs.
Before adding it, extract the duplicated reference backtest into a shared module
and check numerical equivalence. Keep the frozen PTO comparison. Evaluate DFL
without claiming it can recover a signal absent from the input data.
