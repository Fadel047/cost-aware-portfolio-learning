# Findings

## Research question

Does training a model for portfolio decisions improve allocation after
transaction costs compared with predicting returns first and optimizing
the portfolio afterward?

This project compares a Ridge predict-then-optimize (PTO) model with a
linear decision-focused learning (DFL) prototype and simple allocation
benchmarks on seven ETFs.

## Portfolio results

The exploratory comparison covers January 3, 2022 to November 6, 2026.
All strategies include transaction costs of 10 basis points per unit of
traded notional.

| Strategy | Annualized return | Sharpe, rf = 0 | Maximum drawdown | Annual traded notional |
|---|---:|---:|---:|---:|
| Equal weight | 8.82% | 0.669 | -23.96% | 0.500 |
| Inverse volatility | 7.85% | 0.716 | -21.11% | 0.525 |
| Minimum variance | 1.67% | 0.260 | -18.77% | 0.552 |
| Ridge PTO | 2.37% | 0.311 | -20.78% | 0.261 |
| Linear DFL | 1.33% | 0.202 | -20.10% | 0.259 |

Inverse volatility achieved the highest Sharpe, while equal weight achieved
the highest annualized return. Neither learned strategy outperformed these
benchmarks.

DFL produced a slightly smaller maximum drawdown than Ridge, but its return
and Sharpe were lower. Their similar traded notional suggests that the
performance gap cannot be explained by greater trading activity alone.

## Forecasting and learning diagnostics

Ridge forecasts showed weak predictive performance:

- Pooled Pearson correlation: -0.1331.
- Mean monthly cross-sectional rank correlation: -0.0320.
- Forecast MSE divided by zero-forecast MSE: 1.0617.

DFL changed the training objective from prediction accuracy to portfolio
utility after costs, with a variance penalty and regularization toward
the initial Ridge coefficients.

In the reported run, 36 of 59 rolling fits met the solver's convergence
criteria; the other 23 reached the 30-iteration limit and were accepted
because they satisfied the training-loss acceptance criterion.

Increasing the limit to 100 iterations on three selected windows improved
the objective, but the improvement was small in two cases. All three
terminated because objective changes became small, rather than because
the gradient met the stopping threshold. These checks do not establish
that the numerical optimization problem is fully resolved.

## Historical block validation

Four non-overlapping validation blocks ended before 2022. Each used
35 training months, a one-month label gap and 12 validation months.
Model coefficients remained fixed during validation.

Utility is mean net holding-period return minus the estimated variance
penalty. Differences below are DFL minus Ridge, in basis points per period.

| Validation block | Training utility gain | Validation utility gain |
|---|---:|---:|
| December 2017–December 2018 | +21.061 | +6.501 |
| December 2018–December 2019 | +5.294 | -10.359 |
| December 2019–December 2020 | +5.303 | +4.308 |
| December 2020–December 2021 | +33.403 | -57.356 |

DFL improved training utility in every block, but validation utility in
only two. The mean validation difference was -14.226 bps and the median
was -3.025 bps.

In the weakest block, DFL reduced mean net return by approximately
25.3 bps and increased the variance penalty by approximately 32.1 bps.
The deterioration therefore involved both return and estimated risk.

## Interpretation

This implementation demonstrates that the model can improve its historical
decision objective. It does not demonstrate that those improvements
generalize reliably.

The results are consistent with unstable generalization and sensitivity
to market conditions. They do not isolate a single cause: weak input
signals, model specification, regularization, objective choice and numerical
optimization may all contribute.

## Limitations

- The study uses seven ETFs and a limited set of historical features.
- The 2022–2026 period was inspected during development and is not an
  untouched final test.
- PTO hyperparameters were selected using data overlapping the historical
  block diagnostics; these are not independent validation results.
- Block diagnostics use shorter training windows, fixed coefficients and
  a cash restart, unlike the monthly-refitted full backtest.
- Training uses smoothed transaction costs; the daily backtest charges
  absolute traded notional. Market impact and taxes are excluded.
- The learning objective is a return–variance trade-off, not Sharpe.
- Four validation blocks are insufficient for strong statistical claims.
- Numerical convergence remains a limitation of the prototype.

## Conclusion

The current linear DFL prototype does not establish an advantage over
Ridge PTO or simple allocation benchmarks.

Its historical training improvements are not consistently retained in
validation, and its exploratory full-period Sharpe is lower than Ridge,
equal weight and inverse volatility. The experiment is retained as a
negative result rather than followed by further parameter searches aimed
at improving already observed outcomes.

This conclusion applies to this implementation, asset universe and
evaluation protocol. It does not establish that decision-focused learning
is ineffective in general.

## Reference

Jensen, T. I., Kelly, B., Malamud, S., and Pedersen, L. H. (2026),
[Machine Learning and the Implementable Efficient Frontier](https://doi.org/10.1093/rfs/hhag022),
The Review of Financial Studies.

This project is inspired by the paper's focus on portfolio learning after
trading costs. It is a simplified independent experiment, not a replication
of its methodology or empirical results.