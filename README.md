# Cost-Aware Portfolio Learning

Does training a return model directly for portfolio decisions improve allocation after transaction costs compared with predict-then-optimize and simple benchmarks?

The study uses adjusted daily prices for seven ETFs: SPY, QQQ, IWM, EFA, EEM, IEF and GLD. A Ridge model predicts holding-period returns from past returns, volatility and asset indicators. A linear decision-focused learning (DFL) model starts from Ridge and adjusts its coefficients through a constrained portfolio optimizer to improve a return–risk objective after costs.

Portfolios are long-only, fully invested and rebalanced monthly. Signals are formed at month-end and executed at the next trading-day close. The comparison uses rolling training windows, a gap for unavailable training targets and transaction costs of 10 basis points per unit of traded notional.

The current DFL prototype does not outperform Ridge, equal weight or inverse-volatility allocation in the exploratory 2022–2026 comparison. Additional historical block diagnostics show inconsistent generalization. This period has already informed project development and is not an untouched final test.

## Reference

Jensen, T. I., Kelly, B., Malamud, S., and Pedersen, L. H. (2026),
[Machine Learning and the Implementable Efficient Frontier](https://doi.org/10.1093/rfs/hhag022),
The Review of Financial Studies.

Inspired by the paper’s focus on learning portfolio decisions after trading
costs, this project compares predict-then-optimize with a linear
decision-focused prototype on seven ETFs. It is a simplified independent
experiment, not a replication of the paper’s methodology or empirical results.