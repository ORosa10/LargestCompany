# LargestCompany daily report - 2026-07-27

## Verdict: FAVORABLE  (edge +26.3%)
- Target 2026-07-31 (4 days left) | traded AAPL | side auto-picked **YES** @ 0.55
- Fragility (Phase 7): **MEDIUM - some assumptions move it (convergence / tail dependence)**
- Data: Yahoo live

## Edge: market vs simulation
| | Value |
|---|---|
| Polymarket says (YES) | 54.9% |
| Simulation says (YES fair) | 81.2% |
| Edge (fair - price) | **+26.3%** |
| Grade | **FAVORABLE** (>5% favorable, 0-5% marginal, <=0 unfavorable) |

## Best structure: 2/2/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $26.36 |
| Payoff SD | $38.32 |
| VaR 5% | $53.91 |
| VaR 1% | $68.21 |
| Worst case | $101.29 |
| Probability of profit | 81.1% |
| Return on VaR 5% | 48.9% |
| Return on VaR 1% | 38.7% |
| Return on worst case | 26.0% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 2/2/3/3 (best) | 0.80 | 81% | +0.69 | $62.70 | +48.9% | $26.36 |
| 1/1/3/3 | 0.79 | 81% | +0.69 | $61.81 | +48.1% | $26.35 |
| 1/1/4/4 | 0.42 | 81% | +0.69 | $63.81 | +48.0% | $26.38 |
| 2/2/4/4 | 0.21 | 81% | +0.68 | $64.70 | +48.7% | $26.39 |

## Consistency check (across assumptions & simulation reruns)
Overall fragility: **MEDIUM - some assumptions move it (convergence / tail dependence)**. P(AAPL #1) ranges 81.2%-84.7% across models (spread 3.6%).
| Check | Result |
|---|---|
| Across simulation reruns (seeds) | Central metrics are converged; trust them. |
| Across models & tails (IV surface / ATM / copula) | Edge keeps its sign across every model - tradeable. |
| Tail dependence (joint crashes) | Edge is sensitive to tail dependence (change +13.6%). |
| Dominant lever | gap-dominated (structural) (IV range 0.282 vs gap range 0.286). |

P(AAPL #1) by model:
| Model | P(#1) |
|---|---|
| IV surface + Gaussian copula | 81.2% |
| ATM lognormal + Normal | 81.3% |
| ATM lognormal + Student-t copula df=5 | 84.7% |
| ATM lognormal + Student-t df=6 (fat marginals) | 83.9% |

## Watch-outs
- No material risk flags on these scenarios.