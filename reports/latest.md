# LargestCompany daily report - 2026-07-30

## Verdict: FAVORABLE  (edge +15.6%)
- Resolution 2026-08-31 | option expiry 2026-08-28 (29 days) | traded NVDA | side auto-picked **NO** @ 0.61
- Fragility (Phase 7): **MEDIUM - some assumptions move it (convergence / tail dependence)**
- Data: Yahoo live

## Summary
- Expected profit **$15.83** on $102.27 capital at risk (RoCaR 15.5%).
- Side auto-picked **NO** by composite (P(win), EV/SD, CVaR5%, RoC/VaR5%); naked YES EV -16.6% vs naked NO EV +15.6%.
- Your NO edge: model P(NVDA NOT #1) 76.6% vs NO price 61% -> +15.6%.
- Robustness: probability estimate is NOT fully robust (stability of the estimate, not the trade direction).

## Edge: market vs simulation
| | Value |
|---|---|
| Polymarket says (NO) | 61.0% |
| Simulation says (NO fair) | 76.6% |
| Edge (fair - price) | **+15.6%** |
| Grade | **FAVORABLE** (>5% favorable, 0-5% marginal, <=0 unfavorable) |

## Trade candidates (best edge per side)
| | Ticker | Model fair | Price | Edge | Composite |
|---|---|---|---|---|---|
| Best YES | AAPL | 75.2% | 58% | +17.2% | 0.25 |
| Best NO (traded) | NVDA | 76.6% | 61% | +15.6% | 0.75 |
Default = max composite: **NVDA NO** (composite 0.75, edge +15.6%). Composite = P(win), EV/SD, CVaR5%, RoC/VaR5% - risk-adjusted, not raw edge. Both sides get a full risk block below.

## Probability by name (model vs market)
| Ticker | Model P(#1) | Market YES | Market NO | YES edge | NO edge |
|---|---|---|---|---|---|
| AAPL | 75.2% | 58% | 44% | +17.2% | -19.2% |
| NVDA * | 23.4% | 40% | 61% | -16.6% | +15.6% |
| GOOGL | 1.4% | 4% | 97% | -2.6% | +1.7% |
* = traded name.

---
# Best YES trade: AAPL YES @ 0.58
Edge +17.2% | Verdict FAVORABLE | Fragility MEDIUM

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $14.01 |
| Payoff SD | $43.77 |
| VaR 5% | $69.54 |
| VaR 1% | $95.21 |
| Worst case | $102.68 |
| Probability of profit | 74.1% |
| Return on VaR 5% | 20.1% |
| Return on VaR 1% | 14.7% |
| Return on worst case | 13.6% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 1.00 | 74% | +0.32 | $84.84 | +20.1% | $14.01 |
| 2/2/3/3 | 0.48 | 64% | +0.29 | $90.70 | +18.8% | $13.33 |
| 1/1/4/4 | 0.36 | 64% | +0.29 | $91.83 | +18.1% | $13.18 |
| 2/2/4/4 | 0.00 | 62% | +0.26 | $97.69 | +16.8% | $12.50 |

## Consistency check (across assumptions & simulation reruns)
Overall fragility: **MEDIUM - some assumptions move it (convergence / tail dependence)**. P(AAPL #1) ranges 74.6%-78.0% across models (spread 3.3%).
| Check | Result |
|---|---|
| Across simulation reruns (seeds) | Central metrics are converged; trust them. |
| Across models & tails (IV surface / ATM / copula) | Edge keeps its sign across every model - tradeable. |
| Tail dependence (joint crashes) | Edge is sensitive to tail dependence (change +21.7%). |
| Dominant lever | gap-dominated (structural) (IV range 0.303 vs gap range 0.322). |

P(AAPL #1) by model:
| Model | P(#1) |
|---|---|
| IV surface + Gaussian copula | 75.2% |
| ATM lognormal + Normal | 74.6% |
| ATM lognormal + Student-t copula df=5 | 78.0% |
| ATM lognormal + Student-t df=6 (fat marginals) | 77.1% |

## Watch-outs
- No material risk flags on these scenarios.

---
# Best NO trade: NVDA NO @ 0.61 (traded / app preset)
Edge +15.6% | Verdict FAVORABLE | Fragility MEDIUM

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $15.83 |
| Payoff SD | $39.23 |
| VaR 5% | $37.27 |
| VaR 1% | $86.54 |
| Worst case | $102.27 |
| Probability of profit | 66.2% |
| Return on VaR 5% | 42.5% |
| Return on VaR 1% | 18.3% |
| Return on worst case | 15.5% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 0.75 | 66% | +0.40 | $46.36 | +42.5% | $15.83 |
| 1/1/4/4 | 0.34 | 56% | +0.39 | $59.92 | +51.4% | $15.91 |
| 2/2/3/3 | 0.29 | 58% | +0.38 | $60.73 | +48.5% | $15.81 |
| 2/2/4/4 | 0.26 | 54% | +0.37 | $60.16 | +60.5% | $15.88 |

## Consistency check (across assumptions & simulation reruns)
Overall fragility: **MEDIUM - some assumptions move it (convergence / tail dependence)**. P(NVDA #1) ranges 20.9%-24.5% across models (spread 3.6%).
| Check | Result |
|---|---|
| Across simulation reruns (seeds) | Central metrics are converged; trust them. |
| Across models & tails (IV surface / ATM / copula) | Edge keeps its sign across every model - tradeable. |
| Tail dependence (joint crashes) | Edge is sensitive to tail dependence (change -24.8%). |
| Dominant lever | gap-dominated (structural) (IV range 0.218 vs gap range 0.239). |

P(NVDA #1) by model:
| Model | P(#1) |
|---|---|
| IV surface + Gaussian copula | 23.4% |
| ATM lognormal + Normal | 24.5% |
| ATM lognormal + Student-t copula df=5 | 20.9% |
| ATM lognormal + Student-t df=6 (fat marginals) | 21.6% |

## Watch-outs
- No material risk flags on these scenarios.
