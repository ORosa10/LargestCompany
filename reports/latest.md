# LargestCompany daily report - 2026-07-30

## Verdict: FAVORABLE  (edge +17.2%)
- Resolution 2026-08-31 | option expiry 2026-08-28 (29 days) | traded AAPL | side auto-picked **YES** @ 0.58
- Fragility (Phase 7): **MEDIUM - some assumptions move it (convergence / tail dependence)**
- Data: Yahoo live

## Summary
- Expected profit **$15.05** on $101.65 capital at risk (RoCaR 14.8%).
- Side auto-picked **YES**: naked YES EV +17.2% vs naked NO EV -19.2% (traded = higher EV).
- Your YES edge: model P(AAPL #1) 75.2% vs YES price 58% -> +17.2%.
- Robustness: probability estimate is NOT fully robust (stability of the estimate, not the trade direction).

## Edge: market vs simulation
| | Value |
|---|---|
| Polymarket says (YES) | 58.0% |
| Simulation says (YES fair) | 75.2% |
| Edge (fair - price) | **+17.2%** |
| Grade | **FAVORABLE** (>5% favorable, 0-5% marginal, <=0 unfavorable) |

## Trade candidates (best edge per side)
| | Ticker | Model fair | Price | Edge |
|---|---|---|---|---|
| Best YES (traded) | AAPL | 75.2% | 58% | +17.2% |
| Best NO | NVDA | 76.6% | 61% | +15.6% |
Traded = max edge: **AAPL YES** (+17.2%). Both sides get a full risk block below. NO is more robust to an unmodeled surprise winner; YES is more direct but optimistic given the 3-name universe.

## Probability by name (model vs market)
| Ticker | Model P(#1) | Market YES | Market NO | YES edge | NO edge |
|---|---|---|---|---|---|
| AAPL * | 75.2% | 58% | 44% | +17.2% | -19.2% |
| NVDA | 23.4% | 40% | 61% | -16.6% | +15.6% |
| GOOGL | 1.4% | 4% | 97% | -2.6% | +1.7% |
* = traded name.

---
# Best YES trade: AAPL YES @ 0.58 (traded / app preset)
Edge +17.2% | Verdict FAVORABLE | Fragility MEDIUM

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $15.05 |
| Payoff SD | $43.77 |
| VaR 5% | $68.51 |
| VaR 1% | $94.18 |
| Worst case | $101.65 |
| Probability of profit | 74.2% |
| Return on VaR 5% | 22.0% |
| Return on VaR 1% | 16.0% |
| Return on worst case | 14.8% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 1.00 | 74% | +0.34 | $83.80 | +22.0% | $15.05 |
| 2/2/3/3 | 0.54 | 65% | +0.33 | $89.28 | +21.3% | $14.75 |
| 1/1/4/4 | 0.32 | 64% | +0.32 | $90.58 | +20.1% | $14.43 |
| 2/2/4/4 | 0.00 | 62% | +0.30 | $96.05 | +19.5% | $14.13 |

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
# Best NO trade: NVDA NO @ 0.61
Edge +15.6% | Verdict FAVORABLE | Fragility MEDIUM

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $16.95 |
| Payoff SD | $39.23 |
| VaR 5% | $36.16 |
| VaR 1% | $85.43 |
| Worst case | $101.15 |
| Probability of profit | 67.5% |
| Return on VaR 5% | 46.9% |
| Return on VaR 1% | 19.8% |
| Return on worst case | 16.8% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 0.75 | 68% | +0.43 | $56.47 | +46.9% | $16.95 |
| 2/2/3/3 | 0.42 | 60% | +0.41 | $58.01 | +54.8% | $17.13 |
| 1/1/4/4 | 0.37 | 57% | +0.43 | $59.84 | +58.7% | $17.32 |
| 2/2/4/4 | 0.28 | 55% | +0.40 | $59.45 | +71.1% | $17.51 |

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
