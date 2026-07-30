# LargestCompany daily report - 2026-07-30

## Verdict: FAVORABLE  (edge +17.2%)
- Resolution 2026-08-31 | option expiry 2026-08-28 (29 days) | traded AAPL | side auto-picked **YES** @ 0.58
- Fragility (Phase 7): **MEDIUM - some assumptions move it (convergence / tail dependence)**
- Data: Yahoo live

## Summary
- Expected profit **$17.19** on $99.50 capital at risk (RoCaR 17.3%).
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
| Expected profit | $17.19 |
| Payoff SD | $43.77 |
| VaR 5% | $66.36 |
| VaR 1% | $92.03 |
| Worst case | $99.50 |
| Probability of profit | 75.2% |
| Return on VaR 5% | 25.9% |
| Return on VaR 1% | 18.7% |
| Return on worst case | 17.3% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 1.00 | 75% | +0.39 | $81.66 | +25.9% | $17.19 |
| 2/2/3/3 | 0.56 | 67% | +0.38 | $86.84 | +25.7% | $17.19 |
| 1/1/4/4 | 0.37 | 67% | +0.38 | $87.82 | +24.9% | $17.19 |
| 2/2/4/4 | 0.00 | 63% | +0.36 | $93.00 | +24.7% | $17.19 |

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

## Best structure: 2/2/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $15.64 |
| Payoff SD | $41.29 |
| VaR 5% | $32.75 |
| VaR 1% | $90.59 |
| Worst case | $117.74 |
| Probability of profit | 58.3% |
| Return on VaR 5% | 47.7% |
| Return on VaR 1% | 17.3% |
| Return on worst case | 13.3% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 2/2/3/3 (best) | 0.55 | 58% | +0.38 | $56.74 | +47.7% | $15.64 |
| 1/1/3/3 | 0.50 | 66% | +0.40 | $61.21 | +41.7% | $15.63 |
| 1/1/4/4 | 0.37 | 56% | +0.38 | $60.15 | +50.1% | $15.64 |
| 2/2/4/4 | 0.29 | 54% | +0.36 | $60.57 | +59.0% | $15.64 |

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
- This saved portfolio is lightly hedged (probability of loss 45%); the option legs are barely active. Re-run on the fully hedged Phase 5/6 structure before trusting the tail metrics.
