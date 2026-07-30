# LargestCompany daily report - 2026-07-30

## Verdict: FAVORABLE  (edge +17.2%)
- Resolution 2026-08-31 | option expiry 2026-08-28 (29 days) | traded AAPL | side auto-picked **YES** @ 0.58
- Fragility (Phase 7): **MEDIUM - some assumptions move it (convergence / tail dependence)**
- Data: Yahoo live

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
Traded = max edge: **AAPL YES** (+17.2%). NO is more robust to an unmodeled surprise winner; YES is more direct but optimistic given the 3-name universe.

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