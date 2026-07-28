# LargestCompany daily report - 2026-07-28

## Verdict: FAVORABLE  (edge +13.0%)
- Resolution 2026-08-31 | option expiry 2026-08-28 (31 days) | traded NVDA | side auto-picked **NO** @ 0.53
- Fragility (Phase 7): **MEDIUM - some assumptions move it (convergence / tail dependence)**
- Data: Yahoo live

## Edge: market vs simulation
| | Value |
|---|---|
| Polymarket says (NO) | 53.0% |
| Simulation says (NO fair) | 66.0% |
| Edge (fair - price) | **+13.0%** |
| Grade | **FAVORABLE** (>5% favorable, 0-5% marginal, <=0 unfavorable) |

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $14.51 |
| Payoff SD | $40.82 |
| VaR 5% | $44.41 |
| VaR 1% | $85.13 |
| Worst case | $92.89 |
| Probability of profit | 66.0% |
| Return on VaR 5% | 32.7% |
| Return on VaR 1% | 17.0% |
| Return on worst case | 15.6% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 0.60 | 66% | +0.36 | $64.98 | +32.7% | $14.51 |
| 1/1/4/4 | 0.51 | 56% | +0.36 | $68.45 | +34.7% | $14.95 |
| 2/2/3/3 | 0.43 | 55% | +0.35 | $66.54 | +37.1% | $14.73 |
| 2/2/4/4 | 0.25 | 49% | +0.35 | $70.01 | +39.5% | $15.17 |

## Consistency check (across assumptions & simulation reruns)
Overall fragility: **MEDIUM - some assumptions move it (convergence / tail dependence)**. P(NVDA #1) ranges 31.3%-34.3% across models (spread 3.1%).
| Check | Result |
|---|---|
| Across simulation reruns (seeds) | Central metrics are converged; trust them. |
| Across models & tails (IV surface / ATM / copula) | Edge keeps its sign across every model - tradeable. |
| Tail dependence (joint crashes) | Edge is sensitive to tail dependence (change -21.8%). |
| Dominant lever | gap-dominated (structural) (IV range 0.137 vs gap range 0.163). |

P(NVDA #1) by model:
| Model | P(#1) |
|---|---|
| IV surface + Gaussian copula | 34.0% |
| ATM lognormal + Normal | 34.3% |
| ATM lognormal + Student-t copula df=5 | 31.3% |
| ATM lognormal + Student-t df=6 (fat marginals) | 33.1% |

## Watch-outs
- No material risk flags on these scenarios.