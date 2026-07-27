# LargestCompany daily report - 2026-07-27

## Verdict: FAVORABLE  (edge +12.6%)
- Target 2026-07-31 (4 days left) | traded NVDA | side auto-picked **NO** @ 0.24
- Fragility (Phase 7): **MEDIUM - some assumptions move it (convergence / tail dependence)**
- Data: Yahoo live

## Edge: market vs simulation
| | Value |
|---|---|
| Polymarket says (NO) | 24.0% |
| Simulation says (NO fair) | 36.6% |
| Edge (fair - price) | **+12.6%** |
| Grade | **FAVORABLE** (>5% favorable, 0-5% marginal, <=0 unfavorable) |

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $12.57 |
| Payoff SD | $41.70 |
| VaR 5% | $33.19 |
| VaR 1% | $42.34 |
| Worst case | $64.11 |
| Probability of profit | 36.6% |
| Return on VaR 5% | 37.9% |
| Return on VaR 1% | 29.7% |
| Return on worst case | 19.6% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 0.50 | 37% | +0.30 | $38.81 | +37.9% | $12.57 |
| 2/2/4/4 | 0.50 | 47% | +0.31 | $45.52 | +33.8% | $12.53 |
| 2/2/3/3 | 0.40 | 37% | +0.31 | $41.46 | +36.4% | $12.53 |
| 1/1/4/4 | 0.36 | 37% | +0.31 | $42.87 | +35.1% | $12.57 |

## Consistency check (across assumptions & simulation reruns)
Overall fragility: **MEDIUM - some assumptions move it (convergence / tail dependence)**. P(NVDA #1) ranges 62.4%-64.9% across models (spread 2.6%).
| Check | Result |
|---|---|
| Across simulation reruns (seeds) | Central metrics are converged; trust them. |
| Across models & tails (IV surface / ATM / copula) | Edge keeps its sign across every model - tradeable. |
| Tail dependence (joint crashes) | Edge is sensitive to tail dependence (change +18.7%). |
| Dominant lever | randomness-dominated (IV lever) (IV range 0.185 vs gap range 0.179). |

P(NVDA #1) by model:
| Model | P(#1) |
|---|---|
| IV surface + Gaussian copula | 63.4% |
| ATM lognormal + Normal | 62.4% |
| ATM lognormal + Student-t copula df=5 | 64.9% |
| ATM lognormal + Student-t df=6 (fat marginals) | 63.8% |

## Watch-outs
- Outcome is strongly IV-driven; the edge leans on the implied-volatility assumption. Get IV right before sizing up.
- This saved portfolio is lightly hedged (probability of loss 63%); the option legs are barely active. Re-run on the fully hedged Phase 5/6 structure before trusting the tail metrics.