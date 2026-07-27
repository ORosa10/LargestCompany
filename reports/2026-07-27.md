# LargestCompany daily report - 2026-07-27

## Verdict: FAVORABLE  (edge +54.8%)
- Target 2026-07-31 (4 days left) | traded NVDA | side auto-picked **NO** @ 0.24
- Fragility (Phase 7): **MEDIUM - some assumptions move it (convergence / tail dependence)**
- Data: Yahoo live

## Edge: market vs simulation
| | Value |
|---|---|
| Polymarket says (NO) | 24.0% |
| Simulation says (NO fair) | 78.8% |
| Edge (fair - price) | **+54.8%** |
| Grade | **FAVORABLE** (>5% favorable, 0-5% marginal, <=0 unfavorable) |

## Best structure: 2/2/4/4 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $54.68 |
| Payoff SD | $37.03 |
| VaR 5% | $13.66 |
| VaR 1% | $27.89 |
| Worst case | $65.80 |
| Probability of profit | 78.8% |
| Return on VaR 5% | 400.2% |
| Return on VaR 1% | 196.1% |
| Return on worst case | 83.1% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 2/2/4/4 (best) | 0.68 | 79% | +1.48 | $22.23 | +400.2% | $54.68 |
| 1/1/4/4 | 0.49 | 79% | +1.49 | $23.15 | +358.4% | $54.72 |
| 2/2/3/3 | 0.48 | 79% | +1.47 | $22.22 | +354.0% | $54.68 |
| 1/1/3/3 | 0.19 | 79% | +1.48 | $23.13 | +320.9% | $54.72 |

## Consistency check (across assumptions & simulation reruns)
Overall fragility: **MEDIUM - some assumptions move it (convergence / tail dependence)**. P(NVDA #1) ranges 17.1%-21.2% across models (spread 4.2%).
| Check | Result |
|---|---|
| Across simulation reruns (seeds) | Central metrics are converged; trust them. |
| Across models & tails (IV surface / ATM / copula) | Edge keeps its sign across every model - tradeable. |
| Tail dependence (joint crashes) | Edge is sensitive to tail dependence (change -6.6%). |
| Dominant lever | gap-dominated (structural) (IV range 0.281 vs gap range 0.285). |

P(NVDA #1) by model:
| Model | P(#1) |
|---|---|
| IV surface + Gaussian copula | 21.2% |
| ATM lognormal + Normal | 20.8% |
| ATM lognormal + Student-t copula df=5 | 17.1% |
| ATM lognormal + Student-t df=6 (fat marginals) | 19.1% |

## Watch-outs
- Worst-case payoff is noisy (+/- 4.5, 7% relative). Size against Expected shortfall / P1, or raise simulations for a firm worst case.