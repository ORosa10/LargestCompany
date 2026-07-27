# LargestCompany daily report - 2026-07-27

## Verdict: MARGINAL  (edge +2.5%)
- Target 2026-07-31 (4 days left) | traded NVDA | side auto-picked **NO** @ 0.24
- Fragility (Phase 7): **HIGH - edge is model-dependent (flips sign across models)**
- Data: Yahoo live

## Edge: market vs simulation
| | Value |
|---|---|
| Polymarket says (NO) | 24.0% |
| Simulation says (NO fair) | 26.5% |
| Edge (fair - price) | **+2.5%** |
| Grade | **MARGINAL** (>5% favorable, 0-5% marginal, <=0 unfavorable) |

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $2.47 |
| Payoff SD | $37.94 |
| VaR 5% | $36.47 |
| VaR 1% | $45.88 |
| Worst case | $64.27 |
| Probability of profit | 26.5% |
| Return on VaR 5% | 6.8% |
| Return on VaR 1% | 5.4% |
| Return on worst case | 3.8% |

## Structure comparison (weights: put/put/call/call)
| Weights | Score | EV/SD | RoCaR | RoC/ES5% | P(win) | Expected | Max loss |
|---|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 0.62 | +0.07 | +3.8% | +5.9% | 27% | $2.47 | $64.27 |
| 1/1/4/4 | 0.60 | +0.07 | +3.3% | +5.3% | 27% | $2.47 | $74.07 |
| 2/2/3/3 | 0.37 | +0.07 | +3.2% | +5.3% | 27% | $2.43 | $75.14 |
| 2/2/4/4 | 0.35 | +0.07 | +2.9% | +4.8% | 27% | $2.43 | $84.94 |

## Consistency check (across assumptions & simulation reruns)
Overall fragility: **HIGH - edge is model-dependent (flips sign across models)**. P(NVDA #1) ranges 72.9%-76.8% across models (spread 3.8%).
| Check | Result |
|---|---|
| Across simulation reruns (seeds) | Central metrics are NOT converged - raise the simulation count. |
| Across models & tails (IV surface / ATM / copula) | Edge flips sign across models - model-dependent, caution. |
| Tail dependence (joint crashes) | Edge is sensitive to tail dependence (change +97.7%). |
| Dominant lever | randomness-dominated (IV lever) (IV range 0.280 vs gap range 0.275). |

P(NVDA #1) by model:
| Model | P(#1) |
|---|---|
| IV surface + Gaussian copula | 73.5% |
| ATM lognormal + Normal | 72.9% |
| ATM lognormal + Student-t copula df=5 | 76.8% |
| ATM lognormal + Student-t df=6 (fat marginals) | 75.1% |

## Watch-outs
- Outcome is strongly IV-driven; the edge leans on the implied-volatility assumption. Get IV right before sizing up.
- Edge changes sign somewhere in the model grid: not robust to the model choice.
- This saved portfolio is lightly hedged (probability of loss 73%); the option legs are barely active. Re-run on the fully hedged Phase 5/6 structure before trusting the tail metrics.