# LargestCompany daily report - 2026-08-03

## Verdict: MARGINAL  (edge +3.0%)
- Resolution 2026-08-31 | option expiry 2026-08-28 (25 days) | traded GOOGL | side auto-picked **NO** @ 0.85
- Fragility (Phase 7): **HIGH - edge is model-dependent (flips sign across models)**
- Data: Yahoo live

## Summary
- Expected profit **$3.07** on $129.04 capital at risk (RoCaR 2.4%).
- Side auto-picked **NO** by composite (P(win), EV/SD, CVaR5%, RoC/VaR5%); naked YES EV -3.5% vs naked NO EV +3.0%.
- Your NO edge: model P(GOOGL NOT #1) 87.9% vs NO price 85% -> +3.0%.
- Robustness: edge is model-dependent (flips sign across models) - treat the direction as uncertain.
- Current market caps (Yahoo, model ranks by these): NVDA $4.98T, GOOGL $4.49T, AAPL $4.49T.

## Earnings before resolution (data caveat)
- **NVDA** reports: 2026-08-26
Heads-up: Yahoo spot/caps use the last **regular** close, so on/after these dates the after-hours earnings move is not yet in the ranking - treat P(#1) and edge as stale for up to a session around them.

## Edge: market vs simulation
| | Value |
|---|---|
| Polymarket says (NO) | 84.9% |
| Simulation says (NO fair) | 87.9% |
| Edge (fair - price) | **+3.0%** |
| Grade | **MARGINAL** (>5% favorable, 0-5% marginal, <=0 unfavorable) |

## Trade candidates (best edge per side)
| | Ticker | Model fair | Price | Edge | Composite |
|---|---|---|---|---|---|
| Best YES | NVDA | 74.2% | 72% | +2.2% | 0.25 |
| Best NO (traded) | GOOGL | 87.9% | 85% | +3.0% | 0.75 |
Default = max composite: **GOOGL NO** (composite 0.75, edge +3.0%). Composite = P(win), EV/SD, CVaR5%, RoC/VaR5% - risk-adjusted, not raw edge. Both sides get a full risk block below.

## Probability by name (model vs market)
| Ticker | Model P(#1) | Market YES | Market NO | YES edge | NO edge |
|---|---|---|---|---|---|
| NVDA | 74.2% | 72% | 30% | +2.2% | -4.2% |
| AAPL | 13.7% | 15% | 86% | -1.3% | +0.3% |
| GOOGL * | 12.1% | 16% | 85% | -3.5% | +3.0% |
* = traded name.

---
# Best YES trade: NVDA YES @ 0.72
Edge +2.2% | Verdict MARGINAL | Fragility HIGH

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $1.94 |
| Payoff SD | $37.73 |
| VaR 5% | $48.02 |
| VaR 1% | $76.95 |
| Worst case | $113.03 |
| Probability of profit | 51.9% |
| Return on VaR 5% | 4.1% |
| Return on VaR 1% | 2.5% |
| Return on worst case | 1.7% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 0.50 | 52% | +0.05 | $63.93 | +4.1% | $1.94 |
| 2/2/4/4 | 0.50 | 45% | +0.04 | $57.68 | +4.8% | $1.79 |
| 2/2/3/3 | 0.43 | 50% | +0.05 | $61.34 | +4.2% | $1.84 |
| 1/1/4/4 | 0.43 | 47% | +0.05 | $62.98 | +4.5% | $1.89 |

## Consistency check (across assumptions & simulation reruns)
Overall fragility: **HIGH - edge is model-dependent (flips sign across models)**. P(NVDA #1) ranges 73.0%-77.9% across models (spread 4.9%).
| Check | Result |
|---|---|
| Across simulation reruns (seeds) | Central metrics are NOT converged - raise the simulation count. |
| Across models & tails (IV surface / ATM / copula) | Edge flips sign across models - model-dependent, caution. |
| Tail dependence (joint crashes) | Edge is sensitive to tail dependence (change +445.1%). |
| Dominant lever | randomness-dominated (IV lever) (IV range 0.410 vs gap range 0.392). |

P(NVDA #1) by model:
| Model | P(#1) |
|---|---|
| IV surface + Gaussian copula | 74.2% |
| ATM lognormal + Normal | 73.0% |
| ATM lognormal + Student-t copula df=5 | 77.9% |
| ATM lognormal + Student-t df=6 (fat marginals) | 75.7% |

## Watch-outs
- Outcome is strongly IV-driven; the edge leans on the implied-volatility assumption. Get IV right before sizing up.
- Edge changes sign somewhere in the model grid: not robust to the model choice.
- This saved portfolio is lightly hedged (probability of loss 49%); the option legs are barely active. Re-run on the fully hedged Phase 5/6 structure before trusting the tail metrics.

---
# Best NO trade: GOOGL NO @ 0.85 (traded / app preset)
Edge +3.0% | Verdict MARGINAL | Fragility HIGH

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $3.07 |
| Payoff SD | $35.34 |
| VaR 5% | $70.85 |
| VaR 1% | $105.77 |
| Worst case | $129.04 |
| Probability of profit | 65.1% |
| Return on VaR 5% | 4.3% |
| Return on VaR 1% | 2.9% |
| Return on worst case | 2.4% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 0.75 | 65% | +0.09 | $90.65 | +4.3% | $3.07 |
| 2/2/3/3 | 0.61 | 64% | +0.08 | $91.01 | +4.5% | $3.05 |
| 1/1/4/4 | 0.36 | 59% | +0.08 | $92.45 | +4.6% | $3.10 |
| 2/2/4/4 | 0.25 | 58% | +0.08 | $92.81 | +4.8% | $3.08 |

## Consistency check (across assumptions & simulation reruns)
Overall fragility: **HIGH - edge is model-dependent (flips sign across models)**. P(GOOGL #1) ranges 11.1%-13.6% across models (spread 2.5%).
| Check | Result |
|---|---|
| Across simulation reruns (seeds) | Central metrics are NOT converged - raise the simulation count. |
| Across models & tails (IV surface / ATM / copula) | Edge flips sign across models - model-dependent, caution. |
| Tail dependence (joint crashes) | Edge is sensitive to tail dependence (change -131.8%). |
| Dominant lever | randomness-dominated (IV lever) (IV range 0.204 vs gap range 0.195). |

P(GOOGL #1) by model:
| Model | P(#1) |
|---|---|
| IV surface + Gaussian copula | 12.1% |
| ATM lognormal + Normal | 13.6% |
| ATM lognormal + Student-t copula df=5 | 11.1% |
| ATM lognormal + Student-t df=6 (fat marginals) | 12.2% |

## Watch-outs
- Outcome is strongly IV-driven; the edge leans on the implied-volatility assumption. Get IV right before sizing up.
- Edge changes sign somewhere in the model grid: not robust to the model choice.
