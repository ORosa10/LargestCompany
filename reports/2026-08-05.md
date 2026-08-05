# LargestCompany daily report - 2026-08-05

## Verdict: MARGINAL  (edge +2.3%)
- Resolution 2026-08-31 | option expiry 2026-08-28 (23 days) | traded AAPL | side auto-picked **NO** @ 0.93
- Fragility (Phase 7): **LOW - edge holds across the realistic stress band**
- Data: Yahoo live

## Summary
- Expected profit **$3.83** on $135.84 capital at risk (RoCaR 2.8%).
- Side auto-picked **NO** by composite (P(win), EV/SD, CVaR5%, RoC/VaR5%); naked YES EV -3.3% vs naked NO EV +2.3%.
- Your NO edge: model P(AAPL NOT #1) 95.3% vs NO price 93% -> +2.3%.
- Robustness: estimate is robust across the Phase 7 tests.
- Current market caps (Yahoo, model ranks by these): NVDA $5.31T, GOOGL $4.65T, AAPL $4.49T.

## Earnings before resolution (data caveat)
- **NVDA** reports: 2026-08-26
Heads-up: Yahoo spot/caps use the last **regular** close, so on/after these dates the after-hours earnings move is not yet in the ranking - treat P(#1) and edge as stale for up to a session around them.

## Edge: market vs simulation
| | Value |
|---|---|
| Polymarket says (NO) | 93.0% |
| Simulation says (NO fair) | 95.3% |
| Edge (fair - price) | **+2.3%** |
| Grade | **MARGINAL** (>5% favorable, 0-5% marginal, <=0 unfavorable) |

## Trade candidates (best edge per side)
| | Ticker | Model fair | Price | Edge | Composite |
|---|---|---|---|---|---|
| Best YES | NVDA | 85.7% | 84% | +1.7% | 0.25 |
| Best NO (traded) | AAPL | 95.3% | 93% | +2.3% | 0.75 |
Default = max composite: **AAPL NO** (composite 0.75, edge +2.3%). Composite = P(win), EV/SD, CVaR5%, RoC/VaR5% - risk-adjusted, not raw edge. Both sides get a full risk block below.

## Probability by name (model vs market)
| Ticker | Model P(#1) | Market YES | Market NO | YES edge | NO edge |
|---|---|---|---|---|---|
| NVDA | 85.7% | 84% | 17% | +1.7% | -2.7% |
| AAPL * | 4.7% | 8% | 93% | -3.3% | +2.3% |
| GOOGL | 9.5% | 8% | 92% | +1.3% | -1.5% |
* = traded name.

---
# Best YES trade: NVDA YES @ 0.84
Edge +1.7% | Verdict MARGINAL | Fragility HIGH

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $1.54 |
| Payoff SD | $34.20 |
| VaR 5% | $60.32 |
| VaR 1% | $65.82 |
| Worst case | $125.32 |
| Probability of profit | 54.3% |
| Return on VaR 5% | 2.6% |
| Return on VaR 1% | 2.3% |
| Return on worst case | 1.2% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 0.50 | 54% | +0.05 | $63.31 | +2.6% | $1.54 |
| 2/2/4/4 | 0.50 | 51% | +0.04 | $56.30 | +2.9% | $1.43 |
| 1/1/4/4 | 0.49 | 51% | +0.04 | $59.24 | +2.8% | $1.50 |
| 2/2/3/3 | 0.46 | 53% | +0.04 | $59.77 | +2.6% | $1.47 |

## Sensitivity & stress
Base edge (surface + est. correlation + Gaussian copula): **+1.7%**. Fragility: **HIGH - edge flips sign even at normal correlation / marginals**. Dominant lever: **correlation**. Edge across the realistic stress band: -0.9% to +9.4%.

**Volatility / marginals** (correlation = est., Gaussian copula; changes only how each name moves):
| Marginal model | P(#1) | Edge | delta vs base |
|---|---|---|---|
| IV surface (base) | 85.7% | +1.7% | base |
| ATM lognormal + Normal | 85.8% | +1.8% | +0.1 pp |
| ATM + Student-t df=6 (fat marginals) | 87.4% | +3.4% | +1.7 pp |
| ATM + Student-t df=10 | 86.5% | +2.5% | +0.7 pp |

**Correlation x tail copula** (marginals = surface; base cell marked):
| correlation \ tail | Gaussian | Student-t df=5 |
|---|---|---|
| 0.3 (calm) | -0.9% (-2.6) | +0.0% (-1.7) |
| ~est (Saved) | +1.7% (base) | +2.5% (+0.8) |
| 0.8 (crisis) | +9.4% (+7.6) | +9.2% (+7.5) |

Reading: down = more correlation, right = joint crashes (tail dependence). () = edge change in pp vs base.

Comment: the edge changes sign even at normal correlation - its direction is not reliable; do not size up on it.

Notes (unchanged checks):
- Simulation reruns (seeds): Central metrics are NOT converged - raise the simulation count.
- Gap vs randomness: randomness-dominated (IV lever) (IV range 0.361 vs gap range 0.346).

## Watch-outs
- Outcome is strongly IV-driven; the edge leans on the implied-volatility assumption. Get IV right before sizing up.
- Edge changes sign somewhere in the model grid: not robust to the model choice.
- This saved portfolio is lightly hedged (probability of loss 43%); the option legs are barely active. Re-run on the fully hedged Phase 5/6 structure before trusting the tail metrics.

---
# Best NO trade: AAPL NO @ 0.93 (traded / app preset)
Edge +2.3% | Verdict MARGINAL | Fragility LOW

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $3.83 |
| Payoff SD | $27.30 |
| VaR 5% | $37.69 |
| VaR 1% | $97.76 |
| Worst case | $135.84 |
| Probability of profit | 66.6% |
| Return on VaR 5% | 10.1% |
| Return on VaR 1% | 3.9% |
| Return on worst case | 2.8% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 0.87 | 67% | +0.14 | $76.89 | +10.1% | $3.83 |
| 1/1/4/4 | 0.61 | 65% | +0.14 | $73.98 | +9.1% | $4.26 |
| 2/2/3/3 | 0.33 | 66% | +0.14 | $79.48 | +7.4% | $4.08 |
| 2/2/4/4 | 0.11 | 65% | +0.13 | $77.48 | +7.0% | $4.51 |

## Sensitivity & stress
Base edge (surface + est. correlation + Gaussian copula): **+2.3%**. Fragility: **LOW - edge holds across the realistic stress band**. Dominant lever: **correlation**. Edge across the realistic stress band: +1.2% to +5.2%.

**Volatility / marginals** (correlation = est., Gaussian copula; changes only how each name moves):
| Marginal model | P(#1) | Edge | delta vs base |
|---|---|---|---|
| IV surface (base) | 4.7% | +2.3% | base |
| ATM lognormal + Normal | 4.1% | +2.9% | +0.7 pp |
| ATM + Student-t df=6 (fat marginals) | 3.6% | +3.4% | +1.1 pp |
| ATM + Student-t df=10 | 3.9% | +3.1% | +0.8 pp |

**Correlation x tail copula** (marginals = surface; base cell marked):
| correlation \ tail | Gaussian | Student-t df=5 |
|---|---|---|
| 0.3 (calm) | +1.2% (-1.1) | +1.5% (-0.7) |
| ~est (Saved) | +2.3% (base) | +2.5% (+0.2) |
| 0.8 (crisis) | +5.2% (+2.9) | +5.2% (+2.9) |

Reading: down = more correlation, right = joint crashes (tail dependence). () = edge change in pp vs base.

Comment: the edge stays positive across the whole realistic band - robust. Correlation is the main lever but never flips it.

Notes (unchanged checks):
- Simulation reruns (seeds): Central metrics are converged; trust them.
- Gap vs randomness: randomness-dominated (IV lever) (IV range 0.146 vs gap range 0.139).

## Watch-outs
- Outcome is strongly IV-driven; the edge leans on the implied-volatility assumption. Get IV right before sizing up.
