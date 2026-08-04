# LargestCompany daily report - 2026-08-04

## Verdict: MARGINAL  (edge +3.9%)
- Resolution 2026-08-31 | option expiry 2026-08-28 (24 days) | traded GOOGL | side auto-picked **YES** @ 0.10
- Fragility (Phase 7): **MEDIUM - edge holds except the crisis corner (high corr + joint crashes)**
- Data: Yahoo live

## Summary
- Expected profit **$3.80** on $54.87 capital at risk (RoCaR 6.9%).
- Side auto-picked **YES** by composite (P(win), EV/SD, CVaR5%, RoC/VaR5%); naked YES EV +3.9% vs naked NO EV -4.0%.
- Your YES edge: model P(GOOGL #1) 14.3% vs YES price 10% -> +3.9%.
- Robustness: probability estimate is NOT fully robust (stability of the estimate, not the trade direction).
- Current market caps (Yahoo, model ranks by these): NVDA $5.09T, GOOGL $4.59T, AAPL $4.45T.

## Earnings before resolution (data caveat)
- **NVDA** reports: 2026-08-26
Heads-up: Yahoo spot/caps use the last **regular** close, so on/after these dates the after-hours earnings move is not yet in the ranking - treat P(#1) and edge as stale for up to a session around them.

## Edge: market vs simulation
| | Value |
|---|---|
| Polymarket says (YES) | 10.4% |
| Simulation says (YES fair) | 14.3% |
| Edge (fair - price) | **+3.9%** |
| Grade | **MARGINAL** (>5% favorable, 0-5% marginal, <=0 unfavorable) |

## Trade candidates (best edge per side)
| | Ticker | Model fair | Price | Edge | Composite |
|---|---|---|---|---|---|
| Best YES (traded) | GOOGL | 14.3% | 10% | +3.9% | 0.50 |
| Best NO | AAPL | 92.5% | 91% | +1.5% | 0.50 |
Default = max composite: **GOOGL YES** (composite 0.50, edge +3.9%). Composite = P(win), EV/SD, CVaR5%, RoC/VaR5% - risk-adjusted, not raw edge. Both sides get a full risk block below.

## Probability by name (model vs market)
| Ticker | Model P(#1) | Market YES | Market NO | YES edge | NO edge |
|---|---|---|---|---|---|
| NVDA | 78.3% | 79% | 22% | -0.7% | -0.3% |
| AAPL | 7.5% | 10% | 91% | -2.5% | +1.5% |
| GOOGL * | 14.3% | 10% | 90% | +3.9% | -4.0% |
* = traded name.

---
# Best YES trade: GOOGL YES @ 0.10 (traded / app preset)
Edge +3.9% | Verdict MARGINAL | Fragility MEDIUM

## Best structure: 1/1/3/3 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $3.80 |
| Payoff SD | $37.31 |
| VaR 5% | $46.49 |
| VaR 1% | $52.71 |
| Worst case | $54.87 |
| Probability of profit | 53.1% |
| Return on VaR 5% | 8.2% |
| Return on VaR 1% | 7.2% |
| Return on worst case | 6.9% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/3/3 (best) | 0.75 | 53% | +0.10 | $49.45 | +8.2% | $3.80 |
| 2/2/3/3 | 0.52 | 57% | +0.10 | $60.98 | +6.8% | $3.76 |
| 1/1/4/4 | 0.46 | 55% | +0.10 | $58.63 | +6.8% | $3.80 |
| 2/2/4/4 | 0.25 | 58% | +0.09 | $70.15 | +5.8% | $3.75 |

## Sensitivity & stress
Base edge (surface + est. correlation + Gaussian copula): **+3.9%**. Fragility: **MEDIUM - edge holds except the crisis corner (high corr + joint crashes)**. Dominant lever: **correlation**. Edge across the realistic stress band: -2.2% to +5.5%.
Simulation reruns (seeds): Central metrics are NOT converged - raise the simulation count.

**Volatility / marginals** (correlation = est., Gaussian copula; changes only how each name moves):
| Marginal model | P(#1) | Edge | delta vs base |
|---|---|---|---|
| IV surface (base) | 14.3% | +3.9% | base |
| ATM lognormal + Normal | 15.6% | +5.2% | +1.3 pp |
| ATM + Student-t df=6 (fat marginals) | 14.1% | +3.7% | -0.1 pp |
| ATM + Student-t df=10 | 15.0% | +4.6% | +0.8 pp |

**Correlation x tail copula** (marginals = surface; base cell marked):
| correlation \ tail | Gaussian | Student-t df=5 |
|---|---|---|
| 0.3 (calm) | +5.5% (+1.6) | +4.6% (+0.7) |
| ~est (Saved) | +3.9% (base) | +3.0% (-0.9) |
| 0.8 (crisis) | -2.2% (-6.1) | -2.0% (-5.9) |

Reading: down = more correlation, right = joint crashes (tail dependence). () = edge change in pp vs base.

## Watch-outs
- Outcome is strongly IV-driven; the edge leans on the implied-volatility assumption. Get IV right before sizing up.
- Edge changes sign somewhere in the model grid: not robust to the model choice.
- This saved portfolio is lightly hedged (probability of loss 45%); the option legs are barely active. Re-run on the fully hedged Phase 5/6 structure before trusting the tail metrics.

---
# Best NO trade: AAPL NO @ 0.91
Edge +1.5% | Verdict MARGINAL | Fragility LOW

## Best structure: 1/1/4/4 (put/put/call/call)
| Metric | Value |
|---|---|
| Expected profit | $3.66 |
| Payoff SD | $33.57 |
| VaR 5% | $64.73 |
| VaR 1% | $105.42 |
| Worst case | $142.01 |
| Probability of profit | 64.0% |
| Return on VaR 5% | 5.6% |
| Return on VaR 1% | 3.5% |
| Return on worst case | 2.6% |

## Structure comparison (weights: put/put/call/call)
Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.
| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |
|---|---|---|---|---|---|---|
| 1/1/4/4 (best) | 0.67 | 64% | +0.11 | $79.60 | +5.6% | $3.66 |
| 2/2/4/4 | 0.55 | 64% | +0.11 | $83.73 | +6.4% | $3.93 |
| 2/2/3/3 | 0.44 | 66% | +0.10 | $82.58 | +5.1% | $3.46 |
| 1/1/3/3 | 0.25 | 66% | +0.10 | $85.81 | +4.5% | $3.19 |

## Sensitivity & stress
Base edge (surface + est. correlation + Gaussian copula): **+1.5%**. Fragility: **LOW - edge holds across the realistic stress band**. Dominant lever: **correlation**. Edge across the realistic stress band: +0.3% to +5.7%.
Simulation reruns (seeds): Central metrics are NOT converged - raise the simulation count.

**Volatility / marginals** (correlation = est., Gaussian copula; changes only how each name moves):
| Marginal model | P(#1) | Edge | delta vs base |
|---|---|---|---|
| IV surface (base) | 7.5% | +1.5% | base |
| ATM lognormal + Normal | 7.0% | +2.0% | +0.5 pp |
| ATM + Student-t df=6 (fat marginals) | 6.1% | +2.9% | +1.3 pp |
| ATM + Student-t df=10 | 6.6% | +2.4% | +0.9 pp |

**Correlation x tail copula** (marginals = surface; base cell marked):
| correlation \ tail | Gaussian | Student-t df=5 |
|---|---|---|
| 0.3 (calm) | +0.3% (-1.2) | +0.7% (-0.8) |
| ~est (Saved) | +1.5% (base) | +1.9% (+0.4) |
| 0.8 (crisis) | +5.6% (+4.1) | +5.7% (+4.2) |

Reading: down = more correlation, right = joint crashes (tail dependence). () = edge change in pp vs base.

## Watch-outs
- Outcome is strongly IV-driven; the edge leans on the implied-volatility assumption. Get IV right before sizing up.
- This saved portfolio is lightly hedged (probability of loss 41%); the option legs are barely active. Re-run on the fully hedged Phase 5/6 structure before trusting the tail metrics.
