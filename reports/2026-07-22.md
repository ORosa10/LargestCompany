# LargestCompany daily report - 2026-07-22

Target 2026-07-31 (9 days) | traded NVDA NO @ 0.33 | 40,000 sims
Data: caps: Yahoo live; spots: fallback (ValueError)

## Verdict: UNFAVORABLE
- Expected profit $-9.65 on $96.43 capital at risk (RoCaR -10.0%).
- Your NO edge: model P(NVDA NOT #1) 23.2% vs NO price 33% -> -9.8%.
- Probability estimate is NOT fully robust across the Phase 7 tests (this is about the estimate's stability, not trade direction).

## Probability & edge
- Model P(NVDA #1) 76.8% vs Polymarket YES 68.0%. A NO bet wins when NVDA does NOT finish #1.
  - NVDA: model 76.8% | market YES 68.0%
  - AAPL: model 23.1% | market YES 30.9%
  - GOOGL: model 0.1% | market YES 1.7%

## Money view
- Expected profit: $-9.65
- Capital at risk (max loss): $96.43 | net cash $56.43
- Return on capital-at-risk: -10.0%
- Loss ladder: VaR5% $55.15 | VaR1% $73.31 | worst $96.43
- P(profit) 23.2% | P(loss) 76.8%

## Sensitivity
- 1. Monte Carlo error: Central metrics are converged; trust them.
- 2. Tail dependence: Edge is sensitive to tail dependence (change +43.8%).
- 3. Gap vs randomness: randomness-dominated (IV lever) (IV range 0.318 vs gap range 0.312).
- 5. Model robustness: Edge keeps its sign across every model - tradeable.

## Watch-outs
- Outcome is strongly IV-driven; the edge leans on the implied-volatility assumption. Get IV right before sizing up.
- This saved portfolio is lightly hedged (probability of loss 76%); the option legs are barely active. Re-run on the fully hedged Phase 5/6 structure before trusting the tail metrics.