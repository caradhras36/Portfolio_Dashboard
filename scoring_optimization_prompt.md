# CSP Scoring Optimization Prompt

## Current Scoring Weights
```python
SCORING_WEIGHTS = {
    'ivr': 0.15,           # IV Rank importance
    'greeks': 0.10,        # Greeks (Delta, Theta, Vega)
    'technical': 0.20,     # Technical indicators
    'liquidity': 0.05,     # Open interest and volume
    'resistance': 0.05,    # Support/resistance levels
    'probability': 0.20,   # Probability of profit
    'roi': 0.20,           # Return on Investment
    'cash_required': 0.05  # Cash required penalty
}
```

## Optimization Goals
1. **Prioritize high annualized returns** - This should be the primary factor
2. **Favor technical indicators that support selling puts** - RSI oversold, bullish patterns, etc.
3. **Weight resistance/support levels heavily** - Strong support below strike price
4. **Maintain probability of profit** - Still important for risk management
5. **Consider Greeks for risk assessment** - Delta, Theta, Vega analysis

## Suggested New Scoring Weights
```python
SCORING_WEIGHTS = {
    'annualized_return': 0.30,  # Primary factor - high returns
    'technical': 0.25,          # Technical indicators supporting put selling
    'resistance': 0.20,         # Support/resistance levels (strong support)
    'probability': 0.15,        # Probability of profit (risk management)
    'greeks': 0.05,            # Greeks for risk assessment
    'ivr': 0.03,               # IV Rank (reduced - less important)
    'liquidity': 0.02,         # Open interest and volume (minimal)
    'cash_required': 0.00      # Remove cash penalty (not relevant for scoring)
}
```

## Technical Indicators for Put Selling
- **RSI < 30** (oversold) - Good for put selling
- **RSI 30-50** (neutral to oversold) - Acceptable
- **RSI > 70** (overbought) - Bad for put selling
- **Price above 20-day MA** - Bullish trend
- **Price above 50-day MA** - Strong bullish trend
- **Volume increasing** - Confirmation of move

## Resistance/Support Scoring
- **Strong support 10-20% below strike** - Excellent (score 90-100)
- **Moderate support 5-10% below strike** - Good (score 70-89)
- **Weak support 0-5% below strike** - Fair (score 50-69)
- **No clear support** - Poor (score 0-49)

## Annualized Return Scoring
- **> 50% annualized** - Excellent (score 90-100)
- **30-50% annualized** - Good (score 70-89)
- **15-30% annualized** - Fair (score 50-69)
- **< 15% annualized** - Poor (score 0-49)

## Implementation Notes
1. **Normalize all scores to 0-100** for consistent weighting
2. **Apply exponential scaling** for annualized returns (higher returns get exponentially better scores)
3. **Combine technical indicators** into a single score (RSI + MA + Volume)
4. **Use resistance analysis** to find the strongest support level below strike
5. **Weight probability** to ensure we don't take excessive risk

## Example Calculation
```
Annualized Return: 45% → Score: 85
Technical (RSI=25, above MA): → Score: 90
Resistance (strong support 15% below): → Score: 95
Probability: 75% → Score: 75
Greeks: Good delta/theta → Score: 80

Final Score = (85×0.30) + (90×0.25) + (95×0.20) + (75×0.15) + (80×0.05) + (70×0.03) + (60×0.02)
Final Score = 25.5 + 22.5 + 19.0 + 11.25 + 4.0 + 2.1 + 1.2 = 85.55
```

This approach prioritizes high-return opportunities with strong technical and fundamental support for put selling.
