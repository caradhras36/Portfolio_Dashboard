#!/usr/bin/env python3
"""
Risk Metrics Tooltips
Provides detailed explanations and examples for all risk metrics
"""

# Risk metric tooltips with definitions and examples
RISK_TOOLTIPS = {
    "overall_risk_score": {
        "title": "Overall Risk Score",
        "description": "A composite score (0-100) measuring your portfolio's overall risk level. Considers concentration, volatility, and liquidity risks combined.",
        "interpretation": "0-25: Low Risk | 25-50: Medium Risk | 50-75: High Risk | 75-100: Critical Risk",
        "example": "A score of 65 means your portfolio has high risk, likely due to concentration in few positions or high volatility."
    },
    "concentration_risk": {
        "title": "Concentration Risk",
        "description": "Measures how much of your portfolio is concentrated in a few positions. Uses the Herfindahl-Hirschman Index (HHI).",
        "interpretation": "Higher scores indicate more concentration risk. Above 50 suggests too much exposure to single positions.",
        "example": "If 40% of your portfolio is in one stock, concentration risk will be high (60+), suggesting you should diversify."
    },
    "volatility_risk": {
        "title": "Volatility Risk",
        "description": "Measures the degree of price fluctuation in your portfolio. Higher volatility means larger price swings.",
        "interpretation": "Low (<25): Stable | Medium (25-50): Moderate swings | High (50+): Large price fluctuations",
        "example": "A portfolio with many options will have higher volatility risk than one with only blue-chip stocks."
    },
    "liquidity_risk": {
        "title": "Liquidity Risk",
        "description": "Measures how easily you can exit positions without significantly affecting prices. Based on trading volume.",
        "interpretation": "Low score: Easy to exit | High score: Difficult to exit positions quickly",
        "example": "Holding large positions in low-volume stocks increases liquidity risk - you can't sell quickly without moving the price."
    },
    "portfolio_beta": {
        "title": "Portfolio Beta",
        "description": "Measures your portfolio's sensitivity to overall market movements. Beta of 1.0 means you move with the market.",
        "interpretation": "Beta < 1: Less volatile than market | Beta = 1: Moves with market | Beta > 1: More volatile than market",
        "example": "Beta of 1.5 means if the market goes up 10%, your portfolio likely goes up 15%. Same for downside."
    },
    "portfolio_volatility": {
        "title": "Portfolio Volatility",
        "description": "Annual volatility (standard deviation) of portfolio returns. Indicates how much returns vary from average.",
        "interpretation": "10-15%: Low | 15-25%: Moderate | 25-40%: High | 40%+: Very High",
        "example": "30% volatility means ~68% of the time, your annual return will be within ±30% of your average return."
    },
    "sharpe_ratio": {
        "title": "Sharpe Ratio",
        "description": "Risk-adjusted return measure. Shows how much extra return you get per unit of risk taken.",
        "interpretation": "< 0: Poor | 0-1: Subpar | 1-2: Good | 2-3: Very Good | 3+: Excellent",
        "example": "Sharpe of 1.5 means you earn 1.5% return for every 1% of risk. Higher is better."
    },
    "sortino_ratio": {
        "title": "Sortino Ratio",
        "description": "Like Sharpe ratio but only considers downside volatility. Penalizes bad volatility, not good volatility.",
        "interpretation": "Higher is better. Typically higher than Sharpe ratio since it ignores upside volatility.",
        "example": "Sortino of 2.0 means good risk-adjusted returns considering only downside risk."
    },
    "max_drawdown": {
        "title": "Maximum Drawdown",
        "description": "The largest peak-to-trough decline in portfolio value. Shows worst historical loss from a high point.",
        "interpretation": "10%: Low | 10-20%: Moderate | 20-30%: High | 30%+: Very High",
        "example": "15% max drawdown means your portfolio fell 15% from its peak at worst. You need 17.6% gain to recover."
    },
    "var_95": {
        "title": "Value at Risk (95%)",
        "description": "Maximum expected loss over 1 day with 95% confidence. In other words, 95% of days, losses won't exceed this.",
        "interpretation": "Shows your 'bad day' loss threshold. Only 5% of days should see losses exceeding this amount.",
        "example": "VaR 95% of $5,000 means on 95% of days, you won't lose more than $5,000."
    },
    "var_99": {
        "title": "Value at Risk (99%)",
        "description": "Maximum expected loss over 1 day with 99% confidence. Your 'very bad day' loss threshold.",
        "interpretation": "More extreme than VaR 95%. Only 1% of days (about 2-3 trading days/year) should exceed this loss.",
        "example": "VaR 99% of $8,000 means only 2-3 days per year should you expect losses over $8,000."
    },
    "expected_shortfall": {
        "title": "Expected Shortfall (CVaR)",
        "description": "Average loss on days when VaR threshold is exceeded. Also called Conditional Value at Risk (CVaR).",
        "interpretation": "Shows how bad things get when they go really bad. Higher values mean more tail risk.",
        "example": "If VaR is $5k and CVaR is $7k, when losses exceed $5k, they average $7k."
    },
    "total_delta": {
        "title": "Total Portfolio Delta",
        "description": "Net directional exposure of all options. Positive = bullish, negative = bearish. Measures market direction risk.",
        "interpretation": "+50: Equivalent to 50 shares long | -30: Equivalent to 30 shares short | 0: Market neutral",
        "example": "Delta of +100 means your options behave like 100 shares of stock. A 1% market move affects you like 100 shares."
    },
    "total_gamma": {
        "title": "Total Portfolio Gamma",
        "description": "Rate of change of delta. High gamma means delta changes quickly as price moves. Indicates position stability.",
        "interpretation": "Low (<5): Stable delta | Medium (5-15): Moderate changes | High (15+): Delta changes rapidly",
        "example": "Gamma of 10 means if stock moves $1, your delta changes by 10. Short-dated options have higher gamma."
    },
    "total_theta": {
        "title": "Total Portfolio Theta",
        "description": "Daily time decay in dollars. Shows how much value you lose per day just from passage of time.",
        "interpretation": "Negative = Losing money to time decay | Positive = Earning money from time decay",
        "example": "Theta of -$200/day means you lose $200 daily to time decay if nothing else changes. Over 30 days: -$6,000."
    },
    "total_vega": {
        "title": "Total Portfolio Vega",
        "description": "Sensitivity to implied volatility changes. Shows P&L impact if IV changes by 1%.",
        "interpretation": "High positive vega: Profit from volatility increase | High negative vega: Profit from volatility decrease",
        "example": "Vega of +500 means if IV increases 1%, you gain $500. If IV drops 5%, you lose $2,500."
    },
    "theta_30d": {
        "title": "30-Day Theta Decay",
        "description": "Total expected time decay over the next 30 days, assuming no other changes.",
        "interpretation": "Shows your time decay exposure over a month. Critical for option sellers.",
        "example": "30-day theta of -$6,000 means you'll lose $6k to time decay over the next month if prices don't move."
    },
    "expiring_7d_value": {
        "title": "Expiring in 7 Days",
        "description": "Total notional value of options expiring within 7 days. Requires immediate attention.",
        "interpretation": "High values indicate urgent action needed - must close, roll, or let expire.",
        "example": "$15k expiring in 7 days means you have $15k of options that need decisions very soon."
    },
    "expiring_30d_value": {
        "title": "Expiring in 30 Days",
        "description": "Total notional value of options expiring within 30 days. Should start planning actions.",
        "interpretation": "Shows positions entering the high time-decay zone. Plan to roll or close soon.",
        "example": "$40k expiring in 30 days means you should start considering which positions to roll or close."
    },
    "top_5_concentration": {
        "title": "Top 5 Concentration",
        "description": "Percentage of portfolio in your 5 largest positions. Measures concentration risk.",
        "interpretation": "< 50%: Well diversified | 50-70%: Moderate concentration | 70%+: High concentration risk",
        "example": "75% in top 5 means if those 5 positions perform poorly, your entire portfolio suffers significantly."
    },
    "liquidity_score": {
        "title": "Liquidity Score",
        "description": "Composite score (0-100%) measuring how easily you can exit all positions. Based on volume and position size.",
        "interpretation": "80-100%: Highly liquid | 50-80%: Adequate | 30-50%: Concerning | <30%: Poor liquidity",
        "example": "65% liquidity means you can exit most positions quickly, but some may take time or move the price."
    },
    "positions_at_risk": {
        "title": "Positions at Risk",
        "description": "Number of positions exceeding risk thresholds (concentration >15%, high volatility, expiring soon).",
        "interpretation": "0: All positions healthy | 1-3: Monitor closely | 4+: Consider reducing exposure",
        "example": "3 positions at risk might include: 1 concentrated position (25% of portfolio), 2 expiring in 3 days."
    },
    "expiring_soon": {
        "title": "Expiring Soon Count",
        "description": "Number of option positions expiring within 7 days. These require immediate attention.",
        "interpretation": "0-2: Manageable | 3-5: Busy week ahead | 6+: Very high management workload",
        "example": "5 options expiring soon means you need to decide this week: close, roll, or let expire on each."
    },
    "max_position_risk": {
        "title": "Maximum Position Size",
        "description": "Percentage of portfolio in your single largest position. Key concentration risk metric.",
        "interpretation": "<10%: Well diversified | 10-20%: Acceptable | 20-30%: High risk | 30%+: Critical risk",
        "example": "28% in largest position means if that stock drops 20%, your entire portfolio drops 5.6%."
    }
}

def get_tooltip_html(metric_key: str) -> str:
    """Generate HTML for a risk metric tooltip"""
    if metric_key not in RISK_TOOLTIPS:
        return ""
    
    tooltip = RISK_TOOLTIPS[metric_key]
    
    return f"""
        <span class="tooltip-container">
            <span class="tooltip-icon">?</span>
            <div class="tooltip-content">
                <div class="tooltip-title">{tooltip['title']}</div>
                <div class="tooltip-description">{tooltip['description']}</div>
                <div class="tooltip-example">
                    <div class="tooltip-example-title">Interpretation:</div>
                    {tooltip['interpretation']}
                </div>
                <div class="tooltip-example">
                    <div class="tooltip-example-title">Example:</div>
                    {tooltip['example']}
                </div>
            </div>
        </span>
    """
