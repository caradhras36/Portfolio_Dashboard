# Risk Score Calculation Methodology

## Overview
All risk scores are on a 0-100 scale, where higher scores indicate higher risk.

## Current Risk Score Formulas

### 1. Concentration Risk Score (0-100)

**Formula**: Uses Herfindahl-Hirschman Index (HHI)

```python
# Calculate weight of each position
weight_i = (position_value_i / portfolio_value)

# HHI = sum of squared weights
HHI = Σ(weight_i²)

# Convert to 0-100 scale
concentration_score = HHI × 100
```

**Examples**:
- **10 equal positions** (10% each): HHI = 10 × (0.10²) = 0.10 → Score: **10** (Low risk)
- **1 position at 50%, rest small**: HHI ≈ 0.25 → Score: **25** (Medium risk)
- **1 position at 100%**: HHI = 1.0 → Score: **100** (Critical risk)

**Interpretation**:
- 0-20: Well diversified
- 20-40: Moderate concentration
- 40-60: High concentration
- 60-100: Critical concentration

---

### 2. Volatility Risk Score (0-100)

**Current Formula** (Simple):
```python
options_ratio = options_count / total_positions
volatility_score = options_ratio × 100
```

**Examples**:
- **All stocks (0% options)**: Score: **0** (Low volatility)
- **50% stocks, 50% options**: Score: **50** (Medium volatility)
- **All options (100% options)**: Score: **100** (High volatility)

**Better Formula** (Improved):
```python
# Factor in multiple volatility drivers:
# 1. Options ratio (0-40 points)
options_factor = (options_count / total_positions) × 40

# 2. Short-dated options (<30 days) (0-30 points)
short_dated_ratio = short_dated_options / total_options
short_dated_factor = short_dated_ratio × 30

# 3. Concentrated positions (0-30 points)
concentration_factor = max_position_weight × 30

volatility_score = min(100, options_factor + short_dated_factor + concentration_factor)
```

---

### 3. Liquidity Risk Score (0-100)

**Current**: Placeholder (30)

**Better Formula**:
```python
# For each position:
position_volume_ratio = position_value / (daily_volume × price)

# Score = average of all ratios
liquidity_score = average(position_volume_ratios) × 100

# Interpretation:
# < 1%: Highly liquid (score: 0-10)
# 1-5%: Adequate (score: 10-50)
# 5-10%: Concerning (score: 50-75)
# > 10%: Poor liquidity (score: 75-100)
```

**Example**:
- Stock worth $10k, daily volume = $5M → Ratio = 0.2% → Score: **2** (Highly liquid)
- Stock worth $50k, daily volume = $100k → Ratio = 50% → Score: **100** (Very illiquid)

---

### 4. Overall Risk Score (0-100)

**Formula**: Weighted average of component scores

```python
overall_score = (
    concentration_score × 0.40 +  # 40% weight
    volatility_score × 0.35 +      # 35% weight
    liquidity_score × 0.25          # 25% weight
) / 100 × 100
```

**Example Calculation**:
```
Concentration: 35/100  (moderate)
Volatility:    60/100  (high - lots of options)
Liquidity:     25/100  (good)

Overall = (35×0.40 + 60×0.35 + 25×0.25)
        = (14 + 21 + 6.25)
        = 41.25
        → Score: 41 (Medium Risk)
```

---

### 5. Stock Concentration Risk Score (0-100)

**Formula**: HHI for stocks only

```python
# Calculate for stocks only
stock_portfolio_value = sum(stock values)
weight_i = stock_value_i / stock_portfolio_value

HHI_stocks = Σ(weight_i²)
stock_concentration_score = HHI_stocks × 100
```

**Example**:
- 5 stocks, largest is 30% → HHI ≈ 0.24 → Score: **24**
- 3 stocks: 50%, 30%, 20% → HHI = 0.38 → Score: **38**

---

### 6. Time Decay Risk Score (0-100)

**Formula**: Based on days to expiry and theta

```python
# Time factor (closer to expiry = higher risk)
time_factor = max(0, 100 - (avg_days_to_expiry / 60 × 100))

# Theta factor (higher theta decay = higher risk)
theta_factor = min(100, abs(total_theta_per_day) / 10)

# Combined score
time_decay_risk_score = (time_factor + theta_factor) / 2
```

**Examples**:
- **60+ days to expiry, -$100/day theta**: 
  - time_factor = 0, theta_factor = 10
  - Score: **5** (Low risk)

- **30 days to expiry, -$500/day theta**:
  - time_factor = 50, theta_factor = 50  
  - Score: **50** (Medium risk)

- **7 days to expiry, -$1000/day theta**:
  - time_factor = 88, theta_factor = 100
  - Score: **94** (Critical risk)

---

## Risk Level Interpretation

### Overall Risk Score → Risk Level:
```python
if score < 25:   return "Low"        # 🟢 Green
if score < 50:   return "Medium"     # 🟡 Yellow
if score < 75:   return "High"       # 🟠 Orange
else:            return "Critical"   # 🔴 Red
```

### What Each Level Means:

**Low Risk (0-24)**: 
- Well diversified portfolio
- Mostly stocks or long-dated options
- Good liquidity
- **Action**: Maintain current strategy

**Medium Risk (25-49)**:
- Moderate concentration
- Mix of stocks and options
- Acceptable for growth portfolios
- **Action**: Monitor regularly

**High Risk (50-74)**:
- High concentration or many options
- Significant time decay exposure
- **Action**: Consider risk reduction

**Critical Risk (75-100)**:
- Extremely concentrated positions
- Heavy short-dated options exposure
- Liquidity concerns
- **Action**: Immediate rebalancing recommended

---

## Improved Calculation Examples

### Example Portfolio:

**Positions**:
- AAPL: $40,000 (40%)
- TSLA: $30,000 (30%)
- MSFT: $20,000 (20%)
- GOOGL Call (30d): $5,000 (5%, delta=0.6, theta=-$50/day)
- NVDA Put (7d): $5,000 (5%, delta=-0.7, theta=-$100/day)

**Total Portfolio**: $100,000

### Calculations:

**1. Concentration Risk**:
```
HHI = 0.40² + 0.30² + 0.20² + 0.05² + 0.05²
    = 0.16 + 0.09 + 0.04 + 0.0025 + 0.0025
    = 0.295
Score = 29.5 ≈ 30 (Medium concentration)
```

**2. Volatility Risk**:
```
Options ratio = 2/5 = 40%
Score = 40 (Medium volatility)
```

**3. Liquidity Risk**:
```
Assuming all highly liquid (FAANG stocks)
Score = 15 (Low liquidity risk)
```

**4. Time Decay Risk** (Options only):
```
Avg days to expiry = (30 + 7) / 2 = 18.5 days
Total theta = -$50 + -$100 = -$150/day

time_factor = 100 - (18.5/60 × 100) = 69
theta_factor = 150/10 = 15

Score = (69 + 15) / 2 = 42 (Medium time decay risk)
```

**5. Overall Risk**:
```
Overall = (30×0.40 + 40×0.35 + 15×0.25)
        = 12 + 14 + 3.75
        = 29.75
        ≈ 30 → Medium Risk 🟡
```

---

## Advanced Formulas (For Future Implementation)

### Real Volatility Score:
```python
# Using actual historical volatility
portfolio_vol = sqrt(Σ(weight_i² × vol_i² + 2 × Σ(weight_i × weight_j × correlation_ij × vol_i × vol_j)))
volatility_score = min(100, portfolio_vol / 0.5 × 100)  # Normalize to 50% max vol
```

### Real Liquidity Score:
```python
# Days to liquidate portfolio
for each position:
    days_to_liquidate = position_value / (daily_volume × 0.25)  # Assume 25% of volume max

liquidity_score = weighted_average(days_to_liquidate) × 20  # 5 days = score 100
```

### Beta-Adjusted Risk:
```python
# Account for market sensitivity
beta_adjustment = abs(portfolio_beta - 1.0) × 20
overall_score = base_score + beta_adjustment
```

---

## Thresholds Reference

```python
thresholds = {
    'concentration_warning': 0.15,    # 15% in single position
    'concentration_critical': 0.25,   # 25% in single position
    'volatility_warning': 0.30,       # 30% annual volatility
    'volatility_critical': 0.50,      # 50% annual volatility
    'delta_warning': 0.3,             # 30% delta exposure
    'delta_critical': 0.5,            # 50% delta exposure
    'gamma_warning': 0.1,
    'gamma_critical': 0.2,
    'theta_warning': -500,            # -$500/day
    'theta_critical': -1000,          # -$1000/day
    'expiration_warning': 7,          # 7 days
    'expiration_critical': 3,         # 3 days
}
```

---

## Summary

**Risk scores are calculated using**:
1. **Mathematical formulas** (HHI, ratios, weighted averages)
2. **Industry-standard thresholds** (concentration limits, time decay)
3. **Portfolio theory** (diversification benefits)
4. **Practical experience** (what actually matters in trading)

The scores provide **quick visual assessment** while tooltips give **detailed context** for interpretation.

---

**Last Updated**: September 30, 2025
**Version**: 1.0
**Status**: ✅ Production
