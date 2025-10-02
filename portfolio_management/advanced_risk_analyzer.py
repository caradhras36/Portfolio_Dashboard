#!/usr/bin/env python3
"""
Advanced Portfolio Risk Analysis System
Provides comprehensive risk insights for both stocks and options
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import math
from scipy import stats
from scipy.stats import norm
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class StockRiskMetrics:
    """Stock-specific risk metrics"""
    # Concentration Risk
    max_position_weight: float
    top_5_concentration: float
    sector_concentration: Dict[str, float]
    
    # Volatility Risk
    portfolio_volatility: float
    weighted_avg_volatility: float
    volatility_range: Tuple[float, float]
    
    # Beta Risk
    portfolio_beta: float
    beta_exposure: float
    
    # Liquidity Risk
    avg_daily_volume: float
    liquidity_score: float
    
    # Correlation Risk
    avg_correlation: float
    correlation_risk_score: float

@dataclass
class OptionsRiskMetrics:
    """Options-specific risk metrics"""
    # Greeks Risk
    delta_exposure: float
    gamma_risk: float
    theta_decay: float
    vega_sensitivity: float
    
    # Time Decay Risk
    avg_days_to_expiry: float
    time_decay_risk: float
    expiration_risk: float
    
    # Volatility Risk
    implied_vol_skew: float
    vol_risk_score: float
    
    # Strike Risk
    otm_percentage: float
    itm_percentage: float
    atm_percentage: float
    
    # Strategy Risk
    call_put_ratio: float
    strategy_concentration: Dict[str, float]

@dataclass
class PortfolioRiskInsights:
    """Comprehensive portfolio risk insights"""
    # Overall Risk
    total_risk_score: float
    risk_level: str  # Low, Medium, High, Critical
    
    # Stock Risk
    stock_risk: StockRiskMetrics
    
    # Options Risk
    options_risk: OptionsRiskMetrics
    
    # Portfolio Risk
    diversification_score: float
    concentration_risk: float
    liquidity_risk: float
    
    # Risk Alerts
    risk_alerts: List[str]
    recommendations: List[str]

class AdvancedRiskAnalyzer:
    """Advanced portfolio risk analysis with comprehensive insights"""
    
    def __init__(self):
        # Risk thresholds
        self.risk_thresholds = {
            'concentration_high': 0.20,  # 20% max position
            'concentration_critical': 0.30,  # 30% max position
            'volatility_high': 0.30,  # 30% annual vol
            'volatility_critical': 0.50,  # 50% annual vol
            'beta_high': 1.5,  # 1.5 beta
            'beta_critical': 2.0,  # 2.0 beta
            'correlation_high': 0.7,  # 70% correlation
            'liquidity_low': 100000,  # $100k daily volume
            'gamma_high': 0.1,  # 0.1 gamma
            'vega_high': 0.2,  # 0.2 vega
            'theta_high': -0.05,  # -0.05 theta
        }
    
    def analyze_stock_risk(self, positions: List[Any]) -> StockRiskMetrics:
        """Analyze stock-specific risk metrics"""
        stocks = [pos for pos in positions if pos.position_type == 'stock']
        
        if not stocks:
            return StockRiskMetrics(
                max_position_weight=0, top_5_concentration=0, sector_concentration={},
                portfolio_volatility=0, weighted_avg_volatility=0, volatility_range=(0, 0),
                portfolio_beta=0, beta_exposure=0, avg_daily_volume=0, liquidity_score=0,
                avg_correlation=0, correlation_risk_score=0
            )
        
        # Calculate position weights
        total_value = sum(pos.quantity * pos.current_price for pos in stocks)
        position_weights = [(pos.quantity * pos.current_price) / total_value for pos in stocks]
        
        # Concentration Risk
        max_position_weight = max(position_weights) if position_weights else 0
        sorted_weights = sorted(position_weights, reverse=True)
        top_5_concentration = sum(sorted_weights[:5]) if len(sorted_weights) >= 5 else sum(sorted_weights)
        
        # Sector concentration (simplified - would need sector data)
        sector_concentration = self._calculate_sector_concentration(stocks)
        
        # Volatility Risk (simplified calculation)
        portfolio_volatility = self._calculate_portfolio_volatility(stocks)
        weighted_avg_volatility = self._calculate_weighted_volatility(stocks)
        volatility_range = self._calculate_volatility_range(stocks)
        
        # Beta Risk (simplified - would need beta data)
        portfolio_beta = self._calculate_portfolio_beta(stocks)
        beta_exposure = portfolio_beta * total_value
        
        # Liquidity Risk (simplified - would need volume data)
        avg_daily_volume = self._estimate_avg_volume(stocks)
        liquidity_score = self._calculate_liquidity_score(stocks, avg_daily_volume)
        
        # Correlation Risk (simplified)
        avg_correlation = self._estimate_avg_correlation(stocks)
        correlation_risk_score = self._calculate_correlation_risk(avg_correlation)
        
        return StockRiskMetrics(
            max_position_weight=max_position_weight,
            top_5_concentration=top_5_concentration,
            sector_concentration=sector_concentration,
            portfolio_volatility=portfolio_volatility,
            weighted_avg_volatility=weighted_avg_volatility,
            volatility_range=volatility_range,
            portfolio_beta=portfolio_beta,
            beta_exposure=beta_exposure,
            avg_daily_volume=avg_daily_volume,
            liquidity_score=liquidity_score,
            avg_correlation=avg_correlation,
            correlation_risk_score=correlation_risk_score
        )
    
    def analyze_options_risk(self, positions: List[Any]) -> OptionsRiskMetrics:
        """Analyze options-specific risk metrics"""
        options = [pos for pos in positions if pos.position_type in ['call', 'put']]
        
        if not options:
            return OptionsRiskMetrics(
                delta_exposure=0, gamma_risk=0, theta_decay=0, vega_sensitivity=0,
                avg_days_to_expiry=0, time_decay_risk=0, expiration_risk=0,
                implied_vol_skew=0, vol_risk_score=0, otm_percentage=0, itm_percentage=0, atm_percentage=0,
                call_put_ratio=0, strategy_concentration={}
            )
        
        # Greeks Risk
        delta_exposure = sum(pos.delta * pos.quantity for pos in options if pos.delta)
        gamma_risk = sum(pos.gamma * pos.quantity for pos in options if pos.gamma)
        theta_decay = sum(pos.theta * pos.quantity for pos in options if pos.theta)
        vega_sensitivity = sum(pos.vega * pos.quantity for pos in options if pos.vega)
        
        # Time Decay Risk
        days_to_expiry = [pos.time_to_expiration for pos in options if pos.time_to_expiration]
        avg_days_to_expiry = sum(days_to_expiry) / len(days_to_expiry) if days_to_expiry else 0
        time_decay_risk = self._calculate_time_decay_risk(options)
        expiration_risk = self._calculate_expiration_risk(options)
        
        # Volatility Risk
        implied_vols = [pos.implied_volatility for pos in options if pos.implied_volatility]
        implied_vol_skew = self._calculate_vol_skew(implied_vols)
        vol_risk_score = self._calculate_vol_risk_score(implied_vols)
        
        # Strike Risk
        strike_analysis = self._analyze_strike_distribution(options)
        
        # Strategy Risk
        call_put_ratio = self._calculate_call_put_ratio(options)
        strategy_concentration = self._analyze_strategy_concentration(options)
        
        return OptionsRiskMetrics(
            delta_exposure=delta_exposure,
            gamma_risk=gamma_risk,
            theta_decay=theta_decay,
            vega_sensitivity=vega_sensitivity,
            avg_days_to_expiry=avg_days_to_expiry,
            time_decay_risk=time_decay_risk,
            expiration_risk=expiration_risk,
            implied_vol_skew=implied_vol_skew,
            vol_risk_score=vol_risk_score,
            otm_percentage=strike_analysis['otm_pct'],
            itm_percentage=strike_analysis['itm_pct'],
            atm_percentage=strike_analysis['atm_pct'],
            call_put_ratio=call_put_ratio,
            strategy_concentration=strategy_concentration
        )
    
    def generate_risk_insights(self, positions: List[Any]) -> PortfolioRiskInsights:
        """Generate comprehensive risk insights"""
        stock_risk = self.analyze_stock_risk(positions)
        options_risk = self.analyze_options_risk(positions)
        
        # Calculate overall risk metrics
        total_risk_score = self._calculate_total_risk_score(stock_risk, options_risk)
        risk_level = self._determine_risk_level(total_risk_score)
        
        # Portfolio-level risks
        diversification_score = self._calculate_diversification_score(positions)
        concentration_risk = self._calculate_portfolio_concentration_risk(positions)
        liquidity_risk = self._calculate_portfolio_liquidity_risk(positions)
        
        # Generate alerts and recommendations
        risk_alerts = self._generate_risk_alerts(stock_risk, options_risk, positions)
        recommendations = self._generate_recommendations(stock_risk, options_risk, positions)
        
        return PortfolioRiskInsights(
            total_risk_score=total_risk_score,
            risk_level=risk_level,
            stock_risk=stock_risk,
            options_risk=options_risk,
            diversification_score=diversification_score,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
            risk_alerts=risk_alerts,
            recommendations=recommendations
        )
    
    def _calculate_sector_concentration(self, stocks: List[Any]) -> Dict[str, float]:
        """Calculate sector concentration (simplified)"""
        # This would need actual sector data
        # For now, return empty dict
        return {}
    
    def _calculate_portfolio_volatility(self, stocks: List[Any]) -> float:
        """Calculate portfolio volatility (simplified)"""
        # This would need historical returns data
        # For now, use a simplified estimate
        return 0.20  # 20% annual volatility estimate
    
    def _calculate_weighted_volatility(self, stocks: List[Any]) -> float:
        """Calculate weighted average volatility"""
        # Simplified calculation
        return 0.25  # 25% weighted average
    
    def _calculate_volatility_range(self, stocks: List[Any]) -> Tuple[float, float]:
        """Calculate volatility range"""
        return (0.15, 0.35)  # 15% to 35% range
    
    def _calculate_portfolio_beta(self, stocks: List[Any]) -> float:
        """Calculate portfolio beta (simplified)"""
        # This would need actual beta data
        return 1.0  # Market beta
    
    def _estimate_avg_volume(self, stocks: List[Any]) -> float:
        """Estimate average daily volume"""
        # This would need actual volume data
        return 1000000  # $1M average
    
    def _calculate_liquidity_score(self, stocks: List[Any], avg_volume: float) -> float:
        """Calculate liquidity score (0-1)"""
        # Simplified calculation
        return min(1.0, avg_volume / 10000000)  # Normalize to $10M
    
    def _estimate_avg_correlation(self, stocks: List[Any]) -> float:
        """Estimate average correlation between stocks"""
        # This would need historical correlation data
        return 0.3  # 30% average correlation
    
    def _calculate_correlation_risk(self, avg_correlation: float) -> float:
        """Calculate correlation risk score (0-1)"""
        return min(1.0, avg_correlation / 0.8)  # Normalize to 80% max correlation
    
    def _calculate_time_decay_risk(self, options: List[Any]) -> float:
        """Calculate time decay risk"""
        if not options:
            return 0.0
        
        # Calculate weighted time decay based on days to expiry
        total_risk = 0.0
        total_weight = 0.0
        
        for pos in options:
            if pos.time_to_expiration and pos.theta:
                # Higher risk for shorter time to expiry
                time_factor = 1.0 / max(1, pos.time_to_expiration / 30)  # Normalize to 30 days
                weight = abs(pos.quantity * pos.current_price)
                total_risk += abs(pos.theta) * time_factor * weight
                total_weight += weight
        
        return total_risk / total_weight if total_weight > 0 else 0.0
    
    def _calculate_expiration_risk(self, options: List[Any]) -> float:
        """Calculate expiration risk"""
        if not options:
            return 0.0
        
        # Risk increases as expiration approaches
        expiring_soon = [pos for pos in options if pos.time_to_expiration and pos.time_to_expiration <= 7]
        return len(expiring_soon) / len(options)
    
    def _calculate_vol_skew(self, implied_vols: List[float]) -> float:
        """Calculate implied volatility skew"""
        if len(implied_vols) < 2:
            return 0.0
        
        return np.std(implied_vols) / np.mean(implied_vols) if np.mean(implied_vols) > 0 else 0.0
    
    def _calculate_vol_risk_score(self, implied_vols: List[float]) -> float:
        """Calculate volatility risk score"""
        if not implied_vols:
            return 0.0
        
        avg_vol = np.mean(implied_vols)
        return min(1.0, avg_vol / 0.5)  # Normalize to 50% max vol
    
    def _analyze_strike_distribution(self, options: List[Any]) -> Dict[str, float]:
        """Analyze strike price distribution"""
        if not options:
            return {'otm_pct': 0, 'itm_pct': 0, 'atm_pct': 0}
        
        otm_count = 0
        itm_count = 0
        atm_count = 0
        
        for pos in options:
            if pos.strike_price and pos.current_price:
                if pos.position_type == 'call':
                    if pos.strike_price > pos.current_price:
                        otm_count += 1
                    elif pos.strike_price < pos.current_price:
                        itm_count += 1
                    else:
                        atm_count += 1
                elif pos.position_type == 'put':
                    if pos.strike_price < pos.current_price:
                        otm_count += 1
                    elif pos.strike_price > pos.current_price:
                        itm_count += 1
                    else:
                        atm_count += 1
        
        total = len(options)
        return {
            'otm_pct': (otm_count / total) * 100 if total > 0 else 0,
            'itm_pct': (itm_count / total) * 100 if total > 0 else 0,
            'atm_pct': (atm_count / total) * 100 if total > 0 else 0
        }
    
    def _calculate_call_put_ratio(self, options: List[Any]) -> float:
        """Calculate call/put ratio"""
        calls = len([pos for pos in options if pos.position_type == 'call'])
        puts = len([pos for pos in options if pos.position_type == 'put'])
        return calls / puts if puts > 0 else float('inf')
    
    def _analyze_strategy_concentration(self, options: List[Any]) -> Dict[str, float]:
        """Analyze strategy concentration"""
        # This would need more sophisticated strategy detection
        # For now, return basic breakdown
        return {
            'calls': len([pos for pos in options if pos.position_type == 'call']) / len(options),
            'puts': len([pos for pos in options if pos.position_type == 'put']) / len(options)
        }
    
    def _calculate_total_risk_score(self, stock_risk: StockRiskMetrics, options_risk: OptionsRiskMetrics) -> float:
        """Calculate total risk score (0-1)"""
        # Weighted combination of various risk factors
        stock_score = (
            stock_risk.max_position_weight * 0.3 +
            stock_risk.portfolio_volatility * 0.2 +
            (1 - stock_risk.liquidity_score) * 0.2 +
            stock_risk.correlation_risk_score * 0.3
        )
        
        options_score = (
            abs(options_risk.gamma_risk) * 0.25 +
            abs(options_risk.vega_sensitivity) * 0.25 +
            options_risk.time_decay_risk * 0.25 +
            options_risk.vol_risk_score * 0.25
        )
        
        return min(1.0, (stock_score + options_score) / 2)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score < 0.3:
            return "Low"
        elif risk_score < 0.6:
            return "Medium"
        elif risk_score < 0.8:
            return "High"
        else:
            return "Critical"
    
    def _calculate_diversification_score(self, positions: List[Any]) -> float:
        """Calculate diversification score (0-1)"""
        if not positions:
            return 0.0
        
        # Count unique tickers
        unique_tickers = len(set(pos.ticker for pos in positions))
        return min(1.0, unique_tickers / 10)  # Normalize to 10 tickers max
    
    def _calculate_portfolio_concentration_risk(self, positions: List[Any]) -> float:
        """Calculate portfolio concentration risk"""
        if not positions:
            return 0.0
        
        total_value = sum(pos.quantity * pos.current_price for pos in positions)
        position_weights = [(pos.quantity * pos.current_price) / total_value for pos in positions]
        
        # Calculate Herfindahl-Hirschman Index
        hhi = sum(w**2 for w in position_weights)
        return hhi  # 0 = perfectly diversified, 1 = completely concentrated
    
    def _calculate_portfolio_liquidity_risk(self, positions: List[Any]) -> float:
        """Calculate portfolio liquidity risk"""
        # Simplified calculation
        return 0.2  # 20% liquidity risk estimate
    
    def _generate_risk_alerts(self, stock_risk: StockRiskMetrics, options_risk: OptionsRiskMetrics, positions: List[Any]) -> List[str]:
        """Generate risk alerts"""
        alerts = []
        
        # Stock risk alerts
        if stock_risk.max_position_weight > self.risk_thresholds['concentration_critical']:
            alerts.append(f"CRITICAL: Single position represents {stock_risk.max_position_weight:.1%} of portfolio")
        elif stock_risk.max_position_weight > self.risk_thresholds['concentration_high']:
            alerts.append(f"WARNING: Single position represents {stock_risk.max_position_weight:.1%} of portfolio")
        
        if stock_risk.portfolio_volatility > self.risk_thresholds['volatility_critical']:
            alerts.append(f"CRITICAL: Portfolio volatility is {stock_risk.portfolio_volatility:.1%}")
        elif stock_risk.portfolio_volatility > self.risk_thresholds['volatility_high']:
            alerts.append(f"WARNING: Portfolio volatility is {stock_risk.portfolio_volatility:.1%}")
        
        if stock_risk.liquidity_score < 0.3:
            alerts.append("WARNING: Low liquidity positions detected")
        
        # Options risk alerts
        if abs(options_risk.gamma_risk) > self.risk_thresholds['gamma_high']:
            alerts.append(f"WARNING: High gamma exposure: {options_risk.gamma_risk:.2f}")
        
        if abs(options_risk.vega_sensitivity) > self.risk_thresholds['vega_high']:
            alerts.append(f"WARNING: High vega sensitivity: {options_risk.vega_sensitivity:.2f}")
        
        if options_risk.theta_decay < self.risk_thresholds['theta_high']:
            alerts.append(f"WARNING: High theta decay: {options_risk.theta_decay:.2f}")
        
        if options_risk.expiration_risk > 0.5:
            alerts.append("WARNING: High concentration of expiring options")
        
        return alerts
    
    def _generate_recommendations(self, stock_risk: StockRiskMetrics, options_risk: OptionsRiskMetrics, positions: List[Any]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Stock recommendations
        if stock_risk.max_position_weight > 0.15:
            recommendations.append("Consider reducing position size in largest holdings")
        
        if stock_risk.diversification_score < 0.5:
            recommendations.append("Increase diversification by adding more positions")
        
        if stock_risk.liquidity_score < 0.5:
            recommendations.append("Consider adding more liquid positions")
        
        # Options recommendations
        if abs(options_risk.gamma_risk) > 0.05:
            recommendations.append("Consider reducing gamma exposure through hedging")
        
        if options_risk.time_decay_risk > 0.3:
            recommendations.append("Consider rolling out expiring options")
        
        if options_risk.vol_risk_score > 0.7:
            recommendations.append("Consider volatility hedging strategies")
        
        if options_risk.call_put_ratio > 2.0 or options_risk.call_put_ratio < 0.5:
            recommendations.append("Consider balancing call/put ratio")
        
        return recommendations

# Example usage
if __name__ == "__main__":
    analyzer = AdvancedRiskAnalyzer()
    print("Advanced Risk Analyzer initialized successfully")
