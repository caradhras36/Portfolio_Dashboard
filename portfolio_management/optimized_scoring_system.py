"""
Optimized Expected Value-Based Options Scoring System
Incorporates market regimes, portfolio context, and confidence-based delta selection
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class OptimizedOptionsScorer:
    """
    Advanced options scoring system based on Expected Value optimization
    Incorporates market regimes, portfolio context, and confidence-based selection
    """
    
    def __init__(self):
        # Different risk tolerances for CCs vs CSPs
        self.max_assignment_probability_cc = 0.35  # 35% max for covered calls (more conservative but reasonable)
        self.max_assignment_probability_csp = 0.50  # 50% max for CSPs (less conservative)
        self.min_dte = 7
        self.max_dte = 90
        
        # Market regime detection parameters
        self.vix_thresholds = {
            'low_vol': 15,
            'high_vol': 25
        }
        
        # Confidence calculation weights
        self.confidence_weights = {
            'rsi': 0.25,
            'macd': 0.20,
            'bollinger_bands': 0.20,
            'support_resistance': 0.15,
            'volume': 0.10,
            'momentum': 0.10
        }
    
    def detect_market_regime(self, market_data: Dict) -> Dict:
        """Detect current market regime using multiple indicators"""
        
        # VIX analysis
        vix = market_data.get('vix', 20)
        vix_20_ma = market_data.get('vix_20_ma', 20)
        
        # SPY trend analysis
        spy_price = market_data.get('spy_price', 400)
        spy_20_ma = market_data.get('spy_20_ma', 400)
        spy_50_ma = market_data.get('spy_50_ma', 400)
        
        # Market breadth
        advancers = market_data.get('advancers', 0.5)
        decliners = market_data.get('decliners', 0.5)
        
        regime_scores = {
            'bull_market': 0,
            'bear_market': 0,
            'sideways_market': 0,
            'high_volatility': 0,
            'low_volatility': 0
        }
        
        # Bull market indicators
        if spy_price > spy_20_ma > spy_50_ma:
            regime_scores['bull_market'] += 0.4
        if advancers > 0.6:
            regime_scores['bull_market'] += 0.3
        if vix < 20:
            regime_scores['bull_market'] += 0.3
        
        # Bear market indicators
        if spy_price < spy_20_ma < spy_50_ma:
            regime_scores['bear_market'] += 0.4
        if decliners > 0.6:
            regime_scores['bear_market'] += 0.3
        if vix > 30:
            regime_scores['bear_market'] += 0.3
        
        # Volatility regime
        if vix > 25:
            regime_scores['high_volatility'] = 0.8
        elif vix < 15:
            regime_scores['low_volatility'] = 0.8
        
        # Sideways market
        if 0.98 < spy_price / spy_20_ma < 1.02:
            regime_scores['sideways_market'] = 0.6
        
        # Determine primary regime
        primary_regime = max(regime_scores, key=regime_scores.get)
        
        return {
            'primary_regime': primary_regime,
            'regime_confidence': regime_scores[primary_regime],
            'regime_scores': regime_scores,
            'vix_level': vix
        }
    
    def calculate_enhanced_confidence(self, stock_data: Dict, market_regime: Dict) -> float:
        """Enhanced confidence calculation with multiple indicators"""
        
        confidence = 0.5  # Base confidence
        
        # RSI analysis
        rsi = stock_data.get('rsi', 50)
        if rsi > 75:  # Very overbought - excellent for call selling
            confidence += 0.25
        elif rsi > 65:
            confidence += 0.15
        elif rsi > 55:
            confidence += 0.05
        elif rsi < 45:  # Getting risky
            confidence -= 0.15
        elif rsi < 35:  # Very risky for call selling
            confidence -= 0.25
        
        # MACD analysis
        macd = stock_data.get('macd', {})
        macd_hist = macd.get('histogram', 0)
        
        if macd_hist < -0.5:  # Strong negative momentum
            confidence += 0.20
        elif macd_hist < 0:
            confidence += 0.10
        elif macd_hist > 0.5:  # Positive momentum - risky for calls
            confidence -= 0.15
        
        # Bollinger Bands position
        bb = stock_data.get('bollinger_bands', {})
        bb_position = bb.get('position', 0.5)
        if bb_position > 0.85:  # Near upper band
            confidence += 0.20
        elif bb_position > 0.7:
            confidence += 0.10
        elif bb_position < 0.3:  # Near lower band
            confidence -= 0.20
        
        # Support/Resistance analysis
        current_price = stock_data.get('current_price', 0)
        support_resistance = stock_data.get('support_resistance', {})
        resistance = support_resistance.get('resistance', current_price * 1.05)
        
        resistance_distance = (resistance - current_price) / current_price
        if resistance_distance > 0.05:  # Strong resistance above
            confidence += 0.15
        elif resistance_distance < 0.02:  # Close to resistance
            confidence -= 0.10
        
        # Volume analysis
        volume_ratio = stock_data.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:  # High volume - more conviction
            confidence += 0.10
        elif volume_ratio < 0.5:  # Low volume - less conviction
            confidence -= 0.05
        
        # Price momentum
        price_change_5d = stock_data.get('price_change_5d', 0)
        if price_change_5d < -0.02:  # Recent decline - good for call selling
            confidence += 0.10
        elif price_change_5d > 0.05:  # Strong recent gains - risky
            confidence -= 0.15
        
        # Market regime adjustment
        if market_regime['primary_regime'] == 'bull_market':
            confidence *= 1.1
        elif market_regime['primary_regime'] == 'bear_market':
            confidence *= 0.9
        elif market_regime['primary_regime'] == 'high_volatility':
            confidence *= 0.8
        
        return max(0.1, min(0.95, confidence))
    
    def calculate_optimal_delta_range(self, confidence: float, market_regime: Dict, option_type: str = 'call') -> Tuple[float, float]:
        """Calculate optimal delta range based on confidence, market regime, and option type"""
        
        # Different delta ranges for CCs (more conservative) vs CSPs (less conservative)
        if option_type.lower() == 'call':
            # Covered Calls - more conservative delta ranges (but not too restrictive)
            if confidence >= 0.7:  # High confidence
                base_min, base_max = 0.15, 0.40  # More inclusive range
            elif confidence >= 0.5:  # Medium confidence
                base_min, base_max = 0.15, 0.40  # More inclusive range
            else:  # Low confidence
                base_min, base_max = 0.15, 0.40  # More inclusive range
        else:
            # Cash Secured Puts - less conservative delta ranges
            if confidence >= 0.7:  # High confidence
                base_min, base_max = 0.25, 0.45  # Less conservative
            elif confidence >= 0.5:  # Medium confidence
                base_min, base_max = 0.15, 0.35  # Less conservative
            else:  # Low confidence
                base_min, base_max = 0.10, 0.25  # Less conservative
        
        # Market regime adjustments
        if market_regime['primary_regime'] == 'high_volatility':
            # More conservative in high vol
            base_min *= 0.8
            base_max *= 0.8
        elif market_regime['primary_regime'] == 'bull_market':
            # Slightly more aggressive in bull markets
            base_min *= 1.1
            base_max *= 1.1
        elif market_regime['primary_regime'] == 'bear_market':
            # More conservative in bear markets
            base_min *= 0.9
            base_max *= 0.9
        
        return (max(0.05, min(0.5, base_min)), max(0.05, min(0.5, base_max)))
    
    def analyze_portfolio_context(self, portfolio_positions: List[Dict], new_ticker: str) -> Dict:
        """Analyze portfolio context for risk management"""
        
        # Calculate sector concentration (simplified - you'd need sector data)
        total_value = sum(pos['quantity'] * pos['current_price'] 
                         for pos in portfolio_positions 
                         if pos['position_type'] == 'stock')
        
        # Check existing options on same ticker
        existing_options = [pos for pos in portfolio_positions 
                           if pos['ticker'] == new_ticker and pos['position_type'] in ['call', 'put']]
        
        # Calculate portfolio diversification (simplified)
        unique_tickers = len(set(pos['ticker'] for pos in portfolio_positions 
                               if pos['position_type'] == 'stock'))
        diversification_score = min(1.0, unique_tickers / 20)  # Normalize to 20 tickers
        
        return {
            'existing_options_count': len(existing_options),
            'portfolio_diversification_score': diversification_score,
            'ticker_already_has_options': len(existing_options) > 0,
            'total_portfolio_value': total_value
        }
    
    def calculate_expected_value(self, option_data: Dict, confidence: float, market_regime: Dict, option_type: str = 'call') -> float:
        """Calculate true expected value for option selling with different risk tolerances"""
        
        premium = option_data['premium']
        strike = option_data['strike_price']
        current_price = option_data['current_stock_price']
        delta = abs(option_data['delta'])
        days_to_exp = option_data['days_to_expiration']
        theta = option_data.get('theta', 0)
        
        # Different risk tolerances for CCs vs CSPs
        max_assignment_prob = self.max_assignment_probability_cc if option_type.lower() == 'call' else self.max_assignment_probability_csp
        
        # Probability calculations
        prob_profit = 1 - delta  # Delta approximates probability of assignment
        prob_loss = delta
        
        # Apply risk tolerance filter
        if delta > max_assignment_prob:
            # Penalize options with too high assignment probability
            risk_penalty = (delta - max_assignment_prob) * 2.0
            prob_loss += risk_penalty
            prob_profit = max(0.1, prob_profit - risk_penalty)
        
        # Loss amount calculation
        if option_type.lower() == 'call':
            # Covered calls: if assigned, you sell at strike
            # Loss = max(0, current_price - strike) - premium
            max_loss = max(0, current_price - strike) - premium
        else:
            # CSPs: if assigned, you buy at strike
            # Loss = max(0, strike - current_price) - premium
            max_loss = max(0, strike - current_price) - premium
        
        # Base Expected Value
        ev = (prob_profit * premium) - (prob_loss * max_loss)
        
        # Confidence adjustment (more conservative for CCs)
        if option_type.lower() == 'call':
            confidence_multiplier = 0.4 + (confidence * 0.6)  # 0.4 to 1.0 range (more conservative)
        else:
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range (less conservative)
        
        ev *= confidence_multiplier
        
        # Time decay bonus (theta advantage)
        theta_bonus = theta * days_to_exp * 0.1
        ev += theta_bonus
        
        # Market regime adjustment
        if market_regime['primary_regime'] == 'high_volatility':
            ev *= 0.8  # More conservative in high vol
        elif market_regime['primary_regime'] == 'bull_market':
            ev *= 1.1  # Slightly more aggressive in bull markets
        
        return ev
    
    def get_dynamic_weights(self, confidence: float, market_regime: Dict, portfolio_context: Dict) -> Dict:
        """Get dynamic weights based on confidence, market conditions, and portfolio context"""
        
        # Base weights by confidence level
        if confidence >= 0.7:  # High confidence - maximize premium
            base_weights = {
                'expected_value': 0.40,
                'annualized_return': 0.25,
                'delta_optimization': 0.20,
                'technical': 0.10,
                'liquidity': 0.05
            }
        elif confidence >= 0.5:  # Medium confidence - balanced
            base_weights = {
                'expected_value': 0.35,
                'annualized_return': 0.20,
                'delta_optimization': 0.15,
                'technical': 0.15,
                'probability': 0.10,
                'liquidity': 0.05
            }
        else:  # Low confidence - safety first
            base_weights = {
                'expected_value': 0.30,
                'probability': 0.25,
                'annualized_return': 0.15,
                'delta_optimization': 0.10,
                'technical': 0.15,
                'liquidity': 0.05
            }
        
        # Market regime adjustments
        if market_regime['primary_regime'] == 'high_volatility':
            base_weights['probability'] = base_weights.get('probability', 0) + 0.10
            base_weights['expected_value'] -= 0.05
        elif market_regime['primary_regime'] == 'bull_market':
            base_weights['annualized_return'] += 0.05
            base_weights['expected_value'] += 0.05
        
        # Portfolio context adjustments
        if portfolio_context['existing_options_count'] > 2:
            base_weights['expected_value'] -= 0.05
            base_weights['probability'] = base_weights.get('probability', 0) + 0.05
        
        if portfolio_context['portfolio_diversification_score'] > 0.7:
            base_weights['expected_value'] += 0.05
        
        return base_weights
    
    def calculate_optimized_score(self, option_data: Dict, stock_data: Dict, 
                                market_data: Dict, portfolio_positions: List[Dict], option_type: str = 'call') -> Dict:
        """Calculate optimized score using all factors"""
        
        # Detect market regime
        market_regime = self.detect_market_regime(market_data)
        
        # Calculate confidence
        confidence = self.calculate_enhanced_confidence(stock_data, market_regime)
        
        # Get optimal delta range
        delta_min, delta_max = self.calculate_optimal_delta_range(confidence, market_regime, option_type)
        
        # Check if option is in optimal delta range
        option_delta = abs(option_data['delta'])
        delta_optimization_score = 0
        if delta_min <= option_delta <= delta_max:
            delta_optimization_score = 100
        else:
            # Penalize for being outside optimal range
            if option_delta < delta_min:
                delta_optimization_score = 50 * (option_delta / delta_min)
            else:
                delta_optimization_score = 50 * (delta_max / option_delta)
        
        # Calculate expected value
        expected_value = self.calculate_expected_value(option_data, confidence, market_regime, option_type)
        
        # Analyze portfolio context
        portfolio_context = self.analyze_portfolio_context(portfolio_positions, option_data['ticker'])
        
        # Get dynamic weights
        weights = self.get_dynamic_weights(confidence, market_regime, portfolio_context)
        
        # Calculate individual scores
        annualized_return = option_data.get('annualized_return', 0)
        probability_of_profit = option_data.get('probability_of_profit', 0.5)
        
        # Technical score (simplified - you'd use your existing technical scoring)
        technical_score = 50  # Placeholder
        
        # Liquidity score (simplified)
        liquidity_score = min(100, option_data.get('open_interest', 0) / 10)
        
        # Calculate weighted score
        weighted_score = (
            expected_value * weights['expected_value'] +
            annualized_return * weights['annualized_return'] +
            delta_optimization_score * weights['delta_optimization'] +
            technical_score * weights['technical'] +
            probability_of_profit * 100 * weights.get('probability', 0) +
            liquidity_score * weights['liquidity']
        )
        
        # Apply portfolio context adjustment
        portfolio_adjustment = 1.0
        if portfolio_context['existing_options_count'] > 2:
            portfolio_adjustment *= 0.9
        if portfolio_context['portfolio_diversification_score'] > 0.7:
            portfolio_adjustment *= 1.1
        
        final_score = weighted_score * portfolio_adjustment
        
        return {
            'final_score': round(final_score, 2),
            'expected_value': round(expected_value, 4),
            'confidence': round(confidence, 3),
            'market_regime': market_regime['primary_regime'],
            'delta_optimization_score': round(delta_optimization_score, 2),
            'optimal_delta_range': (delta_min, delta_max),
            'portfolio_context': portfolio_context,
            'weights_used': weights,
            'score_breakdown': {
                'expected_value_component': expected_value * weights['expected_value'],
                'annualized_return_component': annualized_return * weights['annualized_return'],
                'delta_optimization_component': delta_optimization_score * weights['delta_optimization'],
                'technical_component': technical_score * weights['technical'],
                'probability_component': probability_of_profit * 100 * weights.get('probability', 0),
                'liquidity_component': liquidity_score * weights['liquidity']
            }
        }
