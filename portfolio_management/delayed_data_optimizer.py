"""
Optimizations for 15-minute delayed data from Polygon.io
Enhances the EV system to work optimally with delayed market data
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class DelayedDataOptimizer:
    """
    Optimizes the EV scoring system for 15-minute delayed data
    Adds safety margins and volatility filters
    """
    
    def __init__(self):
        self.delay_minutes = 15
        self.volatility_threshold = 0.02  # 2% price change threshold
        self.earnings_buffer_days = 3  # Avoid trading 3 days before earnings
        
    def add_safety_margins(self, option_data: Dict, stock_data: Dict) -> Dict:
        """Add safety margins to account for 15-minute delay"""
        
        # Calculate recent volatility
        price_change_15min = self._estimate_15min_price_change(stock_data)
        
        # Add safety margins to premium calculations
        safety_margin = 0.02  # 2% safety margin
        
        # Adjust premium for delayed data
        adjusted_premium = option_data['premium'] * (1 - safety_margin)
        
        # Adjust strike prices for calls (more conservative)
        if option_data['contract_type'] == 'call':
            # Use slightly higher strike for calls to account for upward price movement
            adjusted_strike = option_data['strike_price'] * (1 + safety_margin * 0.5)
        else:
            # Use slightly lower strike for puts to account for downward price movement
            adjusted_strike = option_data['strike_price'] * (1 - safety_margin * 0.5)
        
        # Adjust current price estimate
        current_price = stock_data['current_price']
        if price_change_15min > self.volatility_threshold:
            # High volatility - be more conservative
            price_adjustment = price_change_15min * 1.5
        else:
            # Normal volatility - standard adjustment
            price_adjustment = price_change_15min * 0.5
        
        adjusted_price = current_price + price_adjustment
        
        return {
            'adjusted_premium': adjusted_premium,
            'adjusted_strike': adjusted_strike,
            'adjusted_current_price': adjusted_price,
            'safety_margin_applied': safety_margin,
            'price_adjustment': price_adjustment
        }
    
    def _estimate_15min_price_change(self, stock_data: Dict) -> float:
        """Estimate potential price change in 15 minutes based on recent volatility"""
        
        # Use recent price changes to estimate volatility
        price_change_1h = stock_data.get('price_change_1h', 0)
        price_change_4h = stock_data.get('price_change_4h', 0)
        
        # Estimate 15-minute change based on recent volatility
        if abs(price_change_1h) > 0.01:  # High volatility
            estimated_change = price_change_1h * 0.25  # 15min = 1/4 of 1h
        elif abs(price_change_4h) > 0.02:  # Medium volatility
            estimated_change = price_change_4h * 0.0625  # 15min = 1/16 of 4h
        else:
            estimated_change = 0.001  # Low volatility - minimal change
        
        return estimated_change
    
    def filter_high_volatility_periods(self, stock_data: Dict, market_data: Dict) -> bool:
        """Filter out high volatility periods where 15-min delay is problematic"""
        
        # Check for recent high volatility
        price_change_1h = abs(stock_data.get('price_change_1h', 0))
        price_change_4h = abs(stock_data.get('price_change_4h', 0))
        
        # Check VIX level
        vix = market_data.get('vix', 20)
        
        # Filter conditions
        if price_change_1h > 0.05:  # 5% change in 1 hour
            logger.warning(f"High volatility detected: {price_change_1h:.1%} in 1h - skipping")
            return False
        
        if price_change_4h > 0.10:  # 10% change in 4 hours
            logger.warning(f"High volatility detected: {price_change_4h:.1%} in 4h - skipping")
            return False
        
        if vix > 35:  # Very high VIX
            logger.warning(f"High VIX detected: {vix} - skipping")
            return False
        
        return True
    
    def check_earnings_calendar(self, ticker: str, expiration_date: str) -> bool:
        """Check if option expires near earnings (avoid assignment risk)"""
        
        # This would integrate with an earnings calendar API
        # For now, return True (no earnings conflict)
        # In production, you'd check:
        # - Earnings date vs expiration date
        # - Avoid options expiring within 3 days of earnings
        
        return True
    
    def enhance_confidence_for_delayed_data(self, confidence: float, stock_data: Dict) -> float:
        """Adjust confidence calculation for delayed data"""
        
        # Reduce confidence if recent volatility is high
        price_change_1h = abs(stock_data.get('price_change_1h', 0))
        price_change_4h = abs(stock_data.get('price_change_4h', 0))
        
        volatility_penalty = 0
        if price_change_1h > 0.03:  # 3% in 1 hour
            volatility_penalty += 0.1
        if price_change_4h > 0.06:  # 6% in 4 hours
            volatility_penalty += 0.1
        
        # Reduce confidence for delayed data
        delayed_data_penalty = 0.05  # 5% penalty for 15-min delay
        
        adjusted_confidence = confidence - volatility_penalty - delayed_data_penalty
        
        return max(0.1, min(0.95, adjusted_confidence))
    
    def get_optimal_delta_range_for_delayed_data(self, base_delta_range: tuple, 
                                               stock_data: Dict) -> tuple:
        """Adjust delta range for delayed data safety"""
        
        min_delta, max_delta = base_delta_range
        
        # Check recent volatility
        price_change_1h = abs(stock_data.get('price_change_1h', 0))
        
        if price_change_1h > 0.02:  # High volatility
            # More conservative delta range
            min_delta *= 0.8
            max_delta *= 0.8
        elif price_change_1h > 0.01:  # Medium volatility
            # Slightly more conservative
            min_delta *= 0.9
            max_delta *= 0.9
        
        return (max(0.05, min_delta), max(0.5, max_delta))
    
    def calculate_delayed_data_expected_value(self, option_data: Dict, 
                                            stock_data: Dict, 
                                            confidence: float) -> float:
        """Calculate EV with delayed data adjustments"""
        
        # Get safety-adjusted data
        safety_data = self.add_safety_margins(option_data, stock_data)
        
        # Use adjusted values for EV calculation
        adjusted_premium = safety_data['adjusted_premium']
        adjusted_strike = safety_data['adjusted_strike']
        adjusted_price = safety_data['adjusted_current_price']
        
        # Calculate EV with adjusted values
        delta = abs(option_data['delta'])
        prob_profit = 1 - delta
        prob_loss = delta
        
        # More conservative loss calculation
        max_loss = max(0, adjusted_price - adjusted_strike) - adjusted_premium
        
        # Base EV
        ev = (prob_profit * adjusted_premium) - (prob_loss * max_loss)
        
        # Apply confidence adjustment
        confidence_multiplier = 0.4 + (confidence * 0.6)  # 0.4 to 1.0 range
        ev *= confidence_multiplier
        
        # Apply delayed data penalty
        ev *= 0.95  # 5% penalty for delayed data
        
        return ev
    
    def get_delayed_data_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Filter and adjust recommendations for delayed data"""
        
        filtered_recommendations = []
        
        for rec in recommendations:
            try:
                # Extract data for analysis
                stock_data = {
                    'current_price': rec['current_stock_price'],
                    'price_change_1h': rec.get('price_change_1h', 0),
                    'price_change_4h': rec.get('price_change_4h', 0)
                }
                
                market_data = {
                    'vix': rec.get('vix', 20)
                }
                
                # Check if we should filter this recommendation
                if not self.filter_high_volatility_periods(stock_data, market_data):
                    continue
                
                # Check earnings calendar
                if not self.check_earnings_calendar(rec['ticker'], rec['expiration_date']):
                    continue
                
                # Adjust confidence for delayed data
                original_confidence = rec.get('confidence', 0.5)
                adjusted_confidence = self.enhance_confidence_for_delayed_data(
                    original_confidence, stock_data
                )
                
                # Recalculate EV with delayed data adjustments
                option_data = {
                    'premium': rec['premium'],
                    'strike_price': rec['strike_price'],
                    'delta': rec['delta'],
                    'contract_type': 'call'  # Assuming calls for now
                }
                
                adjusted_ev = self.calculate_delayed_data_expected_value(
                    option_data, stock_data, adjusted_confidence
                )
                
                # Update recommendation with delayed data adjustments
                enhanced_rec = rec.copy()
                enhanced_rec.update({
                    'delayed_data_adjusted': True,
                    'original_confidence': original_confidence,
                    'adjusted_confidence': adjusted_confidence,
                    'delayed_data_ev': adjusted_ev,
                    'safety_margin_applied': 0.02,
                    'data_delay_minutes': 15
                })
                
                # Use adjusted EV as the primary score
                enhanced_rec['score'] = adjusted_ev
                enhanced_rec['expected_value'] = adjusted_ev
                
                filtered_recommendations.append(enhanced_rec)
                
            except Exception as e:
                logger.error(f"Error processing recommendation for delayed data: {e}")
                continue
        
        # Sort by adjusted EV score
        filtered_recommendations.sort(key=lambda x: x['delayed_data_ev'], reverse=True)
        
        return filtered_recommendations

# Integration with existing EV system
def enhance_ev_system_for_delayed_data(ev_scorer, market_data_client):
    """Enhance the EV system to work optimally with 15-minute delayed data"""
    
    # Add delayed data optimizer
    delayed_data_optimizer = DelayedDataOptimizer()
    
    # Enhance the EV scorer's calculate_optimized_score method
    original_calculate_score = ev_scorer.calculate_optimized_score
    
    def enhanced_calculate_score(option_data, stock_data, market_data, portfolio_positions, option_type='call'):
        # Get original score
        original_result = original_calculate_score(option_data, stock_data, market_data, portfolio_positions, option_type)
        
        # Add delayed data enhancements
        if not delayed_data_optimizer.filter_high_volatility_periods(stock_data, market_data):
            return None  # Filter out high volatility periods
        
        # Adjust confidence for delayed data
        original_confidence = original_result['confidence']
        adjusted_confidence = delayed_data_optimizer.enhance_confidence_for_delayed_data(
            original_confidence, stock_data
        )
        
        # Recalculate EV with delayed data adjustments
        adjusted_ev = delayed_data_optimizer.calculate_delayed_data_expected_value(
            option_data, stock_data, adjusted_confidence
        )
        
        # Update result
        enhanced_result = original_result.copy()
        enhanced_result.update({
            'delayed_data_adjusted': True,
            'original_confidence': original_confidence,
            'adjusted_confidence': adjusted_confidence,
            'delayed_data_ev': adjusted_ev,
            'final_score': adjusted_ev,  # Use adjusted EV as final score
            'expected_value': adjusted_ev
        })
        
        return enhanced_result
    
    # Replace the method
    ev_scorer.calculate_optimized_score = enhanced_calculate_score
    
    return ev_scorer, delayed_data_optimizer
