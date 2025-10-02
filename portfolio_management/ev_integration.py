"""
Integration module for Expected Value-based scoring system
Shows how to integrate the optimized scoring with existing recommenders
"""

import logging
from typing import List, Dict, Any, Optional
from optimized_scoring_system import OptimizedOptionsScorer

logger = logging.getLogger(__name__)

class EVEnhancedRecommender:
    """
    Enhanced recommender that uses Expected Value-based scoring
    Integrates with existing CC and CSP recommenders
    """
    
    def __init__(self, base_recommender, market_data_client=None):
        self.base_recommender = base_recommender
        self.ev_scorer = OptimizedOptionsScorer()
        self.market_data_client = market_data_client
    
    def get_market_data(self) -> Dict:
        """Get current market data for regime detection"""
        # This would integrate with your market data client
        # For now, returning mock data - you'd replace with real data
        return {
            'vix': 20.5,
            'vix_20_ma': 22.1,
            'spy_price': 420.5,
            'spy_20_ma': 415.2,
            'spy_50_ma': 410.8,
            'advancers': 0.55,
            'decliners': 0.45
        }
    
    def enhance_recommendations_with_ev(self, recommendations: List[Dict], 
                                      portfolio_positions: List[Dict]) -> List[Dict]:
        """Enhance existing recommendations with Expected Value scoring"""
        
        # Get market data for regime detection
        market_data = self.get_market_data()
        
        enhanced_recommendations = []
        
        for rec in recommendations:
            try:
                # Extract option data
                option_data = {
                    'ticker': rec['ticker'],
                    'premium': rec['premium'],
                    'strike_price': rec['strike_price'],
                    'current_stock_price': rec['current_stock_price'],
                    'delta': rec['delta'],
                    'days_to_expiration': rec['days_to_expiration'],
                    'annualized_return': rec['annualized_return'],
                    'probability_of_profit': rec['probability_of_profit'],
                    'theta': rec.get('theta', 0),
                    'open_interest': rec.get('open_interest', 0)
                }
                
                # Extract stock data (you'd get this from your technical analysis)
                stock_data = {
                    'rsi': rec.get('rsi', 50),
                    'macd': {
                        'histogram': rec.get('macd_histogram', 0),
                        'signal': rec.get('macd_signal', 0)
                    },
                    'bollinger_bands': {
                        'position': rec.get('bb_position', 0.5)
                    },
                    'support_resistance': {
                        'resistance': rec.get('resistance_level', rec['current_stock_price'] * 1.05),
                        'support': rec.get('support_level', rec['current_stock_price'] * 0.95)
                    },
                    'current_price': rec['current_stock_price'],
                    'volume_ratio': rec.get('volume_ratio', 1.0),
                    'price_change_5d': rec.get('price_change_5d', 0)
                }
                
                # Calculate EV-optimized score
                ev_score_data = self.ev_scorer.calculate_optimized_score(
                    option_data, stock_data, market_data, portfolio_positions
                )
                
                # Add EV data to recommendation
                enhanced_rec = rec.copy()
                enhanced_rec.update({
                    'ev_score': ev_score_data['final_score'],
                    'expected_value': ev_score_data['expected_value'],
                    'confidence': ev_score_data['confidence'],
                    'market_regime': ev_score_data['market_regime'],
                    'delta_optimization_score': ev_score_data['delta_optimization_score'],
                    'optimal_delta_range': ev_score_data['optimal_delta_range'],
                    'ev_score_breakdown': ev_score_data['score_breakdown'],
                    'portfolio_context': ev_score_data['portfolio_context']
                })
                
                # Replace original score with EV score
                enhanced_rec['score'] = ev_score_data['final_score']
                
                enhanced_recommendations.append(enhanced_rec)
                
            except Exception as e:
                logger.error(f"Error enhancing recommendation: {e}")
                # Keep original recommendation if enhancement fails
                enhanced_recommendations.append(rec)
        
        # Sort by EV score
        enhanced_recommendations.sort(key=lambda x: x['ev_score'], reverse=True)
        
        return enhanced_recommendations
    
    def get_ev_enhanced_cc_recommendations(self, stock_positions: List[Dict], 
                                         blocked_tickers: set, 
                                         cc_shares_committed: Dict[str, int] = None) -> Dict:
        """Get EV-enhanced covered call recommendations"""
        
        # Get base recommendations
        base_result = self.base_recommender.get_recommendations(
            stock_positions, blocked_tickers, cc_shares_committed
        )
        
        # Convert to portfolio positions format for context analysis
        portfolio_positions = []
        for pos in stock_positions:
            portfolio_positions.append({
                'ticker': pos['ticker'],
                'position_type': 'stock',
                'quantity': pos['quantity'],
                'current_price': pos['current_price']
            })
        
        # Enhance with EV scoring
        enhanced_recommendations = self.enhance_recommendations_with_ev(
            base_result['recommendations'], portfolio_positions
        )
        
        # Update result
        result = base_result.copy()
        result['recommendations'] = enhanced_recommendations
        result['scoring_method'] = 'Expected Value Optimized'
        result['market_regime'] = self.get_market_data().get('primary_regime', 'unknown')
        
        return result
    
    def get_ev_enhanced_csp_recommendations(self, tickers_input: str = None, 
                                          combine_with_watchlist: bool = False) -> Dict:
        """Get EV-enhanced CSP recommendations"""
        
        # Get base recommendations
        base_result = self.base_recommender.get_recommendations(
            tickers_input, combine_with_watchlist
        )
        
        # For CSP, we need to get portfolio positions differently
        # This would integrate with your portfolio API
        portfolio_positions = []  # You'd get this from your portfolio API
        
        # Enhance with EV scoring
        enhanced_recommendations = self.enhance_recommendations_with_ev(
            base_result['recommendations'], portfolio_positions
        )
        
        # Update result
        result = base_result.copy()
        result['recommendations'] = enhanced_recommendations
        result['scoring_method'] = 'Expected Value Optimized'
        result['market_regime'] = self.get_market_data().get('primary_regime', 'unknown')
        
        return result

# Example usage function
def create_ev_enhanced_recommender():
    """Example of how to create an EV-enhanced recommender"""
    
    # This would be your existing recommenders
    from cc_recommender import CoveredCallRecommender
    from csp_recommender import CashSecuredPutRecommender
    
    # Create base recommenders
    cc_recommender = CoveredCallRecommender()
    csp_recommender = CashSecuredPutRecommender()
    
    # Create EV-enhanced versions
    ev_cc_recommender = EVEnhancedRecommender(cc_recommender)
    ev_csp_recommender = EVEnhancedRecommender(csp_recommender)
    
    return ev_cc_recommender, ev_csp_recommender

# Integration with your existing API
def integrate_with_portfolio_api():
    """
    Example of how to integrate with your existing portfolio API
    This would replace the current scoring in your API endpoints
    """
    
    # In your portfolio_api.py, you would modify the endpoints like this:
    
    # @app.get("/api/options/recommendations/covered-calls")
    # async def get_cc_recommendations_ev_enhanced():
    #     # ... existing code ...
    #     
    #     # Instead of using the base recommender directly:
    #     # result = recommender.get_recommendations(stock_pos_dicts, blocked_tickers, cc_shares_committed)
    #     
    #     # Use the EV-enhanced version:
    #     ev_recommender = EVEnhancedRecommender(recommender)
    #     result = ev_recommender.get_ev_enhanced_cc_recommendations(
    #         stock_pos_dicts, blocked_tickers, cc_shares_committed
    #     )
    #     
    #     return result
    pass
