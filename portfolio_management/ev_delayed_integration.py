"""
Complete integration of EV system with delayed data optimization
Shows how to modify your existing API endpoints to use the enhanced system
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

# Import your existing components
from cc_recommender import CoveredCallRecommender
from csp_recommender import CashSecuredPutRecommender
from optimized_scoring_system import OptimizedOptionsScorer
from delayed_data_optimizer import DelayedDataOptimizer, enhance_ev_system_for_delayed_data

logger = logging.getLogger(__name__)

class EVDelayedDataIntegration:
    """
    Complete integration of EV system with delayed data optimization
    Ready to replace your existing API endpoints
    """
    
    def __init__(self, polygon_api_key: str = None, supabase_client = None):
        self.polygon_api_key = polygon_api_key or os.getenv('POLYGON_API_KEY')
        self.supabase = supabase_client
        
        # Initialize base recommenders
        self.cc_recommender = CoveredCallRecommender(
            polygon_api_key=self.polygon_api_key, 
            supabase_client=self.supabase
        )
        self.csp_recommender = CashSecuredPutRecommender(
            polygon_api_key=self.polygon_api_key, 
            supabase_client=self.supabase
        )
        
        # Initialize EV system with delayed data optimization
        self.ev_scorer = OptimizedOptionsScorer()
        self.delayed_data_optimizer = DelayedDataOptimizer()
        
        # Enhance EV system for delayed data
        self.enhanced_ev_scorer, _ = enhance_ev_system_for_delayed_data(
            self.ev_scorer, None
        )
        
        logger.info("✅ EV Delayed Data Integration initialized")
    
    def get_market_data_for_regime_detection(self) -> Dict:
        """Get market data for regime detection (you'd integrate with your market data client)"""
        
        # This would integrate with your existing market_data_client
        # For now, returning mock data - replace with real data from your system
        return {
            'vix': 20.5,
            'vix_20_ma': 22.1,
            'spy_price': 420.5,
            'spy_20_ma': 415.2,
            'spy_50_ma': 410.8,
            'advancers': 0.55,
            'decliners': 0.45
        }
    
    def enhance_recommendation_with_ev_delayed_data(self, recommendation: Dict, 
                                                  stock_data: Dict, 
                                                  portfolio_positions: List[Dict], 
                                                  option_type: str = 'call') -> Dict:
        """Enhance a single recommendation with EV scoring and delayed data optimization"""
        
        try:
            # Get market data for regime detection
            market_data = self.get_market_data_for_regime_detection()
            
            # Extract option data
            option_data = {
                'ticker': recommendation['ticker'],
                'premium': recommendation['premium'],
                'strike_price': recommendation['strike_price'],
                'current_stock_price': recommendation['current_stock_price'],
                'delta': recommendation['delta'],
                'days_to_expiration': recommendation['days_to_expiration'],
                'annualized_return': recommendation['annualized_return'],
                'probability_of_profit': recommendation['probability_of_profit'],
                'theta': recommendation.get('theta', 0),
                'open_interest': recommendation.get('open_interest', 0),
                'contract_type': 'call'  # Assuming calls for covered calls
            }
            
            # Calculate EV-optimized score with delayed data adjustments
            try:
                ev_result = self.enhanced_ev_scorer.calculate_optimized_score(
                    option_data, stock_data, market_data, portfolio_positions, option_type
                )
                
                if ev_result is None:
                    # Filtered out due to high volatility or other safety checks
                    return None
                    
                # Check if we have the required keys
                if 'final_score' not in ev_result:
                    logger.error(f"Missing 'final_score' in EV result: {ev_result}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error calculating EV score for {recommendation.get('ticker', 'unknown')}: {e}")
                return None
            
            # Enhance recommendation with EV data
            enhanced_rec = recommendation.copy()
            enhanced_rec.update({
                'ev_score': ev_result['final_score'],
                'expected_value': ev_result['expected_value'],
                'confidence': ev_result['confidence'],
                'market_regime': ev_result['market_regime'],
                'delta_optimization_score': ev_result['delta_optimization_score'],
                'optimal_delta_range': ev_result['optimal_delta_range'],
                'delayed_data_adjusted': ev_result.get('delayed_data_adjusted', False),
                'original_confidence': ev_result.get('original_confidence', ev_result['confidence']),
                'adjusted_confidence': ev_result.get('adjusted_confidence', ev_result['confidence']),
                'delayed_data_ev': ev_result.get('delayed_data_ev', ev_result['expected_value']),
                'safety_margin_applied': 0.02,
                'data_delay_minutes': 15,
                'ev_score_breakdown': ev_result['score_breakdown'],
                'portfolio_context': ev_result['portfolio_context']
            })
            
            # Replace original score with EV score
            enhanced_rec['score'] = ev_result['final_score']
            
            return enhanced_rec
            
        except Exception as e:
            logger.error(f"Error enhancing recommendation with EV delayed data: {e}")
            return recommendation  # Return original if enhancement fails
    
    def get_enhanced_cc_recommendations(self, stock_positions: List[Dict], 
                                      blocked_tickers: set, 
                                      cc_shares_committed: Dict[str, int] = None) -> Dict:
        """Get EV-enhanced covered call recommendations with delayed data optimization"""
        
        try:
            logger.info("🔍 Starting EV-enhanced CC recommendations with delayed data optimization...")
            
            # Get base recommendations from existing system
            base_result = self.cc_recommender.get_recommendations(
                stock_positions, blocked_tickers, cc_shares_committed
            )
            
            logger.info(f"📊 Base CC recommendations: {len(base_result.get('recommendations', []))}")
            
            # Convert to portfolio positions format for context analysis
            portfolio_positions = []
            for pos in stock_positions:
                portfolio_positions.append({
                    'ticker': pos['ticker'],
                    'position_type': 'stock',
                    'quantity': pos['quantity'],
                    'current_price': pos['current_price']
                })
            
            # Enhance each recommendation with EV scoring and delayed data optimization
            enhanced_recommendations = []
            filtered_count = 0
            
            for rec in base_result['recommendations']:
                try:
                    # Extract stock data for technical analysis
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
                        'price_change_5d': rec.get('price_change_5d', 0),
                        'price_change_1h': rec.get('price_change_1h', 0),
                        'price_change_4h': rec.get('price_change_4h', 0)
                    }
                    
                    # Enhance with EV scoring and delayed data optimization
                    enhanced_rec = self.enhance_recommendation_with_ev_delayed_data(
                        rec, stock_data, portfolio_positions
                    )
                    
                    if enhanced_rec is None:
                        filtered_count += 1
                        logger.info(f"Filtered out recommendation for {rec.get('ticker', 'unknown')} - EV scoring returned None")
                        continue
                    
                    enhanced_recommendations.append(enhanced_rec)
                    
                except Exception as e:
                    logger.error(f"Error processing recommendation: {e}")
                    continue
            
            # Sort by EV score (with fallback to regular score)
            enhanced_recommendations.sort(key=lambda x: x.get('ev_score', x.get('score', 0)), reverse=True)
            
            # Update result
            result = base_result.copy()
            result.update({
                'recommendations': enhanced_recommendations,
                'scoring_method': 'Expected Value Optimized with Delayed Data',
                'market_regime': self.get_market_data_for_regime_detection().get('primary_regime', 'unknown'),
                'delayed_data_optimization': True,
                'safety_margins_applied': True,
                'filtered_recommendations': filtered_count,
                'total_enhanced': len(enhanced_recommendations)
            })
            
            logger.info(f"✅ Generated {len(enhanced_recommendations)} EV-enhanced recommendations")
            logger.info(f"📊 Filtered out {filtered_count} recommendations due to safety checks")
            logger.info(f"📊 Market regime: {result['market_regime']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in EV-enhanced CC recommendations: {e}", exc_info=True)
            raise Exception(f"Error getting EV-enhanced recommendations: {str(e)}")
    
    def get_enhanced_csp_recommendations(self, tickers_input: str = None, 
                                       combine_with_watchlist: bool = False) -> Dict:
        """Get EV-enhanced CSP recommendations with delayed data optimization"""
        
        try:
            logger.info("🔍 Starting EV-enhanced CSP recommendations with delayed data optimization...")
            
            # Get base recommendations from existing system
            base_result = self.csp_recommender.get_recommendations(
                tickers_input, combine_with_watchlist
            )
            
            logger.info(f"📊 Base CSP recommendations: {len(base_result.get('recommendations', []))}")
            
            # For CSP, we need to get portfolio positions differently
            # This would integrate with your portfolio API
            portfolio_positions = []  # You'd get this from your portfolio API
            
            # Enhance each recommendation with EV scoring and delayed data optimization
            enhanced_recommendations = []
            filtered_count = 0
            
            for rec in base_result['recommendations']:
                try:
                    # Extract stock data for technical analysis
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
                        'price_change_5d': rec.get('price_change_5d', 0),
                        'price_change_1h': rec.get('price_change_1h', 0),
                        'price_change_4h': rec.get('price_change_4h', 0)
                    }
                    
                    # Enhance with EV scoring and delayed data optimization
                    enhanced_rec = self.enhance_recommendation_with_ev_delayed_data(
                        rec, stock_data, portfolio_positions
                    )
                    
                    if enhanced_rec is None:
                        filtered_count += 1
                        logger.info(f"Filtered out recommendation for {rec.get('ticker', 'unknown')} - EV scoring returned None")
                        continue
                    
                    enhanced_recommendations.append(enhanced_rec)
                    
                except Exception as e:
                    logger.error(f"Error processing recommendation: {e}")
                    continue
            
            # Sort by EV score (with fallback to regular score)
            enhanced_recommendations.sort(key=lambda x: x.get('ev_score', x.get('score', 0)), reverse=True)
            
            # Update result
            result = base_result.copy()
            result.update({
                'recommendations': enhanced_recommendations,
                'scoring_method': 'Expected Value Optimized with Delayed Data',
                'market_regime': self.get_market_data_for_regime_detection().get('primary_regime', 'unknown'),
                'delayed_data_optimization': True,
                'safety_margins_applied': True,
                'filtered_recommendations': filtered_count,
                'total_enhanced': len(enhanced_recommendations)
            })
            
            logger.info(f"✅ Generated {len(enhanced_recommendations)} EV-enhanced CSP recommendations")
            logger.info(f"📊 Filtered out {filtered_count} recommendations due to safety checks")
            logger.info(f"📊 Market regime: {result['market_regime']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in EV-enhanced CSP recommendations: {e}", exc_info=True)
            raise Exception(f"Error getting EV-enhanced CSP recommendations: {str(e)}")

# Example of how to modify your existing API endpoints
def create_enhanced_api_endpoints():
    """
    Example of how to modify your existing portfolio_api.py endpoints
    This shows the exact changes needed
    """
    
    # In your portfolio_api.py, you would replace these endpoints:
    
    # OLD ENDPOINT:
    # @app.get("/api/options/recommendations/covered-calls")
    # async def get_cc_recommendations():
    #     # ... existing code ...
    #     result = recommender.get_recommendations(stock_pos_dicts, blocked_tickers, cc_shares_committed)
    #     return result
    
    # NEW ENDPOINT:
    # @app.get("/api/options/recommendations/covered-calls")
    # async def get_cc_recommendations():
    #     try:
    #         logger.info("🔍 Starting EV-enhanced covered call recommendations...")
    #         
    #         # Get current portfolio positions
    #         positions = await get_portfolio_positions()
    #         # ... existing code ...
    #         
    #         # Create enhanced recommender
    #         enhanced_recommender = EVDelayedDataIntegration(
    #             polygon_api_key=polygon_api_key, 
    #             supabase_client=supabase
    #         )
    #         
    #         # Get enhanced recommendations
    #         result = enhanced_recommender.get_enhanced_cc_recommendations(
    #             stock_pos_dicts, blocked_tickers, cc_shares_committed
    #         )
    #         
    #         return result
    #     except Exception as e:
    #         logger.error(f"❌ Error in EV-enhanced CC recommendations: {e}", exc_info=True)
    #         raise HTTPException(status_code=500, detail=f"Error getting EV-enhanced recommendations: {str(e)}")
    
    pass

# Example usage
def example_usage():
    """Example of how to use the enhanced system"""
    
    # Initialize the enhanced system
    enhanced_recommender = EVDelayedDataIntegration(
        polygon_api_key=os.getenv('POLYGON_API_KEY'),
        supabase_client=None  # Your supabase client
    )
    
    # Sample stock positions
    sample_positions = [
        {
            'ticker': 'AAPL',
            'quantity': 100,
            'current_price': 150.0
        },
        {
            'ticker': 'MSFT',
            'quantity': 50,
            'current_price': 300.0
        }
    ]
    
    # Get enhanced recommendations
    result = enhanced_recommender.get_enhanced_cc_recommendations(
        stock_positions=sample_positions,
        blocked_tickers=set(),
        cc_shares_committed={}
    )
    
    print(f"Generated {len(result['recommendations'])} enhanced recommendations")
    print(f"Market regime: {result['market_regime']}")
    print(f"Delayed data optimization: {result['delayed_data_optimization']}")
    
    return result

if __name__ == "__main__":
    example_usage()
