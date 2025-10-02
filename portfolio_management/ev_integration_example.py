"""
Complete example of integrating Expected Value scoring with existing system
Shows how to modify your current API endpoints
"""

import logging
from typing import List, Dict, Any, Optional
from ev_integration import EVEnhancedRecommender
from cc_recommender import CoveredCallRecommender
from csp_recommender import CashSecuredPutRecommender

logger = logging.getLogger(__name__)

# Example of how to modify your existing portfolio_api.py endpoints

async def get_cc_recommendations_ev_enhanced(stock_positions: List[Dict], 
                                           blocked_tickers: set, 
                                           cc_shares_committed: Dict[str, int] = None):
    """
    EV-Enhanced Covered Call Recommendations
    This would replace your existing get_cc_recommendations function
    """
    try:
        logger.info("🔍 Starting EV-enhanced covered call recommendations...")
        
        # Get current portfolio positions for context
        positions = await get_portfolio_positions()  # Your existing function
        
        # Create base recommender
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        base_recommender = CoveredCallRecommender(polygon_api_key=polygon_api_key, supabase_client=supabase)
        
        # Create EV-enhanced recommender
        ev_recommender = EVEnhancedRecommender(base_recommender)
        
        # Get EV-enhanced recommendations
        result = ev_recommender.get_ev_enhanced_cc_recommendations(
            stock_positions, blocked_tickers, cc_shares_committed
        )
        
        # Add additional EV-specific metadata
        result.update({
            'scoring_method': 'Expected Value Optimized',
            'max_assignment_probability': 0.40,
            'confidence_thresholds': {
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            },
            'delta_optimization': True,
            'market_regime_aware': True,
            'portfolio_context_aware': True
        })
        
        logger.info(f"✅ Generated {len(result['recommendations'])} EV-enhanced recommendations")
        logger.info(f"📊 Market regime: {result.get('market_regime', 'unknown')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in EV-enhanced CC recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting EV-enhanced recommendations: {str(e)}")

async def get_csp_recommendations_ev_enhanced(tickers_input: str = None, 
                                            combine_with_watchlist: bool = False):
    """
    EV-Enhanced CSP Recommendations
    This would replace your existing get_csp_recommendations function
    """
    try:
        logger.info("🔍 Starting EV-enhanced CSP recommendations...")
        
        # Get current portfolio positions for context
        positions = await get_portfolio_positions()  # Your existing function
        
        # Create base recommender
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        base_recommender = CashSecuredPutRecommender(polygon_api_key=polygon_api_key, supabase_client=supabase)
        
        # Create EV-enhanced recommender
        ev_recommender = EVEnhancedRecommender(base_recommender)
        
        # Get EV-enhanced recommendations
        result = ev_recommender.get_ev_enhanced_csp_recommendations(
            tickers_input, combine_with_watchlist
        )
        
        # Add additional EV-specific metadata
        result.update({
            'scoring_method': 'Expected Value Optimized',
            'max_assignment_probability': 0.40,
            'confidence_thresholds': {
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            },
            'delta_optimization': True,
            'market_regime_aware': True,
            'portfolio_context_aware': True
        })
        
        logger.info(f"✅ Generated {len(result['recommendations'])} EV-enhanced CSP recommendations")
        logger.info(f"📊 Market regime: {result.get('market_regime', 'unknown')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in EV-enhanced CSP recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting EV-enhanced CSP recommendations: {str(e)}")

# Example of how to add new EV-specific endpoints

async def get_ev_analysis(ticker: str, option_type: str = 'call'):
    """
    Get detailed Expected Value analysis for a specific ticker
    New endpoint for deep-dive analysis
    """
    try:
        logger.info(f"🔍 Starting EV analysis for {ticker} {option_type}s...")
        
        # Get market data
        market_data = {
            'vix': 20.5,  # You'd get this from your market data client
            'vix_20_ma': 22.1,
            'spy_price': 420.5,
            'spy_20_ma': 415.2,
            'spy_50_ma': 410.8,
            'advancers': 0.55,
            'decliners': 0.45
        }
        
        # Get technical data
        stock_data = get_technical_indicators(ticker)  # Your existing function
        
        # Get options data
        options_data = await market_analyzer.fetch_options_data(ticker)  # Your existing function
        
        # Filter by option type
        if option_type.lower() == 'call':
            options_data = [opt for opt in options_data if opt['contract_type'] == 'call']
        else:
            options_data = [opt for opt in options_data if opt['contract_type'] == 'put']
        
        # Get portfolio context
        positions = await get_portfolio_positions()
        
        # Create EV scorer
        from optimized_scoring_system import OptimizedOptionsScorer
        ev_scorer = OptimizedOptionsScorer()
        
        # Analyze each option
        ev_analysis = []
        for option in options_data:
            try:
                # Calculate EV score
                ev_data = ev_scorer.calculate_optimized_score(
                    option, stock_data, market_data, positions
                )
                
                ev_analysis.append({
                    'strike': option['strike_price'],
                    'premium': option['current_price'],
                    'delta': option['delta'],
                    'dte': option['days_to_expiration'],
                    'ev_score': ev_data['final_score'],
                    'expected_value': ev_data['expected_value'],
                    'confidence': ev_data['confidence'],
                    'delta_optimization_score': ev_data['delta_optimization_score'],
                    'optimal_delta_range': ev_data['optimal_delta_range'],
                    'market_regime': ev_data['market_regime']
                })
                
            except Exception as e:
                logger.error(f"Error analyzing option {option['strike_price']}: {e}")
                continue
        
        # Sort by EV score
        ev_analysis.sort(key=lambda x: x['ev_score'], reverse=True)
        
        return {
            'ticker': ticker,
            'option_type': option_type,
            'market_regime': ev_scorer.detect_market_regime(market_data),
            'analysis': ev_analysis,
            'recommendations': ev_analysis[:10],  # Top 10
            'analysis_date': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in EV analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting EV analysis: {str(e)}")

# Example of how to modify your existing API endpoints
def modify_existing_endpoints():
    """
    Example of how to modify your existing API endpoints in portfolio_api.py
    """
    
    # In your portfolio_api.py, you would replace these lines:
    
    # OLD CODE:
    # @app.get("/api/options/recommendations/covered-calls")
    # async def get_cc_recommendations():
    #     # ... existing code ...
    #     result = recommender.get_recommendations(stock_pos_dicts, blocked_tickers, cc_shares_committed)
    #     return result
    
    # NEW CODE:
    # @app.get("/api/options/recommendations/covered-calls")
    # async def get_cc_recommendations():
    #     # ... existing code ...
    #     result = await get_cc_recommendations_ev_enhanced(stock_pos_dicts, blocked_tickers, cc_shares_committed)
    #     return result
    
    # Similarly for CSP recommendations:
    # @app.post("/api/options/recommendations/cash-secured-puts")
    # async def get_csp_recommendations(request_data: dict = None):
    #     # ... existing code ...
    #     result = await get_csp_recommendations_ev_enhanced(tickers_input, combine_with_watchlist)
    #     return result
    
    # And add new EV-specific endpoints:
    # @app.get("/api/options/ev-analysis/{ticker}")
    # async def get_ev_analysis_endpoint(ticker: str, option_type: str = 'call'):
    #     return await get_ev_analysis(ticker, option_type)
    
    pass

# Example of how to test the new system
async def test_ev_system():
    """Test the EV system with sample data"""
    
    # Sample portfolio positions
    sample_positions = [
        {
            'ticker': 'AAPL',
            'position_type': 'stock',
            'quantity': 100,
            'current_price': 150.0
        },
        {
            'ticker': 'MSFT',
            'position_type': 'stock',
            'quantity': 50,
            'current_price': 300.0
        }
    ]
    
    # Sample market data
    sample_market_data = {
        'vix': 20.5,
        'vix_20_ma': 22.1,
        'spy_price': 420.5,
        'spy_20_ma': 415.2,
        'spy_50_ma': 410.8,
        'advancers': 0.55,
        'decliners': 0.45
    }
    
    # Test EV scorer
    from optimized_scoring_system import OptimizedOptionsScorer
    ev_scorer = OptimizedOptionsScorer()
    
    # Test market regime detection
    regime = ev_scorer.detect_market_regime(sample_market_data)
    print(f"Market regime: {regime}")
    
    # Test confidence calculation
    sample_stock_data = {
        'rsi': 75,
        'macd': {'histogram': -0.5},
        'bollinger_bands': {'position': 0.85},
        'support_resistance': {'resistance': 155.0},
        'current_price': 150.0,
        'volume_ratio': 1.5,
        'price_change_5d': -0.02
    }
    
    confidence = ev_scorer.calculate_enhanced_confidence(sample_stock_data, regime)
    print(f"Confidence: {confidence}")
    
    # Test delta range calculation
    delta_range = ev_scorer.calculate_optimal_delta_range(confidence, regime)
    print(f"Optimal delta range: {delta_range}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ev_system())
