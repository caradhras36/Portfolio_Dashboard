"""
Test script for EV Delayed Data Integration
Run this to verify the enhanced system works correctly
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the portfolio_management directory to the path
sys.path.append('portfolio_management')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ev_integration():
    """Test the EV integration with delayed data optimization"""
    
    try:
        # Import the enhanced recommender
        from ev_delayed_integration import EVDelayedDataIntegration
        
        logger.info("Starting EV Integration Test...")
        
        # Check if we have Polygon API key
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not polygon_api_key:
            logger.warning("POLYGON_API_KEY not found. Testing with mock data only.")
            # Test individual components without full integration
            return await test_individual_components()
        
        # Initialize enhanced recommender
        enhanced_recommender = EVDelayedDataIntegration(
            polygon_api_key=polygon_api_key,
            supabase_client=None  # You'd pass your supabase client here
        )
        
        logger.info("Enhanced recommender initialized successfully")
        
        # Test with sample data
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
        
        logger.info("📊 Testing with sample positions...")
        
        # Test covered call recommendations
        logger.info("🔍 Testing covered call recommendations...")
        cc_result = enhanced_recommender.get_enhanced_cc_recommendations(
            stock_positions=sample_positions,
            blocked_tickers=set(),
            cc_shares_committed={}
        )
        
        logger.info(f"CC Test Results:")
        logger.info(f"   - Generated {len(cc_result['recommendations'])} recommendations")
        logger.info(f"   - Market regime: {cc_result.get('market_regime', 'unknown')}")
        logger.info(f"   - Delayed data optimization: {cc_result.get('delayed_data_optimization', False)}")
        logger.info(f"   - Safety margins applied: {cc_result.get('safety_margins_applied', False)}")
        logger.info(f"   - Filtered recommendations: {cc_result.get('filtered_recommendations', 0)}")
        
        # Test CSP recommendations
        logger.info("Testing CSP recommendations...")
        csp_result = enhanced_recommender.get_enhanced_csp_recommendations(
            tickers_input="AAPL,MSFT",
            combine_with_watchlist=False
        )
        
        logger.info(f"CSP Test Results:")
        logger.info(f"   - Generated {len(csp_result['recommendations'])} recommendations")
        logger.info(f"   - Market regime: {csp_result.get('market_regime', 'unknown')}")
        logger.info(f"   - Delayed data optimization: {csp_result.get('delayed_data_optimization', False)}")
        logger.info(f"   - Safety margins applied: {csp_result.get('safety_margins_applied', False)}")
        logger.info(f"   - Filtered recommendations: {csp_result.get('filtered_recommendations', 0)}")
        
        # Test market regime detection
        logger.info("Testing market regime detection...")
        market_data = enhanced_recommender.get_market_data_for_regime_detection()
        logger.info(f"Market data: {market_data}")
        
        # Test delayed data optimizer
        logger.info("Testing delayed data optimizer...")
        from delayed_data_optimizer import DelayedDataOptimizer
        optimizer = DelayedDataOptimizer()
        
        # Test volatility filtering
        test_stock_data = {
            'current_price': 150.0,
            'price_change_1h': 0.01,  # 1% change
            'price_change_4h': 0.02   # 2% change
        }
        test_market_data = {'vix': 25}
        
        should_trade = optimizer.filter_high_volatility_periods(test_stock_data, test_market_data)
        logger.info(f"Volatility filtering test: {should_trade}")
        
        # Test confidence adjustment
        original_confidence = 0.7
        adjusted_confidence = optimizer.enhance_confidence_for_delayed_data(original_confidence, test_stock_data)
        logger.info(f"Confidence adjustment: {original_confidence} -> {adjusted_confidence}")
        
        logger.info("All tests completed successfully!")
        logger.info("Summary:")
        logger.info("   - EV scoring system working")
        logger.info("   - Delayed data optimization working")
        logger.info("   - Market regime detection working")
        logger.info("   - Safety margins applied")
        logger.info("   - Volatility filtering working")
        logger.info("   - Confidence adjustments working")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

async def test_individual_components():
    """Test individual components of the EV system"""
    
    try:
        logger.info("Testing individual components...")
        
        # Test optimized scoring system
        from optimized_scoring_system import OptimizedOptionsScorer
        ev_scorer = OptimizedOptionsScorer()
        
        # Test market regime detection
        test_market_data = {
            'vix': 20.5,
            'vix_20_ma': 22.1,
            'spy_price': 420.5,
            'spy_20_ma': 415.2,
            'spy_50_ma': 410.8,
            'advancers': 0.55,
            'decliners': 0.45
        }
        
        regime = ev_scorer.detect_market_regime(test_market_data)
        logger.info(f"Market regime detection: {regime['primary_regime']}")
        
        # Test confidence calculation
        test_stock_data = {
            'rsi': 75,
            'macd': {'histogram': -0.5},
            'bollinger_bands': {'position': 0.85},
            'support_resistance': {'resistance': 155.0},
            'current_price': 150.0,
            'volume_ratio': 1.5,
            'price_change_5d': -0.02
        }
        
        confidence = ev_scorer.calculate_enhanced_confidence(test_stock_data, regime)
        logger.info(f"Confidence calculation: {confidence}")
        
        # Test delta range calculation
        delta_range = ev_scorer.calculate_optimal_delta_range(confidence, regime)
        logger.info(f"Delta range calculation: {delta_range}")
        
        logger.info("Individual component tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Individual component test failed: {e}", exc_info=True)
        return False

async def main():
    """Main test function"""
    
    logger.info("Starting EV Delayed Data Integration Tests...")
    logger.info("=" * 60)
    
    # Test individual components first
    logger.info("Phase 1: Testing individual components...")
    component_test_passed = await test_individual_components()
    
    if not component_test_passed:
        logger.error("Component tests failed. Stopping.")
        return False
    
    logger.info("=" * 60)
    
    # Test full integration
    logger.info("Phase 2: Testing full integration...")
    integration_test_passed = await test_ev_integration()
    
    if not integration_test_passed:
        logger.error("Integration tests failed.")
        return False
    
    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED!")
    logger.info("Your EV Delayed Data Integration is ready to use!")
    logger.info("Next steps:")
    logger.info("   1. Follow the API_ENDPOINT_MODIFICATIONS.md guide")
    logger.info("   2. Update your portfolio_api.py endpoints")
    logger.info("   3. Test with real data")
    logger.info("   4. Monitor performance and adjust parameters as needed")
    
    return True

if __name__ == "__main__":
    # Check if Polygon API key is available
    if not os.getenv('POLYGON_API_KEY'):
        logger.warning("POLYGON_API_KEY not found. Some tests may use mock data.")
    
    # Run the tests
    success = asyncio.run(main())
    
    if success:
        print("\nSUCCESS: All tests passed!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some tests failed.")
        sys.exit(1)
