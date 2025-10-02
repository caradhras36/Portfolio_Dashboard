"""
Debug the EV integration to see what's happening
"""

import sys
sys.path.append('portfolio_management')

def debug_ev_integration():
    """Debug the EV integration"""
    
    print("Debugging EV Integration...")
    print("=" * 40)
    
    try:
        from ev_delayed_integration import EVDelayedDataIntegration
        from optimized_scoring_system import OptimizedOptionsScorer
        
        # Test EV scorer directly
        print("Testing EV Scorer Directly:")
        ev_scorer = OptimizedOptionsScorer()
        
        # Test with a simple option
        option_data = {
            'ticker': 'AAPL',
            'premium': 2.50,
            'strike_price': 150.0,
            'current_stock_price': 155.0,
            'delta': 0.25,
            'days_to_expiration': 30,
            'theta': -0.05
        }
        
        stock_data = {
            'rsi': 70,
            'macd': {'histogram': -0.3},
            'bollinger_bands': {'position': 0.8},
            'support_resistance': {'resistance': 160.0},
            'current_price': 155.0,
            'volume_ratio': 1.2,
            'price_change_5d': -0.01,
            'price_change_1h': 0.01,
            'price_change_4h': 0.02
        }
        
        market_data = {
            'vix': 20.0,
            'vix_20_ma': 22.0,
            'spy_price': 420.0,
            'spy_20_ma': 415.0,
            'spy_50_ma': 410.0,
            'advancers': 0.55,
            'decliners': 0.45
        }
        
        portfolio_positions = []
        
        # Test covered call
        cc_result = ev_scorer.calculate_optimized_score(
            option_data, stock_data, market_data, portfolio_positions, 'call'
        )
        print(f"  CC Result: {cc_result is not None}")
        if cc_result:
            print(f"  CC Score: {cc_result.get('final_score', 'N/A')}")
            print(f"  CC Expected Value: {cc_result.get('expected_value', 'N/A')}")
        
        # Test CSP
        csp_result = ev_scorer.calculate_optimized_score(
            option_data, stock_data, market_data, portfolio_positions, 'put'
        )
        print(f"  CSP Result: {csp_result is not None}")
        if csp_result:
            print(f"  CSP Score: {csp_result.get('final_score', 'N/A')}")
            print(f"  CSP Expected Value: {csp_result.get('expected_value', 'N/A')}")
        
        # Test with a recommendation-like object
        print("\nTesting with Recommendation Object:")
        recommendation = {
            'ticker': 'AAPL',
            'premium': 2.50,
            'strike_price': 150.0,
            'current_stock_price': 155.0,
            'delta': 0.25,
            'days_to_expiration': 30,
            'theta': -0.05,
            'rsi': 70,
            'macd_histogram': -0.3,
            'bb_position': 0.8,
            'resistance_level': 160.0,
            'volume_ratio': 1.2,
            'price_change_5d': -0.01,
            'price_change_1h': 0.01,
            'price_change_4h': 0.02
        }
        
        # Test the integration function
        integration = EVDelayedDataIntegration()
        
        # Test enhance_recommendation_with_ev_delayed_data
        enhanced_rec = integration.enhance_recommendation_with_ev_delayed_data(
            recommendation, stock_data, portfolio_positions, 'call'
        )
        
        print(f"  Enhanced Result: {enhanced_rec is not None}")
        if enhanced_rec:
            print(f"  Enhanced Score: {enhanced_rec.get('ev_score', 'N/A')}")
            print(f"  Enhanced Expected Value: {enhanced_rec.get('expected_value', 'N/A')}")
        else:
            print("  Enhanced Result is None - this is why all recommendations are filtered out!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ev_integration()
