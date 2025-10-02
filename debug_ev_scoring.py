"""
Debug script to see what's happening with EV scoring
"""

import sys
sys.path.append('portfolio_management')

from optimized_scoring_system import OptimizedOptionsScorer

def debug_ev_scoring():
    """Debug the EV scoring system"""
    
    print("Debugging EV Scoring System...")
    print("=" * 40)
    
    # Initialize EV scorer
    ev_scorer = OptimizedOptionsScorer()
    
    # Test with a simple option
    option_data = {
        'ticker': 'AAPL',
        'premium': 2.50,
        'strike_price': 150.0,
        'current_stock_price': 155.0,
        'delta': 0.25,  # 25% assignment probability
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
    
    print("Testing Covered Call (Conservative):")
    try:
        cc_result = ev_scorer.calculate_optimized_score(
            option_data, stock_data, market_data, portfolio_positions, 'call'
        )
        print(f"  Result: {cc_result is not None}")
        if cc_result:
            print(f"  Final Score: {cc_result.get('final_score', 'N/A')}")
            print(f"  Expected Value: {cc_result.get('expected_value', 'N/A')}")
            print(f"  Confidence: {cc_result.get('confidence', 'N/A')}")
            print(f"  Delta Range: {cc_result.get('optimal_delta_range', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nTesting CSP (Less Conservative):")
    try:
        csp_result = ev_scorer.calculate_optimized_score(
            option_data, stock_data, market_data, portfolio_positions, 'put'
        )
        print(f"  Result: {csp_result is not None}")
        if csp_result:
            print(f"  Final Score: {csp_result.get('final_score', 'N/A')}")
            print(f"  Expected Value: {csp_result.get('expected_value', 'N/A')}")
            print(f"  Confidence: {csp_result.get('confidence', 'N/A')}")
            print(f"  Delta Range: {csp_result.get('optimal_delta_range', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"\nRisk Tolerances:")
    print(f"  CC Max Assignment: {ev_scorer.max_assignment_probability_cc * 100}%")
    print(f"  CSP Max Assignment: {ev_scorer.max_assignment_probability_csp * 100}%")

if __name__ == "__main__":
    debug_ev_scoring()
