"""
Test script to verify different risk tolerances for CCs vs CSPs
"""

import sys
sys.path.append('portfolio_management')

from optimized_scoring_system import OptimizedOptionsScorer

def test_risk_tolerances():
    """Test different risk tolerances for covered calls vs CSPs"""
    
    print("Testing Risk Tolerance Differences...")
    print("=" * 50)
    
    # Initialize EV scorer
    ev_scorer = OptimizedOptionsScorer()
    
    # Test data
    option_data = {
        'ticker': 'AAPL',
        'premium': 2.50,
        'strike_price': 150.0,
        'current_stock_price': 155.0,
        'delta': 0.30,  # 30% assignment probability
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
        'price_change_5d': -0.01
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
    
    print(f"Testing option with 30% assignment probability (delta = 0.30)")
    print(f"Covered Call Risk Tolerance: {ev_scorer.max_assignment_probability_cc * 100}%")
    print(f"CSP Risk Tolerance: {ev_scorer.max_assignment_probability_csp * 100}%")
    print()
    
    # Test Covered Call (more conservative)
    print("1. Testing Covered Call (More Conservative):")
    cc_result = ev_scorer.calculate_optimized_score(
        option_data, stock_data, market_data, portfolio_positions, 'call'
    )
    
    print(f"   EV Score: {cc_result['final_score']:.2f}")
    print(f"   Expected Value: {cc_result['expected_value']:.4f}")
    print(f"   Confidence: {cc_result['confidence']:.2f}")
    print(f"   Delta Range: {cc_result['optimal_delta_range']}")
    print(f"   Delta Optimization Score: {cc_result['delta_optimization_score']:.2f}")
    
    # Test CSP (less conservative)
    print("\n2. Testing Cash Secured Put (Less Conservative):")
    csp_result = ev_scorer.calculate_optimized_score(
        option_data, stock_data, market_data, portfolio_positions, 'put'
    )
    
    print(f"   EV Score: {csp_result['final_score']:.2f}")
    print(f"   Expected Value: {csp_result['expected_value']:.4f}")
    print(f"   Confidence: {csp_result['confidence']:.2f}")
    print(f"   Delta Range: {csp_result['optimal_delta_range']}")
    print(f"   Delta Optimization Score: {csp_result['delta_optimization_score']:.2f}")
    
    print("\n" + "=" * 50)
    print("Risk Tolerance Test Complete!")
    
    # Show the differences
    print(f"\nKey Differences:")
    print(f"- CC Max Assignment: {ev_scorer.max_assignment_probability_cc * 100}% (more conservative)")
    print(f"- CSP Max Assignment: {ev_scorer.max_assignment_probability_csp * 100}% (less conservative)")
    print(f"- CC Delta Range: {cc_result['optimal_delta_range']} (more conservative)")
    print(f"- CSP Delta Range: {csp_result['optimal_delta_range']} (less conservative)")
    print(f"- CC Confidence Multiplier: 0.4-1.0 (more conservative)")
    print(f"- CSP Confidence Multiplier: 0.5-1.0 (less conservative)")

if __name__ == "__main__":
    test_risk_tolerances()
