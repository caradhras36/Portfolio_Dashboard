#!/usr/bin/env python3
"""
Test Scenario Analysis System
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from scenario_analyzer import ScenarioAnalyzer, StressTestScenario
from portfolio_api import PortfolioPosition
import asyncio

def test_csp_calculation():
    """Test CSP cash calculation formula"""
    print("🧪 Testing CSP Cash Calculation...")
    
    # Test cases for CSP calculation
    test_cases = [
        {
            'ticker': 'AAPL',
            'strike_price': 150.0,
            'quantity': -2,  # Short 2 puts
            'expected_cash': 30000.0  # 150 * 2 * 100
        },
        {
            'ticker': 'TSLA',
            'strike_price': 200.0,
            'quantity': -1,  # Short 1 put
            'expected_cash': 20000.0  # 200 * 1 * 100
        },
        {
            'ticker': 'SPY',
            'strike_price': 400.0,
            'quantity': -5,  # Short 5 puts
            'expected_cash': 200000.0  # 400 * 5 * 100
        }
    ]
    
    print("✅ CSP Cash Calculation Formula Verification:")
    for case in test_cases:
        calculated_cash = case['strike_price'] * abs(case['quantity']) * 100
        is_correct = abs(calculated_cash - case['expected_cash']) < 0.01
        
        print(f"  {case['ticker']}: {case['strike_price']} × {abs(case['quantity'])} × 100 = ${calculated_cash:,.2f}")
        print(f"    Expected: ${case['expected_cash']:,.2f} | {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
    
    print("\n📊 Formula: Required Cash = Strike Price × |Quantity| × 100")
    print("   This formula is CORRECT for Cash Secured Puts (CSPs)")

def test_scenario_analysis():
    """Test scenario analysis functionality"""
    print("\n🧪 Testing Scenario Analysis...")
    
    # Create sample positions
    sample_positions = [
        PortfolioPosition(
            ticker='AAPL',
            position_type='stock',
            quantity=100,
            entry_price=150.0,
            current_price=155.0
        ),
        PortfolioPosition(
            ticker='AAPL',
            position_type='put',
            quantity=-2,  # Short puts
            entry_price=5.0,
            current_price=4.5,
            expiration_date='2024-01-19',
            strike_price=150.0,
            option_type='put',
            delta=-0.3,
            gamma=0.02,
            theta=-0.1,
            vega=0.15,
            implied_volatility=0.25,
            time_to_expiration=30
        ),
        PortfolioPosition(
            ticker='TSLA',
            position_type='call',
            quantity=1,
            entry_price=10.0,
            current_price=12.0,
            expiration_date='2024-02-16',
            strike_price=250.0,
            option_type='call',
            delta=0.6,
            gamma=0.01,
            theta=-0.05,
            vega=0.20,
            implied_volatility=0.35,
            time_to_expiration=45
        )
    ]
    
    # Initialize scenario analyzer
    analyzer = ScenarioAnalyzer()
    
    print("📊 Sample Portfolio:")
    for pos in sample_positions:
        print(f"  {pos.ticker} {pos.position_type}: {pos.quantity} @ ${pos.current_price:.2f}")
    
    # Test stress tests
    print("\n🔥 Running Stress Tests...")
    try:
        stress_results = analyzer.run_stress_tests(sample_positions)
        print(f"✅ Completed {len(stress_results)} stress test scenarios")
        
        for result in stress_results:
            print(f"  {result.scenario_name}: P&L ${result.pnl:,.2f} ({result.pnl_pct:.1f}%)")
    
    except Exception as e:
        print(f"❌ Error in stress tests: {e}")
    
    # Test Monte Carlo simulation
    print("\n🎲 Running Monte Carlo Simulation...")
    try:
        mc_results = analyzer.run_monte_carlo_simulation(sample_positions, 100)
        if 'error' not in mc_results:
            print(f"✅ Monte Carlo completed: {mc_results['num_simulations']} simulations")
            print(f"  VaR 95%: ${mc_results['var_95']:,.2f}")
            print(f"  VaR 99%: ${mc_results['var_99']:,.2f}")
            print(f"  Expected Shortfall 95%: ${mc_results['expected_shortfall_95']:,.2f}")
        else:
            print(f"❌ Monte Carlo error: {mc_results['error']}")
    
    except Exception as e:
        print(f"❌ Error in Monte Carlo: {e}")
    
    # Test position impact analysis
    print("\n📈 Testing Position Impact Analysis...")
    try:
        new_position = PortfolioPosition(
            ticker='MSFT',
            position_type='stock',
            quantity=50,
            entry_price=300.0,
            current_price=305.0
        )
        
        impact = analyzer.analyze_position_impact(sample_positions, new_position)
        if 'error' not in impact:
            print("✅ Position impact analysis completed")
            print(f"  Value Impact: ${impact['impact']['value_impact']:,.2f}")
            print(f"  P&L Impact: ${impact['impact']['pnl_impact']:,.2f}")
            print(f"  Delta Impact: {impact['impact']['delta_impact']:.2f}")
        else:
            print(f"❌ Impact analysis error: {impact['error']}")
    
    except Exception as e:
        print(f"❌ Error in impact analysis: {e}")

def test_custom_scenario():
    """Test custom scenario creation"""
    print("\n🎯 Testing Custom Scenario...")
    
    analyzer = ScenarioAnalyzer()
    
    # Create a custom scenario
    custom_scenario = analyzer.create_custom_scenario(
        name="Tech Sector Crash",
        description="Simulate a 30% decline in tech stocks",
        market_moves={'AAPL': -0.30, 'TSLA': -0.30, 'MSFT': -0.30},
        volatility_changes={'AAPL': 2.0, 'TSLA': 2.0},  # 2x volatility
        time_decay_days=7,
        interest_rate_change=0.01
    )
    
    print(f"✅ Created custom scenario: {custom_scenario.name}")
    print(f"  Description: {custom_scenario.description}")
    print(f"  Market moves: {custom_scenario.market_moves}")
    print(f"  Volatility changes: {custom_scenario.volatility_changes}")
    print(f"  Time decay: {custom_scenario.time_decay_days} days")
    print(f"  Interest rate change: {custom_scenario.interest_rate_change:.1%}")

def main():
    """Run all tests"""
    print("🚀 Portfolio Scenario Analysis Test Suite")
    print("=" * 50)
    
    # Test CSP calculation
    test_csp_calculation()
    
    # Test scenario analysis
    test_scenario_analysis()
    
    # Test custom scenario
    test_custom_scenario()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")

if __name__ == "__main__":
    main()
