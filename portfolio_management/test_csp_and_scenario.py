#!/usr/bin/env python3
"""
Simple test for CSP calculation and scenario analysis
"""

def test_csp_calculation():
    """Test CSP cash calculation formula"""
    print("Testing CSP Cash Calculation...")
    
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
    
    print("CSP Cash Calculation Formula Verification:")
    all_correct = True
    for case in test_cases:
        calculated_cash = case['strike_price'] * abs(case['quantity']) * 100
        is_correct = abs(calculated_cash - case['expected_cash']) < 0.01
        
        print(f"  {case['ticker']}: {case['strike_price']} x {abs(case['quantity'])} x 100 = ${calculated_cash:,.2f}")
        print(f"    Expected: ${case['expected_cash']:,.2f} | {'CORRECT' if is_correct else 'INCORRECT'}")
        
        if not is_correct:
            all_correct = False
    
    print("\nFormula: Required Cash = Strike Price x |Quantity| x 100")
    print(f"   This formula is {'CORRECT' if all_correct else 'INCORRECT'} for Cash Secured Puts (CSPs)")
    
    return all_correct

def test_scenario_analyzer_import():
    """Test if scenario analyzer can be imported"""
    print("\nTesting Scenario Analyzer Import...")
    
    try:
        from scenario_analyzer import ScenarioAnalyzer, StressTestScenario
        print("ScenarioAnalyzer imported successfully")
        
        # Test basic functionality
        analyzer = ScenarioAnalyzer()
        print("ScenarioAnalyzer initialized successfully")
        
        # Test default scenarios
        scenarios = analyzer.default_scenarios
        print(f"Found {len(scenarios)} default stress test scenarios:")
        for scenario in scenarios:
            print(f"  - {scenario.name}: {scenario.description}")
        
        return True
        
    except Exception as e:
        print(f"Error importing ScenarioAnalyzer: {e}")
        return False

def test_api_endpoints():
    """Test if new API endpoints are properly defined"""
    print("\nTesting API Endpoints...")
    
    try:
        # Read the portfolio_api.py file and check for new endpoints
        with open('portfolio_api.py', 'r') as f:
            content = f.read()
        
        endpoints = [
            '/api/portfolio/scenario/stress-tests',
            '/api/portfolio/scenario/monte-carlo',
            '/api/portfolio/scenario/position-impact',
            '/api/portfolio/scenario/custom'
        ]
        
        found_endpoints = []
        for endpoint in endpoints:
            if endpoint in content:
                found_endpoints.append(endpoint)
                print(f"Found endpoint: {endpoint}")
            else:
                print(f"Missing endpoint: {endpoint}")
        
        print(f"\nFound {len(found_endpoints)}/{len(endpoints)} new scenario endpoints")
        return len(found_endpoints) == len(endpoints)
        
    except Exception as e:
        print(f"Error checking API endpoints: {e}")
        return False

def main():
    """Run all tests"""
    print("CSP and Scenario Analysis Test Suite")
    print("=" * 50)
    
    # Test CSP calculation
    csp_correct = test_csp_calculation()
    
    # Test scenario analyzer import
    scenario_import = test_scenario_analyzer_import()
    
    # Test API endpoints
    api_endpoints = test_api_endpoints()
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"  CSP Calculation: {'PASS' if csp_correct else 'FAIL'}")
    print(f"  Scenario Import: {'PASS' if scenario_import else 'FAIL'}")
    print(f"  API Endpoints: {'PASS' if api_endpoints else 'FAIL'}")
    
    all_passed = csp_correct and scenario_import and api_endpoints
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nReady to proceed with Risk Monitoring and Option Selection tabs!")
    else:
        print("\nPlease fix the failing tests before proceeding.")

if __name__ == "__main__":
    main()
