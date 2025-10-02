"""
Test script to verify API endpoints work with EV integration
"""

import requests
import json
import time

def test_api_endpoints():
    """Test the API endpoints with EV integration"""
    
    base_url = "http://localhost:8000"
    
    print("Testing EV-Enhanced API Endpoints...")
    print("=" * 50)
    
    # Test 1: Covered Call Recommendations
    print("\n1. Testing Covered Call Recommendations...")
    try:
        response = requests.get(f"{base_url}/api/options/recommendations/covered-calls")
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS Success! Generated {len(data.get('recommendations', []))} recommendations")
            print(f"   Scoring Method: {data.get('scoring_method', 'Unknown')}")
            print(f"   Market Regime: {data.get('market_regime', 'Unknown')}")
            print(f"   Delayed Data Optimization: {data.get('delayed_data_optimization', False)}")
            print(f"   Safety Margins Applied: {data.get('safety_margins_applied', False)}")
            print(f"   Filtered Recommendations: {data.get('filtered_recommendations', 0)}")
            
            # Show top recommendation details
            if data.get('recommendations'):
                top_rec = data['recommendations'][0]
                print(f"   Top Recommendation:")
                print(f"     Ticker: {top_rec.get('ticker', 'N/A')}")
                print(f"     EV Score: {top_rec.get('ev_score', top_rec.get('score', 'N/A'))}")
                print(f"     Expected Value: {top_rec.get('expected_value', 'N/A')}")
                print(f"     Confidence: {top_rec.get('confidence', 'N/A')}")
                print(f"     Delta: {top_rec.get('delta', 'N/A')}")
        else:
            print(f"FAILED Failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"FAILED Error: {e}")
    
    # Test 2: CSP Recommendations
    print("\n2. Testing Cash Secured Put Recommendations...")
    try:
        payload = {"tickers": "AAPL,MSFT"}
        response = requests.post(f"{base_url}/api/options/recommendations/cash-secured-puts", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS Success! Generated {len(data.get('recommendations', []))} recommendations")
            print(f"   Scoring Method: {data.get('scoring_method', 'Unknown')}")
            print(f"   Market Regime: {data.get('market_regime', 'Unknown')}")
            print(f"   Delayed Data Optimization: {data.get('delayed_data_optimization', False)}")
            print(f"   Safety Margins Applied: {data.get('safety_margins_applied', False)}")
            print(f"   Filtered Recommendations: {data.get('filtered_recommendations', 0)}")
            
            # Show top recommendation details
            if data.get('recommendations'):
                top_rec = data['recommendations'][0]
                print(f"   Top Recommendation:")
                print(f"     Ticker: {top_rec.get('ticker', 'N/A')}")
                print(f"     EV Score: {top_rec.get('ev_score', top_rec.get('score', 'N/A'))}")
                print(f"     Expected Value: {top_rec.get('expected_value', 'N/A')}")
                print(f"     Confidence: {top_rec.get('confidence', 'N/A')}")
                print(f"     Delta: {top_rec.get('delta', 'N/A')}")
        else:
            print(f"FAILED Failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"FAILED Error: {e}")
    
    print("\n" + "=" * 50)
    print("API Integration Test Complete!")

if __name__ == "__main__":
    print("Make sure your API server is running on http://localhost:8000")
    print("Starting test in 3 seconds...")
    time.sleep(3)
    test_api_endpoints()
