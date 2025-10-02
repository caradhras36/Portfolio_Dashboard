"""
Test raw recommendations without EV enhancement
"""

import requests
import json

def test_raw_recommendations():
    """Test raw recommendations"""
    
    base_url = "http://localhost:8000"
    
    print("Testing Raw Recommendations...")
    print("=" * 40)
    
    try:
        # Test covered calls with a simple request
        print("Testing Covered Calls (Raw)...")
        response = requests.get(f"{base_url}/api/options/recommendations/covered-calls")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {response.status_code}")
            print(f"  Total recommendations: {len(data.get('recommendations', []))}")
            print(f"  Total available: {data.get('total_available', 0)}")
            print(f"  Total considered: {data.get('total_considered', 0)}")
            print(f"  Filtered: {data.get('filtered_recommendations', 0)}")
            print(f"  Enhanced: {data.get('total_enhanced', 0)}")
            print(f"  Blocked tickers: {data.get('blocked_tickers', [])}")
            print(f"  Market regime: {data.get('market_regime', 'unknown')}")
            
            if data.get('recommendations'):
                top_rec = data['recommendations'][0]
                print(f"  Top recommendation: {top_rec.get('ticker', 'N/A')}")
                print(f"  Score: {top_rec.get('score', 'N/A')}")
                print(f"  EV Score: {top_rec.get('ev_score', 'N/A')}")
        else:
            print(f"  Error: {response.status_code} - {response.text}")
        
        # Test CSP
        print("\nTesting CSP (Raw)...")
        payload = {"tickers": "AAPL,MSFT"}
        response = requests.post(f"{base_url}/api/options/recommendations/cash-secured-puts", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {response.status_code}")
            print(f"  Total recommendations: {len(data.get('recommendations', []))}")
            print(f"  Total available: {data.get('total_available', 0)}")
            print(f"  Total considered: {data.get('total_considered', 0)}")
            print(f"  Filtered: {data.get('filtered_recommendations', 0)}")
            print(f"  Enhanced: {data.get('total_enhanced', 0)}")
            print(f"  Market regime: {data.get('market_regime', 'unknown')}")
            
            if data.get('recommendations'):
                top_rec = data['recommendations'][0]
                print(f"  Top recommendation: {top_rec.get('ticker', 'N/A')}")
                print(f"  Score: {top_rec.get('score', 'N/A')}")
                print(f"  EV Score: {top_rec.get('ev_score', 'N/A')}")
        else:
            print(f"  Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    test_raw_recommendations()
