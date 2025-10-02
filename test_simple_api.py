"""
Simple test to check if the API is working
"""

import requests
import json

def test_simple_api():
    """Test if the API is responding"""
    
    base_url = "http://localhost:8000"
    
    print("Testing API Response...")
    print("=" * 30)
    
    try:
        # Test a simple endpoint first
        response = requests.get(f"{base_url}/api/portfolio/positions")
        print(f"Portfolio positions status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                print(f"Portfolio positions: {len(data)} positions")
            else:
                print(f"Portfolio positions: {len(data.get('positions', []))} positions")
        
        # Test covered calls
        print("\nTesting Covered Calls...")
        response = requests.get(f"{base_url}/api/options/recommendations/covered-calls")
        print(f"Covered calls status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Covered calls: {len(data.get('recommendations', []))} recommendations")
            if data.get('recommendations'):
                top_rec = data['recommendations'][0]
                print(f"Top recommendation: {top_rec.get('ticker', 'N/A')} - Score: {top_rec.get('score', 'N/A')}")
        else:
            print(f"Error: {response.text}")
        
        # Test CSP
        print("\nTesting CSP...")
        payload = {"tickers": "AAPL"}
        response = requests.post(f"{base_url}/api/options/recommendations/cash-secured-puts", json=payload)
        print(f"CSP status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"CSP: {len(data.get('recommendations', []))} recommendations")
            if data.get('recommendations'):
                top_rec = data['recommendations'][0]
                print(f"Top recommendation: {top_rec.get('ticker', 'N/A')} - Score: {top_rec.get('score', 'N/A')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    test_simple_api()
