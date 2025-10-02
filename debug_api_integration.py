"""
Debug script to test the API integration layer
"""

import sys
sys.path.append('portfolio_management')

from ev_delayed_integration import EVDelayedDataIntegration

def debug_api_integration():
    """Debug the API integration"""
    
    print("Debugging API Integration...")
    print("=" * 40)
    
    # Initialize integration
    integration = EVDelayedDataIntegration()
    
    print("Testing Covered Call Integration:")
    try:
        cc_result = integration.get_enhanced_cc_recommendations()
        print(f"  Status: Success")
        print(f"  Total recommendations: {len(cc_result.get('recommendations', []))}")
        print(f"  Total available: {cc_result.get('total_available', 0)}")
        print(f"  Total considered: {cc_result.get('total_considered', 0)}")
        print(f"  Filtered: {cc_result.get('filtered_recommendations', 0)}")
        print(f"  Enhanced: {cc_result.get('total_enhanced', 0)}")
        
        if cc_result.get('recommendations'):
            top_rec = cc_result['recommendations'][0]
            print(f"  Top recommendation: {top_rec.get('ticker', 'N/A')}")
            print(f"  Score: {top_rec.get('score', 'N/A')}")
            print(f"  EV Score: {top_rec.get('ev_score', 'N/A')}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting CSP Integration:")
    try:
        csp_result = integration.get_enhanced_csp_recommendations("AAPL,MSFT")
        print(f"  Status: Success")
        print(f"  Total recommendations: {len(csp_result.get('recommendations', []))}")
        print(f"  Total available: {csp_result.get('total_available', 0)}")
        print(f"  Total considered: {csp_result.get('total_considered', 0)}")
        print(f"  Filtered: {csp_result.get('filtered_recommendations', 0)}")
        print(f"  Enhanced: {csp_result.get('total_enhanced', 0)}")
        
        if csp_result.get('recommendations'):
            top_rec = csp_result['recommendations'][0]
            print(f"  Top recommendation: {top_rec.get('ticker', 'N/A')}")
            print(f"  Score: {top_rec.get('score', 'N/A')}")
            print(f"  EV Score: {top_rec.get('ev_score', 'N/A')}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_api_integration()
