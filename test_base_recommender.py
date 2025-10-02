"""
Test the base recommender to see if it's generating recommendations
"""

import sys
sys.path.append('portfolio_management')

def test_base_recommender():
    """Test the base recommender"""
    
    print("Testing Base Recommender...")
    print("=" * 30)
    
    try:
        from cc_recommender import CoveredCallRecommender
        from csp_recommender import CashSecuredPutRecommender
        
        # Test covered calls
        print("Testing Covered Call Recommender:")
        cc_recommender = CoveredCallRecommender()
        
        # Get some stock positions (mock data)
        stock_positions = [
            {'ticker': 'AAPL', 'shares': 100, 'current_price': 155.0},
            {'ticker': 'MSFT', 'shares': 50, 'current_price': 350.0}
        ]
        
        blocked_tickers = set()
        cc_shares_committed = {}
        
        result = cc_recommender.get_recommendations(
            stock_positions, blocked_tickers, cc_shares_committed
        )
        
        print(f"  Status: Success")
        print(f"  Total recommendations: {len(result.get('recommendations', []))}")
        print(f"  Total available: {result.get('total_available', 0)}")
        print(f"  Total considered: {result.get('total_considered', 0)}")
        
        if result.get('recommendations'):
            top_rec = result['recommendations'][0]
            print(f"  Top recommendation: {top_rec.get('ticker', 'N/A')}")
            print(f"  Score: {top_rec.get('score', 'N/A')}")
        else:
            print("  No recommendations generated")
        
        # Test CSP
        print("\nTesting CSP Recommender:")
        csp_recommender = CashSecuredPutRecommender()
        
        result = csp_recommender.get_recommendations("AAPL,MSFT")
        
        print(f"  Status: Success")
        print(f"  Total recommendations: {len(result.get('recommendations', []))}")
        print(f"  Total available: {result.get('total_available', 0)}")
        print(f"  Total considered: {result.get('total_considered', 0)}")
        
        if result.get('recommendations'):
            top_rec = result['recommendations'][0]
            print(f"  Top recommendation: {top_rec.get('ticker', 'N/A')}")
            print(f"  Score: {top_rec.get('score', 'N/A')}")
        else:
            print("  No recommendations generated")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_base_recommender()
