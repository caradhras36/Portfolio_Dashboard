#!/usr/bin/env python3
"""
Test liquidity calculation directly
"""

import asyncio
import sys
sys.path.append('.')
sys.path.append('./shared')
sys.path.append('./portfolio_management')

from datetime import datetime, timedelta
from price_cache import get_price_cache

async def test_liquidity():
    """Test getting price data with volume"""
    try:
        price_cache = get_price_cache()
        
        # Test with one ticker
        ticker = 'APLD'
        end_date = (datetime.now() - timedelta(days=1)).date()
        start_date = end_date - timedelta(days=365)
        
        print(f"\nFetching 1 year of data for {ticker}...")
        print(f"Date range: {start_date} to {end_date}\n")
        
        # Single ticker fetch
        prices = await price_cache.get_cached_prices(ticker, start_date, end_date)
        
        print(f"Got {len(prices)} price records")
        
        if prices:
            # Check volume data
            with_volume = [p for p in prices if p.volume and p.volume > 0]
            print(f"Records with volume > 0: {len(with_volume)}")
            
            # Show first and last
            print(f"\nFirst record: date={prices[0].date}, close=${prices[0].close_price:.2f}, volume={prices[0].volume}")
            print(f"Last record: date={prices[-1].date}, close=${prices[-1].close_price:.2f}, volume={prices[-1].volume}")
            
            # Now test batch fetch
            print(f"\n\nTesting BATCH fetch...")
            tickers = ['APLD', 'ROOT', 'NBIS']
            batch_data = await price_cache.get_cached_prices_batch(tickers, start_date, end_date)
            
            for t, p_list in batch_data.items():
                with_vol = [p for p in p_list if p.volume and p.volume > 0]
                print(f"{t}: {len(p_list)} records, {len(with_vol)} with volume")
        else:
            print("NO DATA RETURNED!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_liquidity())
