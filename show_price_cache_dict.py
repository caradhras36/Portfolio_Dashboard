#!/usr/bin/env python3
"""
Show what's in price_cache for each ticker (as a dictionary)
"""

import asyncio
import sys
sys.path.append('.')
sys.path.append('./shared')
sys.path.append('./portfolio_management')

from datetime import datetime, timedelta
from price_cache import get_price_cache

async def show_cache_dict():
    """Show cached prices as a dictionary for each ticker"""
    try:
        price_cache = get_price_cache()
        
        # Get tickers from portfolio
        from supabase import create_client
        from shared.config import SUPABASE_URL, SUPABASE_KEY
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        response = supabase.table('portfolio_positions')\
            .select('ticker')\
            .eq('position_type', 'stock')\
            .execute()
        
        tickers = list(set([row['ticker'] for row in response.data]))  # Unique tickers
        print(f"\nFound {len(tickers)} unique stock tickers\n")
        
        # Fetch 1 year for all tickers using batch (like the real code does)
        end_date = (datetime.now() - timedelta(days=1)).date()
        start_date = end_date - timedelta(days=365)
        
        print(f"Fetching batch data from {start_date} to {end_date}...\n")
        price_data_cache = await price_cache.get_cached_prices_batch(tickers, start_date, end_date)
        
        print(f"=== PRICE_DATA_CACHE DICTIONARY ===\n")
        print(f"Type: {type(price_data_cache)}")
        print(f"Keys (tickers): {len(price_data_cache.keys())}\n")
        
        # Show first 10 tickers
        for i, (ticker, prices) in enumerate(list(price_data_cache.items())[:10]):
            with_volume = [p for p in prices if p.volume and p.volume > 0]
            print(f"{ticker}:")
            print(f"  Total prices: {len(prices)}")
            print(f"  With volume: {len(with_volume)}")
            if prices:
                print(f"  First date: {prices[0].date}")
                print(f"  Last date: {prices[-1].date}")
            print()
        
        # Check if any have 0 or < 20 records
        empty_or_small = {t: len(p) for t, p in price_data_cache.items() if len(p) < 20}
        if empty_or_small:
            print(f"\n=== PROBLEM: Tickers with < 20 records ===")
            for ticker, count in empty_or_small.items():
                print(f"{ticker}: {count} records (WOULD GET 71 FALLBACK!)")
        else:
            print("\n[OK] All tickers have >= 20 records")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(show_cache_dict())
