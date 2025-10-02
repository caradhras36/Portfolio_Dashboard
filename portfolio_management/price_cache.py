#!/usr/bin/env python3
"""
Historical Price Cache - Stores daily prices in database to reduce API calls
"""

import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta, date
from dataclasses import dataclass
import os

# Import from config (not environment variables - they may not be set!)
try:
    import sys
    sys.path.append('.')
    sys.path.append('./shared')
    from supabase import create_client, Client
    from shared.config import SUPABASE_URL, SUPABASE_KEY
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
    if not supabase:
        print("WARNING: price_cache.py - No Supabase connection (check shared/config.py)")
except Exception as e:
    print(f"ERROR: price_cache.py - Failed to initialize Supabase: {e}")
    supabase = None

logger = logging.getLogger(__name__)

@dataclass
class PriceData:
    """Daily price data"""
    ticker: str
    date: date
    close_price: float
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    volume: Optional[int] = None

class PriceCache:
    """Manages caching of historical prices in database"""
    
    def __init__(self, supabase_client=None):
        self.supabase = supabase_client or supabase
        self._batch_cache = {}  # In-memory cache for current session
        
    async def get_cached_prices_batch(self, tickers: List[str], start_date: date, end_date: date) -> Dict[str, List[PriceData]]:
        """Get cached prices for multiple tickers in ONE database query"""
        try:
            if not self.supabase or not tickers:
                return {}
            
            # Check in-memory cache first
            cache_key = f"{start_date}_{end_date}"
            if cache_key in self._batch_cache:
                logger.debug(f"Using in-memory cache for {len(tickers)} tickers")
                return {ticker: self._batch_cache[cache_key].get(ticker, []) for ticker in tickers}
            
            # Fetch ALL prices for ALL tickers using pagination
            # Supabase has a hard server limit (usually 1000 rows per request)
            # We need to paginate to get all data
            all_rows = []
            page_size = 1000
            offset = 0
            
            while True:
                response = self.supabase.table('historical_prices')\
                    .select('*')\
                    .in_('ticker', tickers)\
                    .gte('date', start_date.isoformat())\
                    .lte('date', end_date.isoformat())\
                    .order('ticker')\
                    .order('date')\
                    .range(offset, offset + page_size - 1)\
                    .execute()
                
                if not response.data or len(response.data) == 0:
                    break
                    
                all_rows.extend(response.data)
                
                if len(response.data) < page_size:
                    break
                    
                offset += page_size
            
            # Group by ticker
            prices_by_ticker = {}
            for ticker in tickers:
                prices_by_ticker[ticker] = []
            
            if all_rows:
                for row in all_rows:
                    ticker = row['ticker']
                    if ticker in prices_by_ticker:
                        prices_by_ticker[ticker].append(PriceData(
                            ticker=row['ticker'],
                            date=datetime.fromisoformat(row['date']).date(),
                            close_price=float(row['close_price']),
                            open_price=float(row['open_price']) if row.get('open_price') else None,
                            high_price=float(row['high_price']) if row.get('high_price') else None,
                            low_price=float(row['low_price']) if row.get('low_price') else None,
                            volume=int(row['volume']) if row.get('volume') else None
                        ))
            
            # Cache in memory
            self._batch_cache[cache_key] = prices_by_ticker
            
            found_count = sum(len(prices) for prices in prices_by_ticker.values())
            logger.info(f"✅ Batch loaded {found_count} price records for {len(tickers)} tickers in ONE query")
            
            return prices_by_ticker
            
        except Exception as e:
            logger.error(f"Error in batch price fetch: {e}")
            return {}
    
    async def get_cached_prices(self, ticker: str, start_date: date, end_date: date) -> List[PriceData]:
        """Get cached prices from database for a date range"""
        try:
            if not self.supabase:
                logger.warning("No database connection available")
                return []
            
            response = self.supabase.table('historical_prices')\
                .select('*')\
                .eq('ticker', ticker)\
                .gte('date', start_date.isoformat())\
                .lte('date', end_date.isoformat())\
                .order('date')\
                .execute()
            
            if response.data:
                prices = []
                for row in response.data:
                    prices.append(PriceData(
                        ticker=row['ticker'],
                        date=datetime.fromisoformat(row['date']).date(),
                        close_price=float(row['close_price']),
                        open_price=float(row['open_price']) if row.get('open_price') else None,
                        high_price=float(row['high_price']) if row.get('high_price') else None,
                        low_price=float(row['low_price']) if row.get('low_price') else None,
                        volume=int(row['volume']) if row.get('volume') else None
                    ))
                logger.debug(f"Found {len(prices)} cached prices for {ticker}")
                return prices
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching cached prices for {ticker}: {e}")
            return []
    
    async def save_prices(self, prices: List[PriceData]) -> bool:
        """Save prices to database (upsert - update if exists, insert if not)"""
        try:
            if not self.supabase or not prices:
                return False
            
            # Convert to dict format for database
            records = []
            for price in prices:
                records.append({
                    'ticker': price.ticker,
                    'date': price.date.isoformat(),
                    'close_price': price.close_price,
                    'open_price': price.open_price,
                    'high_price': price.high_price,
                    'low_price': price.low_price,
                    'volume': int(price.volume) if price.volume else None
                })
            
            # Upsert (insert or update on conflict)
            response = self.supabase.table('historical_prices')\
                .upsert(records, on_conflict='ticker,date')\
                .execute()
            
            logger.info(f"✅ Saved {len(records)} price records to cache")
            return True
            
        except Exception as e:
            logger.error(f"Error saving prices to cache: {e}")
            return False
    
    async def get_missing_dates(self, ticker: str, start_date: date, end_date: date) -> List[date]:
        """Find which dates are missing from cache in a date range"""
        try:
            # Get cached prices
            cached_prices = await self.get_cached_prices(ticker, start_date, end_date)
            cached_dates = {p.date for p in cached_prices}
            
            # Generate all trading days (Monday-Friday, excluding weekends)
            all_dates = []
            current = start_date
            while current <= end_date:
                # Skip weekends (Saturday=5, Sunday=6)
                if current.weekday() < 5:
                    all_dates.append(current)
                current += timedelta(days=1)
            
            # Find missing dates
            missing = [d for d in all_dates if d not in cached_dates]
            
            if missing:
                logger.debug(f"{ticker}: {len(missing)} dates missing from cache")
            
            return missing
            
        except Exception as e:
            logger.error(f"Error checking missing dates for {ticker}: {e}")
            return []

# Global instance
_price_cache = None

def get_price_cache():
    """Get or create price cache instance"""
    global _price_cache
    if _price_cache is None:
        _price_cache = PriceCache()
    return _price_cache
