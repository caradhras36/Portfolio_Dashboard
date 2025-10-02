#!/usr/bin/env python3
"""
Greeks Manager - Handles reading and writing Greeks to separate table
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class OptionGreeks:
    """Greeks data structure"""
    ticker: str
    strike_price: float
    expiration_date: str
    option_type: str
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None
    time_to_expiration: Optional[float] = None
    underlying_price: Optional[float] = None
    fetched_at: Optional[datetime] = None

class GreeksManager:
    """Manages Greeks in separate database table for fast access"""
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.cache = {}  # In-memory cache for ultra-fast access
        
    async def get_greeks(self, ticker: str, strike: float, expiration: str, option_type: str) -> Optional[OptionGreeks]:
        """Get Greeks for a specific option from database"""
        try:
            # Check memory cache first
            cache_key = f"{ticker}_{strike}_{expiration}_{option_type}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Query database
            response = self.supabase.table('option_greeks').select('*').eq(
                'ticker', ticker
            ).eq('strike_price', float(strike)).eq(
                'expiration_date', expiration
            ).eq('option_type', option_type).execute()
            
            if response.data and len(response.data) > 0:
                row = response.data[0]
                greeks = OptionGreeks(
                    ticker=row['ticker'],
                    strike_price=float(row['strike_price']),
                    expiration_date=row['expiration_date'],
                    option_type=row['option_type'],
                    delta=float(row['delta']),
                    gamma=float(row['gamma']),
                    theta=float(row['theta']),
                    vega=float(row['vega']),
                    rho=float(row['rho']) if row.get('rho') else None,
                    implied_volatility=float(row['implied_volatility']) if row.get('implied_volatility') else None,
                    time_to_expiration=float(row['time_to_expiration']) if row.get('time_to_expiration') else None,
                    underlying_price=float(row['underlying_price']) if row.get('underlying_price') else None,
                    fetched_at=datetime.fromisoformat(row['fetched_at']) if row.get('fetched_at') else None
                )
                
                # Cache in memory
                self.cache[cache_key] = greeks
                return greeks
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Greeks for {ticker}: {e}")
            return None
    
    async def get_greeks_batch(self, options: List[Dict]) -> Dict[str, OptionGreeks]:
        """Get Greeks for multiple options in one query (FAST!)"""
        try:
            if not options:
                return {}
            
            # Build OR conditions for batch query
            # Query all Greeks in one database call
            response = self.supabase.table('option_greeks').select('*').execute()
            
            if not response.data:
                logger.warning("No Greeks found in database")
                return {}
            
            # Build lookup dict
            greeks_dict = {}
            for row in response.data:
                key = f"{row['ticker']}_{row['strike_price']}_{row['expiration_date']}_{row['option_type']}"
                greeks_dict[key] = OptionGreeks(
                    ticker=row['ticker'],
                    strike_price=float(row['strike_price']),
                    expiration_date=row['expiration_date'],
                    option_type=row['option_type'],
                    delta=float(row['delta']),
                    gamma=float(row['gamma']),
                    theta=float(row['theta']),
                    vega=float(row['vega']),
                    rho=float(row['rho']) if row.get('rho') else None,
                    implied_volatility=float(row['implied_volatility']) if row.get('implied_volatility') else None,
                    time_to_expiration=float(row['time_to_expiration']) if row.get('time_to_expiration') else None,
                    underlying_price=float(row['underlying_price']) if row.get('underlying_price') else None
                )
            
            logger.info(f"✅ Loaded {len(greeks_dict)} Greeks from database in single query")
            return greeks_dict
            
        except Exception as e:
            logger.error(f"Error in batch Greeks fetch: {e}")
            return {}
    
    async def save_greeks(self, greeks: OptionGreeks):
        """Save or update Greeks in database"""
        try:
            data = {
                'ticker': greeks.ticker,
                'strike_price': float(greeks.strike_price),
                'expiration_date': greeks.expiration_date,
                'option_type': greeks.option_type,
                'delta': float(greeks.delta),
                'gamma': float(greeks.gamma),
                'theta': float(greeks.theta),
                'vega': float(greeks.vega),
                'rho': float(greeks.rho) if greeks.rho else None,
                'implied_volatility': float(greeks.implied_volatility) if greeks.implied_volatility else None,
                'time_to_expiration': float(greeks.time_to_expiration) if greeks.time_to_expiration else None,
                'underlying_price': float(greeks.underlying_price) if greeks.underlying_price else None,
                'updated_at': datetime.now().isoformat()
            }
            
            # Upsert (insert or update if exists)
            self.supabase.table('option_greeks').upsert(data).execute()
            
            # Update cache
            cache_key = f"{greeks.ticker}_{greeks.strike_price}_{greeks.expiration_date}_{greeks.option_type}"
            self.cache[cache_key] = greeks
            
            logger.debug(f"💾 Saved Greeks for {greeks.ticker} ${greeks.strike_price}")
            
        except Exception as e:
            logger.error(f"Error saving Greeks: {e}")
            raise
    
    async def save_greeks_batch(self, greeks_list: List[OptionGreeks]):
        """Save multiple Greeks efficiently"""
        try:
            data_list = []
            for greeks in greeks_list:
                data_list.append({
                    'ticker': greeks.ticker,
                    'strike_price': float(greeks.strike_price),
                    'expiration_date': greeks.expiration_date,
                    'option_type': greeks.option_type,
                    'delta': float(greeks.delta),
                    'gamma': float(greeks.gamma),
                    'theta': float(greeks.theta),
                    'vega': float(greeks.vega),
                    'rho': float(greeks.rho) if greeks.rho else None,
                    'implied_volatility': float(greeks.implied_volatility) if greeks.implied_volatility else None,
                    'time_to_expiration': float(greeks.time_to_expiration) if greeks.time_to_expiration else None,
                    'underlying_price': float(greeks.underlying_price) if greeks.underlying_price else None,
                    'updated_at': datetime.now().isoformat()
                })
            
            # Batch upsert
            self.supabase.table('option_greeks').upsert(data_list).execute()
            
            logger.info(f"💾 Batch saved {len(data_list)} Greeks to database")
            
        except Exception as e:
            logger.error(f"Error in batch save Greeks: {e}")
            raise
    
    def clear_cache(self):
        """Clear in-memory cache"""
        self.cache = {}
        logger.info("🗑️  Cleared Greeks cache")
