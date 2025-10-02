#!/usr/bin/env python3
"""
Stock Metrics Cache - Stores pre-calculated beta, volatility, liquidity
Avoids expensive recalculation every time
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, date
import os

try:
    import sys
    sys.path.append('.')
    sys.path.append('./shared')
    from supabase import create_client, Client
    from shared.config import SUPABASE_URL, SUPABASE_KEY
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
    if not supabase:
        print("WARNING: metrics_cache.py - No Supabase connection (check shared/config.py)")
except Exception as e:
    print(f"ERROR: metrics_cache.py - Failed to initialize Supabase: {e}")
    supabase = None

logger = logging.getLogger(__name__)

class MetricsCache:
    """Manages caching of calculated stock metrics"""
    
    def __init__(self, supabase_client=None):
        self.supabase = supabase_client or supabase
    
    async def get_metrics_batch(self, tickers: list[str]) -> Dict[str, Tuple[float, float, float]]:
        """Get cached metrics for multiple tickers
        
        Returns: Dict[ticker -> (beta, volatility, liquidity)]
        Only returns metrics calculated TODAY
        """
        try:
            if not self.supabase or not tickers:
                return {}
            
            today = date.today()
            
            response = self.supabase.table('stock_metrics_cache')\
                .select('*')\
                .in_('ticker', tickers)\
                .eq('calculation_date', today.isoformat())\
                .execute()
            
            metrics = {}
            if response.data:
                for row in response.data:
                    metrics[row['ticker']] = (
                        float(row['beta']),
                        float(row['volatility']),
                        float(row['liquidity_score'])
                    )
            
            logger.info(f"✅ Loaded cached metrics for {len(metrics)}/{len(tickers)} stocks from today")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching cached metrics: {e}")
            return {}
    
    async def save_metrics_batch(self, metrics: Dict[str, Tuple[float, float, float]]) -> bool:
        """Save calculated metrics for multiple tickers
        
        Args:
            metrics: Dict[ticker -> (beta, volatility, liquidity)]
        """
        try:
            if not self.supabase or not metrics:
                return False
            
            today = date.today()
            
            records = []
            for ticker, (beta, vol, liq) in metrics.items():
                records.append({
                    'ticker': ticker,
                    'beta': beta,
                    'volatility': vol,
                    'liquidity_score': liq,
                    'calculation_date': today.isoformat()
                })
            
            # Upsert (update if exists, insert if not)
            response = self.supabase.table('stock_metrics_cache')\
                .upsert(records, on_conflict='ticker')\
                .execute()
            
            logger.info(f"✅ Saved metrics for {len(records)} stocks to cache")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metrics cache: {e}")
            return False

# Global instance
_metrics_cache = None

def get_metrics_cache():
    """Get or create metrics cache instance"""
    global _metrics_cache
    if _metrics_cache is None:
        _metrics_cache = MetricsCache()
    return _metrics_cache
