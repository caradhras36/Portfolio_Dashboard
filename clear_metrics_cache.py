#!/usr/bin/env python3
"""
Clear the stock_metrics_cache table to force fresh calculations
"""

import sys
sys.path.append('.')
sys.path.append('./shared')

from supabase import create_client, Client
from shared.config import SUPABASE_URL, SUPABASE_KEY

def clear_cache():
    """Delete all cached metrics"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Delete all rows
        response = supabase.table('stock_metrics_cache')\
            .delete()\
            .neq('ticker', 'IMPOSSIBLE_VALUE_TO_MATCH_ALL')\
            .execute()
        
        print(f"Cache cleared! Deleted all cached metrics.")
        print("Now trigger a fresh calculation with force_recalc=true")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clear_cache()
