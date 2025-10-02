#!/usr/bin/env python3
"""Force clear ALL cached metrics using SQL DELETE"""

import sys
sys.path.append('.')
sys.path.append('./shared')

from supabase import create_client, Client
from shared.config import SUPABASE_URL, SUPABASE_KEY

def force_clear():
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get count before
        before = supabase.table('stock_metrics_cache').select('ticker', count='exact').execute()
        print(f"Records before: {before.count}")
        
        # Delete ALL using a simple query
        result = supabase.table('stock_metrics_cache').delete().gte('calculation_date', '2000-01-01').execute()
        
        # Check after
        after = supabase.table('stock_metrics_cache').select('ticker', count='exact').execute()
        print(f"Records after: {after.count}")
        print("\nCache completely cleared!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    force_clear()
