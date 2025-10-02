#!/usr/bin/env python3
"""
Quick diagnostic script to check if Greeks are in the database
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "shared"))

from shared.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

def check_greeks():
    """Check if Greeks exist in database"""
    print("=" * 60)
    print("Checking Greeks in Database")
    print("=" * 60)
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Fetch all positions
        response = supabase.table('portfolio_positions').select('*').execute()
        
        if not response.data:
            print("❌ No positions found in database")
            return
        
        total_positions = len(response.data)
        options_count = 0
        options_with_greeks = 0
        options_without_greeks = 0
        
        print(f"\nTotal positions: {total_positions}")
        print("\n" + "=" * 60)
        print("OPTIONS GREEK STATUS:")
        print("=" * 60)
        
        for row in response.data:
            if row['position_type'] in ['call', 'put']:
                options_count += 1
                has_greeks = row.get('delta') is not None
                
                if has_greeks:
                    options_with_greeks += 1
                    print(f"[OK] {row['ticker']} {row['position_type']} ${row.get('strike_price')}: "
                          f"Delta={row.get('delta'):.3f}, Theta={row.get('theta'):.4f}")
                else:
                    options_without_greeks += 1
                    print(f"[MISSING] {row['ticker']} {row['position_type']} ${row.get('strike_price')}: "
                          f"NO GREEKS (delta={row.get('delta')})")
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        print(f"Total options: {options_count}")
        print(f"[OK] With Greeks: {options_with_greeks} ({options_with_greeks/options_count*100:.1f}%)" if options_count > 0 else "No options")
        print(f"[MISSING] Without Greeks: {options_without_greeks} ({options_without_greeks/options_count*100:.1f}%)" if options_count > 0 else "")
        
        if options_without_greeks > 0:
            print("\n" + "WARNING! " * 8)
            print("*** GREEKS ARE MISSING IN DATABASE! ***")
            print("*** This causes SLOW loading (fetches from Polygon.io every time) ***")
            print("WARNING! " * 8)
            print("\nTO FIX:")
            print("1. Load the dashboard once (it will fetch and save Greeks)")
            print("2. Or run: python portfolio_management/update_greeks.py")
            print("\nAfter that, risk monitoring will be FAST (<50ms)")
        else:
            print("\n[SUCCESS] All Greeks are in database - Risk monitoring should be FAST!")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_greeks()
