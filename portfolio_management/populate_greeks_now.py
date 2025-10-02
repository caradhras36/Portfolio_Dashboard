#!/usr/bin/env python3
"""
Immediately populate Greeks in database from Polygon.io
Run this ONCE to fill the database, then risk monitoring will be fast
"""

import asyncio
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "shared"))
sys.path.append(os.path.join(project_root, "portfolio_management"))

async def main():
    print("=" * 70)
    print("POPULATING GREEKS IN DATABASE FROM POLYGON.IO")
    print("=" * 70)
    print("\nThis will:")
    print("1. Fetch all options positions from database")
    print("2. Get fresh Greeks from Polygon.io for each")
    print("3. Save Greeks back to database")
    print("4. After this, risk monitoring will be FAST (<50ms)")
    print("\n" + "=" * 70)
    
    try:
        from portfolio_api import supabase, calculate_greeks_batch, save_greeks_to_greeks_table
        from portfolio_api import PortfolioPosition
        from datetime import datetime
        
        # Fetch all positions
        response = supabase.table('portfolio_positions').select('*').execute()
        
        if not response.data:
            print("\n[ERROR] No positions found in database")
            return
        
        # Extract options
        option_positions = []
        for row in response.data:
            if row['position_type'] in ['call', 'put']:
                position = PortfolioPosition(
                    ticker=row['ticker'],
                    position_type=row['position_type'],
                    quantity=row['quantity'],
                    entry_price=float(row['entry_price']),
                    current_price=float(row['current_price']),
                    expiration_date=row.get('expiration_date'),
                    strike_price=float(row['strike_price']) if row.get('strike_price') else None,
                    option_type=row.get('option_type'),
                    delta=None,  # Will fetch from Polygon.io
                    gamma=None,
                    theta=None,
                    vega=None,
                    implied_volatility=None,
                    time_to_expiration=None,
                    created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else None
                )
                option_positions.append(position)
        
        print(f"\n[INFO] Found {len(option_positions)} options to populate\n")
        
        if not option_positions:
            print("[INFO] No options found - nothing to do!")
            return
        
        # Fetch Greeks from Polygon.io
        print(f"[STEP 1/2] Fetching Greeks from Polygon.io...")
        print("           This may take 10-30 seconds depending on position count...")
        
        import time
        start_time = time.time()
        
        positions_with_greeks = await calculate_greeks_batch(option_positions)
        
        fetch_time = time.time() - start_time
        print(f"\n[SUCCESS] Fetched Greeks in {fetch_time:.2f} seconds")
        print(f"          ({len(positions_with_greeks)}/{len(option_positions)} positions)")
        
        # Save to NEW option_greeks table (not portfolio_positions)
        print(f"\n[STEP 2/2] Saving Greeks to option_greeks table...")
        
        await save_greeks_to_greeks_table(positions_with_greeks)
        
        print(f"\n[SUCCESS] Greeks saved to database!")
        
        # Verify
        print("\n" + "=" * 70)
        print("VERIFICATION:")
        print("=" * 70)
        
        response = supabase.table('option_greeks').select('*').execute()
        verified = len(response.data) if response.data else 0
        
        print(f"[OK] {verified} Greeks records in option_greeks table")
        
        if verified == len(option_positions):
            print("\n" + "SUCCESS! " * 10)
            print("All Greeks are now in the database!")
            print("Risk monitoring will now load in <50ms!")
            print("SUCCESS! " * 10)
        else:
            print(f"\n[WARNING] Only {verified}/{len(option_positions)} options have Greeks")
            print("          Some may have failed - check logs above")
        
        print("\n" + "=" * 70)
        print("DONE! You can now use the dashboard with fast risk monitoring.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
