#!/usr/bin/env python3
"""
Debug CSP values to see what's going wrong
"""

import pandas as pd
import os
import sys
from pprint import pprint

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csv_parser import BrokerCSVParser
from csp_cash_allocator import CSPCashAllocator

def debug_csp_values():
    parser = BrokerCSVParser()
    allocator = CSPCashAllocator()
    
    # Replace with the actual path to your Merrill Edge CSV file
    csv_file = r"C:\Users\Ardaz\Downloads\merrill_portfolio\ExportData29092025175309.csv"
    
    print("🔍 Debugging CSP Values...")
    
    try:
        # Load portfolio from CSV
        portfolio_df = parser.parse_csv(csv_file)
        print(f"📊 Portfolio loaded: {len(portfolio_df)} positions")
        
        # Show all options first
        options_df = portfolio_df[portfolio_df['position_type'].isin(['call', 'put'])]
        print(f"\n🎯 All Options ({len(options_df)}):")
        for _, row in options_df.iterrows():
            print(f"  {row['ticker']}: {row['option_type']} - Qty: {row['quantity']}, Strike: ${row['strike_price']:.2f}")
        
        # Show short positions
        short_positions = portfolio_df[portfolio_df['quantity'] < 0]
        print(f"\n📉 Short Positions ({len(short_positions)}):")
        for _, row in short_positions.iterrows():
            print(f"  {row['ticker']}: {row['position_type']} - Qty: {row['quantity']}, Strike: ${row['strike_price']:.2f}")
        
        # Identify CSPs specifically
        csps = allocator.identify_csps(portfolio_df)
        print(f"\n💰 CSPs Found ({len(csps)}):")
        if len(csps) > 0:
            print(f"  First CSP keys: {list(csps[0].keys())}")
        for csp in csps:
            print(f"  {csp['ticker']}: {csp.get('option_type', 'N/A')} - Qty: {csp['quantity']}, Strike: ${csp['strike_price']:.2f}")
            print(f"    Required Cash: ${abs(csp['quantity']) * csp['strike_price'] * 100:,.2f}")
        
        # Show some specific examples
        print(f"\n🔍 Specific Examples:")
        if len(csps) > 0:
            for csp in csps[:5]:
                required_cash = abs(csp['quantity']) * csp['strike_price'] * 100
                print(f"  {csp['ticker']} {csp['strike_price']:.2f}P x{csp['quantity']}")
                print(f"    Strike: ${csp['strike_price']:.2f}")
                print(f"    Quantity: {csp['quantity']} (abs: {abs(csp['quantity'])})")
                print(f"    Required: {abs(csp['quantity'])} × ${csp['strike_price']:.2f} × 100 = ${required_cash:,.2f}")
                print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_csp_values()
