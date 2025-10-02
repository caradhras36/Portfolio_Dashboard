#!/usr/bin/env python3
"""
Debug options parsing to see why CSPs aren't detected
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from merrill_edge_parser import MerrillEdgeParser
import pandas as pd

def debug_options():
    """Debug options parsing"""
    
    # Load your parsed portfolio
    parser = MerrillEdgeParser()
    csv_file = r"C:\Users\Ardaz\Downloads\merrill_portfolio\ExportData29092025175309.csv"
    portfolio_df = parser.parse_merrill_csv(csv_file)
    
    print("🔍 Debugging Options Parsing...")
    print(f"📊 Total positions: {len(portfolio_df)}")
    
    # Check position types
    position_types = portfolio_df['position_type'].value_counts()
    print(f"\n📊 Position types:")
    for ptype, count in position_types.items():
        print(f"  {ptype}: {count}")
    
    # Look for options specifically
    options = portfolio_df[portfolio_df['position_type'].isin(['call', 'put'])]
    print(f"\n🎯 Options found: {len(options)}")
    
    if len(options) > 0:
        print("\n📋 Options details:")
        for _, option in options.iterrows():
            print(f"  {option['ticker']}: {option['position_type']} - Qty: {option['quantity']}, Strike: {option['strike_price']}")
    
    # Look for negative quantities (short positions)
    short_positions = portfolio_df[portfolio_df['quantity'] < 0]
    print(f"\n📉 Short positions: {len(short_positions)}")
    
    if len(short_positions) > 0:
        print("\n📋 Short positions details:")
        for _, pos in short_positions.iterrows():
            print(f"  {pos['ticker']}: {pos['position_type']} - Qty: {pos['quantity']}, Strike: {pos['strike_price']}")
    
    # Look for specific tickers that should be puts
    put_candidates = portfolio_df[portfolio_df['ticker'].str.contains('V1725D|K2125D', na=False)]
    print(f"\n🎯 Put candidates (V1725D/K2125D): {len(put_candidates)}")
    
    if len(put_candidates) > 0:
        print("\n📋 Put candidates:")
        for _, pos in put_candidates.iterrows():
            print(f"  {pos['ticker']}: {pos['position_type']} - Qty: {pos['quantity']}, Strike: {pos['strike_price']}")

if __name__ == "__main__":
    debug_options()
