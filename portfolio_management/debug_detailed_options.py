#!/usr/bin/env python3
"""
Detailed debug of option parsing to understand the patterns
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from merrill_edge_parser import MerrillEdgeParser
import pandas as pd
import re

def debug_detailed_options():
    """Debug option parsing in detail"""
    
    # Load your parsed portfolio
    parser = MerrillEdgeParser()
    csv_file = r"C:\Users\Ardaz\Downloads\merrill_portfolio\ExportData29092025175309.csv"
    portfolio_df = parser.parse_merrill_csv(csv_file)
    
    print("🔍 Detailed Option Parsing Debug...")
    
    # Look at the raw ticker symbols that look like options
    option_like_tickers = portfolio_df[portfolio_df['ticker'].str.len() > 10]
    print(f"\n📊 Tickers longer than 10 chars: {len(option_like_tickers)}")
    
    for _, row in option_like_tickers.iterrows():
        ticker = row['ticker']
        print(f"\n🔍 Analyzing: '{ticker}'")
        print(f"  Position type: {row['position_type']}")
        print(f"  Quantity: {row['quantity']}")
        print(f"  Strike: {row['strike_price']}")
        
        # Manual parsing attempt
        print(f"  Manual analysis:")
        
        # Check for V1725D pattern
        if 'V1725D' in ticker:
            parts = ticker.split('V1725D')
            base = parts[0]
            suffix = parts[1] if len(parts) > 1 else ''
            print(f"    V1725D pattern: base='{base}', suffix='{suffix}'")
            
            # Extract strike from suffix
            numbers = re.findall(r'\d+', suffix)
            if numbers:
                strike = float(numbers[0]) / 1000
                print(f"    Extracted strike: {strike}")
            else:
                print(f"    No numbers found in suffix")
        
        # Check for K2125D pattern
        elif 'K2125D' in ticker:
            parts = ticker.split('K2125D')
            base = parts[0]
            suffix = parts[1] if len(parts) > 1 else ''
            print(f"    K2125D pattern: base='{base}', suffix='{suffix}'")
            
            numbers = re.findall(r'\d+', suffix)
            if numbers:
                strike = float(numbers[0]) / 1000
                print(f"    Extracted strike: {strike}")
            else:
                print(f"    No numbers found in suffix")
        
        # Check for J1725D pattern
        elif 'J1725D' in ticker:
            parts = ticker.split('J1725D')
            base = parts[0]
            suffix = parts[1] if len(parts) > 1 else ''
            print(f"    J1725D pattern: base='{base}', suffix='{suffix}'")
            
            numbers = re.findall(r'\d+', suffix)
            if numbers:
                strike = float(numbers[0]) / 1000
                print(f"    Extracted strike: {strike}")
            else:
                print(f"    No numbers found in suffix")
        
        # Check for C pattern at end
        elif ticker.endswith('C'):
            print(f"    Ends with C (call)")
            numbers = re.findall(r'\d+', ticker)
            if numbers:
                strike = float(numbers[-1]) / 1000
                print(f"    Extracted strike: {strike}")
        
        # Check for other patterns
        else:
            print(f"    No recognized pattern")
            print(f"    All numbers: {re.findall(r'\d+', ticker)}")

if __name__ == "__main__":
    debug_detailed_options()
