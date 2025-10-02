#!/usr/bin/env python3
"""
Test script for real Merrill Edge CSV file
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from csv_parser import BrokerCSVParser
import pandas as pd

def test_real_csv():
    """Test with the actual Merrill Edge CSV file"""
    parser = BrokerCSVParser()
    
    csv_file = r"C:\Users\Ardaz\Downloads\merrill_portfolio\ExportData29092025175309.csv"
    
    print("🧪 Testing with real Merrill Edge CSV...")
    print(f"📁 File: {csv_file}")
    
    try:
        # Parse it directly with our specialized parser
        print("\n🔄 Parsing with our specialized Merrill Edge parser...")
        result = parser.parse_csv(csv_file)
        
        print(f"✅ Successfully parsed {len(result)} positions")
        
        if len(result) > 0:
            print("\n📊 Parsed positions:")
            print(result.to_string(index=False))
            
            print(f"\n📈 Position types:")
            print(result['position_type'].value_counts().to_dict())
            
            print(f"\n💰 Total portfolio value: ${(result['quantity'] * result['current_price']).sum():,.2f}")
            
            # Show some statistics
            print(f"\n📊 Statistics:")
            print(f"  • Total positions: {len(result)}")
            print(f"  • Unique tickers: {result['ticker'].nunique()}")
            print(f"  • Stocks: {len(result[result['position_type'] == 'stock'])}")
            print(f"  • Options: {len(result[result['position_type'].isin(['call', 'put'])])}")
            
        else:
            print("❌ No positions were parsed. Check the CSV format.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_csv()
