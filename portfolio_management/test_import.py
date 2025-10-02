#!/usr/bin/env python3
"""
Test script for CSV import functionality
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from csv_parser import BrokerCSVParser
import pandas as pd

def test_csv_parser():
    """Test the CSV parser with sample data"""
    parser = BrokerCSVParser()
    
    print("🧪 Testing CSV Parser...")
    
    # Test with sample portfolio
    try:
        result = parser.parse_csv("sample_portfolio.csv")
        print(f"✅ Successfully parsed {len(result)} positions")
        print("\nParsed positions:")
        print(result.to_string(index=False))
        
        # Test broker format detection
        df = pd.read_csv("sample_portfolio.csv")
        format_detected = parser.detect_broker_format(df)
        print(f"\n🔍 Detected broker format: {format_detected}")
        
    except Exception as e:
        print(f"❌ Error testing CSV parser: {e}")

def test_merrill_edge_format():
    """Test with Merrill Edge format"""
    parser = BrokerCSVParser()
    
    # Create sample Merrill Edge data
    merrill_data = {
        'Symbol': ['AAPL', 'TSLA', 'GOOGL'],
        'Quantity': [100, 50, 25],
        'Average Price': [150.00, 200.00, 2800.00],
        'Current Price': [155.00, 210.00, 2850.00]
    }
    
    df = pd.DataFrame(merrill_data)
    df.to_csv("test_merrill.csv", index=False)
    
    try:
        result = parser.parse_csv("test_merrill.csv")
        print(f"\n✅ Successfully parsed Merrill Edge format: {len(result)} positions")
        print(result.to_string(index=False))
        
        # Clean up
        os.remove("test_merrill.csv")
        
    except Exception as e:
        print(f"❌ Error testing Merrill Edge format: {e}")

if __name__ == "__main__":
    test_csv_parser()
    test_merrill_edge_format()
