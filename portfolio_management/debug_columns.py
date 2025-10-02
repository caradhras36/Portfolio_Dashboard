#!/usr/bin/env python3
"""
Debug the actual column values in the CSV
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

def debug_columns():
    """Debug the actual column values"""
    csv_file = r"C:\Users\Ardaz\Downloads\merrill_portfolio\ExportData29092025175309.csv"
    
    print("🔍 Debugging column values...")
    
    # Read the file line by line to find the data section
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find where the actual data starts
    data_start_line = None
    for i, line in enumerate(lines):
        if 'Symbol' in line and 'Quantity' in line:
            data_start_line = i
            break
    
    if data_start_line is None:
        print("❌ Could not find data section")
        return
    
    print(f"📊 Data starts at line {data_start_line + 1}")
    
    # Extract the data section
    data_lines = lines[data_start_line:]
    
    # Create a temporary CSV with just the data
    temp_csv = '\n'.join(data_lines)
    
    # Parse with pandas
    df = pd.read_csv(pd.io.common.StringIO(temp_csv))
    
    print(f"📋 Columns: {list(df.columns)}")
    print(f"📊 Shape: {df.shape}")
    
    # Show first few rows with actual values
    print("\n📊 First 5 rows with actual values:")
    for i in range(min(5, len(df))):
        print(f"\nRow {i+1}:")
        for col in df.columns:
            value = df.iloc[i][col]
            print(f"  {col}: {repr(value)}")
    
    # Check for different price columns
    print("\n💰 Price column analysis:")
    for col in df.columns:
        if 'price' in col.lower() or 'cost' in col.lower() or 'value' in col.lower():
            print(f"\n{col}:")
            sample_values = df[col].head(10).tolist()
            print(f"  Sample values: {sample_values}")

if __name__ == "__main__":
    debug_columns()
