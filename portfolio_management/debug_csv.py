#!/usr/bin/env python3
"""
Debug CSV file to understand its structure
"""

import pandas as pd
import csv
import os

def debug_csv_file(file_path):
    """Debug the CSV file to understand its structure"""
    print(f"🔍 Debugging CSV file: {file_path}")
    print(f"📁 File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        print("❌ File not found!")
        return
    
    # Get file size
    file_size = os.path.getsize(file_path)
    print(f"📏 File size: {file_size:,} bytes")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        print(f"\n🔤 Trying encoding: {encoding}")
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Read first few lines
                lines = []
                for i, line in enumerate(f):
                    if i >= 10:  # Read first 10 lines
                        break
                    lines.append(line.strip())
                
                print(f"✅ Successfully read with {encoding}")
                print("📋 First 10 lines:")
                for i, line in enumerate(lines):
                    print(f"  {i+1}: {repr(line)}")
                
                # Try to parse as CSV
                f.seek(0)
                reader = csv.reader(f)
                rows = []
                for i, row in enumerate(reader):
                    if i >= 5:  # Read first 5 rows
                        break
                    rows.append(row)
                
                print(f"\n📊 CSV structure (first 5 rows):")
                for i, row in enumerate(rows):
                    print(f"  Row {i+1}: {row}")
                
                if rows:
                    print(f"\n📋 Column count per row:")
                    for i, row in enumerate(rows):
                        print(f"  Row {i+1}: {len(row)} columns")
                
                break
                
        except Exception as e:
            print(f"❌ Failed with {encoding}: {e}")
            continue
    
    # Try pandas with different parameters
    print(f"\n🐼 Trying pandas with different parameters:")
    
    for sep in [',', ';', '\t', '|']:
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding, nrows=5)
                print(f"✅ Success with sep='{sep}', encoding='{encoding}'")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Shape: {df.shape}")
                break
            except Exception as e:
                continue

if __name__ == "__main__":
    csv_file = r"C:\Users\Ardaz\Downloads\merrill_portfolio\ExportData29092025175309.csv"
    debug_csv_file(csv_file)
