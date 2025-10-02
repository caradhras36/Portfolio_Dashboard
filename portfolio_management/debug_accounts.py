#!/usr/bin/env python3
"""
Debug account structure and money accounts in the CSV
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

def debug_accounts():
    """Debug account structure in the CSV"""
    
    csv_file = r"C:\Users\Ardaz\Downloads\merrill_portfolio\ExportData29092025175309.csv"
    
    print("🔍 Debugging Account Structure...")
    
    # Read the file line by line to understand structure
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 Total lines in CSV: {len(lines)}")
    
    # Look for account information
    print("\n📋 First 20 lines:")
    for i, line in enumerate(lines[:20]):
        print(f"  {i+1:2d}: {line.strip()}")
    
    # Look for account names
    print("\n🏦 Looking for account names:")
    for i, line in enumerate(lines):
        if 'Main' in line or 'Duru' in line or 'Bonds' in line:
            print(f"  Line {i+1}: {line.strip()}")
    
    # Look for money accounts
    print("\n💰 Looking for money accounts:")
    for i, line in enumerate(lines):
        if 'money' in line.lower() or 'Money' in line:
            print(f"  Line {i+1}: {line.strip()}")
    
    # Look for cash-related information
    print("\n💵 Looking for cash information:")
    for i, line in enumerate(lines):
        if 'cash' in line.lower() or 'Cash' in line:
            print(f"  Line {i+1}: {line.strip()}")

if __name__ == "__main__":
    debug_accounts()
