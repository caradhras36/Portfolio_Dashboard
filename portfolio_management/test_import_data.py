#!/usr/bin/env python3
"""
Test importing data directly to database
"""

import os
import sys
import asyncio
import pandas as pd
from datetime import datetime

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csv_parser import BrokerCSVParser
from portfolio_api import PortfolioPosition, save_portfolio_position, get_portfolio_positions

async def test_import_data():
    print("🧪 Testing data import to database...")
    
    # Parse the CSV
    csv_file = r"C:\Users\Ardaz\Downloads\merrill_portfolio\ExportData29092025175309.csv"
    parser = BrokerCSVParser()
    
    try:
        df = parser.parse_csv(csv_file)
        print(f"📊 Parsed {len(df)} positions from CSV")
        
        # Clear existing data
        print("🗑️ Clearing existing positions...")
        # This would need to be done through the API or direct database call
        
        # Import each position
        imported_count = 0
        for _, row in df.iterrows():
            try:
                position = PortfolioPosition(
                    ticker=str(row['ticker']),
                    position_type=str(row.get('position_type', 'stock')),
                    quantity=int(row['quantity']),
                    entry_price=float(row['entry_price']),
                    current_price=float(row.get('current_price', row['entry_price'])),
                    expiration_date=str(row.get('expiration_date', '')) if pd.notna(row.get('expiration_date')) else None,
                    strike_price=float(row['strike_price']) if pd.notna(row.get('strike_price')) else None,
                    option_type=str(row.get('option_type', '')) if pd.notna(row.get('option_type')) else None,
                    created_at=datetime.now()
                )
                
                # Save to database
                success = await save_portfolio_position(position)
                if success:
                    imported_count += 1
                    
            except Exception as e:
                print(f"❌ Error importing row {_}: {e}")
                continue
        
        print(f"✅ Successfully imported {imported_count} positions")
        
        # Test reading back
        positions = await get_portfolio_positions()
        print(f"📊 Retrieved {len(positions)} positions from database")
        
        # Count by type
        stocks = [p for p in positions if p.position_type == 'stock']
        options = [p for p in positions if p.position_type in ['call', 'put']]
        
        print(f"📈 Stocks: {len(stocks)}")
        print(f"📊 Options: {len(options)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_import_data())
