#!/usr/bin/env python3
"""
Debug dashboard data issues
"""

import os
import sys
import asyncio

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_api import get_portfolio_positions, PortfolioRiskAnalyzer

async def debug_dashboard_data():
    print("🔍 Debugging dashboard data...")
    
    # Get positions
    positions = await get_portfolio_positions()
    print(f"📊 Total positions: {len(positions)}")
    
    # Count by type
    stocks = [p for p in positions if p.position_type == 'stock']
    options = [p for p in positions if p.position_type in ['call', 'put']]
    
    print(f"📈 Stocks: {len(stocks)}")
    print(f"📊 Options: {len(options)}")
    
    # Show first few of each
    print("\n📈 First 3 stocks:")
    for i, stock in enumerate(stocks[:3]):
        print(f"  {i+1}. {stock.ticker}: {stock.quantity} @ ${stock.current_price}")
    
    print("\n📊 First 3 options:")
    for i, option in enumerate(options[:3]):
        print(f"  {i+1}. {stock.ticker} {option.option_type}: {option.quantity} @ ${option.current_price} (strike: ${option.strike_price})")
    
    # Test risk analysis
    risk_analyzer = PortfolioRiskAnalyzer()
    risk_data = risk_analyzer.analyze_portfolio_risk(positions)
    
    print(f"\n📊 Risk Analysis:")
    print(f"  Portfolio Value: ${risk_data['portfolio_value']:,.2f}")
    print(f"  Stock Count: {risk_data['stocks']['count']}")
    print(f"  Options Count: {risk_data['options']['count']}")
    print(f"  Total Delta: {risk_data['greeks']['total_delta']:,.2f}")
    print(f"  CSP Required Cash: ${risk_data['csp_required_cash']:,.2f}")

if __name__ == "__main__":
    asyncio.run(debug_dashboard_data())
