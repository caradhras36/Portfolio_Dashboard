#!/usr/bin/env python3
"""
Test CSP Cash Analysis with your portfolio
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from csp_cash_allocator import CSPCashAllocator
import pandas as pd

def test_csp_cash_analysis():
    """Test CSP cash analysis with your portfolio"""
    
    # Load your parsed portfolio
    from merrill_edge_parser import MerrillEdgeParser
    parser = MerrillEdgeParser()
    
    csv_file = r"C:\Users\Ardaz\Downloads\merrill_portfolio\ExportData29092025175309.csv"
    portfolio_df = parser.parse_merrill_csv(csv_file)
    
    print("🧪 Testing CSP Cash Analysis...")
    print(f"📊 Portfolio loaded: {len(portfolio_df)} positions")
    
    # Initialize CSP allocator
    allocator = CSPCashAllocator()
    
    # Identify CSPs
    csps = allocator.identify_csps(portfolio_df)
    print(f"\n💰 Found {len(csps)} CSP positions:")
    
    for csp in csps:
        print(f"  {csp['ticker']} {csp['strike_price']}P: {csp['quantity']} contracts")
    
    if not csps:
        print("❌ No CSP positions found")
        return
    
    # Calculate required cash
    allocations = allocator.calculate_required_cash(csps)
    
    print(f"\n📊 Required Cash Analysis:")
    total_required = sum(alloc.required_cash for alloc in allocations)
    print(f"  Total Required: ${total_required:,.2f}")
    
    for alloc in allocations:
        print(f"  {alloc.ticker} {alloc.strike_price}P: ${alloc.required_cash:,.2f}")
    
    # Test with different cash amounts
    test_cash_amounts = [200000, 300000, 400000, 500000]
    
    for total_cash in test_cash_amounts:
        print(f"\n💵 Testing with ${total_cash:,} available cash:")
        
        # Allocate cash
        test_allocations = allocator.allocate_cash(allocations.copy(), total_cash)
        
        # Analyze risk
        risk_analysis = allocator.analyze_csp_risk(test_allocations)
        
        print(f"  Cash Coverage: {risk_analysis['cash_coverage_pct']:.1f}%")
        print(f"  Allocated: ${risk_analysis['total_allocated_cash']:,.2f}")
        print(f"  Shortfall: ${risk_analysis['total_shortfall']:,.2f}")
        
        # Show individual allocations
        for alloc in test_allocations:
            if alloc.allocated_cash > 0:
                print(f"    {alloc.ticker} {alloc.strike_price}P: {alloc.cash_utilization:.1f}% covered")
    
    # Test with your actual cash balance
    print(f"\n🏦 Using your actual cash balance:")
    your_cash = 222939.46 + 11943.00  # Cash balance + Money accounts
    print(f"  Available Cash: ${your_cash:,.2f}")
    
    final_allocations = allocator.allocate_cash(allocations.copy(), your_cash)
    final_risk = allocator.analyze_csp_risk(final_allocations)
    
    print(f"  Cash Coverage: {final_risk['cash_coverage_pct']:.1f}%")
    print(f"  Allocated: ${final_risk['total_allocated_cash']:,.2f}")
    print(f"  Shortfall: ${final_risk['total_shortfall']:,.2f}")
    
    if final_risk['total_shortfall'] > 0:
        print(f"\n⚠️  WARNING: You need ${final_risk['total_shortfall']:,.2f} more cash to fully secure all CSPs!")
    else:
        print(f"\n✅ All CSPs are fully secured with available cash!")

if __name__ == "__main__":
    test_csp_cash_analysis()
