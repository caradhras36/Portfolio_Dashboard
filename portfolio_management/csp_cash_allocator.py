#!/usr/bin/env python3
"""
CSP Cash Allocation System
Allocates cash balances against Cash Secured Puts for proper risk analysis
"""

import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CSPCashAllocation:
    """CSP cash allocation data structure"""
    ticker: str
    strike_price: float
    quantity: int  # Negative for short puts
    expiration_date: str
    required_cash: float  # Strike * |Quantity| * 100
    allocated_cash: float
    remaining_cash: float
    cash_utilization: float  # Percentage of cash used

class CSPCashAllocator:
    """Allocate cash balances against CSPs"""
    
    def __init__(self):
        self.cash_per_contract = 100  # Standard option contract size
    
    def identify_csps(self, positions: pd.DataFrame) -> List[Dict]:
        """Identify CSP positions from portfolio"""
        csps = []
        
        for _, position in positions.iterrows():
            # CSPs are short puts (negative quantity, put option type)
            if (position['position_type'] == 'put' and 
                position['quantity'] < 0 and 
                position['strike_price'] is not None):
                
                csps.append({
                    'ticker': position['ticker'],
                    'strike_price': position['strike_price'],
                    'quantity': position['quantity'],  # Negative
                    'expiration_date': position['expiration_date'],
                    'current_price': position['current_price']
                })
        
        return csps
    
    def calculate_required_cash(self, csps: List[Dict]) -> List[CSPCashAllocation]:
        """Calculate required cash for each CSP"""
        allocations = []
        
        for csp in csps:
            # Required cash = Strike Price * |Quantity| * 100
            required_cash = csp['strike_price'] * abs(csp['quantity']) * self.cash_per_contract
            
            allocation = CSPCashAllocation(
                ticker=csp['ticker'],
                strike_price=csp['strike_price'],
                quantity=csp['quantity'],
                expiration_date=csp['expiration_date'],
                required_cash=required_cash,
                allocated_cash=0.0,  # Will be calculated
                remaining_cash=0.0,  # Will be calculated
                cash_utilization=0.0  # Will be calculated
            )
            allocations.append(allocation)
        
        return allocations
    
    def allocate_cash(self, allocations: List[CSPCashAllocation], 
                     total_cash: float) -> List[CSPCashAllocation]:
        """Allocate available cash against CSPs"""
        
        # Sort by expiration date (earliest first) and then by strike price
        sorted_allocations = sorted(allocations, 
                                  key=lambda x: (x.expiration_date or '9999-12-31', x.strike_price))
        
        remaining_cash = total_cash
        
        for allocation in sorted_allocations:
            if remaining_cash >= allocation.required_cash:
                # Full allocation possible
                allocation.allocated_cash = allocation.required_cash
                allocation.remaining_cash = 0.0
                allocation.cash_utilization = 100.0
                remaining_cash -= allocation.required_cash
            else:
                # Partial allocation
                allocation.allocated_cash = remaining_cash
                allocation.remaining_cash = allocation.required_cash - remaining_cash
                allocation.cash_utilization = (remaining_cash / allocation.required_cash) * 100
                remaining_cash = 0.0
        
        return allocations
    
    def analyze_csp_risk(self, allocations: List[CSPCashAllocation]) -> Dict:
        """Analyze CSP risk metrics"""
        total_required = sum(alloc.required_cash for alloc in allocations)
        total_allocated = sum(alloc.allocated_cash for alloc in allocations)
        total_shortfall = sum(alloc.remaining_cash for alloc in allocations)
        
        # Calculate risk metrics
        cash_coverage = (total_allocated / total_required * 100) if total_required > 0 else 100
        
        # Group by expiration
        by_expiration = {}
        for alloc in allocations:
            exp = alloc.expiration_date or 'Unknown'
            if exp not in by_expiration:
                by_expiration[exp] = []
            by_expiration[exp].append(alloc)
        
        # Calculate concentration risk
        max_single_csp = max((alloc.required_cash for alloc in allocations), default=0)
        concentration_risk = (max_single_csp / total_required * 100) if total_required > 0 else 0
        
        return {
            'total_required_cash': total_required,
            'total_allocated_cash': total_allocated,
            'total_shortfall': total_shortfall,
            'cash_coverage_pct': cash_coverage,
            'concentration_risk_pct': concentration_risk,
            'csp_count': len(allocations),
            'by_expiration': by_expiration,
            'allocations': allocations
        }
    
    def create_cash_positions(self, allocations: List[CSPCashAllocation], 
                            unallocated_cash: float) -> List[Dict]:
        """Create cash positions for the portfolio"""
        positions = []
        
        # Add allocated cash for each CSP
        for alloc in allocations:
            if alloc.allocated_cash > 0:
                positions.append({
                    'ticker': f"CASH_SECURED_{alloc.ticker}",
                    'position_type': 'cash_secured',
                    'quantity': 1,
                    'entry_price': alloc.allocated_cash,
                    'current_price': alloc.allocated_cash,
                    'expiration_date': alloc.expiration_date,
                    'strike_price': alloc.strike_price,
                    'option_type': 'put',
                    'description': f"Cash secured for {alloc.ticker} {alloc.strike_price}P"
                })
        
        # Add unallocated cash
        if unallocated_cash > 0:
            positions.append({
                'ticker': 'CASH_UNALLOCATED',
                'position_type': 'cash',
                'quantity': 1,
                'entry_price': unallocated_cash,
                'current_price': unallocated_cash,
                'expiration_date': None,
                'strike_price': None,
                'option_type': None,
                'description': 'Unallocated cash'
            })
        
        return positions

# Example usage
if __name__ == "__main__":
    allocator = CSPCashAllocator()
    
    # Example CSP positions
    sample_positions = pd.DataFrame([
        {'ticker': 'SOFI', 'position_type': 'put', 'quantity': -5, 'strike_price': 26.0, 'expiration_date': '2025-01-17'},
        {'ticker': 'HIMS', 'position_type': 'put', 'quantity': -2, 'strike_price': 48.0, 'expiration_date': '2025-01-17'},
        {'ticker': 'HIMS', 'position_type': 'put', 'quantity': -5, 'strike_price': 49.0, 'expiration_date': '2025-01-17'},
    ])
    
    # Identify CSPs
    csps = allocator.identify_csps(sample_positions)
    print(f"Found {len(csps)} CSP positions")
    
    # Calculate required cash
    allocations = allocator.calculate_required_cash(csps)
    
    # Allocate cash (assuming $300,000 available)
    total_cash = 300000
    allocations = allocator.allocate_cash(allocations, total_cash)
    
    # Analyze risk
    risk_analysis = allocator.analyze_csp_risk(allocations)
    
    print(f"\n💰 CSP Cash Analysis:")
    print(f"  Total Required: ${risk_analysis['total_required_cash']:,.2f}")
    print(f"  Total Allocated: ${risk_analysis['total_allocated_cash']:,.2f}")
    print(f"  Cash Coverage: {risk_analysis['cash_coverage_pct']:.1f}%")
    print(f"  CSP Count: {risk_analysis['csp_count']}")
    
    print(f"\n📊 Individual CSPs:")
    for alloc in allocations:
        print(f"  {alloc.ticker} {alloc.strike_price}P: ${alloc.required_cash:,.2f} required, {alloc.cash_utilization:.1f}% covered")
