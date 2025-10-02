"""
Comprehensive test suite for Portfolio API
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import sys
import os

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'portfolio_management'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from portfolio_api import (
    PortfolioPosition, 
    PortfolioRiskAnalyzer, 
    get_portfolio_positions,
    save_portfolio_position,
    risk_analyzer
)
from options_analyzer import OptionsAnalyzer, MarketAwareOptionsAnalyzer
from csv_parser import BrokerCSVParser
from csp_cash_allocator import CSPCashAllocator

class TestPortfolioPosition:
    """Test PortfolioPosition dataclass"""
    
    def test_portfolio_position_creation(self):
        """Test creating a portfolio position"""
        position = PortfolioPosition(
            ticker="AAPL",
            position_type="stock",
            quantity=100,
            entry_price=150.00,
            current_price=175.50,
            created_at=datetime.now()
        )
        
        assert position.ticker == "AAPL"
        assert position.position_type == "stock"
        assert position.quantity == 100
        assert position.entry_price == 150.00
        assert position.current_price == 175.50
        assert position.expiration_date is None
        assert position.strike_price is None
        assert position.option_type is None
    
    def test_option_position_creation(self):
        """Test creating an option position"""
        position = PortfolioPosition(
            ticker="GOOGL",
            position_type="call",
            quantity=10,
            entry_price=5.50,
            current_price=6.20,
            expiration_date="2025-01-17",
            strike_price=140.00,
            option_type="call",
            delta=0.65,
            gamma=0.02,
            theta=-0.15,
            vega=0.25,
            implied_volatility=0.28,
            created_at=datetime.now()
        )
        
        assert position.ticker == "GOOGL"
        assert position.position_type == "call"
        assert position.strike_price == 140.00
        assert position.delta == 0.65

class TestOptionsAnalyzer:
    """Test OptionsAnalyzer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = OptionsAnalyzer()
    
    def test_calculate_greeks_call_option(self):
        """Test Greeks calculation for call option"""
        option_data = {
            'ticker': 'AAPL',
            'strike_price': 175.0,
            'current_price': 175.50,
            'time_to_expiration': 30,
            'implied_volatility': 0.25,
            'contract_type': 'call'
        }
        
        greeks = self.analyzer.calculate_greeks(option_data)
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 0 <= greeks['delta'] <= 1  # Call delta should be between 0 and 1
        assert greeks['gamma'] > 0  # Gamma should be positive
        assert greeks['vega'] > 0  # Vega should be positive
    
    def test_calculate_greeks_put_option(self):
        """Test Greeks calculation for put option"""
        option_data = {
            'ticker': 'AAPL',
            'strike_price': 175.0,
            'current_price': 175.50,
            'time_to_expiration': 30,
            'implied_volatility': 0.25,
            'contract_type': 'put'
        }
        
        greeks = self.analyzer.calculate_greeks(option_data)
        
        assert -1 <= greeks['delta'] <= 0  # Put delta should be between -1 and 0
        assert greeks['gamma'] > 0  # Gamma should be positive
        assert greeks['vega'] > 0  # Vega should be positive
    
    def test_invalid_option_data(self):
        """Test handling of invalid option data"""
        invalid_data = {
            'ticker': 'INVALID',
            'strike_price': 0,  # Invalid strike
            'current_price': -1,  # Invalid price
            'time_to_expiration': 0,  # Invalid time
            'implied_volatility': 0,
            'contract_type': 'call'
        }
        
        greeks = self.analyzer.calculate_greeks(invalid_data)
        
        # Should return zero Greeks for invalid data
        assert greeks['delta'] == 0
        assert greeks['gamma'] == 0
        assert greeks['theta'] == 0
        assert greeks['vega'] == 0

class TestMarketAwareOptionsAnalyzer:
    """Test MarketAwareOptionsAnalyzer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = MarketAwareOptionsAnalyzer()
    
    @pytest.mark.asyncio
    async def test_get_current_price(self):
        """Test getting current price for ticker"""
        price = await self.analyzer.get_current_price("AAPL")
        assert isinstance(price, float)
        assert price > 0
    
    @pytest.mark.asyncio
    async def test_fetch_options_data(self):
        """Test fetching options data"""
        options_data = await self.analyzer.fetch_options_data("AAPL")
        
        assert isinstance(options_data, list)
        assert len(options_data) > 0
        
        for option in options_data:
            assert 'ticker' in option
            assert 'strike_price' in option
            assert 'contract_type' in option
            assert option['contract_type'] in ['call', 'put']

class TestPortfolioRiskAnalyzer:
    """Test PortfolioRiskAnalyzer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = PortfolioRiskAnalyzer()
        self.sample_positions = [
            PortfolioPosition(
                ticker="AAPL",
                position_type="stock",
                quantity=100,
                entry_price=150.00,
                current_price=175.50,
                created_at=datetime.now()
            ),
            PortfolioPosition(
                ticker="GOOGL",
                position_type="call",
                quantity=10,
                entry_price=5.50,
                current_price=6.20,
                expiration_date="2025-01-17",
                strike_price=140.00,
                option_type="call",
                delta=0.65,
                gamma=0.02,
                theta=-0.15,
                vega=0.25,
                created_at=datetime.now()
            )
        ]
    
    def test_calculate_portfolio_greeks(self):
        """Test portfolio Greeks calculation"""
        greeks = self.analyzer.calculate_portfolio_greeks(self.sample_positions)
        
        assert 'total_delta' in greeks
        assert 'total_gamma' in greeks
        assert 'total_theta' in greeks
        assert 'total_vega' in greeks
        
        # Stock should contribute 100 to delta (100 shares * 1.0 delta)
        # Option should contribute 6.5 to delta (10 contracts * 0.65 delta)
        expected_delta = 100 + 6.5
        assert abs(greeks['total_delta'] - expected_delta) < 0.01
    
    def test_calculate_portfolio_value(self):
        """Test portfolio value calculation"""
        value = self.analyzer.calculate_portfolio_value(self.sample_positions)
        
        expected_value = (100 * 175.50) + (10 * 6.20)  # 17550 + 62 = 17612
        assert abs(value - expected_value) < 0.01
    
    def test_calculate_position_weights(self):
        """Test position weight calculation"""
        weights = self.analyzer.calculate_position_weights(self.sample_positions)
        
        assert "AAPL" in weights
        assert "GOOGL" in weights
        
        # Weights should sum to 100%
        total_weight = sum(weights.values())
        assert abs(total_weight - 100.0) < 0.01
    
    def test_calculate_concentration_risk(self):
        """Test concentration risk calculation"""
        concentration = self.analyzer.calculate_concentration_risk(self.sample_positions)
        
        assert 'top_5_concentration' in concentration
        assert 'hhi' in concentration
        assert 'max_position_weight' in concentration
        assert 'position_count' in concentration
        
        assert concentration['position_count'] == 2
        assert concentration['top_5_concentration'] <= 100.0
        assert concentration['max_position_weight'] > 0
    
    def test_analyze_portfolio_risk(self):
        """Test comprehensive portfolio risk analysis"""
        risk_analysis = self.analyzer.analyze_portfolio_risk(self.sample_positions)
        
        assert 'portfolio_value' in risk_analysis
        assert 'total_pnl' in risk_analysis
        assert 'total_pnl_pct' in risk_analysis
        assert 'greeks' in risk_analysis
        assert 'stocks' in risk_analysis
        assert 'options' in risk_analysis
        assert 'concentration' in risk_analysis
        
        assert risk_analysis['stocks']['count'] == 1
        assert risk_analysis['options']['count'] == 1
        assert risk_analysis['position_count'] == 2

class TestCSVParser:
    """Test CSV parser functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.parser = BrokerCSVParser()
    
    def test_detect_broker_format(self):
        """Test broker format detection"""
        import pandas as pd
        
        # Test Merrill Edge format
        merrill_df = pd.DataFrame({
            'Symbol': ['AAPL', 'TSLA'],
            'Quantity': [100, 50],
            'Average Price': [150.00, 200.00],
            'Current Price': [175.50, 245.30]
        })
        
        format_type = self.parser.detect_broker_format(merrill_df)
        assert format_type == 'merrill_edge_stocks'
    
    def test_safe_float_conversion(self):
        """Test safe float conversion"""
        assert self.parser._safe_float("150.50") == 150.50
        assert self.parser._safe_float("$150.50") == 150.50
        assert self.parser._safe_float("1,500.50") == 1500.50
        assert self.parser._safe_float("") == 0.0
        assert self.parser._safe_float(None) == 0.0
        assert self.parser._safe_float("invalid") == 0.0

class TestCSPCashAllocator:
    """Test CSP cash allocator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.allocator = CSPCashAllocator()
        import pandas as pd
        
        self.sample_positions = pd.DataFrame([
            {
                'ticker': 'SOFI',
                'position_type': 'put',
                'quantity': -5,
                'strike_price': 26.0,
                'expiration_date': '2025-01-17'
            },
            {
                'ticker': 'HIMS',
                'position_type': 'put',
                'quantity': -2,
                'strike_price': 48.0,
                'expiration_date': '2025-01-17'
            }
        ])
    
    def test_identify_csps(self):
        """Test CSP identification"""
        csps = self.allocator.identify_csps(self.sample_positions)
        
        assert len(csps) == 2
        assert csps[0]['ticker'] == 'SOFI'
        assert csps[1]['ticker'] == 'HIMS'
    
    def test_calculate_required_cash(self):
        """Test required cash calculation"""
        csps = self.allocator.identify_csps(self.sample_positions)
        allocations = self.allocator.calculate_required_cash(csps)
        
        # SOFI: 5 contracts * $26 * 100 = $13,000
        # HIMS: 2 contracts * $48 * 100 = $9,600
        sofi_required = 5 * 26.0 * 100
        hims_required = 2 * 48.0 * 100
        
        assert allocations[0].required_cash == sofi_required
        assert allocations[1].required_cash == hims_required
    
    def test_allocate_cash(self):
        """Test cash allocation"""
        csps = self.allocator.identify_csps(self.sample_positions)
        allocations = self.allocator.calculate_required_cash(csps)
        
        total_cash = 50000  # $50,000 available
        allocations = self.allocator.allocate_cash(allocations, total_cash)
        
        # Should fully allocate both CSPs
        assert allocations[0].cash_utilization == 100.0
        assert allocations[1].cash_utilization == 100.0
    
    def test_analyze_csp_risk(self):
        """Test CSP risk analysis"""
        csps = self.allocator.identify_csps(self.sample_positions)
        allocations = self.allocator.calculate_required_cash(csps)
        allocations = self.allocator.allocate_cash(allocations, 50000)
        
        risk_analysis = self.allocator.analyze_csp_risk(allocations)
        
        assert 'total_required_cash' in risk_analysis
        assert 'total_allocated_cash' in risk_analysis
        assert 'cash_coverage_pct' in risk_analysis
        assert 'csp_count' in risk_analysis
        
        assert risk_analysis['csp_count'] == 2
        assert risk_analysis['cash_coverage_pct'] == 100.0

class TestDatabaseFunctions:
    """Test database helper functions"""
    
    @pytest.mark.asyncio
    @patch('portfolio_api.supabase', None)
    async def test_get_portfolio_positions_no_db(self):
        """Test getting positions when database is not available"""
        positions = await get_portfolio_positions()
        
        # Should return sample positions when no database
        assert isinstance(positions, list)
        assert len(positions) > 0
    
    @pytest.mark.asyncio
    async def test_save_portfolio_position(self):
        """Test saving a portfolio position"""
        position = PortfolioPosition(
            ticker="TEST",
            position_type="stock",
            quantity=10,
            entry_price=100.00,
            current_price=105.00,
            created_at=datetime.now()
        )
        
        # This will use in-memory storage if no database
        result = await save_portfolio_position(position)
        assert result is True

# Integration tests
class TestIntegration:
    """Integration tests for the portfolio system"""
    
    @pytest.mark.asyncio
    async def test_full_portfolio_analysis(self):
        """Test full portfolio analysis workflow"""
        # Create sample positions
        positions = [
            PortfolioPosition(
                ticker="AAPL",
                position_type="stock",
                quantity=100,
                entry_price=150.00,
                current_price=175.50,
                created_at=datetime.now()
            ),
            PortfolioPosition(
                ticker="GOOGL",
                position_type="call",
                quantity=10,
                entry_price=5.50,
                current_price=6.20,
                expiration_date="2025-01-17",
                strike_price=140.00,
                option_type="call",
                delta=0.65,
                gamma=0.02,
                theta=-0.15,
                vega=0.25,
                created_at=datetime.now()
            )
        ]
        
        # Analyze risk
        risk_analysis = risk_analyzer.analyze_portfolio_risk(positions)
        
        # Verify results
        assert risk_analysis['portfolio_value'] > 0
        assert risk_analysis['position_count'] == 2
        assert 'greeks' in risk_analysis
        assert risk_analysis['stocks']['count'] == 1
        assert risk_analysis['options']['count'] == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
