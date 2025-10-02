"""
Pytest configuration and fixtures for portfolio dashboard tests
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'portfolio_management'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from portfolio_api import PortfolioPosition

@pytest.fixture
def sample_stock_position():
    """Sample stock position for testing"""
    return PortfolioPosition(
        ticker="AAPL",
        position_type="stock",
        quantity=100,
        entry_price=150.00,
        current_price=175.50,
        created_at=datetime.now()
    )

@pytest.fixture
def sample_call_option():
    """Sample call option position for testing"""
    return PortfolioPosition(
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

@pytest.fixture
def sample_put_option():
    """Sample put option position for testing"""
    return PortfolioPosition(
        ticker="TSLA",
        position_type="put",
        quantity=5,
        entry_price=8.75,
        current_price=7.50,
        expiration_date="2025-01-17",
        strike_price=250.00,
        option_type="put",
        delta=-0.35,
        gamma=0.015,
        theta=-0.12,
        vega=0.18,
        implied_volatility=0.32,
        created_at=datetime.now()
    )

@pytest.fixture
def sample_portfolio():
    """Sample portfolio with multiple positions"""
    return [
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
            implied_volatility=0.28,
            created_at=datetime.now()
        ),
        PortfolioPosition(
            ticker="TSLA",
            position_type="stock",
            quantity=50,
            entry_price=200.00,
            current_price=245.30,
            created_at=datetime.now()
        )
    ]

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_supabase():
    """Mock Supabase client for testing"""
    from unittest.mock import Mock
    
    mock_client = Mock()
    mock_table = Mock()
    mock_client.table.return_value = mock_table
    mock_table.select.return_value.execute.return_value.data = []
    mock_table.insert.return_value.execute.return_value = {}
    mock_table.delete.return_value.eq.return_value.execute.return_value = {}
    
    return mock_client

@pytest.fixture
def mock_options_analyzer():
    """Mock options analyzer for testing"""
    from unittest.mock import Mock
    
    analyzer = Mock()
    analyzer.calculate_greeks.return_value = {
        'delta': 0.5,
        'gamma': 0.02,
        'theta': -0.1,
        'vega': 0.2,
        'implied_volatility': 0.25
    }
    
    return analyzer

@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return """Symbol,Quantity,Average Price,Current Price
AAPL,100,150.00,175.50
TSLA,50,200.00,245.30
GOOGL,25,2800.00,2850.00"""

@pytest.fixture
def sample_options_csv_data():
    """Sample options CSV data for testing"""
    return """Symbol,Quantity,Call/Put,Strike Price,Expiration Date,Average Price,Current Price
AAPL,10,Call,175.00,2025-01-17,5.50,6.20
GOOGL,5,Put,140.00,2025-01-17,3.25,2.80"""

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test"""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    
    yield
    
    # Clean up after test
    if 'TESTING' in os.environ:
        del os.environ['TESTING']
