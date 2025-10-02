"""Configuration settings for the Options Trade Search system."""

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas_market_calendars as mcal
from dotenv import load_dotenv
from typing import Dict, List

# Load environment variables
load_dotenv()

# Logging Configuration
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "options_trade_search.log"

# API Keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ZAPIER_WEBHOOK_URL = os.getenv("ZAPIER_WEBHOOK_URL")

# Market Settings
MARKET_CLOSE_TIME = "20:00"  # UTC time for market close

# Market Calendar
NYSE = mcal.get_calendar('NYSE')

# Options Trading Configuration
DEFAULT_STOCK_LIST = [
    # High IV stocks from your existing list
    "TSLA", "NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN",
    "AMD", "PLTR", "CRM", "SNOW", "DDOG", "NET", "U", "CRWD",
    "COIN", "PYPL", "SOFI", "HOOD", "MSTR", "RIVN", "LCID",
    "NFLX", "DIS", "UBER", "SHOP", "ENPH", "PLUG", "PANW"
]

# Options Analysis Parameters
MIN_DAYS_TO_EXPIRATION = 30
MAX_DAYS_TO_EXPIRATION = 60
MIN_IV_THRESHOLD = 0.20  # 20% minimum IV
MIN_OPEN_INTEREST = 100  # Minimum open interest for liquidity

# Technical Indicator Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Risk Management
MAX_POSITION_SIZE = 0.02  # 2% of portfolio per position
MAX_DAILY_LOSS = 0.05     # 5% max daily loss
STOP_LOSS_PCT = 0.20      # 20% stop loss

# Scoring Weights (for non-ML approach)
SCORING_WEIGHTS = {
    'ivr': 0.10,           # IV Rank importance (reduced)
    'greeks': 0.15,        # Greeks (Delta, Theta, Vega) (reduced)
    'technical': 0.15,     # Technical indicators (reduced)
    'liquidity': 0.10,     # Open interest and volume (reduced)
    'resistance': 0.10,    # Support/resistance levels (reduced)
    'probability': 0.25,   # Probability of profit (NEW - highest weight)
    'roi': 0.15            # Return on Investment (only after probability threshold)
}

# Data Storage
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

def is_trading_day(date: datetime = None) -> bool:
    """Check if a given date is a trading day."""
    if date is None:
        date = datetime.now()
    return NYSE.valid_days(start_date=date.date(), end_date=date.date()).size > 0

def get_last_trading_day(date: datetime = None) -> datetime:
    """Get the last completed trading day."""
    if date is None:
        date = datetime.now(timezone.utc)
    
    # Get the last 10 trading days
    start_date = date - timedelta(days=10)
    trading_days = NYSE.valid_days(start_date=start_date.date(), end_date=date.date())
    
    if trading_days.size > 0:
        # Get the most recent completed trading day
        last_day = trading_days[-1].to_pydatetime().replace(tzinfo=timezone.utc)
        
        # If it's today and market is still open, use previous day
        if last_day.date() == date.date():
            market_close = datetime.combine(date.date(), datetime.strptime(MARKET_CLOSE_TIME, "%H:%M").time(), tzinfo=timezone.utc)
            if date < market_close:
                if trading_days.size > 1:
                    return trading_days[-2].to_pydatetime().replace(tzinfo=timezone.utc)
                else:
                    raise ValueError("No previous trading day found")
        return last_day
    
    raise ValueError("No trading days found in the last 10 days")

def get_polygon_api_key() -> str:
    """Get the Polygon API key from environment variables."""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY not found in environment variables")
    return api_key

def get_supabase_credentials() -> Dict[str, str]:
    """Get Supabase credentials from environment variables."""
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    return {'url': url, 'key': key}

def get_zapier_webhook_url() -> str:
    """Get the Zapier webhook URL from environment variables."""
    url = os.getenv('ZAPIER_WEBHOOK_URL')
    if not url:
        raise ValueError("ZAPIER_WEBHOOK_URL not found in environment variables")
    return url
