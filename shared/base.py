"""Base classes and shared utilities for the Options Trade Search system."""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Union, Any
from polygon import RESTClient
from supabase import create_client
import pandas as pd
import numpy as np
from config import get_polygon_api_key, get_supabase_credentials

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptionsDataFetcher:
    """Base class for fetching options and market data from Polygon.io and Supabase."""
    
    def __init__(self):
        """Initialize the data fetcher with API clients."""
        try:
            self.polygon_client = RESTClient(get_polygon_api_key())
            supabase_creds = get_supabase_credentials()
            self.supabase = create_client(supabase_creds['url'], supabase_creds['key'])
            logger.info("Successfully initialized OptionsDataFetcher with Polygon.io and Supabase clients")
        except Exception as e:
            logger.error(f"Failed to initialize OptionsDataFetcher: {e}")
            raise
    
    def _retry_with_backoff(self, func, max_retries: int = 5, base_delay: float = 0.5) -> Optional[Any]:
        """Retry a function with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if "Rate limited" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (1.5 ** attempt)
                    logger.warning(f"Rate limited. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Error after {attempt + 1} attempts: {str(e)}")
                    raise
        return None
    
    def get_stock_price(self, ticker: str, date: datetime = None) -> Optional[float]:
        """Get the current or historical stock price with market-aware data fetching."""
        try:
            from config import is_trading_day, get_last_trading_day
            
            # Determine if market is open (ET timezone)
            from pytz import timezone as pytz_timezone
            et_tz = pytz_timezone('US/Eastern')
            now_et = datetime.now(et_tz)
            market_is_open = is_trading_day(now_et) and 9.5 <= now_et.hour + now_et.minute/60 <= 16
            
            if market_is_open:
                # Market is open - use 15-minute delayed data from Polygon
                logger.info(f"Market is open - using 15-minute delayed data for {ticker}")
                try:
                    # Use Polygon.io with 15-minute delay
                    aggs = list(self.polygon_client.list_aggs(
                        ticker=ticker,
                        multiplier=1,
                        timespan="minute",
                        from_=now_et.strftime("%Y-%m-%d"),
                        to=now_et.strftime("%Y-%m-%d"),
                        limit=1
                    ))
                    
                    if aggs:
                        return float(aggs[0].close)
                except Exception as e:
                    logger.warning(f"Polygon.io failed for {ticker}, using yfinance: {e}")
            
            # Market is closed - use market close price from previous trading day
            logger.info(f"Market is closed - using market close price for {ticker}")
            try:
                # Get previous trading day
                last_trading_day = get_last_trading_day(now_et)
                date_str = last_trading_day.strftime("%Y-%m-%d")
                
                # Try Polygon.io first for previous day
                aggs = list(self.polygon_client.list_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=date_str,
                    to=date_str,
                    limit=1
                ))
                
                if aggs:
                    close_price = float(aggs[0].close)
                    logger.info(f"Got market close price for {ticker}: ${close_price:.2f}")
                    return close_price
            except Exception as e:
                logger.warning(f"Polygon.io previous day failed for {ticker}: {e}")
            
            # Fallback to yfinance
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            
            if not hist.empty:
                close_price = float(hist['Close'].iloc[-1])
                logger.info(f"Got market close price from yfinance for {ticker}: ${close_price:.2f}")
                return close_price
            return None
            
        except Exception as e:
            logger.error(f"Error getting stock price for {ticker}: {e}")
            return None
    
    def get_ivr_from_supabase(self, ticker: str) -> Optional[float]:
        """Get the current IVR (Implied Volatility Rank) from Supabase."""
        try:
            result = self.supabase.table('iv_history').select('*').eq('ticker', ticker).order('date', desc=True).limit(252).execute()
            
            if not result.data or len(result.data) < 20:
                logger.warning(f"Insufficient IV history for {ticker}")
                return None
            
            # Calculate IVR
            current_iv = result.data[0]['iv']
            iv_values = [entry['iv'] for entry in result.data]
            iv_min = min(iv_values)
            iv_max = max(iv_values)
            
            if iv_max > iv_min:
                ivr = ((current_iv - iv_min) / (iv_max - iv_min)) * 100
                return round(ivr, 1)
            
            return None
        except Exception as e:
            logger.error(f"Error getting IVR for {ticker}: {e}")
            return None
    
    def get_options_chain(self, ticker: str, expiration_date: str) -> List[Dict]:
        """Get options chain for a specific ticker and expiration date with market-aware fetching."""
        try:
            from config import is_trading_day
            
            # Determine if market is open (ET timezone)
            from pytz import timezone as pytz_timezone
            et_tz = pytz_timezone('US/Eastern')
            now_et = datetime.now(et_tz)
            market_is_open = is_trading_day(now_et) and 9.5 <= now_et.hour + now_et.minute/60 <= 16
            
            contracts = list(self.polygon_client.list_options_contracts(
                underlying_ticker=ticker,
                expiration_date=expiration_date,
                limit=1000
            ))
            
            options_data = []
            for contract in contracts:
                if not hasattr(contract, 'strike_price') or not hasattr(contract, 'contract_type'):
                    continue
                
                # Get snapshot data for Greeks and pricing
                try:
                    snapshot = self.polygon_client.get_snapshot_option(
                        contract.underlying_ticker,
                        contract.ticker
                    )
                    
                    if snapshot:
                        # Use appropriate data based on market status
                        if market_is_open:
                            # Market open - use 15-minute delayed data
                            bid = getattr(snapshot, 'bid', 0)
                            ask = getattr(snapshot, 'ask', 0)
                            last = getattr(snapshot, 'last', 0)
                            volume = getattr(snapshot, 'volume', 0)
                        else:
                            # Market closed - use previous day's data
                            bid = getattr(snapshot, 'bid', 0)
                            ask = getattr(snapshot, 'ask', 0)
                            last = getattr(snapshot, 'last', 0)
                            volume = getattr(snapshot, 'volume', 0)
                        
                        option_data = {
                            'ticker': contract.ticker,
                            'underlying': contract.underlying_ticker,
                            'strike': contract.strike_price,
                            'expiration': contract.expiration_date,
                            'contract_type': contract.contract_type,
                            'bid': bid,
                            'ask': ask,
                            'last': last,
                            'volume': volume,
                            'open_interest': getattr(snapshot, 'open_interest', 0),
                            'implied_volatility': getattr(snapshot, 'implied_volatility', 0),
                            'delta': getattr(snapshot, 'delta', 0),
                            'gamma': getattr(snapshot, 'gamma', 0),
                            'theta': getattr(snapshot, 'theta', 0),
                            'vega': getattr(snapshot, 'vega', 0),
                            'rho': getattr(snapshot, 'rho', 0),
                            'market_status': 'open' if market_is_open else 'closed'
                        }
                        options_data.append(option_data)
                except Exception as e:
                    logger.warning(f"Error getting snapshot for {contract.ticker}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(options_data)} options for {ticker} exp {expiration_date} (market: {'open' if market_is_open else 'closed'})")
            return options_data
        except Exception as e:
            logger.error(f"Error getting options chain for {ticker} exp {expiration_date}: {e}")
            return []
    
    def get_available_expirations(self, ticker: str, min_days: int = 30, max_days: int = 60) -> List[str]:
        """Get available expiration dates within the specified range."""
        try:
            start_date = datetime.now(timezone.utc) + timedelta(days=min_days)
            end_date = datetime.now(timezone.utc) + timedelta(days=max_days)
            
            # Get all available expirations
            contracts = list(self.polygon_client.list_options_contracts(
                underlying_ticker=ticker,
                limit=1000
            ))
            
            # Extract unique expiration dates
            expirations = set()
            for contract in contracts:
                if hasattr(contract, 'expiration_date'):
                    # Parse expiration date and make it timezone-aware
                    exp_date = datetime.strptime(contract.expiration_date, '%Y-%m-%d')
                    exp_date = exp_date.replace(tzinfo=timezone.utc)  # Make timezone-aware
                    
                    if start_date <= exp_date <= end_date:
                        expirations.add(contract.expiration_date)
            
            # Sort and return
            sorted_expirations = sorted(list(expirations))
            logger.info(f"Found {len(sorted_expirations)} expiration dates for {ticker}")
            return sorted_expirations
            
        except Exception as e:
            logger.error(f"Error getting expirations for {ticker}: {e}")
            return []

class SupabaseFetcher:
    """Fetches IVR data from Supabase."""
    
    def __init__(self):
        """Initialize Supabase client."""
        try:
            from supabase import create_client
            from config import get_supabase_credentials
            
            creds = get_supabase_credentials()
            self.supabase = create_client(creds['url'], creds['key'])
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {e}")
            self.supabase = None
    
    def get_ivr(self, ticker: str) -> Optional[float]:
        """Get the current IVR (Implied Volatility Rank) for a ticker."""
        try:
            if not self.supabase:
                logger.error("Supabase client not initialized")
                return None
            
            result = self.supabase.table('iv_history').select('*').eq('ticker', ticker).order('date', desc=True).limit(252).execute()
            
            if not result.data:
                logger.warning(f"No IV data found for {ticker}")
                return None
            
            # Calculate IVR from the data
            iv_values = [float(row['iv']) for row in result.data if row['iv'] is not None]
            
            if len(iv_values) < 20:  # Need at least 20 data points
                logger.warning(f"Insufficient IV data for {ticker}: {len(iv_values)} points")
                return None
            
            # Calculate IVR (percentile rank of current IV)
            current_iv = iv_values[0]  # Most recent IV
            ivr = (sum(1 for iv in iv_values if iv < current_iv) / len(iv_values)) * 100
            
            logger.info(f"IVR for {ticker}: {ivr:.1f}%")
            return ivr
            
        except Exception as e:
            logger.error(f"Error getting IVR for {ticker}: {e}")
            return None
    
    def get_available_expirations(self, ticker: str, min_days: int = 30, max_days: int = 60) -> List[str]:
        """Get available expiration dates within the specified range."""
        try:
            start_date = datetime.now(timezone.utc) + timedelta(days=min_days)
            end_date = datetime.now(timezone.utc) + timedelta(days=max_days)
            
            contracts = list(self.polygon_client.list_options_contracts(
                underlying_ticker=ticker,
                expiration_date_gte=start_date.strftime("%Y-%m-%d"),
                expiration_date_lte=end_date.strftime("%Y-%m-%d"),
                limit=1000
            ))
            
            # Extract unique expiration dates
            expirations = set()
            for contract in contracts:
                if hasattr(contract, 'expiration_date'):
                    expirations.add(contract.expiration_date)
            
            return sorted(list(expirations))
        except Exception as e:
            logger.error(f"Error getting expirations for {ticker}: {e}")
            return []

class TechnicalAnalyzer:
    """Class for calculating technical indicators."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if insufficient data
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(prices) < slow + signal:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0,
            'signal': signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0,
            'histogram': histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 0.5}
        
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        current_price = prices.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_middle = middle.iloc[-1]
        
        # Calculate position within bands (0 = lower band, 1 = upper band)
        if current_upper != current_lower:
            position = (current_price - current_lower) / (current_upper - current_lower)
        else:
            position = 0.5
        
        return {
            'upper': current_upper if not pd.isna(current_upper) else 0,
            'middle': current_middle if not pd.isna(current_middle) else 0,
            'lower': current_lower if not pd.isna(current_lower) else 0,
            'position': position if not pd.isna(position) else 0.5
        }
    
    @staticmethod
    def calculate_support_resistance(prices: pd.Series, window: int = 20) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        if len(prices) < window:
            current_price = prices.iloc[-1] if len(prices) > 0 else 0
            return {'support': current_price * 0.95, 'resistance': current_price * 1.05}
        
        recent_prices = prices.tail(window)
        highs = recent_prices.rolling(window=5).max()
        lows = recent_prices.rolling(window=5).min()
        
        # Find significant levels
        resistance_levels = highs.nlargest(3).values
        support_levels = lows.nsmallest(3).values
        
        current_price = prices.iloc[-1]
        
        # Find nearest support and resistance
        resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.05)
        support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
        
        return {
            'support': resistance if not pd.isna(resistance) else current_price * 0.95,
            'resistance': support if not pd.isna(support) else current_price * 1.05
        }
