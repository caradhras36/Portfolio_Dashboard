"""
Polygon.io Market Data Client
Provides real-time stock and options data integration
"""

import httpx
import asyncio
import logging
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PriceData:
    """Price data structure"""
    ticker: str
    price: float
    timestamp: datetime
    volume: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    prev_close: Optional[float] = None

@dataclass
class OptionData:
    """Options data structure"""
    ticker: str
    strike_price: float
    expiration_date: str
    contract_type: str  # 'call' or 'put'
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

class PolygonMarketDataClient:
    """Polygon.io market data client with caching and error handling"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        self.base_url = "https://api.polygon.io"
        self.cache = {}
        self.cache_ttl = 60  # 1 minute cache for stock prices
        self.options_cache_ttl = 300  # 5 minutes cache for options data
        self.rate_limit_delay = 0.1  # 100ms between requests to respect rate limits
        
        if not self.api_key:
            logger.warning("Polygon API key not found. Using mock data.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            logger.info("Polygon.io client initialized with API key")
    
    async def get_stock_price(self, ticker: str) -> Optional[PriceData]:
        """Get current stock price for a ticker"""
        try:
            # Check cache first
            cache_key = f"stock_{ticker}"
            cached_data = self._get_cached_data(cache_key, self.cache_ttl)
            if cached_data:
                logger.debug(f"Using cached price for {ticker}")
                return cached_data
            
            if self.mock_mode:
                return self._get_mock_stock_price(ticker)
            
            # Fetch from Polygon.io
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
                params = {"apikey": self.api_key}
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("status") == "OK" and data.get("results"):
                    result = data["results"]
                    price_data = PriceData(
                        ticker=ticker.upper(),
                        price=result.get("lastTrade", {}).get("p", 0),
                        timestamp=datetime.now(),
                        volume=result.get("lastTrade", {}).get("s", 0),
                        high=result.get("day", {}).get("h", None),
                        low=result.get("day", {}).get("l", None),
                        open=result.get("day", {}).get("o", None),
                        prev_close=result.get("prevDay", {}).get("c", None)
                    )
                    
                    # Cache the result
                    self._cache_data(cache_key, price_data)
                    
                    # Respect rate limits
                    await asyncio.sleep(self.rate_limit_delay)
                    
                    return price_data
                else:
                    logger.warning(f"No data returned for {ticker}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching stock price for {ticker}: {e}")
            return self._get_mock_stock_price(ticker)
    
    async def get_multiple_stock_prices(self, tickers: List[str]) -> Dict[str, PriceData]:
        """Get prices for multiple tickers efficiently"""
        results = {}
        
        # Process in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            tasks = [self.get_stock_price(ticker) for ticker in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for ticker, result in zip(batch, batch_results):
                if isinstance(result, PriceData):
                    results[ticker] = result
                elif isinstance(result, Exception):
                    logger.error(f"Error fetching {ticker}: {result}")
                else:
                    logger.warning(f"No data for {ticker}")
            
            # Small delay between batches
            if i + batch_size < len(tickers):
                await asyncio.sleep(0.5)
        
        return results
    
    async def get_batch_option_prices(self, option_tickers: List[str], underlying_ticker: str = None) -> Dict[str, Dict]:
        """Get prices and Greeks for multiple options using v3 snapshot endpoint
        
        Args:
            option_tickers: List of option contract tickers (e.g., 'O:BILL251017C00059000')
            underlying_ticker: The underlying asset ticker (e.g., 'BILL')
        """
        if not option_tickers:
            return {}
        
        results = {}
        
        try:
            # Use v3 snapshot endpoint: /v3/snapshot/options/{underlyingAsset}/{optionContract}
            # Process in parallel for better performance
            async with httpx.AsyncClient(timeout=30.0) as client:
                tasks = []
                for option_ticker in option_tickers:
                    # Extract underlying ticker from option ticker if not provided
                    # Format: O:BILL251017C00059000 -> BILL
                    if not underlying_ticker:
                        # Remove O: prefix and extract ticker before date
                        ticker_part = option_ticker.replace('O:', '')
                        # Find where the date starts (6 digits: YYMMDD)
                        import re
                        match = re.match(r'([A-Z]+)\d{6}', ticker_part)
                        if match:
                            underlying = match.group(1)
                        else:
                            logger.warning(f"Could not extract underlying ticker from {option_ticker}")
                            continue
                    else:
                        underlying = underlying_ticker
                    
                    url = f"{self.base_url}/v3/snapshot/options/{underlying}/{option_ticker}"
                    params = {"apiKey": self.api_key}
                    tasks.append((option_ticker, client.get(url, params=params)))
                
                # Execute all requests in parallel
                responses = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
                
                for (option_ticker, _), response in zip(tasks, responses):
                    if isinstance(response, Exception):
                        logger.warning(f"Error fetching {option_ticker}: {response}")
                        continue
                    
                    try:
                        if response.status_code == 200:
                            data = response.json()
                            
                            if data.get("status") == "OK" and data.get("results"):
                                result = data["results"]
                                
                                # Extract Greeks from the greeks object
                                greeks = result.get("greeks", {})
                                last_quote = result.get("last_quote", {})
                                last_trade = result.get("last_trade", {})
                                day = result.get("day", {})
                                
                                results[option_ticker] = {
                                    "last_price": last_trade.get("price", 0) or day.get("close", 0) or last_quote.get("midpoint", 0),
                                    "bid": last_quote.get("bid", 0),
                                    "ask": last_quote.get("ask", 0),
                                    "volume": day.get("volume", 0),
                                    "open_interest": result.get("open_interest", 0),
                                    "implied_volatility": result.get("implied_volatility"),
                                    "delta": greeks.get("delta"),
                                    "gamma": greeks.get("gamma"),
                                    "theta": greeks.get("theta"),
                                    "vega": greeks.get("vega")
                                }
                                
                                logger.info(f"✅ Got Greeks for {option_ticker}: Delta={greeks.get('delta')}, Gamma={greeks.get('gamma')}, Theta={greeks.get('theta')}, Vega={greeks.get('vega')}, IV={result.get('implied_volatility')}")
                            else:
                                logger.warning(f"No results for {option_ticker}: {data.get('status')}")
                    except Exception as e:
                        logger.warning(f"Error parsing response for {option_ticker}: {e}")
                        continue
                
                return results
                
        except Exception as e:
            logger.error(f"Error fetching batch option prices: {e}")
            return {}
    
    async def get_options_chain(self, ticker: str, expiration_date: Optional[str] = None) -> List[OptionData]:
        """Get options chain for a ticker"""
        try:
            # Check cache first
            cache_key = f"options_{ticker}_{expiration_date or 'all'}"
            cached_data = self._get_cached_data(cache_key, self.options_cache_ttl)
            if cached_data:
                logger.debug(f"Using cached options data for {ticker}")
                return cached_data
            
            # Use real API data but with proper batching to avoid rate limits
            logger.info(f"Fetching real options chain for {ticker} with batch processing")
            
            # Fetch from Polygon.io with proper batching
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/v3/reference/options/contracts"
                params = {
                    "underlying_ticker": ticker,
                    "apikey": self.api_key,
                    "limit": 1000
                }
                
                if expiration_date:
                    params["expiration_date"] = expiration_date
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                options_data = []
                
                if data.get("status") == "OK" and data.get("results"):
                    # Get contract details without individual price calls
                    for contract in data["results"]:
                        # Create option data with basic info, prices will be fetched separately
                        option_data = OptionData(
                            ticker=contract["ticker"],
                            strike_price=contract["strike_price"],
                            expiration_date=contract["expiration_date"],
                            contract_type=contract["contract_type"],
                            last_price=0.0,  # Will be updated by batch price fetching
                            bid=0.0,
                            ask=0.0,
                            volume=0,
                            open_interest=0,
                            implied_volatility=0.30,  # Default IV
                            delta=0.0,  # Will be calculated
                            gamma=0.0,
                            theta=0.0,
                            vega=0.0
                        )
                        options_data.append(option_data)
                    
                    # Cache the result
                    self._cache_data(cache_key, options_data)
                    
                    # Respect rate limits
                    await asyncio.sleep(self.rate_limit_delay)
                
                return options_data
                
        except Exception as e:
            logger.error(f"Error fetching options chain for {ticker}: {e}")
            return self._get_mock_options_chain(ticker, expiration_date)
    
    async def _get_option_price(self, option_ticker: str) -> Optional[Dict]:
        """Get price data for a specific option"""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/v2/snapshot/options/{option_ticker}"
                params = {"apikey": self.api_key}
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("status") == "OK" and data.get("results"):
                    result = data["results"]
                    return {
                        "last_price": result.get("lastTrade", {}).get("p", 0),
                        "bid": result.get("details", {}).get("bid", 0),
                        "ask": result.get("details", {}).get("ask", 0),
                        "volume": result.get("lastTrade", {}).get("s", 0),
                        "open_interest": result.get("details", {}).get("open_interest", 0),
                        "implied_volatility": result.get("details", {}).get("implied_volatility"),
                        "delta": result.get("details", {}).get("delta"),
                        "gamma": result.get("details", {}).get("gamma"),
                        "theta": result.get("details", {}).get("theta"),
                        "vega": result.get("details", {}).get("vega")
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error fetching option price for {option_ticker}: {e}")
            return None
    
    def _get_mock_stock_price(self, ticker: str) -> PriceData:
        """Get mock stock price for testing/demo purposes"""
        mock_prices = {
            'AAPL': 175.50,
            'TSLA': 245.30,
            'MSFT': 380.20,
            'GOOGL': 142.80,
            'AMZN': 155.40,
            'NVDA': 875.60,
            'META': 485.30,
            'NFLX': 485.90,
            'AMD': 135.70,
            'SOFI': 8.45,
            'HIMS': 9.20,
            'PLTR': 18.75,
            'CRM': 275.80,
            'SNOW': 185.40
        }
        
        # Add some random variation to simulate real price movement
        base_price = mock_prices.get(ticker, 100.0)
        variation = random.uniform(-0.02, 0.02)  # ±2% variation
        current_price = base_price * (1 + variation)
        
        return PriceData(
            ticker=ticker.upper(),
            price=round(current_price, 2),
            timestamp=datetime.now(),
            volume=random.randint(100000, 10000000),
            high=round(current_price * 1.01, 2),
            low=round(current_price * 0.99, 2),
            open=round(current_price * random.uniform(0.998, 1.002), 2),
            prev_close=base_price
        )
    
    def _get_mock_options_chain(self, ticker: str, expiration_date: Optional[str] = None) -> List[OptionData]:
        """Get mock options chain for testing/demo purposes"""
        stock_price = self._get_mock_stock_price(ticker)
        current_price = stock_price.price
        
        options = []
        
        # Generate options around current price
        strikes = []
        base_strike = round(current_price / 5) * 5  # Round to nearest $5
        
        for i in range(-5, 6):
            strikes.append(base_strike + (i * 5))
        
        for strike in strikes:
            for contract_type in ['call', 'put']:
                # Calculate mock option price using simplified Black-Scholes approximation
                if contract_type == 'call':
                    option_price = max(0.01, (current_price - strike) * 0.1 + random.uniform(0.5, 3.0))
                    delta = min(0.95, max(0.05, (current_price - strike) / current_price + 0.5))
                else:  # put
                    option_price = max(0.01, (strike - current_price) * 0.1 + random.uniform(0.5, 3.0))
                    delta = max(-0.95, min(-0.05, (strike - current_price) / current_price - 0.5))
                
                option_data = OptionData(
                    ticker=f"{ticker}{strike}C" if contract_type == 'call' else f"{ticker}{strike}P",
                    strike_price=strike,
                    expiration_date=expiration_date or "2025-01-17",
                    contract_type=contract_type,
                    last_price=round(option_price, 2),
                    bid=round(option_price * 0.98, 2),
                    ask=round(option_price * 1.02, 2),
                    volume=random.randint(10, 1000),
                    open_interest=random.randint(100, 10000),
                    implied_volatility=round(random.uniform(0.15, 0.45), 3),
                    delta=round(delta, 3),
                    gamma=round(random.uniform(0.01, 0.05), 3),
                    theta=round(random.uniform(-0.05, -0.01), 3),
                    vega=round(random.uniform(0.1, 0.3), 3)
                )
                options.append(option_data)
        
        return options
    
    def _get_cached_data(self, key: str, ttl: int) -> Optional[Any]:
        """Get cached data if still valid"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < ttl:
                return data
            else:
                # Remove expired cache
                del self.cache[key]
        return None
    
    def _cache_data(self, key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[key] = (data, time.time())
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        try:
            if self.mock_mode:
                return {
                    "market": "open",
                    "exchanges": {
                        "NASDAQ": "open",
                        "NYSE": "open"
                    },
                    "serverTime": datetime.now().isoformat()
                }
            
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/v1/marketstatus/now"
                params = {"apikey": self.api_key}
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                return response.json()
                
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return {
                "market": "unknown",
                "exchanges": {},
                "serverTime": datetime.now().isoformat()
            }

# Global instance
market_data_client = PolygonMarketDataClient()

# Example usage and testing
async def test_polygon_client():
    """Test the Polygon client functionality"""
    client = PolygonMarketDataClient()
    
    print("Testing Polygon.io Market Data Client")
    print("=" * 50)
    
    # Test stock price fetching
    print("1. Testing stock price fetching...")
    price = await client.get_stock_price("AAPL")
    if price:
        print(f"   AAPL: ${price.price} (Volume: {price.volume:,})")
    
    # Test multiple stock prices
    print("\n2. Testing multiple stock prices...")
    prices = await client.get_multiple_stock_prices(["AAPL", "TSLA", "MSFT"])
    for ticker, price_data in prices.items():
        print(f"   {ticker}: ${price_data.price}")
    
    # Test options chain
    print("\n3. Testing options chain...")
    options = await client.get_options_chain("AAPL")
    print(f"   Found {len(options)} options contracts")
    for option in options[:5]:  # Show first 5
        print(f"   {option.ticker}: ${option.last_price} (Strike: ${option.strike_price})")
    
    # Test market status
    print("\n4. Testing market status...")
    status = await client.get_market_status()
    print(f"   Market: {status.get('market', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(test_polygon_client())
