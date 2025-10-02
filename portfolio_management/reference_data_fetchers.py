"""Data fetchers for options chains, stock data, and technical indicators."""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
from base import OptionsDataFetcher, TechnicalAnalyzer
from config import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD,
    MIN_DAYS_TO_EXPIRATION, MAX_DAYS_TO_EXPIRATION
)

logger = logging.getLogger(__name__)

class StockDataFetcher(OptionsDataFetcher):
    """Fetches stock data and technical indicators."""
    
    def get_stock_data(self, ticker: str, days: int = 252) -> Optional[pd.DataFrame]:
        """Get historical stock data for technical analysis."""
        try:
            # Use yfinance for historical data (more reliable for free data)
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")  # Get 1 year of data
            
            if hist.empty:
                logger.warning(f"No historical data found for {ticker}")
                return None
            
            # Ensure we have enough data
            if len(hist) < 50:
                logger.warning(f"Insufficient data for {ticker}: {len(hist)} days")
                return None
            
            return hist
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {e}")
            return None
    
    def get_technical_indicators(self, ticker: str) -> Dict[str, Any]:
        """Get all technical indicators for a stock."""
        try:
            stock_data = self.get_stock_data(ticker)
            if stock_data is None:
                return {}
            
            prices = stock_data['Close']
            
            # Calculate technical indicators
            rsi = TechnicalAnalyzer.calculate_rsi(prices, RSI_PERIOD)
            macd = TechnicalAnalyzer.calculate_macd(prices, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            bollinger = TechnicalAnalyzer.calculate_bollinger_bands(prices, BOLLINGER_PERIOD, BOLLINGER_STD)
            support_resistance = TechnicalAnalyzer.calculate_support_resistance(prices)
            
            # Get current price and IVR
            current_price = self.get_stock_price(ticker)
            ivr = self.get_ivr_from_supabase(ticker)
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'rsi': rsi,
                'macd': macd,
                'bollinger_bands': bollinger,
                'support_resistance': support_resistance,
                'ivr': ivr,
                'volume': stock_data['Volume'].iloc[-1] if not stock_data['Volume'].empty else 0,
                'price_change_pct': ((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100) if len(prices) > 1 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {ticker}: {e}")
            return {}

class OptionsChainFetcher(OptionsDataFetcher):
    """Fetches and analyzes options chains."""
    
    def get_options_data(self, ticker: str) -> List[Dict[str, Any]]:
        """Get comprehensive options data for a ticker."""
        try:
            # Get available expirations
            expirations = self.get_available_expirations(ticker, MIN_DAYS_TO_EXPIRATION, MAX_DAYS_TO_EXPIRATION)
            
            if not expirations:
                logger.warning(f"No suitable expirations found for {ticker}")
                return []
            
            all_options = []
            
            # Get options for each expiration
            for exp_date in expirations[:3]:  # Limit to first 3 expirations
                options_chain = self.get_options_chain(ticker, exp_date)
                
                # Filter for liquid options
                liquid_options = [
                    opt for opt in options_chain
                    if opt['open_interest'] >= 100 and opt['volume'] >= 10
                ]
                
                all_options.extend(liquid_options)
            
            return all_options
        except Exception as e:
            logger.error(f"Error fetching options data for {ticker}: {e}")
            return []
    
    def filter_options_for_selling(self, options_data: List[Dict[str, Any]], 
                                 current_price: float) -> List[Dict[str, Any]]:
        """Filter options suitable for selling strategies."""
        suitable_options = []
        
        for option in options_data:
            strike = option['strike']
            contract_type = option['contract_type']
            iv = option['implied_volatility']
            delta = abs(option['delta'])
            
            # Filter criteria for selling
            if iv < 0.15:  # Skip low IV options
                continue
            
            # For calls: sell OTM (strike > current_price)
            if contract_type == 'call' and strike > current_price * 1.02:
                # Prefer strikes with delta between 0.15-0.35
                if 0.15 <= delta <= 0.35:
                    option['strategy'] = 'call_sell'
                    option['distance_from_price'] = (strike - current_price) / current_price
                    suitable_options.append(option)
            
            # For puts: sell OTM (strike < current_price)
            elif contract_type == 'put' and strike < current_price * 0.98:
                # Prefer strikes with delta between 0.15-0.35
                if 0.15 <= delta <= 0.35:
                    option['strategy'] = 'put_sell'
                    option['distance_from_price'] = (current_price - strike) / current_price
                    suitable_options.append(option)
        
        return suitable_options

class MarketDataFetcher(OptionsDataFetcher):
    """Fetches broader market data and sentiment indicators."""
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Get market sentiment indicators."""
        try:
            # VIX data
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="5d")
            vix_current = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 20
            vix_change = ((vix_current - vix_hist['Close'].iloc[-2]) / vix_hist['Close'].iloc[-2] * 100) if len(vix_hist) > 1 else 0
            
            # SPY data for market direction
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="5d")
            spy_current = spy_hist['Close'].iloc[-1] if not spy_hist.empty else 400
            spy_change = ((spy_current - spy_hist['Close'].iloc[-2]) / spy_hist['Close'].iloc[-2] * 100) if len(spy_hist) > 1 else 0
            
            return {
                'vix': vix_current,
                'vix_change': vix_change,
                'spy_price': spy_current,
                'spy_change': spy_change,
                'market_direction': 'bullish' if spy_change > 0 else 'bearish' if spy_change < 0 else 'neutral'
            }
        except Exception as e:
            logger.error(f"Error fetching market sentiment: {e}")
            return {
                'vix': 20,
                'vix_change': 0,
                'spy_price': 400,
                'spy_change': 0,
                'market_direction': 'neutral'
            }
    
    def get_sector_performance(self) -> Dict[str, float]:
        """Get sector ETF performance."""
        try:
            sector_etfs = {
                'XLK': 'Technology',
                'XLF': 'Financials',
                'XLV': 'Healthcare',
                'XLE': 'Energy',
                'XLY': 'Consumer Discretionary',
                'XLI': 'Industrials',
                'XLP': 'Consumer Staples',
                'XLU': 'Utilities'
            }
            
            sector_performance = {}
            for etf, name in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="2d")
                    if not hist.empty and len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2]
                        change = ((current - prev) / prev) * 100
                        sector_performance[name] = round(change, 2)
                except Exception as e:
                    logger.warning(f"Error fetching {etf}: {e}")
                    continue
            
            return sector_performance
        except Exception as e:
            logger.error(f"Error fetching sector performance: {e}")
            return {}

class DataAggregator:
    """Aggregates data from multiple sources for comprehensive analysis."""
    
    def __init__(self):
        self.stock_fetcher = StockDataFetcher()
        self.options_fetcher = OptionsChainFetcher()
        self.market_fetcher = MarketDataFetcher()
    
    def get_comprehensive_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive data for a single ticker."""
        try:
            logger.info(f"Fetching comprehensive data for {ticker}")
            
            # Get stock and technical data
            stock_data = self.stock_fetcher.get_technical_indicators(ticker)
            if not stock_data:
                logger.warning(f"No stock data available for {ticker}")
                return {}
            
            # Get options data
            options_data = self.options_fetcher.get_options_data(ticker)
            if not options_data:
                logger.warning(f"No options data available for {ticker}")
                return stock_data
            
            # Filter suitable options for selling
            suitable_options = self.options_fetcher.filter_options_for_selling(
                options_data, stock_data['current_price']
            )
            
            # Get market sentiment
            market_sentiment = self.market_fetcher.get_market_sentiment()
            
            return {
                'ticker': ticker,
                'stock_data': stock_data,
                'options_data': suitable_options,
                'market_sentiment': market_sentiment,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting comprehensive data for {ticker}: {e}")
            return {}
    
    def get_multiple_tickers_data(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive data for multiple tickers."""
        results = {}
        
        for ticker in tickers:
            try:
                data = self.get_comprehensive_data(ticker)
                if data:
                    results[ticker] = data
                else:
                    logger.warning(f"Skipping {ticker} due to insufficient data")
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        return results
