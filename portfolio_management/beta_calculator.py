#!/usr/bin/env python3
"""
Beta Calculator - Calculate stock beta from historical returns
Uses Polygon.io historical price data
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
import httpx
import os

logger = logging.getLogger(__name__)

class BetaCalculator:
    """Calculate stock beta using historical price data from Polygon.io"""
    
    def __init__(self, polygon_api_key: str):
        self.api_key = polygon_api_key
        self.base_url = "https://api.polygon.io"
        self.beta_cache = {}  # Cache betas for 24 hours
        
    async def calculate_beta(self, ticker: str, benchmark: str = 'SPY') -> tuple[float, float]:
        """Calculate beta and volatility for a stock using last year's data
        
        Uses all available daily data from the past year for accurate calculation
        Beta = Covariance(stock_returns, benchmark_returns) / Variance(benchmark_returns)
        Volatility = StdDev(stock_returns) * sqrt(252) (annualized)
        
        Returns: (beta, volatility)
        """
        try:
            # Check cache first (24 hour TTL)
            cache_key = f"beta_vol_{ticker}_{benchmark}"
            if cache_key in self.beta_cache:
                cache_time, cached_values = self.beta_cache[cache_key]
                if (datetime.now() - cache_time).seconds < 86400:
                    return cached_values
            
            # Fetch 1 year of historical data (all daily returns)
            stock_returns, benchmark_returns = await asyncio.gather(
                self._get_daily_returns(ticker),
                self._get_daily_returns(benchmark)
            )
            
            if len(stock_returns) < 20 or len(benchmark_returns) < 20:
                logger.warning(f"Insufficient data for {ticker} beta/volatility calculation (got {len(stock_returns)} points)")
                return 1.0, 0.25
            
            # Align the returns (same length)
            min_len = min(len(stock_returns), len(benchmark_returns))
            stock_returns = stock_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
            
            # Calculate beta
            covariance = np.cov(stock_returns, benchmark_returns)[0][1]
            variance = np.var(benchmark_returns)
            beta = covariance / variance if variance > 0 else 1.0
            
            # Calculate annualized volatility
            # Volatility = daily std dev * sqrt(252 trading days)
            daily_volatility = np.std(stock_returns)
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            beta = round(beta, 2)
            volatility = round(annualized_volatility, 2)
            
            # Cache the results
            self.beta_cache[cache_key] = (datetime.now(), (beta, volatility))
            
            logger.info(f"Calculated {ticker}: beta={beta:.2f}, volatility={volatility:.1%} (from {min_len} data points)")
            return beta, volatility
            
        except Exception as e:
            logger.error(f"Error calculating beta/volatility for {ticker}: {e}")
            return 1.0, 0.25
    
    async def _get_daily_returns(self, ticker: str) -> List[float]:
        """Get all daily returns from the past year - uses cached prices when available"""
        try:
            # Import price cache
            from price_cache import get_price_cache, PriceData
            price_cache = get_price_cache()
            
            # Use yesterday as end date (market data not available for today yet)
            end_date_dt = datetime.now() - timedelta(days=1)
            start_date_dt = end_date_dt - timedelta(days=365)  # 1 year lookback
            end_date = end_date_dt.date()
            start_date = start_date_dt.date()
            
            # Try to get cached prices first
            cached_prices = await price_cache.get_cached_prices(ticker, start_date, end_date)
            
            if cached_prices and len(cached_prices) >= 20:
                # Use cached prices
                all_prices = [p.close_price for p in cached_prices]
                logger.debug(f"{ticker}: Using {len(all_prices)} cached prices ⚡")
            else:
                # Fetch from Polygon.io
                logger.debug(f"{ticker}: Fetching from Polygon.io (cache miss)")
                url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date_dt.strftime('%Y-%m-%d')}/{end_date_dt.strftime('%Y-%m-%d')}"
                params = {
                    'apikey': self.api_key,
                    'adjusted': 'true',
                    'sort': 'asc',
                    'limit': 50000
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params, timeout=10.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get('status') != 'OK' or not data.get('results'):
                            return []
                        
                        results = data['results']
                        all_prices = [bar['c'] for bar in results]
                        logger.debug(f"{ticker}: Got {len(all_prices)} trading days from Polygon.io")
                        
                        # Save to cache for future use (async, don't wait)
                        price_data_list = []
                        for bar in results:
                            price_date = datetime.fromtimestamp(bar['t'] / 1000).date()
                            volume_value = bar.get('v', 0)  # Get volume, default to 0
                            
                            price_data_list.append(PriceData(
                                ticker=ticker,
                                date=price_date,
                                close_price=bar['c'],
                                open_price=bar.get('o'),
                                high_price=bar.get('h'),
                                low_price=bar.get('l'),
                                volume=volume_value
                            ))
                        
                        # Debug: Check if we got volumes
                        has_volume = sum(1 for p in price_data_list if p.volume and p.volume > 0)
                        logger.info(f"{ticker}: Saving {len(price_data_list)} prices, {has_volume} have volume data")
                        
                        # Save to cache (fire and forget)
                        asyncio.create_task(price_cache.save_prices(price_data_list))
                    else:
                        return []
            
            # Calculate returns from all consecutive days
            returns = []
            for i in range(1, len(all_prices)):
                ret = (all_prices[i] - all_prices[i-1]) / all_prices[i-1]
                returns.append(ret)
            
            logger.debug(f"{ticker}: Calculated {len(returns)} daily returns from {len(all_prices)} trading days")
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching sampled returns for {ticker}: {e}")
            return []
    
    async def calculate_portfolio_beta_and_volatility(self, positions: List[Any]) -> tuple[float, float]:
        """Calculate portfolio beta and volatility as weighted averages"""
        stocks = [pos for pos in positions if pos.position_type == 'stock']
        
        if not stocks:
            return 1.0, 0.20
        
        # Calculate beta and volatility for all stocks in parallel
        metrics_tasks = [self.calculate_beta(stock.ticker) for stock in stocks]
        metrics = await asyncio.gather(*metrics_tasks)
        
        # Weighted averages
        total_value = sum(pos.quantity * pos.current_price for pos in stocks)
        if total_value == 0:
            return 1.0, 0.20
        
        weighted_beta = 0.0
        weighted_volatility = 0.0
        
        for pos, (beta, volatility) in zip(stocks, metrics):
            position_value = pos.quantity * pos.current_price
            weight = position_value / total_value
            weighted_beta += beta * weight
            weighted_volatility += volatility * weight
        
        logger.info(f"Portfolio: beta={weighted_beta:.2f}, volatility={weighted_volatility:.1%} (from {len(stocks)} stocks)")
        return round(weighted_beta, 2), round(weighted_volatility, 2)

# Global instance
beta_calculator = None

def get_beta_calculator():
    """Get or create beta calculator instance"""
    global beta_calculator
    if beta_calculator is None:
        api_key = os.getenv('POLYGON_API_KEY')
        if api_key:
            beta_calculator = BetaCalculator(api_key)
    return beta_calculator
