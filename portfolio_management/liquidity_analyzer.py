#!/usr/bin/env python3
"""
Liquidity Analyzer - Calculate real liquidity scores based on volume and position size
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class LiquidityAnalyzer:
    """Analyzes portfolio liquidity based on trading volume"""
    
    def __init__(self):
        self.volume_cache = {}  # Cache average volumes
        
    async def calculate_stock_liquidity(self, ticker: str, position_size_dollars: float) -> float:
        """Calculate liquidity score for a single stock
        
        Liquidity Score (0-100):
        - 90-100: Highly liquid (can exit easily)
        - 70-90: Good liquidity
        - 50-70: Moderate liquidity
        - 30-50: Some concerns
        - 0-30: Poor liquidity
        
        Based on:
        1. Average daily volume
        2. Position size relative to daily volume
        3. Days to liquidate (assuming 10% of daily volume)
        """
        try:
            # Get average volume from price cache
            from price_cache import get_price_cache
            price_cache = get_price_cache()
            
            # Get 1 year of historical data for accurate volume average
            end_date = (datetime.now() - timedelta(days=1)).date()
            start_date = end_date - timedelta(days=365)
            
            cached_prices = await price_cache.get_cached_prices(ticker, start_date, end_date)
            
            if not cached_prices or len(cached_prices) < 5:
                logger.warning(f"{ticker}: Insufficient volume data for liquidity calculation")
                return 54.0  # FALLBACK: Insufficient cached prices (< 5)
            
            # Calculate average daily volume
            volumes = [p.volume for p in cached_prices if p.volume and p.volume > 0]
            if not volumes:
                return 50.0
            
            avg_daily_volume = np.mean(volumes)
            avg_daily_dollar_volume = avg_daily_volume * cached_prices[-1].close_price
            
            # Calculate position as % of daily dollar volume
            position_pct_of_volume = (position_size_dollars / avg_daily_dollar_volume) * 100
            
            # Calculate days to liquidate (assuming we trade 10% of daily volume)
            days_to_liquidate = position_pct_of_volume / 10
            
            # Score based on liquidity
            if days_to_liquidate < 1:
                score = 95  # Can exit in less than a day
            elif days_to_liquidate < 3:
                score = 85  # Can exit in 2-3 days
            elif days_to_liquidate < 5:
                score = 75  # Can exit in a week
            elif days_to_liquidate < 10:
                score = 60  # 1-2 weeks
            elif days_to_liquidate < 20:
                score = 40  # 2-4 weeks
            else:
                score = 20  # Over a month
            
            # Adjust for absolute volume (smaller volume = lower score)
            if avg_daily_dollar_volume < 1_000_000:  # < $1M daily volume
                score = max(20, score - 30)
            elif avg_daily_dollar_volume < 10_000_000:  # < $10M daily volume
                score = max(30, score - 20)
            elif avg_daily_dollar_volume < 50_000_000:  # < $50M daily volume
                score = max(40, score - 10)
            
            logger.debug(f"{ticker}: Liquidity score={score:.0f} (avg_vol=${avg_daily_dollar_volume:,.0f}, position={position_pct_of_volume:.1f}% of daily vol, days_to_liquidate={days_to_liquidate:.1f})")
            
            return round(score, 1)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity for {ticker}: {e}")
            return 58.0  # FALLBACK: Error in calculate_stock_liquidity
    
    async def calculate_portfolio_liquidity(self, positions: List[Any]) -> float:
        """Calculate overall portfolio liquidity score (weighted by position size)"""
        try:
            stocks = [pos for pos in positions if pos.position_type == 'stock']
            
            if not stocks:
                return 73.0  # FALLBACK: No stocks in portfolio
            
            # Calculate liquidity for each stock
            liquidity_tasks = []
            for stock in stocks:
                position_value = stock.quantity * stock.current_price
                liquidity_tasks.append(self.calculate_stock_liquidity(stock.ticker, position_value))
            
            liquidity_scores = await asyncio.gather(*liquidity_tasks)
            
            # Weighted average by position size
            total_value = sum(pos.quantity * pos.current_price for pos in stocks)
            if total_value == 0:
                return 74.0  # FALLBACK: Zero total value
            
            weighted_score = 0.0
            for stock, score in zip(stocks, liquidity_scores):
                position_value = stock.quantity * stock.current_price
                weight = position_value / total_value
                weighted_score += score * weight
            
            logger.info(f"Portfolio liquidity score: {weighted_score:.1f} (from {len(stocks)} stocks)")
            return round(weighted_score, 1)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio liquidity: {e}")
            return 76.0  # FALLBACK: Error in calculate_portfolio_liquidity
    
    async def find_illiquid_positions(self, positions: List[Any], threshold: float = 50.0) -> List[Dict[str, Any]]:
        """Find positions with liquidity concerns (score < threshold)"""
        try:
            stocks = [pos for pos in positions if pos.position_type == 'stock']
            
            illiquid = []
            for stock in stocks:
                position_value = stock.quantity * stock.current_price
                score = await self.calculate_stock_liquidity(stock.ticker, position_value)
                
                if score < threshold:
                    illiquid.append({
                        'ticker': stock.ticker,
                        'liquidity_score': score,
                        'position_value': position_value,
                        'quantity': stock.quantity
                    })
            
            # Sort by score (worst first)
            illiquid.sort(key=lambda x: x['liquidity_score'])
            
            return illiquid
            
        except Exception as e:
            logger.error(f"Error finding illiquid positions: {e}")
            return []

# Global instance
_liquidity_analyzer = None

def get_liquidity_analyzer():
    """Get or create liquidity analyzer instance"""
    global _liquidity_analyzer
    if _liquidity_analyzer is None:
        _liquidity_analyzer = LiquidityAnalyzer()
    return _liquidity_analyzer
