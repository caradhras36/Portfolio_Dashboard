#!/usr/bin/env python3
"""
Real Market Risk Analyzer
Uses actual market data from Polygon.io for sophisticated risk calculations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import math
from scipy import stats
from scipy.stats import norm
import asyncio
import aiohttp
import json

logger = logging.getLogger(__name__)

@dataclass
class RealMarketData:
    """Real market data structure"""
    ticker: str
    current_price: float
    daily_return: float
    volatility: float
    volume: int
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None

@dataclass
class RealRiskMetrics:
    """Real risk metrics using actual market data"""
    # Portfolio-level metrics
    portfolio_value: float
    portfolio_volatility: float
    portfolio_beta: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Stock-specific metrics
    stock_concentration_risk: float
    sector_concentration_risk: float
    liquidity_risk: float
    correlation_risk: float
    
    # Options-specific metrics
    delta_exposure: float
    gamma_risk: float
    theta_decay_risk: float
    vega_risk: float
    time_decay_risk: float
    
    # Advanced metrics
    tail_risk: float
    skewness: float
    kurtosis: float
    information_ratio: float
    treynor_ratio: float

class RealMarketRiskAnalyzer:
    """Advanced risk analysis using real market data from Polygon.io"""
    
    def __init__(self, polygon_api_key: str):
        self.polygon_api_key = polygon_api_key
        self.base_url = "https://api.polygon.io"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_real_market_data(self, ticker: str) -> RealMarketData:
        """Get real market data for a ticker from Polygon.io"""
        try:
            # Get current price and daily data
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/prev"
            params = {
                'apikey': self.polygon_api_key,
                'adjusted': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('results'):
                        result = data['results'][0]
                        current_price = result['c']  # Close price
                        
                        # Get historical data for volatility calculation
                        hist_data = await self.get_historical_data(ticker, days=30)
                        volatility = self.calculate_volatility(hist_data)
                        daily_return = (current_price - result['o']) / result['o'] if result['o'] > 0 else 0
                        
                        return RealMarketData(
                            ticker=ticker,
                            current_price=current_price,
                            daily_return=daily_return,
                            volatility=volatility,
                            volume=result['v'],
                            market_cap=result.get('market_cap'),
                            beta=await self.get_beta(ticker),
                            sector=await self.get_sector(ticker)
                        )
                else:
                    logger.error(f"Error fetching data for {ticker}: {response.status}")
                    return self.get_default_market_data(ticker)
                    
        except Exception as e:
            logger.error(f"Error getting market data for {ticker}: {e}")
            return self.get_default_market_data(ticker)
    
    async def get_historical_data(self, ticker: str, days: int = 30) -> List[float]:
        """Get historical price data for volatility calculation"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {
                'apikey': self.polygon_api_key,
                'adjusted': 'true',
                'sort': 'asc'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('results'):
                        prices = [result['c'] for result in data['results']]
                        returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
                        return returns
                return []
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {e}")
            return []
    
    def calculate_volatility(self, returns: List[float]) -> float:
        """Calculate annualized volatility from returns"""
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * math.sqrt(252)  # Annualized
    
    async def get_beta(self, ticker: str) -> float:
        """Get beta from Polygon.io or calculate it"""
        try:
            # Try to get beta from Polygon.io fundamentals
            url = f"{self.base_url}/v1/meta/symbols/{ticker}/company"
            params = {'apikey': self.polygon_api_key}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('beta', 1.0)
                return 1.0
        except Exception as e:
            logger.error(f"Error getting beta for {ticker}: {e}")
            return 1.0
    
    async def get_sector(self, ticker: str) -> str:
        """Get sector information from Polygon.io"""
        try:
            url = f"{self.base_url}/v1/meta/symbols/{ticker}/company"
            params = {'apikey': self.polygon_api_key}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('industry', 'Unknown')
                return 'Unknown'
        except Exception as e:
            logger.error(f"Error getting sector for {ticker}: {e}")
            return 'Unknown'
    
    def get_default_market_data(self, ticker: str) -> RealMarketData:
        """Fallback market data when API fails"""
        return RealMarketData(
            ticker=ticker,
            current_price=100.0,  # Default price
            daily_return=0.0,
            volatility=0.20,  # 20% default volatility
            volume=1000000,
            market_cap=None,
            beta=1.0,
            sector='Unknown'
        )
    
    async def analyze_portfolio_risk(self, positions: List[Any]) -> RealRiskMetrics:
        """Comprehensive portfolio risk analysis using real market data"""
        if not positions:
            return self.get_empty_risk_metrics()
        
        # Get real market data for all positions
        market_data = {}
        for position in positions:
            market_data[position.ticker] = await self.get_real_market_data(position.ticker)
        
        # Calculate portfolio-level metrics
        portfolio_value = sum(pos.quantity * market_data[pos.ticker].current_price for pos in positions)
        portfolio_volatility = self.calculate_portfolio_volatility(positions, market_data)
        portfolio_beta = self.calculate_portfolio_beta(positions, market_data)
        
        # Calculate risk metrics
        var_95, var_99 = self.calculate_var(portfolio_value, portfolio_volatility)
        cvar_95, cvar_99 = self.calculate_cvar(portfolio_value, portfolio_volatility)
        
        # Calculate advanced metrics
        returns = self.calculate_portfolio_returns(positions, market_data)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(returns)
        
        # Calculate concentration risks
        stock_concentration_risk = self.calculate_stock_concentration_risk(positions, market_data)
        sector_concentration_risk = self.calculate_sector_concentration_risk(positions, market_data)
        liquidity_risk = self.calculate_liquidity_risk(positions, market_data)
        correlation_risk = self.calculate_correlation_risk(positions, market_data)
        
        # Calculate options-specific risks
        delta_exposure = self.calculate_delta_exposure(positions)
        gamma_risk = self.calculate_gamma_risk(positions)
        theta_decay_risk = self.calculate_theta_decay_risk(positions)
        vega_risk = self.calculate_vega_risk(positions)
        time_decay_risk = self.calculate_time_decay_risk(positions)
        
        # Calculate advanced statistical metrics
        tail_risk = self.calculate_tail_risk(returns)
        skewness = self.calculate_skewness(returns)
        kurtosis = self.calculate_kurtosis(returns)
        information_ratio = self.calculate_information_ratio(returns)
        treynor_ratio = self.calculate_treynor_ratio(returns, portfolio_beta)
        
        return RealRiskMetrics(
            portfolio_value=portfolio_value,
            portfolio_volatility=portfolio_volatility,
            portfolio_beta=portfolio_beta,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            stock_concentration_risk=stock_concentration_risk,
            sector_concentration_risk=sector_concentration_risk,
            liquidity_risk=liquidity_risk,
            correlation_risk=correlation_risk,
            delta_exposure=delta_exposure,
            gamma_risk=gamma_risk,
            theta_decay_risk=theta_decay_risk,
            vega_risk=vega_risk,
            time_decay_risk=time_decay_risk,
            tail_risk=tail_risk,
            skewness=skewness,
            kurtosis=kurtosis,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio
        )
    
    def calculate_portfolio_volatility(self, positions: List[Any], market_data: Dict[str, RealMarketData]) -> float:
        """Calculate portfolio volatility using real market data"""
        if not positions:
            return 0.0
        
        # Calculate weighted volatility
        total_value = sum(pos.quantity * market_data[pos.ticker].current_price for pos in positions)
        weighted_vol = 0.0
        
        for pos in positions:
            position_value = pos.quantity * market_data[pos.ticker].current_price
            weight = position_value / total_value
            weighted_vol += market_data[pos.ticker].volatility * weight
        
        return weighted_vol
    
    def calculate_portfolio_beta(self, positions: List[Any], market_data: Dict[str, RealMarketData]) -> float:
        """Calculate portfolio beta using real market data"""
        if not positions:
            return 0.0
        
        total_value = sum(pos.quantity * market_data[pos.ticker].current_price for pos in positions)
        weighted_beta = 0.0
        
        for pos in positions:
            position_value = pos.quantity * market_data[pos.ticker].current_price
            weight = position_value / total_value
            weighted_beta += market_data[pos.ticker].beta * weight
        
        return weighted_beta
    
    def calculate_var(self, portfolio_value: float, volatility: float) -> Tuple[float, float]:
        """Calculate Value at Risk at 95% and 99% confidence levels"""
        var_95 = portfolio_value * volatility * norm.ppf(0.05)
        var_99 = portfolio_value * volatility * norm.ppf(0.01)
        return var_95, var_99
    
    def calculate_cvar(self, portfolio_value: float, volatility: float) -> Tuple[float, float]:
        """Calculate Conditional Value at Risk"""
        cvar_95 = portfolio_value * volatility * norm.pdf(norm.ppf(0.05)) / 0.05
        cvar_99 = portfolio_value * volatility * norm.pdf(norm.ppf(0.01)) / 0.01
        return cvar_95, cvar_99
    
    def calculate_portfolio_returns(self, positions: List[Any], market_data: Dict[str, RealMarketData]) -> List[float]:
        """Calculate portfolio returns using real market data"""
        # This would use historical data to calculate actual returns
        # For now, return a simplified calculation
        returns = []
        for pos in positions:
            if pos.position_type == 'stock':
                returns.append(market_data[pos.ticker].daily_return)
        return returns
    
    def calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return mean_return / std_return if std_return > 0 else 0.0
    
    def calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if not returns or len(returns) < 2:
            return 0.0
        mean_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if downside_returns else 0.0
        return mean_return / downside_std if downside_std > 0 else 0.0
    
    def calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def calculate_stock_concentration_risk(self, positions: List[Any], market_data: Dict[str, RealMarketData]) -> float:
        """Calculate stock concentration risk using Herfindahl-Hirschman Index"""
        if not positions:
            return 0.0
        
        total_value = sum(pos.quantity * market_data[pos.ticker].current_price for pos in positions)
        weights = [(pos.quantity * market_data[pos.ticker].current_price) / total_value for pos in positions]
        hhi = sum(w**2 for w in weights)
        return hhi
    
    def calculate_sector_concentration_risk(self, positions: List[Any], market_data: Dict[str, RealMarketData]) -> float:
        """Calculate sector concentration risk"""
        if not positions:
            return 0.0
        
        sector_weights = {}
        total_value = sum(pos.quantity * market_data[pos.ticker].current_price for pos in positions)
        
        for pos in positions:
            sector = market_data[pos.ticker].sector
            position_value = pos.quantity * market_data[pos.ticker].current_price
            weight = position_value / total_value
            
            if sector in sector_weights:
                sector_weights[sector] += weight
            else:
                sector_weights[sector] = weight
        
        # Calculate HHI for sectors
        hhi = sum(w**2 for w in sector_weights.values())
        return hhi
    
    def calculate_liquidity_risk(self, positions: List[Any], market_data: Dict[str, RealMarketData]) -> float:
        """Calculate liquidity risk based on volume and position size"""
        if not positions:
            return 0.0
        
        liquidity_scores = []
        for pos in positions:
            position_value = pos.quantity * market_data[pos.ticker].current_price
            daily_volume_value = market_data[pos.ticker].volume * market_data[pos.ticker].current_price
            liquidity_ratio = position_value / daily_volume_value if daily_volume_value > 0 else 1.0
            liquidity_scores.append(min(1.0, liquidity_ratio))
        
        return np.mean(liquidity_scores)
    
    def calculate_correlation_risk(self, positions: List[Any], market_data: Dict[str, RealMarketData]) -> float:
        """Calculate correlation risk (simplified)"""
        # This would require historical correlation data
        # For now, return a simplified estimate
        return 0.3  # 30% average correlation estimate
    
    def calculate_delta_exposure(self, positions: List[Any]) -> float:
        """Calculate total delta exposure"""
        delta_exposure = 0.0
        for pos in positions:
            if pos.position_type in ['call', 'put'] and pos.delta:
                delta_exposure += pos.delta * pos.quantity
            elif pos.position_type == 'stock':
                delta_exposure += pos.quantity  # Stocks have delta = 1
        return delta_exposure
    
    def calculate_gamma_risk(self, positions: List[Any]) -> float:
        """Calculate gamma risk"""
        gamma_risk = 0.0
        for pos in positions:
            if pos.position_type in ['call', 'put'] and pos.gamma:
                gamma_risk += pos.gamma * pos.quantity
        return gamma_risk
    
    def calculate_theta_decay_risk(self, positions: List[Any]) -> float:
        """Calculate theta decay risk"""
        theta_risk = 0.0
        for pos in positions:
            if pos.position_type in ['call', 'put'] and pos.theta:
                theta_risk += pos.theta * pos.quantity
        return theta_risk
    
    def calculate_vega_risk(self, positions: List[Any]) -> float:
        """Calculate vega risk"""
        vega_risk = 0.0
        for pos in positions:
            if pos.position_type in ['call', 'put'] and pos.vega:
                vega_risk += pos.vega * pos.quantity
        return vega_risk
    
    def calculate_time_decay_risk(self, positions: List[Any]) -> float:
        """Calculate time decay risk based on days to expiration"""
        if not positions:
            return 0.0
        
        options = [pos for pos in positions if pos.position_type in ['call', 'put']]
        if not options:
            return 0.0
        
        # Calculate weighted average days to expiration
        total_value = sum(pos.quantity * pos.current_price for pos in options)
        weighted_days = 0.0
        
        for pos in options:
            # Check if time_to_expiration exists, otherwise calculate from expiration_date
            if hasattr(pos, 'time_to_expiration') and pos.time_to_expiration:
                days_to_exp = pos.time_to_expiration
            elif hasattr(pos, 'expiration_date') and pos.expiration_date:
                try:
                    from datetime import datetime
                    exp_date = datetime.fromisoformat(pos.expiration_date.replace('Z', '+00:00'))
                    days_to_exp = (exp_date - datetime.now()).days
                except:
                    days_to_exp = 30  # Default
            else:
                days_to_exp = 30  # Default
            
            position_value = pos.quantity * pos.current_price
            weight = position_value / total_value if total_value > 0 else 0
            weighted_days += days_to_exp * weight
        
        # Risk increases as expiration approaches
        return 1.0 / max(1, weighted_days / 30)  # Normalize to 30 days
    
    def calculate_tail_risk(self, returns: List[float]) -> float:
        """Calculate tail risk (extreme negative returns)"""
        if not returns or len(returns) < 10:
            return 0.0
        return np.percentile(returns, 5)  # 5th percentile
    
    def calculate_skewness(self, returns: List[float]) -> float:
        """Calculate skewness of returns"""
        if not returns or len(returns) < 3:
            return 0.0
        return stats.skew(returns)
    
    def calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis of returns"""
        if not returns or len(returns) < 4:
            return 0.0
        return stats.kurtosis(returns)
    
    def calculate_information_ratio(self, returns: List[float]) -> float:
        """Calculate information ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        # Assuming benchmark return of 0 for simplicity
        excess_returns = returns
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
    
    def calculate_treynor_ratio(self, returns: List[float], beta: float) -> float:
        """Calculate Treynor ratio"""
        if not returns or beta == 0:
            return 0.0
        return np.mean(returns) / beta
    
    def get_empty_risk_metrics(self) -> RealRiskMetrics:
        """Return empty risk metrics when no positions"""
        return RealRiskMetrics(
            portfolio_value=0.0,
            portfolio_volatility=0.0,
            portfolio_beta=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            stock_concentration_risk=0.0,
            sector_concentration_risk=0.0,
            liquidity_risk=0.0,
            correlation_risk=0.0,
            delta_exposure=0.0,
            gamma_risk=0.0,
            theta_decay_risk=0.0,
            vega_risk=0.0,
            time_decay_risk=0.0,
            tail_risk=0.0,
            skewness=0.0,
            kurtosis=0.0,
            information_ratio=0.0,
            treynor_ratio=0.0
        )

# Example usage
if __name__ == "__main__":
    async def main():
        analyzer = RealMarketRiskAnalyzer("your_polygon_api_key")
        async with analyzer:
            # Example usage
            pass
    
    asyncio.run(main())
