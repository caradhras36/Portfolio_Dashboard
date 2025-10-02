#!/usr/bin/env python3
"""
Fast Risk Monitoring System
Provides real-time risk alerts and fast risk calculations with progressive loading
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

@dataclass
class RiskAlert:
    """Real-time risk alert"""
    level: str  # 'critical', 'warning', 'info'
    category: str  # 'concentration', 'volatility', 'liquidity', 'greeks', 'expiration'
    message: str
    value: float
    threshold: float
    timestamp: datetime
    affected_positions: List[str]
    recommendation: str

@dataclass
class VolatileStock:
    """Individual stock with beta information"""
    ticker: str
    beta: float
    value: float
    weight: float  # % of portfolio

@dataclass
class StockRiskSummary:
    """Stock-specific risk summary"""
    stock_count: int
    stock_value: float
    stock_pnl_pct: float
    concentration_risk_score: int
    max_position_pct: float
    sector_concentration: Dict[str, float]
    unique_tickers: int
    portfolio_beta: float = 1.0
    portfolio_volatility: float = 0.20
    liquidity_score: int = 50
    most_volatile_stocks: List = field(default_factory=list)  # Top 5 most volatile

@dataclass
class OptionsRiskSummary:
    """Options-specific risk summary"""
    options_count: int
    options_value: float
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    expiring_7d_count: int
    expiring_30d_count: int
    time_decay_risk_score: int
    avg_days_to_expiry: float

@dataclass
class FastRiskMetrics:
    """Fast-loading risk metrics with essential data"""
    # Critical metrics (load first)
    portfolio_value: float
    total_pnl: float
    total_pnl_pct: float
    max_position_risk: float
    
    # Risk levels (0-100 score)
    overall_risk_score: int
    concentration_risk_score: int
    volatility_risk_score: int
    liquidity_risk_score: int
    
    # Separate stock and options risk
    stock_risk: StockRiskSummary
    options_risk: OptionsRiskSummary
    
    # Quick counts
    total_positions: int
    positions_at_risk: int
    expiring_soon: int
    
    # Alerts
    active_alerts: List[RiskAlert]
    
    # Timestamp
    calculated_at: datetime
    calculation_time_ms: float

@dataclass
class DetailedRiskMetrics:
    """Detailed risk metrics (load second)"""
    # Greeks exposure
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    
    # Portfolio metrics
    portfolio_beta: float
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Risk measurements
    var_95: float
    var_99: float
    expected_shortfall: float
    
    # Concentration analysis
    top_5_concentration: float
    sector_concentration: Dict[str, float]
    
    # Time decay analysis
    theta_decay_30d: float
    expiring_7d_value: float
    expiring_30d_value: float
    
    # Liquidity analysis
    avg_liquidity_score: float
    illiquid_positions: List[str]

class FastRiskMonitor:
    """Fast risk monitoring with progressive loading and real-time alerts"""
    
    def __init__(self):
        self._risk_cache = {}  # In-memory cache for risk calculations
        self._cache_ttl = 60  # 1 minute cache
        
        # Risk thresholds for alerts
        self.thresholds = {
            'concentration_warning': 0.15,  # 15% in single position
            'concentration_critical': 0.25,  # 25% in single position
            'volatility_warning': 0.30,     # 30% portfolio volatility
            'volatility_critical': 0.50,    # 50% portfolio volatility
            'delta_warning': 0.3,           # 30% delta exposure
            'delta_critical': 0.5,          # 50% delta exposure
            'gamma_warning': 0.1,           # Gamma threshold
            'gamma_critical': 0.2,          # Critical gamma
            'theta_warning': -500,          # $500/day theta decay
            'theta_critical': -1000,        # $1000/day theta decay
            'expiration_warning': 7,        # 7 days to expiration
            'expiration_critical': 3,       # 3 days to expiration
            'liquidity_warning': 0.05,      # 5% of daily volume
            'liquidity_critical': 0.15,     # 15% of daily volume
        }
        
        # Cache for risk calculations and beta values
        self._cache = {}
        self._cache_ttl = 30  # 30 seconds cache for risk metrics
        # Beta values cached for 24 hours (86400 seconds)
        self._most_volatile_stocks = []  # Track volatile stocks between calls
        self._portfolio_liquidity_score = 75.0  # Track liquidity score between calls
        
    async def get_fast_risk_metrics(self, positions: List[Any], force_recalc: bool = False) -> FastRiskMetrics:
        """Get fast-loading essential risk metrics - ALL CALCULATIONS ARE O(n)"""
        start_time = time.time()
        
        try:
            logger.info(f"🔢 Starting risk calculation for {len(positions)} positions...")
            
            # Separate stocks and options (single pass - fast)
            calc_start = time.time()
            stocks = [pos for pos in positions if pos.position_type == 'stock']
            options = [pos for pos in positions if pos.position_type in ['call', 'put']]
            logger.info(f"✅ Separated positions in {(time.time()-calc_start)*1000:.2f}ms: {len(stocks)} stocks, {len(options)} options")
            
            # Calculate essential metrics quickly (O(n) - should be <1ms for 100 positions)
            calc_start = time.time()
            portfolio_value = self._calculate_portfolio_value(positions)
            # Calculate total cost with options multiplier
            total_cost = sum(
                pos.quantity * pos.entry_price * 100 if pos.position_type in ['call', 'put'] else pos.quantity * pos.entry_price
                for pos in positions
            )
            total_pnl = portfolio_value - total_cost
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
            logger.info(f"✅ Basic metrics in {(time.time()-calc_start)*1000:.2f}ms")
            
            # Calculate position risks (with correct multipliers)
            position_values = [
                (pos.ticker, 
                 pos.quantity * pos.current_price * 100 if pos.position_type in ['call', 'put'] else pos.quantity * pos.current_price,
                 pos) 
                for pos in positions
            ]
            max_position = max(position_values, key=lambda x: x[1]) if position_values else (None, 0, None)
            max_position_risk = (max_position[1] / portfolio_value * 100) if portfolio_value > 0 else 0
            
            # Calculate risk scores (0-100)
            concentration_score = self._calculate_concentration_score(positions, portfolio_value)
            volatility_score = self._calculate_volatility_score(positions)
            liquidity_score = self._calculate_liquidity_score(positions)
            
            overall_risk_score = int((concentration_score + volatility_score + liquidity_score) / 3)
            
            # Calculate stock-specific risk (async for beta calculation)
            calc_start = time.time()
            stock_risk = await self._calculate_stock_risk_summary(stocks, portfolio_value, force_recalc=force_recalc)
            logger.info(f"✅ Stock risk in {(time.time()-calc_start)*1000:.2f}ms")
            
            # Calculate options-specific risk
            calc_start = time.time()
            options_risk = self._calculate_options_risk_summary(options)
            logger.info(f"✅ Options risk in {(time.time()-calc_start)*1000:.2f}ms")
            
            # Count positions at risk
            positions_at_risk = self._count_positions_at_risk(positions, portfolio_value)
            
            # Count expiring positions
            expiring_soon = self._count_expiring_soon(positions, days=7)
            
            # Generate real-time alerts
            calc_start = time.time()
            alerts = await self._generate_real_time_alerts(positions, portfolio_value)
            logger.info(f"✅ Alerts generated in {(time.time()-calc_start)*1000:.2f}ms ({len(alerts)} alerts)")
            
            calculation_time = (time.time() - start_time) * 1000  # Convert to ms
            logger.info(f"🎯 TOTAL RISK CALCULATION TIME: {calculation_time:.2f}ms")
            
            return FastRiskMetrics(
                portfolio_value=portfolio_value,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                max_position_risk=max_position_risk,
                overall_risk_score=overall_risk_score,
                concentration_risk_score=concentration_score,
                volatility_risk_score=volatility_score,
                liquidity_risk_score=liquidity_score,
                stock_risk=stock_risk,
                options_risk=options_risk,
                total_positions=len(positions),
                positions_at_risk=positions_at_risk,
                expiring_soon=expiring_soon,
                active_alerts=alerts,
                calculated_at=datetime.now(),
                calculation_time_ms=calculation_time
            )
            
        except Exception as e:
            logger.error(f"Error calculating fast risk metrics: {e}")
            raise
    
    async def get_detailed_risk_metrics(self, positions: List[Any]) -> DetailedRiskMetrics:
        """Get detailed risk metrics (load after fast metrics)"""
        try:
            import time
            start_time = time.time()
            
            # Check cache first
            positions_hash = str(hash(tuple((pos.ticker, pos.quantity, pos.current_price) for pos in positions)))
            if positions_hash in self._risk_cache:
                cache_time, cached_data = self._risk_cache[positions_hash]
                if time.time() - cache_time < self._cache_ttl:
                    logger.info("⚡ Using cached detailed risk metrics")
                    return cached_data
                else:
                    del self._risk_cache[positions_hash]
            
            logger.info(f"🔄 Starting detailed risk calculation for {len(positions)} positions...")
            # Calculate Greeks exposure
            options = [pos for pos in positions if pos.position_type in ['call', 'put']]
            total_delta = sum(getattr(pos, 'delta', 0) * pos.quantity for pos in options)
            total_gamma = sum(getattr(pos, 'gamma', 0) * pos.quantity for pos in options)
            total_theta = sum(getattr(pos, 'theta', 0) * pos.quantity * 100 for pos in options)  # Convert to dollars
            total_vega = sum(getattr(pos, 'vega', 0) * pos.quantity for pos in options)
            
            # Calculate portfolio metrics (beta, volatility, volatile stocks, liquidity) - reuse stock list
            stocks = [pos for pos in positions if pos.position_type == 'stock']
            logger.info(f"📊 Calculating portfolio metrics for {len(stocks)} stocks...")
            metrics_start = time.time()
            portfolio_beta, portfolio_volatility, most_volatile, liquidity_score = await self._calculate_portfolio_metrics(stocks, force_recalc=False)
            logger.info(f"⏱️ Portfolio metrics calculated in {time.time() - metrics_start:.2f}s")
            
            # Store volatile stocks and liquidity for later use
            self._most_volatile_stocks = most_volatile
            self._portfolio_liquidity_score = liquidity_score
            self._portfolio_volatility = portfolio_volatility  # Store for max drawdown calculation
            sharpe_ratio = self._calculate_sharpe_ratio(positions)
            max_drawdown = self._calculate_max_drawdown(positions)
            
            # Calculate VaR (using correct portfolio value with options multiplier)
            portfolio_value = self._calculate_portfolio_value(positions)
            
            # VaR calculations - using negative values for losses
            from scipy.stats import norm
            import math
            
            # Convert annual volatility to daily volatility (assuming 252 trading days)
            daily_volatility = portfolio_volatility / math.sqrt(252)
            
            var_95 = abs(portfolio_value * daily_volatility * norm.ppf(0.05))  # 95% confidence (loss)
            var_99 = abs(portfolio_value * daily_volatility * norm.ppf(0.01))  # 99% confidence (loss)
            
            # Expected Shortfall (CVaR) - proper calculation
            z_95 = norm.ppf(0.05)
            expected_shortfall = abs(portfolio_value * daily_volatility * norm.pdf(z_95) / 0.05)
            
            # Concentration analysis
            top_5_concentration = self._calculate_top_5_concentration(positions)
            sector_concentration = self._calculate_sector_concentration(positions)
            
            # Time decay analysis
            theta_decay_30d = total_theta * 30  # 30-day theta decay
            expiring_7d_value = self._calculate_expiring_value(positions, 7)
            expiring_30d_value = self._calculate_expiring_value(positions, 30, 7)  # 8-30 days (excludes 7d)
            
            # Liquidity analysis - use the already calculated score (converted to 0-1 scale)
            avg_liquidity_score = liquidity_score / 100  # Use the calculated value from line 253
            illiquid_positions = self._find_illiquid_positions(positions)
            
            result = DetailedRiskMetrics(
                total_delta=total_delta,
                total_gamma=total_gamma,
                total_theta=total_theta,
                total_vega=total_vega,
                portfolio_beta=portfolio_beta,
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                top_5_concentration=top_5_concentration,
                sector_concentration=sector_concentration,
                theta_decay_30d=theta_decay_30d,
                expiring_7d_value=expiring_7d_value,
                expiring_30d_value=expiring_30d_value,
                avg_liquidity_score=avg_liquidity_score,
                illiquid_positions=illiquid_positions
            )
            
            # Cache the result
            self._risk_cache[positions_hash] = (time.time(), result)
            
            total_time = time.time() - start_time
            logger.info(f"✅ Detailed risk metrics completed in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating detailed risk metrics: {e}")
            raise
    
    async def _generate_real_time_alerts(self, positions: List[Any], portfolio_value: float) -> List[RiskAlert]:
        """Generate real-time risk alerts - OPTIMIZED for speed"""
        alerts = []
        
        if not positions or portfolio_value == 0:
            return alerts
        
        try:
            # Pre-calculate datetime once
            now = datetime.now()
            
            # Separate positions types once
            options = [pos for pos in positions if pos.position_type in ['call', 'put']]
            
            # SINGLE PASS through all positions for concentration and expiration
            total_theta = 0.0
            theta_positions = []
            
            for pos in positions:
                # Concentration check
                position_value = pos.quantity * pos.current_price
                concentration = position_value / portfolio_value
                
                if concentration > self.thresholds['concentration_critical']:
                    alerts.append(RiskAlert(
                        level='critical',
                        category='concentration',
                        message=f'{pos.ticker} represents {concentration*100:.1f}% of portfolio',
                        value=concentration,
                        threshold=self.thresholds['concentration_critical'],
                        timestamp=now,
                        affected_positions=[pos.ticker],
                        recommendation=f'Consider reducing {pos.ticker} position to below 25%'
                    ))
                elif concentration > self.thresholds['concentration_warning']:
                    alerts.append(RiskAlert(
                        level='warning',
                        category='concentration',
                        message=f'{pos.ticker} represents {concentration*100:.1f}% of portfolio',
                        value=concentration,
                        threshold=self.thresholds['concentration_warning'],
                        timestamp=now,
                        affected_positions=[pos.ticker],
                        recommendation=f'Monitor {pos.ticker} position size'
                    ))
                
                # For options: check expiration and accumulate theta
                if pos.position_type in ['call', 'put']:
                    # Get days to expiration
                    days_to_exp = getattr(pos, 'time_to_expiration', None)
                    
                    # Expiration alerts
                    if days_to_exp is not None:
                        if days_to_exp <= self.thresholds['expiration_critical']:
                            alerts.append(RiskAlert(
                                level='critical',
                                category='expiration',
                                message=f'{pos.ticker} expires in {days_to_exp} days',
                                value=days_to_exp,
                                threshold=self.thresholds['expiration_critical'],
                                timestamp=now,
                                affected_positions=[pos.ticker],
                                recommendation='Take action NOW - close or roll'
                            ))
                        elif days_to_exp <= self.thresholds['expiration_warning']:
                            alerts.append(RiskAlert(
                                level='warning',
                                category='expiration',
                                message=f'{pos.ticker} expires in {days_to_exp} days',
                                value=days_to_exp,
                                threshold=self.thresholds['expiration_warning'],
                                timestamp=now,
                                affected_positions=[pos.ticker],
                                recommendation='Consider rolling or closing soon'
                            ))
                    
                    # Accumulate theta
                    theta = getattr(pos, 'theta', 0)
                    if theta and theta < 0:
                        total_theta += theta * pos.quantity * 100
                        theta_positions.append(pos.ticker)
            
            # Check theta risk (only if we have theta decay)
            if total_theta < self.thresholds['theta_critical']:
                alerts.append(RiskAlert(
                    level='critical',
                    category='greeks',
                    message=f'High theta decay: ${abs(total_theta):.0f}/day',
                    value=total_theta,
                    threshold=self.thresholds['theta_critical'],
                    timestamp=now,
                    affected_positions=theta_positions[:5],  # Limit to 5 for brevity
                    recommendation='Reduce short-dated options exposure'
                ))
            elif total_theta < self.thresholds['theta_warning']:
                alerts.append(RiskAlert(
                    level='warning',
                    category='greeks',
                    message=f'Moderate theta decay: ${abs(total_theta):.0f}/day',
                    value=total_theta,
                    threshold=self.thresholds['theta_warning'],
                    timestamp=now,
                    affected_positions=theta_positions[:5],
                    recommendation='Monitor time decay closely'
                ))
            
            # Sort alerts by level (critical first) - fast sort
            alert_priority = {'critical': 0, 'warning': 1, 'info': 2}
            alerts.sort(key=lambda x: alert_priority.get(x.level, 3))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return []
    
    def _calculate_concentration_score(self, positions: List[Any], portfolio_value: float) -> int:
        """Calculate concentration risk score (0-100, higher = more risk) - OPTIMIZED"""
        if not positions or portfolio_value == 0:
            return 0
        
        # Fast HHI calculation in single pass
        hhi = sum((pos.quantity * pos.current_price / portfolio_value) ** 2 for pos in positions)
        
        # Convert to 0-100 scale (1.0 = 100, 0 = 0)
        return min(100, int(hhi * 100))
    
    def _calculate_volatility_score(self, positions: List[Any]) -> int:
        """Calculate volatility risk score (0-100, higher = more risk)"""
        # Simplified volatility score based on position diversity
        if not positions:
            return 0
        
        options_ratio = len([p for p in positions if p.position_type in ['call', 'put']]) / len(positions)
        
        # Higher options ratio = higher volatility risk
        return min(100, int(options_ratio * 100))
    
    def _calculate_liquidity_score(self, positions: List[Any]) -> int:
        """Calculate liquidity risk score (0-100, higher = more risk)"""
        # Simplified liquidity score
        # In practice, would check actual volume data
        return 30  # Placeholder
    
    def _count_positions_at_risk(self, positions: List[Any], portfolio_value: float) -> int:
        """Count positions that exceed risk thresholds - OPTIMIZED"""
        if portfolio_value == 0:
            return 0
        
        # Fast count using generator expression
        threshold = self.thresholds['concentration_warning']
        return sum(1 for pos in positions 
                  if (pos.quantity * pos.current_price / portfolio_value) > threshold)
    
    def _count_expiring_soon(self, positions: List[Any], days: int) -> int:
        """Count options expiring within specified days - OPTIMIZED"""
        count = 0
        now = datetime.now()  # Calculate once
        
        for pos in positions:
            if pos.position_type not in ['call', 'put']:
                continue
                
            days_to_exp = getattr(pos, 'time_to_expiration', None)
            if days_to_exp is None and hasattr(pos, 'expiration_date') and pos.expiration_date:
                try:
                    exp_date = datetime.fromisoformat(pos.expiration_date.replace('Z', '+00:00'))
                    days_to_exp = (exp_date - now).days
                except:
                    continue
            
            if days_to_exp is not None and days_to_exp <= days:
                count += 1
        
        return count
    
    async def _calculate_portfolio_metrics(self, positions: List[Any], force_recalc: bool = False) -> tuple[float, float, List[VolatileStock], float]:
        """Calculate portfolio beta, volatility, volatile stocks, AND liquidity - all in one pass"""
        
        import time
        start_time = time.time()
        
        stocks = [pos for pos in positions if pos.position_type == 'stock']
        
        if not stocks:
            return 1.0, 0.20, [], 70.0
        
        try:
            # Import analyzers
            from beta_calculator import get_beta_calculator
            from liquidity_analyzer import get_liquidity_analyzer
            from price_cache import get_price_cache
            
            calculator = get_beta_calculator()
            liquidity_analyzer = get_liquidity_analyzer()
            price_cache = get_price_cache()
            
            if not calculator:
                return 1.0, 0.20, [], 70.0
            
            # FIRST: Check if we have TODAY's pre-calculated metrics in cache (unless forced)
            from metrics_cache import get_metrics_cache
            from datetime import datetime, timedelta
            
            metrics_cache = get_metrics_cache()
            tickers = [stock.ticker for stock in stocks]
            
            # Try to get cached metrics first (instant!) - skip if force_recalc
            cache_start = time.time()
            cached_metrics = {} if force_recalc else await metrics_cache.get_metrics_batch(tickers)
            logger.info(f"⏱️ Cache lookup took {time.time() - cache_start:.3f}s")
            
            # Separate stocks with cached metrics vs those that need calculation
            stocks_with_cache = []
            stocks_need_calc = []
            
            for stock in stocks:
                if stock.ticker in cached_metrics:
                    stocks_with_cache.append((stock, cached_metrics[stock.ticker]))
                else:
                    stocks_need_calc.append(stock)
            
            logger.info(f"📊 Cache hit: {len(stocks_with_cache)}/{len(stocks)} stocks, need to calculate: {len(stocks_need_calc)}")
            
            # If we have ALL stocks cached, skip expensive calculations entirely
            if not stocks_need_calc:
                logger.info("⚡ All metrics cached - skipping historical data fetch!")
                
                # Calculate weighted averages directly from cached data
                total_value = sum(pos.quantity * pos.current_price for pos in stocks)
                if total_value == 0:
                    return 1.0, 0.20, [], 70.0
                
                weighted_beta = 0.0
                weighted_volatility = 0.0
                weighted_liquidity = 0.0
                stock_betas = []
                
                for pos, (beta, volatility, liquidity) in stocks_with_cache:
                    position_value = pos.quantity * pos.current_price
                    weight = position_value / total_value
                    
                    # Cap individual stock volatility at 30% to prevent outliers from skewing portfolio
                    capped_volatility = min(volatility, 0.30)
                    
                    weighted_beta += beta * weight
                    weighted_volatility += capped_volatility * weight
                    weighted_liquidity += liquidity * weight
                    
                    stock_betas.append(VolatileStock(
                        ticker=pos.ticker,
                        beta=beta,
                        value=position_value,
                        weight=weight * 100
                    ))
                
                # Get top 5 most volatile
                most_volatile = sorted(stock_betas, key=lambda x: x.beta, reverse=True)[:5]
                
                portfolio_beta = round(weighted_beta, 2)
                portfolio_volatility = round(weighted_volatility, 2)
                liquidity_score = round(weighted_liquidity, 1)
                
                logger.info(f"Portfolio: beta={portfolio_beta:.2f}, volatility={portfolio_volatility:.1%}, liquidity={liquidity_score:.1f}/100 (from {len(stocks)} stocks)")
                logger.info(f"⏱️ Portfolio metrics (all cached) completed in {time.time() - start_time:.3f}s")
                
                return portfolio_beta, portfolio_volatility, most_volatile, liquidity_score
            
            # Only fetch historical data for stocks that need calculation
            if stocks_need_calc:
                end_date = (datetime.now() - timedelta(days=1)).date()
                start_date = end_date - timedelta(days=365)
                
                # Batch fetch all needed tickers in single query (FAST!)
                tickers_to_fetch = [stock.ticker for stock in stocks_need_calc]
                price_data_cache = await price_cache.get_cached_prices_batch(tickers_to_fetch, start_date, end_date)
                
                total_records = sum(len(prices) for prices in price_data_cache.values())
                logger.info(f"✅ Fetched {total_records} price records for {len(tickers_to_fetch)} stocks in batch query")
            else:
                price_data_cache = {}
                logger.info(f"⚡ All metrics cached - skipping historical data fetch!")
            
            # Calculate metrics only for stocks that need it
            all_metrics = []
            newly_calculated = {}
            
            # Add cached metrics
            for stock, (beta, vol, liq) in stocks_with_cache:
                all_metrics.append((beta, vol, liq))
            
            # Calculate missing metrics
            if stocks_need_calc:
                metrics_tasks = [
                    self._calculate_stock_metrics_with_cache(stock, price_data_cache.get(stock.ticker, []))
                    for stock in stocks_need_calc
                ]
                calculated_metrics = await asyncio.gather(*metrics_tasks)
                
                # Store for saving to cache
                for stock, metrics in zip(stocks_need_calc, calculated_metrics):
                    newly_calculated[stock.ticker] = metrics
                    all_metrics.append(metrics)
                
                # Save newly calculated metrics to cache (fire and forget)
                if newly_calculated:
                    asyncio.create_task(metrics_cache.save_metrics_batch(newly_calculated))
            
            # Merge the stocks list back in the correct order
            all_stocks = stocks_with_cache + [(stock, None) for stock in stocks_need_calc]
            all_stocks = [s[0] for s in all_stocks]  # Extract just the stock objects
            
            # Calculate weighted averages
            total_value = sum(pos.quantity * pos.current_price for pos in stocks)
            if total_value == 0:
                return 1.0, 0.20, [], 70.0
            
            weighted_beta = 0.0
            weighted_volatility = 0.0
            weighted_liquidity = 0.0
            stock_betas = []
            
            for pos, (beta, volatility, liquidity) in zip(all_stocks, all_metrics):
                position_value = pos.quantity * pos.current_price
                weight = position_value / total_value
                
                # Cap individual stock volatility at 30% to prevent outliers from skewing portfolio
                capped_volatility = min(volatility, 0.30)
                
                weighted_beta += beta * weight
                weighted_volatility += capped_volatility * weight
                weighted_liquidity += liquidity * weight
                
                stock_betas.append(VolatileStock(
                    ticker=pos.ticker,
                    beta=beta,
                    value=position_value,
                    weight=weight * 100
                ))
            
            # Get top 5 most volatile
            most_volatile = sorted(stock_betas, key=lambda x: x.beta, reverse=True)[:5]
            
            portfolio_beta = round(weighted_beta, 2)
            portfolio_volatility = round(weighted_volatility, 2)
            liquidity_score = round(weighted_liquidity, 1)
            
            logger.info(f"Portfolio: beta={portfolio_beta:.2f}, volatility={portfolio_volatility:.1%}, liquidity={liquidity_score:.1f}/100 (from {len(stocks)} stocks)")
            
            return portfolio_beta, portfolio_volatility, most_volatile, liquidity_score
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return 1.0, 0.20, [], 77.0  # FALLBACK: Error in _calculate_portfolio_metrics
    
    async def _calculate_stock_metrics_with_cache(self, stock, cached_prices: List) -> tuple[float, float, float]:
        """Calculate beta, volatility, and liquidity for one stock using pre-fetched cache"""
        try:
            from beta_calculator import get_beta_calculator
            
            calculator = get_beta_calculator()
            
            # Check if we have enough data
            cached_count = len(cached_prices) if cached_prices else 0
            
            if cached_count < 20:
                logger.debug(f"{stock.ticker}: Only {cached_count} cached prices - using fallback")
            
            # If we have cached prices, use them; otherwise fetch
            if cached_prices and len(cached_prices) >= 20:
                # Calculate returns from cached prices
                all_prices = [p.close_price for p in cached_prices]
                returns = []
                for i in range(1, len(all_prices)):
                    ret = (all_prices[i] - all_prices[i-1]) / all_prices[i-1]
                    returns.append(ret)
                
                # Get SPY returns for beta
                from datetime import datetime, timedelta
                end_date = (datetime.now() - timedelta(days=1)).date()
                start_date = end_date - timedelta(days=365)
                
                benchmark_returns = await calculator._get_daily_returns('SPY')
                
                if len(returns) >= 20 and len(benchmark_returns) >= 20:
                    min_len = min(len(returns), len(benchmark_returns))
                    stock_returns = returns[:min_len]
                    bench_returns = benchmark_returns[:min_len]
                    
                    # Calculate beta
                    import numpy as np
                    covariance = np.cov(stock_returns, bench_returns)[0][1]
                    variance = np.var(bench_returns)
                    beta = covariance / variance if variance > 0 else 1.0
                    
                    # Calculate volatility
                    daily_vol = np.std(stock_returns)
                    annualized_vol = daily_vol * np.sqrt(252)
                    
                    # Calculate liquidity using ALL available historical data (full year)
                    # Use entire year of volume data for accurate average
                    prices_with_volume = [p for p in cached_prices if p.volume and p.volume > 0]
                    
                    # Debug: Check volume data
                    logger.debug(f"{stock.ticker}: Full year has {len(cached_prices)} prices, {len(prices_with_volume)} with volume > 0")
                    
                    if prices_with_volume:
                        volumes = [p.volume for p in prices_with_volume]
                        avg_daily_volume = np.mean(volumes)
                        avg_daily_dollar_volume = avg_daily_volume * prices_with_volume[-1].close_price
                        
                        logger.debug(f"{stock.ticker}: avg_volume={avg_daily_volume:,.0f}, avg_dollar_volume=${avg_daily_dollar_volume:,.0f}")
                        
                        position_value = stock.quantity * stock.current_price
                        position_pct = (position_value / avg_daily_dollar_volume) * 100 if avg_daily_dollar_volume > 0 else 100
                        days_to_liquidate = position_pct / 10
                        
                        # Score based on liquidity
                        if days_to_liquidate < 1:
                            liquidity = 95
                        elif days_to_liquidate < 3:
                            liquidity = 85
                        elif days_to_liquidate < 5:
                            liquidity = 75
                        elif days_to_liquidate < 10:
                            liquidity = 60
                        elif days_to_liquidate < 20:
                            liquidity = 40
                        else:
                            liquidity = 20
                        
                        # Adjust for absolute volume
                        if avg_daily_dollar_volume < 1_000_000:
                            liquidity = max(20, liquidity - 30)
                        elif avg_daily_dollar_volume < 10_000_000:
                            liquidity = max(30, liquidity - 20)
                        elif avg_daily_dollar_volume < 50_000_000:
                            liquidity = max(40, liquidity - 10)
                    else:
                        liquidity = 52.0  # FALLBACK: No volume in price data
                    
                    return round(beta, 2), round(annualized_vol, 2), round(liquidity, 1)
            
            # Fallback to calculator methods if cache insufficient
            beta, volatility = await calculator.calculate_beta(stock.ticker)
            liquidity = 71.0  # FALLBACK: Insufficient cached prices (< 20 records)
            
            return beta, volatility, liquidity
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {stock.ticker}: {e}")
            return 1.0, 0.25, 50.0
    
    async def _calculate_portfolio_beta_and_volatility(self, positions: List[Any]) -> tuple[float, float, List[VolatileStock]]:
        """Calculate portfolio beta, volatility, and identify most volatile stocks"""
        stocks = [pos for pos in positions if pos.position_type == 'stock']
        
        if not stocks:
            return 1.0, 0.20, []
        
        try:
            # Use the beta calculator to get real calculated betas and volatilities
            from beta_calculator import get_beta_calculator
            
            calculator = get_beta_calculator()
            if calculator:
                # Calculate beta and volatility for each stock
                metrics_tasks = [calculator.calculate_beta(stock.ticker) for stock in stocks]
                metrics = await asyncio.gather(*metrics_tasks)
                
                # Calculate total stock value
                total_value = sum(pos.quantity * pos.current_price for pos in stocks)
                
                # Create list of stocks with their betas
                stock_betas = []
                weighted_beta = 0.0
                weighted_volatility = 0.0
                
                for pos, (beta, volatility) in zip(stocks, metrics):
                    position_value = pos.quantity * pos.current_price
                    weight = position_value / total_value if total_value > 0 else 0
                    weighted_beta += beta * weight
                    weighted_volatility += volatility * weight
                    
                    stock_betas.append(VolatileStock(
                        ticker=pos.ticker,
                        beta=beta,
                        value=position_value,
                        weight=weight * 100  # Convert to percentage
                    ))
                
                # Sort by beta and get top 5 most volatile
                most_volatile = sorted(stock_betas, key=lambda x: x.beta, reverse=True)[:5]
                
                portfolio_beta = round(weighted_beta, 2)
                portfolio_volatility = round(weighted_volatility, 2)
                logger.info(f"Portfolio: beta={portfolio_beta:.2f}, volatility={portfolio_volatility:.1%} (from {len(stocks)} stocks)")
                
                return portfolio_beta, portfolio_volatility, most_volatile
            else:
                logger.warning("Beta calculator not available - using estimate")
                return 1.0, 0.20, []
                
        except Exception as e:
            logger.error(f"Error calculating portfolio beta/volatility: {e}")
            return 1.0, 0.20, []
    
    def _calculate_portfolio_volatility(self, positions: List[Any]) -> float:
        """Calculate portfolio volatility using stock volatilities and option leverage"""
        stocks = [pos for pos in positions if pos.position_type == 'stock']
        options = [pos for pos in positions if pos.position_type in ['call', 'put']]
        
        if not positions:
            return 0.20
        
        # Estimate based on portfolio composition
        stock_ratio = len(stocks) / len(positions) if positions else 0
        options_ratio = len(options) / len(positions) if positions else 0
        
        # Base volatility: stocks ~20%, options ~50%
        estimated_vol = (stock_ratio * 0.20) + (options_ratio * 0.50)
        
        return round(min(0.80, max(0.10, estimated_vol)), 2)
    
    def _calculate_sharpe_ratio(self, positions: List[Any]) -> float:
        """Calculate Sharpe ratio (simplified)"""
        # In practice, would use actual returns data
        return 1.5
    
    def _calculate_max_drawdown(self, positions: List[Any]) -> float:
        """Calculate maximum drawdown based on portfolio volatility and concentration risk"""
        if not positions:
            return 0.0
            
        # Calculate portfolio volatility for stocks only (more stable for drawdown estimation)
        stocks = [pos for pos in positions if pos.position_type == 'stock']
        if not stocks:
            return 0.0
            
        # Calculate max drawdown based on actual portfolio volatility
        if hasattr(self, '_portfolio_volatility'):
            # Use the actual calculated volatility from the detailed metrics
            actual_vol = self._portfolio_volatility
        else:
            # Fallback to estimation
            actual_vol = self._estimate_portfolio_volatility(stocks)
        
        # Calculate concentration risk multiplier
        # Higher concentration = higher drawdown risk
        total_value = sum(pos.quantity * pos.current_price for pos in stocks)
        if total_value == 0:
            return 0.0
            
        # Calculate top 5 concentration
        stock_values = [(pos, pos.quantity * pos.current_price) for pos in stocks]
        stock_values.sort(key=lambda x: x[1], reverse=True)
        top_5_value = sum(value for _, value in stock_values[:5])
        top_5_concentration = top_5_value / total_value
        
        # Concentration risk multiplier: 1.0 (diversified) to 2.5 (highly concentrated)
        # 57.5% concentration = ~2.0x multiplier
        concentration_multiplier = 1.0 + (top_5_concentration - 0.2) * 2.5  # Scale from 20% to 80%
        concentration_multiplier = max(1.0, min(2.5, concentration_multiplier))
        
        # Base drawdown = 1.8x volatility, adjusted for concentration
        max_drawdown = actual_vol * 1.8 * concentration_multiplier
        
        return max_drawdown
    
    def _estimate_portfolio_volatility(self, positions: List[Any]) -> float:
        """Estimate portfolio volatility for risk calculations"""
        if not positions:
            return 0.0
            
        # Simple volatility estimation based on position types
        # In practice, would use historical price data
        total_value = sum(pos.quantity * pos.current_price for pos in positions)
        if total_value == 0:
            return 0.0
            
        weighted_vol = 0.0
        for pos in positions:
            position_value = pos.quantity * pos.current_price
            weight = position_value / total_value
            
            # Estimate volatility based on position type and size
            if pos.position_type == 'stock':
                # Stocks: estimate 20-40% annual volatility based on typical ranges
                estimated_vol = 0.30  # 30% annual
            elif pos.position_type in ['call', 'put']:
                # Options: much higher volatility
                estimated_vol = 0.60  # 60% annual
            else:
                estimated_vol = 0.20  # 20% annual for other assets
                
            weighted_vol += estimated_vol * weight
            
        return weighted_vol
    
    def _calculate_top_5_concentration(self, positions: List[Any]) -> float:
        """Calculate top 5 positions concentration"""
        if not positions:
            return 0.0
        
        portfolio_value = self._calculate_portfolio_value(positions)
        # Calculate position values with correct multipliers
        position_values = sorted([
            pos.quantity * pos.current_price * 100 if pos.position_type in ['call', 'put'] else pos.quantity * pos.current_price
            for pos in positions
        ], reverse=True)
        top_5_value = sum(position_values[:5])
        
        return (top_5_value / portfolio_value * 100) if portfolio_value > 0 else 0
    
    def _calculate_sector_concentration(self, positions: List[Any]) -> Dict[str, float]:
        """Calculate sector concentration - FAST (using placeholder data)"""
        # In production, fetch sector from database join or API
        # For now, return empty dict for speed (load from DB if needed)
        return {}
    
    def _calculate_portfolio_value(self, positions: List[Any]) -> float:
        """Calculate total portfolio value correctly accounting for options contract multiplier"""
        total = 0.0
        for pos in positions:
            if pos.position_type in ['call', 'put']:
                # Options: multiply by 100 (contract multiplier)
                total += pos.quantity * pos.current_price * 100
            else:
                # Stocks: no multiplier
                total += pos.quantity * pos.current_price
        return total
    
    def _calculate_expiring_value(self, positions: List[Any], days: int, exclude_days: int = 0) -> float:
        """Calculate value of positions expiring within days range - OPTIMIZED
        
        Args:
            positions: List of positions
            days: Maximum days to expiration (inclusive)
            exclude_days: Minimum days to expiration (exclusive) - use to exclude shorter-term positions
            
        Examples:
            _calculate_expiring_value(positions, 7) -> positions expiring in 0-7 days
            _calculate_expiring_value(positions, 30, 7) -> positions expiring in 8-30 days
        """
        # Single-pass calculation using generator
        return sum(
            pos.quantity * pos.current_price * 100
            for pos in positions
            if pos.position_type in ['call', 'put'] 
            and getattr(pos, 'time_to_expiration', None) is not None
            and pos.time_to_expiration <= days
            and pos.time_to_expiration > exclude_days
        )
    
    def _calculate_avg_liquidity_score(self, positions: List[Any]) -> float:
        """Calculate average liquidity score - uses cached value from fast metrics"""
        # Return the liquidity score calculated in fast metrics (already calculated)
        if hasattr(self, '_portfolio_liquidity_score'):
            return self._portfolio_liquidity_score / 100  # Convert 0-100 scale to 0-1 scale
        return 0.75  # Fallback
    
    def _find_illiquid_positions(self, positions: List[Any]) -> List[str]:
        """Find illiquid positions"""
        # In practice, would check actual volume data
        return []
    
    async def _calculate_stock_risk_summary(self, stocks: List[Any], portfolio_value: float, force_recalc: bool = False) -> StockRiskSummary:
        """Calculate stock-specific risk summary - OPTIMIZED single-pass"""
        if not stocks:
            return StockRiskSummary(
                stock_count=0,
                stock_value=0.0,
                stock_pnl_pct=0.0,
                concentration_risk_score=0,
                max_position_pct=0.0,
                sector_concentration={},
                unique_tickers=0
            )
        
        # SINGLE PASS calculation of all metrics
        stock_value = 0.0
        stock_cost = 0.0
        max_stock_value = 0.0
        hhi = 0.0
        unique_tickers_set = set()
        
        for pos in stocks:
            pos_value = pos.quantity * pos.current_price
            pos_cost = pos.quantity * pos.entry_price
            
            stock_value += pos_value
            stock_cost += pos_cost
            
            if pos_value > max_stock_value:
                max_stock_value = pos_value
            
            unique_tickers_set.add(pos.ticker)
        
        # Calculate derived metrics
        stock_pnl_pct = ((stock_value - stock_cost) / stock_cost * 100) if stock_cost > 0 else 0
        max_position_pct = (max_stock_value / stock_value * 100) if stock_value > 0 else 0
        
        # Calculate HHI in second pass (unavoidable, but fast)
        hhi = sum((pos.quantity * pos.current_price / stock_value) ** 2 for pos in stocks) if stock_value > 0 else 0
        concentration_score = min(100, int(hhi * 100))
        
        # Calculate portfolio beta, volatility, and liquidity in one pass (reuses fetched data)
        portfolio_beta, portfolio_volatility, most_volatile, liquidity_score = await self._calculate_portfolio_metrics(stocks, force_recalc=force_recalc)
        
        return StockRiskSummary(
            stock_count=len(stocks),
            stock_value=stock_value,
            stock_pnl_pct=stock_pnl_pct,
            concentration_risk_score=concentration_score,
            max_position_pct=max_position_pct,
            sector_concentration={},  # Empty for speed
            unique_tickers=len(unique_tickers_set),
            portfolio_beta=portfolio_beta,
            portfolio_volatility=portfolio_volatility,
            liquidity_score=int(liquidity_score),
            most_volatile_stocks=most_volatile
        )
    
    def _calculate_options_risk_summary(self, options: List[Any]) -> OptionsRiskSummary:
        """Calculate options-specific risk summary - ULTRA-OPTIMIZED single-pass"""
        if not options:
            return OptionsRiskSummary(
                options_count=0,
                options_value=0.0,
                total_delta=0.0,
                total_gamma=0.0,
                total_theta=0.0,
                total_vega=0.0,
                expiring_7d_count=0,
                expiring_30d_count=0,
                time_decay_risk_score=0,
                avg_days_to_expiry=0.0
            )
        
        # SINGLE PASS through all options - calculate everything at once
        options_value = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        expiring_7d = 0
        expiring_30d = 0
        days_sum = 0.0
        days_count = 0
        
        # Process all metrics in ONE LOOP (no date parsing!)
        for pos in options:
            # Value
            options_value += pos.quantity * pos.current_price * 100
            
            # Greeks (use getattr with default to avoid None errors)
            total_delta += getattr(pos, 'delta', 0) * pos.quantity
            total_gamma += getattr(pos, 'gamma', 0) * pos.quantity
            total_theta += getattr(pos, 'theta', 0) * pos.quantity * 100
            total_vega += getattr(pos, 'vega', 0) * pos.quantity
            
            # Expiration (use pre-calculated time_to_expiration if available)
            days_to_exp = getattr(pos, 'time_to_expiration', None)
            if days_to_exp is not None:
                days_sum += days_to_exp
                days_count += 1
                if days_to_exp <= 7:
                    expiring_7d += 1
                if days_to_exp <= 30:
                    expiring_30d += 1
        
        # Calculate average days to expiry
        avg_days_to_expiry = (days_sum / days_count) if days_count > 0 else 0
        
        # Calculate time decay risk score (0-100, higher = more risk)
        # Based on avg days to expiry and theta
        if avg_days_to_expiry > 0:
            time_factor = max(0, 100 - (avg_days_to_expiry / 60 * 100))  # Risk increases as expiry approaches
            theta_factor = min(100, abs(total_theta) / 10)  # Higher theta = higher risk
            time_decay_risk_score = int((time_factor + theta_factor) / 2)
        else:
            time_decay_risk_score = 0
        
        return OptionsRiskSummary(
            options_count=len(options),
            options_value=options_value,
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            expiring_7d_count=expiring_7d,
            expiring_30d_count=expiring_30d,
            time_decay_risk_score=time_decay_risk_score,
            avg_days_to_expiry=avg_days_to_expiry
        )

# Global instance
fast_risk_monitor = FastRiskMonitor()
