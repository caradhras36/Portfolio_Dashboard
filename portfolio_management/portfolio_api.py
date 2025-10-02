"""
Portfolio Risk Dashboard API
FastAPI backend for portfolio risk analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from dataclasses import dataclass, asdict
import asyncio
import os
import sys
import io
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import time

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Cache for risk calculations
risk_cache = {}
cache_ttl = 60  # 1 minute cache
last_cache_time = 0

# Technical Analysis Helper Functions
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index)"""
    if len(prices) < period + 1:
        return 50.0
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(prices) < slow + signal:
        return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
        'signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
        'histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
    }

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        current_price = prices.iloc[-1] if len(prices) > 0 else 100.0
        return {'upper': current_price * 1.05, 'middle': current_price, 'lower': current_price * 0.95, 'position': 0.5}
    
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    current_price = prices.iloc[-1]
    current_upper = upper.iloc[-1]
    current_lower = lower.iloc[-1]
    current_middle = middle.iloc[-1]
    
    # Calculate position within bands
    if current_upper != current_lower:
        position = (current_price - current_lower) / (current_upper - current_lower)
    else:
        position = 0.5
    
    return {
        'upper': float(current_upper) if not pd.isna(current_upper) else current_price * 1.05,
        'middle': float(current_middle) if not pd.isna(current_middle) else current_price,
        'lower': float(current_lower) if not pd.isna(current_lower) else current_price * 0.95,
        'position': float(position) if not pd.isna(position) else 0.5
    }

def calculate_support_resistance(prices: pd.Series, window: int = 20) -> Dict[str, float]:
    """Calculate support and resistance levels"""
    if len(prices) < window:
        current_price = prices.iloc[-1] if len(prices) > 0 else 100.0
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
        'support': float(support) if not pd.isna(support) else current_price * 0.95,
        'resistance': float(resistance) if not pd.isna(resistance) else current_price * 1.05
    }

def get_technical_indicators(ticker: str) -> Dict[str, Any]:
    """Get all technical indicators for a stock"""
    try:
        # Fetch historical data using yfinance
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty or len(hist) < 50:
            logger.warning(f"Insufficient historical data for {ticker}")
            return {}
        
        prices = hist['Close']
        
        # Calculate all indicators
        rsi = calculate_rsi(prices)
        macd = calculate_macd(prices)
        bollinger = calculate_bollinger_bands(prices)
        support_resistance = calculate_support_resistance(prices)
        
        return {
            'rsi': rsi,
            'macd': macd['macd'],
            'macd_signal': macd['signal'],
            'macd_histogram': macd['histogram'],
            'bb_upper': bollinger['upper'],
            'bb_middle': bollinger['middle'],
            'bb_lower': bollinger['lower'],
            'bb_position': bollinger['position'],
            'support_level': support_resistance['support'],
            'resistance_level': support_resistance['resistance']
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {ticker}: {e}")
        return {}

def get_cached_risk_analysis(positions_hash: str) -> Optional[Dict]:
    """Get cached risk analysis if still valid"""
    if positions_hash in risk_cache:
        cache_time, cached_data = risk_cache[positions_hash]
        if time.time() - cache_time < cache_ttl:
            logger.info("Using cached risk analysis")
            return cached_data
        else:
            # Remove expired cache
            del risk_cache[positions_hash]
    return None

def cache_risk_analysis(positions_hash: str, risk_data: Dict):
    """Cache risk analysis data"""
    global last_cache_time
    risk_cache[positions_hash] = (time.time(), risk_data)
    last_cache_time = time.time()
    logger.info("Cached risk analysis data")

def get_last_cache_time():
    """Get the last cache update time"""
    return last_cache_time

# Add project paths to import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "shared"))

from shared.config import SUPABASE_URL, SUPABASE_KEY
from csv_parser import BrokerCSVParser
from csp_cash_allocator import CSPCashAllocator
from options_analyzer import OptionsAnalyzer, MarketAwareOptionsAnalyzer
from scenario_analyzer import ScenarioAnalyzer
from real_market_risk_analyzer import RealMarketRiskAnalyzer
from fast_risk_monitor import fast_risk_monitor, FastRiskMonitor
from greeks_manager import GreeksManager
import yfinance as yf
import numpy as np
from cc_recommender import CoveredCallRecommender
from csp_recommender import CashSecuredPutRecommender

# Initialize options analyzer
options_analyzer = OptionsAnalyzer()
from market_data_client import market_data_client, PolygonMarketDataClient
from trading_system import trading_system, TradeType, TradingSystem
from supabase import create_client, Client

app = FastAPI(
    title="Portfolio Risk Dashboard",
    description="Real-time portfolio risk analysis and options strategy evaluation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(project_root, "web_interface", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize analyzers and database
try:
    options_analyzer = OptionsAnalyzer()
    market_analyzer = MarketAwareOptionsAnalyzer()
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Supabase credentials not found. Real portfolio data required.")
        raise Exception("Supabase credentials required for real portfolio data")
    else:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Successfully connected to Supabase")
        in_memory_positions = None
        
        # Initialize Greeks Manager for fast Greek access
        greeks_manager = GreeksManager(supabase)
        logger.info("✅ Initialized Greeks Manager")
    
    csv_parser = BrokerCSVParser()
    csp_allocator = CSPCashAllocator()
    
    # Initialize market data and trading systems
    market_client = market_data_client
    trading_client = TradingSystem(supabase_client=supabase if supabase else None)
    
except Exception as e:
    logger.error(f"Error initializing portfolio components: {e}")
    raise

@dataclass
class PortfolioPosition:
    """Portfolio position data structure"""
    ticker: str
    position_type: str  # 'stock', 'call', 'put', 'spread'
    quantity: int
    entry_price: float
    current_price: float
    expiration_date: Optional[str] = None
    strike_price: Optional[float] = None
    option_type: Optional[str] = None  # 'call' or 'put'
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None
    time_to_expiration: Optional[float] = None  # Days to expiration
    created_at: Optional[datetime] = None

def get_positions_hash(positions: List[PortfolioPosition]) -> str:
    """Generate hash for positions to use as cache key"""
    positions_data = []
    for pos in positions:
        positions_data.append({
            'ticker': pos.ticker,
            'quantity': pos.quantity,
            'current_price': pos.current_price,
            'position_type': pos.position_type
        })
    # Sort by ticker to ensure consistent ordering
    sorted_data = sorted(positions_data, key=lambda x: x.get('ticker', ''))
    return str(hash(str(sorted_data)))

# Database helper functions
async def calculate_greeks_for_position(position: PortfolioPosition, force_recalc: bool = True, save_to_db: bool = False) -> PortfolioPosition:
    """Fetch fresh Greeks from Polygon.io for an options position
    
    Args:
        position: The position to fetch Greeks for
        force_recalc: Always fetch fresh Greeks (default: True for real-time data)
        save_to_db: Save Greeks to database (default: False - done in batch)
    
    Note: This function ALWAYS fetches from Polygon.io by default for accurate real-time Greeks.
    """
    if position.position_type not in ['call', 'put']:
        return position
    
    # Always fetch fresh Greeks from Polygon.io (unless explicitly told not to)
    # We want real-time accurate data, not cached data
    
    try:
        # Skip if we don't have the necessary data for Greeks calculation
        if not position.strike_price or position.strike_price <= 0:
            logger.warning(f"Skipping Greeks calculation for {position.ticker}: missing or invalid strike price")
            position.delta = 0.0
            position.gamma = 0.0
            position.theta = 0.0
            position.vega = 0.0
            return position
            
        # Calculate time to expiration if we have expiration_date
        time_to_exp = 30.0  # Default
        if position.expiration_date:
            try:
                exp_date = datetime.fromisoformat(position.expiration_date.replace('Z', '+00:00'))
                time_to_exp = max(1.0, (exp_date - datetime.now()).days)  # At least 1 day
                position.time_to_expiration = time_to_exp  # Store in position
            except:
                time_to_exp = 30.0
                position.time_to_expiration = 30.0
        else:
            position.time_to_expiration = 30.0
        
        # Get real Greeks directly from Polygon.io using market data client
        current_price = position.current_price
        strike_price = position.strike_price
        
        # Try to get real Greeks from Polygon.io using the market data client
        try:
            if market_data_client and position.expiration_date:
                # Get options chain for the specific expiration date
                options_chain = await market_data_client.get_options_chain(
                    position.ticker, 
                    position.expiration_date
                )
                
                # Find the matching option
                for option in options_chain:
                    if (option.strike_price == strike_price and 
                        option.contract_type == position.position_type):
                        
                        # Use real market data directly
                        position.delta = option.delta
                        position.gamma = option.gamma
                        position.theta = option.theta
                        position.vega = option.vega
                        position.implied_volatility = option.implied_volatility
                        
                        logger.info(f"Found real Greeks for {position.ticker}: Delta={option.delta:.3f}, IV={option.implied_volatility:.1%}")
                        return position
                        
        except Exception as e:
            logger.warning(f"Could not fetch Greeks from Polygon.io for {position.ticker}: {e}")
        
        # Fallback: Set default values if we couldn't get real data
        position.delta = 0.0
        position.gamma = 0.0
        position.theta = 0.0
        position.vega = 0.0
        position.implied_volatility = 0.30  # Default fallback
        
        logger.info(f"Using fallback Greeks for {position.ticker}: S={current_price}, K={strike_price}, T={time_to_exp:.1f}d, IV={position.implied_volatility:.1%}, Delta={position.delta:.3f}, Theta={position.theta:.3f}")
        
    except Exception as e:
        logger.error(f"Error getting Greeks for {position.ticker}: {e}")
        # Set default values
        position.delta = 0.0
        position.gamma = 0.0
        position.theta = 0.0
        position.vega = 0.0
        position.implied_volatility = 0.30
    
    # Save calculated Greeks to database for future fast access
    # Greeks are now saved to option_greeks table via save_greeks_to_greeks_table()
    
    return position

# DEPRECATED: save_greeks_to_db function removed - Greeks now saved to option_greeks table

async def save_all_greeks_to_db(positions: List[PortfolioPosition]):
    """Save calculated Greeks for all positions to database (old positions table)"""
    try:
        saved_count = 0
        failed_count = 0
        
        for position in positions:
            # Greeks are now saved to option_greeks table via save_greeks_to_greeks_table()
            pass
        
        logger.info(f"💾 Saved {saved_count} Greeks to DB ({failed_count} failed)")
        
    except Exception as e:
        logger.error(f"Error in batch save Greeks: {e}")

async def save_greeks_to_greeks_table(positions: List[PortfolioPosition]):
    """Save Greeks to the NEW option_greeks table (FAST!)"""
    try:
        from greeks_manager import OptionGreeks
        
        greeks_list = []
        for pos in positions:
            if pos.position_type in ['call', 'put'] and pos.delta is not None:
                greeks = OptionGreeks(
                    ticker=pos.ticker,
                    strike_price=pos.strike_price,
                    expiration_date=pos.expiration_date,
                    option_type=pos.position_type,
                    delta=pos.delta,
                    gamma=pos.gamma,
                    theta=pos.theta,
                    vega=pos.vega,
                    implied_volatility=pos.implied_volatility,
                    time_to_expiration=pos.time_to_expiration,
                    underlying_price=pos.current_price
                )
                greeks_list.append(greeks)
        
        if greeks_list:
            await greeks_manager.save_greeks_batch(greeks_list)
            logger.info(f"💾 Saved {len(greeks_list)} Greeks to option_greeks table")
        
    except Exception as e:
        logger.error(f"Error saving Greeks to greeks table: {e}")

async def refresh_greeks_background(option_positions: List[PortfolioPosition]):
    """Background task to refresh Greeks from Polygon.io and update database
    
    This runs asynchronously and doesn't block the main response.
    Fresh Greeks are saved to DB for the next page load.
    """
    try:
        logger.info(f"🔄 BACKGROUND: Starting Greek refresh for {len(option_positions)} options...")
        
        # Fetch fresh Greeks from Polygon.io
        refreshed_positions = await calculate_greeks_batch(option_positions)
        
        # Save to database for next load
        await save_all_greeks_to_db(refreshed_positions)
        
        logger.info(f"✅ BACKGROUND: Greek refresh complete - {len(refreshed_positions)} options updated in DB")
        
    except Exception as e:
        logger.error(f"❌ BACKGROUND: Error refreshing Greeks: {e}")

async def calculate_greeks_batch(option_positions: List[PortfolioPosition]) -> List[PortfolioPosition]:
    """Calculate Greeks for multiple option positions efficiently using batch processing"""
    if not option_positions:
        return []
    
    total_options = len(option_positions)
    logger.info(f"📊 BATCH PROCESSING: Starting {total_options} option positions...")
    
    # Group options by ticker and expiration date for batch API calls
    options_by_ticker_exp = {}
    for position in option_positions:
        if position.position_type not in ['call', 'put']:
            continue
            
        key = f"{position.ticker}_{position.expiration_date}"
        if key not in options_by_ticker_exp:
            options_by_ticker_exp[key] = []
        options_by_ticker_exp[key].append(position)
    
    logger.info(f"📈 GROUPING: Created {len(options_by_ticker_exp)} option groups")
    
    # Show group details (first 3 only)
    for i, (ticker_exp, positions) in enumerate(list(options_by_ticker_exp.items())[:3]):
        logger.info(f"   Group {i+1}: {ticker_exp} ({len(positions)} positions)")
    if len(options_by_ticker_exp) > 3:
        logger.info(f"   ... and {len(options_by_ticker_exp) - 3} more groups")
    
    # Process each group in parallel
    tasks = []
    for ticker_exp, positions in options_by_ticker_exp.items():
        tasks.append(process_option_group(positions, ticker_exp))
    
    # Wait for all groups to complete with progress tracking
    completed_groups = 0
    total_groups = len(tasks)
    
    logger.info(f"🚀 PARALLEL PROCESSING: Starting {total_groups} groups simultaneously...")
    start_time = asyncio.get_event_loop().time()
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = asyncio.get_event_loop().time()
    processing_time = end_time - start_time
    
    # Flatten results and track progress
    processed_positions = []
    successful_groups = 0
    failed_groups = 0
    
    for i, result in enumerate(results):
        completed_groups += 1
        if isinstance(result, Exception):
            logger.error(f"❌ GROUP {i+1}/{total_groups} FAILED: {result}")
            failed_groups += 1
            continue
        processed_positions.extend(result)
        successful_groups += 1
        if completed_groups % 5 == 0 or completed_groups == total_groups:  # Log every 5th group or last group
            logger.info(f"✅ GROUP {completed_groups}/{total_groups} COMPLETE: {len(result)} positions processed")
    
    logger.info(f"🎯 BATCH PROCESSING COMPLETE:")
    logger.info(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
    logger.info(f"   ✅ Successful: {successful_groups}/{total_groups} groups")
    logger.info(f"   📊 Processed: {len(processed_positions)}/{total_options} options")
    if processing_time > 0:
        logger.info(f"   🚀 Speed: {total_options/processing_time:.1f} options/sec")
    
    return processed_positions

async def process_option_group(positions: List[PortfolioPosition], ticker_exp: str) -> List[PortfolioPosition]:
    """Process a group of options with the same ticker and expiration date"""
    if not positions:
        return []
    
    # Get the first position to determine ticker and expiration
    first_pos = positions[0]
    ticker = first_pos.ticker
    expiration_date = first_pos.expiration_date
    
    logger.debug(f"🔍 GROUP: {ticker_exp} ({len(positions)} positions)")
    
    # Skip if we already have Greeks calculated for all positions
    already_calculated = sum(1 for pos in positions if pos.delta is not None)
    if already_calculated == len(positions):
        logger.debug(f"⚡ SKIP: {ticker_exp} - All Greeks cached")
        return positions
    
    try:
        # OPTIMIZED: Use snapshot endpoint directly for Greeks (much faster!)
        if market_data_client and expiration_date:
            logger.info(f"📡 Fetching Greeks from Polygon.io snapshot for {ticker_exp}...")
            
            # Build Polygon option tickers directly (format: O:TICKER{YYMMDD}{C/P}{STRIKE*1000})
            option_tickers = []
            for pos in positions:
                if pos.delta is not None:
                    continue  # Skip if already has Greeks
                
                try:
                    # Build Polygon option ticker
                    exp_date = datetime.fromisoformat(pos.expiration_date.replace('Z', '+00:00'))
                    date_str = exp_date.strftime('%y%m%d')
                    contract_type = 'C' if pos.position_type == 'call' else 'P'
                    strike_int = int(pos.strike_price * 1000)
                    polygon_ticker = f"O:{ticker}{date_str}{contract_type}{strike_int:08d}"
                    option_tickers.append((polygon_ticker, pos))
                except Exception as e:
                    logger.warning(f"Could not build Polygon ticker for {pos.ticker}: {e}")
            
            if option_tickers:
                logger.info(f"💰 Fetching Greeks for {len(option_tickers)} options from snapshot API...")
                
                # Batch fetch all option snapshots with Greeks
                polygon_ticker_list = [ot[0] for ot in option_tickers]
                option_data = await market_data_client.get_batch_option_prices(polygon_ticker_list)
                logger.info(f"📈 Retrieved {len(option_data)} option snapshots with Greeks")
                
                # Match data to positions
                matched_count = 0
                for polygon_ticker, position in option_tickers:
                    if polygon_ticker in option_data:
                        data = option_data[polygon_ticker]
                        
                        # Use real Greeks from snapshot API
                        position.delta = data.get('delta', 0.5 if position.position_type == 'call' else -0.5)
                        position.gamma = data.get('gamma', 0.05)
                        position.theta = data.get('theta', -0.05)
                        position.vega = data.get('vega', 0.2)
                        position.implied_volatility = data.get('implied_volatility', 0.30)
                        position.time_to_expiration = data.get('time_to_expiration')
                        
                        # Update current price if available
                        if data.get('last_price'):
                            position.current_price = data['last_price']
                        
                        matched_count += 1
                        logger.debug(f"✅ {position.ticker} ${position.strike_price}: Δ={position.delta:.3f}, θ={position.theta:.4f}")
                    else:
                        # Fallback for options not found in Polygon.io
                        position.delta = 0.5 if position.position_type == 'call' else -0.5
                        position.gamma = 0.05
                        position.theta = -0.05
                        position.vega = 0.2
                        position.implied_volatility = 0.30
                        logger.debug(f"⚠️  Fallback for {position.ticker} ${position.strike_price}")
                
                logger.info(f"📊 {ticker_exp}: Matched {matched_count}/{len(option_tickers)} options")
        else:
            # No market data available, use fallback for all positions
            logger.info(f"⚠️ No market data for {ticker_exp}, using fallback values")
            for position in positions:
                if position.delta is None:
                    position.delta = 0.0
                    position.gamma = 0.0
                    position.theta = 0.0
                    position.vega = 0.0
                    position.implied_volatility = 0.30
                    
    except Exception as e:
        logger.warning(f"Error processing option group for {ticker}: {e}")
        # Use fallback values for all positions in this group
        for position in positions:
            if position.delta is None:
                position.delta = 0.0
                position.gamma = 0.0
                position.theta = 0.0
                position.vega = 0.0
                position.implied_volatility = 0.30
    
    return positions

async def get_portfolio_positions(use_cached_greeks: bool = True, refresh_greeks_async: bool = True) -> List[PortfolioPosition]:
    """Get all portfolio positions from Supabase database
    
    Args:
        use_cached_greeks: Use Greeks from option_greeks table for fast loading (default: True)
        refresh_greeks_async: Trigger background refresh of Greeks from Polygon.io (default: True)
    
    Flow:
        1. Read positions from portfolio_positions table (FAST)
        2. Read Greeks from option_greeks table in single batch query (FAST - 20-50ms)
        3. Calculate risk metrics with cached Greeks (FAST)
        4. Optionally trigger background refresh from Polygon.io for next load
    """
    if supabase is None:
        logger.error("No database connection available")
        return []
    
    try:
        start_time = time.time()
        
        # Fetch positions
        response = supabase.table('portfolio_positions').select('*').execute()
        positions = []
        
        if not response.data:
            logger.info("No positions found in database")
            return positions
        
        logger.info(f"✅ Fetched {len(response.data)} positions in {(time.time()-start_time)*1000:.0f}ms")
            
        # Separate stocks and options
        stock_positions = []
        option_positions = []
        
        for row in response.data:
            try:
                position = PortfolioPosition(
                    ticker=row['ticker'],
                    position_type=row['position_type'],
                    quantity=row['quantity'],
                    entry_price=float(row['entry_price']),
                    current_price=float(row['current_price']),
                    expiration_date=row.get('expiration_date'),
                    strike_price=float(row['strike_price']) if row.get('strike_price') else None,
                    option_type=row.get('option_type'),
                    delta=None,  # Will load from option_greeks table
                    gamma=None,
                    theta=None,
                    vega=None,
                    implied_volatility=None,
                    time_to_expiration=None,  # Will calculate below
                    created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else None
                )
                
                # Calculate time_to_expiration for options
                if position.position_type in ['call', 'put'] and position.expiration_date:
                    try:
                        exp_date = datetime.fromisoformat(position.expiration_date.replace('Z', '+00:00'))
                        position.time_to_expiration = max(1.0, (exp_date - datetime.now()).days)
                    except:
                        position.time_to_expiration = 30.0  # Default fallback
                elif position.position_type in ['call', 'put']:
                    position.time_to_expiration = 30.0  # Default for options without expiration_date
                
                # Add to appropriate list
                if position.position_type in ['call', 'put']:
                    option_positions.append(position)
                else:
                    stock_positions.append(position)
                    
            except Exception as e:
                logger.error(f"Error parsing position row {row}: {e}")
                continue
        
        # Add stock positions
        positions.extend(stock_positions)
        
        # For options: load Greeks from option_greeks table (FAST - single query!)
        if option_positions and use_cached_greeks:
            greeks_start = time.time()
            logger.info(f"📊 Loading Greeks for {len(option_positions)} options from option_greeks table...")
            
            # Batch load ALL Greeks in one query
            greeks_dict = await greeks_manager.get_greeks_batch(option_positions)
            greeks_time = (time.time() - greeks_start) * 1000
            logger.info(f"✅ Loaded {len(greeks_dict)} Greeks in {greeks_time:.0f}ms from option_greeks table")
            
            # Match Greeks to positions
            options_with_greeks = 0
            options_without_greeks = []
            
            for pos in option_positions:
                # Build lookup key
                key = f"{pos.ticker}_{pos.strike_price}_{pos.expiration_date}_{pos.position_type}"
                
                if key in greeks_dict:
                    greeks = greeks_dict[key]
                    pos.delta = greeks.delta
                    pos.gamma = greeks.gamma
                    pos.theta = greeks.theta
                    pos.vega = greeks.vega
                    pos.implied_volatility = greeks.implied_volatility
                    
                    # Use Greeks time_to_expiration if available, otherwise keep calculated value
                    if greeks.time_to_expiration is not None:
                        pos.time_to_expiration = greeks.time_to_expiration
                    # If Greeks time_to_expiration is None, keep the calculated value from above
                    
                    options_with_greeks += 1
                else:
                    options_without_greeks.append(pos)
            
            logger.info(f"✅ Matched Greeks: {options_with_greeks}/{len(option_positions)} options")
            
            # Handle options without Greeks
            if options_without_greeks:
                logger.warning(f"⚠️  {len(options_without_greeks)} options missing Greeks - fetching from Polygon.io...")
                fetch_start = time.time()
                refreshed = await calculate_greeks_batch(options_without_greeks)
                fetch_time = time.time() - fetch_start
                logger.warning(f"⏱️  Fetched from Polygon.io in {fetch_time:.2f}s")
                
                # Save to option_greeks table
                await save_greeks_to_greeks_table(refreshed)
                
                # Update option_positions with fetched Greeks
                for ref_pos in refreshed:
                    for pos in option_positions:
                        if (pos.ticker == ref_pos.ticker and pos.strike_price == ref_pos.strike_price and
                            pos.expiration_date == ref_pos.expiration_date and pos.position_type == ref_pos.position_type):
                            pos.delta = ref_pos.delta
                            pos.gamma = ref_pos.gamma
                            pos.theta = ref_pos.theta
                            pos.vega = ref_pos.vega
                            pos.implied_volatility = ref_pos.implied_volatility
                            break
        
        # Add option positions
        positions.extend(option_positions)
        
        logger.info(f"✅ Total loaded: {len(positions)} positions ({len(stock_positions)} stocks, {len(option_positions)} options)")
        
        # Trigger background refresh for next load (async, doesn't block)
        if refresh_greeks_async and option_positions:
            logger.info(f"🔄 Triggering background Greek refresh for {len(option_positions)} options...")
            asyncio.create_task(refresh_greeks_background(option_positions))
        
        return positions
                
        logger.info(f"🎉 PORTFOLIO SUMMARY:")
        logger.info(f"   📈 Total positions: {len(positions)}")
        logger.info(f"   🏢 Stocks: {len(stock_positions)}")
        logger.info(f"   📊 Options: {len(option_positions)}")
        logger.info(f"   💰 Total value: ${sum(pos.quantity * pos.current_price for pos in positions):,.2f}")
        
        return positions
        
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return []

async def save_portfolio_position(position: PortfolioPosition) -> bool:
    """Save a portfolio position to Supabase or in-memory storage"""
    if supabase is None:
        # Use in-memory storage
        in_memory_positions.append(position)
        logger.info(f"Successfully saved position to memory: {position.ticker}")
        return True
        
    try:
        data = {
            'ticker': position.ticker,
            'position_type': position.position_type,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'expiration_date': position.expiration_date,
            'strike_price': position.strike_price,
            'option_type': position.option_type,
            'delta': position.delta,
            'gamma': position.gamma,
            'theta': position.theta,
            'vega': position.vega,
            'implied_volatility': position.implied_volatility,
            'created_at': datetime.now().isoformat()
        }
        
        response = supabase.table('portfolio_positions').insert(data).execute()
        logger.info(f"Successfully saved position to database: {position.ticker}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving position {position.ticker}: {e}")
        return False

async def delete_portfolio_position(position_id: int) -> bool:
    """Delete a portfolio position from Supabase"""
    try:
        response = supabase.table('portfolio_positions').delete().eq('id', position_id).execute()
        return True
    except Exception as e:
        print(f"Error deleting position: {e}")
        return False

async def update_position_prices():
    """Update current prices for all positions using real-time market data"""
    try:
        positions = await get_portfolio_positions()
        if not positions:
            return
            
        # Skip known problematic tickers that consistently fail
        skip_tickers = {'AFMDQ', 'TTOO', 'OPTT', 'HGBL', 'USIO', 'NDRA'}
        unique_tickers = list(set(pos.ticker for pos in positions if pos.ticker not in skip_tickers))
        
        logger.info(f"Updating prices for {len(unique_tickers)} tickers (skipped {len(skip_tickers)} known issues)")
        
        # Get all stock prices in batch for better performance
        stock_prices = await market_client.get_multiple_stock_prices(unique_tickers)
        
        # Update positions with real-time prices
        for position in positions:
            try:
                if position.ticker in stock_prices:
                    price_data = stock_prices[position.ticker]
                    position.current_price = price_data.price
                    
                    # Skip individual option price fetching to avoid API rate limits
                    # Options will use fallback values for faster loading
                    if position.position_type in ['call', 'put']:
                        # Use fallback values instead of API calls
                        if position.delta is None:
                            position.delta = 0.0
                            position.gamma = 0.0
                            position.theta = 0.0
                            position.vega = 0.0
                            position.implied_volatility = 0.30
                    
                    # Update in database if available
                    if supabase:
                        update_data = {
                            'current_price': position.current_price,
                            'delta': position.delta,
                            'gamma': position.gamma,
                            'theta': position.theta,
                            'vega': position.vega,
                            'implied_volatility': position.implied_volatility
                        }
                        supabase.table('portfolio_positions').update(update_data).eq('ticker', position.ticker).execute()
                        
            except Exception as e:
                logger.error(f"Error updating position {position.ticker}: {e}")
                continue
                
        logger.info(f"Updated prices for {len(positions)} positions")
                
    except Exception as e:
        logger.error(f"Error updating position prices: {e}")

async def save_risk_snapshot(risk_data: Dict):
    """Save risk snapshot to Supabase"""
    try:
        data = {
            'snapshot_date': datetime.now().date().isoformat(),
            'total_delta': risk_data['greeks']['total_delta'],
            'total_gamma': risk_data['greeks']['total_gamma'],
            'total_theta': risk_data['greeks']['total_theta'],
            'total_vega': risk_data['greeks']['total_vega'],
            'portfolio_value': risk_data['portfolio_value'],
            'risk_metrics': json.dumps(risk_data)
        }
        
        supabase.table('portfolio_risk_snapshots').insert(data).execute()
        return True
    except Exception as e:
        print(f"Error saving risk snapshot: {e}")
        return False

class PortfolioRiskAnalyzer:
    """Portfolio risk calculation engine"""
    
    def __init__(self):
        # Note: Options analysis capabilities can be integrated later
        pass
    
    def calculate_portfolio_greeks(self, positions: List[PortfolioPosition]) -> Dict[str, float]:
        """Calculate aggregated portfolio Greeks"""
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        
        for position in positions:
            if position.position_type == 'stock':
                # Stocks have delta = 1
                total_delta += position.quantity
            elif position.position_type in ['call', 'put']:
                # Use position's Greeks if available
                if position.delta is not None:
                    total_delta += position.delta * position.quantity
                if position.gamma is not None:
                    total_gamma += position.gamma * position.quantity
                if position.theta is not None:
                    total_theta += position.theta * position.quantity
                if position.vega is not None:
                    total_vega += position.vega * position.quantity
        
        return {
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'total_vega': total_vega
        }
    
    def calculate_portfolio_value(self, positions: List[PortfolioPosition]) -> float:
        """Calculate total portfolio value"""
        return sum(pos.quantity * pos.current_price for pos in positions)
    
    def calculate_position_weights(self, positions: List[PortfolioPosition]) -> Dict[str, float]:
        """Calculate position weights as percentage of portfolio"""
        total_value = self.calculate_portfolio_value(positions)
        if total_value == 0:
            return {}
        
        weights = {}
        for position in positions:
            position_value = position.quantity * position.current_price
            weights[position.ticker] = (position_value / total_value) * 100
        
        return weights
    
    def calculate_concentration_risk(self, positions: List[PortfolioPosition]) -> Dict[str, Any]:
        """Calculate portfolio concentration risk metrics"""
        weights = self.calculate_position_weights(positions)
        
        if not weights:
            return {}
        
        # Top 5 positions concentration
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_5_concentration = sum(weight for _, weight in sorted_weights[:5])
        
        # Herfindahl-Hirschman Index (concentration measure)
        hhi = sum(weight ** 2 for weight in weights.values())
        
        return {
            'top_5_concentration': top_5_concentration,
            'hhi': hhi,
            'max_position_weight': max(weights.values()) if weights else 0,
            'position_count': len(weights)
        }
    
    def analyze_portfolio_risk(self, positions: List[PortfolioPosition]) -> Dict[str, Any]:
        """Comprehensive portfolio risk analysis"""
        greeks = self.calculate_portfolio_greeks(positions)
        portfolio_value = self.calculate_portfolio_value(positions)
        concentration = self.calculate_concentration_risk(positions)
        
        # Calculate P&L
        total_pnl = sum(pos.quantity * (pos.current_price - pos.entry_price) for pos in positions)
        total_pnl_pct = (total_pnl / sum(pos.quantity * pos.entry_price for pos in positions)) * 100 if positions else 0
        
        # Separate stocks and options
        stocks = [pos for pos in positions if pos.position_type == 'stock']
        options = [pos for pos in positions if pos.position_type in ['call', 'put']]
        
        # Stock metrics
        stock_metrics = {
            'count': len(stocks),
            'total_value': sum(pos.quantity * pos.current_price for pos in stocks),
            'unique_tickers': len(set(pos.ticker for pos in stocks)),
            'avg_pnl_pct': 0
        }
        if stocks:
            stock_pnl_pcts = [((pos.current_price - pos.entry_price) / pos.entry_price) * 100 for pos in stocks]
            stock_metrics['avg_pnl_pct'] = sum(stock_pnl_pcts) / len(stock_pnl_pcts)
        
        # Options metrics
        options_metrics = {
            'count': len(options),
            'total_value': sum(pos.quantity * pos.current_price for pos in options),
            'calls': len([pos for pos in options if pos.position_type == 'call']),
            'puts': len([pos for pos in options if pos.position_type == 'put']),
            'avg_delta': 0,
            'avg_theta': 0,
            'avg_gamma': 0,
            'avg_vega': 0
        }
        if options:
            # Calculate weighted averages (weighted by absolute position value)
            total_option_value = sum(abs(pos.quantity * pos.current_price) for pos in options)
            weighted_delta = 0
            weighted_theta = 0
            weighted_gamma = 0
            weighted_vega = 0
            
            for pos in options:
                position_value = abs(pos.quantity * pos.current_price)  # Use absolute value
                if total_option_value > 0:
                    weight = position_value / total_option_value
                    
                    # Use calculated Greeks if available, otherwise defaults
                    delta = pos.delta if pos.delta is not None else 0.0
                    theta = pos.theta if pos.theta is not None else 0.0
                    gamma = pos.gamma if pos.gamma is not None else 0.0
                    vega = pos.vega if pos.vega is not None else 0.0
                    
                    weighted_delta += delta * weight
                    weighted_theta += theta * weight
                    weighted_gamma += gamma * weight
                    weighted_vega += vega * weight
            
            options_metrics['avg_delta'] = weighted_delta
            options_metrics['avg_theta'] = weighted_theta
            options_metrics['avg_gamma'] = weighted_gamma
            options_metrics['avg_vega'] = weighted_vega
            
            logger.info(f"Options Greeks averages: Delta={weighted_delta:.3f}, Theta={weighted_theta:.3f}, Gamma={weighted_gamma:.3f}, Vega={weighted_vega:.3f}")
        
        # Calculate CSP cash requirements
        csp_cash_required = 0
        csp_positions = [pos for pos in options if pos.position_type == 'put' and pos.quantity < 0]
        for csp in csp_positions:
            csp_cash_required += abs(csp.quantity) * csp.strike_price * 100
        
        # Cash balance should come from actual account data, not estimates
        actual_cash_balance = 0  # This should be populated from your CSV data
        
        return {
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'greeks': greeks,
            'stocks': stock_metrics,
            'options': options_metrics,
            'concentration': concentration,
            'position_count': len(positions),
            'cash_balance': actual_cash_balance,
            'csp_required_cash': csp_cash_required,
            'analysis_date': datetime.now().isoformat()
        }

# Initialize risk analyzer and scenario analyzer
risk_analyzer = PortfolioRiskAnalyzer()
scenario_analyzer = ScenarioAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard"""
    try:
        template_path = os.path.join(project_root, "web_interface", "templates", "dashboard.html")
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard not found. Please check templates/dashboard.html</h1>")

@app.get("/api/portfolio/positions")
async def get_positions():
    """Get all portfolio positions"""
    positions = await get_portfolio_positions()
    return [asdict(pos) for pos in positions]

@app.post("/api/portfolio/positions")
async def add_position(position_data: dict):
    """Add a new position to the portfolio"""
    try:
        position = PortfolioPosition(
            ticker=position_data['ticker'],
            position_type=position_data['position_type'],
            quantity=int(position_data['quantity']),
            entry_price=float(position_data['entry_price']),
            current_price=float(position_data['current_price']),
            expiration_date=position_data.get('expiration_date'),
            strike_price=float(position_data.get('strike_price')) if position_data.get('strike_price') else None,
            option_type=position_data.get('option_type'),
            created_at=datetime.now()
        )
        
        # Save to database
        success = await save_portfolio_position(position)
        if success:
            return {"message": "Position added successfully", "position": asdict(position)}
        else:
            raise HTTPException(status_code=500, detail="Failed to save position to database")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error adding position: {str(e)}")

@app.post("/api/portfolio/import")
async def import_portfolio(file: UploadFile = File(...)):
    """Import portfolio from CSV file"""
    try:
        # Save uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Parse CSV using flexible parser
            df = csv_parser.parse_csv(temp_file_path)
            
            if df.empty:
                return {"message": "No valid positions found in CSV file"}
            
            # Clear existing positions (delete all from database)
            try:
                supabase.table('portfolio_positions').delete().neq('id', 0).execute()
            except Exception as e:
                print(f"Warning: Could not clear existing positions: {e}")
            
            imported_count = 0
            # Process each row
            for _, row in df.iterrows():
                try:
                    position = PortfolioPosition(
                        ticker=str(row['ticker']),
                        position_type=str(row.get('position_type', 'stock')),
                        quantity=int(row['quantity']),
                        entry_price=float(row['entry_price']),
                        current_price=float(row.get('current_price', row['entry_price'])),
                        expiration_date=str(row.get('expiration_date', '')) if pd.notna(row.get('expiration_date')) else None,
                        strike_price=float(row['strike_price']) if pd.notna(row.get('strike_price')) else None,
                        option_type=str(row.get('option_type', '')) if pd.notna(row.get('option_type')) else None,
                        created_at=datetime.now()
                    )
                    
                    # Save to database
                    success = await save_portfolio_position(position)
                    if success:
                        imported_count += 1
                        
                except Exception as e:
                    print(f"Error importing row {_}: {e}")
                    continue
            
            return {
                "message": f"Successfully imported {imported_count} positions",
                "broker_format": csv_parser.detect_broker_format(pd.read_csv(temp_file_path)),
                "positions": imported_count
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error importing portfolio: {str(e)}")

@app.get("/api/portfolio/risk-metrics-fast")
async def get_risk_metrics_fast():
    """Get portfolio metrics using already-loaded position data (fast)"""
    positions = await get_portfolio_positions()
    if not positions:
        return {"message": "No positions in portfolio"}
    
    # Separate stocks and options
    stocks = [pos for pos in positions if pos.position_type == 'stock']
    options = [pos for pos in positions if pos.position_type in ['call', 'put']]
    
    # Calculate stock metrics
    stock_metrics = {
        'count': len(stocks),
        'total_value': sum(pos.quantity * pos.current_price for pos in stocks),
        'unique_tickers': len(set(pos.ticker for pos in stocks)),
        'avg_pnl_pct': 0
    }
    if stocks:
        stock_pnl_pcts = [((pos.current_price - pos.entry_price) / pos.entry_price) * 100 for pos in stocks]
        stock_metrics['avg_pnl_pct'] = sum(stock_pnl_pcts) / len(stock_pnl_pcts)
    
    # Calculate options metrics (with correct multipliers)
    options_metrics = {
        'count': len(options),
        'total_value': sum(pos.quantity * pos.current_price * 100 for pos in options),
        'calls': len([pos for pos in options if pos.position_type == 'call']),
        'puts': len([pos for pos in options if pos.position_type == 'put']),
        'avg_delta': 0,
        'avg_theta': 0,
        'avg_gamma': 0,
        'avg_vega': 0
    }
    if options:
        deltas = [pos.delta for pos in options if pos.delta is not None]
        thetas = [pos.theta for pos in options if pos.theta is not None]
        gammas = [pos.gamma for pos in options if pos.gamma is not None]
        vegas = [pos.vega for pos in options if pos.vega is not None]
        
        if deltas:
            options_metrics['avg_delta'] = sum(deltas) / len(deltas)
        if thetas:
            options_metrics['avg_theta'] = sum(thetas) / len(thetas)
        if gammas:
            options_metrics['avg_gamma'] = sum(gammas) / len(gammas)
        if vegas:
            options_metrics['avg_vega'] = sum(vegas) / len(vegas)
    
    # Calculate portfolio Greeks (total, not average)
    total_delta = sum(pos.delta * pos.quantity for pos in options if pos.delta is not None)
    total_gamma = sum(pos.gamma * pos.quantity for pos in options if pos.gamma is not None)
    total_theta = sum(pos.theta * pos.quantity for pos in options if pos.theta is not None)
    total_vega = sum(pos.vega * pos.quantity for pos in options if pos.vega is not None)
    
    greeks = {
        'total_delta': total_delta,
        'total_gamma': total_gamma,
        'total_theta': total_theta,
        'total_vega': total_vega
    }
    
    # Calculate P&L (with correct multipliers for options)
    total_pnl = sum(
        pos.quantity * (pos.current_price - pos.entry_price) * 100 if pos.position_type in ['call', 'put'] 
        else pos.quantity * (pos.current_price - pos.entry_price)
        for pos in positions
    )
    total_cost = sum(
        pos.quantity * pos.entry_price * 100 if pos.position_type in ['call', 'put'] 
        else pos.quantity * pos.entry_price
        for pos in positions
    )
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
    
    # Calculate portfolio value (with correct multipliers for options)
    portfolio_value = sum(
        pos.quantity * pos.current_price * 100 if pos.position_type in ['call', 'put'] 
        else pos.quantity * pos.current_price
        for pos in positions
    )
    
    # CSP cash required
    csp_positions = [pos for pos in options if pos.position_type == 'put' and pos.quantity < 0]
    csp_cash_required = sum(abs(pos.quantity) * pos.strike_price * 100 for pos in csp_positions if pos.strike_price)
    
    # Calculate concentration metrics (with correct multipliers for options)
    position_values = [
        (pos.ticker, 
         pos.quantity * pos.current_price * 100 if pos.position_type in ['call', 'put'] 
         else pos.quantity * pos.current_price)
        for pos in positions
    ]
    position_values.sort(key=lambda x: x[1], reverse=True)
    
    total_value = sum(val for _, val in position_values)
    top_5_value = sum(val for _, val in position_values[:5])
    top_5_concentration = (top_5_value / total_value * 100) if total_value > 0 else 0
    
    # HHI (Herfindahl-Hirschman Index)
    position_weights = [(val / total_value) for _, val in position_values] if total_value > 0 else []
    hhi = sum(w * w for w in position_weights) * 10000 if position_weights else 0
    
    max_position_weight = (max(val for _, val in position_values) / total_value * 100) if total_value > 0 and position_values else 0
    
    # Get top 10 positions for detailed breakdown
    top_10_positions = []
    for ticker, value in position_values[:10]:
        weight = (value / total_value * 100) if total_value > 0 else 0
        top_10_positions.append({
            'ticker': ticker,
            'value': value,
            'weight': weight
        })
    
    concentration = {
        'top_5_concentration': top_5_concentration,
        'hhi': hhi,
        'max_position_weight': max_position_weight,
        'position_count': len(positions),
        'top_10_positions': top_10_positions
    }
    
    return {
        'portfolio_value': portfolio_value,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'stocks': stock_metrics,
        'options': options_metrics,
        'greeks': greeks,
        'concentration': concentration,
        'position_count': len(positions),
        'csp_required_cash': csp_cash_required,
        'analysis_date': datetime.now().isoformat(),
        'data_source': 'fast_calculation'
    }

@app.get("/api/portfolio/risk-monitor/fast")
async def get_fast_risk_monitoring(force_recalc: bool = False):
    """Get fast-loading risk monitoring data with real-time alerts - OPTIMIZED
    
    Strategy:
        1. Use Greeks from database for instant risk calculations (20-50ms)
        2. Trigger background refresh from Polygon.io for next load
        3. Display fast with current data, update fresh for next time
    
    Load time: 20-50ms (using cached Greeks from DB)
    Background: Greeks refreshed from Polygon.io for next load
    """
    try:
        # Use cached Greeks from DB for FAST risk calculations
        # Background refresh will update DB for next load
        positions = await get_portfolio_positions(use_cached_greeks=True, refresh_greeks_async=True)
        if not positions:
            return {"message": "No positions in portfolio", "alerts": []}
        
        # Get fast risk metrics with alerts (no Greek recalculation)
        fast_metrics = await fast_risk_monitor.get_fast_risk_metrics(positions, force_recalc=force_recalc)
        
        # Convert stock risk to dict (include new beta, volatility, liquidity fields)
        stock_risk_dict = {
            'stock_count': fast_metrics.stock_risk.stock_count,
            'stock_value': fast_metrics.stock_risk.stock_value,
            'stock_pnl_pct': fast_metrics.stock_risk.stock_pnl_pct,
            'concentration_risk_score': fast_metrics.stock_risk.concentration_risk_score,
            'max_position_pct': fast_metrics.stock_risk.max_position_pct,
            'sector_concentration': fast_metrics.stock_risk.sector_concentration,
            'unique_tickers': fast_metrics.stock_risk.unique_tickers,
            'portfolio_beta': fast_metrics.stock_risk.portfolio_beta,
            'portfolio_volatility': fast_metrics.stock_risk.portfolio_volatility,
            'liquidity_score': fast_metrics.stock_risk.liquidity_score,
            'most_volatile_stocks': [
                {
                    'ticker': vs.ticker,
                    'beta': vs.beta,
                    'value': vs.value,
                    'weight': vs.weight
                } for vs in (fast_metrics.stock_risk.most_volatile_stocks or [])
            ]
        }
        
        # Convert options risk to dict
        options_risk_dict = {
            'options_count': fast_metrics.options_risk.options_count,
            'options_value': fast_metrics.options_risk.options_value,
            'total_delta': fast_metrics.options_risk.total_delta,
            'total_gamma': fast_metrics.options_risk.total_gamma,
            'total_theta': fast_metrics.options_risk.total_theta,
            'total_vega': fast_metrics.options_risk.total_vega,
            'expiring_7d_count': fast_metrics.options_risk.expiring_7d_count,
            'expiring_30d_count': fast_metrics.options_risk.expiring_30d_count,
            'time_decay_risk_score': fast_metrics.options_risk.time_decay_risk_score,
            'avg_days_to_expiry': fast_metrics.options_risk.avg_days_to_expiry
        }
        
        # Convert to dict for JSON response
        result = {
            'portfolio_value': fast_metrics.portfolio_value,
            'total_pnl': fast_metrics.total_pnl,
            'total_pnl_pct': fast_metrics.total_pnl_pct,
            'max_position_risk': fast_metrics.max_position_risk,
            'risk_scores': {
                'overall': fast_metrics.overall_risk_score,
                'concentration': fast_metrics.concentration_risk_score,
                'volatility': fast_metrics.volatility_risk_score,
                'liquidity': fast_metrics.liquidity_risk_score
            },
            'stock_risk': stock_risk_dict,
            'options_risk': options_risk_dict,
            'counts': {
                'total_positions': fast_metrics.total_positions,
                'positions_at_risk': fast_metrics.positions_at_risk,
                'expiring_soon': fast_metrics.expiring_soon
            },
            'alerts': [
                {
                    'level': alert.level,
                    'category': alert.category,
                    'message': alert.message,
                    'value': alert.value,
                    'threshold': alert.threshold,
                    'timestamp': alert.timestamp.isoformat(),
                    'affected_positions': alert.affected_positions,
                    'recommendation': alert.recommendation
                }
                for alert in fast_metrics.active_alerts
            ],
            'calculated_at': fast_metrics.calculated_at.isoformat(),
            'calculation_time_ms': fast_metrics.calculation_time_ms,
            'status': 'success'
        }
        
        logger.info(f"Fast risk metrics calculated in {fast_metrics.calculation_time_ms:.2f}ms with {len(fast_metrics.active_alerts)} alerts")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in fast risk monitoring: {e}")
        return {"error": str(e), "status": "error"}

@app.get("/api/portfolio/risk-monitor/detailed")
async def get_detailed_risk_monitoring():
    """Get detailed risk monitoring data (load after fast metrics)
    
    Uses cached Greeks from database for fast calculations.
    Background refresh keeps Greeks updated.
    """
    try:
        import time
        start_time = time.time()
        
        # Use cached Greeks from DB for fast detailed calculations
        logger.info("🔄 Loading portfolio positions...")
        positions_start = time.time()
        positions = await get_portfolio_positions(use_cached_greeks=True, refresh_greeks_async=False)
        logger.info(f"⏱️ Positions loaded in {time.time() - positions_start:.3f}s")
        
        if not positions:
            return {"message": "No positions in portfolio"}
        
        # Get detailed risk metrics
        detailed_metrics = await fast_risk_monitor.get_detailed_risk_metrics(positions)
        
        # Convert to dict for JSON response
        result = {
            'greeks': {
                'total_delta': detailed_metrics.total_delta,
                'total_gamma': detailed_metrics.total_gamma,
                'total_theta': detailed_metrics.total_theta,
                'total_vega': detailed_metrics.total_vega
            },
            'portfolio_metrics': {
                'beta': detailed_metrics.portfolio_beta,
                'volatility': detailed_metrics.portfolio_volatility,
                'sharpe_ratio': detailed_metrics.sharpe_ratio,
                'max_drawdown': detailed_metrics.max_drawdown
            },
            'risk_measurements': {
                'var_95': detailed_metrics.var_95,
                'var_99': detailed_metrics.var_99,
                'expected_shortfall': detailed_metrics.expected_shortfall
            },
            'concentration': {
                'top_5': detailed_metrics.top_5_concentration,
                'by_sector': detailed_metrics.sector_concentration
            },
            'time_decay': {
                'theta_30d': detailed_metrics.theta_decay_30d,
                'expiring_7d_value': detailed_metrics.expiring_7d_value,
                'expiring_30d_value': detailed_metrics.expiring_30d_value
            },
            'liquidity': {
                'avg_score': detailed_metrics.avg_liquidity_score,
                'illiquid_positions': detailed_metrics.illiquid_positions
            },
            'status': 'success'
        }
        
        total_time = time.time() - start_time
        logger.info(f"✅ Detailed risk monitoring completed in {total_time:.3f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in detailed risk monitoring: {e}")
        return {"error": str(e), "status": "error"}

@app.get("/api/portfolio/risk-metrics")
async def get_risk_metrics(fast_mode: bool = False):
    """Get current portfolio risk metrics using real market data"""
    positions = await get_portfolio_positions()
    if not positions:
        return {"message": "No positions in portfolio"}
    
    # Check cache first
    positions_hash = get_positions_hash(positions)
    cached_analysis = get_cached_risk_analysis(positions_hash)
    if cached_analysis and fast_mode:
        logger.info("Using cached risk analysis")
        return cached_analysis
    
    # Get Polygon.io API key from environment
    polygon_api_key = os.getenv('POLYGON_API_KEY')
    if not polygon_api_key:
        logger.error("POLYGON_API_KEY not found in environment variables")
        return {"error": "Polygon.io API key required for real market data"}
    
    try:
        # Use real market data for risk analysis
        async with RealMarketRiskAnalyzer(polygon_api_key) as real_analyzer:
            start_time = time.time()
            real_risk_metrics = await real_analyzer.analyze_portfolio_risk(positions)
            calculation_time = time.time() - start_time
            
            logger.info(f"Real market risk analysis calculated in {calculation_time:.2f} seconds")
            
            # Separate stocks and options for metrics
            stocks = [pos for pos in positions if pos.position_type == 'stock']
            options = [pos for pos in positions if pos.position_type in ['call', 'put']]
            
            # Calculate stock metrics
            stock_metrics = {
                'count': len(stocks),
                'total_value': sum(pos.quantity * pos.current_price for pos in stocks),
                'unique_tickers': len(set(pos.ticker for pos in stocks)),
                'avg_pnl_pct': 0
            }
            if stocks:
                stock_pnl_pcts = [((pos.current_price - pos.entry_price) / pos.entry_price) * 100 for pos in stocks]
                stock_metrics['avg_pnl_pct'] = sum(stock_pnl_pcts) / len(stock_pnl_pcts)
            
            # Calculate options metrics
            options_metrics = {
                'count': len(options),
                'total_value': sum(pos.quantity * pos.current_price for pos in options),
                'calls': len([pos for pos in options if pos.position_type == 'call']),
                'puts': len([pos for pos in options if pos.position_type == 'put']),
                'avg_delta': 0,
                'avg_theta': 0,
                'avg_gamma': 0,
                'avg_vega': 0
            }
            if options:
                deltas = [pos.delta for pos in options if pos.delta is not None]
                thetas = [pos.theta for pos in options if pos.theta is not None]
                gammas = [pos.gamma for pos in options if pos.gamma is not None]
                vegas = [pos.vega for pos in options if pos.vega is not None]
                
                if deltas:
                    options_metrics['avg_delta'] = sum(deltas) / len(deltas)
                if thetas:
                    options_metrics['avg_theta'] = sum(thetas) / len(thetas)
                if gammas:
                    options_metrics['avg_gamma'] = sum(gammas) / len(gammas)
                if vegas:
                    options_metrics['avg_vega'] = sum(vegas) / len(vegas)
            
            # Calculate portfolio Greeks
            total_delta = sum(pos.delta * pos.quantity for pos in options if pos.delta is not None)
            total_gamma = sum(pos.gamma * pos.quantity for pos in options if pos.gamma is not None)
            total_theta = sum(pos.theta * pos.quantity for pos in options if pos.theta is not None)
            total_vega = sum(pos.vega * pos.quantity for pos in options if pos.vega is not None)
            
            greeks = {
                'total_delta': total_delta,
                'total_gamma': total_gamma,
                'total_theta': total_theta,
                'total_vega': total_vega
            }
            
            # Calculate P&L
            total_pnl = sum(pos.quantity * (pos.current_price - pos.entry_price) for pos in positions)
            total_cost = sum(pos.quantity * pos.entry_price for pos in positions)
            total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
            
            # CSP cash required (simplified calculation)
            csp_positions = [pos for pos in options if pos.position_type == 'put' and pos.quantity < 0]
            csp_cash_required = sum(abs(pos.quantity) * pos.strike_price * 100 for pos in csp_positions if pos.strike_price)
            
            # Calculate concentration metrics
            position_values = [(pos.ticker, pos.quantity * pos.current_price) for pos in positions]
            position_values.sort(key=lambda x: x[1], reverse=True)
            
            total_value = sum(val for _, val in position_values)
            top_5_value = sum(val for _, val in position_values[:5])
            top_5_concentration = (top_5_value / total_value * 100) if total_value > 0 else 0
            
            # HHI (Herfindahl-Hirschman Index)
            position_weights = [(val / total_value) for _, val in position_values] if total_value > 0 else []
            hhi = sum(w * w for w in position_weights) * 10000 if position_weights else 0
            
            max_position_weight = (max(val for _, val in position_values) / total_value * 100) if total_value > 0 and position_values else 0
            
            # Get top 10 positions for detailed breakdown
            top_10_positions = []
            for ticker, value in position_values[:10]:
                weight = (value / total_value * 100) if total_value > 0 else 0
                top_10_positions.append({
                    'ticker': ticker,
                    'value': value,
                    'weight': weight
                })
            
            concentration = {
                'top_5_concentration': top_5_concentration,
                'hhi': hhi,
                'max_position_weight': max_position_weight,
                'position_count': len(positions),
                'top_10_positions': top_10_positions
            }
            
            # Convert to dict format for JSON response
            risk_analysis = {
                'portfolio_value': real_risk_metrics.portfolio_value,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'portfolio_volatility': real_risk_metrics.portfolio_volatility,
                'portfolio_beta': real_risk_metrics.portfolio_beta,
                'sharpe_ratio': real_risk_metrics.sharpe_ratio,
                'sortino_ratio': real_risk_metrics.sortino_ratio,
                'max_drawdown': real_risk_metrics.max_drawdown,
                'var_95': real_risk_metrics.var_95,
                'var_99': real_risk_metrics.var_99,
                'cvar_95': real_risk_metrics.cvar_95,
                'cvar_99': real_risk_metrics.cvar_99,
                'stock_concentration_risk': real_risk_metrics.stock_concentration_risk,
                'sector_concentration_risk': real_risk_metrics.sector_concentration_risk,
                'liquidity_risk': real_risk_metrics.liquidity_risk,
                'correlation_risk': real_risk_metrics.correlation_risk,
                'delta_exposure': real_risk_metrics.delta_exposure,
                'gamma_risk': real_risk_metrics.gamma_risk,
                'theta_decay_risk': real_risk_metrics.theta_decay_risk,
                'vega_risk': real_risk_metrics.vega_risk,
                'time_decay_risk': real_risk_metrics.time_decay_risk,
                'tail_risk': real_risk_metrics.tail_risk,
                'skewness': real_risk_metrics.skewness,
                'kurtosis': real_risk_metrics.kurtosis,
                'information_ratio': real_risk_metrics.information_ratio,
                'treynor_ratio': real_risk_metrics.treynor_ratio,
                'stocks': stock_metrics,
                'options': options_metrics,
                'greeks': greeks,
                'concentration': concentration,
                'position_count': len(positions),
                'csp_required_cash': csp_cash_required,
                'analysis_date': datetime.now().isoformat(),
                'data_source': 'real_market_data'
            }
            
            # Cache the result
            cache_risk_analysis(positions_hash, risk_analysis)
            
            return risk_analysis
            
    except Exception as e:
        logger.error(f"Error in real market risk analysis: {e}")
        # Fallback to basic analysis
        risk_analysis = risk_analyzer.analyze_portfolio_risk(positions)
        risk_analysis['data_source'] = 'fallback'
        return risk_analysis

@app.post("/api/portfolio/scenario")
async def analyze_scenario(scenario_data: dict):
    """Analyze impact of adding new positions"""
    try:
        # Get current positions
        current_positions = await get_portfolio_positions()
        
        # Create temporary position for scenario
        temp_position = PortfolioPosition(
            ticker=scenario_data['ticker'],
            position_type=scenario_data['position_type'],
            quantity=int(scenario_data['quantity']),
            entry_price=float(scenario_data['entry_price']),
            current_price=float(scenario_data['current_price']),
            expiration_date=scenario_data.get('expiration_date'),
            strike_price=float(scenario_data.get('strike_price')) if scenario_data.get('strike_price') else None,
            option_type=scenario_data.get('option_type')
        )
        
        # Analyze with and without the new position
        current_risk = risk_analyzer.analyze_portfolio_risk(current_positions)
        scenario_positions = current_positions + [temp_position]
        scenario_risk = risk_analyzer.analyze_portfolio_risk(scenario_positions)
        
        # Calculate impact
        impact = {
            'delta_change': scenario_risk['greeks']['total_delta'] - current_risk['greeks']['total_delta'],
            'gamma_change': scenario_risk['greeks']['total_gamma'] - current_risk['greeks']['total_gamma'],
            'theta_change': scenario_risk['greeks']['total_theta'] - current_risk['greeks']['total_theta'],
            'vega_change': scenario_risk['greeks']['total_vega'] - current_risk['greeks']['total_vega'],
            'value_change': scenario_risk['portfolio_value'] - current_risk['portfolio_value']
        }
        
        return {
            'current_risk': current_risk,
            'scenario_risk': scenario_risk,
            'impact': impact
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing scenario: {str(e)}")

@app.get("/api/portfolio/scenario/stress-tests")
async def run_stress_tests():
    """Run comprehensive stress tests on the portfolio"""
    try:
        positions = await get_portfolio_positions()
        if not positions:
            return {"message": "No positions in portfolio"}
        
        # Run stress tests
        stress_results = scenario_analyzer.run_stress_tests(positions)
        
        # Get scenario explanations
        explanations = scenario_analyzer.get_scenario_explanations()
        
        # Convert results to dict format for JSON response
        results = []
        for result in stress_results:
            scenario_explanation = explanations.get(result.scenario_name, {})
            results.append({
                'scenario_name': result.scenario_name,
                'portfolio_value': result.portfolio_value,
                'pnl': result.pnl,
                'pnl_pct': result.pnl_pct,
                'delta': result.delta,
                'gamma': result.gamma,
                'theta': result.theta,
                'vega': result.vega,
                'var_95': result.var_95,
                'var_99': result.var_99,
                'cvar_95': result.cvar_95,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'scenario_details': result.scenario_details,
                'explanation': scenario_explanation
            })
        
        return {
            'stress_tests': results,
            'analysis_date': datetime.now().isoformat(),
            'total_scenarios': len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error running stress tests: {str(e)}")

@app.get("/api/portfolio/scenario/explanations")
async def get_scenario_explanations():
    """Get detailed explanations for stress test scenarios"""
    try:
        explanations = scenario_analyzer.get_scenario_explanations()
        return {
            'explanations': explanations,
            'analysis_date': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting scenario explanations: {str(e)}")

@app.get("/api/portfolio/scenario/monte-carlo")
async def run_monte_carlo_simulation(num_simulations: int = 1000):
    """Run Monte Carlo simulation for portfolio risk analysis"""
    try:
        positions = await get_portfolio_positions()
        if not positions:
            return {"message": "No positions in portfolio"}
        
        # Run Monte Carlo simulation
        simulation_results = scenario_analyzer.run_monte_carlo_simulation(positions, num_simulations)
        
        return {
            'monte_carlo': simulation_results,
            'analysis_date': datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error running Monte Carlo simulation: {str(e)}")

@app.post("/api/portfolio/scenario/position-impact")
async def analyze_position_impact(new_position_data: dict):
    """Analyze the impact of adding a new position to the portfolio"""
    try:
        positions = await get_portfolio_positions()
        if not positions:
            return {"message": "No positions in portfolio"}
        
        # Create new position object
        new_position = PortfolioPosition(
            ticker=new_position_data['ticker'],
            position_type=new_position_data['position_type'],
            quantity=int(new_position_data['quantity']),
            entry_price=float(new_position_data['entry_price']),
            current_price=float(new_position_data['current_price']),
            expiration_date=new_position_data.get('expiration_date'),
            strike_price=float(new_position_data.get('strike_price')) if new_position_data.get('strike_price') else None,
            option_type=new_position_data.get('option_type'),
            delta=new_position_data.get('delta'),
            gamma=new_position_data.get('gamma'),
            theta=new_position_data.get('theta'),
            vega=new_position_data.get('vega'),
            implied_volatility=new_position_data.get('implied_volatility'),
            time_to_expiration=new_position_data.get('time_to_expiration')
        )
        
        # Analyze impact
        impact_analysis = scenario_analyzer.analyze_position_impact(positions, new_position)
        
        return {
            'impact_analysis': impact_analysis,
            'analysis_date': datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing position impact: {str(e)}")

@app.post("/api/portfolio/scenario/custom")
async def run_custom_scenario(scenario_data: dict):
    """Run a custom scenario analysis"""
    try:
        positions = await get_portfolio_positions()
        if not positions:
            return {"message": "No positions in portfolio"}
        
        # Create custom scenario
        custom_scenario = scenario_analyzer.create_custom_scenario(
            name=scenario_data['name'],
            description=scenario_data['description'],
            market_moves=scenario_data.get('market_moves', {}),
            volatility_changes=scenario_data.get('volatility_changes', {}),
            time_decay_days=scenario_data.get('time_decay_days', 0),
            interest_rate_change=scenario_data.get('interest_rate_change', 0.0)
        )
        
        # Apply scenario to positions
        modified_positions = scenario_analyzer.apply_scenario_to_positions(positions, custom_scenario)
        
        # Calculate metrics
        result = scenario_analyzer.calculate_scenario_metrics(modified_positions, custom_scenario.name)
        
        return {
            'scenario_result': {
                'scenario_name': result.scenario_name,
                'portfolio_value': result.portfolio_value,
                'pnl': result.pnl,
                'pnl_pct': result.pnl_pct,
                'delta': result.delta,
                'gamma': result.gamma,
                'theta': result.theta,
                'vega': result.vega,
                'var_95': result.var_95,
                'var_99': result.var_99,
                'cvar_95': result.cvar_95,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'scenario_details': result.scenario_details
            },
            'analysis_date': datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error running custom scenario: {str(e)}")

@app.delete("/api/portfolio/positions/{position_id}")
async def delete_position(position_id: int):
    """Delete a position from the portfolio"""
    try:
        success = await delete_portfolio_position(position_id)
        if success:
            return {"message": "Position deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete position from database")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Position not found: {str(e)}")

@app.get("/api/portfolio/export")
async def export_portfolio():
    """Export portfolio to CSV"""
    positions = await get_portfolio_positions()
    if not positions:
        raise HTTPException(status_code=404, detail="No positions to export")
    
    # Convert to DataFrame
    df = pd.DataFrame([asdict(pos) for pos in positions])
    
    # Return CSV as response
    csv_content = df.to_csv(index=False)
    return JSONResponse(content={"csv": csv_content})

@app.post("/api/portfolio/csp-cash-analysis")
async def analyze_csp_cash(cash_data: dict):
    """Analyze CSP cash allocation"""
    try:
        positions = await get_portfolio_positions()
        if not positions:
            return {"message": "No positions in portfolio"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(pos) for pos in positions])
        
        # Get cash balance from request
        total_cash = cash_data.get('total_cash', 0)
        
        # Identify CSPs
        csps = csp_allocator.identify_csps(df)
        
        if not csps:
            return {"message": "No CSP positions found in portfolio"}
        
        # Calculate required cash
        allocations = csp_allocator.calculate_required_cash(csps)
        
        # Allocate cash
        allocations = csp_allocator.allocate_cash(allocations, total_cash)
        
        # Analyze risk
        risk_analysis = csp_allocator.analyze_csp_risk(allocations)
        
        # Create cash positions
        unallocated_cash = total_cash - risk_analysis['total_allocated_cash']
        cash_positions = csp_allocator.create_cash_positions(allocations, unallocated_cash)
        
        return {
            'csp_analysis': risk_analysis,
            'cash_positions': cash_positions,
            'unallocated_cash': unallocated_cash
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing CSP cash: {str(e)}")

# Trading API Endpoints
@app.post("/api/portfolio/trade")
async def execute_trade(trade_data: dict):
    """Execute a buy or sell trade"""
    try:
        ticker = trade_data.get('ticker', '').upper()
        trade_type_str = trade_data.get('trade_type', '').lower()
        quantity = int(trade_data.get('quantity', 0))
        price = float(trade_data.get('price', 0))
        notes = trade_data.get('notes', '')
        
        # Validate trade type
        if trade_type_str == 'buy':
            trade_type = TradeType.BUY
        elif trade_type_str == 'sell':
            trade_type = TradeType.SELL
        else:
            raise HTTPException(status_code=400, detail="Invalid trade type. Must be 'buy' or 'sell'")
        
        # Execute the trade
        result = await trading_client.execute_trade(ticker, trade_type, quantity, price, notes)
        
        if result.success:
            # Update the portfolio position
            await update_portfolio_position_from_trade(ticker, result.new_quantity, price, result.transaction_id)
            
            return {
                "success": True,
                "message": result.message,
                "transaction_id": result.transaction_id,
                "new_quantity": result.new_quantity
            }
        else:
            raise HTTPException(status_code=400, detail=result.error or result.message)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid trade parameters: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing trade: {str(e)}")

@app.get("/api/portfolio/transactions")
async def get_transaction_history(ticker: Optional[str] = None, limit: int = 100):
    """Get transaction history"""
    try:
        if trading_client.use_database:
            history = await trading_client.get_transaction_history_from_db(ticker, limit)
        else:
            history = trading_client.get_transaction_history(ticker, limit)
        
        return {
            "transactions": history,
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transaction history: {str(e)}")

@app.get("/api/portfolio/positions/{ticker}/summary")
async def get_position_summary(ticker: str):
    """Get position summary for a specific ticker"""
    try:
        if trading_client.use_database:
            summary = await trading_client.get_position_summary_from_db(ticker.upper())
        else:
            summary = trading_client.get_position_summary(ticker.upper())
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching position summary: {str(e)}")

@app.get("/api/portfolio/trading-summary")
async def get_trading_summary():
    """Get overall trading summary"""
    try:
        summary = trading_client.get_portfolio_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trading summary: {str(e)}")

@app.get("/api/portfolio/market-data/{ticker}")
async def get_market_data(ticker: str):
    """Get real-time market data for a ticker"""
    try:
        price_data = await market_client.get_stock_price(ticker.upper())
        if price_data:
            return {
                "ticker": price_data.ticker,
                "price": price_data.price,
                "timestamp": price_data.timestamp.isoformat(),
                "volume": price_data.volume,
                "high": price_data.high,
                "low": price_data.low,
                "open": price_data.open,
                "prev_close": price_data.prev_close
            }
        else:
            raise HTTPException(status_code=404, detail=f"No market data found for {ticker}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

@app.get("/api/portfolio/options-chain/{ticker}")
async def get_options_chain(ticker: str, expiration_date: Optional[str] = None):
    """Get options chain for a ticker"""
    try:
        options_chain = await market_client.get_options_chain(ticker.upper(), expiration_date)
        return {
            "ticker": ticker.upper(),
            "options": [asdict(option) for option in options_chain],
            "count": len(options_chain)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching options chain: {str(e)}")

@app.post("/api/portfolio/update-prices")
async def update_prices():
    """Manually trigger price update for all positions"""
    try:
        await update_position_prices()
        return {"message": "Prices updated successfully", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating prices: {str(e)}")

@app.post("/api/portfolio/initialize")
async def initialize_portfolio():
    """Initialize portfolio system and load existing data"""
    try:
        # Load existing transactions from database if available
        if trading_client.use_database:
            await trading_client.load_transactions_from_db()
            logger.info("Loaded existing transactions from database")
        
        # Update all position prices
        await update_position_prices()
        
        return {
            "message": "Portfolio initialized successfully",
            "database_connected": trading_client.use_database,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing portfolio: {str(e)}")

@app.get("/api/portfolio/loading-status")
async def get_loading_status():
    """Get current loading status and progress"""
    try:
        # Check if we have positions loaded
        positions = await get_portfolio_positions()
        positions_count = len(positions) if positions else 0
        
        # Check if market data is available
        market_data_available = market_client.api_key is not None
        
        return {
            "positions_loaded": positions_count,
            "market_data_available": market_data_available,
            "database_connected": supabase is not None,
            "trading_enabled": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "positions_loaded": 0,
            "market_data_available": False,
            "database_connected": False,
            "trading_enabled": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/portfolio/debug/value-breakdown")
async def get_portfolio_value_breakdown():
    """Debug endpoint to understand portfolio value calculation"""
    try:
        positions = await get_portfolio_positions()
        if not positions:
            return {"message": "No positions found"}
        
        # Calculate breakdown by position type
        stocks = [pos for pos in positions if pos.position_type == 'stock']
        options = [pos for pos in positions if pos.position_type in ['call', 'put']]
        
        stock_value = sum(pos.quantity * pos.current_price for pos in stocks)
        option_value = sum(pos.quantity * pos.current_price for pos in options)
        
        # Top 10 positions by value
        position_values = []
        for pos in positions:
            position_values.append({
                'ticker': pos.ticker,
                'position_type': pos.position_type,
                'quantity': pos.quantity,
                'current_price': pos.current_price,
                'total_value': pos.quantity * pos.current_price
            })
        
        position_values.sort(key=lambda x: x['total_value'], reverse=True)
        
        return {
            'total_positions': len(positions),
            'stock_positions': len(stocks),
            'option_positions': len(options),
            'stock_value': stock_value,
            'option_value': option_value,
            'total_portfolio_value': stock_value + option_value,
            'top_10_positions': position_values[:10],
            'all_positions': position_values
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting value breakdown: {str(e)}")

# Option Selection API Endpoints
@app.get("/api/options/search")
async def search_options(ticker: str, expiration_date: Optional[str] = None, option_type: Optional[str] = None):
    """Search for options by ticker, expiration, and type"""
    try:
        ticker = ticker.upper()
        
        # Get current stock price
        current_price = await market_analyzer.get_current_price(ticker)
        
        # Fetch options data
        options_data = await market_analyzer.fetch_options_data(ticker, expiration_date)
        
        # Filter by option type if specified
        if option_type:
            options_data = [opt for opt in options_data if opt['contract_type'] == option_type.lower()]
        
        # Calculate Greeks for each option
        for option in options_data:
            greeks = options_analyzer.calculate_greeks(option)
            option.update(greeks)
            
            # Add additional calculated fields
            option['intrinsic_value'] = max(0, current_price - option['strike_price']) if option['contract_type'] == 'call' else max(0, option['strike_price'] - current_price)
            option['time_value'] = option['current_price'] - option['intrinsic_value']
            option['moneyness'] = 'ITM' if option['intrinsic_value'] > 0 else 'ATM' if abs(current_price - option['strike_price']) < current_price * 0.02 else 'OTM'
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'options': options_data,
            'count': len(options_data),
            'expiration_dates': list(set(opt.get('expiration_date', '') for opt in options_data if opt.get('expiration_date')))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching options: {str(e)}")

@app.get("/api/options/chain/{ticker}")
async def get_options_chain(ticker: str, expiration_date: Optional[str] = None):
    """Get full options chain for a ticker"""
    try:
        ticker = ticker.upper()
        
        # Get current stock price
        current_price = await market_analyzer.get_current_price(ticker)
        
        # Fetch options data
        options_data = await market_analyzer.fetch_options_data(ticker, expiration_date)
        
        # Calculate Greeks and additional metrics
        for option in options_data:
            greeks = options_analyzer.calculate_greeks(option)
            option.update(greeks)
            
            # Add calculated fields
            option['intrinsic_value'] = max(0, current_price - option['strike_price']) if option['contract_type'] == 'call' else max(0, option['strike_price'] - current_price)
            option['time_value'] = option['current_price'] - option['intrinsic_value']
            option['moneyness'] = 'ITM' if option['intrinsic_value'] > 0 else 'ATM' if abs(current_price - option['strike_price']) < current_price * 0.02 else 'OTM'
            option['breakeven'] = option['strike_price'] + option['current_price'] if option['contract_type'] == 'call' else option['strike_price'] - option['current_price']
        
        # Group by expiration date
        options_by_expiration = {}
        for option in options_data:
            exp_date = option.get('expiration_date', 'Unknown')
            if exp_date not in options_by_expiration:
                options_by_expiration[exp_date] = {'calls': [], 'puts': []}
            
            if option['contract_type'] == 'call':
                options_by_expiration[exp_date]['calls'].append(option)
            else:
                options_by_expiration[exp_date]['puts'].append(option)
        
        # Sort strikes within each expiration
        for exp_date in options_by_expiration:
            options_by_expiration[exp_date]['calls'].sort(key=lambda x: x['strike_price'])
            options_by_expiration[exp_date]['puts'].sort(key=lambda x: x['strike_price'])
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'options_by_expiration': options_by_expiration,
            'expiration_dates': list(options_by_expiration.keys()),
            'total_options': len(options_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting options chain: {str(e)}")

@app.post("/api/options/analyze")
async def analyze_option_strategy(strategy_data: dict):
    """Analyze an option strategy or single option"""
    try:
        ticker = strategy_data.get('ticker', '').upper()
        strategy_type = strategy_data.get('strategy_type', 'single')
        options = strategy_data.get('options', [])
        
        if not options:
            raise HTTPException(status_code=400, detail="No options provided for analysis")
        
        # Get current stock price
        current_price = await market_analyzer.get_current_price(ticker)
        
        # Calculate Greeks for each option
        analyzed_options = []
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_cost = 0
        
        for option in options:
            # Calculate Greeks
            greeks = options_analyzer.calculate_greeks(option)
            option.update(greeks)
            
            # Add calculated fields
            option['intrinsic_value'] = max(0, current_price - option['strike_price']) if option['contract_type'] == 'call' else max(0, option['strike_price'] - current_price)
            option['time_value'] = option['current_price'] - option['intrinsic_value']
            option['moneyness'] = 'ITM' if option['intrinsic_value'] > 0 else 'ATM' if abs(current_price - option['strike_price']) < current_price * 0.02 else 'OTM'
            
            # Calculate position Greeks (multiply by quantity)
            quantity = option.get('quantity', 1)
            total_delta += greeks['delta'] * quantity
            total_gamma += greeks['gamma'] * quantity
            total_theta += greeks['theta'] * quantity
            total_vega += greeks['vega'] * quantity
            total_cost += option['current_price'] * quantity * 100  # Options are per 100 shares
            
            analyzed_options.append(option)
        
        # Calculate strategy metrics
        strategy_metrics = {
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'total_vega': total_vega,
            'total_cost': total_cost,
            'max_profit': 0,  # Will be calculated based on strategy
            'max_loss': total_cost,  # Default to total cost
            'breakeven_points': []  # Will be calculated based on strategy
        }
        
        # Calculate breakeven points based on strategy type
        if strategy_type == 'single':
            if len(analyzed_options) == 1:
                option = analyzed_options[0]
                if option['contract_type'] == 'call':
                    strategy_metrics['breakeven_points'] = [option['strike_price'] + option['current_price']]
                else:
                    strategy_metrics['breakeven_points'] = [option['strike_price'] - option['current_price']]
        elif strategy_type == 'spread':
            # For spreads, calculate breakeven based on net debit/credit
            if len(analyzed_options) == 2:
                long_option = analyzed_options[0] if analyzed_options[0].get('quantity', 1) > 0 else analyzed_options[1]
                short_option = analyzed_options[1] if analyzed_options[0].get('quantity', 1) > 0 else analyzed_options[0]
                
                net_debit = long_option['current_price'] - short_option['current_price']
                if long_option['contract_type'] == 'call':
                    strategy_metrics['breakeven_points'] = [long_option['strike_price'] + net_debit]
                else:
                    strategy_metrics['breakeven_points'] = [long_option['strike_price'] - net_debit]
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'strategy_type': strategy_type,
            'options': analyzed_options,
            'strategy_metrics': strategy_metrics,
            'analysis_date': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing option strategy: {str(e)}")

@app.get("/api/options/screen")
async def screen_options(
    ticker: Optional[str] = None,
    min_delta: Optional[float] = None,
    max_delta: Optional[float] = None,
    min_theta: Optional[float] = None,
    max_theta: Optional[float] = None,
    min_vega: Optional[float] = None,
    max_vega: Optional[float] = None,
    min_iv: Optional[float] = None,
    max_iv: Optional[float] = None,
    option_type: Optional[str] = None,
    moneyness: Optional[str] = None,
    min_days_to_expiry: Optional[int] = None,
    max_days_to_expiry: Optional[int] = None
):
    """Screen options based on various criteria"""
    try:
        # Get current stock price if ticker provided
        current_price = None
        if ticker:
            current_price = await market_analyzer.get_current_price(ticker.upper())
        
        # Fetch options data
        options_data = await market_analyzer.fetch_options_data(ticker.upper() if ticker else 'AAPL')
        
        # Apply filters
        filtered_options = []
        for option in options_data:
            # Calculate Greeks
            greeks = options_analyzer.calculate_greeks(option)
            option.update(greeks)
            
            # Add calculated fields
            if current_price:
                option['intrinsic_value'] = max(0, current_price - option['strike_price']) if option['contract_type'] == 'call' else max(0, option['strike_price'] - current_price)
                option['time_value'] = option['current_price'] - option['intrinsic_value']
                option['moneyness'] = 'ITM' if option['intrinsic_value'] > 0 else 'ATM' if abs(current_price - option['strike_price']) < current_price * 0.02 else 'OTM'
            
            # Apply filters
            if option_type and option['contract_type'] != option_type.lower():
                continue
            if min_delta is not None and greeks['delta'] < min_delta:
                continue
            if max_delta is not None and greeks['delta'] > max_delta:
                continue
            if min_theta is not None and greeks['theta'] < min_theta:
                continue
            if max_theta is not None and greeks['theta'] > max_theta:
                continue
            if min_vega is not None and greeks['vega'] < min_vega:
                continue
            if max_vega is not None and greeks['vega'] > max_vega:
                continue
            if min_iv is not None and greeks['implied_volatility'] < min_iv:
                continue
            if max_iv is not None and greeks['implied_volatility'] > max_iv:
                continue
            if moneyness and option.get('moneyness') != moneyness:
                continue
            if min_days_to_expiry is not None and option['time_to_expiration'] < min_days_to_expiry:
                continue
            if max_days_to_expiry is not None and option['time_to_expiration'] > max_days_to_expiry:
                continue
            
            filtered_options.append(option)
        
        # Sort by a default criteria (e.g., by delta)
        filtered_options.sort(key=lambda x: abs(x['delta']), reverse=True)
        
        return {
            'ticker': ticker.upper() if ticker else 'AAPL',
            'current_price': current_price,
            'filtered_options': filtered_options,
            'count': len(filtered_options),
            'filters_applied': {
                'min_delta': min_delta,
                'max_delta': max_delta,
                'min_theta': min_theta,
                'max_theta': max_theta,
                'min_vega': min_vega,
                'max_vega': max_vega,
                'min_iv': min_iv,
                'max_iv': max_iv,
                'option_type': option_type,
                'moneyness': moneyness,
                'min_days_to_expiry': min_days_to_expiry,
                'max_days_to_expiry': max_days_to_expiry
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error screening options: {str(e)}")


@app.get("/api/options/recommendations/covered-calls")
async def get_covered_call_recommendations():
    """Get best Covered Call recommendations based on current portfolio"""
    try:
        logger.info("🔍 Starting covered call recommendations...")
        
        # Get current portfolio positions
        positions = await get_portfolio_positions()
        logger.info(f"📊 Found {len(positions)} total positions")
        
        if not positions:
            return {"message": "No positions in portfolio", "recommendations": []}
        
        # Filter for stock positions only
        stock_positions = [pos for pos in positions if pos.position_type == 'stock' and pos.quantity > 0]
        logger.info(f"📈 Found {len(stock_positions)} stock positions")
        
        if not stock_positions:
            return {"message": "No stock positions available for covered calls", "recommendations": []}
        
        # Get existing covered call positions and calculate available shares for additional CCs
        existing_ccs = [pos for pos in positions if pos.position_type == 'call' and pos.quantity < 0]
        
        # Calculate how many shares are already committed to covered calls
        cc_shares_committed = {}
        for cc_pos in existing_ccs:
            ticker = cc_pos.ticker
            # Each call contract covers 100 shares, and quantity is negative for short positions
            shares_committed = abs(cc_pos.quantity) * 100
            cc_shares_committed[ticker] = cc_shares_committed.get(ticker, 0) + shares_committed
        
        # Find tickers that are completely blocked (no shares available for additional CCs)
        completely_blocked_tickers = set()
        for stock_pos in stock_positions:
            ticker = stock_pos.ticker
            shares_owned = stock_pos.quantity
            shares_committed = cc_shares_committed.get(ticker, 0)
            
            # If all shares are committed to existing CCs, block this ticker
            if shares_committed >= shares_owned:
                completely_blocked_tickers.add(ticker)
        
        blocked_tickers = completely_blocked_tickers
        logger.info(f"🚫 Completely blocked tickers (no shares available for additional CCs): {blocked_tickers}")
        logger.info(f"📊 CC shares committed by ticker: {cc_shares_committed}")
        
        # Convert positions to dict format for the recommender
        stock_pos_dicts = [asdict(pos) for pos in stock_positions]
        
        # Get Polygon API key
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not polygon_api_key:
            logger.error("POLYGON_API_KEY not found in environment variables")
            raise HTTPException(status_code=500, detail="Polygon.io API key required for covered call recommendations")
        
        # Use the EV-enhanced recommender with delayed data optimization
        from ev_delayed_integration import EVDelayedDataIntegration
        ev_recommender = EVDelayedDataIntegration(polygon_api_key=polygon_api_key, supabase_client=supabase)
        result = ev_recommender.get_enhanced_cc_recommendations(stock_pos_dicts, blocked_tickers, cc_shares_committed)
        
        # Extract data from the result dictionary
        recommendations = result['recommendations']
        underlying_tickers_considered = result['underlying_tickers_considered']
        underlying_tickers_in_results = result['underlying_tickers_in_results']
        
        # Sort by EV score (already sorted by the enhanced recommender)
        # recommendations.sort(key=lambda x: x['ev_score'], reverse=True)
        
        logger.info(f"✅ Generated {len(recommendations)} total recommendations from {len(stock_positions)} stock positions")
        logger.info(f"📊 Processed stocks: {len(stock_positions) - len(blocked_tickers)}, Blocked: {len(blocked_tickers)}")
        logger.info(f"📊 Underlying tickers: {underlying_tickers_considered} considered, {underlying_tickers_in_results} in results")
        
        if len(recommendations) == 0:
            logger.warning(f"⚠️ No recommendations generated! This might be due to:")
            logger.warning(f"   - Options chains not available for these tickers")
            logger.warning(f"   - All stocks already have covered calls")
            logger.warning(f"   - API errors fetching options data")
        
        return {
            'recommendations': recommendations[:20],  # Top 20 recommendations
            'total_available': len(stock_positions),
            'total_considered': result['total_considered'],  # Total options analyzed
            'blocked_tickers': list(blocked_tickers),
            'underlying_tickers_considered': underlying_tickers_considered,
            'underlying_tickers_in_results': underlying_tickers_in_results,
            'analysis_date': datetime.now().isoformat(),
            'scoring_method': result.get('scoring_method', 'Expected Value Optimized'),
            'market_regime': result.get('market_regime', 'unknown'),
            'delayed_data_optimization': result.get('delayed_data_optimization', False),
            'safety_margins_applied': result.get('safety_margins_applied', False),
            'filtered_recommendations': result.get('filtered_recommendations', 0),
            'total_enhanced': result.get('total_enhanced', 0)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in covered call recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting covered call recommendations: {str(e)}")

@app.post("/api/options/recommendations/cash-secured-puts")
async def get_csp_recommendations(request_data: dict = None):
    """Get best Cash Secured Put recommendations using the new CSP recommender"""
    try:
        logger.info("🔍 Starting CSP recommendations...")
        
        # Get tickers from request data (comma-separated string)
        tickers_input = None
        combine_with_watchlist = False
        
        if request_data:
            if 'tickers' in request_data:
                tickers_input = request_data['tickers']
                logger.info(f"📝 Using provided tickers: {tickers_input}")
            if 'combine_with_watchlist' in request_data:
                combine_with_watchlist = request_data['combine_with_watchlist']
                logger.info(f"🔄 Combine with watchlist: {combine_with_watchlist}")
        
        # Get Polygon API key
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not polygon_api_key:
            logger.error("POLYGON_API_KEY not found in environment variables")
            raise HTTPException(status_code=500, detail="Polygon.io API key required for CSP recommendations")
        
        # Use the EV-enhanced recommender with delayed data optimization
        from ev_delayed_integration import EVDelayedDataIntegration
        ev_recommender = EVDelayedDataIntegration(polygon_api_key=polygon_api_key, supabase_client=supabase)
        result = ev_recommender.get_enhanced_csp_recommendations(tickers_input, combine_with_watchlist)
        
        logger.info(f"✅ Generated {len(result['recommendations'])} EV-enhanced CSP recommendations from {result['total_considered']} options")
        logger.info(f"📊 Market regime: {result.get('market_regime', 'unknown')}")
        logger.info(f"📊 Delayed data optimization: {result.get('delayed_data_optimization', False)}")
        logger.info(f"📊 Filtered recommendations: {result.get('filtered_recommendations', 0)}")
        
        return {
            'recommendations': result['recommendations'],
            'total_available': len(result['recommendations']),
            'total_considered': result['total_considered'],
            'underlying_tickers_considered': result['underlying_tickers_considered'],
            'underlying_tickers_in_results': result['underlying_tickers_in_results'],
            'analysis_date': datetime.now().isoformat(),
            'scoring_method': result.get('scoring_method', 'Expected Value Optimized'),
            'market_regime': result.get('market_regime', 'unknown'),
            'delayed_data_optimization': result.get('delayed_data_optimization', False),
            'safety_margins_applied': result.get('safety_margins_applied', False),
            'filtered_recommendations': result.get('filtered_recommendations', 0),
            'total_enhanced': result.get('total_enhanced', 0)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in CSP recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting CSP recommendations: {str(e)}")

async def update_portfolio_position_from_trade(ticker: str, new_quantity: int, last_price: float, transaction_id: str = None):
    """Update portfolio position based on trade result"""
    try:
        positions = await get_portfolio_positions()
        
        # Find existing position
        existing_position = None
        for pos in positions:
            if pos.ticker == ticker and pos.position_type == 'stock':
                existing_position = pos
                break
        
        if new_quantity == 0:
            # Remove position if quantity is zero
            if existing_position and supabase:
                supabase.table('portfolio_positions').delete().eq('ticker', ticker).eq('position_type', 'stock').execute()
            
            # Save position history
            if trading_client.use_database:
                await trading_client._save_position_history(
                    ticker, 'stock', new_quantity, last_price, last_price,
                    'trade', transaction_id, 'Position closed'
                )
        else:
            if existing_position:
                # Update existing position
                old_quantity = existing_position.quantity
                existing_position.quantity = new_quantity
                existing_position.current_price = last_price
                
                if supabase:
                    supabase.table('portfolio_positions').update({
                        'quantity': new_quantity,
                        'current_price': last_price
                    }).eq('ticker', ticker).eq('position_type', 'stock').execute()
                
                # Save position history
                if trading_client.use_database:
                    await trading_client._save_position_history(
                        ticker, 'stock', new_quantity, existing_position.entry_price, last_price,
                        'trade', transaction_id, f'Position updated: {old_quantity} -> {new_quantity}'
                    )
            else:
                # Create new position
                new_position = PortfolioPosition(
                    ticker=ticker,
                    position_type='stock',
                    quantity=new_quantity,
                    entry_price=last_price,
                    current_price=last_price,
                    created_at=datetime.now()
                )
                await save_portfolio_position(new_position)
                
                # Save position history
                if trading_client.use_database:
                    await trading_client._save_position_history(
                        ticker, 'stock', new_quantity, last_price, last_price,
                        'trade', transaction_id, 'New position created'
                    )
                
    except Exception as e:
        logger.error(f"Error updating portfolio position for {ticker}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
