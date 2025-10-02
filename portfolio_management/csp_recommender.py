"""
Cash Secured Puts (CSP) Recommender - Optimized for Speed and Accuracy
Includes parallel processing, batch operations, and advanced filtering
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from polygon import RESTClient
import os
import yfinance as yf
import pandas as pd
import numpy as np
import math
import pytz
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

# Optimized scoring weights for CSPs - prioritize high returns with technical/resistance support
SCORING_WEIGHTS = {
    'annualized_return': 0.30,  # Primary factor - high annualized returns
    'technical': 0.25,          # Technical indicators supporting put selling
    'resistance': 0.20,         # Support/resistance levels (strong support below strike)
    'probability': 0.15,        # Probability of profit (risk management)
    'greeks': 0.05,            # Greeks for risk assessment
    'ivr': 0.03,               # IV Rank (reduced - less important)
    'liquidity': 0.02,         # Open interest and volume (minimal)
    'cash_required': 0.00      # Remove cash penalty (not relevant for scoring)
}

# RSI constants
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

class CashSecuredPutRecommender:
    """Find best Cash Secured Put opportunities with optimized parallel processing"""
    
    def __init__(self, polygon_api_key: str = None, supabase_client = None):
        self.polygon_client = RESTClient(polygon_api_key or os.getenv('POLYGON_API_KEY'))
        self.supabase = supabase_client  # For IVR data and stock selection
        self.weights = SCORING_WEIGHTS
        self.technical_cache = {}  # Cache for technical indicators
        self.ivr_cache = {}  # Cache for IVR data
        self.option_cache = {}  # Cache for option chains
        self.cache_ttl = 86400  # 24 hours (1 day) cache for technical indicators
        self.option_cache_ttl = 300  # 5 minutes cache for option chains
        self.cache_date = datetime.now().date()  # Track cache date
        
        # Performance optimization settings
        self.max_concurrent_tickers = 10  # Limit concurrent ticker processing
        self.max_concurrent_options = 20  # Limit concurrent option processing
        
        # ETFs and bonds to skip (no options or not suitable for CSPs)
        self.skip_tickers = {
            'SCHD', 'JEPQ', 'JEPI', 'DIVO', 'QYLD', 'RYLD', 'XYLD',  # Income ETFs
            'VOO', 'SPY', 'QQQ', 'IWM', 'DIA',  # Index ETFs
            'BND', 'AGG', 'BNDX', 'HYG', 'LQD', 'MUB', 'VGIT', 'TLT',  # Bond ETFs
            'GLD', 'SLV', 'USO', 'UNG'  # Commodity ETFs
        }
    
    def _is_third_friday(self, date: datetime) -> bool:
        """Check if a date is the 3rd Friday of the month"""
        # Get the first day of the month
        first_day = date.replace(day=1)
        
        # Find the first Friday of the month
        first_friday = first_day
        while first_friday.weekday() != 4:  # Friday is weekday 4
            first_friday += pd.Timedelta(days=1)
        
        # The 3rd Friday is 14 days after the first Friday
        third_friday = first_friday + pd.Timedelta(days=14)
        
        return date.date() == third_friday.date()
    
    def _find_monthly_expirations(self, ticker: str, min_dte: int = 7, max_dte: int = 90) -> List[str]:
        """Find next 2 monthly expiration dates (3rd Friday of month)"""
        from datetime import timedelta
        
        today = datetime.now()
        min_date = today + timedelta(days=min_dte)
        max_date = today + timedelta(days=max_dte)
        
        monthly_expirations = []
        current_date = min_date
        
        # Search for monthly expirations
        while current_date <= max_date and len(monthly_expirations) < 2:
            if self._is_third_friday(current_date):
                monthly_expirations.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        logger.info(f"📅 Found {len(monthly_expirations)} monthly expirations: {monthly_expirations}")
        return monthly_expirations
    
    def get_recommendations(self, tickers_input: str = None, combine_with_watchlist: bool = False) -> Dict:
        """Get CSP recommendations with optimized parallel processing"""
        start_time = time.time()
        logger.info("🚀 Starting optimized CSP recommendations with parallel processing")
        
        # Get tickers to analyze
        tickers = self._get_tickers_to_analyze(tickers_input, combine_with_watchlist)
        
        if not tickers:
            logger.warning("⚠️  No tickers to analyze")
            return {'recommendations': [], 'total_considered': 0, 'processing_time': 0}
        
        logger.info(f"📊 Analyzing {len(tickers)} tickers with parallel processing")
        
        # Phase 1: Batch fetch IVR data for all tickers (5x faster)
        ivr_data = self._batch_fetch_ivr_data(tickers)
        
        # Phase 2: Parallel stock data fetching (3-4x faster)
        stock_data_results = self._parallel_fetch_stock_data(tickers)
        
        # Phase 3: Filter and process valid tickers
        valid_tickers = self._filter_valid_tickers(tickers, stock_data_results, ivr_data)
        
        if not valid_tickers:
            logger.warning("⚠️  No valid tickers after filtering")
            return {'recommendations': [], 'total_considered': 0, 'processing_time': time.time() - start_time}
        
        # Phase 4: Parallel option processing with pre-filtering
        recommendations = self._parallel_process_options(valid_tickers, stock_data_results, ivr_data)
        
        # Phase 5: Apply final filtering and sorting
        final_recommendations = self._apply_final_filtering(recommendations)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Completed in {processing_time:.2f}s - {len(final_recommendations)} recommendations from {len(recommendations)} options")
        
        # Count unique tickers processed for backward compatibility
        processed_tickers = len(set(rec['ticker'] for rec in recommendations))
        final_ticker_count = len(set(rec['ticker'] for rec in final_recommendations))
        
        return {
            'recommendations': final_recommendations,
            'total_considered': len(recommendations),
            'underlying_tickers_considered': processed_tickers,
            'underlying_tickers_in_results': final_ticker_count,
            'processing_time': processing_time,
            'tickers_analyzed': len(valid_tickers)
        }
    
    def _get_tickers_to_analyze(self, tickers_input: str = None, combine_with_watchlist: bool = False) -> List[str]:
        """Get tickers to analyze with optimized logic"""
        if tickers_input:
            # Parse provided tickers (comma or space separated)
            provided_tickers = [t.strip().upper() for t in tickers_input.replace(',', ' ').split() if t.strip()]
            logger.info(f"🎯 Using provided tickers: {provided_tickers}")
            
            if combine_with_watchlist:
                # For "all available" mode, also get popular/volatile tickers and combine
                popular_volatile_tickers = self._get_popular_volatile_tickers()
                logger.info(f"📊 Found {len(popular_volatile_tickers)} popular/volatile tickers from iv_history")
                
                # Combine provided tickers with popular/volatile tickers (remove duplicates)
                all_tickers = list(set(provided_tickers + popular_volatile_tickers))
                tickers = all_tickers
                logger.info(f"🔄 Combined mode: {len(provided_tickers)} provided + {len(popular_volatile_tickers)} popular/volatile = {len(tickers)} total")
            else:
                # Custom mode: use only provided tickers
                tickers = provided_tickers
                logger.info(f"🎯 Custom mode: analyzing only {len(tickers)} provided tickers")
        else:
            # Get popular and high volatility stocks from iv_history
            tickers = self._get_popular_volatile_tickers()
            logger.info(f"📊 Found {len(tickers)} popular/volatile tickers from iv_history")
        
        return tickers
    
    def _batch_fetch_ivr_data(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Batch fetch IVR data for all tickers (5x faster than individual queries)"""
        if not self.supabase:
            logger.warning("No Supabase client - skipping IVR batch fetch")
            return {ticker: None for ticker in tickers}
        
        logger.info(f"📊 Batch fetching IVR data for {len(tickers)} tickers")
        start_time = time.time()
        
        try:
            # Single query to get IVR data for all tickers
            result = self.supabase.table('iv_history').select('ticker, iv, date').in_('ticker', tickers).order('date', desc=True).execute()
            
            # Group by ticker and calculate IVR for each
            ivr_data = {}
            ticker_iv_history = {}
            
            for row in result.data:
                ticker = row['ticker']
                if ticker not in ticker_iv_history:
                    ticker_iv_history[ticker] = []
                ticker_iv_history[ticker].append(float(row['iv']))
            
            # Calculate IVR for each ticker
            for ticker in tickers:
                if ticker in ticker_iv_history and len(ticker_iv_history[ticker]) >= 20:
                    iv_values = ticker_iv_history[ticker][:252]  # Last 252 trading days
                    current_iv = iv_values[0]
                    iv_min = min(iv_values)
                    iv_max = max(iv_values)
                    
                    if iv_max > iv_min:
                        ivr = ((current_iv - iv_min) / (iv_max - iv_min)) * 100
                        ivr_data[ticker] = round(ivr, 1)
                    else:
                        ivr_data[ticker] = None
                else:
                    ivr_data[ticker] = None
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Batch IVR fetch completed in {elapsed:.2f}s")
            return ivr_data
            
        except Exception as e:
            logger.error(f"Error in batch IVR fetch: {e}")
            return {ticker: None for ticker in tickers}
    
    def _parallel_fetch_stock_data(self, tickers: List[str]) -> Dict[str, Optional[Dict]]:
        """Parallel fetch stock data for all tickers (3-4x faster)"""
        logger.info(f"📊 Parallel fetching stock data for {len(tickers)} tickers")
        start_time = time.time()
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tickers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self._get_stock_data, ticker): ticker 
                for ticker in tickers
            }
            
            # Collect results
            results = {}
            for future in future_to_ticker:
                ticker = future_to_ticker[future]
                try:
                    results[ticker] = future.result(timeout=30)  # 30 second timeout per ticker
                except Exception as e:
                    logger.warning(f"Error fetching stock data for {ticker}: {e}")
                    results[ticker] = None
        
        elapsed = time.time() - start_time
        valid_count = sum(1 for v in results.values() if v is not None)
        logger.info(f"✅ Parallel stock data fetch completed in {elapsed:.2f}s - {valid_count}/{len(tickers)} successful")
        
        return results
    
    def _filter_valid_tickers(self, tickers: List[str], stock_data_results: Dict[str, Optional[Dict]], 
                            ivr_data: Dict[str, Optional[float]]) -> List[str]:
        """Filter tickers that have valid stock data and are not ETFs"""
        valid_tickers = []
        
        for ticker in tickers:
            # Skip ETFs
            if ticker in self.skip_tickers:
                continue
            
            # Must have valid stock data
            if stock_data_results.get(ticker) is None:
                continue
            
            # Add IVR data to stock data if available
            if ivr_data.get(ticker) is not None:
                stock_data_results[ticker]['ivr'] = ivr_data[ticker]
            else:
                stock_data_results[ticker]['ivr'] = None
            
            valid_tickers.append(ticker)
        
        logger.info(f"✅ Filtered to {len(valid_tickers)} valid tickers from {len(tickers)} total")
        return valid_tickers
    
    def _parallel_process_options(self, valid_tickers: List[str], stock_data_results: Dict[str, Dict], 
                                ivr_data: Dict[str, Optional[float]]) -> List[Dict]:
        """Parallel process options for all valid tickers with pre-filtering"""
        logger.info(f"📊 Parallel processing options for {len(valid_tickers)} tickers")
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel option processing
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tickers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self._process_ticker_options, ticker, stock_data_results[ticker]): ticker 
                for ticker in valid_tickers
            }
            
            # Collect results
            all_recommendations = []
            for future in future_to_ticker:
                ticker = future_to_ticker[future]
                try:
                    ticker_recommendations = future.result(timeout=60)  # 60 second timeout per ticker
                    if ticker_recommendations:
                        all_recommendations.extend(ticker_recommendations)
                except Exception as e:
                    logger.warning(f"Error processing options for {ticker}: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Parallel option processing completed in {elapsed:.2f}s - {len(all_recommendations)} options found")
        
        return all_recommendations
    
    def _process_ticker_options(self, ticker: str, stock_data: Dict) -> List[Dict]:
        """Process options for a single ticker with optimized pre-filtering"""
        try:
            stock_price = stock_data['current_price']
            
            # Find monthly expirations
            monthly_expirations = self._find_monthly_expirations(ticker)
            if not monthly_expirations:
                return []
            
            # Get option contracts with caching
            all_put_contracts = self._get_cached_option_contracts(ticker, monthly_expirations)
            if not all_put_contracts:
                logger.debug(f"🔍 {ticker}: No put contracts found for expirations {monthly_expirations}")
                return []
            
            logger.debug(f"🔍 {ticker}: Found {len(all_put_contracts)} total put contracts")
            
            # Pre-filter contracts by strike range (fast filter) - Fixed to match CC logic
            lower_strike = stock_price * 0.80  # 20% below current price (more reasonable for puts)
            upper_strike = stock_price * 0.98  # 2% below current price (still below current price for puts)
            
            put_contracts = [
                c for c in all_put_contracts
                if lower_strike <= c.strike_price <= upper_strike
            ]
            
            logger.debug(f"🔍 {ticker}: After strike filtering ({lower_strike:.2f}-{upper_strike:.2f}): {len(put_contracts)} contracts")
            
            if not put_contracts:
                logger.debug(f"🔍 {ticker}: No contracts in strike range {lower_strike:.2f}-{upper_strike:.2f}")
                return []
            
            # Process contracts with optimized scoring
            recommendations = []
            processed_count = 0
            skipped_count = 0
            
            for contract in put_contracts:
                try:
                    recommendation = self._process_single_option(contract, stock_data, stock_price)
                    processed_count += 1
                    if recommendation:
                        recommendations.append(recommendation)
                        logger.debug(f"✅ {contract.underlying_ticker}: Added recommendation (delta={abs(recommendation['delta']):.3f}, dte={recommendation['days_to_expiration']}, strike={contract.strike_price})")
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.debug(f"Error processing option {contract.ticker}: {e}")
                    continue
            
            logger.debug(f"🔍 {ticker}: Processed {processed_count} contracts, {len(recommendations)} recommendations, {skipped_count} skipped")
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error processing ticker {ticker}: {e}")
            return []
    
    def _get_cached_option_contracts(self, ticker: str, expirations: List[str]) -> List:
        """Get option contracts with caching to reduce API calls"""
        cache_key = f"{ticker}_{expirations[0]}_{expirations[-1] if len(expirations) > 1 else expirations[0]}"
        
        # Check cache first
        if cache_key in self.option_cache:
            cached_data, cache_time = self.option_cache[cache_key]
            if time.time() - cache_time < self.option_cache_ttl:
                logger.debug(f"📦 Using cached option contracts for {ticker}")
                return cached_data
        
        # Fetch from API
        all_put_contracts = []
        for exp_date in expirations:
            try:
                contracts = list(self.polygon_client.list_options_contracts(
                    underlying_ticker=ticker,
                    expiration_date=exp_date,
                    contract_type='put',
                    limit=100  # Increased from 50 to 100
                ))
                all_put_contracts.extend(contracts)
            except Exception as e:
                logger.warning(f"Could not fetch contracts for {ticker} exp {exp_date}: {e}")
                continue
        
        # Cache the result
        self.option_cache[cache_key] = (all_put_contracts, time.time())
        
        return all_put_contracts
    
    def _process_single_option(self, contract, stock_data: Dict, stock_price: float) -> Optional[Dict]:
        """Process a single option with optimized calculations and pre-filtering"""
        try:
            # Get snapshot
            snapshot = self.polygon_client.get_snapshot_option(
                contract.underlying_ticker,
                contract.ticker
            )
            
            # Extract price and Greeks
            price = self._extract_option_price(snapshot)
            if not price or price == 0:
                logger.debug(f"🔍 {contract.underlying_ticker}: Skipped due to invalid price {price}")
                return None
            
            greeks = self._extract_greeks(snapshot)
            
            # Pre-filter by delta (fast filter before expensive calculations) - Original range for puts
            abs_delta = abs(greeks['delta'])
            if abs_delta < 0.14 or abs_delta > 0.45:  # Original delta range for puts (0.14-0.45)
                logger.debug(f"🔍 {contract.underlying_ticker}: Skipped due to delta {abs_delta:.3f} (outside 0.14-0.45 range)")
                return None
            
            # Calculate days to expiration
            et_tz = pytz.timezone('US/Eastern')
            now_et = datetime.now(et_tz)
            exp_date = datetime.fromisoformat(contract.expiration_date)
            exp_date_et = et_tz.localize(exp_date.replace(hour=16, minute=0, second=0))
            time_remaining = exp_date_et - now_et
            days_to_exp = max(1, time_remaining.days + (1 if time_remaining.seconds > 0 else 0))
            
            # Pre-filter by DTE
            if days_to_exp < 7 or days_to_exp > 90:
                logger.debug(f"🔍 {contract.underlying_ticker}: Skipped due to DTE {days_to_exp} (outside 7-90 range)")
                return None
            
            # Calculate metrics
            premium_per_contract = price * 100
            max_profit_per_contract = price * 100
            max_loss_per_contract = (contract.strike_price - price) * 100
            breakeven = contract.strike_price - price
            roi = (price / contract.strike_price) * 100
            annualized_return = roi * (365 / days_to_exp)
            
            # Calculate probability of profit
            probability_of_profit = self._calculate_probability_of_profit(
                greeks['delta'], days_to_exp, greeks['implied_volatility'],
                contract.strike_price, stock_price, stock_data
            )
            
            # Calculate scores with optimized methods
            scores = self._calculate_all_scores(stock_data, greeks, days_to_exp, contract, stock_price, annualized_return, probability_of_profit)
            
            return {
                'ticker': contract.underlying_ticker,
                'current_stock_price': stock_price,
                'strike_price': contract.strike_price,
                'premium': price,
                'premium_per_contract': premium_per_contract,
                'expiration_date': contract.expiration_date,
                'days_to_expiration': days_to_exp,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'implied_volatility': greeks['implied_volatility'],
                'moneyness': self._calculate_moneyness(stock_price, contract.strike_price),
                'max_profit_per_contract': max_profit_per_contract,
                'max_loss_per_contract': max_loss_per_contract,
                'breakeven': breakeven,
                'roi': roi,
                'annualized_return': annualized_return,
                'probability_of_profit': probability_of_profit,
                'score': scores['overall_score'],
                'tech_score': scores['technical_score'],
                'greeks_score': scores['greeks_score'],
                'liquidity_score': scores['liquidity_score'],
                'resistance_score': scores['resistance_score'],
                'volume': greeks['volume'],
                'open_interest': greeks['open_interest'],
                'ivr': stock_data.get('ivr'),
                # Technical indicators
                'rsi': stock_data.get('rsi', 50.0),
                'macd': stock_data.get('macd', {}).get('macd', 0.0),
                'macd_signal': stock_data.get('macd', {}).get('signal', 0.0),
                'macd_histogram': stock_data.get('macd', {}).get('histogram', 0.0),
                'bb_upper': stock_data.get('bollinger_bands', {}).get('upper', stock_price * 1.05),
                'bb_middle': stock_data.get('bollinger_bands', {}).get('middle', stock_price),
                'bb_lower': stock_data.get('bollinger_bands', {}).get('lower', stock_price * 0.95),
                'bb_position': stock_data.get('bollinger_bands', {}).get('position', 0.5),
                'support_level': stock_data.get('support_resistance', {}).get('support', stock_price * 0.95),
                'resistance_level': stock_data.get('support_resistance', {}).get('resistance', stock_price * 1.05),
                'score_breakdown': scores
            }
            
        except Exception as e:
            logger.debug(f"Error processing single option: {e}")
            return None
    
    def _calculate_all_scores(self, stock_data: Dict, greeks: Dict, days_to_exp: int, 
                            contract, stock_price: float, annualized_return: float, 
                            probability_of_profit: float) -> Dict:
        """Calculate all scores in a single optimized pass"""
        tech_score = self._calculate_technical_score(stock_data)
        greeks_score = self._calculate_greeks_score(greeks, days_to_exp)
        liquidity_score = self._calculate_liquidity_score(
            greeks['volume'], greeks['open_interest'], contract.strike_price * 0.01  # Estimate premium
        )
        resistance_score = self._calculate_resistance_score(
            contract.strike_price, stock_price, stock_data
        )
        ivr_score = self._calculate_ivr_score(stock_data.get('ivr'))
        annualized_score = self._calculate_annualized_return_score(annualized_return)
        prob_score = probability_of_profit * 100
        
        overall = (
            annualized_score * self.weights['annualized_return'] +
            tech_score * self.weights['technical'] +
            resistance_score * self.weights['resistance'] +
            prob_score * self.weights['probability'] +
            greeks_score * self.weights['greeks'] +
            ivr_score * self.weights['ivr'] +
            liquidity_score * self.weights['liquidity']
        )
        
        return {
            'technical_score': tech_score,
            'greeks_score': greeks_score,
            'liquidity_score': liquidity_score,
            'resistance_score': resistance_score,
            'ivr_score': ivr_score,
            'annualized_return_score': annualized_score,
            'probability_score': prob_score,
            'overall_score': round(overall, 2),
            'weights': self.weights
        }
    
    def _calculate_moneyness(self, stock_price: float, strike_price: float) -> str:
        """Calculate moneyness for puts"""
        intrinsic = max(0, strike_price - stock_price)
        if intrinsic > 0:
            return 'ITM'
        elif abs(stock_price - strike_price) < stock_price * 0.02:
            return 'ATM'
        else:
            return 'OTM'
    
    def _apply_final_filtering(self, recommendations: List[Dict]) -> List[Dict]:
        """Apply final filtering and sorting with ticker diversity"""
        if not recommendations:
            return []
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply ticker diversity limit (max 20 options)
        return self._apply_ticker_priority_limit(recommendations, max_options=20)
    
    def _get_popular_volatile_tickers(self) -> List[str]:
        """Get popular and high volatility tickers from iv_history table"""
        if not self.supabase:
            logger.warning("No Supabase client - cannot get popular/volatile tickers")
            return []
        
        try:
            # Get ALL unique tickers from the database with proper pagination
            logger.info("🔍 Getting ALL unique tickers from database with pagination...")
            all_tickers = set()
            page_size = 1000
            offset = 0
            
            while True:
                logger.info(f"🔍 Fetching page {offset//page_size + 1} (offset: {offset})...")
                result = self.supabase.table('iv_history').select('ticker').range(offset, offset + page_size - 1).execute()
                
                if not result.data:
                    logger.info("🔍 No more data, pagination complete")
                    break
                
                page_tickers = [row['ticker'] for row in result.data if row.get('ticker')]
                all_tickers.update(page_tickers)
                logger.info(f"📊 Page {offset//page_size + 1}: {len(page_tickers)} tickers, total unique: {len(all_tickers)}")
                
                # If we got fewer than page_size, we've reached the end
                if len(result.data) < page_size:
                    logger.info("🔍 Reached end of data (partial page)")
                    break
                
                offset += page_size
                
                # Safety limit to prevent infinite loops
                if offset > 50000:  # Max 50k records
                    logger.warning("🔍 Hit safety limit of 50k records")
                    break
            
            tickers = list(all_tickers)
            logger.info(f"📊 Total unique tickers found: {len(tickers)}")
            
            # Debug: Show some sample tickers
            if tickers:
                sample_tickers = tickers[:10]
                logger.info(f"🔍 Sample tickers: {sample_tickers}")
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error getting popular/volatile tickers: {e}")
            return []
    
    def _apply_ticker_priority_limit(self, recommendations: List[Dict], max_options: int = 20) -> List[Dict]:
        """Apply limit with priority: ensure maximum ticker diversity (1 option per ticker when possible)"""
        if len(recommendations) <= max_options:
            return recommendations
        
        # Group by ticker
        by_ticker = {}
        for rec in recommendations:
            ticker = rec['ticker']
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            by_ticker[ticker].append(rec)
        
        # Sort ticker groups by their best option's score
        ticker_groups = [(ticker, sorted(recs, key=lambda x: x['score'], reverse=True)) 
                        for ticker, recs in by_ticker.items()]
        ticker_groups.sort(key=lambda x: x[1][0]['score'], reverse=True)
        
        final_recommendations = []
        used_tickers = set()
        
        # Strategy: Always prioritize ticker diversity
        # If we have 20+ tickers, take exactly 1 from each of the top 20 tickers
        # If we have fewer than 20 tickers, take the best options overall
        if len(ticker_groups) >= max_options:
            # Take exactly 1 option from the top 20 tickers (by score)
            logger.info(f"🎯 Found {len(ticker_groups)} tickers - taking 1 option from top {max_options} tickers for maximum diversity")
            for i in range(min(max_options, len(ticker_groups))):
                ticker, ticker_recs = ticker_groups[i]
                final_recommendations.append(ticker_recs[0])  # Best option from this ticker
                used_tickers.add(ticker)
        else:
            # We have fewer than 20 tickers, so take best options overall
            logger.info(f"🎯 Found only {len(ticker_groups)} tickers - taking best options overall")
            
            # First pass: take best option from each ticker
            for ticker, ticker_recs in ticker_groups:
                final_recommendations.append(ticker_recs[0])
                used_tickers.add(ticker)
            
            # Second pass: if we still have room, add more options from the same tickers
            if len(final_recommendations) < max_options:
                remaining_recs = []
                for ticker, ticker_recs in ticker_groups:
                    # Add all options except the first (already taken)
                    remaining_recs.extend(ticker_recs[1:])
                
                # Sort remaining by score
                remaining_recs.sort(key=lambda x: x['score'], reverse=True)
                
                # Add remaining options until we reach the limit
                slots_left = max_options - len(final_recommendations)
                final_recommendations.extend(remaining_recs[:slots_left])
        
        # Sort final results by score
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"🎯 Final: {len(final_recommendations)} options from {len(used_tickers)} unique tickers")
        logger.info(f"🎯 Tickers used: {sorted(used_tickers)}")
        
        return final_recommendations
    
    def _get_ivr(self, ticker: str) -> Optional[float]:
        """Get IVR (Implied Volatility Rank) from Supabase - exact copy from reference"""
        import time
        
        if not self.supabase:
            return None
        
        # Check cache first
        cache_key = f"{ticker}_ivr"
        if cache_key in self.ivr_cache:
            cached_ivr, cache_time = self.ivr_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cached_ivr
        
        try:
            # Get IV history from Supabase (last 252 trading days = 1 year)
            result = self.supabase.table('iv_history').select('*').eq('ticker', ticker).order('date', desc=True).limit(252).execute()
            
            if not result.data or len(result.data) < 20:
                logger.warning(f"Insufficient IV history for {ticker}: {len(result.data) if result.data else 0} records")
                return None
            
            # Calculate IVR
            current_iv = float(result.data[0]['iv'])
            iv_values = [float(entry['iv']) for entry in result.data if entry.get('iv') is not None]
            
            if len(iv_values) < 20:
                return None
            
            iv_min = min(iv_values)
            iv_max = max(iv_values)
            
            if iv_max > iv_min:
                ivr = ((current_iv - iv_min) / (iv_max - iv_min)) * 100
                ivr = round(ivr, 1)
                
                # Cache the result
                self.ivr_cache[cache_key] = (ivr, time.time())
                
                return ivr
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting IVR for {ticker}: {e}")
            return None
    
    def _get_stock_data(self, ticker: str) -> Optional[Dict]:
        """Get stock data with enhanced technical indicators (with daily caching)"""
        import time
        
        # Clear cache if it's a new day
        today = datetime.now().date()
        if today != self.cache_date:
            logger.info(f"🗓️  New day detected - clearing technical indicators cache")
            self.technical_cache.clear()
            self.cache_date = today
        
        # Check cache first
        cache_key = f"{ticker}_technical"
        if cache_key in self.technical_cache:
            cached_data, cache_time = self.technical_cache[cache_key]
            logger.info(f"📦 Using cached technical data for {ticker} (age: {(time.time() - cache_time)/60:.1f} min)")
            return cached_data
        
        try:
            # Use Yahoo Finance for both current price and historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if hist.empty:
                logger.warning(f"⚠️  No historical data for {ticker} - hist is empty")
                return None
            
            # Get current price from the most recent close price
            current_price = float(hist['Close'].iloc[-1])
            logger.info(f"✅ Got current price from Yahoo Finance: {ticker} = ${current_price:.2f}")
            
            if len(hist) < 20:  # Reduced from 50 to 20 days minimum
                logger.warning(f"⚠️  Insufficient historical data for {ticker}: {len(hist)} days (need 20)")
                return None
            
            prices = hist['Close']
            volumes = hist['Volume']
            
            # Calculate enhanced technical indicators
            rsi = self._calculate_rsi(prices)
            macd = self._calculate_macd(prices)
            bollinger = self._calculate_bollinger_bands(prices)
            support_resistance = self._calculate_support_resistance(prices)
            
            # Enhanced indicators
            moving_averages = self._calculate_moving_averages(prices)
            volume_analysis = self._calculate_volume_analysis(volumes)
            trend_strength = self._calculate_trend_strength(prices)
            
            stock_data = {
                'current_price': current_price,
                'rsi': rsi,
                'macd': macd,
                'bollinger_bands': bollinger,
                'support_resistance': support_resistance,
                'price_change_pct': self._calculate_price_change_pct(prices),
                'moving_averages': moving_averages,
                'volume_analysis': volume_analysis,
                'trend_strength': trend_strength
            }
            
            # Cache the result
            self.technical_cache[cache_key] = (stock_data, time.time())
            
            return stock_data
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {e}")
            return None
    
    def _calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate multiple moving averages for trend analysis"""
        try:
            ma_20 = prices.rolling(window=20).mean().iloc[-1] if len(prices) >= 20 else float(prices.iloc[-1])
            ma_50 = prices.rolling(window=50).mean().iloc[-1] if len(prices) >= 50 else float(prices.iloc[-1])
            ma_200 = prices.rolling(window=200).mean().iloc[-1] if len(prices) >= 200 else float(prices.iloc[-1])
            
            return {
                'ma_20': float(ma_20) if not pd.isna(ma_20) else float(prices.iloc[-1]),
                'ma_50': float(ma_50) if not pd.isna(ma_50) else float(prices.iloc[-1]),
                'ma_200': float(ma_200) if not pd.isna(ma_200) else float(prices.iloc[-1])
            }
        except:
            current_price = float(prices.iloc[-1]) if not prices.empty else 100.0
            return {
                'ma_20': current_price,
                'ma_50': current_price,
                'ma_200': current_price
            }
    
    def _calculate_volume_analysis(self, volumes: pd.Series) -> Dict[str, float]:
        """Calculate volume-based indicators"""
        try:
            if len(volumes) < 20:
                return {'volume_ratio': 1.0, 'volume_trend': 0.0}
            
            # Current volume vs 20-day average
            avg_volume_20 = volumes.rolling(window=20).mean().iloc[-1]
            current_volume = volumes.iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
            
            # Volume trend (comparing recent vs older periods)
            recent_avg = volumes.tail(5).mean()
            older_avg = volumes.iloc[-20:-5].mean()
            volume_trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
            
            return {
                'volume_ratio': float(volume_ratio) if not pd.isna(volume_ratio) else 1.0,
                'volume_trend': float(volume_trend) if not pd.isna(volume_trend) else 0.0
            }
        except:
            return {'volume_ratio': 1.0, 'volume_trend': 0.0}
    
    def _calculate_trend_strength(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate trend strength indicators"""
        try:
            if len(prices) < 50:
                return {'trend_score': 0.5, 'momentum': 0.0}
            
            # Price momentum over different periods
            momentum_5 = (prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6] if prices.iloc[-6] > 0 else 0
            momentum_10 = (prices.iloc[-1] - prices.iloc[-11]) / prices.iloc[-11] if prices.iloc[-11] > 0 else 0
            momentum_20 = (prices.iloc[-1] - prices.iloc[-21]) / prices.iloc[-21] if prices.iloc[-21] > 0 else 0
            
            # Trend score based on moving average alignment
            ma_20 = prices.rolling(window=20).mean().iloc[-1]
            ma_50 = prices.rolling(window=50).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            trend_score = 0.5  # Neutral
            if current_price > ma_20 > ma_50:
                trend_score = 0.8  # Strong uptrend
            elif current_price > ma_20:
                trend_score = 0.65  # Moderate uptrend
            elif current_price > ma_50:
                trend_score = 0.6  # Weak uptrend
            elif current_price < ma_50 < ma_20:
                trend_score = 0.2  # Strong downtrend
            elif current_price < ma_50:
                trend_score = 0.35  # Moderate downtrend
            
            # Overall momentum
            momentum = (momentum_5 + momentum_10 + momentum_20) / 3
            
            return {
                'trend_score': float(trend_score) if not pd.isna(trend_score) else 0.5,
                'momentum': float(momentum) if not pd.isna(momentum) else 0.0,
                'momentum_5': float(momentum_5) if not pd.isna(momentum_5) else 0.0,
                'momentum_10': float(momentum_10) if not pd.isna(momentum_10) else 0.0,
                'momentum_20': float(momentum_20) if not pd.isna(momentum_20) else 0.0
            }
        except:
            return {'trend_score': 0.5, 'momentum': 0.0}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI - exact copy from reference"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD - exact copy from reference"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': float(macd_line.iloc[-1]) if not macd_line.empty and not pd.isna(macd_line.iloc[-1]) else 0.0,
                'signal': float(signal_line.iloc[-1]) if not signal_line.empty and not pd.isna(signal_line.iloc[-1]) else 0.0,
                'histogram': float(histogram.iloc[-1]) if not histogram.empty and not pd.isna(histogram.iloc[-1]) else 0.0
            }
        except:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands - exact copy from reference"""
        try:
            sma = prices.rolling(window=period).mean()
            std_dev = prices.rolling(window=period).std()
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            
            current_price = float(prices.iloc[-1]) if not prices.empty else 0.0
            upper = float(upper_band.iloc[-1]) if not upper_band.empty and not pd.isna(upper_band.iloc[-1]) else current_price
            middle = float(sma.iloc[-1]) if not sma.empty and not pd.isna(sma.iloc[-1]) else current_price
            lower = float(lower_band.iloc[-1]) if not lower_band.empty and not pd.isna(lower_band.iloc[-1]) else current_price
            
            # Calculate position within bands
            if upper != lower:
                position = (current_price - lower) / (upper - lower)
            else:
                position = 0.5
            
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'position': position
            }
        except:
            current_price = float(prices.iloc[-1]) if not prices.empty else 100.0
            return {
                'upper': current_price * 1.05,
                'middle': current_price,
                'lower': current_price * 0.95,
                'position': 0.5
            }
    
    def _calculate_support_resistance(self, prices: pd.Series, window: int = 20) -> Dict[str, float]:
        """Calculate support and resistance - exact copy from reference"""
        try:
            if len(prices) < window:
                current_price = prices.iloc[-1]
                return {'support': current_price * 0.95, 'resistance': current_price * 1.05}
            
            recent = prices.tail(window * 2)
            current_price = prices.iloc[-1]
            
            # Simple support/resistance using highs and lows
            highs = recent.rolling(window=5).max().dropna()
            lows = recent.rolling(window=5).min().dropna()
            
            # Find resistance (highs above current price)
            resistance_levels = [h for h in highs.values if h > current_price]
            resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
            
            # Find support (lows below current price)
            support_levels = [l for l in lows.values if l < current_price]
            support = max(support_levels) if support_levels else current_price * 0.95
            
            return {
                'support': float(support) if not pd.isna(support) else current_price * 0.95,
                'resistance': float(resistance) if not pd.isna(resistance) else current_price * 1.05
            }
        except:
            current_price = float(prices.iloc[-1]) if not prices.empty else 100.0
            return {'support': current_price * 0.95, 'resistance': current_price * 1.05}
    
    def _calculate_price_change_pct(self, prices: pd.Series, period: int = 5) -> float:
        """Calculate price change percentage"""
        try:
            if len(prices) < period + 1:
                return 0.0
            current = prices.iloc[-1]
            past = prices.iloc[-(period + 1)]
            if past == 0:
                return 0.0
            return float((current - past) / past * 100)
        except:
            return 0.0
    
    def _extract_option_price(self, snapshot) -> float:
        """Extract option price from snapshot"""
        try:
            if hasattr(snapshot, 'day') and snapshot.day:
                price = getattr(snapshot.day, 'close', None)
                if price and price > 0:
                    return float(price)
            
            if hasattr(snapshot, 'last_trade') and snapshot.last_trade:
                price = getattr(snapshot.last_trade, 'price', None)
                if price and price > 0:
                    return float(price)
            
            if hasattr(snapshot, 'last_quote') and snapshot.last_quote:
                bid = getattr(snapshot.last_quote, 'bid', 0) or 0
                ask = getattr(snapshot.last_quote, 'ask', 0) or 0
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2
        except:
            pass
        return 0.0
    
    def _extract_greeks(self, snapshot) -> Dict[str, float]:
        """Extract Greeks from snapshot"""
        greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'implied_volatility': 0.3,
            'volume': 0,
            'open_interest': 0
        }
        
        try:
            if hasattr(snapshot, 'greeks') and snapshot.greeks:
                greeks['delta'] = getattr(snapshot.greeks, 'delta', 0.0) or 0.0
                greeks['gamma'] = getattr(snapshot.greeks, 'gamma', 0.0) or 0.0
                greeks['theta'] = getattr(snapshot.greeks, 'theta', 0.0) or 0.0
                greeks['vega'] = getattr(snapshot.greeks, 'vega', 0.0) or 0.0
            
            if hasattr(snapshot, 'implied_volatility'):
                greeks['implied_volatility'] = getattr(snapshot, 'implied_volatility', 0.3) or 0.3
            
            if hasattr(snapshot, 'day') and snapshot.day:
                greeks['volume'] = getattr(snapshot.day, 'volume', 0) or 0
            
            if hasattr(snapshot, 'open_interest'):
                greeks['open_interest'] = getattr(snapshot, 'open_interest', 0) or 0
        except:
            pass
        
        return greeks
    
    def _calculate_probability_of_profit(self, delta: float, days_to_exp: int,
                                         iv: float, strike: float, stock_price: float,
                                         stock_data: Dict) -> float:
        """Calculate probability of profit using enhanced Black-Scholes approximation"""
        try:
            # Enhanced probability calculation using multiple factors
            
            # Base probability from delta (for puts, delta is negative)
            # Delta represents the probability that the option will be ITM at expiration
            base_prob = max(0.05, min(0.95, 1 - abs(delta)))
            
            # Time decay adjustment (more sophisticated)
            time_factor = self._calculate_time_decay_factor(days_to_exp)
            
            # Volatility adjustment (market-aware)
            vol_factor = self._calculate_volatility_factor(iv, stock_data)
            
            # Strike distance adjustment (enhanced)
            distance_factor = self._calculate_strike_distance_factor(strike, stock_price)
            
            # Technical analysis adjustment (market-aware)
            tech_factor = self._calculate_technical_probability_factor(stock_data)
            
            # Market regime adjustment
            market_factor = self._calculate_market_regime_factor(stock_data)
            
            # Combine all factors
            prob = base_prob * time_factor * vol_factor * distance_factor * tech_factor * market_factor
            
            return max(0.05, min(0.95, prob))
        except:
            return 0.5
    
    def _calculate_time_decay_factor(self, days_to_exp: int) -> float:
        """Calculate time decay adjustment factor"""
        if days_to_exp < 7:
            return 1.05  # Short-term options have higher probability
        elif days_to_exp < 15:
            return 1.02
        elif days_to_exp < 30:
            return 1.00  # Baseline
        elif days_to_exp < 45:
            return 0.98
        elif days_to_exp < 60:
            return 0.95
        else:
            return 0.92  # Long-term options have lower probability
    
    def _calculate_volatility_factor(self, iv: float, stock_data: Dict) -> float:
        """Calculate volatility adjustment factor"""
        # Get IVR for more sophisticated volatility analysis
        ivr = stock_data.get('ivr')
        
        if ivr is not None:
            # High IVR means volatility is elevated relative to historical
            if ivr > 70:
                return 0.90  # High IV environment - reduce probability
            elif ivr > 50:
                return 0.95  # Moderate IV environment
            elif ivr > 30:
                return 1.00  # Normal IV environment
            else:
                return 1.05  # Low IV environment - increase probability
        
        # Fallback to raw IV
        if iv > 0.8:
            return 0.85
        elif iv > 0.6:
            return 0.90
        elif iv > 0.4:
            return 0.95
        elif iv > 0.2:
            return 1.00
        else:
            return 1.05
    
    def _calculate_strike_distance_factor(self, strike: float, stock_price: float) -> float:
        """Calculate strike distance adjustment factor"""
        distance_pct = (stock_price - strike) / stock_price
        
        if distance_pct < 0.02:  # Very close to current price (ATM)
            return 0.90  # Lower probability due to uncertainty
        elif distance_pct < 0.05:  # Close to current price
            return 0.95
        elif distance_pct < 0.10:  # Moderate distance
            return 1.00  # Baseline
        elif distance_pct < 0.15:  # Further out
            return 1.02
        else:  # Far out of the money
            return 1.05  # Higher probability but lower premium
    
    def _calculate_technical_probability_factor(self, stock_data: Dict) -> float:
        """Calculate technical analysis adjustment factor"""
        try:
            factor = 1.0
            
            rsi = stock_data.get('rsi', 50)
            macd = stock_data.get('macd', {})
            bb = stock_data.get('bollinger_bands', {})
            
            # RSI analysis for put selling
            if rsi < 25:  # Extremely oversold
                factor *= 1.08  # Strong support for put selling
            elif rsi < 30:  # Oversold
                factor *= 1.05
            elif rsi < 40:  # Approaching oversold
                factor *= 1.02
            elif rsi > 70:  # Overbought
                factor *= 0.95
            elif rsi > 60:  # Approaching overbought
                factor *= 0.98
            
            # MACD momentum analysis
            histogram = macd.get('histogram', 0)
            if histogram > 0:  # Bullish momentum
                factor *= 1.03
            elif histogram < -0.1:  # Strong bearish momentum
                factor *= 0.97
            
            # Bollinger Bands position
            bb_position = bb.get('position', 0.5)
            if bb_position < 0.2:  # Near lower band
                factor *= 1.04
            elif bb_position < 0.3:  # Below middle
                factor *= 1.02
            elif bb_position > 0.8:  # Near upper band
                factor *= 0.96
            elif bb_position > 0.7:  # Above middle
                factor *= 0.98
            
            return factor
        except:
            return 1.0
    
    def _calculate_market_regime_factor(self, stock_data: Dict) -> float:
        """Calculate market regime adjustment factor"""
        try:
            factor = 1.0
            
            # Analyze price momentum
            price_change = stock_data.get('price_change_pct', 0)
            
            # Strong positive momentum (good for put selling)
            if price_change > 3:
                factor *= 1.05
            elif price_change > 1:
                factor *= 1.02
            elif price_change < -3:  # Strong negative momentum
                factor *= 0.95
            elif price_change < -1:
                factor *= 0.98
            
            # Support/resistance analysis
            support_resistance = stock_data.get('support_resistance', {})
            support = support_resistance.get('support', 0)
            resistance = support_resistance.get('resistance', 0)
            current_price = stock_data.get('current_price', 0)
            
            if support > 0 and resistance > 0:
                # Calculate position within support/resistance range
                range_size = resistance - support
                position = (current_price - support) / range_size if range_size > 0 else 0.5
                
                # Better probability when closer to support
                if position < 0.3:  # Near support
                    factor *= 1.03
                elif position > 0.7:  # Near resistance
                    factor *= 0.98
            
            return factor
        except:
            return 1.0
    
    def _calculate_technical_score(self, stock_data: Dict) -> float:
        """Calculate enhanced technical score optimized for put selling"""
        try:
            score = 50.0
            
            rsi = stock_data.get('rsi', 50)
            macd = stock_data.get('macd', {})
            bb = stock_data.get('bollinger_bands', {})
            price_change = stock_data.get('price_change_pct', 0)
            current_price = stock_data.get('current_price', 0)
            
            # Enhanced indicators
            moving_averages = stock_data.get('moving_averages', {})
            volume_analysis = stock_data.get('volume_analysis', {})
            trend_strength = stock_data.get('trend_strength', {})
            
            # RSI analysis for put selling (heavily weighted)
            if rsi < 25:  # Extremely oversold - excellent for selling puts
                score += 30
            elif rsi < RSI_OVERSOLD:  # Oversold - very good for selling puts
                score += 25
            elif rsi < 40:  # Approaching oversold - good for selling puts
                score += 20
            elif rsi < 50:  # Neutral to oversold - acceptable
                score += 10
            elif rsi > RSI_OVERBOUGHT:  # Overbought - bad for selling puts
                score -= 25
            elif rsi > 60:  # Approaching overbought - not ideal
                score -= 15
            
            # Enhanced moving average analysis
            ma_20 = moving_averages.get('ma_20', current_price)
            ma_50 = moving_averages.get('ma_50', current_price)
            ma_200 = moving_averages.get('ma_200', current_price)
            
            if current_price > ma_20 > ma_50 > ma_200:  # Perfect bullish alignment
                score += 25
            elif current_price > ma_20 > ma_50:  # Strong bullish trend
                score += 20
            elif current_price > ma_20:  # Above 20-day MA
                score += 15
            elif current_price > ma_50:  # Above 50-day MA
                score += 10
            elif current_price < ma_50 < ma_20:  # Below both MAs - concerning
                score -= 15
            elif current_price < ma_50:  # Below 50-day MA
                score -= 10
            
            # Trend strength analysis
            trend_score = trend_strength.get('trend_score', 0.5)
            momentum = trend_strength.get('momentum', 0.0)
            
            if trend_score > 0.7:  # Strong uptrend
                score += 20
            elif trend_score > 0.6:  # Moderate uptrend
                score += 15
            elif trend_score > 0.5:  # Weak uptrend
                score += 10
            elif trend_score < 0.3:  # Strong downtrend
                score -= 20
            elif trend_score < 0.4:  # Moderate downtrend
                score -= 15
            
            # Momentum analysis
            if momentum > 0.05:  # Strong positive momentum
                score += 15
            elif momentum > 0.02:  # Moderate positive momentum
                score += 10
            elif momentum > 0:  # Weak positive momentum
                score += 5
            elif momentum < -0.05:  # Strong negative momentum
                score -= 20
            elif momentum < -0.02:  # Moderate negative momentum
                score -= 15
            
            # MACD analysis for momentum
            histogram = macd.get('histogram', 0)
            is_bullish = histogram > 0
            if histogram > 0.1:  # Strong bullish momentum
                score += 15
            elif is_bullish:
                score += 10  # Bullish - good for selling puts
            elif histogram < -0.1:  # Strong bearish momentum
                score -= 15
            else:
                score -= 5  # Bearish - bad for selling puts
            
            # Bollinger Bands analysis
            position = bb.get('position', 0.5)
            if position < 0.2:  # Near lower band - excellent for put selling
                score += 20
            elif position < 0.3:  # Below middle - good for put selling
                score += 10
            elif position > 0.8:  # Near upper band - bad for put selling
                score -= 20
            elif position > 0.7:  # Above middle - not ideal
                score -= 10
            elif 0.3 <= position <= 0.7:
                score += 5
            
            # Enhanced volume analysis
            volume_ratio = volume_analysis.get('volume_ratio', 1.0)
            volume_trend = volume_analysis.get('volume_trend', 0.0)
            
            if volume_ratio > 2.0:  # Very high volume - strong confirmation
                score += 15
            elif volume_ratio > 1.5:  # High volume - confirms move
                score += 10
            elif volume_ratio > 1.2:  # Above average volume
                score += 5
            elif volume_ratio < 0.5:  # Low volume - weak confirmation
                score -= 10
            elif volume_ratio < 0.8:  # Below average volume
                score -= 5
            
            # Volume trend analysis
            if volume_trend > 0.2:  # Increasing volume trend
                score += 10
            elif volume_trend > 0:  # Positive volume trend
                score += 5
            elif volume_trend < -0.2:  # Decreasing volume trend
                score -= 10
            elif volume_trend < 0:  # Negative volume trend
                score -= 5
            
            # Price momentum analysis (enhanced)
            if price_change > 3:  # Very strong positive momentum
                score += 20
            elif price_change > 2:  # Strong positive momentum - excellent for put selling
                score += 15
            elif price_change > 1:  # Moderate positive momentum
                score += 10
            elif price_change > 0:  # Positive momentum - good for put selling
                score += 5
            elif price_change < -5:  # Very strong negative momentum
                score -= 25
            elif price_change < -3:  # Strong negative momentum - bad for put selling
                score -= 20
            elif price_change < -2:  # Moderate negative momentum
                score -= 15
            elif price_change < -1:  # Weak negative momentum
                score -= 10
            
            return max(0, min(100, score))
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 50.0
    
    def _calculate_greeks_score(self, greeks: Dict, days_to_exp: int) -> float:
        """Calculate Greeks score - adapted for puts"""
        try:
            score = 50.0
            
            # For puts, delta is negative, so we use absolute value
            delta = abs(greeks['delta'])
            
            # Delta scoring for selling puts
            if 0.15 <= delta <= 0.35:
                score += 20
            elif 0.1 <= delta <= 0.5:
                score += 10
            elif 0.5 < delta <= 0.7:
                score -= 15
            elif delta > 0.7:
                score -= 30
            else:
                score -= 10
            
            # Theta scoring (same for puts and calls)
            theta = abs(greeks['theta'])
            if theta > 0.1:
                score += 15
            elif theta > 0.05:
                score += 5
            else:
                score -= 10
            
            # Vega scoring (same for puts and calls)
            vega = abs(greeks['vega'])
            if vega < 0.1:
                score += 10
            elif vega > 0.3:
                score -= 15
            
            # Gamma scoring (same for puts and calls)
            gamma = abs(greeks['gamma'])
            if gamma < 0.01:
                score += 10
            elif gamma > 0.05:
                score -= 15
            
            return max(0, min(100, score))
        except:
            return 50.0
    
    def _calculate_liquidity_score(self, volume: int, open_interest: int, price: float) -> float:
        """Calculate liquidity score - same as calls"""
        try:
            score = 0.0
            
            # Open interest
            if open_interest >= 1000:
                score += 40
            elif open_interest >= 500:
                score += 30
            elif open_interest >= 100:
                score += 20
            else:
                score += 5
            
            # Volume
            if volume >= 100:
                score += 30
            elif volume >= 50:
                score += 20
            elif volume >= 10:
                score += 10
            else:
                score += 5
            
            # Price (bid-ask spread proxy)
            if price > 1.0:
                score += 20
            elif price > 0.5:
                score += 15
            else:
                score += 10
            
            return min(100, score)
        except:
            return 50.0
    
    def _calculate_resistance_score(self, strike: float, stock_price: float, stock_data: Dict) -> float:
        """Calculate resistance score optimized for put selling - prioritize strong support below strike"""
        try:
            score = 50.0
            
            support = stock_data.get('support_resistance', {}).get('support', stock_price * 0.95)
            resistance = stock_data.get('support_resistance', {}).get('resistance', stock_price * 1.05)
            
            # Calculate support strength based on distance from current price
            support_distance_pct = ((stock_price - support) / stock_price) * 100
            strike_distance_pct = ((stock_price - strike) / stock_price) * 100
            
            # For put selling: prefer strikes with strong support below
            if support_distance_pct > 15:  # Strong support 15%+ below current price
                if strike_distance_pct <= 5:  # Strike close to current price
                    score += 30  # Excellent - strong support far below strike
                elif strike_distance_pct <= 10:  # Strike 5-10% below current price
                    score += 25  # Very good
                else:
                    score += 20  # Good
            elif support_distance_pct > 10:  # Moderate support 10-15% below
                if strike_distance_pct <= 5:
                    score += 25  # Very good
                elif strike_distance_pct <= 10:
                    score += 20  # Good
                else:
                    score += 15  # Fair
            elif support_distance_pct > 5:  # Weak support 5-10% below
                if strike_distance_pct <= 5:
                    score += 20  # Good
                elif strike_distance_pct <= 10:
                    score += 15  # Fair
                else:
                    score += 10  # Poor
            else:  # No clear support or very close
                if strike_distance_pct <= 5:
                    score += 10  # Fair
                else:
                    score += 5   # Poor
            
            # Additional scoring based on resistance levels
            resistance_distance_pct = ((resistance - stock_price) / stock_price) * 100
            if resistance_distance_pct > 20:  # Strong resistance far above
                score += 10  # Good for put selling
            elif resistance_distance_pct > 10:  # Moderate resistance above
                score += 5   # Neutral
            
            # Penalize if strike is too close to resistance
            if strike > resistance * 0.95:  # Strike near resistance
                score -= 15  # Bad for put selling
            
            return max(0, min(100, score))
        except Exception as e:
            logger.error(f"Error calculating resistance score: {e}")
            return 50.0
    
    def _calculate_ivr_score(self, ivr: Optional[float]) -> float:
        """Calculate IVR score component - same as calls"""
        if ivr is None:
            return 50.0
        
        if ivr >= 70:
            return 90.0
        elif ivr >= 50:
            return 70.0
        elif ivr >= 30:
            return 50.0
        else:
            return 30.0
    
    def _calculate_cash_required_score(self, strike_price: float, premium: float) -> float:
        """Calculate cash required score - penalizes higher cash requirements"""
        cash_required = strike_price * 100  # Cash required for CSP
        
        # Score based on cash required tiers
        # Lower cash requirements get higher scores
        if cash_required <= 10000:  # <= $10k
            return 90.0
        elif cash_required <= 25000:  # <= $25k
            return 80.0
        elif cash_required <= 50000:  # <= $50k
            return 70.0
        elif cash_required <= 100000:  # <= $100k
            return 60.0
        elif cash_required <= 250000:  # <= $250k
            return 50.0
        elif cash_required <= 500000:  # <= $500k
            return 40.0
        else:  # > $500k
            return 30.0

    def _calculate_annualized_return_score(self, annualized_return: float) -> float:
        """Calculate annualized return score with exponential scaling for high returns"""
        try:
            # Exponential scaling for high returns
            if annualized_return >= 50:  # > 50% annualized - Excellent
                return 90.0 + min(10.0, (annualized_return - 50) * 0.2)
            elif annualized_return >= 30:  # 30-50% annualized - Good
                return 70.0 + (annualized_return - 30) * 1.0
            elif annualized_return >= 15:  # 15-30% annualized - Fair
                return 50.0 + (annualized_return - 15) * 1.33
            else:  # < 15% annualized - Poor
                return max(0.0, annualized_return * 3.33)
        except Exception as e:
            logger.error(f"Error calculating annualized return score: {e}")
            return 50.0

    def _calculate_overall_score(self, tech_score: float, greeks_score: float,
                                 liquidity_score: float, resistance_score: float,
                                 probability: float, annualized_return: float,
                                 ivr: Optional[float] = None) -> float:
        """Calculate optimized weighted overall score prioritizing high returns with technical/resistance support"""
        try:
            # IVR score calculation (reduced importance)
            ivr_score = 50.0  # Default
            if ivr is not None:
                if ivr >= 70:  # High IVR - excellent for selling
                    ivr_score = 90.0
                elif ivr >= 50:  # Good IVR
                    ivr_score = 70.0
                elif ivr >= 30:  # Medium IVR
                    ivr_score = 50.0
                else:  # Low IVR
                    ivr_score = 30.0
            
            # Probability score (convert to 0-100 scale)
            prob_score = probability * 100
            
            # Annualized return score with exponential scaling
            annualized_score = self._calculate_annualized_return_score(annualized_return)
            
            # Weighted average using optimized weights
            overall = (
                annualized_score * self.weights['annualized_return'] +      # 30% - Primary factor
                tech_score * self.weights['technical'] +                    # 25% - Technical support
                resistance_score * self.weights['resistance'] +             # 20% - Support levels
                prob_score * self.weights['probability'] +                  # 15% - Risk management
                greeks_score * self.weights['greeks'] +                     # 5% - Risk assessment
                ivr_score * self.weights['ivr'] +                           # 3% - IV Rank
                liquidity_score * self.weights['liquidity']                 # 2% - Liquidity
            )
            
            return round(overall, 2)
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 50.0
