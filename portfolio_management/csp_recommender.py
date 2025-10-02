"""
Cash Secured Puts (CSP) Recommender - Uses the exact proven architecture from Covered Calls
Includes rigorous scoring, probability calculations, and technical analysis for PUT options
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
    """Find best Cash Secured Put opportunities from popular/high volatility stocks"""
    
    def __init__(self, polygon_api_key: str = None, supabase_client = None):
        self.polygon_client = RESTClient(polygon_api_key or os.getenv('POLYGON_API_KEY'))
        self.supabase = supabase_client  # For IVR data and stock selection
        self.weights = SCORING_WEIGHTS
        self.technical_cache = {}  # Cache for technical indicators
        self.ivr_cache = {}  # Cache for IVR data
        self.cache_ttl = 86400  # 24 hours (1 day) cache for technical indicators
        self.cache_date = datetime.now().date()  # Track cache date
        
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
    
    def _find_monthly_expirations(self, ticker: str, min_dte: int = 15, max_dte: int = 90) -> List[str]:
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
        """Get CSP recommendations for popular/high volatility stocks or provided tickers"""
        recommendations = []
        total_options_considered = 0
        
        # DEBUG: Comprehensive tracking of ticker elimination
        debug_stats = {
            'initial_tickers': 0,
            'after_stock_data': 0,
            'after_skip_filter': 0,
            'after_monthly_exp': 0,
            'after_put_contracts': 0,
            'after_strike_range': 0,
            'after_delta_filter': 0,
            'final_with_recommendations': 0,
            'skip_reasons': {},
            'ticker_details': {}
        }
        
        # Get tickers to analyze
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
                # Remove artificial limit to allow all available tickers to be considered
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
        
        debug_stats['initial_tickers'] = len(tickers)
        
        if not tickers:
            logger.warning("⚠️  No tickers to analyze")
            return []
        
        # Process each ticker
        tickers_processed = 0
        tickers_skipped = 0
        skip_reasons = {}
        tickers_with_recommendations = 0
        
        logger.info(f"🔍 DEBUG: Starting to process {len(tickers)} tickers: {tickers[:10]}...")
        
        for ticker in tickers:
            ticker_debug = {
                'stock_data_ok': False,
                'skipped_etf': False,
                'monthly_exp_ok': False,
                'put_contracts_ok': False,
                'strike_range_ok': False,
                'delta_filter_ok': False,
                'final_recommendations': 0,
                'skip_reason': None
            }
            
            try:
                tickers_processed += 1
                logger.info(f"🔄 Processing {ticker} ({tickers_processed}/{len(tickers)})...")
                
                # Get stock price and technical data
                stock_data = self._get_stock_data(ticker)
                if not stock_data:
                    tickers_skipped += 1
                    skip_reasons['No stock data'] = skip_reasons.get('No stock data', 0) + 1
                    ticker_debug['skip_reason'] = 'No stock data'
                    logger.warning(f"⚠️  No stock data for {ticker} (skipped {tickers_skipped} so far)")
                    logger.warning(f"🔍 DEBUG: Failed to get stock data for {ticker} - this could be due to:")
                    logger.warning(f"   - Yahoo Finance API issues")
                    logger.warning(f"   - Insufficient historical data (< 20 days)")
                    logger.warning(f"   - Price calculation errors")
                    logger.warning(f"   - Technical indicator calculation errors")
                    debug_stats['ticker_details'][ticker] = ticker_debug
                    continue
                
                ticker_debug['stock_data_ok'] = True
                debug_stats['after_stock_data'] += 1
                
                stock_price = stock_data['current_price']
                logger.info(f"💰 {ticker} price: ${stock_price:.2f}")
                
                # Note: Removed penny stock filter - we'll let liquidity criteria handle this
                
                if ticker in self.skip_tickers:
                    tickers_skipped += 1
                    skip_reasons['ETF/Bond'] = skip_reasons.get('ETF/Bond', 0) + 1
                    ticker_debug['skipped_etf'] = True
                    ticker_debug['skip_reason'] = 'ETF/Bond'
                    logger.warning(f"⏭️  DEBUG: Skipping {ticker} - ETF/Bond (skipped {tickers_skipped}/{tickers_processed})")
                    debug_stats['ticker_details'][ticker] = ticker_debug
                    continue
                
                debug_stats['after_skip_filter'] += 1
                
                # Get IVR from Supabase
                ivr = self._get_ivr(ticker)
                if ivr is not None:
                    stock_data['ivr'] = ivr
                    logger.info(f"📊 IVR for {ticker}: {ivr:.1f}%")
                else:
                    stock_data['ivr'] = None
                    logger.info(f"⚠️  No IVR data for {ticker}")
                
                # Find next 2 monthly expirations
                monthly_expirations = self._find_monthly_expirations(ticker)
                if not monthly_expirations:
                    tickers_skipped += 1
                    skip_reasons['No monthly expirations'] = skip_reasons.get('No monthly expirations', 0) + 1
                    ticker_debug['skip_reason'] = 'No monthly expirations'
                    logger.warning(f"⏭️  DEBUG: Skipping {ticker} - No monthly expirations (skipped {tickers_skipped}/{tickers_processed})")
                    debug_stats['ticker_details'][ticker] = ticker_debug
                    continue
                else:
                    ticker_debug['monthly_exp_ok'] = True
                    debug_stats['after_monthly_exp'] += 1
                    logger.info(f"📅 DEBUG: {ticker} has {len(monthly_expirations)} monthly expirations: {monthly_expirations}")
                
                # Get options contracts for monthly expirations only
                all_put_contracts = []
                for exp_date in monthly_expirations:
                    try:
                        contracts = list(self.polygon_client.list_options_contracts(
                            underlying_ticker=ticker,
                            expiration_date=exp_date,
                            contract_type='put',
                            limit=50
                        ))
                        all_put_contracts.extend(contracts)
                        logger.info(f"✅ Found {len(contracts)} put contracts for {ticker} exp {exp_date}")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not fetch contracts for {ticker} exp {exp_date}: {e}")
                        continue
                
                if not all_put_contracts:
                    tickers_skipped += 1
                    skip_reasons['No put contracts'] = skip_reasons.get('No put contracts', 0) + 1
                    ticker_debug['skip_reason'] = 'No put contracts'
                    logger.warning(f"⏭️  DEBUG: Skipping {ticker} - No put contracts (skipped {tickers_skipped}/{tickers_processed})")
                    debug_stats['ticker_details'][ticker] = ticker_debug
                    continue
                else:
                    ticker_debug['put_contracts_ok'] = True
                    debug_stats['after_put_contracts'] += 1
                    logger.info(f"📞 DEBUG: {ticker} has {len(all_put_contracts)} total put contracts")
                
                logger.info(f"📞 Total {len(all_put_contracts)} monthly put contracts for {ticker}")
                
                # Filter for relevant strike range (for CSPs, we want strikes below current price)
                lower_strike = stock_price * 0.75  # 25% below current price (more inclusive)
                upper_strike = stock_price * 0.99  # 1% below current price (more inclusive)
                
                put_contracts = [
                    c for c in all_put_contracts
                    if lower_strike <= c.strike_price <= upper_strike
                ]
                
                if not put_contracts:
                    tickers_skipped += 1
                    skip_reasons['No strikes in range'] = skip_reasons.get('No strikes in range', 0) + 1
                    ticker_debug['skip_reason'] = 'No strikes in range'
                    logger.warning(f"⏭️  DEBUG: Skipping {ticker} - No strikes in range ${lower_strike:.2f}-${upper_strike:.2f} (skipped {tickers_skipped}/{tickers_processed})")
                    debug_stats['ticker_details'][ticker] = ticker_debug
                    continue
                
                ticker_debug['strike_range_ok'] = True
                debug_stats['after_strike_range'] += 1
                
                logger.info(f"🎯 Found {len(put_contracts)} put options in strike range for {ticker}")
                logger.info(f"📊 Strike range: ${lower_strike:.2f} - ${upper_strike:.2f} (stock price: ${stock_price:.2f})")
                total_options_considered += len(put_contracts)
                
                # Process contracts for this ticker
                ticker_recommendations = []
                contracts_passed_delta = 0
                for contract in put_contracts:
                    try:
                        # Get snapshot
                        snapshot = self.polygon_client.get_snapshot_option(
                            contract.underlying_ticker,
                            contract.ticker
                        )
                        
                        # Extract price and Greeks
                        price = self._extract_option_price(snapshot)
                        if not price or price == 0:
                            continue
                        
                        greeks = self._extract_greeks(snapshot)
                        
                        # Filter out options with extremely low deltas (not suitable for CSPs)
                        # For puts, delta is negative, so we check absolute value
                        abs_delta = abs(greeks['delta'])
                        if abs_delta < 0.14:  # Skip options with delta < 0.14 (too far OTM)
                            logger.debug(f"Skipping {contract.ticker} - delta too low: {greeks['delta']:.3f}")
                            continue
                        if abs_delta > 0.45:  # Skip options with delta > 0.45 (too close to ATM)
                            logger.debug(f"Skipping {contract.ticker} - delta too high: {greeks['delta']:.3f}")
                            continue
                        
                        contracts_passed_delta += 1
                        
                        # Calculate days to expiration (using US/Eastern timezone for market)
                        et_tz = pytz.timezone('US/Eastern')
                        now_et = datetime.now(et_tz)
                        
                        # Parse expiration date and set to market close time (4:00 PM ET)
                        exp_date = datetime.fromisoformat(contract.expiration_date)
                        exp_date_et = et_tz.localize(exp_date.replace(hour=16, minute=0, second=0))
                        
                        # Calculate days remaining (including partial days)
                        time_remaining = exp_date_et - now_et
                        days_to_exp = max(1, time_remaining.days + (1 if time_remaining.seconds > 0 else 0))
                        
                        # Calculate moneyness
                        intrinsic = max(0, contract.strike_price - stock_price)
                        if intrinsic > 0:
                            moneyness = 'ITM'
                        elif abs(stock_price - contract.strike_price) < stock_price * 0.02:
                            moneyness = 'ATM'
                        else:
                            moneyness = 'OTM'
                        
                        # Calculate metrics (PER CONTRACT - 100 shares)
                        # For SELLING a cash secured put:
                        premium_per_contract = price * 100
                        
                        # Max profit: Just the premium received (you keep the premium no matter what)
                        max_profit_per_contract = price * 100
                        
                        # Max loss: Stock drops to $0, but you keep the premium
                        # Loss = strike_price - 0 = strike_price
                        # But you collected premium, so net loss = strike_price - premium
                        max_loss_per_contract = (contract.strike_price - price) * 100
                        
                        # Breakeven
                        breakeven = contract.strike_price - price
                        
                        # ROI (premium / strike price)
                        roi = (price / contract.strike_price) * 100
                        
                        # Annualized return
                        annualized_return = roi * (365 / days_to_exp)
                        
                        # Calculate probability of profit (for puts: profit if stock stays above strike)
                        probability_of_profit = self._calculate_probability_of_profit(
                            greeks['delta'], days_to_exp, greeks['implied_volatility'],
                            contract.strike_price, stock_price, stock_data
                        )
                        
                        # Calculate individual scores using reference implementation
                        tech_score = self._calculate_technical_score(stock_data)
                        greeks_score = self._calculate_greeks_score(greeks, days_to_exp)
                        liquidity_score = self._calculate_liquidity_score(
                            greeks['volume'], greeks['open_interest'], price
                        )
                        resistance_score = self._calculate_resistance_score(
                            contract.strike_price, stock_price, stock_data
                        )
                        
                        # Calculate cash required score
                        cash_required_score = self._calculate_cash_required_score(
                            contract.strike_price, price
                        )
                        
                        # Calculate overall score using optimized weighted components
                        ivr_value = stock_data.get('ivr', None)
                        score = self._calculate_overall_score(
                            tech_score, greeks_score, liquidity_score,
                            resistance_score, probability_of_profit, annualized_return, ivr_value
                        )
                        
                        recommendation = {
                            'ticker': ticker,
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
                            'moneyness': moneyness,
                            'max_profit_per_contract': max_profit_per_contract,
                            'max_loss_per_contract': max_loss_per_contract,
                            'breakeven': breakeven,
                            'roi': roi,
                            'annualized_return': annualized_return,
                            'probability_of_profit': probability_of_profit,
                            'score': score,
                            'tech_score': tech_score,
                            'greeks_score': greeks_score,
                            'liquidity_score': liquidity_score,
                            'resistance_score': resistance_score,
                            'volume': greeks['volume'],
                            'open_interest': greeks['open_interest'],
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
                            'ivr': ivr_value,
                            # Score breakdown for detailed view
                            'score_breakdown': {
                                'technical_score': tech_score,
                                'greeks_score': greeks_score,
                                'liquidity_score': liquidity_score,
                                'resistance_score': resistance_score,
                                'probability_score': probability_of_profit * 100,
                                'roi_score': min(100, max(0, roi * 5)),
                                'cash_required_score': cash_required_score,
                                'ivr_score': self._calculate_ivr_score(ivr_value),
                                'ivr_value': ivr_value,
                                'weights': SCORING_WEIGHTS,
                                'overall_score': score
                            }
                        }
                        
                        ticker_recommendations.append(recommendation)
                        
                    except Exception as e:
                        logger.warning(f"Error processing option: {e}")
                        continue
                
                # Sort by score and take best options for this ticker
                ticker_recommendations.sort(key=lambda x: x['score'], reverse=True)
                recommendations.extend(ticker_recommendations)
                
                # Update debug tracking
                if ticker_recommendations:
                    ticker_debug['delta_filter_ok'] = True
                    ticker_debug['final_recommendations'] = len(ticker_recommendations)
                    debug_stats['after_delta_filter'] += 1
                    debug_stats['final_with_recommendations'] += 1
                    tickers_with_recommendations += 1
                
                debug_stats['ticker_details'][ticker] = ticker_debug
                
                logger.info(f"✅ Generated {len(ticker_recommendations)} recommendations for {ticker}")
                logger.info(f"🔍 DEBUG: {ticker} - Strike range: {len(put_contracts)} contracts, Delta filter: {contracts_passed_delta} passed, Final: {len(ticker_recommendations)} recommendations")
                
            except Exception as e:
                logger.error(f"❌ Error processing {ticker}: {e}")
                ticker_debug['skip_reason'] = f'Processing error: {str(e)[:50]}'
                debug_stats['ticker_details'][ticker] = ticker_debug
                continue
        
        # Sort all recommendations by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply 20 options limit with priority: at least 1 per ticker, then fill by score
        final_recommendations = self._apply_ticker_priority_limit(recommendations, max_options=20)
        
        # Count unique tickers processed
        processed_tickers = len([ticker for ticker in set(rec['ticker'] for rec in recommendations)])
        final_ticker_count = len(set(rec['ticker'] for rec in final_recommendations))
        
        logger.info(f"🎯 Final CSP recommendations: {len(final_recommendations)} options from {total_options_considered} total analyzed")
        logger.info(f"📊 Processed {processed_tickers} unique tickers, found options for {final_ticker_count} tickers")
        
        # Comprehensive debug summary
        logger.info("=" * 80)
        logger.info("🔍 CSP RECOMMENDER DEBUG SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📊 PIPELINE FILTERING RESULTS:")
        logger.info(f"   • Initial tickers: {debug_stats['initial_tickers']}")
        logger.info(f"   • After stock data: {debug_stats['after_stock_data']} (-{debug_stats['initial_tickers'] - debug_stats['after_stock_data']})")
        logger.info(f"   • After skip filter: {debug_stats['after_skip_filter']} (-{debug_stats['after_stock_data'] - debug_stats['after_skip_filter']})")
        logger.info(f"   • After monthly exp: {debug_stats['after_monthly_exp']} (-{debug_stats['after_skip_filter'] - debug_stats['after_monthly_exp']})")
        logger.info(f"   • After put contracts: {debug_stats['after_put_contracts']} (-{debug_stats['after_monthly_exp'] - debug_stats['after_put_contracts']})")
        logger.info(f"   • After strike range: {debug_stats['after_strike_range']} (-{debug_stats['after_put_contracts'] - debug_stats['after_strike_range']})")
        logger.info(f"   • After delta filter: {debug_stats['after_delta_filter']} (-{debug_stats['after_strike_range'] - debug_stats['after_delta_filter']})")
        logger.info(f"   • Final with recommendations: {debug_stats['final_with_recommendations']}")
        
        logger.info(f"\n📋 DETAILED SKIP REASONS:")
        for reason, count in skip_reasons.items():
            logger.info(f"   • {reason}: {count} tickers")
        
        logger.info(f"\n🎯 TICKER DIVERSITY ANALYSIS:")
        logger.info(f"   • Total tickers with recommendations: {debug_stats['final_with_recommendations']}")
        logger.info(f"   • Unique tickers in final results: {final_ticker_count}")
        logger.info(f"   • Diversity rule triggered: {'YES' if debug_stats['final_with_recommendations'] >= 20 else 'NO'}")
        
        if debug_stats['final_with_recommendations'] < 20:
            logger.warning(f"⚠️  DIVERSITY ISSUE: Only {debug_stats['final_with_recommendations']} tickers made it through filtering!")
            logger.warning(f"   This prevents the diversity rule from working (needs 20+ tickers)")
            logger.warning(f"   Main bottlenecks:")
            if debug_stats['initial_tickers'] - debug_stats['after_stock_data'] > 5:
                logger.warning(f"     - Stock data retrieval: {debug_stats['initial_tickers'] - debug_stats['after_stock_data']} tickers failed")
            if debug_stats['after_stock_data'] - debug_stats['after_monthly_exp'] > 5:
                logger.warning(f"     - Monthly expirations: {debug_stats['after_stock_data'] - debug_stats['after_monthly_exp']} tickers failed")
            if debug_stats['after_monthly_exp'] - debug_stats['after_put_contracts'] > 5:
                logger.warning(f"     - Put contracts: {debug_stats['after_monthly_exp'] - debug_stats['after_put_contracts']} tickers failed")
            if debug_stats['after_put_contracts'] - debug_stats['after_strike_range'] > 5:
                logger.warning(f"     - Strike range (80%-98%): {debug_stats['after_put_contracts'] - debug_stats['after_strike_range']} tickers failed")
            if debug_stats['after_strike_range'] - debug_stats['after_delta_filter'] > 5:
                logger.warning(f"     - Delta filter (0.05-0.80): {debug_stats['after_strike_range'] - debug_stats['after_delta_filter']} tickers failed")
        
        logger.info("=" * 80)
        
        # Summary of skip reasons
        if skip_reasons:
            logger.info(f"📋 Skip Summary: {tickers_skipped}/{len(tickers)} tickers skipped")
            for reason, count in skip_reasons.items():
                logger.info(f"   • {reason}: {count} tickers")
        else:
            logger.info(f"✅ All {len(tickers)} tickers processed successfully")
        
        return {
            'recommendations': final_recommendations,
            'total_considered': total_options_considered,
            'underlying_tickers_considered': processed_tickers,
            'underlying_tickers_in_results': final_ticker_count,
            'debug_stats': debug_stats
        }
    
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
        """Get stock data with technical indicators (with daily caching)"""
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
            
            # Calculate technical indicators
            rsi = self._calculate_rsi(prices)
            macd = self._calculate_macd(prices)
            bollinger = self._calculate_bollinger_bands(prices)
            support_resistance = self._calculate_support_resistance(prices)
            
            stock_data = {
                'current_price': current_price,
                'rsi': rsi,
                'macd': macd,
                'bollinger_bands': bollinger,
                'support_resistance': support_resistance,
                'price_change_pct': self._calculate_price_change_pct(prices)
            }
            
            # Cache the result
            self.technical_cache[cache_key] = (stock_data, time.time())
            
            return stock_data
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {e}")
            return None
    
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
        """Calculate probability of profit - adapted for puts"""
        try:
            # For cash secured puts: profit if stock stays above strike
            # Use Delta-based approximation: P(profit) ≈ 1 - |Delta| (for puts, delta is negative)
            prob = max(0.05, min(0.95, 1 - abs(delta)))
            
            # Adjust for time to expiration
            if days_to_exp > 45:
                prob *= 0.92
            elif days_to_exp > 30:
                prob *= 0.95
            elif days_to_exp < 7:
                prob *= 1.03
            
            # Adjust for IV
            if iv > 0.8:
                prob *= 0.85
            elif iv > 0.6:
                prob *= 0.90
            elif iv > 0.4:
                prob *= 0.95
            elif iv < 0.2:
                prob *= 1.05
            
            # Adjust for strike distance (for puts, closer to current price is better)
            distance_pct = (stock_price - strike) / stock_price
            if distance_pct < 0.05:  # Very close to current price
                prob *= 0.90
            elif distance_pct < 0.10:  # Close to current price
                prob *= 0.95
            
            # Technical indicator adjustments for puts
            try:
                rsi = stock_data.get('rsi', 50)
                macd = stock_data.get('macd', {})
                bb = stock_data.get('bollinger_bands', {})
                
                # RSI adjustment for puts
                if rsi < 30:  # Oversold - good for put selling
                    prob *= 1.05
                elif rsi > 70:  # Overbought - bad for put selling
                    prob *= 0.95
                
                # MACD adjustment
                is_bullish = macd.get('histogram', 0) > 0
                if is_bullish:
                    prob *= 1.03
                
                # BB position adjustment
                bb_position = bb.get('position', 0.5)
                if bb_position < 0.2:  # Near lower band - good for put selling
                    prob *= 1.02
            except:
                pass
            
            return max(0.05, min(0.95, prob))
        except:
            return 0.5
    
    def _calculate_technical_score(self, stock_data: Dict) -> float:
        """Calculate technical score optimized for put selling - prioritize oversold conditions and bullish trends"""
        try:
            score = 50.0
            
            rsi = stock_data.get('rsi', 50)
            macd = stock_data.get('macd', {})
            bb = stock_data.get('bollinger_bands', {})
            price_change = stock_data.get('price_change_pct', 0)
            ma_20 = stock_data.get('ma_20', 0)
            ma_50 = stock_data.get('ma_50', 0)
            current_price = stock_data.get('current_price', 0)
            volume_ratio = stock_data.get('volume_ratio', 1.0)
            
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
            
            # Moving average analysis for trend confirmation
            if current_price > ma_20 > ma_50:  # Strong bullish trend
                score += 20
            elif current_price > ma_20:  # Above 20-day MA
                score += 15
            elif current_price > ma_50:  # Above 50-day MA
                score += 10
            elif current_price < ma_50:  # Below 50-day MA - concerning
                score -= 10
            
            # MACD analysis for momentum
            histogram = macd.get('histogram', 0)
            is_bullish = histogram > 0
            if is_bullish:
                score += 15  # Bullish - good for selling puts
            else:
                score -= 10  # Bearish - bad for selling puts
            
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
            
            # Price momentum analysis
            if price_change > 2:  # Strong positive momentum - excellent for put selling
                score += 15
            elif price_change > 0:  # Positive momentum - good for put selling
                score += 10
            elif price_change < -5:  # Strong negative momentum - bad for put selling
                score -= 20
            elif price_change < -2:  # Negative momentum - concerning
                score -= 10
            
            # Volume confirmation
            if volume_ratio > 1.5:  # High volume - confirms move
                score += 10
            elif volume_ratio > 1.2:  # Above average volume
                score += 5
            elif volume_ratio < 0.5:  # Low volume - weak confirmation
                score -= 5
            
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
