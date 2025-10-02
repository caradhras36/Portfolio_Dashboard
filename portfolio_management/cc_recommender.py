"""
Covered Calls Recommender - Uses the exact proven architecture from Options_Trade_Search
Includes rigorous scoring, probability calculations, and technical analysis
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

# Scoring weights (from reference implementation - config.py)
SCORING_WEIGHTS = {
    'ivr': 0.10,           # IV Rank importance
    'greeks': 0.15,        # Greeks (Delta, Theta, Vega)
    'technical': 0.15,     # Technical indicators
    'liquidity': 0.10,     # Open interest and volume
    'resistance': 0.10,    # Support/resistance levels
    'probability': 0.25,   # Probability of profit (HIGHEST weight)
    'roi': 0.15            # Return on Investment
}

# RSI constants
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

class CoveredCallRecommender:
    """Find best covered call opportunities from portfolio stocks"""
    
    def __init__(self, polygon_api_key: str = None, supabase_client = None):
        self.polygon_client = RESTClient(polygon_api_key or os.getenv('POLYGON_API_KEY'))
        self.supabase = supabase_client  # For IVR data
        self.weights = SCORING_WEIGHTS
        self.technical_cache = {}  # Cache for technical indicators
        self.ivr_cache = {}  # Cache for IVR data
        self.cache_ttl = 86400  # 24 hours (1 day) cache for technical indicators
        self.cache_date = datetime.now().date()  # Track cache date
        
        # ETFs and bonds to skip (no options or not suitable for CCs)
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
    
    def get_recommendations(self, stock_positions: List[Dict], blocked_tickers: set) -> List[Dict]:
        """Get covered call recommendations for portfolio stocks"""
        recommendations = []
        
        for stock_pos in stock_positions:
            ticker = stock_pos['ticker']
            shares_owned = stock_pos['quantity']
            
            # Smart filtering for faster processing
            if ticker in blocked_tickers:
                logger.info(f"⏭️  Skipping {ticker} - already has covered call")
                continue
            
            if shares_owned < 100:
                logger.info(f"⏭️  Skipping {ticker} - only {shares_owned} shares (need 100 for 1 contract)")
                continue
            
            if ticker in self.skip_tickers:
                logger.info(f"⏭️  Skipping {ticker} - ETF/Bond (no suitable options)")
                continue
            
            # Note: Removed penny stock filter - we'll let liquidity criteria handle this
            
            try:
                logger.info(f"🔄 Processing {ticker} ({shares_owned} shares)...")
                
                # Get stock price and technical data
                stock_data = self._get_stock_data(ticker)
                if not stock_data:
                    logger.warning(f"⚠️  No stock data for {ticker}")
                    continue
                
                stock_price = stock_data['current_price']
                logger.info(f"💰 {ticker} price: ${stock_price:.2f}")
                
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
                    logger.warning(f"⚠️  No monthly expirations found for {ticker}")
                    continue
                
                # Get options contracts for monthly expirations only
                all_call_contracts = []
                for exp_date in monthly_expirations:
                    try:
                        contracts = list(self.polygon_client.list_options_contracts(
                            underlying_ticker=ticker,
                            expiration_date=exp_date,
                            contract_type='call',
                            limit=50
                        ))
                        all_call_contracts.extend(contracts)
                        logger.info(f"✅ Found {len(contracts)} call contracts for {ticker} exp {exp_date}")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not fetch contracts for {ticker} exp {exp_date}: {e}")
                        continue
                
                if not all_call_contracts:
                    logger.warning(f"⚠️  No call contracts for {ticker} monthly expirations")
                    continue
                
                logger.info(f"📞 Total {len(all_call_contracts)} monthly call contracts for {ticker}")
                
                # Filter for relevant strike range
                lower_strike = stock_price * 0.95
                upper_strike = stock_price * 1.20
                
                call_contracts = [
                    c for c in all_call_contracts
                    if lower_strike <= c.strike_price <= upper_strike
                ]
                
                logger.info(f"🎯 Found {len(call_contracts)} call options in strike range")
                
                # Process top contracts
                for contract in call_contracts[:10]:  # Limit to 10 per stock
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
                        intrinsic = max(0, stock_price - contract.strike_price)
                        if intrinsic > 0:
                            moneyness = 'ITM'
                        elif abs(stock_price - contract.strike_price) < stock_price * 0.02:
                            moneyness = 'ATM'
                        else:
                            moneyness = 'OTM'
                        
                        # Calculate metrics (PER CONTRACT - 100 shares)
                        # For SELLING a covered call on stock you ALREADY OWN:
                        premium_per_contract = price * 100
                        
                        # Max profit: Just the premium received (you keep the premium no matter what)
                        # Note: If stock goes above strike, you're capped at strike price
                        # But the "profit from selling the call" is just the premium
                        max_profit_per_contract = price * 100
                        
                        # Max loss: Stock drops to $0, but you keep the premium
                        # Loss on the stock position = stock_price - 0 = stock_price
                        # But you collected premium, so net loss = stock_price - premium
                        max_loss_per_contract = (stock_price - price) * 100
                        
                        # Breakeven
                        breakeven = stock_price - price
                        
                        # ROI
                        roi = (price / stock_price) * 100
                        
                        # Annualized return
                        annualized_return = roi * (365 / days_to_exp)
                        
                        # Calculate probability of profit (using Delta-based method from reference)
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
                        
                        # Calculate overall score using weighted components (with IVR)
                        ivr_value = stock_data.get('ivr', None)
                        score = self._calculate_overall_score(
                            tech_score, greeks_score, liquidity_score,
                            resistance_score, probability_of_profit, roi, ivr_value
                        )
                        
                        recommendation = {
                            'ticker': ticker,
                            'shares_owned': shares_owned,
                            'contracts_available': int(shares_owned / 100),
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
                                'ivr_score': self._calculate_ivr_score(ivr_value),
                                'ivr_value': ivr_value,
                                'weights': SCORING_WEIGHTS,
                                'overall_score': score
                            }
                        }
                        
                        recommendations.append(recommendation)
                        
                    except Exception as e:
                        logger.warning(f"Error processing option: {e}")
                        continue
                
                logger.info(f"✅ Generated {len([r for r in recommendations if r['ticker'] == ticker])} recommendations for {ticker}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {ticker}: {e}")
                continue
        
        # Sort by score first
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply ticker diversity limit (same logic as CSP)
        final_recommendations = self._apply_ticker_priority_limit(recommendations, max_options=20)
        
        # Count unique tickers processed
        processed_tickers = len([ticker for ticker in set(rec['ticker'] for rec in recommendations)])
        final_ticker_count = len(set(rec['ticker'] for rec in final_recommendations))
        
        logger.info(f"🎯 Final CC recommendations: {len(final_recommendations)} options from {processed_tickers} underlying tickers")
        logger.info(f"📊 Processed {processed_tickers} unique tickers, found options for {final_ticker_count} tickers")
        
        return {
            'recommendations': final_recommendations,
            'underlying_tickers_considered': processed_tickers,
            'underlying_tickers_in_results': final_ticker_count
        }
    
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
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if hist.empty or len(hist) < 50:
                return None
            
            prices = hist['Close']
            current_price = float(prices.iloc[-1])
            
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
        """Calculate probability of profit - exact copy from reference"""
        try:
            # For covered calls: profit if stock stays below strike
            # Use Delta-based approximation: P(profit) ≈ 1 - Delta
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
            
            # Adjust for strike distance
            distance_pct = (strike - stock_price) / stock_price
            if distance_pct < 0.05:
                prob *= 0.90
            elif distance_pct < 0.10:
                prob *= 0.95
            
            # Technical indicator adjustments
            try:
                rsi = stock_data.get('rsi', 50)
                macd = stock_data.get('macd', {})
                bb = stock_data.get('bollinger_bands', {})
                
                # RSI adjustment
                if rsi > 70:  # Overbought - good for call selling
                    prob *= 1.05
                elif rsi < 30:  # Oversold - bad for call selling
                    prob *= 0.95
                
                # MACD adjustment
                is_bearish = macd.get('histogram', 0) < 0
                if is_bearish:
                    prob *= 1.03
                
                # BB position adjustment
                bb_position = bb.get('position', 0.5)
                if bb_position > 0.8:
                    prob *= 1.02
            except:
                pass
            
            return max(0.05, min(0.95, prob))
        except:
            return 0.5
    
    def _calculate_technical_score(self, stock_data: Dict) -> float:
        """Calculate technical score - exact copy from reference"""
        try:
            score = 50.0
            
            rsi = stock_data.get('rsi', 50)
            macd = stock_data.get('macd', {})
            bb = stock_data.get('bollinger_bands', {})
            price_change = stock_data.get('price_change_pct', 0)
            
            # RSI analysis for covered calls
            if rsi > RSI_OVERBOUGHT:
                score += 20  # Overbought - good for selling calls
            elif rsi < RSI_OVERSOLD:
                score -= 20  # Oversold - bad for selling calls
            
            # MACD analysis
            histogram = macd.get('histogram', 0)
            is_bearish = histogram < 0
            if is_bearish:
                score += 10  # Bearish - good for selling calls
            else:
                score -= 10  # Bullish - bad for selling calls
            
            # Bollinger Bands position
            position = bb.get('position', 0.5)
            if position > 0.8:
                score += 15  # Near upper band - good for selling calls
            elif position < 0.2:
                score -= 15  # Near lower band - bad for selling calls
            elif 0.3 <= position <= 0.7:
                score += 5
            
            # Price momentum
            if abs(price_change) > 5:
                score -= 10  # High volatility
            elif abs(price_change) < 1:
                score += 5  # Low volatility
            
            return max(0, min(100, score))
        except:
            return 50.0
    
    def _calculate_greeks_score(self, greeks: Dict, days_to_exp: int) -> float:
        """Calculate Greeks score - exact copy from reference"""
        try:
            score = 50.0
            
            delta = abs(greeks['delta'])
            
            # Delta scoring for selling
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
            
            # Theta scoring
            theta = abs(greeks['theta'])
            if theta > 0.1:
                score += 15
            elif theta > 0.05:
                score += 5
            else:
                score -= 10
            
            # Vega scoring
            vega = abs(greeks['vega'])
            if vega < 0.1:
                score += 10
            elif vega > 0.3:
                score -= 15
            
            # Gamma scoring
            gamma = abs(greeks['gamma'])
            if gamma < 0.01:
                score += 10
            elif gamma > 0.05:
                score -= 15
            
            return max(0, min(100, score))
        except:
            return 50.0
    
    def _calculate_liquidity_score(self, volume: int, open_interest: int, price: float) -> float:
        """Calculate liquidity score - exact copy from reference"""
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
        """Calculate resistance score - exact copy from reference"""
        try:
            score = 50.0
            
            resistance = stock_data.get('support_resistance', {}).get('resistance', stock_price * 1.05)
            
            # For calls: prefer strikes near resistance
            strike_ratio = strike / resistance
            if 0.95 <= strike_ratio <= 1.05:
                score += 15
            elif 0.9 <= strike_ratio <= 1.1:
                score += 8
            else:
                score += 5
            
            return max(0, min(100, score))
        except:
            return 50.0
    
    def _calculate_ivr_score(self, ivr: Optional[float]) -> float:
        """Calculate IVR score component"""
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
    
    def _calculate_overall_score(self, tech_score: float, greeks_score: float,
                                 liquidity_score: float, resistance_score: float,
                                 probability: float, roi: float, ivr: Optional[float] = None) -> float:
        """Calculate weighted overall score - exact copy from reference"""
        try:
            # IVR score calculation (from reference implementation)
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
            
            # ROI score (normalize)
            roi_score = min(100, max(0, roi * 5))
            
            # Weighted average
            overall = (
                ivr_score * self.weights['ivr'] +
                greeks_score * self.weights['greeks'] +
                tech_score * self.weights['technical'] +
                liquidity_score * self.weights['liquidity'] +
                resistance_score * self.weights['resistance'] +
                prob_score * self.weights['probability'] +
                roi_score * self.weights['roi']
            )
            
            return round(overall, 2)
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 50.0
