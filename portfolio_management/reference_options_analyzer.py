"""Options analysis engine with Greeks, IVR, and technical analysis."""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from config import SCORING_WEIGHTS, MIN_IV_THRESHOLD

# RSI constants
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

logger = logging.getLogger(__name__)

@dataclass
class OptionScenario:
    """Represents a single option trading scenario."""
    ticker: str
    option_ticker: str
    strike: float
    expiration: str
    contract_type: str
    strategy: str
    current_price: float
    option_price: float
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    open_interest: int
    volume: int
    days_to_expiration: int
    distance_from_price: float
    ivr: Optional[float] = None
    technical_score: float = 0.0
    greeks_score: float = 0.0
    liquidity_score: float = 0.0
    resistance_score: float = 0.0
    overall_score: float = 0.0
    probability_of_profit: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven: float = 0.0
    roi: float = 0.0  # Return on Investment percentage

class OptionsAnalyzer:
    """Analyzes options for selling strategies."""
    
    def __init__(self):
        self.weights = SCORING_WEIGHTS
    
    def analyze_option_scenario(self, option_data: Dict[str, Any], 
                              stock_data: Dict[str, Any], current_price: float) -> OptionScenario:
        """Analyze a single option scenario."""
        try:
            # Calculate days to expiration
            exp_date = datetime.strptime(option_data['expiration'], '%Y-%m-%d').date()
            current_date = datetime.now(timezone.utc).date()
            days_to_exp = (exp_date - current_date).days
            
            # Create base scenario
            scenario = OptionScenario(
                ticker=option_data['underlying'],
                option_ticker=option_data['ticker'],
                strike=option_data['strike'],
                expiration=option_data['expiration'],
                contract_type=option_data['contract_type'],
                strategy=self._determine_strategy(option_data['contract_type']),
                current_price=stock_data['current_price'],
                option_price=option_data.get('option_price', 0),
                implied_volatility=option_data['implied_volatility'],
                delta=option_data['delta'],
                gamma=option_data['gamma'],
                theta=option_data['theta'],
                vega=option_data['vega'],
                rho=option_data['rho'],
                open_interest=option_data['open_interest'],
                volume=option_data['volume'],
                days_to_expiration=days_to_exp,
                distance_from_price=option_data.get('distance_from_price', 0),
                ivr=stock_data.get('ivr')
            )
            
            # Calculate scores
            # Add strategy to stock_data for technical analysis
            stock_data_with_strategy = stock_data.copy()
            stock_data_with_strategy['strategy'] = scenario.strategy
            scenario.technical_score = self._calculate_technical_score(stock_data_with_strategy)
            scenario.greeks_score = self._calculate_greeks_score(scenario)
            scenario.liquidity_score = self._calculate_liquidity_score(scenario)
            scenario.resistance_score = self._calculate_resistance_score(scenario, stock_data)
            
            # Calculate overall score
            scenario.overall_score = self._calculate_overall_score(scenario)
            
            # Calculate trading metrics
            self._calculate_trading_metrics(scenario, stock_data)
            
            return scenario
            
        except Exception as e:
            logger.error(f"Error analyzing option scenario: {e}")
            return None
    
    def _determine_strategy(self, contract_type: str) -> str:
        """Determine trading strategy based on contract type."""
        if contract_type == 'call':
            return 'covered_call'
        elif contract_type == 'put':
            return 'cash_secured_put'
        else:
            return 'unknown'
    
    def _should_reject_scenario(self, scenario: OptionScenario, filter_level: str = "none") -> bool:
        """Check if scenario should be rejected based on progressive filter level."""
        try:
            delta = abs(scenario.delta) if scenario.delta is not None else 0
            
            if filter_level == "none":
                # No filters - accept everything
                return False
            elif filter_level == "iv_only":
                # Only IV filter (minimum 30%)
                iv = scenario.implied_volatility if scenario.implied_volatility is not None else 0
                return iv < 0.30
            elif filter_level == "basic":
                # Basic filters: IV + very basic probability
                iv = scenario.implied_volatility if scenario.implied_volatility is not None else 0
                if iv < 0.30:
                    return True
                if scenario.probability_of_profit < 0.15:  # Very low threshold
                    return True
                return False
            elif filter_level == "moderate":
                # Moderate filters: IV + probability + delta
                iv = scenario.implied_volatility if scenario.implied_volatility is not None else 0
                if iv < 0.35:
                    return True
                if scenario.probability_of_profit < 0.25:
                    return True
                if delta > 0.8:  # Very high delta threshold
                    return True
                return False
            elif filter_level == "strict":
                # Strict filters: Original criteria
                iv = scenario.implied_volatility if scenario.implied_volatility is not None else 0
                if iv < 0.40:
                    return True
                
                if scenario.contract_type == 'call':
                    # CCs: 40% min probability, delta < 0.6
                    if scenario.probability_of_profit < 0.40:
                        return True
                    if delta > 0.6:
                        return True
                elif scenario.contract_type == 'put':
                    # CSPs: 50% min probability, delta < 0.5
                    if scenario.probability_of_profit < 0.50:
                        return True
                    if delta > 0.5:
                        return True
                
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error in rejection check: {e}")
            return True  # Reject on error
    
    def analyze_options(self, ticker: str, current_price: float, options_data: List[Dict], 
                       stock_data: pd.DataFrame, ivr: Optional[float] = None) -> List[OptionScenario]:
        """Analyze multiple options and return scenarios."""
        try:
            scenarios = []
            
            # Convert stock data to dict for technical analysis
            # Try to fetch technical indicators from Polygon.io API first
            try:
                rsi_data = self._fetch_polygon_rsi(ticker)
                macd_data = self._fetch_polygon_macd(ticker)
                # Note: Bollinger Bands API not available in current SDK version
                bollinger_data = self._calculate_bollinger_bands(stock_data['Close'])
                logger.info(f"✅ Successfully fetched RSI and MACD from Polygon.io for {ticker}")
            except Exception as e:
                logger.warning(f"Could not fetch Polygon.io technical indicators for {ticker}: {e}")
                # Fallback to calculated indicators
                rsi_data = self._calculate_rsi(stock_data['Close'])
                macd_data = self._calculate_macd(stock_data['Close'])
                bollinger_data = self._calculate_bollinger_bands(stock_data['Close'])
            
            stock_dict = {
                'current_price': current_price,
                'rsi': rsi_data,
                'macd': macd_data,
                'bollinger_bands': bollinger_data,
                'support_resistance': {
                    'support': self._find_support_levels(stock_data['Close']),
                    'resistance': self._find_resistance_levels(stock_data['Close'])
                },
                'price_change_pct': self._calculate_price_change_pct(stock_data['Close'])
            }
            
            # Progressive filtering: Start with no filters, add one by one until we have 10+ options
            filter_levels = ["none", "iv_only", "basic", "moderate", "strict"]
            scenarios = []
            applied_filter = "none"
            
            for filter_level in filter_levels:
                scenarios = []
                filtered_count = 0
                
                for option in options_data:
                    try:
                        # Analyze each option
                        scenario = self.analyze_option_scenario(option, stock_dict, current_price)
                        if scenario:
                            # Add IVR if available
                            if ivr:
                                scenario.ivr = ivr
                            
                            # Apply progressive filtering
                            if self._should_reject_scenario(scenario, filter_level):
                                filtered_count += 1
                                continue
                            
                            scenarios.append(scenario)
                    except Exception as e:
                        logger.warning(f"Error analyzing option {option.get('ticker', 'unknown')}: {e}")
                        continue
                
                applied_filter = filter_level
                logger.info(f"Filter level '{filter_level}': Generated {len(scenarios)} scenarios for {ticker} (filtered out {filtered_count})")
                
                # If we have 10+ scenarios, stop here
                if len(scenarios) >= 10:
                    break
            
            logger.info(f"Final filter level for {ticker}: '{applied_filter}' with {len(scenarios)} scenarios")
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error analyzing options for {ticker}: {e}")
            return []
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI for the given prices."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not rsi.empty else 50.0
        except:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD for the given prices."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': float(macd_line.iloc[-1]) if not macd_line.empty else 0.0,
                'signal': float(signal_line.iloc[-1]) if not signal_line.empty else 0.0,
                'histogram': float(histogram.iloc[-1]) if not histogram.empty else 0.0
            }
        except:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands for the given prices."""
        try:
            sma = prices.rolling(window=period).mean()
            std_dev = prices.rolling(window=period).std()
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            
            current_price = float(prices.iloc[-1]) if not prices.empty else 0.0
            upper = float(upper_band.iloc[-1]) if not upper_band.empty else current_price
            middle = float(sma.iloc[-1]) if not sma.empty else current_price
            lower = float(lower_band.iloc[-1]) if not lower_band.empty else current_price
            
            # Calculate position within bands (0 = lower band, 1 = upper band)
            if upper != lower:
                position = (current_price - lower) / (upper - lower)
            else:
                position = 0.5
            
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'position': position  # Add position for technical scoring
            }
        except:
            current_price = float(prices.iloc[-1]) if not prices.empty else 0.0
            return {
                'upper': current_price, 
                'middle': current_price, 
                'lower': current_price,
                'position': 0.5
            }
    
    def _calculate_price_change_pct(self, prices: pd.Series, period: int = 5) -> float:
        """Calculate price change percentage over specified period."""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            current_price = prices.iloc[-1]
            past_price = prices.iloc[-(period + 1)]
            
            if past_price == 0:
                return 0.0
            
            change_pct = ((current_price - past_price) / past_price) * 100
            return float(change_pct)
        except:
            return 0.0
    
    def _fetch_polygon_rsi(self, ticker: str, window: int = 14) -> float:
        """Fetch RSI from Polygon.io API."""
        try:
            from polygon import RESTClient
            import os
            
            api_key = os.getenv('POLYGON_API_KEY')
            if not api_key:
                raise ValueError("POLYGON_API_KEY not found")
            
            client = RESTClient(api_key)
            
            # Fetch RSI data
            response = client.get_rsi(
                ticker=ticker,
                window=window,
                series_type="close",
                timespan="day",
                limit=1
            )
            
            if response.values and len(response.values) > 0:
                return float(response.values[0].value)
            else:
                raise ValueError("No RSI data returned")
                
        except Exception as e:
            logger.warning(f"Error fetching RSI from Polygon.io for {ticker}: {e}")
            raise
    
    def _fetch_polygon_macd(self, ticker: str) -> Dict[str, float]:
        """Fetch MACD from Polygon.io API."""
        try:
            from polygon import RESTClient
            import os
            
            api_key = os.getenv('POLYGON_API_KEY')
            if not api_key:
                raise ValueError("POLYGON_API_KEY not found")
            
            client = RESTClient(api_key)
            
            # Fetch MACD data
            response = client.get_macd(
                ticker=ticker,
                series_type="close",
                timespan="day",
                limit=1
            )
            
            if response.values and len(response.values) > 0:
                macd_value = response.values[0]
                return {
                    'macd': float(macd_value.value),
                    'signal': float(macd_value.signal),
                    'histogram': float(macd_value.histogram)
                }
            else:
                raise ValueError("No MACD data returned")
                
        except Exception as e:
            logger.warning(f"Error fetching MACD from Polygon.io for {ticker}: {e}")
            raise
    
    
    def _find_resistance_levels(self, prices: pd.Series, window: int = 20) -> List[float]:
        """Find resistance levels in the price data."""
        try:
            # Simple peak detection
            peaks = []
            for i in range(window, len(prices) - window):
                if all(prices.iloc[i] >= prices.iloc[i-j] for j in range(1, window+1)) and \
                   all(prices.iloc[i] >= prices.iloc[i+j] for j in range(1, window+1)):
                    peaks.append(float(prices.iloc[i]))
            
            # Return top 3 resistance levels
            return sorted(peaks, reverse=True)[:3]
        except:
            return []
    
    def _find_support_levels(self, prices: pd.Series, window: int = 20) -> List[float]:
        """Find support levels in the price data."""
        try:
            # Simple trough detection
            troughs = []
            for i in range(window, len(prices) - window):
                if all(prices.iloc[i] <= prices.iloc[i-j] for j in range(1, window+1)) and \
                   all(prices.iloc[i] <= prices.iloc[i+j] for j in range(1, window+1)):
                    troughs.append(float(prices.iloc[i]))
            
            # Return top 3 support levels
            return sorted(troughs)[:3]
        except:
            return []
    
    def _calculate_technical_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate technical analysis score (0-100)."""
        try:
            score = 50.0  # Base score
            
            # RSI analysis - strategy-specific logic
            rsi = stock_data.get('rsi', 50)
            strategy = stock_data.get('strategy', 'unknown')
            
            if strategy == 'covered_call':
                if rsi > RSI_OVERBOUGHT:  # RSI > 70
                    score += 20  # Overbought - GOOD for selling calls (stock likely to decline)
                elif rsi < RSI_OVERSOLD:  # RSI < 30
                    score -= 20  # Oversold - BAD for selling calls (stock likely to bounce up)
            elif strategy == 'cash_secured_put':
                if rsi < RSI_OVERSOLD:  # RSI < 30
                    score += 20  # Oversold - GOOD for selling puts (stock likely to stay above support)
                elif rsi > RSI_OVERBOUGHT:  # RSI > 70
                    score -= 20  # Overbought - BAD for selling puts (stock likely to decline)
            
            # MACD analysis - strategy-specific logic
            macd = stock_data.get('macd', {})
            macd_line = macd.get('macd', 0)
            signal_line = macd.get('signal', 0)
            histogram = macd.get('histogram', 0)
            
            is_bullish = macd_line > signal_line and histogram > 0
            is_bearish = macd_line < signal_line and histogram < 0
            
            if strategy == 'covered_call':
                if is_bearish:
                    score += 10  # Bearish MACD - GOOD for selling calls (downward momentum)
                elif is_bullish:
                    score -= 10  # Bullish MACD - BAD for selling calls (upward momentum)
            elif strategy == 'cash_secured_put':
                if is_bullish:
                    score += 10  # Bullish MACD - GOOD for selling puts (upward momentum)
                elif is_bearish:
                    score -= 10  # Bearish MACD - BAD for selling puts (downward momentum)
            
            # Bollinger Bands analysis - strategy-specific logic
            bb = stock_data.get('bollinger_bands', {})
            position = bb.get('position', 0.5)  # 0 = lower band, 1 = upper band
            
            if strategy == 'covered_call':
                if position > 0.8:  # Near upper band
                    score += 15  # GOOD for selling calls (stock likely to bounce down from resistance)
                elif position < 0.2:  # Near lower band
                    score -= 15  # BAD for selling calls (stock likely to bounce up)
                elif 0.3 <= position <= 0.7:  # Middle range
                    score += 5   # Neutral to good
            elif strategy == 'cash_secured_put':
                if position < 0.2:  # Near lower band
                    score += 15  # GOOD for selling puts (stock likely to bounce up from support)
                elif position > 0.8:  # Near upper band
                    score -= 15  # BAD for selling puts (stock likely to bounce down)
                elif 0.3 <= position <= 0.7:  # Middle range
                    score += 5   # Neutral to good
            
            # Price momentum
            price_change = stock_data.get('price_change_pct', 0)
            if abs(price_change) > 5:  # High volatility
                score -= 10
            elif abs(price_change) < 1:  # Low volatility
                score += 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 50.0
    
    def _calculate_greeks_score(self, scenario: OptionScenario) -> float:
        """Calculate Greeks-based score (0-100)."""
        try:
            score = 50.0  # Base score
            
            # Delta analysis (for selling strategies)
            delta = abs(scenario.delta) if scenario.delta is not None else 0
            if 0.15 <= delta <= 0.35:  # Sweet spot for selling
                score += 20
            elif 0.1 <= delta <= 0.5:  # Acceptable range
                score += 10
            elif 0.5 < delta <= 0.7:  # High delta - moderate penalty
                score -= 15
            elif 0.7 < delta <= 0.9:  # Very high delta - large penalty
                score -= 30
            elif delta > 0.9:  # Extremely high delta (deep ITM) - very large penalty
                score -= 50
            else:  # Very low delta
                score -= 10
            
            # Theta analysis (time decay)
            theta = abs(scenario.theta) if scenario.theta is not None else 0
            if theta > 0.1:  # Good time decay
                score += 15
            elif theta > 0.05:  # Decent time decay
                score += 5
            else:
                score -= 10
            
            # Vega analysis (volatility sensitivity)
            vega = abs(scenario.vega) if scenario.vega is not None else 0
            if vega < 0.1:  # Low volatility sensitivity
                score += 10
            elif vega > 0.3:  # High volatility sensitivity
                score -= 15
            
            # Gamma analysis (acceleration risk)
            gamma = abs(scenario.gamma) if scenario.gamma is not None else 0
            if gamma < 0.01:  # Low gamma risk
                score += 10
            elif gamma > 0.05:  # High gamma risk
                score -= 15
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating Greeks score: {e}")
            return 50.0
    
    def _calculate_liquidity_score(self, scenario: OptionScenario) -> float:
        """Calculate liquidity score (0-100)."""
        try:
            score = 0.0
            
            # Open interest scoring
            oi = scenario.open_interest
            if oi >= 1000:
                score += 40
            elif oi >= 500:
                score += 30
            elif oi >= 100:
                score += 20
            else:
                score += 5
            
            # Volume scoring
            volume = scenario.volume
            if volume >= 100:
                score += 30
            elif volume >= 50:
                score += 20
            elif volume >= 10:
                score += 10
            else:
                score += 5
            
            # Bid-ask spread (estimated from option price)
            if scenario.option_price > 0:
                # Assume reasonable spread for liquid options
                if scenario.option_price > 1.0:
                    score += 20
                elif scenario.option_price > 0.5:
                    score += 15
                else:
                    score += 10
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 50.0
    
    def _calculate_resistance_score(self, scenario: OptionScenario, 
                                  stock_data: Dict[str, Any]) -> float:
        """Calculate support/resistance score (0-100)."""
        try:
            score = 50.0  # Base score
            
            support_resistance = stock_data.get('support_resistance', {})
            
            # Handle lists returned by support/resistance functions
            support_list = support_resistance.get('support', [])
            resistance_list = support_resistance.get('resistance', [])
            
            # Use the closest support/resistance level to current price
            current_price = scenario.current_price
            
            if isinstance(support_list, list) and support_list:
                # Find closest support level below current price
                support = max([s for s in support_list if s < current_price], default=current_price * 0.95)
            else:
                support = current_price * 0.95
                
            if isinstance(resistance_list, list) and resistance_list:
                # Find closest resistance level above current price
                resistance = min([r for r in resistance_list if r > current_price], default=current_price * 1.05)
            else:
                resistance = current_price * 1.05
            
            # Calculate distance from current price to strike
            price_distance_pct = abs(scenario.strike - current_price) / current_price
            
            # For call selling: prefer strikes near resistance but not too close
            if scenario.contract_type == 'call':
                strike_ratio = scenario.strike / resistance
                if 0.95 <= strike_ratio <= 1.05:  # Near resistance
                    score += 15  # Reduced from 20
                elif 0.9 <= strike_ratio <= 1.1:  # Close to resistance
                    score += 8   # Reduced from 10
                elif price_distance_pct > 0.15:  # Too far out
                    score -= 5   # Reduced penalty
                else:
                    score += 5   # Give some credit for reasonable strikes
            
            # For put selling: prefer strikes near support but not too close
            elif scenario.contract_type == 'put':
                strike_ratio = scenario.strike / support
                if 0.95 <= strike_ratio <= 1.05:  # Near support
                    score += 15  # Reduced from 20
                elif 0.9 <= strike_ratio <= 1.1:  # Close to support
                    score += 8   # Reduced from 10
                elif price_distance_pct > 0.15:  # Too far out
                    score -= 5   # Reduced penalty
                else:
                    score += 5   # Give some credit for reasonable strikes
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating resistance score: {e}")
            return 50.0
    
    def _calculate_overall_score(self, scenario: OptionScenario) -> float:
        """Calculate weighted overall score."""
        try:
            # IVR score (if available)
            ivr_score = 50.0
            if scenario.ivr is not None:
                if scenario.ivr >= 70:  # High IVR - good for selling
                    ivr_score = 90
                elif scenario.ivr >= 50:
                    ivr_score = 70
                elif scenario.ivr >= 30:
                    ivr_score = 50
                else:
                    ivr_score = 30
            
            # Calculate probability score (0-100 scale) - NEW MAJOR COMPONENT
            prob_score = scenario.probability_of_profit * 100  # Convert to 0-100 scale
            
            # Calculate ROI score (0-100 scale) - only matters after probability threshold
            # Since we now filter out options with < 50% probability, ROI becomes secondary
            roi_score = min(100, max(0, scenario.roi * 5))  # Reduced multiplier (1% ROI = 5 points)
            
            # Strategy balance factor - give slight bonus to puts to balance systematic bias
            strategy_bonus = 0
            if scenario.strategy == 'cash_secured_put':
                strategy_bonus = 3  # 3-point bonus to help balance the scoring
            
            # Weighted average with probability as major component
            overall = (
                ivr_score * self.weights['ivr'] +
                scenario.greeks_score * self.weights['greeks'] +
                scenario.technical_score * self.weights['technical'] +
                scenario.liquidity_score * self.weights['liquidity'] +
                scenario.resistance_score * self.weights['resistance'] +
                prob_score * self.weights['probability'] +  # NEW: Probability as major component
                roi_score * self.weights['roi'] +
                strategy_bonus  # Add strategy balance bonus
            )
            
            return round(overall, 2)
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 50.0
    
    def _calculate_trading_metrics(self, scenario: OptionScenario, stock_data: Dict[str, Any]):
        """Calculate trading metrics for the scenario."""
        try:
            # Max profit (premium received)
            scenario.max_profit = scenario.option_price
            if scenario.option_price == 0:
                logger.warning(f"Option price is 0 for {scenario.option_ticker}: {scenario.option_price}")
            
            # CORRECTED Max loss calculation
            if scenario.strategy == 'covered_call':
                # Covered call: Max loss occurs if stock goes to $0
                # You lose the stock value but keep the premium
                # Max loss = current_price - option_price (if stock goes to $0)
                scenario.max_loss = max(0, scenario.current_price - scenario.option_price)
                scenario.breakeven = scenario.strike + scenario.option_price  # Stock can be called away
            elif scenario.strategy == 'cash_secured_put':
                # Cash secured put: Max loss occurs if stock goes to $0
                # You're forced to buy at strike, but keep the premium
                # Max loss = strike - option_price (if stock goes to $0)
                scenario.max_loss = max(0, scenario.strike - scenario.option_price)
                scenario.breakeven = scenario.strike - scenario.option_price  # Stock can be put to you
            else:
                # Fallback for other strategies
                scenario.max_loss = 0
                scenario.breakeven = 0
            
            # Calculate Return on Investment (ROI)
            if scenario.strategy == 'covered_call':
                # ROI = Premium / Current Stock Price
                scenario.roi = (scenario.option_price / scenario.current_price) * 100 if scenario.current_price > 0 else 0
            elif scenario.strategy == 'cash_secured_put':
                # ROI = Premium / Strike Price (secured cash)
                scenario.roi = (scenario.option_price / scenario.strike) * 100 if scenario.strike > 0 else 0
            else:
                scenario.roi = 0
            
            # IMPROVED Probability of profit calculation using Black-Scholes approximation
            import math
            
            # Use Delta for a more accurate probability estimate
            delta = abs(scenario.delta) if scenario.delta is not None else 0.5
            
            if scenario.strategy == 'covered_call':
                # For covered calls, profit if stock stays below strike
                # Use normal distribution approximation based on Delta
                # Delta ≈ N(d1), so probability of finishing below strike ≈ 1 - Delta
                scenario.probability_of_profit = max(0.05, min(0.95, 1 - delta))
            elif scenario.strategy == 'cash_secured_put':
                # For cash-secured puts, profit if stock stays above strike
                # Similar logic but for puts
                scenario.probability_of_profit = max(0.05, min(0.95, 1 - delta))
            else:
                # Fallback to Delta-based calculation
                scenario.probability_of_profit = max(0.05, min(0.95, delta))
            
            # Adjust for time to expiration (more time = more uncertainty)
            dte = scenario.days_to_expiration
            if dte > 45:
                scenario.probability_of_profit *= 0.92  # Reduce more for longer time
            elif dte > 30:
                scenario.probability_of_profit *= 0.95  # Slightly reduce for longer time
            elif dte < 7:
                scenario.probability_of_profit *= 1.03  # Slightly increase for shorter time
            
            # Adjust for implied volatility (higher IV = more uncertainty)
            iv = scenario.implied_volatility if scenario.implied_volatility is not None else 0.3
            if iv > 0.8:  # Extremely high IV
                scenario.probability_of_profit *= 0.85
            elif iv > 0.6:  # Very high IV
                scenario.probability_of_profit *= 0.90
            elif iv > 0.4:  # High IV
                scenario.probability_of_profit *= 0.95
            elif iv < 0.2:  # Low IV
                scenario.probability_of_profit *= 1.05
            
            # Adjust for strike distance (closer to money = higher risk)
            if scenario.strategy == 'covered_call':
                distance_pct = (scenario.strike - scenario.current_price) / scenario.current_price
                if distance_pct < 0.05:  # Very close to current price
                    scenario.probability_of_profit *= 0.90
                elif distance_pct < 0.10:  # Close to current price
                    scenario.probability_of_profit *= 0.95
            elif scenario.strategy == 'cash_secured_put':
                distance_pct = (scenario.current_price - scenario.strike) / scenario.current_price
                if distance_pct < 0.05:  # Very close to current price
                    scenario.probability_of_profit *= 0.90
                elif distance_pct < 0.10:  # Close to current price
                    scenario.probability_of_profit *= 0.95
            
            # Technical indicator adjustments (optional enhancement)
            # Note: Technical indicators are already incorporated in technical_score
            # This is an additional refinement to probability calculation
            try:
                # Get technical data from stock_data (if available)
                rsi = stock_data.get('rsi', 50)
                macd = stock_data.get('macd', {})
                macd_line = macd.get('macd', 0)
                signal_line = macd.get('signal', 0)
                bb = stock_data.get('bollinger_bands', {})
                bb_position = bb.get('position', 0.5)
                
                # RSI-based probability adjustment
                if scenario.strategy == 'covered_call':
                    if rsi > 70:  # Overbought - good for call selling
                        scenario.probability_of_profit *= 1.05
                    elif rsi < 30:  # Oversold - bad for call selling
                        scenario.probability_of_profit *= 0.95
                elif scenario.strategy == 'cash_secured_put':
                    if rsi < 30:  # Oversold - good for put selling
                        scenario.probability_of_profit *= 1.05
                    elif rsi > 70:  # Overbought - bad for put selling
                        scenario.probability_of_profit *= 0.95
                
                # MACD-based probability adjustment
                is_bullish = macd_line > signal_line
                if scenario.strategy == 'covered_call' and not is_bullish:
                    scenario.probability_of_profit *= 1.03  # Bearish MACD good for call selling
                elif scenario.strategy == 'cash_secured_put' and is_bullish:
                    scenario.probability_of_profit *= 1.03  # Bullish MACD good for put selling
                
                # Bollinger Bands position adjustment
                if scenario.strategy == 'covered_call' and bb_position > 0.8:
                    scenario.probability_of_profit *= 1.02  # Near upper band good for call selling
                elif scenario.strategy == 'cash_secured_put' and bb_position < 0.2:
                    scenario.probability_of_profit *= 1.02  # Near lower band good for put selling
                    
            except Exception as e:
                logger.debug(f"Technical indicator probability adjustment failed: {e}")
            
            # Final bounds check
            scenario.probability_of_profit = max(0.05, min(0.95, scenario.probability_of_profit))
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            scenario.max_profit = 0
            scenario.max_loss = 0
            scenario.breakeven = 0
            scenario.probability_of_profit = 0.5
            scenario.roi = 0

class ScenarioComparator:
    """Compares and ranks option scenarios."""
    
    def __init__(self):
        self.analyzer = OptionsAnalyzer()
    
    def analyze_scenarios(self, ticker_data: Dict[str, Any]) -> List[OptionScenario]:
        """Analyze all option scenarios for a ticker."""
        try:
            scenarios = []
            stock_data = ticker_data.get('stock_data', {})
            options_data = ticker_data.get('options_data', [])
            
            if not stock_data or not options_data:
                logger.warning(f"Insufficient data for {ticker_data.get('ticker', 'unknown')}")
                return scenarios
            
            for option in options_data:
                scenario = self.analyzer.analyze_option_scenario(option, stock_data)
                if scenario:
                    scenarios.append(scenario)
            
            # Sort by overall score (descending)
            scenarios.sort(key=lambda x: x.overall_score, reverse=True)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error analyzing scenarios: {e}")
            return []
    
    def get_top_scenarios(self, scenarios: List[OptionScenario], 
                         limit: int = 10) -> List[OptionScenario]:
        """Get top N scenarios based on overall score."""
        return scenarios[:limit]
    
    def filter_by_criteria(self, scenarios: List[OptionScenario],
                          min_score: float = 60.0,
                          min_probability: float = 0.5,
                          max_days_to_exp: int = 60) -> List[OptionScenario]:
        """Filter scenarios by specific criteria."""
        filtered = []
        
        for scenario in scenarios:
            if (scenario.overall_score >= min_score and
                scenario.probability_of_profit >= min_probability and
                scenario.days_to_expiration <= max_days_to_exp):
                filtered.append(scenario)
        
        return filtered
    
    def group_by_strategy(self, scenarios: List[OptionScenario]) -> Dict[str, List[OptionScenario]]:
        """Group scenarios by strategy type."""
        grouped = {}
        
        for scenario in scenarios:
            strategy = scenario.strategy
            if strategy not in grouped:
                grouped[strategy] = []
            grouped[strategy].append(scenario)
        
        return grouped
