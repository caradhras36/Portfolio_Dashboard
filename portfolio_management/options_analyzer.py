"""
Options Analysis Engine
Provides Greeks calculation and options analysis capabilities
"""

import math
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptionData:
    """Option data structure"""
    ticker: str
    strike_price: float
    current_price: float
    time_to_expiration: float  # in days
    risk_free_rate: float = 0.05  # 5% default
    implied_volatility: Optional[float] = None

class OptionsAnalyzer:
    """Options analysis and Greeks calculation"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # Default 5%
    
    def calculate_greeks(self, option_data: Dict) -> Dict[str, float]:
        """Calculate option Greeks"""
        try:
            # Extract data from option_data dict
            ticker = option_data.get('ticker', 'UNKNOWN')
            strike = float(option_data.get('strike_price', 0))
            current_price = float(option_data.get('current_price', 0))
            time_to_exp = float(option_data.get('time_to_expiration', 30))
            iv = float(option_data.get('implied_volatility', 0.3))
            option_type = option_data.get('contract_type', 'call')
            
            if strike <= 0 or current_price <= 0 or time_to_exp <= 0:
                logger.warning(f"Invalid option data for {ticker}: S={strike}, P={current_price}, T={time_to_exp}")
                return self._get_zero_greeks()
            
            # Convert time to years
            T = time_to_exp / 365.0
            
            # Calculate d1 and d2
            d1 = self._calculate_d1(current_price, strike, T, self.risk_free_rate, iv)
            d2 = d1 - iv * math.sqrt(T)
            
            # Calculate Greeks
            delta = self._calculate_delta(d1, option_type)
            gamma = self._calculate_gamma(current_price, T, iv, d1)
            theta = self._calculate_theta(current_price, strike, T, self.risk_free_rate, iv, d1, d2, option_type)
            vega = self._calculate_vega(current_price, T, iv, d1)
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'implied_volatility': iv
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks for {option_data.get('ticker', 'UNKNOWN')}: {e}")
            return self._get_zero_greeks()
    
    def _calculate_d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 for Black-Scholes formula"""
        if sigma <= 0 or T <= 0:
            return 0
        return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    
    def _calculate_delta(self, d1: float, option_type: str) -> float:
        """Calculate option delta"""
        if option_type.lower() == 'call':
            return self._normal_cdf(d1)
        else:  # put
            return self._normal_cdf(d1) - 1
    
    def _calculate_gamma(self, S: float, T: float, sigma: float, d1: float) -> float:
        """Calculate option gamma"""
        if S <= 0 or T <= 0 or sigma <= 0:
            return 0
        return self._normal_pdf(d1) / (S * sigma * math.sqrt(T))
    
    def _calculate_theta(self, S: float, K: float, T: float, r: float, sigma: float, d1: float, d2: float, option_type: str) -> float:
        """Calculate option theta"""
        if T <= 0:
            return 0
            
        term1 = -S * self._normal_pdf(d1) * sigma / (2 * math.sqrt(T))
        
        if option_type.lower() == 'call':
            term2 = -r * K * math.exp(-r * T) * self._normal_cdf(d2)
        else:  # put
            term2 = r * K * math.exp(-r * T) * self._normal_cdf(-d2)
        
        return (term1 + term2) / 365.0  # Convert to daily theta
    
    def _calculate_vega(self, S: float, T: float, sigma: float, d1: float) -> float:
        """Calculate option vega"""
        if S <= 0 or T <= 0:
            return 0
        return S * self._normal_pdf(d1) * math.sqrt(T) / 100.0  # Vega per 1% change in IV
    
    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function of standard normal distribution"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Probability density function of standard normal distribution"""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    
    def _get_zero_greeks(self) -> Dict[str, float]:
        """Return zero Greeks as fallback"""
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'implied_volatility': 0.0
        }
    
    def calculate_option_price(self, option_data: OptionData) -> float:
        """Calculate theoretical option price using Black-Scholes"""
        try:
            S = option_data.current_price
            K = option_data.strike_price
            T = option_data.time_to_expiration / 365.0
            r = option_data.risk_free_rate
            sigma = option_data.implied_volatility or 0.3
            
            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
                return 0.0
            
            d1 = self._calculate_d1(S, K, T, r, sigma)
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_data.ticker.endswith('C') or 'call' in option_data.ticker.lower():
                # Call option
                price = S * self._normal_cdf(d1) - K * math.exp(-r * T) * self._normal_cdf(d2)
            else:
                # Put option
                price = K * math.exp(-r * T) * self._normal_cdf(-d2) - S * self._normal_cdf(-d1)
            
            return max(0.0, price)  # Option price cannot be negative
            
        except Exception as e:
            logger.error(f"Error calculating option price: {e}")
            return 0.0

class MarketAwareOptionsAnalyzer(OptionsAnalyzer):
    """Market-aware options analyzer with real-time data integration"""
    
    def __init__(self):
        super().__init__()
        self.price_cache = {}
        self.cache_duration = 300  # 5 minutes
    
    async def get_current_price(self, ticker: str) -> float:
        """Get current market price for ticker"""
        # For now, return a mock price
        # In production, this would integrate with real market data APIs
        mock_prices = {
            'AAPL': 175.50,
            'TSLA': 245.30,
            'MSFT': 380.20,
            'GOOGL': 142.80,
            'AMZN': 155.40,
            'NVDA': 875.60,
            'META': 485.30,
            'NFLX': 485.90,
            'AMD': 135.70,
            'SOFI': 8.45
        }
        
        # Clean ticker (remove option suffixes)
        clean_ticker = ticker.split(' ')[0].split('C')[0].split('P')[0]
        
        return mock_prices.get(clean_ticker, 100.0)  # Default price
    
    async def fetch_options_data(self, ticker: str, target_expiration: Optional[str] = None) -> List[Dict]:
        """Fetch options data for a ticker"""
        # Mock options data - in production, this would call real options APIs
        current_price = await self.get_current_price(ticker)
        
        # Generate mock options around current price
        strike_prices = []
        base_strike = round(current_price / 5) * 5  # Round to nearest $5
        
        for i in range(-3, 4):
            strike_prices.append(base_strike + (i * 5))
        
        options_data = []
        for strike in strike_prices:
            for option_type in ['call', 'put']:
                options_data.append({
                    'ticker': ticker,
                    'strike_price': strike,
                    'contract_type': option_type,
                    'current_price': max(0.01, abs(current_price - strike) * 0.1),
                    'implied_volatility': 0.25 + (abs(current_price - strike) / current_price) * 0.2,
                    'time_to_expiration': 30.0,  # 30 days
                    'volume': 100,
                    'open_interest': 500
                })
        
        return options_data
    
    def analyze_portfolio_options(self, positions: List[Dict]) -> Dict:
        """Analyze options positions in portfolio"""
        options_positions = [p for p in positions if p.get('position_type') in ['call', 'put']]
        
        if not options_positions:
            return {'total_options': 0, 'analysis': 'No options positions found'}
        
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        
        for position in options_positions:
            greeks = self.calculate_greeks(position)
            quantity = position.get('quantity', 0)
            
            total_delta += greeks['delta'] * quantity
            total_gamma += greeks['gamma'] * quantity
            total_theta += greeks['theta'] * quantity
            total_vega += greeks['vega'] * quantity
        
        return {
            'total_options': len(options_positions),
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'total_vega': total_vega,
            'avg_iv': sum(p.get('implied_volatility', 0) for p in options_positions) / len(options_positions)
        }

# Example usage
if __name__ == "__main__":
    analyzer = OptionsAnalyzer()
    
    # Test option data
    option_data = {
        'ticker': 'AAPL',
        'strike_price': 175.0,
        'current_price': 175.50,
        'time_to_expiration': 30,
        'implied_volatility': 0.25,
        'contract_type': 'call'
    }
    
    greeks = analyzer.calculate_greeks(option_data)
    print(f"Greeks for AAPL 175C: {greeks}")
