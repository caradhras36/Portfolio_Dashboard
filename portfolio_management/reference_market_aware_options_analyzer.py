#!/usr/bin/env python3
"""
Market-Aware Options Analyzer - Works during market hours (full Greeks) and after hours (limited data)
"""

from polygon import RESTClient
from config import get_polygon_api_key
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Optional
import pandas as pd

class MarketAwareOptionsAnalyzer:
    """Options analyzer that adapts to market hours"""
    
    def __init__(self):
        self.client = RESTClient(get_polygon_api_key())
        self.et_tz = pytz.timezone('US/Eastern')
        
    def is_market_open(self) -> bool:
        """Check if US stock market is open (9:30 AM - 4:00 PM ET, Mon-Fri)"""
        now = datetime.now(self.et_tz)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Check if it's within market hours
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def get_market_status(self) -> str:
        """Get current market status"""
        if self.is_market_open():
            return "OPEN - Full Greeks data available"
        else:
            return "CLOSED - Limited Greeks data (using previous close)"
    
    def find_monthly_expirations(self, ticker: str, limit: int = 50, 
                               min_dte: int = 15, max_dte: int = 90) -> List[str]:
        """Find monthly expiration dates (3rd Friday of each month) within DTE range"""
        from datetime import datetime, timedelta
        import calendar
        
        # Calculate date range
        today = datetime.now()
        min_date = today + timedelta(days=min_dte)
        max_date = today + timedelta(days=max_dte)
        
        print(f"📅 Searching monthly options: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({min_dte}-{max_dte} DTE)")
        
        # For now, return a simple fallback - calculate next few 3rd Fridays manually
        # This avoids the problematic API call that hangs
        monthly_expirations = []
        current_date = min_date
        
        while current_date <= max_date and len(monthly_expirations) < 3:
            if self._is_third_friday(current_date):
                monthly_expirations.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        print(f"  📅 Found {len(monthly_expirations)} monthly expirations: {monthly_expirations}")
        return monthly_expirations
    
    def _is_third_friday(self, date: datetime) -> bool:
        """Check if a date is the 3rd Friday of the month"""
        # Get the first day of the month
        first_day = date.replace(day=1)
        
        # Find the first Friday of the month
        first_friday = first_day
        while first_friday.weekday() != 4:  # Friday is weekday 4
            first_friday += timedelta(days=1)
        
        # The 3rd Friday is 14 days after the first Friday
        third_friday = first_friday + timedelta(days=14)
        
        return date.date() == third_friday.date()
    
    def fetch_options_data(self, ticker: str, target_expiration: str = None, 
                          strike_range_pct: float = 0.15, max_expirations: int = 2) -> List[Dict]:
        """Fetch real options data from Polygon.io - Simplified approach"""
        
        print(f"\n🔍 Fetching options data for {ticker}...")
        print(f"📊 Market Status: {self.get_market_status()}")
        
        # Debug: Check API key
        from config import get_polygon_api_key
        try:
            api_key = get_polygon_api_key()
            print(f"🔑 API Key configured: {'*' * (len(api_key) - 4) + api_key[-4:] if api_key else 'NOT SET'}")
        except Exception as e:
            print(f"❌ API Key error: {e}")
            return []
        
        try:
            print(f"📡 Fetching options contracts for {ticker}...")
            
            # Simple, direct approach - get contracts and let the API handle pagination
            contracts = list(self.client.list_options_contracts(
                underlying_ticker=ticker,
                limit=100,  # Reasonable limit
                order="asc",
                sort="expiration_date"
            ))
            
            print(f"✅ Successfully fetched {len(contracts)} contracts")
            
            if not contracts:
                print(f"❌ No contracts found for {ticker}")
                return []
            
            # If no target expiration specified, use the next available monthly expirations
            if not target_expiration:
                monthly_exps = self.find_monthly_expirations(ticker)
                if monthly_exps:
                    # Use up to max_expirations monthly expirations
                    target_expirations = monthly_exps[:max_expirations]
                    print(f"📅 Using monthly expirations: {target_expirations}")
                else:
                    # Use the first available expiration as fallback
                    target_expirations = [contracts[0].expiration_date]
                    print(f"📅 Using first available expiration: {target_expirations[0]}")
            else:
                target_expirations = [target_expiration]
            
            # Filter contracts by target expirations
            filtered_contracts = [c for c in contracts if c.expiration_date in target_expirations]
            print(f"📋 Found {len(filtered_contracts)} contracts for expirations {target_expirations}")
            
            if not filtered_contracts:
                print(f"❌ No contracts found for expirations {target_expirations}")
                return []
            
            contracts = filtered_contracts
            
        except Exception as e:
            print(f"❌ Error fetching options data for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        
        # Get current stock price to determine strike range
        try:
            # Try to get current stock price (this might be delayed or previous close)
            stock_price = self.get_stock_price(ticker)
            if stock_price:
                lower_strike = stock_price * (1 - strike_range_pct)
                upper_strike = stock_price * (1 + strike_range_pct)
                print(f"💰 Current {ticker} price: ${stock_price:.2f}")
                print(f"🎯 Strike range: ${lower_strike:.2f} - ${upper_strike:.2f}")
            else:
                # Use default range if we can't get stock price
                lower_strike = 100
                upper_strike = 500
                print(f"⚠️  Using default strike range: ${lower_strike:.2f} - ${upper_strike:.2f}")
        except:
            lower_strike = 100
            upper_strike = 500
        
        # Filter contracts within strike range
        relevant_contracts = []
        for contract in contracts:
            if lower_strike <= contract.strike_price <= upper_strike:
                relevant_contracts.append(contract)
        
        print(f"🎯 Found {len(relevant_contracts)} contracts within strike range")
        
        # Get snapshot data for each contract with retry mechanism
        options_data = []
        for i, contract in enumerate(relevant_contracts):  # Process all valid contracts
            print(f"  📊 Fetching data for {contract.ticker}...", end="")
            
            # Retry mechanism for snapshot fetching with timeout
            snapshot = None
            for attempt in range(3):  # Try 3 times
                try:
                    # Direct API call without timeout
                    snapshot = self.client.get_snapshot_option(
                        contract.underlying_ticker,
                        contract.ticker
                    )
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < 2:  # Not the last attempt
                        print(f" (retry {attempt + 1})", end="")
                        import time
                        time.sleep(0.5)  # Brief delay before retry
                    else:
                        print(f" ❌ (Failed after 3 attempts: {str(e)[:30]})")
                        continue
            
            if snapshot is None:
                continue  # Skip this contract if we couldn't get snapshot
            
            option_data = self.extract_option_data(contract, snapshot)
            options_data.append(option_data)
            
            if option_data['has_greeks']:
                print(" ✅ (Greeks available)")
            elif option_data['has_price']:
                print(" ⚠️  (Price only)")
            else:
                print(" ❌ (No data)")
        
        return options_data
    
    def get_stock_price(self, ticker: str) -> Optional[float]:
        """Get current stock price (delayed or previous close)"""
        try:
            # Try to get previous close
            from data_fetchers import StockDataFetcher
            stock_fetcher = StockDataFetcher()
            stock_data = stock_fetcher.get_stock_data(ticker)
            if stock_data is not None and not stock_data.empty:
                return float(stock_data['Close'].iloc[-1])
        except:
            pass
        return None
    
    def get_stock_data(self, ticker: str):
        """Get stock data for technical analysis"""
        try:
            from data_fetchers import StockDataFetcher
            stock_fetcher = StockDataFetcher()
            return stock_fetcher.get_stock_data(ticker)
        except Exception as e:
            print(f"⚠️  Error getting stock data for {ticker}: {e}")
            return None
    
    def extract_option_data(self, contract, snapshot) -> Dict:
        """Extract all available data from option snapshot"""
        
        option_data = {
            'ticker': contract.ticker,
            'underlying_ticker': contract.underlying_ticker,
            'type': contract.contract_type,
            'strike': contract.strike_price,
            'expiration': contract.expiration_date,
            'exercise_style': contract.exercise_style,
            
            # Price data
            'close_price': None,
            'bid': None,
            'ask': None,
            'last_trade_price': None,
            'volume': None,
            'open_interest': None,
            'vwap': None,
            
            # Greeks
            'delta': None,
            'gamma': None,
            'theta': None,
            'vega': None,
            'implied_volatility': None,
            
            # Other data
            'break_even_price': None,
            'underlying_price': None,
            'has_greeks': False,
            'has_price': False,
            'has_complete_data': False
        }
        
        # Handle OptionContractSnapshot object
        try:
            # The snapshot is likely an OptionContractSnapshot object
            # Extract data using attribute access
            results = {
                'day': snapshot.day if hasattr(snapshot, 'day') and snapshot.day else None,
                'last_quote': snapshot.last_quote if hasattr(snapshot, 'last_quote') and snapshot.last_quote else None,
                'last_trade': snapshot.last_trade if hasattr(snapshot, 'last_trade') and snapshot.last_trade else None,
                'greeks': snapshot.greeks if hasattr(snapshot, 'greeks') and snapshot.greeks else None,
                'implied_volatility': getattr(snapshot, 'implied_volatility', None),
                'break_even_price': getattr(snapshot, 'break_even_price', None),
                'open_interest': getattr(snapshot, 'open_interest', None),
                'underlying_asset': snapshot.underlying_asset if hasattr(snapshot, 'underlying_asset') and snapshot.underlying_asset else None
            }
        except Exception as e:
            print(f"⚠️  Error processing snapshot: {e}")
            # Fallback to empty results
            results = {}
        
        # Extract price data from day section
        if results.get('day'):
            day_data = results['day']
            option_data['close_price'] = getattr(day_data, 'close', None)
            option_data['volume'] = getattr(day_data, 'volume', None)
            option_data['vwap'] = getattr(day_data, 'vwap', None)
        
        # Extract bid/ask from last_quote section
        if results.get('last_quote'):
            quote_data = results['last_quote']
            option_data['bid'] = getattr(quote_data, 'bid', None)
            option_data['ask'] = getattr(quote_data, 'ask', None)
        
        # Extract last trade price
        if results.get('last_trade'):
            trade_data = results['last_trade']
            option_data['last_trade_price'] = getattr(trade_data, 'price', None)
        
        # Extract Greeks
        if results.get('greeks'):
            greeks_data = results['greeks']
            option_data['delta'] = getattr(greeks_data, 'delta', None)
            option_data['gamma'] = getattr(greeks_data, 'gamma', None)
            option_data['theta'] = getattr(greeks_data, 'theta', None)
            option_data['vega'] = getattr(greeks_data, 'vega', None)
        
        # Extract other data
        option_data['implied_volatility'] = results.get('implied_volatility')
        option_data['break_even_price'] = results.get('break_even_price')
        option_data['open_interest'] = results.get('open_interest')
        
        # Extract underlying price
        if results.get('underlying_asset'):
            underlying_data = results['underlying_asset']
            option_data['underlying_price'] = getattr(underlying_data, 'price', None)
        
        # Determine data completeness
        option_data['has_price'] = any([
            option_data['close_price'],
            option_data['last_trade_price'],
            option_data['bid']
        ])
        
        option_data['has_greeks'] = all([
            option_data['delta'] is not None,
            option_data['gamma'] is not None,
            option_data['theta'] is not None,
            option_data['vega'] is not None
        ])
        
        option_data['has_complete_data'] = (
            option_data['has_price'] and 
            option_data['has_greeks'] and 
            option_data['implied_volatility'] is not None
        )
        
        return option_data
    
    def display_results(self, options_data: List[Dict]):
        """Display options analysis results"""
        
        if not options_data:
            print("❌ No options data to display")
            return
        
        # Separate calls and puts
        calls = [opt for opt in options_data if opt['type'] == 'call']
        puts = [opt for opt in options_data if opt['type'] == 'put']
        
        # Count data completeness
        complete_calls = sum(1 for opt in calls if opt['has_complete_data'])
        complete_puts = sum(1 for opt in puts if opt['has_complete_data'])
        greeks_calls = sum(1 for opt in calls if opt['has_greeks'])
        greeks_puts = sum(1 for opt in puts if opt['has_greeks'])
        
        print(f"\n{'='*120}")
        print(f"OPTIONS ANALYSIS RESULTS - {self.get_market_status()}")
        print(f"{'='*120}")
        print(f"📊 Data Summary:")
        print(f"   Calls: {len(calls)} total, {complete_calls} complete, {greeks_calls} with Greeks")
        print(f"   Puts:  {len(puts)} total, {complete_puts} complete, {greeks_puts} with Greeks")
        
        # Display calls
        if calls:
            print(f"\n📞 CALL OPTIONS:")
            self._display_options_table(calls)
        
        # Display puts
        if puts:
            print(f"\n📉 PUT OPTIONS:")
            self._display_options_table(puts)
        
        # Summary
        total_complete = complete_calls + complete_puts
        total_greeks = greeks_calls + greeks_puts
        
        print(f"\n🎯 SUMMARY:")
        if total_complete > 0:
            print(f"   ✅ {total_complete} options with complete data (price + Greeks + IV)")
        if total_greeks > 0:
            print(f"   📊 {total_greeks} options with Greeks data")
        
        if self.is_market_open():
            print(f"   🕐 Market is OPEN - Full data available")
        else:
            print(f"   🕐 Market is CLOSED - Limited data (previous close prices)")
    
    def _display_options_table(self, options: List[Dict]):
        """Display options in table format"""
        
        print(f"{'Strike':<8} {'Price':<8} {'Bid':<6} {'Ask':<6} {'Δ':<6} {'Γ':<6} {'θ':<6} {'ν':<6} {'IV%':<6} {'Vol':<6} {'OI':<8} {'Status':<10}")
        print(f"{'-'*90}")
        
        for opt in sorted(options, key=lambda x: x['strike']):
            price = opt['close_price'] or opt['last_trade_price'] or 0
            bid = opt['bid'] or 0
            ask = opt['ask'] or 0
            delta = opt['delta'] or 0
            gamma = opt['gamma'] or 0
            theta = opt['theta'] or 0
            vega = opt['vega'] or 0
            iv = (opt['implied_volatility'] or 0) * 100
            volume = opt['volume'] or 0
            oi = opt['open_interest'] or 0
            
            # Status indicator
            if opt['has_complete_data']:
                status = "Complete"
            elif opt['has_greeks']:
                status = "Greeks"
            elif opt['has_price']:
                status = "Price"
            else:
                status = "Limited"
            
            print(f"${opt['strike']:<7.0f} ${price:<7.2f} ${bid:<5.2f} ${ask:<5.2f} "
                  f"{delta:<5.3f} {gamma:<5.3f} {theta:<5.2f} {vega:<5.2f} "
                  f"{iv:<5.1f} {volume:<6} {oi:<8} {status:<10}")

def main():
    """Main function to demonstrate the market-aware analyzer"""
    
    print("🚀 Market-Aware Options Analyzer")
    print("=" * 50)
    
    analyzer = MarketAwareOptionsAnalyzer()
    
    # Test with AAPL
    ticker = "AAPL"
    print(f"\nAnalyzing {ticker}...")
    
    options_data = analyzer.fetch_options_data(ticker)
    
    if options_data:
        analyzer.display_results(options_data)
        
        # Show what we can do with this data
        print(f"\n🔧 NEXT STEPS:")
        print(f"   1. Integrate with main options analysis system")
        print(f"   2. Run covered calls and cash secured puts analysis")
        print(f"   3. Generate charts and recommendations")
        print(f"   4. Post to Twitter via Zapier")
    else:
        print(f"❌ Failed to get options data for {ticker}")

if __name__ == "__main__":
    main()
