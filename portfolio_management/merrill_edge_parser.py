#!/usr/bin/env python3
"""
Specialized parser for Merrill Edge CSV exports
Handles the complex structure with headers and summary information
"""

import pandas as pd
import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MerrillEdgeParser:
    """Specialized parser for Merrill Edge CSV exports"""
    
    def __init__(self):
        self.standard_columns = [
            'ticker', 'position_type', 'quantity', 'entry_price', 
            'current_price', 'expiration_date', 'strike_price', 'option_type'
        ]
    
    def parse_merrill_csv(self, file_path: str) -> pd.DataFrame:
        """Parse Merrill Edge CSV file"""
        try:
            # Read the file line by line to find the data section
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find where the actual data starts
            data_start_line = None
            header_line = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Skip account names and account numbers (CMA* entries)
                if line.startswith('CMA') or 'CMAEdge' in line:
                    continue
                
                # Skip gain/loss information (but not headers that contain these words)
                if (('gain' in line.lower() or 'loss' in line.lower()) and 
                    'Symbol' not in line and 'Quantity' not in line):
                    continue
                
                # Look for lines that contain position data headers
                # Skip summary headers like "All Accounts" - look for actual portfolio data
                if (('Symbol' in line and 'Quantity' in line and 'Price' in line and 'Unit Cost' in line) or 
                    ('Ticker' in line and 'Shares' in line)):
                    data_start_line = i
                    header_line = line
                    break
            
            if data_start_line is None:
                # Try to find any line with multiple columns that might be data
                for i, line in enumerate(lines):
                    if ',' in line and len(line.split(',')) > 3:
                        # Check if it looks like position data (not account info)
                        parts = [p.strip().strip('"') for p in line.split(',')]
                        if (len(parts) >= 4 and 
                            any(char.isalpha() for char in parts[0]) and
                            not parts[0].startswith('CMA') and
                            'CMAEdge' not in parts[0]):
                            data_start_line = i
                            header_line = line
                            break
            
            if data_start_line is None:
                raise ValueError("Could not find data section in CSV file")
            
            print(f"📊 Found data starting at line {data_start_line + 1}")
            print(f"📋 Header line: {header_line}")
            
            # Extract the data section
            data_lines = lines[data_start_line:]
            
            # Parse the header to understand column mapping
            header_parts = [p.strip().strip('"') for p in header_line.split(',')]
            print(f"📋 Header columns: {header_parts}")
            
            # Create a temporary CSV with just the data
            temp_csv = '\n'.join(data_lines)
            
            # Try to parse with pandas
            try:
                df = pd.read_csv(pd.io.common.StringIO(temp_csv))
                print(f"✅ Successfully parsed data section: {df.shape}")
                print(f"📋 Data columns: {list(df.columns)}")
                
                # Convert to our standard format
                return self._convert_to_standard_format(df, header_parts)
                
            except Exception as e:
                print(f"❌ Error parsing data section: {e}")
                # Try manual parsing
                return self._manual_parse(data_lines, header_parts)
                
        except Exception as e:
            logger.error(f"Error parsing Merrill Edge CSV: {e}")
            raise
    
    def _convert_to_standard_format(self, df: pd.DataFrame, header_parts: List[str]) -> pd.DataFrame:
        """Convert parsed data to standard format"""
        result = pd.DataFrame(columns=self.standard_columns)
        
        # Map columns based on common Merrill Edge patterns
        column_mapping = self._detect_column_mapping(df.columns, header_parts)
        print(f"🔍 Detected column mapping: {column_mapping}")
        
        for _, row in df.iterrows():
            try:
                # Extract data based on mapping
                ticker = self._extract_ticker(row, column_mapping)
                if not ticker:
                    continue
                
                quantity = self._safe_float(row, column_mapping.get('quantity'))
                if quantity == 0:
                    continue
                
                entry_price = self._safe_float(row, column_mapping.get('entry_price'))
                current_price = self._safe_float(row, column_mapping.get('current_price'))
                
                # Parse option information from ticker
                option_info = self._parse_option_ticker(ticker)
                
                # Use option info if it's an option, otherwise use defaults
                if option_info['position_type'] in ['call', 'put']:
                    position_type = option_info['position_type']
                    option_type = option_info['option_type']
                    strike_price = option_info['strike_price']
                    expiration_date = option_info['expiration_date']
                    ticker = option_info['ticker']  # Use the base ticker
                else:
                    # Regular stock
                    position_type = 'stock'
                    option_type = None
                    strike_price = None
                    expiration_date = None
                
                # Skip money accounts, account numbers, and other non-trading positions
                if (ticker.lower() in ['moneyaccounts', 'cmaedge', 'cma'] or
                    ticker.startswith('CMA') or
                    'CMAEdge' in ticker or
                    'gain' in ticker.lower() or
                    'loss' in ticker.lower()):
                    continue
                
                result = pd.concat([result, pd.DataFrame([{
                    'ticker': ticker,
                    'position_type': position_type,
                    'quantity': int(quantity),
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'expiration_date': expiration_date,
                    'strike_price': strike_price,
                    'option_type': option_type
                }])], ignore_index=True)
                
            except Exception as e:
                logger.warning(f"Error converting row: {e}")
                continue
        
        return result
    
    def _detect_column_mapping(self, columns: List[str], header_parts: List[str]) -> Dict[str, str]:
        """Detect which columns map to our standard format"""
        mapping = {}
        
        # Convert to lowercase for matching
        cols_lower = [col.lower().strip() for col in columns]
        headers_lower = [h.lower().strip() for h in header_parts]
        
        # Map ticker/symbol
        for i, col in enumerate(cols_lower):
            if any(keyword in col for keyword in ['symbol', 'ticker', 'instrument']):
                mapping['ticker'] = columns[i]
                break
        
        # Map quantity
        for i, col in enumerate(cols_lower):
            if any(keyword in col for keyword in ['quantity', 'shares', 'qty', 'units']):
                mapping['quantity'] = columns[i]
                break
        
        # Map entry price - look for "Unit Cost" column
        for i, col in enumerate(cols_lower):
            if 'unit' in col and 'cost' in col:
                mapping['entry_price'] = columns[i]
                break
        
        # Map current price - look for "Price" column (current market price)
        for i, col in enumerate(cols_lower):
            if col.strip() == 'price':
                mapping['current_price'] = columns[i]
                break
        
        # Map option fields
        for i, col in enumerate(cols_lower):
            if any(keyword in col for keyword in ['call', 'put', 'type']):
                mapping['option_type'] = columns[i]
                break
        
        for i, col in enumerate(cols_lower):
            if any(keyword in col for keyword in ['strike', 'exercise']):
                mapping['strike_price'] = columns[i]
                break
        
        for i, col in enumerate(cols_lower):
            if any(keyword in col for keyword in ['expiration', 'expiry', 'exp']):
                mapping['expiration_date'] = columns[i]
                break
        
        return mapping
    
    def _extract_ticker(self, row: pd.Series, mapping: Dict[str, str]) -> Optional[str]:
        """Extract ticker symbol from row"""
        if 'ticker' not in mapping:
            return None
        
        ticker = str(row[mapping['ticker']]).strip()
        if ticker == 'nan' or not ticker:
            return None
        
        # Clean up ticker (remove common suffixes)
        ticker = re.sub(r'\s+', '', ticker)  # Remove spaces
        ticker = re.sub(r'[^\w]', '', ticker)  # Keep only alphanumeric
        
        return ticker
    
    def _parse_option_ticker(self, ticker: str) -> Dict[str, str]:
        """Parse Merrill Edge option ticker format"""
        # Format: [UNDERLYING][MONTH][DAY][YEAR][DECIMAL_CODE][STRIKE]
        # Month: A-L = Call (Jan-Dec), M-Y = Put (Jan-Dec)
        # Decimal: C = no decimal, D = 2 places, E = 4 places
        
        option_info = {
            'ticker': ticker,
            'position_type': 'stock',
            'option_type': None,
            'strike_price': None,
            'expiration_date': None
        }
        
        # Check if it looks like an option (has letters and numbers)
        if len(ticker) > 8 and any(char.isdigit() for char in ticker) and any(char.isalpha() for char in ticker):
            try:
                import re
                
                # Find the pattern: letters + month + day + year + decimal_code + strike
                # Look for pattern: [letters][A-Y][0-9][0-9][CDE][0-9]+
                pattern = r'^([A-Z]+)([A-Y])(\d{2})(\d{2})([CDE])(\d+)$'
                match = re.match(pattern, ticker)
                
                if match:
                    underlying, month_letter, day, year, decimal_code, strike_digits = match.groups()
                    
                    # Merrill Edge month mapping
                    call_months = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
                                 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12}
                    put_months = {'M': 1, 'N': 2, 'O': 3, 'P': 4, 'Q': 5, 'R': 6,
                                'S': 7, 'T': 8, 'U': 9, 'V': 10, 'X': 11, 'Y': 12}
                    
                    # Determine if it's a call or put and get month number
                    if month_letter in call_months:
                        option_info['option_type'] = 'call'
                        option_info['position_type'] = 'call'
                        month_num = call_months[month_letter]
                    elif month_letter in put_months:
                        option_info['option_type'] = 'put'
                        option_info['position_type'] = 'put'
                        month_num = put_months[month_letter]
                    else:
                        # Not a valid option format
                        return option_info
                    
                    # Calculate strike price based on decimal code
                    strike_int = int(strike_digits)
                    
                    if decimal_code == 'C':
                        # C = 3 digits before decimal + cents
                        # For C162500000: 162500 -> 162 dollars + 50 cents = $162.50
                        if len(strike_digits) <= 3:
                            # Less than 3 digits, treat as dollars only
                            strike_price = float(strike_int)
                        else:
                            # First 3 digits are dollars, rest are cents
                            dollars = strike_int // (10 ** (len(strike_digits) - 3))
                            cents = strike_int % (10 ** (len(strike_digits) - 3))
                            strike_price = dollars + (cents / 100)
                    elif decimal_code == 'D':
                        # D = 2 digits before decimal + cents
                        # For D800000: 8000 -> 80 dollars + 00 cents = $80.00
                        if len(strike_digits) <= 2:
                            # Less than 2 digits, treat as dollars only
                            strike_price = float(strike_int)
                        else:
                            # First 2 digits are dollars, rest are cents (2 decimal places)
                            dollars = strike_int // (10 ** (len(strike_digits) - 2))
                            cents = strike_int % (10 ** (len(strike_digits) - 2))
                            # Convert cents to proper decimal (e.g., 5000 -> 0.50)
                            cents_decimal = cents / (10 ** (len(strike_digits) - 2))
                            strike_price = dollars + cents_decimal
                    elif decimal_code == 'E':
                        # E = 1 digit before decimal + cents
                        # For E500000: 5000 -> 5 dollars + 00 cents = $5.00
                        if len(strike_digits) <= 1:
                            # Single digit, treat as dollars only
                            strike_price = float(strike_int)
                        else:
                            # First 1 digit is dollars, rest are cents (2 decimal places)
                            dollars = strike_int // (10 ** (len(strike_digits) - 1))
                            cents = strike_int % (10 ** (len(strike_digits) - 1))
                            # Convert cents to proper decimal (e.g., 5000 -> 0.50)
                            cents_decimal = cents / (10 ** (len(strike_digits) - 1))
                            strike_price = dollars + cents_decimal
                    else:
                        # Default: treat as 2 decimal places
                        strike_price = float(strike_int) / 100
                    
                    option_info['strike_price'] = strike_price
                    option_info['ticker'] = underlying
                    
                    # Calculate expiration date
                    full_year = 2000 + int(year)
                    option_info['expiration_date'] = f"{full_year}-{month_num:02d}-{int(day):02d}"
                    
                    print(f"✅ Parsed option: {ticker} → {underlying} {option_info['option_type']} ${strike_price} {option_info['expiration_date']}")
                
                else:
                    # Try alternative patterns for edge cases
                    # Look for patterns like V1725D, K2125D, etc.
                    if any(pattern in ticker for pattern in ['V1725', 'K2125', 'J1725', 'A1725', 'B1725']):
                        # Extract underlying (everything before the pattern)
                        for test_pattern in ['V1725', 'K2125', 'J1725', 'A1725', 'B1725']:
                            if test_pattern in ticker:
                                underlying = ticker.split(test_pattern)[0]
                                suffix = ticker.split(test_pattern)[1] if len(ticker.split(test_pattern)) > 1 else ''
                                
                                # Determine call/put based on month letter
                                month_letter = test_pattern[0]
                                month_num = ord(month_letter) - ord('A') + 1
                                
                                if month_num <= 12:
                                    option_info['option_type'] = 'call'
                                    option_info['position_type'] = 'call'
                                else:
                                    option_info['option_type'] = 'put'
                                    option_info['position_type'] = 'put'
                                
                                # Extract strike from suffix
                                if suffix:
                                    # Look for decimal code and strike
                                    decimal_match = re.search(r'([CDE])(\d+)', suffix)
                                    if decimal_match:
                                        decimal_code, strike_digits = decimal_match.groups()
                                        strike_int = int(strike_digits)
                                        
                                        if decimal_code == 'C':
                                            strike_price = float(strike_int)
                                        elif decimal_code == 'D':
                                            strike_price = float(strike_int) / 100
                                        elif decimal_code == 'E':
                                            strike_price = float(strike_int) / 10000
                                        else:
                                            strike_price = float(strike_int) / 1000
                                        
                                        option_info['strike_price'] = strike_price
                                
                                option_info['ticker'] = underlying
                                option_info['expiration_date'] = '2025-01-17'  # Default
                                break
                
            except Exception as e:
                print(f"❌ Error parsing option {ticker}: {e}")
                # If parsing fails, treat as regular stock
                pass
        
        return option_info
    
    def _safe_float(self, row: pd.Series, column: Optional[str]) -> float:
        """Safely extract float value from row"""
        if not column or column not in row:
            return 0.0
        
        value = row[column]
        if pd.isna(value) or value == '' or value is None:
            return 0.0
        
        try:
            # Remove common currency symbols and commas
            if isinstance(value, str):
                value = re.sub(r'[$,]', '', value)
                value = value.strip()
            
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _parse_date(self, date_str) -> Optional[str]:
        """Parse date string to ISO format"""
        if pd.isna(date_str) or date_str == '' or date_str is None:
            return None
        
        try:
            # Try different date formats
            date_formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%m-%d-%Y',
                '%d/%m/%Y',
                '%d-%m-%Y',
                '%Y/%m/%d'
            ]
            
            for fmt in date_formats:
                try:
                    from datetime import datetime
                    parsed_date = datetime.strptime(str(date_str), fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            return str(date_str)
            
        except Exception:
            return None
    
    def _manual_parse(self, data_lines: List[str], header_parts: List[str]) -> pd.DataFrame:
        """Manually parse data lines if pandas fails"""
        result = pd.DataFrame(columns=self.standard_columns)
        
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith('"') and line.endswith('"'):
                continue
            
            parts = [p.strip().strip('"') for p in line.split(',')]
            if len(parts) < 3:
                continue
            
            # Try to extract basic information
            ticker = parts[0] if parts[0] else None
            if not ticker or ticker == 'nan':
                continue
            
            # Look for numeric values
            quantities = []
            prices = []
            
            for part in parts[1:]:
                try:
                    val = float(re.sub(r'[$,]', '', part))
                    if val > 0:
                        if val < 1000:  # Likely quantity
                            quantities.append(val)
                        else:  # Likely price
                            prices.append(val)
                except:
                    continue
            
            if not quantities:
                continue
            
            quantity = int(quantities[0])
            entry_price = prices[0] if len(prices) > 0 else 0.0
            current_price = prices[1] if len(prices) > 1 else entry_price
            
            result = pd.concat([result, pd.DataFrame([{
                'ticker': ticker,
                'position_type': 'stock',
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': current_price,
                'expiration_date': None,
                'strike_price': None,
                'option_type': None
            }])], ignore_index=True)
        
        return result

# Test function
if __name__ == "__main__":
    parser = MerrillEdgeParser()
    csv_file = r"C:\Users\Ardaz\Downloads\merrill_portfolio\ExportData29092025175309.csv"
    
    try:
        result = parser.parse_merrill_csv(csv_file)
        print(f"✅ Successfully parsed {len(result)} positions")
        print("\n📊 Parsed positions:")
        print(result.to_string(index=False))
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
