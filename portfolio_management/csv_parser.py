"""
Flexible CSV Parser for Broker Data
Handles different broker CSV formats and converts them to our standard format
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BrokerCSVParser:
    """Parse CSV files from different brokers and convert to standard format"""
    
    def __init__(self):
        self.standard_columns = [
            'ticker', 'position_type', 'quantity', 'entry_price', 
            'current_price', 'expiration_date', 'strike_price', 'option_type'
        ]
    
    def detect_broker_format(self, df: pd.DataFrame) -> str:
        """Detect which broker format the CSV is in"""
        columns = [col.lower().strip() for col in df.columns]
        
        # Merrill Edge detection
        if any('symbol' in col for col in columns) and any('quantity' in col for col in columns):
            if any('call' in col or 'put' in col for col in columns):
                return 'merrill_edge_options'
            else:
                return 'merrill_edge_stocks'
    
    def detect_broker_format_from_file(self, file_path: str) -> str:
        """Detect broker format by reading file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Look for Merrill Edge specific patterns
            for line in lines[:20]:  # Check first 20 lines
                if 'Merrill' in line or 'Edge' in line:
                    return 'merrill_edge_stocks'
                if 'Symbol' in line and 'Quantity' in line and 'Price' in line:
                    return 'merrill_edge_stocks'
            
            return 'unknown'
        except:
            return 'unknown'
        
        # Fidelity detection
        if any('symbol' in col for col in columns) and any('shares' in col for col in columns):
            return 'fidelity'
        
        # Schwab detection
        if any('symbol' in col for col in columns) and any('quantity' in col for col in columns):
            return 'schwab'
        
        # TD Ameritrade detection
        if any('symbol' in col for col in columns) and any('quantity' in col for col in columns):
            return 'td_ameritrade'
        
        # Generic format
        if 'ticker' in columns or 'symbol' in columns:
            return 'generic'
        
        return 'unknown'
    
    def parse_merrill_edge_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse Merrill Edge stocks CSV using specialized parser"""
        try:
            # Import the specialized Merrill Edge parser
            from merrill_edge_parser import MerrillEdgeParser
            merrill_parser = MerrillEdgeParser()
            
            # Save the dataframe to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                result = merrill_parser.parse_merrill_csv(f.name)
                os.unlink(f.name)  # Clean up temp file
                return result
                
        except Exception as e:
            logger.warning(f"Error using specialized Merrill Edge parser: {e}")
            # Fallback to basic parsing
            return self._parse_merrill_edge_basic(df)
    
    def _parse_merrill_edge_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic Merrill Edge parsing as fallback"""
        result = pd.DataFrame(columns=self.standard_columns)
        
        for _, row in df.iterrows():
            try:
                # Extract symbol
                symbol = str(row.get('Symbol', '')).strip()
                if not symbol or symbol == 'nan':
                    continue
                
                # Extract quantity
                quantity = self._safe_float(row.get('Quantity', 0))
                if quantity == 0:
                    continue
                
                # Extract prices
                entry_price = self._safe_float(row.get('Average Price', row.get('Cost Basis', 0)))
                current_price = self._safe_float(row.get('Current Price', row.get('Last Price', entry_price)))
                
                result = pd.concat([result, pd.DataFrame([{
                    'ticker': symbol,
                    'position_type': 'stock',
                    'quantity': int(quantity),
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'expiration_date': None,
                    'strike_price': None,
                    'option_type': None
                }])], ignore_index=True)
                
            except Exception as e:
                logger.warning(f"Error parsing Merrill Edge stock row: {e}")
                continue
        
        return result
    
    def parse_merrill_edge_options(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse Merrill Edge options CSV"""
        result = pd.DataFrame(columns=self.standard_columns)
        
        for _, row in df.iterrows():
            try:
                # Extract symbol
                symbol = str(row.get('Symbol', '')).strip()
                if not symbol or symbol == 'nan':
                    continue
                
                # Extract quantity
                quantity = self._safe_float(row.get('Quantity', 0))
                if quantity == 0:
                    continue
                
                # Determine if it's a call or put
                option_type = None
                if 'call' in str(row.get('Call/Put', '')).lower():
                    option_type = 'call'
                elif 'put' in str(row.get('Call/Put', '')).lower():
                    option_type = 'put'
                
                if not option_type:
                    continue
                
                # Extract strike price
                strike_price = self._safe_float(row.get('Strike Price', 0))
                
                # Extract expiration date
                expiration_date = self._parse_date(row.get('Expiration Date', ''))
                
                # Extract prices
                entry_price = self._safe_float(row.get('Average Price', row.get('Cost Basis', 0)))
                current_price = self._safe_float(row.get('Current Price', row.get('Last Price', entry_price)))
                
                result = pd.concat([result, pd.DataFrame([{
                    'ticker': symbol,
                    'position_type': option_type,
                    'quantity': int(quantity),
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'expiration_date': expiration_date,
                    'strike_price': strike_price,
                    'option_type': option_type
                }])], ignore_index=True)
                
            except Exception as e:
                logger.warning(f"Error parsing Merrill Edge option row: {e}")
                continue
        
        return result
    
    def parse_fidelity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse Fidelity CSV"""
        result = pd.DataFrame(columns=self.standard_columns)
        
        for _, row in df.iterrows():
            try:
                symbol = str(row.get('Symbol', '')).strip()
                if not symbol or symbol == 'nan':
                    continue
                
                quantity = self._safe_float(row.get('Shares', row.get('Quantity', 0)))
                if quantity == 0:
                    continue
                
                entry_price = self._safe_float(row.get('Average Cost', row.get('Cost Basis', 0)))
                current_price = self._safe_float(row.get('Last Price', row.get('Current Price', entry_price)))
                
                result = pd.concat([result, pd.DataFrame([{
                    'ticker': symbol,
                    'position_type': 'stock',
                    'quantity': int(quantity),
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'expiration_date': None,
                    'strike_price': None,
                    'option_type': None
                }])], ignore_index=True)
                
            except Exception as e:
                logger.warning(f"Error parsing Fidelity row: {e}")
                continue
        
        return result
    
    def parse_schwab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse Schwab CSV"""
        result = pd.DataFrame(columns=self.standard_columns)
        
        for _, row in df.iterrows():
            try:
                symbol = str(row.get('Symbol', '')).strip()
                if not symbol or symbol == 'nan':
                    continue
                
                quantity = self._safe_float(row.get('Quantity', 0))
                if quantity == 0:
                    continue
                
                entry_price = self._safe_float(row.get('Average Cost', row.get('Cost Basis', 0)))
                current_price = self._safe_float(row.get('Last Price', row.get('Current Price', entry_price)))
                
                result = pd.concat([result, pd.DataFrame([{
                    'ticker': symbol,
                    'position_type': 'stock',
                    'quantity': int(quantity),
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'expiration_date': None,
                    'strike_price': None,
                    'option_type': None
                }])], ignore_index=True)
                
            except Exception as e:
                logger.warning(f"Error parsing Schwab row: {e}")
                continue
        
        return result
    
    def parse_generic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse generic CSV format"""
        result = pd.DataFrame(columns=self.standard_columns)
        
        # Map common column names
        column_mapping = {
            'ticker': ['ticker', 'symbol', 'symbol'],
            'quantity': ['quantity', 'shares', 'qty'],
            'entry_price': ['entry_price', 'average_price', 'cost_basis', 'avg_price'],
            'current_price': ['current_price', 'last_price', 'price', 'current_value'],
            'expiration_date': ['expiration_date', 'expiry', 'exp_date'],
            'strike_price': ['strike_price', 'strike', 'strike_price'],
            'option_type': ['option_type', 'call_put', 'type']
        }
        
        # Find matching columns
        mapped_columns = {}
        for standard_col, possible_names in column_mapping.items():
            for col in df.columns:
                if col.lower().strip() in [name.lower() for name in possible_names]:
                    mapped_columns[standard_col] = col
                    break
        
        for _, row in df.iterrows():
            try:
                ticker = str(row.get(mapped_columns.get('ticker', ''), '')).strip()
                if not ticker or ticker == 'nan':
                    continue
                
                quantity = self._safe_float(row.get(mapped_columns.get('quantity', ''), 0))
                if quantity == 0:
                    continue
                
                entry_price = self._safe_float(row.get(mapped_columns.get('entry_price', ''), 0))
                current_price = self._safe_float(row.get(mapped_columns.get('current_price', ''), entry_price))
                
                # Determine position type
                position_type = 'stock'
                option_type = None
                strike_price = None
                expiration_date = None
                
                if mapped_columns.get('option_type'):
                    option_type_str = str(row.get(mapped_columns['option_type'], '')).lower()
                    if 'call' in option_type_str:
                        position_type = 'call'
                        option_type = 'call'
                    elif 'put' in option_type_str:
                        position_type = 'put'
                        option_type = 'put'
                
                if mapped_columns.get('strike_price'):
                    strike_price = self._safe_float(row.get(mapped_columns['strike_price'], 0))
                    if strike_price > 0:
                        position_type = option_type or 'call'
                
                if mapped_columns.get('expiration_date'):
                    expiration_date = self._parse_date(row.get(mapped_columns['expiration_date'], ''))
                
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
                logger.warning(f"Error parsing generic row: {e}")
                continue
        
        return result
    
    def parse_csv(self, file_path: str) -> pd.DataFrame:
        """Parse CSV file and return standardized format"""
        try:
            # First, try to detect broker format from file content
            broker_format = self.detect_broker_format_from_file(file_path)
            logger.info(f"Detected broker format from file: {broker_format}")
            
            # If it's Merrill Edge, use specialized parser directly
            if broker_format == 'merrill_edge_stocks':
                from merrill_edge_parser import MerrillEdgeParser
                merrill_parser = MerrillEdgeParser()
                return merrill_parser.parse_merrill_csv(file_path)
            
            # For other formats, try to read with pandas first
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any encoding")
            
            # Detect broker format from dataframe
            broker_format = self.detect_broker_format(df)
            logger.info(f"Detected broker format from dataframe: {broker_format}")
            
            # Parse based on format
            if broker_format == 'merrill_edge_stocks':
                return self.parse_merrill_edge_stocks(df)
            elif broker_format == 'merrill_edge_options':
                return self.parse_merrill_edge_options(df)
            elif broker_format == 'fidelity':
                return self.parse_fidelity(df)
            elif broker_format == 'schwab':
                return self.parse_schwab(df)
            elif broker_format == 'generic':
                return self.parse_generic(df)
            else:
                logger.warning(f"Unknown broker format, trying generic parser")
                return self.parse_generic(df)
                
        except Exception as e:
            logger.error(f"Error parsing CSV file: {e}")
            raise
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float"""
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
                    parsed_date = datetime.strptime(str(date_str), fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # If no format works, return as is
            return str(date_str)
            
        except Exception:
            return None

# Example usage
if __name__ == "__main__":
    parser = BrokerCSVParser()
    
    # Test with sample file
    try:
        result = parser.parse_csv("sample_portfolio.csv")
        print("Parsed portfolio:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
