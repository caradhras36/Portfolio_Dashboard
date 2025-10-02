"""
Trading System for Portfolio Management
Handles buy/sell transactions and position management
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import os
import sys

# Add project paths for database integration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "shared"))

logger = logging.getLogger(__name__)

class TradeType(Enum):
    BUY = "buy"
    SELL = "sell"

class TransactionStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class TradeTransaction:
    """Trade transaction record"""
    id: str
    ticker: str
    trade_type: TradeType
    quantity: int
    price: float
    timestamp: datetime
    status: TransactionStatus
    notes: Optional[str] = None
    commission: float = 0.0
    total_amount: Optional[float] = None
    
    def __post_init__(self):
        if self.total_amount is None:
            self.total_amount = (self.quantity * self.price) + self.commission

@dataclass
class PositionUpdate:
    """Position update result"""
    success: bool
    message: str
    new_quantity: int
    transaction_id: Optional[str] = None
    error: Optional[str] = None

class TradingSystem:
    """Trading system for managing buy/sell transactions"""
    
    def __init__(self, supabase_client=None):
        self.transactions: List[TradeTransaction] = []
        self.commission_rate = 0.0  # No commission for demo
        self.min_order_size = 1
        self.max_order_size = 10000
        self.supabase = supabase_client
        self.use_database = supabase_client is not None
        
        if self.use_database:
            logger.info("Trading system initialized with database support")
        else:
            logger.info("Trading system initialized with in-memory storage only")
        
    async def execute_trade(self, 
                          ticker: str, 
                          trade_type: TradeType, 
                          quantity: int, 
                          price: float,
                          notes: Optional[str] = None) -> PositionUpdate:
        """Execute a buy or sell trade"""
        try:
            # Validate trade parameters
            validation_result = self._validate_trade(ticker, trade_type, quantity, price)
            if not validation_result["valid"]:
                return PositionUpdate(
                    success=False,
                    message=validation_result["error"],
                    new_quantity=0,
                    error=validation_result["error"]
                )
            
            # Create transaction record
            transaction = TradeTransaction(
                id=f"TXN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ticker}",
                ticker=ticker.upper(),
                trade_type=trade_type,
                quantity=quantity,
                price=price,
                timestamp=datetime.now(),
                status=TransactionStatus.COMPLETED,  # Auto-complete for demo
                notes=notes,
                commission=self.commission_rate * quantity * price
            )
            
            # Store transaction
            self.transactions.append(transaction)
            
            # Save to database if available
            if self.use_database:
                await self._save_transaction_to_db(transaction)
            
            # Calculate new position quantity
            current_quantity = await self._get_current_position_quantity(ticker)
            
            if trade_type == TradeType.BUY:
                new_quantity = current_quantity + quantity
            else:  # SELL
                new_quantity = current_quantity - quantity
                
                # Check if we have enough shares to sell
                if new_quantity < 0:
                    return PositionUpdate(
                        success=False,
                        message=f"Insufficient shares to sell. Current position: {current_quantity} shares",
                        new_quantity=current_quantity,
                        error="Insufficient shares"
                    )
            
            logger.info(f"Trade executed: {trade_type.value.upper()} {quantity} {ticker} @ ${price}")
            
            return PositionUpdate(
                success=True,
                message=f"Successfully {trade_type.value} {quantity} shares of {ticker} at ${price}",
                new_quantity=new_quantity,
                transaction_id=transaction.id
            )
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return PositionUpdate(
                success=False,
                message=f"Error executing trade: {str(e)}",
                new_quantity=0,
                error=str(e)
            )
    
    def _validate_trade(self, ticker: str, trade_type: TradeType, quantity: int, price: float) -> Dict[str, Any]:
        """Validate trade parameters"""
        # Check ticker
        if not ticker or not ticker.strip():
            return {"valid": False, "error": "Ticker symbol is required"}
        
        # Check quantity
        if quantity < self.min_order_size:
            return {"valid": False, "error": f"Minimum order size is {self.min_order_size} shares"}
        
        if quantity > self.max_order_size:
            return {"valid": False, "error": f"Maximum order size is {self.max_order_size} shares"}
        
        # Check price
        if price <= 0:
            return {"valid": False, "error": "Price must be greater than 0"}
        
        return {"valid": True}
    
    async def _get_current_position_quantity(self, ticker: str) -> int:
        """Get current position quantity for a ticker"""
        # This would integrate with the portfolio positions
        # For now, calculate from transaction history
        total_quantity = 0
        
        for transaction in self.transactions:
            if transaction.ticker.upper() == ticker.upper() and transaction.status == TransactionStatus.COMPLETED:
                if transaction.trade_type == TradeType.BUY:
                    total_quantity += transaction.quantity
                else:  # SELL
                    total_quantity -= transaction.quantity
        
        return total_quantity
    
    def get_transaction_history(self, ticker: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get transaction history"""
        filtered_transactions = self.transactions
        
        if ticker:
            filtered_transactions = [t for t in filtered_transactions if t.ticker.upper() == ticker.upper()]
        
        # Sort by timestamp (newest first)
        filtered_transactions.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Limit results
        filtered_transactions = filtered_transactions[:limit]
        
        # Convert to dictionaries for JSON serialization
        return [asdict(transaction) for transaction in filtered_transactions]
    
    def get_position_summary(self, ticker: str) -> Dict[str, Any]:
        """Get position summary for a ticker"""
        ticker_transactions = [t for t in self.transactions if t.ticker.upper() == ticker.upper()]
        
        total_shares = 0
        total_cost = 0.0
        total_sales = 0.0
        buy_transactions = 0
        sell_transactions = 0
        
        for transaction in ticker_transactions:
            if transaction.status != TransactionStatus.COMPLETED:
                continue
                
            if transaction.trade_type == TradeType.BUY:
                total_shares += transaction.quantity
                total_cost += transaction.total_amount
                buy_transactions += 1
            else:  # SELL
                total_shares -= transaction.quantity
                total_sales += transaction.total_amount
                sell_transactions += 1
        
        avg_cost = total_cost / max(total_shares, 1) if total_shares > 0 else 0
        
        return {
            "ticker": ticker.upper(),
            "current_quantity": total_shares,
            "total_cost": total_cost,
            "total_sales": total_sales,
            "average_cost": avg_cost,
            "buy_transactions": buy_transactions,
            "sell_transactions": sell_transactions,
            "total_transactions": len(ticker_transactions)
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio trading summary"""
        tickers = set(t.ticker for t in self.transactions)
        
        total_transactions = len(self.transactions)
        total_volume = sum(t.quantity for t in self.transactions)
        total_commission = sum(t.commission for t in self.transactions)
        
        position_summaries = {}
        for ticker in tickers:
            position_summaries[ticker] = self.get_position_summary(ticker)
        
        return {
            "total_tickers": len(tickers),
            "total_transactions": total_transactions,
            "total_volume": total_volume,
            "total_commission": total_commission,
            "positions": position_summaries
        }
    
    def cancel_transaction(self, transaction_id: str) -> bool:
        """Cancel a pending transaction"""
        for transaction in self.transactions:
            if transaction.id == transaction_id and transaction.status == TransactionStatus.PENDING:
                transaction.status = TransactionStatus.CANCELLED
                logger.info(f"Transaction {transaction_id} cancelled")
                return True
        
        return False
    
    def get_transaction_by_id(self, transaction_id: str) -> Optional[TradeTransaction]:
        """Get transaction by ID"""
        for transaction in self.transactions:
            if transaction.id == transaction_id:
                return transaction
        return None
    
    async def _save_transaction_to_db(self, transaction: TradeTransaction):
        """Save transaction to database"""
        try:
            if not self.supabase:
                return
                
            data = {
                'transaction_id': transaction.id,
                'ticker': transaction.ticker,
                'trade_type': transaction.trade_type.value,
                'quantity': transaction.quantity,
                'price': transaction.price,
                'commission': transaction.commission,
                'total_amount': transaction.total_amount,
                'status': transaction.status.value,
                'notes': transaction.notes,
                'created_at': transaction.timestamp.isoformat()
            }
            
            self.supabase.table('trade_transactions').insert(data).execute()
            logger.info(f"Transaction {transaction.id} saved to database")
            
        except Exception as e:
            logger.error(f"Error saving transaction to database: {e}")
    
    async def _save_position_history(self, ticker: str, position_type: str, quantity: int, 
                                   entry_price: float, current_price: float, 
                                   change_type: str, reference_transaction_id: str = None,
                                   notes: str = None):
        """Save position change to history"""
        try:
            if not self.supabase:
                return
                
            data = {
                'ticker': ticker,
                'position_type': position_type,
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': current_price,
                'change_type': change_type,
                'reference_transaction_id': reference_transaction_id,
                'notes': notes,
                'created_at': datetime.now().isoformat()
            }
            
            self.supabase.table('position_history').insert(data).execute()
            logger.debug(f"Position history saved for {ticker}")
            
        except Exception as e:
            logger.error(f"Error saving position history: {e}")
    
    async def load_transactions_from_db(self):
        """Load transactions from database on startup"""
        try:
            if not self.supabase:
                return
                
            response = self.supabase.table('trade_transactions').select('*').order('created_at', desc=True).execute()
            
            if response.data:
                self.transactions = []
                for row in response.data:
                    transaction = TradeTransaction(
                        id=row['transaction_id'],
                        ticker=row['ticker'],
                        trade_type=TradeType(row['trade_type']),
                        quantity=row['quantity'],
                        price=float(row['price']),
                        timestamp=datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')),
                        status=TransactionStatus(row['status']),
                        notes=row.get('notes'),
                        commission=float(row.get('commission', 0)),
                        total_amount=float(row['total_amount'])
                    )
                    self.transactions.append(transaction)
                
                logger.info(f"Loaded {len(self.transactions)} transactions from database")
            
        except Exception as e:
            logger.error(f"Error loading transactions from database: {e}")
    
    async def get_transaction_history_from_db(self, ticker: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get transaction history from database"""
        try:
            if not self.supabase:
                return self.get_transaction_history(ticker, limit)
            
            query = self.supabase.table('trade_transactions').select('*')
            
            if ticker:
                query = query.eq('ticker', ticker.upper())
            
            response = query.order('created_at', desc=True).limit(limit).execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error fetching transaction history from database: {e}")
            return self.get_transaction_history(ticker, limit)
    
    async def get_position_summary_from_db(self, ticker: str) -> Dict[str, Any]:
        """Get position summary from database transactions"""
        try:
            if not self.supabase:
                return self.get_position_summary(ticker)
            
            response = self.supabase.table('trade_transactions').select('*').eq('ticker', ticker.upper()).execute()
            
            if not response.data:
                return {
                    "ticker": ticker.upper(),
                    "current_quantity": 0,
                    "total_cost": 0,
                    "total_sales": 0,
                    "average_cost": 0,
                    "buy_transactions": 0,
                    "sell_transactions": 0,
                    "total_transactions": 0
                }
            
            total_shares = 0
            total_cost = 0.0
            total_sales = 0.0
            buy_transactions = 0
            sell_transactions = 0
            
            for transaction in response.data:
                if transaction['status'] != 'completed':
                    continue
                    
                if transaction['trade_type'] == 'buy':
                    total_shares += transaction['quantity']
                    total_cost += transaction['total_amount']
                    buy_transactions += 1
                else:  # sell
                    total_shares -= transaction['quantity']
                    total_sales += transaction['total_amount']
                    sell_transactions += 1
            
            avg_cost = total_cost / max(total_shares, 1) if total_shares > 0 else 0
            
            return {
                "ticker": ticker.upper(),
                "current_quantity": total_shares,
                "total_cost": total_cost,
                "total_sales": total_sales,
                "average_cost": avg_cost,
                "buy_transactions": buy_transactions,
                "sell_transactions": sell_transactions,
                "total_transactions": len(response.data)
            }
            
        except Exception as e:
            logger.error(f"Error fetching position summary from database: {e}")
            return self.get_position_summary(ticker)

# Global trading system instance
trading_system = TradingSystem()

# Example usage
async def test_trading_system():
    """Test the trading system functionality"""
    trading = TradingSystem()
    
    print("Testing Trading System")
    print("=" * 30)
    
    # Test buy order
    result = await trading.execute_trade("AAPL", TradeType.BUY, 100, 175.50, "Initial position")
    print(f"Buy order: {result.message}")
    print(f"New quantity: {result.new_quantity}")
    
    # Test sell order
    result = await trading.execute_trade("AAPL", TradeType.SELL, 50, 180.00, "Partial profit taking")
    print(f"\nSell order: {result.message}")
    print(f"New quantity: {result.new_quantity}")
    
    # Test insufficient shares
    result = await trading.execute_trade("AAPL", TradeType.SELL, 100, 185.00, "Too many shares")
    print(f"\nSell order (insufficient): {result.message}")
    
    # Get transaction history
    history = trading.get_transaction_history("AAPL")
    print(f"\nTransaction History for AAPL:")
    for transaction in history:
        print(f"  {transaction['timestamp']}: {transaction['trade_type'].upper()} {transaction['quantity']} @ ${transaction['price']}")
    
    # Get position summary
    summary = trading.get_position_summary("AAPL")
    print(f"\nPosition Summary for AAPL:")
    print(f"  Current Quantity: {summary['current_quantity']}")
    print(f"  Average Cost: ${summary['average_cost']:.2f}")
    print(f"  Total Transactions: {summary['total_transactions']}")

if __name__ == "__main__":
    asyncio.run(test_trading_system())
