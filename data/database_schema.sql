-- Portfolio Dashboard Database Schema
-- Run this in your Supabase SQL editor

-- Portfolio positions table
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(50) NOT NULL,
    position_type VARCHAR(20) NOT NULL CHECK (position_type IN ('stock', 'call', 'put', 'spread')),
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,2) NOT NULL,
    current_price DECIMAL(10,2) NOT NULL,
    expiration_date DATE,
    strike_price DECIMAL(10,2),
    option_type VARCHAR(10) CHECK (option_type IN ('call', 'put')),
    delta DECIMAL(15,6),
    gamma DECIMAL(15,6),
    theta DECIMAL(15,6),
    vega DECIMAL(15,6),
    implied_volatility DECIMAL(8,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolio risk snapshots table
CREATE TABLE IF NOT EXISTS portfolio_risk_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE DEFAULT CURRENT_DATE,
    total_delta DECIMAL(15,2),
    total_gamma DECIMAL(15,2),
    total_theta DECIMAL(15,2),
    total_vega DECIMAL(15,2),
    portfolio_value DECIMAL(15,2),
    risk_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_ticker ON portfolio_positions(ticker);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_type ON portfolio_positions(position_type);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_created_at ON portfolio_positions(created_at);
CREATE INDEX IF NOT EXISTS idx_risk_snapshots_date ON portfolio_risk_snapshots(snapshot_date);

-- Create updated_at trigger for portfolio_positions
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_portfolio_positions_updated_at 
    BEFORE UPDATE ON portfolio_positions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Sample data for testing (optional)
-- INSERT INTO portfolio_positions (ticker, position_type, quantity, entry_price, current_price) VALUES
-- ('AAPL', 'stock', 100, 150.00, 155.00),
-- ('TSLA', 'stock', 50, 200.00, 210.00),
-- ('GOOGL', 'call', 10, 5.50, 6.20),
-- ('MSFT', 'put', 5, 3.25, 2.80);

-- Trade transactions table
CREATE TABLE IF NOT EXISTS trade_transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    ticker VARCHAR(50) NOT NULL,
    trade_type VARCHAR(10) NOT NULL CHECK (trade_type IN ('buy', 'sell')),
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    commission DECIMAL(10,2) DEFAULT 0.00,
    total_amount DECIMAL(15,2) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'completed' CHECK (status IN ('pending', 'completed', 'cancelled', 'failed')),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolio positions history table (for tracking position changes)
CREATE TABLE IF NOT EXISTS position_history (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(50) NOT NULL,
    position_type VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,2) NOT NULL,
    current_price DECIMAL(10,2) NOT NULL,
    change_type VARCHAR(20) NOT NULL CHECK (change_type IN ('trade', 'price_update', 'import', 'manual')),
    reference_transaction_id VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_trade_transactions_ticker ON trade_transactions(ticker);
CREATE INDEX IF NOT EXISTS idx_trade_transactions_type ON trade_transactions(trade_type);
CREATE INDEX IF NOT EXISTS idx_trade_transactions_created_at ON trade_transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_trade_transactions_transaction_id ON trade_transactions(transaction_id);

CREATE INDEX IF NOT EXISTS idx_position_history_ticker ON position_history(ticker);
CREATE INDEX IF NOT EXISTS idx_position_history_created_at ON position_history(created_at);
CREATE INDEX IF NOT EXISTS idx_position_history_change_type ON position_history(change_type);

-- Create updated_at trigger for trade_transactions
CREATE TRIGGER update_trade_transactions_updated_at 
    BEFORE UPDATE ON trade_transactions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT ALL ON portfolio_positions TO authenticated;
-- GRANT ALL ON portfolio_risk_snapshots TO authenticated;
-- GRANT ALL ON trade_transactions TO authenticated;
-- GRANT ALL ON position_history TO authenticated;
-- GRANT USAGE ON SEQUENCE portfolio_positions_id_seq TO authenticated;
-- GRANT USAGE ON SEQUENCE portfolio_risk_snapshots_id_seq TO authenticated;
-- GRANT USAGE ON SEQUENCE trade_transactions_id_seq TO authenticated;
-- GRANT USAGE ON SEQUENCE position_history_id_seq TO authenticated;
