-- Option Greeks Table
-- Stores current Greeks data separately from position data for better performance

CREATE TABLE IF NOT EXISTS option_greeks (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(50) NOT NULL,
    strike_price DECIMAL(10,2) NOT NULL,
    expiration_date DATE NOT NULL,
    option_type VARCHAR(10) NOT NULL CHECK (option_type IN ('call', 'put')),
    
    -- Greeks from Polygon.io
    delta DECIMAL(15,6) NOT NULL,
    gamma DECIMAL(15,6) NOT NULL,
    theta DECIMAL(15,6) NOT NULL,
    vega DECIMAL(15,6) NOT NULL,
    rho DECIMAL(15,6),
    
    -- Additional market data
    implied_volatility DECIMAL(8,4),
    time_to_expiration DECIMAL(10,2),
    underlying_price DECIMAL(10,2),
    
    -- Metadata
    data_source VARCHAR(20) DEFAULT 'polygon.io',
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint: one record per option contract
    CONSTRAINT unique_option_greeks UNIQUE (ticker, strike_price, expiration_date, option_type)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_option_greeks_ticker ON option_greeks(ticker);
CREATE INDEX IF NOT EXISTS idx_option_greeks_expiration ON option_greeks(expiration_date);
CREATE INDEX IF NOT EXISTS idx_option_greeks_updated_at ON option_greeks(updated_at);
CREATE INDEX IF NOT EXISTS idx_option_greeks_lookup ON option_greeks(ticker, strike_price, expiration_date, option_type);

-- Auto-update timestamp trigger
CREATE TRIGGER update_option_greeks_updated_at 
    BEFORE UPDATE ON option_greeks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
-- GRANT ALL ON option_greeks TO authenticated;
-- GRANT USAGE ON SEQUENCE option_greeks_id_seq TO authenticated;

-- Note: Run this in your Supabase SQL editor to create the table
