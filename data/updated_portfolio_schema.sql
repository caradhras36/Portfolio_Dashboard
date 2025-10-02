-- Updated Portfolio Positions Table Schema
-- Greeks removed - now in separate option_greeks table

-- Portfolio positions table (CLEANED - no Greeks)
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(50) NOT NULL,
    position_type VARCHAR(20) NOT NULL CHECK (position_type IN ('stock', 'call', 'put', 'spread')),
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,2) NOT NULL,
    current_price DECIMAL(10,2) NOT NULL,
    
    -- Options-specific fields (basic info only)
    expiration_date DATE,
    strike_price DECIMAL(10,2),
    option_type VARCHAR(10) CHECK (option_type IN ('call', 'put')),
    
    -- Greeks are NO LONGER HERE - they're in option_greeks table!
    -- This table is now just for position tracking
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_ticker ON portfolio_positions(ticker);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_type ON portfolio_positions(position_type);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_created_at ON portfolio_positions(created_at);

-- Create updated_at trigger
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

-- Grant permissions
-- GRANT ALL ON portfolio_positions TO authenticated;
-- GRANT USAGE ON SEQUENCE portfolio_positions_id_seq TO authenticated;
