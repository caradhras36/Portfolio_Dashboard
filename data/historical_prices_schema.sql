-- Historical Prices Cache Table
-- Stores daily closing prices for stocks to avoid repeated Polygon.io API calls

CREATE TABLE IF NOT EXISTS historical_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    close_price DECIMAL(12, 4) NOT NULL,
    open_price DECIMAL(12, 4),
    high_price DECIMAL(12, 4),
    low_price DECIMAL(12, 4),
    volume BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure one price record per ticker per day
    UNIQUE (ticker, date)
);

-- Index for fast lookups by ticker and date range
CREATE INDEX IF NOT EXISTS idx_historical_prices_ticker_date 
ON historical_prices(ticker, date DESC);

-- Index for fast date-only queries
CREATE INDEX IF NOT EXISTS idx_historical_prices_date 
ON historical_prices(date DESC);

COMMENT ON TABLE historical_prices IS 'Cache of daily stock prices to reduce API calls to Polygon.io';
COMMENT ON COLUMN historical_prices.ticker IS 'Stock ticker symbol';
COMMENT ON COLUMN historical_prices.date IS 'Trading date';
COMMENT ON COLUMN historical_prices.close_price IS 'Closing price for the day';
