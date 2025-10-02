-- Stock Metrics Cache Table
-- Stores pre-calculated beta, volatility, and liquidity to avoid recalculation

CREATE TABLE IF NOT EXISTS stock_metrics_cache (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL UNIQUE,
    beta DECIMAL(6, 3) NOT NULL,
    volatility DECIMAL(6, 4) NOT NULL,
    liquidity_score DECIMAL(5, 2) NOT NULL,
    calculation_date DATE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- One record per ticker
    UNIQUE (ticker)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_stock_metrics_ticker 
ON stock_metrics_cache(ticker);

-- Index for checking freshness
CREATE INDEX IF NOT EXISTS idx_stock_metrics_date 
ON stock_metrics_cache(calculation_date DESC);

COMMENT ON TABLE stock_metrics_cache IS 'Cached beta, volatility, and liquidity calculations to avoid expensive recomputation';
COMMENT ON COLUMN stock_metrics_cache.beta IS 'Portfolio beta calculated from historical returns';
COMMENT ON COLUMN stock_metrics_cache.volatility IS 'Annualized volatility';
COMMENT ON COLUMN stock_metrics_cache.liquidity_score IS 'Liquidity score 0-100';
COMMENT ON COLUMN stock_metrics_cache.calculation_date IS 'Date when metrics were last calculated';
