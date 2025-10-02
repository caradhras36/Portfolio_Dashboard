-- Migration: Remove Greek columns from portfolio_positions table
-- These are now stored in the dedicated option_greeks table

-- IMPORTANT: Run this AFTER creating option_greeks table and populating it!
-- This is a one-way migration - make sure you have a backup!

BEGIN;

-- Step 1: Verify option_greeks table exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'option_greeks') THEN
        RAISE EXCEPTION 'option_greeks table does not exist! Create it first using option_greeks_schema.sql';
    END IF;
END $$;

-- Step 2: Remove Greek-related columns from portfolio_positions
-- These are now in option_greeks table

ALTER TABLE portfolio_positions 
    DROP COLUMN IF EXISTS delta,
    DROP COLUMN IF EXISTS gamma,
    DROP COLUMN IF EXISTS theta,
    DROP COLUMN IF EXISTS vega,
    DROP COLUMN IF EXISTS implied_volatility,
    DROP COLUMN IF EXISTS time_to_expiration;

-- Step 3: The cleaned portfolio_positions table now only has:
-- - Position data (ticker, quantity, prices)
-- - Stock and option basics
-- - Greeks are in separate option_greeks table

COMMIT;

-- Verify the changes
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'portfolio_positions'
ORDER BY ordinal_position;

-- You should NOT see: delta, gamma, theta, vega, implied_volatility, time_to_expiration
