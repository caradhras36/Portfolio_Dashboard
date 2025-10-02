#!/usr/bin/env python3
"""
Utility script to update Greeks in the database
Run this periodically (e.g., every hour) to refresh Greek values
"""

import asyncio
import logging
from datetime import datetime
import sys
import os

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "shared"))
sys.path.append(os.path.join(project_root, "portfolio_management"))

from shared.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client
from portfolio_api import get_portfolio_positions, calculate_greeks_for_position

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def update_all_greeks():
    """Update Greeks for all options positions in the database"""
    try:
        # Initialize Supabase
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Connected to Supabase")
        
        # Get all positions (without recalculating Greeks)
        positions = await get_portfolio_positions(recalculate_greeks=False)
        logger.info(f"Found {len(positions)} positions")
        
        # Filter options
        options = [pos for pos in positions if pos.position_type in ['call', 'put']]
        logger.info(f"Found {len(options)} options positions to update")
        
        if not options:
            logger.info("No options to update")
            return
        
        # Recalculate Greeks for all options
        updated_count = 0
        for pos in options:
            try:
                # Recalculate Greeks
                updated_pos = await calculate_greeks_for_position(pos, force_recalc=True)
                
                # Update in database
                supabase.table('positions').update({
                    'delta': updated_pos.delta,
                    'gamma': updated_pos.gamma,
                    'theta': updated_pos.theta,
                    'vega': updated_pos.vega,
                    'implied_volatility': updated_pos.implied_volatility,
                    'time_to_expiration': updated_pos.time_to_expiration,
                    'updated_at': datetime.now().isoformat()
                }).eq('ticker', pos.ticker).eq('position_type', pos.position_type).execute()
                
                updated_count += 1
                logger.info(f"Updated Greeks for {pos.ticker} {pos.position_type}: delta={updated_pos.delta:.4f}, theta={updated_pos.theta:.4f}")
                
            except Exception as e:
                logger.error(f"Error updating Greeks for {pos.ticker}: {e}")
                continue
        
        logger.info(f"Successfully updated Greeks for {updated_count}/{len(options)} options")
        
    except Exception as e:
        logger.error(f"Error in update_all_greeks: {e}")
        raise

if __name__ == "__main__":
    print("=" * 60)
    print("Greek Values Update Utility")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    asyncio.run(update_all_greeks())
    
    print()
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
