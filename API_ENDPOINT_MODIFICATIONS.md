# API Endpoint Modifications for EV Delayed Data Integration

## 🎯 **Step-by-Step Integration Guide**

### **Step 1: Add Imports to portfolio_api.py**

Add these imports at the top of your `portfolio_api.py` file:

```python
# Add these imports
from ev_delayed_integration import EVDelayedDataIntegration
```

### **Step 2: Initialize Enhanced Recommender**

Add this initialization after your existing analyzer initialization:

```python
# Add after your existing initialization (around line 250)
# Initialize EV-enhanced recommender with delayed data optimization
ev_enhanced_recommender = EVDelayedDataIntegration(
    polygon_api_key=polygon_api_key,
    supabase_client=supabase
)
logger.info("✅ EV Enhanced Recommender with Delayed Data Optimization initialized")
```

### **Step 3: Replace Covered Call Recommendations Endpoint**

Replace your existing covered call endpoint (around line 2400) with this:

```python
@app.get("/api/options/recommendations/covered-calls")
async def get_cc_recommendations():
    """Get EV-enhanced covered call recommendations with delayed data optimization"""
    try:
        logger.info("🔍 Starting EV-enhanced covered call recommendations...")
        
        # Get current portfolio positions
        positions = await get_portfolio_positions()
        logger.info(f"📊 Found {len(positions)} total positions")
        
        if not positions:
            return {"message": "No positions in portfolio", "recommendations": []}
        
        # Filter for stock positions only
        stock_positions = [pos for pos in positions if pos.position_type == 'stock' and pos.quantity > 0]
        logger.info(f"📈 Found {len(stock_positions)} stock positions")
        
        if not stock_positions:
            return {"message": "No stock positions available for covered calls", "recommendations": []}
        
        # Get existing covered call positions and calculate available shares for additional CCs
        existing_ccs = [pos for pos in positions if pos.position_type == 'call' and pos.quantity < 0]
        
        # Calculate how many shares are already committed to covered calls
        cc_shares_committed = {}
        for cc_pos in existing_ccs:
            ticker = cc_pos.ticker
            # Each call contract covers 100 shares, and quantity is negative for short positions
            shares_committed = abs(cc_pos.quantity) * 100
            cc_shares_committed[ticker] = cc_shares_committed.get(ticker, 0) + shares_committed
        
        # Find tickers that are completely blocked (no shares available for additional CCs)
        completely_blocked_tickers = set()
        for stock_pos in stock_positions:
            ticker = stock_pos.ticker
            shares_owned = stock_pos.quantity
            shares_committed = cc_shares_committed.get(ticker, 0)
            
            # If all shares are committed to existing CCs, block this ticker
            if shares_committed >= shares_owned:
                completely_blocked_tickers.add(ticker)
        
        blocked_tickers = completely_blocked_tickers
        logger.info(f"🚫 Completely blocked tickers (no shares available for additional CCs): {blocked_tickers}")
        logger.info(f"📊 CC shares committed by ticker: {cc_shares_committed}")
        
        # Convert positions to dict format for the recommender
        stock_pos_dicts = [asdict(pos) for pos in stock_positions]
        
        # Get Polygon API key
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not polygon_api_key:
            logger.error("POLYGON_API_KEY not found in environment variables")
            raise HTTPException(status_code=500, detail="Polygon.io API key required for covered call recommendations")
        
        # Use the EV-enhanced recommender with delayed data optimization
        result = ev_enhanced_recommender.get_enhanced_cc_recommendations(
            stock_pos_dicts, blocked_tickers, cc_shares_committed
        )
        
        # Extract data from the result dictionary
        recommendations = result['recommendations']
        underlying_tickers_considered = result['underlying_tickers_considered']
        underlying_tickers_in_results = result['underlying_tickers_in_results']
        
        # Sort by EV score (already sorted by the enhanced recommender)
        # recommendations.sort(key=lambda x: x['ev_score'], reverse=True)
        
        logger.info(f"✅ Generated {len(recommendations)} EV-enhanced recommendations from {len(stock_positions)} stock positions")
        logger.info(f"📊 Processed stocks: {len(stock_positions) - len(blocked_tickers)}, Blocked: {len(blocked_tickers)}")
        logger.info(f"📊 Underlying tickers: {underlying_tickers_considered} considered, {underlying_tickers_in_results} in results")
        logger.info(f"📊 Market regime: {result.get('market_regime', 'unknown')}")
        logger.info(f"📊 Delayed data optimization: {result.get('delayed_data_optimization', False)}")
        logger.info(f"📊 Filtered recommendations: {result.get('filtered_recommendations', 0)}")
        
        if len(recommendations) == 0:
            logger.warning(f"⚠️ No recommendations generated! This might be due to:")
            logger.warning(f"   - Options chains not available for these tickers")
            logger.warning(f"   - All stocks already have covered calls")
            logger.warning(f"   - API errors fetching options data")
            logger.warning(f"   - High volatility filtering (delayed data safety)")
        
        return {
            'recommendations': recommendations[:20],  # Top 20 recommendations
            'total_available': len(stock_positions),
            'total_considered': result['total_considered'],  # Total options analyzed
            'blocked_tickers': list(blocked_tickers),
            'underlying_tickers_considered': underlying_tickers_considered,
            'underlying_tickers_in_results': underlying_tickers_in_results,
            'analysis_date': datetime.now().isoformat(),
            'scoring_method': result.get('scoring_method', 'Expected Value Optimized'),
            'market_regime': result.get('market_regime', 'unknown'),
            'delayed_data_optimization': result.get('delayed_data_optimization', False),
            'safety_margins_applied': result.get('safety_margins_applied', False),
            'filtered_recommendations': result.get('filtered_recommendations', 0),
            'total_enhanced': result.get('total_enhanced', 0)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in EV-enhanced covered call recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting EV-enhanced recommendations: {str(e)}")
```

### **Step 4: Replace CSP Recommendations Endpoint**

Replace your existing CSP endpoint (around line 2488) with this:

```python
@app.post("/api/options/recommendations/cash-secured-puts")
async def get_csp_recommendations(request_data: dict = None):
    """Get EV-enhanced CSP recommendations with delayed data optimization"""
    try:
        logger.info("🔍 Starting EV-enhanced CSP recommendations...")
        
        # Get tickers from request data (comma-separated string)
        tickers_input = None
        combine_with_watchlist = False
        
        if request_data:
            if 'tickers' in request_data:
                tickers_input = request_data['tickers']
                logger.info(f"📝 Using provided tickers: {tickers_input}")
            if 'combine_with_watchlist' in request_data:
                combine_with_watchlist = request_data['combine_with_watchlist']
                logger.info(f"🔄 Combine with watchlist: {combine_with_watchlist}")
        
        # Get Polygon API key
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not polygon_api_key:
            logger.error("POLYGON_API_KEY not found in environment variables")
            raise HTTPException(status_code=500, detail="Polygon.io API key required for CSP recommendations")
        
        # Use the EV-enhanced recommender with delayed data optimization
        result = ev_enhanced_recommender.get_enhanced_csp_recommendations(
            tickers_input, combine_with_watchlist
        )
        
        logger.info(f"✅ Generated {len(result['recommendations'])} EV-enhanced CSP recommendations from {result['total_considered']} options")
        logger.info(f"📊 Market regime: {result.get('market_regime', 'unknown')}")
        logger.info(f"📊 Delayed data optimization: {result.get('delayed_data_optimization', False)}")
        logger.info(f"📊 Filtered recommendations: {result.get('filtered_recommendations', 0)}")
        
        return {
            'recommendations': result['recommendations'],
            'total_available': len(result['recommendations']),
            'total_considered': result['total_considered'],
            'underlying_tickers_considered': result['underlying_tickers_considered'],
            'underlying_tickers_in_results': result['underlying_tickers_in_results'],
            'analysis_date': datetime.now().isoformat(),
            'scoring_method': result.get('scoring_method', 'Expected Value Optimized'),
            'market_regime': result.get('market_regime', 'unknown'),
            'delayed_data_optimization': result.get('delayed_data_optimization', False),
            'safety_margins_applied': result.get('safety_margins_applied', False),
            'filtered_recommendations': result.get('filtered_recommendations', 0),
            'total_enhanced': result.get('total_enhanced', 0)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in EV-enhanced CSP recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting EV-enhanced CSP recommendations: {str(e)}")
```

### **Step 5: Add New EV Analysis Endpoint**

Add this new endpoint for deep-dive EV analysis:

```python
@app.get("/api/options/ev-analysis/{ticker}")
async def get_ev_analysis(ticker: str, option_type: str = 'call'):
    """Get detailed Expected Value analysis for a specific ticker with delayed data optimization"""
    try:
        logger.info(f"🔍 Starting EV analysis for {ticker} {option_type}s with delayed data optimization...")
        
        # Get market data for regime detection
        market_data = ev_enhanced_recommender.get_market_data_for_regime_detection()
        
        # Get technical data (you'd integrate with your existing technical analysis)
        stock_data = get_technical_indicators(ticker)  # Your existing function
        
        # Get options data
        options_data = await market_analyzer.fetch_options_data(ticker)  # Your existing function
        
        # Filter by option type
        if option_type.lower() == 'call':
            options_data = [opt for opt in options_data if opt['contract_type'] == 'call']
        else:
            options_data = [opt for opt in options_data if opt['contract_type'] == 'put']
        
        # Get portfolio context
        positions = await get_portfolio_positions()
        portfolio_positions = [asdict(pos) for pos in positions]
        
        # Analyze each option with EV scoring and delayed data optimization
        ev_analysis = []
        filtered_count = 0
        
        for option in options_data:
            try:
                # Calculate EV score with delayed data optimization
                ev_result = ev_enhanced_recommender.enhanced_ev_scorer.calculate_optimized_score(
                    option, stock_data, market_data, portfolio_positions
                )
                
                if ev_result is None:
                    filtered_count += 1
                    continue
                
                ev_analysis.append({
                    'strike': option['strike_price'],
                    'premium': option['current_price'],
                    'delta': option['delta'],
                    'dte': option['days_to_expiration'],
                    'ev_score': ev_result['final_score'],
                    'expected_value': ev_result['expected_value'],
                    'confidence': ev_result['confidence'],
                    'delta_optimization_score': ev_result['delta_optimization_score'],
                    'optimal_delta_range': ev_result['optimal_delta_range'],
                    'market_regime': ev_result['market_regime'],
                    'delayed_data_adjusted': ev_result.get('delayed_data_adjusted', False),
                    'safety_margin_applied': 0.02,
                    'data_delay_minutes': 15
                })
                
            except Exception as e:
                logger.error(f"Error analyzing option {option['strike_price']}: {e}")
                continue
        
        # Sort by EV score
        ev_analysis.sort(key=lambda x: x['ev_score'], reverse=True)
        
        return {
            'ticker': ticker,
            'option_type': option_type,
            'market_regime': market_data.get('primary_regime', 'unknown'),
            'analysis': ev_analysis,
            'recommendations': ev_analysis[:10],  # Top 10
            'filtered_options': filtered_count,
            'delayed_data_optimization': True,
            'safety_margins_applied': True,
            'analysis_date': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in EV analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting EV analysis: {str(e)}")
```

## 🚀 **Testing the Integration**

### **Step 6: Test the Enhanced System**

Create a test script to verify everything works:

```python
# test_ev_integration.py
import asyncio
from ev_delayed_integration import EVDelayedDataIntegration

async def test_enhanced_system():
    """Test the enhanced EV system"""
    
    # Initialize enhanced recommender
    enhanced_recommender = EVDelayedDataIntegration(
        polygon_api_key=os.getenv('POLYGON_API_KEY'),
        supabase_client=supabase
    )
    
    # Test with sample data
    sample_positions = [
        {
            'ticker': 'AAPL',
            'quantity': 100,
            'current_price': 150.0
        }
    ]
    
    # Get enhanced recommendations
    result = enhanced_recommender.get_enhanced_cc_recommendations(
        stock_positions=sample_positions,
        blocked_tickers=set(),
        cc_shares_committed={}
    )
    
    print(f"✅ Test successful!")
    print(f"Generated {len(result['recommendations'])} recommendations")
    print(f"Market regime: {result['market_regime']}")
    print(f"Delayed data optimization: {result['delayed_data_optimization']}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_system())
```

## 📊 **Expected Results**

After integration, you'll see these new fields in your API responses:

- `ev_score`: Expected Value-based score
- `expected_value`: Mathematical expected value
- `confidence`: Confidence level (0.1-0.95)
- `market_regime`: Bull/Bear/Sideways/High-Vol
- `delayed_data_adjusted`: True/False
- `safety_margin_applied`: True/False
- `data_delay_minutes`: 15
- `filtered_recommendations`: Number filtered out for safety

## 🎯 **Key Benefits**

1. **Mathematical Optimization**: True Expected Value scoring
2. **Delayed Data Safety**: Built-in safety margins and volatility filtering
3. **Market Regime Awareness**: Adapts to different market conditions
4. **Portfolio Context**: Considers existing positions and diversification
5. **Enhanced Risk Management**: 40% max assignment probability with confidence-based selection

The integration is designed to be a drop-in replacement for your existing endpoints while adding all the EV optimization and delayed data safety features!
