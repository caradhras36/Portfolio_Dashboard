#!/usr/bin/env python3
"""
Portfolio Scenario Analysis System
Provides comprehensive scenario analysis including stress testing, Monte Carlo simulation, and impact analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import math
from scipy import stats
from scipy.stats import norm
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class ScenarioResult:
    """Result of a scenario analysis"""
    scenario_name: str
    portfolio_value: float
    pnl: float
    pnl_pct: float
    delta: float
    gamma: float
    theta: float
    vega: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional Value at Risk 95%
    max_drawdown: float
    sharpe_ratio: float
    scenario_details: Dict[str, Any]

@dataclass
class StressTestScenario:
    """Definition of a stress test scenario"""
    name: str
    description: str
    market_moves: Dict[str, float]  # Ticker -> percentage change
    volatility_changes: Dict[str, float]  # Ticker -> volatility multiplier
    time_decay_days: int  # Days of time decay to apply
    interest_rate_change: float  # Change in risk-free rate

class ScenarioAnalyzer:
    """Comprehensive scenario analysis for portfolio risk management"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.default_scenarios = self._create_default_scenarios()
    
    def _create_default_scenarios(self) -> List[StressTestScenario]:
        """Create default stress test scenarios with detailed explanations"""
        return [
            StressTestScenario(
                name="Market Crash -20%",
                description="Simulate a 20% market decline across all positions",
                market_moves={},  # Will be applied to all
                volatility_changes={},  # Will be applied to all
                time_decay_days=7,
                interest_rate_change=0.0
            ),
            StressTestScenario(
                name="Market Rally +15%",
                description="Simulate a 15% market rally across all positions",
                market_moves={},
                volatility_changes={},
                time_decay_days=7,
                interest_rate_change=0.0
            ),
            StressTestScenario(
                name="Volatility Spike",
                description="Simulate a volatility spike (2x normal) with 5% market decline",
                market_moves={},
                volatility_changes={},
                time_decay_days=7,
                interest_rate_change=0.0
            ),
            StressTestScenario(
                name="Interest Rate Shock",
                description="Simulate a 2% interest rate increase with 10% market decline",
                market_moves={},
                volatility_changes={},
                time_decay_days=14,
                interest_rate_change=0.02
            ),
            StressTestScenario(
                name="Sector Rotation",
                description="Simulate tech sector decline (-25%) and financial sector rally (+15%)",
                market_moves={},
                volatility_changes={},
                time_decay_days=7,
                interest_rate_change=0.0
            )
        ]
    
    def get_scenario_explanations(self) -> Dict[str, Dict[str, str]]:
        """Get detailed explanations for each stress test scenario"""
        return {
            "Market Crash -20%": {
                "purpose": "Tests portfolio resilience during severe market downturns",
                "what_happens": "All stock positions decline by 20%, options experience significant losses due to delta exposure",
                "key_risks": "High delta exposure amplifies losses, gamma risk increases as options move further OTM",
                "what_to_look_for": "Portfolio value decline, P&L impact, VaR levels, Greeks changes",
                "interpretation": "A healthy portfolio should show manageable losses. High losses may indicate overexposure to market risk.",
                "historical_context": "Similar to March 2020 COVID crash, 2008 financial crisis, or October 1987 crash",
                "mitigation": "Consider reducing delta exposure, adding protective puts, or increasing cash allocation"
            },
            "Market Rally +15%": {
                "purpose": "Tests portfolio performance during strong market upswings",
                "what_happens": "All stock positions gain 15%, call options benefit significantly, puts lose value",
                "key_risks": "Short call positions may face unlimited losses, put options become worthless",
                "what_to_look_for": "Portfolio value increase, P&L gains, call option performance, put option losses",
                "interpretation": "Good performance shows proper delta exposure. Poor performance may indicate over-hedging.",
                "historical_context": "Similar to post-2009 recovery, 2017 Trump rally, or 2020-2021 tech boom",
                "mitigation": "Ensure call options are properly hedged, consider covered call strategies"
            },
            "Volatility Spike": {
                "purpose": "Tests portfolio behavior during extreme market volatility",
                "what_happens": "Market drops 5% but volatility doubles, options experience massive price swings",
                "key_risks": "Vega exposure causes large option price changes, gamma risk increases dramatically",
                "what_to_look_for": "Option price changes, vega impact, gamma risk, portfolio volatility",
                "interpretation": "High vega exposure means large option price swings. Low vega means more stable option prices.",
                "historical_context": "Similar to VIX spikes during 2008, 2011 debt crisis, or 2020 COVID volatility",
                "mitigation": "Consider vega hedging, reduce short volatility positions, add volatility protection"
            },
            "Interest Rate Shock": {
                "purpose": "Tests portfolio sensitivity to rising interest rates",
                "what_happens": "Rates rise 2%, market drops 10%, options lose value due to higher discount rates",
                "key_risks": "Higher rates reduce option values, especially longer-dated options",
                "what_to_look_for": "Option value changes, theta impact, portfolio duration risk",
                "interpretation": "High sensitivity indicates long-dated options exposure. Low sensitivity shows good rate hedging.",
                "historical_context": "Similar to 1994 bond massacre, 2013 taper tantrum, or 2018 rate hikes",
                "mitigation": "Reduce long-dated options, consider rate hedging, focus on shorter-term strategies"
            },
            "Sector Rotation": {
                "purpose": "Tests portfolio resilience during sector-specific market movements",
                "what_happens": "Tech stocks drop 25%, financials gain 15%, creating sector-specific impacts",
                "key_risks": "Concentrated sector exposure amplifies losses, correlation risk becomes apparent",
                "what_to_look_for": "Sector concentration impact, correlation effects, diversification benefits",
                "interpretation": "High sector concentration means larger impacts. Good diversification reduces overall risk.",
                "historical_context": "Similar to 2000 dot-com crash, 2008 financial crisis, or 2022 tech correction",
                "mitigation": "Increase sector diversification, reduce concentration in single sectors"
            }
        }
    
    def apply_scenario_to_positions(self, positions: List[Any], scenario: StressTestScenario) -> List[Any]:
        """Apply scenario changes to positions"""
        modified_positions = []
        
        for pos in positions:
            # Create a copy of the position
            modified_pos = type(pos)(**pos.__dict__)
            
            # Apply market moves
            if pos.ticker in scenario.market_moves:
                move_pct = scenario.market_moves[pos.ticker]
            elif scenario.market_moves:  # If scenario has specific moves, don't apply default
                move_pct = 0.0
            else:
                # Apply default market move based on scenario name
                if "Crash" in scenario.name:
                    move_pct = -0.20
                elif "Rally" in scenario.name:
                    move_pct = 0.15
                elif "Volatility" in scenario.name:
                    move_pct = -0.05
                elif "Interest Rate" in scenario.name:
                    move_pct = -0.10
                else:
                    move_pct = 0.0
            
            # Apply price change
            if move_pct != 0:
                modified_pos.current_price = pos.current_price * (1 + move_pct)
            
            # Apply volatility changes for options
            if pos.position_type in ['call', 'put'] and hasattr(modified_pos, 'implied_volatility'):
                if pos.ticker in scenario.volatility_changes:
                    vol_multiplier = scenario.volatility_changes[pos.ticker]
                elif scenario.volatility_changes:
                    vol_multiplier = 1.0
                else:
                    # Apply default volatility change
                    if "Volatility" in scenario.name:
                        vol_multiplier = 2.0
                    else:
                        vol_multiplier = 1.0
                
                if hasattr(modified_pos, 'implied_volatility') and modified_pos.implied_volatility:
                    modified_pos.implied_volatility = modified_pos.implied_volatility * vol_multiplier
            
            # Apply time decay for options
            if pos.position_type in ['call', 'put'] and scenario.time_decay_days > 0:
                if hasattr(modified_pos, 'expiration_date') and modified_pos.expiration_date:
                    # Reduce time to expiration
                    if hasattr(modified_pos, 'time_to_expiration'):
                        modified_pos.time_to_expiration = max(0, modified_pos.time_to_expiration - scenario.time_decay_days)
            
            # Apply interest rate changes for options
            if pos.position_type in ['call', 'put'] and scenario.interest_rate_change != 0:
                # This would require re-calculating Greeks with new interest rate
                # For now, we'll note this in the scenario details
                pass
            
            modified_positions.append(modified_pos)
        
        return modified_positions
    
    def calculate_scenario_metrics(self, positions: List[Any], scenario_name: str) -> ScenarioResult:
        """Calculate comprehensive metrics for a scenario"""
        # Calculate portfolio value
        portfolio_value = sum(pos.quantity * pos.current_price for pos in positions)
        
        # Calculate P&L
        total_cost = sum(pos.quantity * pos.entry_price for pos in positions)
        pnl = portfolio_value - total_cost
        pnl_pct = (pnl / total_cost * 100) if total_cost != 0 else 0
        
        # Calculate Greeks
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        
        for pos in positions:
            if pos.position_type == 'stock':
                total_delta += pos.quantity
            elif pos.position_type in ['call', 'put']:
                if hasattr(pos, 'delta') and pos.delta:
                    total_delta += pos.delta * pos.quantity
                if hasattr(pos, 'gamma') and pos.gamma:
                    total_gamma += pos.gamma * pos.quantity
                if hasattr(pos, 'theta') and pos.theta:
                    total_theta += pos.theta * pos.quantity
                if hasattr(pos, 'vega') and pos.vega:
                    total_vega += pos.vega * pos.quantity
        
        # Calculate VaR using parametric method
        # This is a simplified calculation - in practice, you'd use historical data
        portfolio_volatility = self._estimate_portfolio_volatility(positions)
        var_95 = portfolio_value * portfolio_volatility * norm.ppf(0.05)
        var_99 = portfolio_value * portfolio_volatility * norm.ppf(0.01)
        cvar_95 = portfolio_value * portfolio_volatility * norm.pdf(norm.ppf(0.05)) / 0.05
        
        # Calculate max drawdown (simplified)
        max_drawdown = abs(var_99)  # Simplified approximation
        
        # Calculate Sharpe ratio (simplified)
        expected_return = pnl_pct / 100
        sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return ScenarioResult(
            scenario_name=scenario_name,
            portfolio_value=portfolio_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            scenario_details={
                'position_count': len(positions),
                'stock_count': len([p for p in positions if p.position_type == 'stock']),
                'option_count': len([p for p in positions if p.position_type in ['call', 'put']]),
                'total_cost': total_cost
            }
        )
    
    def _estimate_portfolio_volatility(self, positions: List[Any]) -> float:
        """Estimate portfolio volatility (simplified)"""
        # This is a simplified calculation
        # In practice, you'd use historical returns and correlation matrix
        total_value = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        if total_value == 0:
            return 0.0
        
        weighted_vol = 0.0
        for pos in positions:
            position_value = abs(pos.quantity * pos.current_price)
            weight = position_value / total_value
            
            # Estimate individual position volatility
            if pos.position_type == 'stock':
                # Assume 20% annual volatility for stocks
                pos_vol = 0.20
            elif pos.position_type in ['call', 'put']:
                # Use implied volatility if available, otherwise estimate
                if hasattr(pos, 'implied_volatility') and pos.implied_volatility:
                    pos_vol = pos.implied_volatility
                else:
                    pos_vol = 0.30  # Default 30% for options
            else:
                pos_vol = 0.0
            
            weighted_vol += pos_vol * weight
        
        return weighted_vol
    
    def run_stress_tests(self, positions: List[Any]) -> List[ScenarioResult]:
        """Run all default stress test scenarios"""
        results = []
        
        for scenario in self.default_scenarios:
            try:
                # Apply scenario to positions
                modified_positions = self.apply_scenario_to_positions(positions, scenario)
                
                # Calculate metrics
                result = self.calculate_scenario_metrics(modified_positions, scenario.name)
                results.append(result)
                
                logger.info(f"Completed stress test: {scenario.name}")
                
            except Exception as e:
                logger.error(f"Error running stress test {scenario.name}: {e}")
                # Create error result
                error_result = ScenarioResult(
                    scenario_name=scenario.name,
                    portfolio_value=0,
                    pnl=0,
                    pnl_pct=0,
                    delta=0,
                    gamma=0,
                    theta=0,
                    vega=0,
                    var_95=0,
                    var_99=0,
                    cvar_95=0,
                    max_drawdown=0,
                    sharpe_ratio=0,
                    scenario_details={'error': str(e)}
                )
                results.append(error_result)
        
        return results
    
    def run_monte_carlo_simulation(self, positions: List[Any], num_simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for portfolio risk analysis"""
        try:
            portfolio_value = sum(pos.quantity * pos.current_price for pos in positions)
            portfolio_vol = self._estimate_portfolio_volatility(positions)
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            returns = np.random.normal(0, portfolio_vol, num_simulations)
            
            # Calculate portfolio values for each simulation
            simulated_values = portfolio_value * (1 + returns)
            
            # Calculate statistics
            var_95 = np.percentile(simulated_values, 5)
            var_99 = np.percentile(simulated_values, 1)
            cvar_95 = np.mean(simulated_values[simulated_values <= var_95])
            cvar_99 = np.mean(simulated_values[simulated_values <= var_99])
            
            # Calculate expected shortfall
            expected_shortfall_95 = portfolio_value - cvar_95
            expected_shortfall_99 = portfolio_value - cvar_99
            
            return {
                'num_simulations': num_simulations,
                'current_value': portfolio_value,
                'mean_simulated_value': np.mean(simulated_values),
                'std_simulated_value': np.std(simulated_values),
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'expected_shortfall_95': expected_shortfall_95,
                'expected_shortfall_99': expected_shortfall_99,
                'probability_of_loss': np.mean(simulated_values < portfolio_value),
                'max_loss': np.min(simulated_values),
                'max_gain': np.max(simulated_values),
                'simulated_values': simulated_values.tolist()[:100]  # Return first 100 for plotting
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {'error': str(e)}
    
    def analyze_position_impact(self, positions: List[Any], new_position: Any) -> Dict[str, Any]:
        """Analyze the impact of adding a new position to the portfolio"""
        try:
            # Calculate current portfolio metrics
            current_metrics = self.calculate_scenario_metrics(positions, "Current Portfolio")
            
            # Add new position
            new_positions = positions + [new_position]
            new_metrics = self.calculate_scenario_metrics(new_positions, "With New Position")
            
            # Calculate impact
            impact = {
                'value_impact': new_metrics.portfolio_value - current_metrics.portfolio_value,
                'pnl_impact': new_metrics.pnl - current_metrics.pnl,
                'pnl_pct_impact': new_metrics.pnl_pct - current_metrics.pnl_pct,
                'delta_impact': new_metrics.delta - current_metrics.delta,
                'gamma_impact': new_metrics.gamma - current_metrics.gamma,
                'theta_impact': new_metrics.theta - current_metrics.theta,
                'vega_impact': new_metrics.vega - current_metrics.vega,
                'var_95_impact': new_metrics.var_95 - current_metrics.var_95,
                'var_99_impact': new_metrics.var_99 - current_metrics.var_99,
                'sharpe_impact': new_metrics.sharpe_ratio - current_metrics.sharpe_ratio
            }
            
            return {
                'current_metrics': current_metrics.__dict__,
                'new_metrics': new_metrics.__dict__,
                'impact': impact
            }
            
        except Exception as e:
            logger.error(f"Error analyzing position impact: {e}")
            return {'error': str(e)}
    
    def create_custom_scenario(self, name: str, description: str, 
                             market_moves: Dict[str, float] = None,
                             volatility_changes: Dict[str, float] = None,
                             time_decay_days: int = 0,
                             interest_rate_change: float = 0.0) -> StressTestScenario:
        """Create a custom stress test scenario"""
        return StressTestScenario(
            name=name,
            description=description,
            market_moves=market_moves or {},
            volatility_changes=volatility_changes or {},
            time_decay_days=time_decay_days,
            interest_rate_change=interest_rate_change
        )

# Example usage
if __name__ == "__main__":
    # This would be used with actual position data
    analyzer = ScenarioAnalyzer()
    print("Scenario Analyzer initialized successfully")
