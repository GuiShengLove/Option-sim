"""
Analytics Engine
================
Advanced Analysis & Attribution for Option Strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict

class AttributionEngine:
    """
    Decomposes PnL into Greek Drivers.
    PnL ≈ Delta_PnL + Gamma_PnL + Vega_PnL + Theta_PnL + Pthers
    
    Delta_PnL = Σ (Position_Delta * Change_In_Underlying)
    Gamma_PnL = 0.5 * Σ (Position_Gamma * Change_In_Underlying^2)
    Vega_PnL  = Σ (Position_Vega * Change_In_Vol)
    Theta_PnL = Σ (Position_Theta * Time_Decay)
    """
    
    def analyze(self, backtest_results: pd.DataFrame, underlying_price_history: list) -> Dict[str, float]:
        """
        Perform attribution analysis on a completed backtest.
        
        Args:
            backtest_results: DataFrame containing daily 'total_delta', 'total_gamma', etc.
            underlying_price_history: List or Series of S
            
        Returns:
            Dictionary of attributed PnL values.
        """
        # Note: This requires the backtest to have recorded daily portfolio Greeks.
        # If not present, we perform a naive estimation or return placeholders.
        
        if 'total_delta' not in backtest_results.columns:
            return {
                "delta_pnl": 0.0,
                "gamma_pnl": 0.0,
                "vega_pnl": 0.0,
                "theta_pnl": 0.0,
                "residual": 0.0,
                "explanation": "Detailed Greeks history not found in backtest results."
            }

        # Calculate Move
        S = np.array(underlying_price_history)
        dS = np.diff(S)
        
        # We need alignment. backtest_results usually has length N, dS has length N-1.
        # Assuming index alignment.
        
        total_pnl = backtest_results['equity'].iloc[-1] - backtest_results['equity'].iloc[0]
        
        # Vectorized approx
        # Shift greeks to be "previous day's exposure" applying to "today's move"
        pos_delta = backtest_results['total_delta'].iloc[:-1].values
        pos_gamma = backtest_results['total_gamma'].iloc[:-1].values
        pos_vega = backtest_results['total_vega'].iloc[:-1].values
        pos_theta = backtest_results['total_theta'].iloc[:-1].values
        
        # Truncate dS to match if needed (assuming same freq)
        n = min(len(pos_delta), len(dS))
        
        delta_pnl = np.sum(pos_delta[:n] * dS[:n])
        gamma_pnl = 0.5 * np.sum(pos_gamma[:n] * (dS[:n]**2))
        
        # Vol change is harder to track without separate vol history.
        # Assuming constant vol for MVP or getting dSig if available.
        vega_pnl = 0.0 # Placeholder
        
        theta_pnl = np.sum(pos_theta[:n]) / 365.0 # Daily approximation
        
        explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
        residual = total_pnl - explained
        
        return {
            "total_pnl": float(total_pnl),
            "delta_pnl": float(delta_pnl),
            "gamma_pnl": float(gamma_pnl),
            "vega_pnl": float(vega_pnl),
            "theta_pnl": float(theta_pnl),
            "residual": float(residual),
            "explanation": "Attribution based on localized Greeks approximation."
        }
