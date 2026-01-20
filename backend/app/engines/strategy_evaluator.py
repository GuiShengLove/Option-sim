"""
Strategy Evaluator
==================
Evaluates option strategy PnL on Monte Carlo simulation paths.

Integrates:
- PricingEngine for option pricing
- StrategyTemplates for strategy definitions
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .pricing import PricingEngine
from .strategy_templates import STRATEGY_TEMPLATES, StrategyTemplate, OptionLeg


@dataclass
class OptionPosition:
    """Single option position for evaluation."""
    type: str  # 'call' or 'put'
    strike: float
    quantity: int  # Positive for long, negative for short
    premium: float  # Premium paid/received per unit
    expiry_days: int


@dataclass
class StrategyEvaluationResult:
    """Results of strategy evaluation."""
    pnl_distribution: List[float]
    avg_pnl: float
    win_rate: float
    max_profit: float
    max_loss: float
    var_95: float
    cvar_95: float
    final_price_distribution: List[float]
    strategy_name: str
    strategy_type: str


class StrategyEvaluator:
    """
    Evaluates option strategies on simulated price paths.
    
    For each path, calculates:
    1. Option prices at entry (using initial spot and IV)
    2. Option prices at expiry (intrinsic value)
    3. Total PnL including premium paid/received
    """
    
    MULTIPLIER = 10000  # 50ETF contract multiplier
    
    def __init__(self, pricing_engine: Optional[PricingEngine] = None):
        self.pricing = pricing_engine or PricingEngine()
    
    def evaluate_strategy(
        self,
        paths: np.ndarray,
        strategy_id: str,
        spot: float,
        expiry_days: int,
        initial_iv: float = 0.20,
        risk_free_rate: float = 0.03,
        custom_strikes: Optional[List[float]] = None
    ) -> StrategyEvaluationResult:
        """
        Evaluate a strategy template on simulation paths.
        
        Args:
            paths: (n_paths, n_steps+1) array of price paths
            strategy_id: Strategy template ID (e.g., 'covered_call')
            spot: Initial spot price
            expiry_days: Days to expiry for the strategy
            initial_iv: Initial implied volatility for pricing
            risk_free_rate: Risk-free rate
            custom_strikes: Optional custom strikes (otherwise computed from spot)
            
        Returns:
            StrategyEvaluationResult with PnL distribution and statistics
        """
        # Get strategy template
        template = STRATEGY_TEMPLATES.get(strategy_id)
        if template is None:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        
        # Build option positions
        positions = self._build_positions(
            template, spot, expiry_days, initial_iv, 
            risk_free_rate, custom_strikes
        )
        
        # Get final prices from paths
        # If paths have more steps than expiry_days, take the price at expiry
        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1
        
        # Index for expiry (0-indexed, considering paths start at index 0)
        expiry_idx = min(expiry_days, n_steps)
        final_prices = paths[:, expiry_idx]
        
        # Calculate PnL for each path
        pnl_list = []
        for path_idx in range(n_paths):
            S_T = final_prices[path_idx]
            path_pnl = self._calculate_strategy_pnl(positions, S_T)
            pnl_list.append(path_pnl)
        
        pnl_array = np.array(pnl_list)
        
        # Calculate statistics
        sorted_pnl = np.sort(pnl_array)
        var_95_idx = int(0.05 * len(sorted_pnl))
        var_95 = sorted_pnl[var_95_idx] if var_95_idx < len(sorted_pnl) else sorted_pnl[0]
        cvar_95 = np.mean(sorted_pnl[:var_95_idx + 1]) if var_95_idx > 0 else var_95
        
        win_rate = np.sum(pnl_array > 0) / len(pnl_array)
        
        return StrategyEvaluationResult(
            pnl_distribution=pnl_list,
            avg_pnl=float(np.mean(pnl_array)),
            win_rate=float(win_rate),
            max_profit=float(np.max(pnl_array)),
            max_loss=float(np.min(pnl_array)),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            final_price_distribution=final_prices.tolist(),
            strategy_name=template.chinese_name,
            strategy_type=template.type.value
        )
    
    def _build_positions(
        self,
        template: StrategyTemplate,
        spot: float,
        expiry_days: int,
        initial_iv: float,
        risk_free_rate: float,
        custom_strikes: Optional[List[float]] = None
    ) -> List[OptionPosition]:
        """
        Build option positions from strategy template.
        """
        positions = []
        T_years = expiry_days / 365.0
        
        # Calculate ATM strike (round to nearest tick)
        atm_strike = round(spot, 2)
        
        # Strike interval (approximate for 50ETF)
        strike_interval = 0.05 if spot < 3.0 else 0.10
        
        for i, leg in enumerate(template.legs):
            # Calculate strike based on offset
            if custom_strikes and i < len(custom_strikes):
                strike = custom_strikes[i]
            else:
                strike = atm_strike + leg.strike_offset * strike_interval
            
            # Calculate option premium using BSM
            is_call = leg.type == 'call'
            premium = self.pricing.price(
                S=spot,
                K=strike,
                T=T_years,
                sigma=initial_iv,
                is_call=is_call
            )
            
            # Determine quantity (positive for buy, negative for sell)
            quantity = leg.quantity if leg.action == 'buy' else -leg.quantity
            
            positions.append(OptionPosition(
                type=leg.type,
                strike=strike,
                quantity=quantity,
                premium=premium,
                expiry_days=expiry_days
            ))
        
        return positions
    
    def _calculate_strategy_pnl(
        self,
        positions: List[OptionPosition],
        S_T: float
    ) -> float:
        """
        Calculate total PnL for a strategy at expiry.
        
        PnL = (Intrinsic Value at Expiry - Premium Paid) * Quantity * Multiplier
        """
        total_pnl = 0.0
        
        for pos in positions:
            # Calculate intrinsic value at expiry
            if pos.type == 'call':
                intrinsic = max(0, S_T - pos.strike)
            else:  # put
                intrinsic = max(0, pos.strike - S_T)
            
            # PnL per unit
            # If long (qty > 0): PnL = (Intrinsic - Premium)
            # If short (qty < 0): PnL = (Premium - Intrinsic)
            # Both simplify to: (Intrinsic - Premium) * Quantity
            pnl = (intrinsic - pos.premium) * pos.quantity * self.MULTIPLIER
            
            total_pnl += pnl
        
        return total_pnl
    
    def evaluate_custom_positions(
        self,
        paths: np.ndarray,
        positions: List[Dict],
        expiry_days: int
    ) -> StrategyEvaluationResult:
        """
        Evaluate custom option positions (not from template).
        
        Args:
            paths: Price paths
            positions: List of position dicts with keys:
                       {'type': 'call'/'put', 'strike': float, 
                        'quantity': int, 'premium': float}
            expiry_days: Days to expiry
            
        Returns:
            StrategyEvaluationResult
        """
        # Convert dicts to OptionPosition objects
        option_positions = [
            OptionPosition(
                type=p['type'],
                strike=p['strike'],
                quantity=p['quantity'],
                premium=p['premium'],
                expiry_days=expiry_days
            )
            for p in positions
        ]
        
        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1
        expiry_idx = min(expiry_days, n_steps)
        final_prices = paths[:, expiry_idx]
        
        pnl_list = []
        for path_idx in range(n_paths):
            S_T = final_prices[path_idx]
            path_pnl = self._calculate_strategy_pnl(option_positions, S_T)
            pnl_list.append(path_pnl)
        
        pnl_array = np.array(pnl_list)
        sorted_pnl = np.sort(pnl_array)
        var_95_idx = int(0.05 * len(sorted_pnl))
        var_95 = sorted_pnl[var_95_idx] if var_95_idx < len(sorted_pnl) else sorted_pnl[0]
        cvar_95 = np.mean(sorted_pnl[:var_95_idx + 1]) if var_95_idx > 0 else var_95
        win_rate = np.sum(pnl_array > 0) / len(pnl_array)
        
        return StrategyEvaluationResult(
            pnl_distribution=pnl_list,
            avg_pnl=float(np.mean(pnl_array)),
            win_rate=float(win_rate),
            max_profit=float(np.max(pnl_array)),
            max_loss=float(np.min(pnl_array)),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            final_price_distribution=final_prices.tolist(),
            strategy_name="Custom Strategy",
            strategy_type="custom"
        )
