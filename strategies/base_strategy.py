"""
Base Strategy Framework
=======================
Abstract base class for all option strategies.
Designed to be compatible with UI auto-configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

# ============================================================
# Data Structures
# ============================================================

@dataclass
class Position:
    """Represents a single option position."""
    symbol: str
    quantity: float      # Positive for Long, Negative for Short
    entry_price: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price * 10000  # Multiplier usually 10000 for 50ETF
    
    @property
    def pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity * 10000

@dataclass
class Order:
    """Trading Order."""
    symbol: str
    quantity: float      # + for Buy, - for Sell
    price: Optional[float] = None  # Limit price, None for Market
    type: str = "MARKET" # MARKET or LIMIT
    status: str = "PENDING"

@dataclass
class BacktestContext:
    """Context passed to strategy methods."""
    current_date: datetime
    cash: float
    portfolio_value: float
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)
    
    # Place order helper
    def order(self, symbol: str, quantity: float, price: Optional[float] = None):
        """Place an order."""
        self.orders.append(Order(symbol, quantity, price))
        
    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)
    
    def close_all_positions(self):
        """Generate closing orders for all open positions."""
        for symbol, pos in self.positions.items():
            if pos.quantity != 0:
                self.order(symbol, -pos.quantity)

# ============================================================
# Base Strategy Class
# ============================================================

class BaseStrategy(ABC):
    """
    Abstract Base Class for user-defined strategies.
    
    Attributes:
        params (dict): Strategy parameters definition for UI generation.
                       Format:
                       {
                           "name": {"type": "int|float|select", "default": val, ...}
                       }
    """
    
    # Define parameters here in subclasses
    params: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        """
        Initialize strategy with parameter values.
        kwargs should match keys in self.params.
        """
        self.config = kwargs
        # Set defaults if not provided
        for k, v in self.params.items():
            if k not in self.config:
                self.config[k] = v.get("default")
                
    @abstractmethod
    def on_init(self, context: BacktestContext):
        """
        Called once at the beginning of the backtest.
        Use this to setup indicators or static data.
        """
        pass
        
    @abstractmethod
    def on_bar(self, context: BacktestContext, data: Dict[str, pd.DataFrame]):
        """
        Called on every time step (e.g. daily).
        
        Args:
            context: Current account state and order placement
            data: Dictionary containing current market data
                  {'options': pd.DataFrame, 'underlying': float}
        """
        pass
        
    def on_order_filled(self, context: BacktestContext, order: Order):
        """
        Optional: Called when an order is executed.
        """
        pass

# ============================================================
# Example Strategy Implementation
# ============================================================

class DemoStrategy(BaseStrategy):
    """
    Simple Demo Strategy: Long Straddle if IV is low.
    """
    params = {
        "iv_threshold": {"type": "float", "default": 0.15, "min": 0.1, "max": 0.5, "step": 0.01},
        "days_to_expiry": {"type": "int", "default": 30, "min": 10, "max": 90}
    }
    
    def on_init(self, context: BacktestContext):
        print("Strategy initialized with config:", self.config)
        
    def on_bar(self, context: BacktestContext, data: Dict[str, pd.DataFrame]):
        # Example logic - buys ATM options on first bar
        options_df = data.get('options')
        underlying_price = data.get('underlying_price', 3.0)
        
        if options_df is None or options_df.empty:
            return
            
        # Only open once
        if len(context.positions) > 0:
            return
            
        # Find ATM call
        calls = options_df[options_df['type'] == 'C']
        if not calls.empty:
            atm_call = calls.iloc[(calls['strike'] - underlying_price).abs().argmin()]
            context.order(atm_call['symbol'], 1)
            print(f"[DemoStrategy] Buy ATM Call @ {atm_call['strike']:.2f}")

