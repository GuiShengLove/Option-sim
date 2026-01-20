"""
Base Strategy Framework
=======================
Abstract base class for all option strategies.
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
    API exposed to user.
    """
    
    # Define parameters here in subclasses
    params: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        """
        Initialize strategy with parameter values.
        """
        self.config = kwargs
        # Set defaults if not provided
        for k, v in self.params.items():
            if k not in self.config:
                self.config[k] = v.get("default")
        
        # Risk Engine reference (Injected by BacktestEngine)
        self.risk = None
        self.account = None # Account info injection

    @abstractmethod
    def on_init(self, context: BacktestContext):
        """
        Called once at the beginning of the backtest.
        """
        pass
        
    @abstractmethod
    def on_bar(self, context: BacktestContext, data: Dict[str, pd.DataFrame]):
        """
        Called on every time step (e.g. daily).
        Args:
            context: Current account state and order placement
            data: Dictionary containing current market data
                  {'options': pd.DataFrame, 'underlying_price': float}
        """
        pass
        
    def on_order_filled(self, context: BacktestContext, order: Order):
        """
        Optional: Called when an order is executed.
        """
        pass
