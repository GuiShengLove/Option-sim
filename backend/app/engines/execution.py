"""
Execution Engine
================
Handles trade execution realism: Slippage, Liquidity, Assignment.
"""
import numpy as np
import random
from dataclasses import dataclass
from typing import Literal

@dataclass
class ExecutionConfig:
    fill_mode: Literal['MID', 'BID_ASK', 'WORST'] = 'MID'
    slippage_bps: float = 5.0 # Basic slippage in basis points
    liquidity_limit_pct: float = 0.10 # Max 10% of volume
    early_assignment_prob: float = 0.0 # Base probability

class ExecutionEngine:
    def __init__(self, config: ExecutionConfig):
        self.config = config

    def calculate_fill_price(self, order_type: str, side: str, market_data: dict) -> float:
        """
        Calculate fill price based on mode.
        market_data: {'close': float, 'high': float, 'low': float, 'volume': int, 'bid': float, 'ask': float}
        """
        # Fallback if no bid/ask
        mid = market_data.get('close')
        if not mid: return 0.0
        
        bid = market_data.get('bid', mid * 0.999) # Mock spread if missing
        ask = market_data.get('ask', mid * 1.001)
        
        if self.config.fill_mode == 'MID':
            return mid
            
        if self.config.fill_mode == 'BID_ASK':
            base_price = ask if side == 'BUY' else bid
            # Add slippage
            slip = base_price * (self.config.slippage_bps / 10000.0)
            return base_price + slip if side == 'BUY' else base_price - slip
            
        if self.config.fill_mode == 'WORST':
            # Market Maker Killer Mode: Buy at High, Sell at Low (approximation)
            bad_price = market_data.get('high') if side == 'BUY' else market_data.get('low')
            if not bad_price: bad_price = ask * 1.01 if side == 'BUY' else bid * 0.99
            return bad_price
            
        return mid

    def check_liquidity(self, quantity: float, market_volume: float) -> float:
        """
        Check liquidity constraints.
        Returns filled_quantity (partial fill).
        """
        if self.config.liquidity_limit_pct <= 0:
            return quantity
            
        max_qty = market_volume * self.config.liquidity_limit_pct
        if abs(quantity) > max_qty:
            # Partial fill
            return max_qty * (1 if quantity > 0 else -1)
            
        return quantity

    def check_assignment(self, position_record: dict, underlying_price: float) -> bool:
        """
        Check for early assignment on Short American Options.
        Rules:
        1. Deep ITM (Delta ~ 1)
        2. Dividend Risk (Not modeled yet for ETF, but general logic)
        3. Random "Unlucky" assignment
        """
        if position_record['quantity'] >= 0: return False # Longs control exercise
        if position_record['type'] not in ['Call', 'Put']: return False # Stocks
        
        strike = position_record['strike']
        otype = position_record['option_type']
        
        is_itm = (otype == 'Call' and underlying_price > strike) or \
                 (otype == 'Put' and underlying_price < strike)
                 
        if not is_itm: return False
        
        # Simple Model
        prob = self.config.early_assignment_prob
        
        # Increase prob if deep ITM
        moneyness = abs(underlying_price - strike) / strike
        if moneyness > 0.10: # >10% ITM
            prob += 0.05
        if moneyness > 0.20:
            prob += 0.20
            
        return random.random() < prob
