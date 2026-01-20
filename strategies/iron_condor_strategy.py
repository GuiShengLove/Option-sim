"""
铁鹰式策略 (Iron Condor Strategy)
==================================
同时卖出虚值认购和认沽，买入更虚值的期权保护。
适合低波动、盘整市场。

风险等级: ★★★☆☆ (中等)
收益特征: 有限收益，有限风险，高胜率
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional
import pandas as pd
from backend.app.engines.strategy import BaseStrategy, BacktestContext


class IronCondorStrategy(BaseStrategy):
    """
    铁鹰式策略 - 继承自 BaseStrategy
    
    组合结构:
    - 买入虚值认沽 (保护下行)
    - 卖出虚值认沽 (收取权利金)
    - 卖出虚值认购 (收取权利金)
    - 买入虚值认购 (保护上行)
    
    盈利区间: 卖沽行权价 < 标的价格 < 卖购行权价
    """
    
    params = {
        "put_sell_offset": {"type": "float", "default": -0.03, "min": -0.10, "max": 0.0, "step": 0.01},
        "put_buy_offset": {"type": "float", "default": -0.06, "min": -0.15, "max": -0.01, "step": 0.01},
        "call_sell_offset": {"type": "float", "default": 0.03, "min": 0.0, "max": 0.10, "step": 0.01},
        "call_buy_offset": {"type": "float", "default": 0.06, "min": 0.01, "max": 0.15, "step": 0.01},
        "min_iv": {"type": "float", "default": 0.15, "min": 0.10, "max": 0.50, "step": 0.01}
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_position = False
        self.legs = {}
        self.entry_premium = 0
        
    def on_init(self, context: BacktestContext):
        """策略初始化"""
        print(f"[IronCondor] 策略初始化")
        print(f"  - 卖沽偏移: {self.config.get('put_sell_offset', -0.03)*100:.1f}%")
        print(f"  - 卖购偏移: {self.config.get('call_sell_offset', 0.03)*100:.1f}%")
        
    def on_bar(self, context: BacktestContext, data: Dict[str, pd.DataFrame]):
        """每根K线触发"""
        options_df = data.get('options')
        underlying_price = data.get('underlying_price', 3.0)
        
        if options_df is None or options_df.empty:
            return
            
        # 只开仓一次
        if self.has_position:
            return
            
        # 开仓
        legs = self._find_legs(options_df, underlying_price)
        if legs:
            self._open_position(context, legs, underlying_price)
            
    def _find_legs(self, options_df: pd.DataFrame, underlying_price: float) -> Optional[Dict]:
        """寻找四条腿"""
        put_sell_offset = self.config.get('put_sell_offset', -0.03)
        put_buy_offset = self.config.get('put_buy_offset', -0.06)
        call_sell_offset = self.config.get('call_sell_offset', 0.03)
        call_buy_offset = self.config.get('call_buy_offset', 0.06)
        
        puts = options_df[options_df['type'] == 'P']
        calls = options_df[options_df['type'] == 'C']
        
        if puts.empty or calls.empty:
            return None
            
        def find_closest(df, target_strike):
            return df.iloc[(df['strike'] - target_strike).abs().argmin()]
            
        legs = {
            'buy_put': find_closest(puts, underlying_price * (1 + put_buy_offset)),
            'sell_put': find_closest(puts, underlying_price * (1 + put_sell_offset)),
            'sell_call': find_closest(calls, underlying_price * (1 + call_sell_offset)),
            'buy_call': find_closest(calls, underlying_price * (1 + call_buy_offset)),
        }
        
        return legs
        
    def _open_position(self, context: BacktestContext, legs: Dict, underlying_price: float):
        """开仓"""
        # 买入虚值认沽 (保护)
        context.order(legs['buy_put']['symbol'], 1)
        # 卖出虚值认沽 (收权利金)
        context.order(legs['sell_put']['symbol'], -1)
        # 卖出虚值认购 (收权利金)
        context.order(legs['sell_call']['symbol'], -1)
        # 买入虚值认购 (保护)
        context.order(legs['buy_call']['symbol'], 1)
        
        self.has_position = True
        self.legs = {k: v['symbol'] for k, v in legs.items()}
        
        # 计算净权利金
        buy_put_price = legs['buy_put'].get('close', 0.02)
        sell_put_price = legs['sell_put'].get('close', 0.03)
        sell_call_price = legs['sell_call'].get('close', 0.03)
        buy_call_price = legs['buy_call'].get('close', 0.02)
        
        self.entry_premium = sell_put_price + sell_call_price - buy_put_price - buy_call_price
        
        print(f"[IronCondor] 开仓:")
        print(f"  买入认沽 K={legs['buy_put']['strike']:.2f}")
        print(f"  卖出认沽 K={legs['sell_put']['strike']:.2f}")
        print(f"  卖出认购 K={legs['sell_call']['strike']:.2f}")
        print(f"  买入认购 K={legs['buy_call']['strike']:.2f}")
        print(f"  净权利金收入: {self.entry_premium:.4f}")


# 策略元数据
STRATEGY_META = {
    'name': 'iron_condor',
    'display_name': '铁鹰式策略',
    'type': 'income',
    'risk_level': 3,
    'description': '同时卖出虚值认购和认沽，四腿策略，适合盘整市场',
    'class': IronCondorStrategy
}
