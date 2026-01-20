"""
Delta对冲策略 (Delta Hedge Strategy)
=====================================
保持投资组合Delta中性，赚取Gamma收益或Theta收益。
适合做市商或对冲基金的风险管理。

风险等级: ★★★★☆ (较低)
复杂度: 高级策略
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
import pandas as pd
from backend.app.engines.strategy import BaseStrategy, BacktestContext


class DeltaHedgeStrategy(BaseStrategy):
    """
    Delta对冲策略 - 继承自 BaseStrategy
    
    核心思路:
    - 持有期权头寸
    - 用标的资产对冲Delta风险
    - 保持组合Delta接近0
    
    收益来源:
    - 卖方: Theta收益 (时间价值衰减)
    - 买方: Gamma收益 (波动带来的非线性收益)
    """
    
    params = {
        "option_position": {"type": "int", "default": 1, "min": 1, "max": 100},
        "option_type": {"type": "select", "default": "call", "options": ["call", "put"]},
        "hedge_threshold": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.20, "step": 0.01},
        "rebalance_interval": {"type": "int", "default": 1, "min": 1, "max": 5}
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.option_position = None
        self.hedge_position = 0  # 标的对冲持仓
        self.total_delta = 0
        self.rebalance_count = 0
        self.has_opened = False
        self.days_since_rebalance = 0
        
    def on_init(self, context: BacktestContext):
        """策略初始化"""
        print(f"[DeltaHedge] 策略初始化")
        print(f"  - 期权持仓: {self.config.get('option_position', 1)}张")
        print(f"  - 对冲阈值: ±{self.config.get('hedge_threshold', 0.05)}")
        
    def on_bar(self, context: BacktestContext, data: Dict[str, pd.DataFrame]):
        """每根K线触发"""
        options_df = data.get('options')
        underlying_price = data.get('underlying_price', 3.0)
        
        if options_df is None or options_df.empty:
            return
            
        # 首次开仓：买入期权
        if not self.has_opened:
            self._open_option_position(context, options_df, underlying_price)
            return
            
        # 每日检查Delta并再平衡
        self.days_since_rebalance += 1
        rebalance_interval = self.config.get('rebalance_interval', 1)
        
        if self.days_since_rebalance >= rebalance_interval:
            self._rebalance_delta(context, options_df, underlying_price)
            self.days_since_rebalance = 0
            
    def _open_option_position(self, context: BacktestContext, options_df: pd.DataFrame, underlying_price: float):
        """开仓期权头寸"""
        option_type = self.config.get('option_type', 'call')
        type_code = 'C' if option_type == 'call' else 'P'
        
        # 筛选期权
        options = options_df[options_df['type'] == type_code]
        if options.empty:
            return
            
        # 选择ATM期权
        atm_option = options.iloc[(options['strike'] - underlying_price).abs().argmin()]
        
        # 买入期权
        position_size = self.config.get('option_position', 1)
        context.order(atm_option['symbol'], position_size)
        
        self.option_position = atm_option['symbol']
        self.has_opened = True
        
        # 初始Delta对冲
        option_delta = atm_option.get('delta', 0.5)
        self.total_delta = option_delta * position_size * 10000
        
        print(f"[DeltaHedge] 买入{position_size}张 {option_type} @ {atm_option['strike']:.2f}")
        print(f"  - 初始Delta: {self.total_delta:.2f}")
        
    def _rebalance_delta(self, context: BacktestContext, options_df: pd.DataFrame, underlying_price: float):
        """再平衡对冲头寸"""
        if not self.option_position:
            return
            
        # 查找当前持仓的Delta
        if self.option_position in options_df['symbol'].values:
            option_row = options_df[options_df['symbol'] == self.option_position].iloc[0]
            option_delta = option_row.get('delta', 0.5)
        else:
            option_delta = 0.5
            
        position_size = self.config.get('option_position', 1)
        position_delta = option_delta * position_size * 10000
        
        # 当前组合Delta = 期权Delta + 对冲头寸
        self.total_delta = position_delta + self.hedge_position
        
        # 检查是否需要再平衡
        delta_per_share = abs(self.total_delta) / (underlying_price * 10000) if underlying_price > 0 else 0
        threshold = self.config.get('hedge_threshold', 0.05)
        
        if delta_per_share > threshold:
            target_hedge = -position_delta
            hedge_change = target_hedge - self.hedge_position
            
            self.hedge_position = target_hedge
            self.rebalance_count += 1
            
            print(f"[DeltaHedge] 再平衡 #{self.rebalance_count}")
            print(f"  - 新Delta: {self.total_delta:.2f} -> ~0")


# 策略元数据
STRATEGY_META = {
    'name': 'delta_hedge',
    'display_name': 'Delta对冲策略',
    'type': 'hedge',
    'risk_level': 4,
    'description': '保持投资组合Delta中性，对冲方向性风险',
    'class': DeltaHedgeStrategy
}
