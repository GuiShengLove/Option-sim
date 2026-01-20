"""
备兑开仓策略 (Covered Call Strategy)
=====================================
持有标的资产同时卖出虚值看涨期权，赚取时间价值。
适合温和看涨或盘整市场。

风险等级: ★★☆☆☆ (较低)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
import pandas as pd
from backend.app.engines.strategy import BaseStrategy, BacktestContext


class CoveredCallStrategy(BaseStrategy):
    """
    备兑开仓策略 - 继承自 BaseStrategy
    
    入场条件:
    - 持有标的资产
    - IV处于较高水平 (获取更高权利金)
    
    出场条件:
    - 期权到期自动平仓
    - 标的价格接近行权价时可提前回购期权
    """
    
    params = {
        "call_strike_offset": {"type": "float", "default": 0.02, "min": 0.01, "max": 0.10, "step": 0.01},
        "expiry_days": {"type": "int", "default": 30, "min": 7, "max": 90},
        "min_premium": {"type": "float", "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_position = False
        self.call_position = None
        
    def on_init(self, context: BacktestContext):
        """策略初始化"""
        print(f"[CoveredCall] 策略初始化")
        print(f"  - Call偏移: {self.config.get('call_strike_offset', 0.02)*100:.1f}% OTM")
        print(f"  - 到期天数: {self.config.get('expiry_days', 30)}天")
        
    def on_bar(self, context: BacktestContext, data: Dict[str, pd.DataFrame]):
        """每根K线触发"""
        options_df = data.get('options')
        underlying_price = data.get('underlying_price', 3.0)
        
        if options_df is None or options_df.empty:
            return
            
        # 只开仓一次
        if self.has_position:
            return
            
        offset = self.config.get('call_strike_offset', 0.02)
        target_strike = underlying_price * (1 + offset)
        min_premium = self.config.get('min_premium', 0.01)
        
        # 筛选认购期权
        calls = options_df[options_df['type'] == 'C']
        
        if calls.empty:
            return
            
        # 找到最接近目标行权价的期权
        best_call = calls.iloc[(calls['strike'] - target_strike).abs().argmin()]
        
        # 检查权利金是否满足最低要求
        premium = best_call['close'] if 'close' in best_call else 0.05
        if premium < min_premium:
            return
            
        # 卖出认购期权 (Short Call = 负数量)
        context.order(best_call['symbol'], -1)
        
        self.has_position = True
        self.call_position = best_call['symbol']
        print(f"[CoveredCall] 卖出认购期权 @ 行权价{best_call['strike']:.2f}, 收取权利金: {premium:.4f}")


# 策略元数据
STRATEGY_META = {
    'name': 'covered_call',
    'display_name': '备兑开仓策略',
    'type': 'income',
    'risk_level': 2,
    'description': '持有标的资产同时卖出虚值看涨期权，赚取时间价值',
    'class': CoveredCallStrategy
}
