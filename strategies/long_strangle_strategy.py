"""
买入宽跨式 (Long Strangle)
=====================
买入虚值认购和认沽，成本比跨式更低，但需要更大波动才能获利。

风险等级: ★★★★☆
类型: volatility
最大收益: 无限
最大亏损: 总权利金
盈亏平衡: 上/下行权价 ± 总权利金
理想IV: 低IV
时间衰减: 负向
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
import pandas as pd
from backend.app.engines.strategy import BaseStrategy, BacktestContext


class LongStrangleStrategy(BaseStrategy):
    """
    买入宽跨式策略 - 继承自 BaseStrategy
    
    同时买入虚值认购和认沽，押注大幅波动。
    收益无限，风险有限（权利金）。
    """
    
    params = {
        "call_strike_offset": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.15, "step": 0.01},
        "put_strike_offset": {"type": "float", "default": -0.05, "min": -0.15, "max": -0.01, "step": 0.01},
        "position_size": {"type": "int", "default": 1, "min": 1, "max": 100}
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_position = False
        self.call_position = None
        self.put_position = None
        
    def on_init(self, context: BacktestContext):
        """策略初始化"""
        print(f"[LongStrangle] 策略初始化")
        print(f"  - Call偏移: {self.config.get('call_strike_offset', 0.05)*100:.1f}% OTM")
        print(f"  - Put偏移: {self.config.get('put_strike_offset', -0.05)*100:.1f}% OTM")
        
    def on_bar(self, context: BacktestContext, data: Dict[str, pd.DataFrame]):
        """每根K线触发"""
        options_df = data.get('options')
        underlying_price = data.get('underlying_price', 3.0)
        
        if options_df is None or options_df.empty:
            return
            
        # 只开仓一次
        if self.has_position:
            return
            
        call_offset = self.config.get('call_strike_offset', 0.05)
        put_offset = self.config.get('put_strike_offset', -0.05)
        
        call_strike_target = underlying_price * (1 + call_offset)
        put_strike_target = underlying_price * (1 + put_offset)
        
        # 筛选期权
        calls = options_df[options_df['type'] == 'C']
        puts = options_df[options_df['type'] == 'P']
        
        if calls.empty or puts.empty:
            return
            
        # 找到最接近目标行权价的期权
        target_call = calls.iloc[(calls['strike'] - call_strike_target).abs().argmin()]
        target_put = puts.iloc[(puts['strike'] - put_strike_target).abs().argmin()]
        
        position_size = self.config.get('position_size', 1)
        
        # 买入认购 (Long Call)
        context.order(target_call['symbol'], position_size)
        # 买入认沽 (Long Put)
        context.order(target_put['symbol'], position_size)
        
        self.has_position = True
        self.call_position = target_call['symbol']
        self.put_position = target_put['symbol']
        
        print(f"[LongStrangle] 开仓: 买Call({target_call['strike']:.2f}) + 买Put({target_put['strike']:.2f})")


# 策略元数据
STRATEGY_META = {
    'name': 'long_strangle',
    'display_name': '买入宽跨式',
    'type': 'volatility',
    'risk_level': 4,
    'description': '买入虚值认购和认沽，成本比跨式更低，但需要更大波动才能获利。'
}
