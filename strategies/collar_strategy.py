"""
领口策略 (Collar)
=============
持有标的，买入虚值看跌对冲，卖出虚值看涨降低成本。锁定区间收益。

风险等级: ★☆☆☆☆
类型: hedge
最大收益: 高行权价 - 现价 + 净权利金
最大亏损: 现价 - 低行权价 + 净权利金
盈亏平衡: 现价 + 净权利金
理想IV: 高IV (卖的比买的贵)
时间衰减: 中性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
import pandas as pd
from backend.app.engines.strategy import BaseStrategy, BacktestContext


class CollarStrategy(BaseStrategy):
    """
    领口策略 - 继承自 BaseStrategy
    
    持有标的，买入虚值看跌对冲，卖出虚值看涨降低成本。
    """
    
    params = {
        "put_strike_offset": {"type": "float", "default": -0.03, "min": -0.10, "max": 0.0, "step": 0.01},
        "call_strike_offset": {"type": "float", "default": 0.03, "min": 0.0, "max": 0.10, "step": 0.01},
        "position_size": {"type": "int", "default": 1, "min": 1, "max": 100}
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.legs = []
        self.has_opened = False
        
    def on_init(self, context: BacktestContext):
        """策略初始化"""
        print(f"[Collar] 策略初始化")
        print(f"  - Put偏移: {self.config.get('put_strike_offset', -0.03)*100:.1f}%")
        print(f"  - Call偏移: {self.config.get('call_strike_offset', 0.03)*100:.1f}%")
        
    def on_bar(self, context: BacktestContext, data: Dict[str, pd.DataFrame]):
        """每根K线触发"""
        options_df = data.get('options')
        underlying_price = data.get('underlying_price', 3.0)
        
        if options_df is None or options_df.empty:
            return
            
        # 只开仓一次
        if self.has_opened:
            return
            
        put_offset = self.config.get('put_strike_offset', -0.03)
        call_offset = self.config.get('call_strike_offset', 0.03)
        
        put_strike_target = underlying_price * (1 + put_offset)
        call_strike_target = underlying_price * (1 + call_offset)
        
        # 筛选认沽和认购
        puts = options_df[options_df['type'] == 'P']
        calls = options_df[options_df['type'] == 'C']
        
        if puts.empty or calls.empty:
            return
            
        # 找到最接近目标行权价的期权
        target_put = puts.iloc[(puts['strike'] - put_strike_target).abs().argmin()]
        target_call = calls.iloc[(calls['strike'] - call_strike_target).abs().argmin()]
        
        # 买入看跌 (Long Put = 买入 = 正数量)
        context.order(target_put['symbol'], 1)
        # 卖出看涨 (Short Call = 卖出 = 负数量)
        context.order(target_call['symbol'], -1)
        
        self.has_opened = True
        print(f"[Collar] 开仓: 买入Put({target_put['strike']:.2f}) & 卖出Call({target_call['strike']:.2f})")


# 策略元数据
STRATEGY_META = {
    'name': 'collar',
    'display_name': '领口策略',
    'type': 'hedge',
    'risk_level': 1,
    'description': '持有标的，买入虚值看跌对冲，卖出虚值看涨降低成本。锁定区间收益。'
}
