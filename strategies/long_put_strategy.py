"""
买入看跌 (Long Put)
===============
最简单的看跌策略，杠杆做空。风险有限。

风险等级: ★★★☆☆
类型: directional
最大收益: 行权价 - 权利金 (标的跌至0)
最大亏损: 权利金
盈亏平衡: 行权价 - 权利金
理想IV: 低IV
时间衰减: 负向
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
import pandas as pd
from backend.app.engines.strategy import BaseStrategy, BacktestContext


class LongPutStrategy(BaseStrategy):
    """
    买入看跌策略 - 继承自 BaseStrategy
    
    简单的方向性策略，看跌市场时使用。
    风险有限（最大亏损为权利金），收益有限但杠杆高。
    """
    
    params = {
        "strike_offset": {"type": "float", "default": 0.0, "min": -0.10, "max": 0.10, "step": 0.01},
        "position_size": {"type": "int", "default": 1, "min": 1, "max": 100}
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_position = False
        self.put_position = None
        
    def on_init(self, context: BacktestContext):
        """策略初始化"""
        print(f"[LongPut] 策略初始化")
        print(f"  - 行权价偏移: {self.config.get('strike_offset', 0.0)*100:.1f}%")
        print(f"  - 持仓规模: {self.config.get('position_size', 1)}张")
        
    def on_bar(self, context: BacktestContext, data: Dict[str, pd.DataFrame]):
        """每根K线触发"""
        options_df = data.get('options')
        underlying_price = data.get('underlying_price', 3.0)
        
        if options_df is None or options_df.empty:
            return
            
        # 只开仓一次
        if self.has_position:
            return
            
        offset = self.config.get('strike_offset', 0.0)
        target_strike = underlying_price * (1 + offset)
        
        # 筛选认沽期权
        puts = options_df[options_df['type'] == 'P']
        
        if puts.empty:
            return
            
        # 找到最接近目标行权价的期权 (ATM or slightly OTM)
        target_put = puts.iloc[(puts['strike'] - target_strike).abs().argmin()]
        
        # 买入认沽期权 (Long Put = 正数量)
        position_size = self.config.get('position_size', 1)
        context.order(target_put['symbol'], position_size)
        
        self.has_position = True
        self.put_position = target_put['symbol']
        
        premium = target_put['close'] if 'close' in target_put else 0.05
        print(f"[LongPut] 开仓: 买入Put({target_put['strike']:.2f}) @ {premium:.4f}")


# 策略元数据
STRATEGY_META = {
    'name': 'long_put',
    'display_name': '买入看跌',
    'type': 'directional',
    'risk_level': 3,
    'description': '最简单的看跌策略，杠杆做空。风险有限。'
}
