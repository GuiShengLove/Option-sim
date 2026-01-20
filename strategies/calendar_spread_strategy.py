"""
日历价差 (Calendar Spread)
======================
卖出近月期权，买入远月同行权价期权。利用近月时间价值衰减更快的特点。

风险等级: ★★★☆☆
类型: volatility
最大收益: 近月期权时间价值
最大亏损: 净权利金
盈亏平衡: 复杂，取决于IV变化
理想IV: 期待远月IV上升
时间衰减: 正向 (近月衰减更快)
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """策略配置参数"""
    capital: float = 100000
    position_size: int = 1


class CalendarSpreadStrategy:
    """
    日历价差策略实现
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.legs = []
        
    def on_init(self, context: Dict[str, Any]):
        """策略初始化 - 设置策略腿"""
        # Leg 1: SELL CALL
        self.add_leg(
            type='call',
            action='sell',
            strike_offset=0,
            expiry_days=30,
            quantity=1
        )
        # Leg 2: BUY CALL
        self.add_leg(
            type='call',
            action='buy',
            strike_offset=0,
            expiry_days=60,
            quantity=1
        )
        
    def add_leg(self, type: str, action: str, strike_offset: float, 
                expiry_days: int, quantity: int):
        """添加策略腿"""
        self.legs.append({
            'type': type,
            'action': action,
            'strike_offset': strike_offset,
            'expiry_days': expiry_days,
            'quantity': quantity
        })
        
    def on_bar(self, context: Dict[str, Any], data: Dict[str, Any]):
        """每根K线触发"""
        spot_price = data.get('spot_price', 0)
        option_chain = data.get('option_chain', [])
        
        # TODO: 实现具体交易逻辑
        pass
        
    def get_greeks(self) -> Dict[str, float]:
        """获取组合Greeks"""
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}


# 策略元数据
STRATEGY_META = {
    'name': 'calendar_spread',
    'display_name': '日历价差',
    'type': 'volatility',
    'risk_level': 3,
    'description': '卖出近月期权，买入远月同行权价期权。利用近月时间价值衰减更快的特点。'
}
