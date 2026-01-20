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

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """策略配置参数"""
    capital: float = 100000
    position_size: int = 1


class CollarStrategy:
    """
    领口策略策略实现
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.legs = []
        
    def on_init(self, context: Dict[str, Any]):
        """策略初始化 - 设置策略腿"""
        # Leg 1: BUY PUT
        self.add_leg(
            type='put',
            action='buy',
            strike_offset=-1,
            expiry_days=30,
            quantity=1
        )
        # Leg 2: SELL CALL
        self.add_leg(
            type='call',
            action='sell',
            strike_offset=1,
            expiry_days=30,
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
    'name': 'collar',
    'display_name': '领口策略',
    'type': 'hedge',
    'risk_level': 1,
    'description': '持有标的，买入虚值看跌对冲，卖出虚值看涨降低成本。锁定区间收益。'
}
