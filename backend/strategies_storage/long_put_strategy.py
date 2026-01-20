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

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """策略配置参数"""
    capital: float = 100000
    position_size: int = 1


class LongPutStrategy:
    """
    买入看跌策略实现
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
            strike_offset=0,
            expiry_days=45,
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
    'name': 'long_put',
    'display_name': '买入看跌',
    'type': 'directional',
    'risk_level': 3,
    'description': '最简单的看跌策略，杠杆做空。风险有限。'
}
