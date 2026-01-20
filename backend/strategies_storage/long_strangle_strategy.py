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

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """策略配置参数"""
    capital: float = 100000
    position_size: int = 1


class LongStrangleStrategy:
    """
    买入宽跨式策略实现
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.legs = []
        
    def on_init(self, context: Dict[str, Any]):
        """策略初始化 - 设置策略腿"""
        # Leg 1: BUY CALL
        self.add_leg(
            type='call',
            action='buy',
            strike_offset=2,
            expiry_days=30,
            quantity=1
        )
        # Leg 2: BUY PUT
        self.add_leg(
            type='put',
            action='buy',
            strike_offset=-2,
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
    'name': 'long_strangle',
    'display_name': '买入宽跨式',
    'type': 'volatility',
    'risk_level': 4,
    'description': '买入虚值认购和认沽，成本比跨式更低，但需要更大波动才能获利。'
}
