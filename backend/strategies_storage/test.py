"""
test.py
自定义策略
"""

from typing import Dict, Any


class MyStrategy:
    """自定义策略类"""
    
    def __init__(self):
        self.positions = []
        
    def on_init(self, context: Dict[str, Any]):
        """策略初始化"""
        print("策略初始化完成")
        
    def on_bar(self, context: Dict[str, Any], data: Dict[str, Any]):
        """每根K线触发"""
        spot_price = data.get('spot_price', 0)
        # TODO: 添加交易逻辑
        pass
        
    def get_greeks(self) -> Dict[str, float]:
        """获取组合Greeks"""
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}


STRATEGY_META = {
    'name': 'test',
    'display_name': '自定义策略',
    'type': 'custom',
    'risk_level': 3,
    'description': '自定义策略描述'
}
