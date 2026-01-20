"""
Delta对冲策略 (Delta Hedge Strategy)
=====================================
保持投资组合Delta中性，赚取Gamma收益或Theta收益。
适合做市商或对冲基金的风险管理。

风险等级: ★★★★☆ (较高)
复杂度: 高级策略
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import math


@dataclass  
class DeltaHedgeConfig:
    """策略配置"""
    option_position: int = 100        # 期权持仓张数
    option_type: str = 'call'         # 期权类型
    hedge_threshold: float = 0.05     # 对冲阈值 (Delta偏离)
    rebalance_interval: int = 1       # 再平衡频率 (天)


class DeltaHedgeStrategy:
    """
    Delta对冲策略
    
    核心思路:
    - 持有期权头寸
    - 用标的资产对冲Delta风险
    - 保持组合Delta接近0
    
    收益来源:
    - 卖方: Theta收益 (时间价值衰减)
    - 买方: Gamma收益 (波动带来的非线性收益)
    """
    
    def __init__(self, config: DeltaHedgeConfig = None):
        self.config = config or DeltaHedgeConfig()
        self.option_position = None
        self.hedge_position = 0  # 标的对冲持仓
        self.total_delta = 0
        self.rebalance_count = 0
        self.pnl_from_hedging = 0
        
    def on_init(self, context: Dict[str, Any]):
        """策略初始化"""
        print(f"[DeltaHedge] 策略初始化")
        print(f"  - 期权持仓: {self.config.option_position}张")
        print(f"  - 对冲阈值: ±{self.config.hedge_threshold}")
        
    def calculate_delta(self, option: Dict, spot_price: float) -> float:
        """
        使用BSM计算期权Delta
        
        简化版: 使用期权数据中的delta
        """
        delta = option.get('delta', 0.5)
        return delta
        
    def on_bar(self, context: Dict[str, Any], data: Dict[str, Any]):
        """每根K线触发"""
        spot_price = data.get('spot_price', 0)
        option = data.get('current_option', {})
        
        if not option or spot_price <= 0:
            return
            
        # 计算期权Delta
        option_delta = self.calculate_delta(option, spot_price)
        
        # 期权头寸的总Delta
        position_delta = option_delta * self.config.option_position * 10000
        
        # 当前组合Delta = 期权Delta + 对冲头寸
        self.total_delta = position_delta + self.hedge_position
        
        # 检查是否需要再平衡
        delta_per_share = self.total_delta / (spot_price * 10000) if spot_price > 0 else 0
        
        if abs(delta_per_share) > self.config.hedge_threshold:
            self._rebalance(spot_price, position_delta, context)
            
    def _rebalance(self, spot_price: float, position_delta: float, context: Dict):
        """再平衡对冲头寸"""
        # 需要对冲的Delta
        target_hedge = -position_delta
        hedge_change = target_hedge - self.hedge_position
        
        # 交易成本 (简化)
        transaction_cost = abs(hedge_change) * spot_price * 0.0001
        
        self.hedge_position = target_hedge
        self.rebalance_count += 1
        self.pnl_from_hedging -= transaction_cost
        
        print(f"[DeltaHedge] 再平衡 #{self.rebalance_count}")
        print(f"  - 标的持仓: {self.hedge_position:,.0f}股")
        print(f"  - 交易成本: {transaction_cost:.2f}")
        print(f"  - 新Delta: {self.total_delta:.2f}")
        
    def get_greeks(self) -> Dict[str, float]:
        """获取组合Greeks"""
        return {
            'delta': self.total_delta / 10000,
            'gamma': 0.02 * self.config.option_position,
            'theta': -0.01 * self.config.option_position,
            'vega': 0.1 * self.config.option_position
        }
        
    def get_performance(self) -> Dict[str, Any]:
        """获取策略表现"""
        return {
            'rebalance_count': self.rebalance_count,
            'hedging_cost': self.pnl_from_hedging,
            'current_delta': self.total_delta,
            'hedge_position': self.hedge_position
        }
        
    def get_summary(self) -> str:
        """策略摘要"""
        return f"""
Delta对冲策略
=============
期权持仓: {self.config.option_position}张 {self.config.option_type}
对冲持仓: {self.hedge_position:,.0f}股
当前Delta: {self.total_delta:.2f}
再平衡次数: {self.rebalance_count}
对冲成本: {self.pnl_from_hedging:.2f}
"""


# 策略元数据
STRATEGY_META = {
    'name': 'delta_hedge',
    'display_name': 'Delta对冲策略',
    'type': 'hedge',
    'risk_level': 4,
    'description': '保持投资组合Delta中性，对冲方向性风险',
    'class': DeltaHedgeStrategy
}
