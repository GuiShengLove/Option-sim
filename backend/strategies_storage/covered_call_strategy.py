"""
备兑开仓策略 (Covered Call Strategy)
=====================================
持有标的资产同时卖出虚值看涨期权，赚取时间价值。
适合温和看涨或盘整市场。

风险等级: ★★☆☆☆ (较低)
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StrategyConfig:
    """策略配置参数"""
    underlying_qty: int = 10000      # 标的持仓数量
    call_strike_offset: float = 0.02  # 虚值程度 (2% OTM)
    expiry_days: int = 30             # 到期天数
    roll_days: int = 7                # 移仓天数 (到期前N天)
    min_premium: float = 0.01         # 最低权利金要求


class CoveredCallStrategy:
    """
    备兑开仓策略
    
    入场条件:
    - 持有标的资产
    - IV处于较高水平 (获取更高权利金)
    
    出场条件:
    - 期权到期自动平仓
    - 标的价格接近行权价时可提前回购期权
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.positions = []
        self.pnl_history = []
        
    def on_init(self, context: Dict[str, Any]):
        """策略初始化"""
        print(f"[CoveredCall] 策略初始化 - 到期天数: {self.config.expiry_days}天")
        context['underlying_position'] = self.config.underlying_qty
        
    def select_option(self, option_chain: List[Dict], spot_price: float) -> Dict:
        """
        选择合适的期权合约
        
        Args:
            option_chain: 期权链数据
            spot_price: 标的现价
            
        Returns:
            选中的期权合约
        """
        target_strike = spot_price * (1 + self.config.call_strike_offset)
        
        # 筛选认购期权
        calls = [opt for opt in option_chain if opt['type'] == 'call']
        
        # 找到最接近目标行权价的期权
        best_option = min(
            calls, 
            key=lambda x: abs(x['strike'] - target_strike)
        )
        
        return best_option
        
    def on_bar(self, context: Dict[str, Any], data: Dict[str, Any]):
        """
        每根K线触发
        
        Args:
            context: 策略上下文
            data: 市场数据
        """
        spot_price = data.get('spot_price', 0)
        option_chain = data.get('option_chain', [])
        current_date = data.get('date', datetime.now())
        
        # 检查是否需要开仓
        if not self.positions:
            option = self.select_option(option_chain, spot_price)
            if option and option['ask'] >= self.config.min_premium:
                self._open_position(option, context)
                
        # 检查是否需要移仓
        for pos in self.positions:
            days_to_expiry = (pos['expiry'] - current_date).days
            if days_to_expiry <= self.config.roll_days:
                self._roll_position(pos, option_chain, spot_price, context)
                
    def _open_position(self, option: Dict, context: Dict):
        """开仓"""
        contracts = self.config.underlying_qty // 10000  # 1张 = 10000份
        premium = option['bid'] * contracts * 10000
        
        position = {
            'id': option['id'],
            'strike': option['strike'],
            'expiry': option['expiry'],
            'premium': premium,
            'contracts': contracts,
            'open_date': context.get('current_date')
        }
        
        self.positions.append(position)
        print(f"[CoveredCall] 卖出 {contracts}张 认购期权 @ 行权价{option['strike']}, 收取权利金: {premium:.2f}")
        
    def _roll_position(self, old_pos: Dict, option_chain: List, spot_price: float, context: Dict):
        """移仓到下月"""
        # 买回旧期权
        # 卖出新期权
        print(f"[CoveredCall] 移仓: 行权价 {old_pos['strike']} -> 新合约")
        self.positions.remove(old_pos)
        
    def get_greeks(self) -> Dict[str, float]:
        """获取组合Greeks"""
        return {
            'delta': 1.0 - 0.3 * len(self.positions),  # 标的delta减去期权delta
            'gamma': -0.05 * len(self.positions),
            'theta': 0.02 * len(self.positions),        # 正theta收益
            'vega': -0.1 * len(self.positions)
        }
        
    def get_summary(self) -> str:
        """策略摘要"""
        return f"""
备兑开仓策略 (Covered Call)
===========================
持仓: {self.config.underlying_qty}股标的 + 卖出{len(self.positions)}张认购
行权价偏移: {self.config.call_strike_offset*100:.1f}% OTM
到期天数: {self.config.expiry_days}天
移仓阈值: {self.config.roll_days}天
"""


# 策略元数据
STRATEGY_META = {
    'name': 'covered_call',
    'display_name': '备兑开仓策略',
    'type': 'income',
    'risk_level': 2,
    'description': '持有标的资产同时卖出虚值看涨期权，赚取时间价值',
    'class': CoveredCallStrategy
}
