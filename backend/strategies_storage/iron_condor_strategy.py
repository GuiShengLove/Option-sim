"""
铁鹰式策略 (Iron Condor Strategy)
==================================
同时卖出虚值认购和认沽，买入更虚值的期权保护。
适合低波动、盘整市场。

风险等级: ★★★☆☆ (中等)
收益特征: 有限收益，有限风险，高胜率
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class IronCondorConfig:
    """策略配置"""
    capital: float = 100000           # 投入资金
    put_sell_offset: float = -0.03    # 卖沽距离ATM (-3%)
    put_buy_offset: float = -0.06     # 买沽距离ATM (-6%)
    call_sell_offset: float = 0.03    # 卖购距离ATM (+3%)
    call_buy_offset: float = 0.06     # 买购距离ATM (+6%)
    expiry_days: int = 30             # 目标到期天数
    profit_target: float = 0.5        # 止盈比例 (50%权利金)
    loss_limit: float = 2.0           # 止损比例 (200%权利金)
    min_iv: float = 0.15              # 最低IV要求


class IronCondorStrategy:
    """
    铁鹰式策略
    
    组合结构:
    - 买入虚值认沽 (保护下行)
    - 卖出虚值认沽 (收取权利金)
    - 卖出虚值认购 (收取权利金)
    - 买入虚值认购 (保护上行)
    
    盈利区间: 卖沽行权价 < 标的价格 < 卖购行权价
    """
    
    def __init__(self, config: IronCondorConfig = None):
        self.config = config or IronCondorConfig()
        self.position = None
        self.entry_premium = 0
        self.pnl_history = []
        
    def on_init(self, context: Dict[str, Any]):
        """策略初始化"""
        print(f"[IronCondor] 策略初始化")
        print(f"  - 卖沽偏移: {self.config.put_sell_offset*100:.1f}%")
        print(f"  - 卖购偏移: {self.config.call_sell_offset*100:.1f}%")
        
    def find_legs(self, option_chain: List[Dict], spot_price: float) -> Optional[Dict]:
        """
        寻找四条腿
        
        Returns:
            {
                'buy_put': {...},
                'sell_put': {...},
                'sell_call': {...},
                'buy_call': {...}
            }
        """
        puts = sorted([o for o in option_chain if o['type'] == 'put'], key=lambda x: x['strike'])
        calls = sorted([o for o in option_chain if o['type'] == 'call'], key=lambda x: x['strike'])
        
        if not puts or not calls:
            return None
            
        def find_closest(options, target_strike):
            return min(options, key=lambda x: abs(x['strike'] - target_strike))
            
        legs = {
            'buy_put': find_closest(puts, spot_price * (1 + self.config.put_buy_offset)),
            'sell_put': find_closest(puts, spot_price * (1 + self.config.put_sell_offset)),
            'sell_call': find_closest(calls, spot_price * (1 + self.config.call_sell_offset)),
            'buy_call': find_closest(calls, spot_price * (1 + self.config.call_buy_offset)),
        }
        
        return legs
        
    def calculate_premium(self, legs: Dict) -> float:
        """计算净权利金收入"""
        premium = (
            - legs['buy_put']['ask']   # 支出
            + legs['sell_put']['bid']  # 收入
            + legs['sell_call']['bid'] # 收入
            - legs['buy_call']['ask']  # 支出
        )
        return premium
        
    def on_bar(self, context: Dict[str, Any], data: Dict[str, Any]):
        """每根K线触发"""
        spot_price = data.get('spot_price', 0)
        option_chain = data.get('option_chain', [])
        current_iv = data.get('iv', 0.2)
        
        # 检查IV是否满足条件
        if current_iv < self.config.min_iv:
            return
            
        # 开仓逻辑
        if self.position is None:
            legs = self.find_legs(option_chain, spot_price)
            if legs:
                premium = self.calculate_premium(legs)
                if premium > 0:
                    self._open_position(legs, premium, context)
                    
        # 持仓管理
        else:
            current_value = self._calculate_position_value(option_chain)
            pnl_ratio = (self.entry_premium - current_value) / self.entry_premium
            
            # 止盈
            if pnl_ratio >= self.config.profit_target:
                self._close_position(context, reason="止盈")
                
            # 止损
            elif pnl_ratio <= -self.config.loss_limit:
                self._close_position(context, reason="止损")
                
    def _open_position(self, legs: Dict, premium: float, context: Dict):
        """开仓"""
        self.position = legs
        self.entry_premium = premium
        
        print(f"[IronCondor] 开仓:")
        print(f"  买入认沽 K={legs['buy_put']['strike']}")
        print(f"  卖出认沽 K={legs['sell_put']['strike']}")
        print(f"  卖出认购 K={legs['sell_call']['strike']}")
        print(f"  买入认购 K={legs['buy_call']['strike']}")
        print(f"  净权利金收入: {premium:.4f}")
        
    def _close_position(self, context: Dict, reason: str):
        """平仓"""
        print(f"[IronCondor] 平仓 - 原因: {reason}")
        self.position = None
        
    def _calculate_position_value(self, option_chain: List) -> float:
        """计算持仓当前价值"""
        if not self.position:
            return 0
        # 简化: 返回entry premium
        return self.entry_premium
        
    def get_greeks(self) -> Dict[str, float]:
        """获取组合Greeks"""
        if not self.position:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
        return {
            'delta': 0.05,    # 接近delta中性
            'gamma': -0.08,   # 负gamma
            'theta': 0.03,    # 正theta
            'vega': -0.15     # 负vega
        }
        
    def get_payoff_chart(self, spot_range: List[float]) -> Dict:
        """
        生成收益图
        
        Returns:
            {'x': [prices], 'y': [pnl]}
        """
        if not self.position:
            return {'x': [], 'y': []}
            
        pnl = []
        for s in spot_range:
            # 简化计算
            legs = self.position
            bp_strike = legs['buy_put']['strike']
            sp_strike = legs['sell_put']['strike']
            sc_strike = legs['sell_call']['strike']
            bc_strike = legs['buy_call']['strike']
            
            payoff = self.entry_premium
            
            # 认沽部分
            if s < bp_strike:
                payoff += sp_strike - bp_strike
            elif s < sp_strike:
                payoff += sp_strike - s
                
            # 认购部分
            if s > bc_strike:
                payoff += sc_strike - bc_strike
            elif s > sc_strike:
                payoff -= s - sc_strike
                
            pnl.append(payoff)
            
        return {'x': spot_range, 'y': pnl}


# 策略元数据
STRATEGY_META = {
    'name': 'iron_condor',
    'display_name': '铁鹰式策略',
    'type': 'income',
    'risk_level': 3,
    'description': '同时卖出虚值认购和认沽，四腿策略，适合盘整市场',
    'class': IronCondorStrategy
}
