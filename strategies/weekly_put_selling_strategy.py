"""
波段卖看跌策略 (Weekly Put Selling Strategy)
==============================================
每周卖出ATM或稍虚值看跌期权，收取权利金。
期权到期前或达到盈利目标时回购平仓，然后开新仓。

这是一个能产生大量交易记录的策略，适合用于回测测试。

风险等级: ★★★☆☆ (中等)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
from backend.app.engines.strategy import BaseStrategy, BacktestContext


class WeeklyPutSellingStrategy(BaseStrategy):
    """
    波段卖看跌策略
    
    交易逻辑:
    1. 每5个交易日开新仓（卖出看跌期权）
    2. 盈利达60%或亏损达100%时平仓
    3. 期权临近到期时（<3天）强制平仓并开新仓
    4. 适应性选择最佳期权：DTE 15-45天，Delta约0.3
    
    这个策略会产生较多交易，适合用于测试回测系统。
    """
    
    params = {
        "put_strike_offset": {
            "type": "float", 
            "default": -0.02,  # 2% OTM (虚值)
            "min": -0.10, 
            "max": 0.0, 
            "step": 0.01,
            "description": "看跌期权行权价偏移（负数为虚值）"
        },
        "target_dte": {
            "type": "int", 
            "default": 30,  # 目标到期天数
            "min": 14, 
            "max": 60,
            "description": "目标到期天数"
        },
        "profit_target": {
            "type": "float", 
            "default": 0.50,  # 50%盈利平仓
            "min": 0.2, 
            "max": 0.9,
            "description": "盈利平仓百分比"
        },
        "stop_loss": {
            "type": "float", 
            "default": 1.0,  # 100%亏损平仓(权利金翻倍)
            "min": 0.5, 
            "max": 2.0,
            "description": "止损百分比"
        },
        "position_size": {
            "type": "int", 
            "default": 20,  # 增加到20张以便测试保证金
            "min": 1, 
            "max": 100,
            "description": "每次交易张数（增大可提高保证金占用率）"
        },
        "reentry_days": {
            "type": "int",
            "default": 5,  # 平仓后等待5天再开仓
            "min": 1,
            "max": 20,
            "description": "重新开仓间隔天数"
        }
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_position: Optional[Dict] = None
        self.days_since_close = 999  # 距上次平仓天数
        self.trade_count = 0
        self.total_premium_collected = 0.0
        self.total_pnl = 0.0
        
    def on_init(self, context: BacktestContext):
        """策略初始化"""
        offset = self.config.get('put_strike_offset', -0.02)
        dte = self.config.get('target_dte', 30)
        profit = self.config.get('profit_target', 0.5)
        stop = self.config.get('stop_loss', 1.0)
        size = self.config.get('position_size', 2)
        
        print(f"═══════════════════════════════════════════════════")
        print(f"【波段卖看跌策略】初始化")
        print(f"═══════════════════════════════════════════════════")
        print(f"  • 行权价偏移: {offset*100:.1f}% ({'虚值' if offset < 0 else 'ATM/实值'})")
        print(f"  • 目标DTE: {dte}天")
        print(f"  • 盈利目标: {profit*100:.0f}%")
        print(f"  • 止损线: {stop*100:.0f}%")
        print(f"  • 每笔张数: {size}张 (预计保证金占用率: ~{size * 0.35:.1f}%)")
        print(f"═══════════════════════════════════════════════════")
        
    def on_bar(self, context: BacktestContext, data: Dict[str, pd.DataFrame]):
        """每根K线触发"""
        options_df = data.get('options')
        underlying_price = data.get('underlying_price', 3.0)
        current_date = context.current_date
        
        if options_df is None or options_df.empty:
            return
            
        self.days_since_close += 1
        
        # ========== 持仓管理 ==========
        if self.current_position is not None:
            self._manage_position(context, options_df, underlying_price, current_date)
            
        # ========== 开新仓 ==========
        if self.current_position is None:
            reentry_days = self.config.get('reentry_days', 5)
            if self.days_since_close >= reentry_days:
                self._open_new_position(context, options_df, underlying_price, current_date)
    
    def _manage_position(self, context: BacktestContext, options_df: pd.DataFrame, 
                         underlying_price: float, current_date: datetime):
        """管理现有持仓"""
        symbol = self.current_position['symbol']
        entry_price = self.current_position['entry_price']
        entry_date = self.current_position['entry_date']
        quantity = self.current_position['quantity']
        expiry = self.current_position.get('expiry')
        
        # 找到当前期权价格
        option_row = options_df[options_df['symbol'] == symbol]
        if option_row.empty:
            # 期权可能已到期，强制平仓
            print(f"[{current_date.strftime('%Y-%m-%d')}] ⚡ 期权到期/不存在，自动平仓")
            self._close_position(context, 0, "到期清算")
            return
            
        current_price = option_row.iloc[0]['close']
        
        # 计算盈亏（卖出期权盈利 = 入场价 - 当前价）
        pnl_per_contract = (entry_price - current_price) * 10000  # 乘数
        pnl_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0
        
        # 检查到期时间
        days_held = (current_date - entry_date).days
        dte_now = None
        if expiry:
            dte_now = (expiry - current_date).days
            if dte_now <= 3:
                print(f"[{current_date.strftime('%Y-%m-%d')}] ⏰ 临近到期({dte_now}天), 平仓避险")
                context.order(symbol, -quantity)  # 买回平仓
                self._close_position(context, current_price, "临近到期")
                return
        
        # 检查盈利目标
        profit_target = self.config.get('profit_target', 0.5)
        if pnl_pct >= profit_target:
            print(f"[{current_date.strftime('%Y-%m-%d')}] 🎯 达到盈利目标 {pnl_pct*100:.1f}%")
            context.order(symbol, -quantity)  # 买回平仓
            self._close_position(context, current_price, "盈利平仓")
            return
            
        # 检查止损
        stop_loss = self.config.get('stop_loss', 1.0)
        if pnl_pct <= -stop_loss:
            print(f"[{current_date.strftime('%Y-%m-%d')}] 🛑 触发止损 {pnl_pct*100:.1f}%")
            context.order(symbol, -quantity)
            self._close_position(context, current_price, "止损平仓")
            return
            
    def _close_position(self, context: BacktestContext, exit_price: float, reason: str):
        """平仓处理"""
        if self.current_position is None:
            return
            
        entry_price = self.current_position['entry_price']
        quantity = self.current_position['quantity']
        symbol = self.current_position['symbol']
        
        # 计算本次交易盈亏（卖出期权: 盈亏 = (入场价 - 出场价) * 数量 * 乘数）
        pnl = (entry_price - exit_price) * abs(quantity) * 10000
        self.total_pnl += pnl
        self.trade_count += 1
        
        print(f"    ├─ 平仓价: {exit_price:.4f} (入场: {entry_price:.4f})")
        print(f"    ├─ 本次盈亏: ¥{pnl:+,.0f}")
        print(f"    └─ 累计交易: {self.trade_count}笔, 累计盈亏: ¥{self.total_pnl:+,.0f}")
        
        self.current_position = None
        self.days_since_close = 0
        
    def _open_new_position(self, context: BacktestContext, options_df: pd.DataFrame,
                           underlying_price: float, current_date: datetime):
        """开新仓"""
        offset = self.config.get('put_strike_offset', -0.02)
        target_dte = self.config.get('target_dte', 30)
        position_size = self.config.get('position_size', 2)
        
        # 计算目标行权价
        target_strike = underlying_price * (1 + offset)
        
        # 筛选看跌期权
        puts = options_df[options_df['type'] == 'P'].copy()
        if puts.empty:
            return
            
        # 计算DTE
        if 'expiry_date' in puts.columns:
            puts['dte'] = puts['expiry_date'].apply(
                lambda x: (pd.Timestamp(x) - pd.Timestamp(current_date)).days
            )
            # 筛选DTE在合适范围内的期权
            puts = puts[(puts['dte'] >= 10) & (puts['dte'] <= 60)]
            if puts.empty:
                return
            
            # 找最接近目标DTE的期权
            puts['dte_diff'] = abs(puts['dte'] - target_dte)
            puts = puts.nsmallest(10, 'dte_diff')  # 取DTE最接近的10个
            
        # 在DTE合适的期权中，找最接近目标行权价的
        if 'strike' not in puts.columns:
            return
            
        best_put = puts.iloc[(puts['strike'] - target_strike).abs().argmin()]
        
        symbol = best_put['symbol']
        strike = best_put['strike']
        premium = best_put['close'] if 'close' in best_put else 0.05
        expiry = pd.Timestamp(best_put['expiry_date']) if 'expiry_date' in best_put else None
        dte = best_put.get('dte', target_dte)
        
        # 检查权利金是否足够（过低的权利金不值得交易）
        if premium < 0.005:
            return
            
        # 卖出看跌期权（Short Put = 负数量）
        context.order(symbol, -position_size)
        
        self.current_position = {
            'symbol': symbol,
            'quantity': -position_size,  # 负数表示空头
            'entry_price': premium,
            'entry_date': current_date,
            'strike': strike,
            'expiry': expiry
        }
        
        self.total_premium_collected += premium * position_size * 10000
        
        print(f"[{current_date.strftime('%Y-%m-%d')}] 📝 开仓: 卖出{position_size}张看跌期权")
        print(f"    ├─ 合约: {symbol}")
        print(f"    ├─ 行权价: {strike:.3f} (标的: {underlying_price:.3f})")
        print(f"    ├─ 权利金: {premium:.4f} (收取: ¥{premium*position_size*10000:,.0f})")
        print(f"    ├─ DTE: {dte}天")
        print(f"    └─ 累计收取权利金: ¥{self.total_premium_collected:,.0f}")


# 策略元数据
STRATEGY_META = {
    'name': 'weekly_put_selling',
    'display_name': '波段卖看跌策略',
    'type': 'income',
    'risk_level': 3,
    'description': '周期性卖出看跌期权收取权利金，适合温和看涨市场。产生较多交易记录，适合测试。',
    'class': WeeklyPutSellingStrategy
}
