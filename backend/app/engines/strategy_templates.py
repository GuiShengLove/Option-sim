"""
Option Strategy Templates
=========================
Professional option strategy templates for common trading patterns.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional
from enum import Enum


class StrategyType(Enum):
    """Strategy classification."""
    INCOME = "income"           # 收益型策略
    DIRECTIONAL = "directional" # 方向性策略
    VOLATILITY = "volatility"   # 波动率策略
    HEDGE = "hedge"             # 对冲策略
    ARBITRAGE = "arbitrage"     # 套利策略


class MarketOutlook(Enum):
    """Market view classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"


@dataclass
class OptionLeg:
    """Single leg of an option strategy."""
    type: Literal['call', 'put']
    action: Literal['buy', 'sell']
    strike_offset: float  # Relative to ATM (e.g., 0 = ATM, -1 = 1 strike OTM for put)
    expiry_offset: int    # Days from now
    quantity: int = 1


@dataclass
class StrategyTemplate:
    """Complete strategy template."""
    name: str
    chinese_name: str
    type: StrategyType
    outlook: List[MarketOutlook]
    legs: List[OptionLeg]
    max_profit: str        # Description
    max_loss: str          # Description
    breakeven: str         # Description
    ideal_iv: str          # High/Low/Neutral
    time_decay: str        # Positive/Negative/Neutral
    description: str
    risk_level: int        # 1-5


# ============================================================
# Strategy Template Library
# ============================================================

STRATEGY_TEMPLATES = {
    # === Income Strategies ===
    "covered_call": StrategyTemplate(
        name="Covered Call",
        chinese_name="备兑开仓",
        type=StrategyType.INCOME,
        outlook=[MarketOutlook.NEUTRAL, MarketOutlook.BULLISH],
        legs=[
            OptionLeg(type='call', action='sell', strike_offset=1, expiry_offset=30),
        ],
        max_profit="期权权利金 + (行权价 - 现价)",
        max_loss="标的跌至0 - 权利金",
        breakeven="现价 - 权利金",
        ideal_iv="高IV (获取更高权利金)",
        time_decay="正向 (卖方获益)",
        description="持有标的资产同时卖出虚值看涨期权，赚取时间价值。适合温和看涨或盘整市场。",
        risk_level=2
    ),
    
    "cash_secured_put": StrategyTemplate(
        name="Cash-Secured Put",
        chinese_name="现金担保卖沽",
        type=StrategyType.INCOME,
        outlook=[MarketOutlook.NEUTRAL, MarketOutlook.BULLISH],
        legs=[
            OptionLeg(type='put', action='sell', strike_offset=-1, expiry_offset=30),
        ],
        max_profit="权利金收入",
        max_loss="行权价 - 权利金 (标的跌至0)",
        breakeven="行权价 - 权利金",
        ideal_iv="高IV",
        time_decay="正向",
        description="卖出看跌期权，准备以较低价格买入标的。适合想抄底但不急于入场的投资者。",
        risk_level=2
    ),
    
    # === Directional Strategies ===
    "long_call": StrategyTemplate(
        name="Long Call",
        chinese_name="买入看涨",
        type=StrategyType.DIRECTIONAL,
        outlook=[MarketOutlook.BULLISH],
        legs=[
            OptionLeg(type='call', action='buy', strike_offset=0, expiry_offset=45),
        ],
        max_profit="无限 (标的无限上涨)",
        max_loss="权利金",
        breakeven="行权价 + 权利金",
        ideal_iv="低IV (买入便宜)",
        time_decay="负向",
        description="最简单的看涨策略，杠杆做多。风险有限，收益无限。",
        risk_level=3
    ),
    
    "long_put": StrategyTemplate(
        name="Long Put",
        chinese_name="买入看跌",
        type=StrategyType.DIRECTIONAL,
        outlook=[MarketOutlook.BEARISH],
        legs=[
            OptionLeg(type='put', action='buy', strike_offset=0, expiry_offset=45),
        ],
        max_profit="行权价 - 权利金 (标的跌至0)",
        max_loss="权利金",
        breakeven="行权价 - 权利金",
        ideal_iv="低IV",
        time_decay="负向",
        description="最简单的看跌策略，杠杆做空。风险有限。",
        risk_level=3
    ),
    
    "bull_call_spread": StrategyTemplate(
        name="Bull Call Spread",
        chinese_name="牛市看涨价差",
        type=StrategyType.DIRECTIONAL,
        outlook=[MarketOutlook.BULLISH],
        legs=[
            OptionLeg(type='call', action='buy', strike_offset=0, expiry_offset=30),
            OptionLeg(type='call', action='sell', strike_offset=2, expiry_offset=30),
        ],
        max_profit="高行权价 - 低行权价 - 净权利金",
        max_loss="净权利金",
        breakeven="低行权价 + 净权利金",
        ideal_iv="中等",
        time_decay="接近平值时负向",
        description="买入较低行权价认购，卖出较高行权价认购。成本更低的看涨策略。",
        risk_level=2
    ),
    
    "bear_put_spread": StrategyTemplate(
        name="Bear Put Spread",
        chinese_name="熊市看跌价差",
        type=StrategyType.DIRECTIONAL,
        outlook=[MarketOutlook.BEARISH],
        legs=[
            OptionLeg(type='put', action='buy', strike_offset=0, expiry_offset=30),
            OptionLeg(type='put', action='sell', strike_offset=-2, expiry_offset=30),
        ],
        max_profit="高行权价 - 低行权价 - 净权利金",
        max_loss="净权利金",
        breakeven="高行权价 - 净权利金",
        ideal_iv="中等",
        time_decay="接近平值时负向",
        description="买入较高行权价认沽，卖出较低行权价认沽。成本更低的看跌策略。",
        risk_level=2
    ),
    
    # === Volatility Strategies ===
    "long_straddle": StrategyTemplate(
        name="Long Straddle",
        chinese_name="买入跨式",
        type=StrategyType.VOLATILITY,
        outlook=[MarketOutlook.HIGH_VOL],
        legs=[
            OptionLeg(type='call', action='buy', strike_offset=0, expiry_offset=30),
            OptionLeg(type='put', action='buy', strike_offset=0, expiry_offset=30),
        ],
        max_profit="无限 (大幅波动)",
        max_loss="双倍权利金 (标的不动)",
        breakeven="行权价 ± 总权利金",
        ideal_iv="低IV买入，期待IV上升",
        time_decay="负向",
        description="同时买入平值认购和认沽，押注大幅波动。不判断方向，只判断波动大小。",
        risk_level=4
    ),
    
    "long_strangle": StrategyTemplate(
        name="Long Strangle",
        chinese_name="买入宽跨式",
        type=StrategyType.VOLATILITY,
        outlook=[MarketOutlook.HIGH_VOL],
        legs=[
            OptionLeg(type='call', action='buy', strike_offset=2, expiry_offset=30),
            OptionLeg(type='put', action='buy', strike_offset=-2, expiry_offset=30),
        ],
        max_profit="无限",
        max_loss="总权利金",
        breakeven="上/下行权价 ± 总权利金",
        ideal_iv="低IV",
        time_decay="负向",
        description="买入虚值认购和认沽，成本比跨式更低，但需要更大波动才能获利。",
        risk_level=4
    ),
    
    "short_straddle": StrategyTemplate(
        name="Short Straddle",
        chinese_name="卖出跨式",
        type=StrategyType.INCOME,
        outlook=[MarketOutlook.LOW_VOL, MarketOutlook.NEUTRAL],
        legs=[
            OptionLeg(type='call', action='sell', strike_offset=0, expiry_offset=30),
            OptionLeg(type='put', action='sell', strike_offset=0, expiry_offset=30),
        ],
        max_profit="双倍权利金 (标的不动)",
        max_loss="无限 (大幅波动)",
        breakeven="行权价 ± 总权利金",
        ideal_iv="高IV卖出",
        time_decay="正向",
        description="同时卖出平值认购和认沽，赚取时间价值。押注低波动，风险较大。",
        risk_level=5
    ),
    
    # === Complex Strategies ===
    "iron_condor": StrategyTemplate(
        name="Iron Condor",
        chinese_name="铁鹰式",
        type=StrategyType.INCOME,
        outlook=[MarketOutlook.NEUTRAL, MarketOutlook.LOW_VOL],
        legs=[
            OptionLeg(type='put', action='buy', strike_offset=-3, expiry_offset=30),
            OptionLeg(type='put', action='sell', strike_offset=-2, expiry_offset=30),
            OptionLeg(type='call', action='sell', strike_offset=2, expiry_offset=30),
            OptionLeg(type='call', action='buy', strike_offset=3, expiry_offset=30),
        ],
        max_profit="净权利金",
        max_loss="价差宽度 - 净权利金",
        breakeven="卖出行权价 ± 净权利金",
        ideal_iv="高IV",
        time_decay="正向",
        description="结合牛市看跌价差和熊市看涨价差，四腿策略。适合盘整市场，胜率高但盈亏比较低。",
        risk_level=3
    ),
    
    "iron_butterfly": StrategyTemplate(
        name="Iron Butterfly",
        chinese_name="铁蝶式",
        type=StrategyType.INCOME,
        outlook=[MarketOutlook.NEUTRAL, MarketOutlook.LOW_VOL],
        legs=[
            OptionLeg(type='put', action='buy', strike_offset=-2, expiry_offset=30),
            OptionLeg(type='put', action='sell', strike_offset=0, expiry_offset=30),
            OptionLeg(type='call', action='sell', strike_offset=0, expiry_offset=30),
            OptionLeg(type='call', action='buy', strike_offset=2, expiry_offset=30),
        ],
        max_profit="净权利金",
        max_loss="价差宽度 - 净权利金",
        breakeven="平值行权价 ± 净权利金",
        ideal_iv="高IV",
        time_decay="正向",
        description="类似铁鹰式，但卖出的是平值期权。收益更高但盈利区间更窄。",
        risk_level=3
    ),
    
    "calendar_spread": StrategyTemplate(
        name="Calendar Spread",
        chinese_name="日历价差",
        type=StrategyType.VOLATILITY,
        outlook=[MarketOutlook.NEUTRAL],
        legs=[
            OptionLeg(type='call', action='sell', strike_offset=0, expiry_offset=30),
            OptionLeg(type='call', action='buy', strike_offset=0, expiry_offset=60),
        ],
        max_profit="近月期权时间价值",
        max_loss="净权利金",
        breakeven="复杂，取决于IV变化",
        ideal_iv="期待远月IV上升",
        time_decay="正向 (近月衰减更快)",
        description="卖出近月期权，买入远月同行权价期权。利用近月时间价值衰减更快的特点。",
        risk_level=3
    ),
    
    # === Hedge Strategies ===
    "protective_put": StrategyTemplate(
        name="Protective Put",
        chinese_name="保护性看跌",
        type=StrategyType.HEDGE,
        outlook=[MarketOutlook.BULLISH],
        legs=[
            OptionLeg(type='put', action='buy', strike_offset=-1, expiry_offset=30),
        ],
        max_profit="无限 (标的上涨) - 权利金",
        max_loss="现价 - 行权价 + 权利金",
        breakeven="现价 + 权利金",
        ideal_iv="低IV",
        time_decay="负向",
        description="持有标的同时买入看跌期权对冲下行风险。相当于买保险。",
        risk_level=1
    ),
    
    "collar": StrategyTemplate(
        name="Collar",
        chinese_name="领口策略",
        type=StrategyType.HEDGE,
        outlook=[MarketOutlook.NEUTRAL],
        legs=[
            OptionLeg(type='put', action='buy', strike_offset=-1, expiry_offset=30),
            OptionLeg(type='call', action='sell', strike_offset=1, expiry_offset=30),
        ],
        max_profit="高行权价 - 现价 + 净权利金",
        max_loss="现价 - 低行权价 + 净权利金",
        breakeven="现价 + 净权利金",
        ideal_iv="高IV (卖的比买的贵)",
        time_decay="中性",
        description="持有标的，买入虚值看跌对冲，卖出虚值看涨降低成本。锁定区间收益。",
        risk_level=1
    ),
}


def get_strategy_by_outlook(outlook: MarketOutlook) -> List[StrategyTemplate]:
    """Get strategies suitable for a market outlook."""
    return [s for s in STRATEGY_TEMPLATES.values() if outlook in s.outlook]


def get_strategy_by_type(stype: StrategyType) -> List[StrategyTemplate]:
    """Get strategies by type."""
    return [s for s in STRATEGY_TEMPLATES.values() if s.type == stype]


def get_strategy_by_name(name: str) -> Optional[StrategyTemplate]:
    """Get strategy by name."""
    return STRATEGY_TEMPLATES.get(name.lower().replace(" ", "_"))
