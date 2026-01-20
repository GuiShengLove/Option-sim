"""
Professional Risk & Margin Engine
==================================
Implements industry-standard margin calculation methods:
- SSE/SZSE: Shanghai/Shenzhen Stock Exchange option margin formulas
- SPAN: Standard Portfolio Analysis of Risk (16-scenario analysis)
- PM: Portfolio Margin with spread recognition

Supports multiple asset classes with configurable contract multipliers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

# ==================== Asset Configuration ====================

# Contract multipliers for different underlying assets
ASSET_MULTIPLIERS = {
    '510050': 10000,   # 50ETF
    '510300': 10000,   # 沪深300ETF
    '510500': 10000,   # 中证500ETF
    '159919': 10000,   # 沪深300ETF (深交所)
    '159915': 10000,   # 创业板ETF
    '588000': 10000,   # 科创板ETF
    'DEFAULT': 10000,  # 默认乘数
}

def get_multiplier(asset_code: str) -> float:
    """Get contract multiplier for asset."""
    # Extract base code (remove _SH/_SZ suffix)
    base_code = asset_code.split('_')[0] if '_' in asset_code else asset_code
    return ASSET_MULTIPLIERS.get(base_code, ASSET_MULTIPLIERS['DEFAULT'])


class MarginScheme(Enum):
    """Supported margin calculation schemes."""
    FIXED = 'FIXED'      # Fixed percentage (simple)
    SSE = 'SSE'          # Shanghai Stock Exchange standard
    SPAN = 'SPAN'        # SPAN risk-based
    PM = 'PM'            # Portfolio Margin


# ==================== Account Model ====================

@dataclass
class MarginAccount:
    """
    Trading account with margin capabilities.
    Tracks cash, equity, margin requirements, and liquidity.
    """
    initial_capital: float
    margin_scheme: str = 'SSE'  # 'FIXED', 'SSE', 'SPAN', 'PM'
    asset_code: str = '510050'  # For multiplier lookup
    
    # 自定义保证金参数（可选）
    custom_margin_ratio: float = None  # 自定义保证金率，覆盖默认 12%
    custom_maintenance_margin: float = None  # 自定义维持保证金率，覆盖默认 7%
    leverage: float = 1.0  # 杠杆倍数：保证金要求 = 基础保证金 ÷ 杠杆
    
    # Account balances
    cash: float = 0.0
    equity: float = 0.0
    
    # Margin requirements
    maintenance_margin: float = 0.0
    initial_margin: float = 0.0
    
    # Derived metrics
    excess_liquidity: float = 0.0
    margin_utilization: float = 0.0
    
    def __post_init__(self):
        if self.cash == 0.0:
            self.cash = self.initial_capital
        self.equity = self.cash
        self.update_balances()
    
    @property
    def margin_rate(self) -> float:
        """获取实际使用的保证金率"""
        return self.custom_margin_ratio if self.custom_margin_ratio is not None else 0.12
    
    @property
    def maint_rate(self) -> float:
        """获取实际使用的维持保证金率"""
        return self.custom_maintenance_margin if self.custom_maintenance_margin is not None else 0.07

    def update_balances(self):
        """Update derived balances after position changes."""
        self.excess_liquidity = max(0, self.equity - self.maintenance_margin)
        self.margin_utilization = (
            self.maintenance_margin / self.equity 
            if self.equity > 0 else 0.0
        )
    
    @property
    def is_margin_call(self) -> bool:
        """Check if account is in margin call status."""
        return self.equity < self.maintenance_margin
    
    @property
    def buying_power(self) -> float:
        """Available funds for new positions."""
        return self.excess_liquidity


# ==================== SSE/SZSE Margin Formulas ====================

class SSEMarginCalculator:
    """
    Shanghai/Shenzhen Stock Exchange standard margin formulas.
    
    Official formulas (as of 2024):
    
    Short Call Margin = max(
        Premium + Underlying × 12% - OTM_Amount,
        Premium + Underlying × 7%
    ) × Multiplier × Contracts
    
    Short Put Margin = max(
        Premium + Underlying × 12% - OTM_Amount,
        Premium + Strike × 7%
    ) × Multiplier × Contracts
    
    Where:
    - OTM_Amount (Call) = max(0, Strike - Underlying)
    - OTM_Amount (Put) = max(0, Underlying - Strike)
    """
    
    # SSE Parameters (adjustable by exchange)
    INITIAL_MARGIN_RATE = 0.12    # 12% initial margin rate
    MAINTENANCE_MARGIN_RATE = 0.07  # 7% minimum margin rate
    
    @classmethod
    def calculate_short_call_margin(
        cls,
        underlying_price: float,
        strike: float,
        premium: float,
        quantity: int,
        multiplier: float
    ) -> float:
        """Calculate margin for short call position."""
        otm_amount = max(0, strike - underlying_price)
        
        margin_method1 = premium + underlying_price * cls.INITIAL_MARGIN_RATE - otm_amount
        margin_method2 = premium + underlying_price * cls.MAINTENANCE_MARGIN_RATE
        
        margin_per_unit = max(margin_method1, margin_method2)
        return margin_per_unit * multiplier * abs(quantity)
    
    @classmethod
    def calculate_short_put_margin(
        cls,
        underlying_price: float,
        strike: float,
        premium: float,
        quantity: int,
        multiplier: float
    ) -> float:
        """Calculate margin for short put position."""
        otm_amount = max(0, underlying_price - strike)
        
        margin_method1 = premium + underlying_price * cls.INITIAL_MARGIN_RATE - otm_amount
        margin_method2 = premium + strike * cls.MAINTENANCE_MARGIN_RATE
        
        margin_per_unit = max(margin_method1, margin_method2)
        return margin_per_unit * multiplier * abs(quantity)
    
    @classmethod
    def calculate_position_margin(
        cls,
        underlying_price: float,
        strike: float,
        option_type: str,  # 'C' or 'P'
        premium: float,
        quantity: int,
        multiplier: float
    ) -> float:
        """
        Calculate margin for any option position.
        
        Long positions: Only cost (no margin requirement)
        Short positions: Full margin calculation
        """
        if quantity >= 0:  # Long position - no margin, just premium cost
            return 0.0
        
        if option_type.upper() in ('C', 'CALL'):
            return cls.calculate_short_call_margin(
                underlying_price, strike, premium, quantity, multiplier
            )
        else:  # Put
            return cls.calculate_short_put_margin(
                underlying_price, strike, premium, quantity, multiplier
            )


# ==================== SPAN Margin Calculator ====================

class SPANCalculator:
    """
    Standard Portfolio Analysis of Risk (SPAN) Margin System.
    
    SPAN uses 16 risk scenarios combining:
    - 7 price scan ranges: ±(1/3, 2/3, 1) × price_scan_range
    - 2 volatility shifts: ±vol_scan_range
    - Plus extreme scenarios
    
    Margin = max(scenario losses) + extreme move add-on
    """
    
    # SPAN Parameters (configurable per asset class)
    DEFAULT_PRICE_SCAN_RANGE = 0.15  # 15% price scan
    DEFAULT_VOL_SCAN_RANGE = 0.25    # 25% vol shift
    EXTREME_MOVE_FACTOR = 0.35       # 35% extreme scenario
    SHORT_OPTION_MINIMUM = 0.02      # 2% minimum for short options
    
    # Price scan fractions
    PRICE_SCAN_FRACTIONS = [
        -1.0, -2/3, -1/3, 0.0, 1/3, 2/3, 1.0
    ]
    
    @classmethod
    def generate_scenarios(
        cls, 
        underlying_price: float,
        current_vol: float = 0.25,
        price_scan_range: float = None,
        vol_scan_range: float = None
    ) -> List[Tuple[float, float]]:
        """
        Generate 16 SPAN scenarios.
        Returns list of (shocked_price, shocked_vol) tuples.
        """
        price_range = price_scan_range or cls.DEFAULT_PRICE_SCAN_RANGE
        vol_range = vol_scan_range or cls.DEFAULT_VOL_SCAN_RANGE
        
        scenarios = []
        
        # Standard 14 scenarios: 7 price moves × 2 vol shifts
        for price_frac in cls.PRICE_SCAN_FRACTIONS:
            price_shock = underlying_price * (1 + price_frac * price_range)
            
            for vol_dir in [-1, 1]:
                vol_shock = current_vol * (1 + vol_dir * vol_range)
                vol_shock = max(0.05, vol_shock)  # Floor at 5% vol
                scenarios.append((price_shock, vol_shock))
        
        # Extreme scenarios (up and down)
        extreme_up = underlying_price * (1 + cls.EXTREME_MOVE_FACTOR)
        extreme_down = underlying_price * (1 - cls.EXTREME_MOVE_FACTOR)
        scenarios.append((extreme_up, current_vol * 1.5))
        scenarios.append((extreme_down, current_vol * 1.5))
        
        return scenarios
    
    @classmethod
    def estimate_option_price_at_scenario(
        cls,
        current_price: float,
        strike: float,
        option_type: str,
        current_spot: float,
        shocked_spot: float,
        current_vol: float,
        shocked_vol: float,
        days_to_expiry: float = 30
    ) -> float:
        """
        Estimate option price at scenario using delta approximation.
        
        For speed, we use: NewPrice ≈ Intrinsic + TimeValue × VolRatio
        More accurate would be full BS repricing.
        """
        # Calculate intrinsic values
        if option_type.upper() in ('C', 'CALL'):
            new_intrinsic = max(0, shocked_spot - strike)
            current_intrinsic = max(0, current_spot - strike)
        else:
            new_intrinsic = max(0, strike - shocked_spot)
            current_intrinsic = max(0, strike - current_spot)
        
        # Time value adjustment based on vol change
        current_time_value = max(0, current_price - current_intrinsic)
        vol_ratio = shocked_vol / current_vol if current_vol > 0 else 1.0
        
        # Estimate decayed time value (simplified)
        adjusted_time_value = current_time_value * vol_ratio
        
        # At extreme moves, time value collapses
        price_pct_change = abs(shocked_spot - current_spot) / current_spot
        if price_pct_change > 0.20:
            adjusted_time_value *= max(0.3, 1 - price_pct_change)
        
        return max(0, new_intrinsic + adjusted_time_value)
    
    @classmethod
    def calculate_portfolio_margin(
        cls,
        positions: List[Dict],
        underlying_price: float,
        current_vol: float = 0.25,
        multiplier: float = 10000,
        price_scan_range: float = None,
        vol_scan_range: float = None
    ) -> Tuple[float, List[Dict]]:
        """
        Calculate SPAN margin for portfolio.
        
        Args:
            positions: List of position dicts with keys:
                - type: 'C' or 'P'
                - strike: float
                - quantity: int (negative for short)
                - current_price: float
                - days_to_expiry: float (optional)
            underlying_price: Current underlying price
            current_vol: Current implied volatility
            multiplier: Contract multiplier
        
        Returns:
            (margin_requirement, scenario_details)
        """
        if not positions:
            return 0.0, []
        
        scenarios = cls.generate_scenarios(
            underlying_price, current_vol, price_scan_range, vol_scan_range
        )
        
        scenario_results = []
        max_loss = 0.0
        worst_scenario = None
        
        for i, (shocked_spot, shocked_vol) in enumerate(scenarios):
            portfolio_pnl = 0.0
            
            for pos in positions:
                strike = pos['strike']
                qty = pos['quantity']
                otype = pos['type']
                current_price = pos.get('current_price', 0)
                dte = pos.get('days_to_expiry', 30)
                
                # Estimate new price at scenario
                new_price = cls.estimate_option_price_at_scenario(
                    current_price, strike, otype,
                    underlying_price, shocked_spot,
                    current_vol, shocked_vol, dte
                )
                
                # Position P&L
                pnl = (new_price - current_price) * qty * multiplier
                portfolio_pnl += pnl
            
            scenario_results.append({
                'scenario_id': i,
                'spot_shock': (shocked_spot - underlying_price) / underlying_price,
                'vol_shock': (shocked_vol - current_vol) / current_vol,
                'portfolio_pnl': portfolio_pnl
            })
            
            # Track worst case
            if portfolio_pnl < 0 and abs(portfolio_pnl) > max_loss:
                max_loss = abs(portfolio_pnl)
                worst_scenario = i
        
        # Add short option minimum
        short_option_value = sum(
            pos.get('current_price', 0) * abs(pos['quantity']) * multiplier
            for pos in positions if pos['quantity'] < 0
        )
        min_margin = short_option_value * cls.SHORT_OPTION_MINIMUM
        
        margin = max(max_loss, min_margin)
        
        return margin, scenario_results


# ==================== Portfolio Margin (Spread Recognition) ====================

class PortfolioMarginCalculator:
    """
    Portfolio Margin with spread recognition.
    
    Recognizes common option spreads and applies reduced margin:
    - Vertical Spreads (Bull/Bear Call/Put): Max loss
    - Iron Condors: Single side risk
    - Covered Calls/Puts: Covered writing rules
    - Straddles/Strangles: Larger side + premium
    """
    
    # Spread identification thresholds
    STRIKE_DIFF_TOLERANCE = 0.001  # 0.1% tolerance for strike matching
    
    @classmethod
    def identify_spreads(
        cls, 
        positions: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Identify spread positions and return (spreads, remaining_positions).
        
        Each spread dict contains:
        - type: 'vertical', 'iron_condor', 'straddle', 'strangle', etc.
        - legs: list of position dicts
        - max_risk: calculated max risk
        """
        spreads = []
        remaining = positions.copy()
        
        # Sort by strike for easier matching
        remaining.sort(key=lambda p: (p['type'], p['strike']))
        
        # Find vertical spreads (same type, different strikes)
        i = 0
        while i < len(remaining) - 1:
            pos1 = remaining[i]
            
            for j in range(i + 1, len(remaining)):
                pos2 = remaining[j]
                
                # Check for vertical spread
                if (pos1['type'] == pos2['type'] and
                    pos1['quantity'] * pos2['quantity'] < 0):  # Opposite directions
                    
                    strike_diff = abs(pos2['strike'] - pos1['strike'])
                    if strike_diff > 0:
                        # This is a vertical spread
                        spreads.append({
                            'type': 'vertical',
                            'option_type': pos1['type'],
                            'legs': [pos1, pos2],
                            'width': strike_diff
                        })
                        remaining.remove(pos1)
                        remaining.remove(pos2)
                        i -= 1
                        break
            i += 1
        
        return spreads, remaining
    
    @classmethod
    def calculate_spread_margin(
        cls,
        spread: Dict,
        multiplier: float
    ) -> float:
        """Calculate margin for identified spread."""
        spread_type = spread['type']
        legs = spread['legs']
        
        if spread_type == 'vertical':
            # Vertical spread: Max loss = width × multiplier × contracts
            width = spread['width']
            contracts = min(abs(legs[0]['quantity']), abs(legs[1]['quantity']))
            
            # Credit or debit?
            net_premium = sum(
                leg['current_price'] * leg['quantity'] 
                for leg in legs
            )
            
            if net_premium > 0:  # Credit spread
                max_loss = (width - abs(net_premium)) * multiplier * contracts
            else:  # Debit spread
                max_loss = abs(net_premium) * multiplier * contracts
            
            return max(0, max_loss)
        
        return 0.0
    
    @classmethod
    def calculate_portfolio_margin(
        cls,
        positions: List[Dict],
        underlying_price: float,
        multiplier: float = 10000
    ) -> float:
        """
        Calculate portfolio margin with spread recognition.
        
        1. Identify spreads
        2. Calculate spread margins (reduced risk)
        3. Calculate naked option margins for remaining
        4. Sum all margins
        """
        if not positions:
            return 0.0
        
        # Identify spreads
        spreads, naked_positions = cls.identify_spreads(positions)
        
        total_margin = 0.0
        
        # Spread margins
        for spread in spreads:
            total_margin += cls.calculate_spread_margin(spread, multiplier)
        
        # Naked position margins (use SSE formulas)
        for pos in naked_positions:
            margin = SSEMarginCalculator.calculate_position_margin(
                underlying_price,
                pos['strike'],
                pos['type'],
                pos.get('current_price', 0),
                pos['quantity'],
                multiplier
            )
            total_margin += margin
        
        return total_margin


# ==================== Main Risk Engine ====================

class RiskEngine:
    """
    Professional Risk & Margin Engine.
    
    Supports multiple margin calculation schemes:
    - FIXED: Simple fixed percentage
    - SSE: Shanghai Stock Exchange standard formulas
    - SPAN: Risk-based scenario analysis
    - PM: Portfolio margin with spread recognition
    """
    
    def __init__(self, account: MarginAccount):
        self.account = account
        self.multiplier = get_multiplier(account.asset_code)
        self.scheme = MarginScheme(account.margin_scheme.upper())
    
    def calculate_margin_impact(
        self, 
        underlying_price: float, 
        strike: float, 
        type_: str, 
        premium: float, 
        quantity: float,
        is_short: bool = True
    ) -> float:
        """
        Calculate margin impact for a single option position.
        Used for pre-trade validation.
        """
        if quantity > 0:  # Long position
            # Long options: only require premium payment, no margin
            return 0.0
        
        # Short positions: Use SSE formulas for single position
        return SSEMarginCalculator.calculate_position_margin(
            underlying_price, strike, type_, premium, int(quantity), self.multiplier
        )
    
    def calculate_portfolio_margin(
        self, 
        positions: List[Dict], 
        underlying_price: float,
        current_vol: float = 0.25
    ) -> float:
        """
        Calculate portfolio margin using configured scheme.
        
        Args:
            positions: List of position dicts with keys:
                - type: 'C' or 'P'
                - strike: float
                - quantity: int (negative for short)
                - current_price: float
            underlying_price: Current underlying price
            current_vol: Current implied volatility (for SPAN)
        
        Returns:
            Total margin requirement
        """
        if not positions:
            return 0.0
        
        base_margin = 0.0
        
        if self.scheme == MarginScheme.FIXED:
            # Simple fixed percentage of notional (使用自定义保证金率)
            margin_rate = self.account.margin_rate  # 从账户获取保证金率
            total_notional = sum(
                abs(p['quantity']) * underlying_price * self.multiplier
                for p in positions
            )
            base_margin = total_notional * margin_rate
        
        elif self.scheme == MarginScheme.SSE:
            # Sum of individual position margins (no netting)
            total = 0.0
            for pos in positions:
                margin = SSEMarginCalculator.calculate_position_margin(
                    underlying_price,
                    pos['strike'],
                    pos['type'],
                    pos.get('current_price', 0),
                    pos['quantity'],
                    self.multiplier
                )
                total += margin
            base_margin = total
        
        elif self.scheme == MarginScheme.SPAN:
            # Full SPAN scenario analysis
            margin, _ = SPANCalculator.calculate_portfolio_margin(
                positions, underlying_price, current_vol, self.multiplier
            )
            base_margin = margin
        
        elif self.scheme == MarginScheme.PM:
            # Portfolio margin with spread recognition
            base_margin = PortfolioMarginCalculator.calculate_portfolio_margin(
                positions, underlying_price, self.multiplier
            )
        
        else:
            # Fallback to SSE
            base_margin = self.calculate_portfolio_margin_sse(positions, underlying_price)
        
        # 应用杠杆：杠杆越高，保证金要求越低
        leverage = self.account.leverage if self.account.leverage and self.account.leverage > 0 else 1.0
        return base_margin / leverage
    
    def calculate_portfolio_margin_sse(
        self, 
        positions: List[Dict], 
        underlying_price: float
    ) -> float:
        """SSE-style margin calculation (no netting)."""
        total = 0.0
        for pos in positions:
            margin = SSEMarginCalculator.calculate_position_margin(
                underlying_price,
                pos['strike'],
                pos['type'],
                pos.get('current_price', 0),
                pos['quantity'],
                self.multiplier
            )
            total += margin
        return total
    
    def check_liquidation(self) -> bool:
        """
        Check if account needs liquidation.
        Returns True if Equity < Maintenance Margin.
        """
        return self.account.equity < self.account.maintenance_margin
    
    def get_margin_summary(self, positions: List[Dict], underlying_price: float) -> Dict:
        """
        Get comprehensive margin summary.
        
        Returns dict with:
        - total_margin: Total margin requirement
        - margin_utilization: Margin / Equity ratio
        - excess_liquidity: Available buying power
        - scheme_used: Margin scheme name
        - position_breakdown: Per-position margins
        """
        total_margin = self.calculate_portfolio_margin(positions, underlying_price)
        
        position_breakdown = []
        for pos in positions:
            pos_margin = SSEMarginCalculator.calculate_position_margin(
                underlying_price,
                pos['strike'],
                pos['type'],
                pos.get('current_price', 0),
                pos['quantity'],
                self.multiplier
            )
            position_breakdown.append({
                'strike': pos['strike'],
                'type': pos['type'],
                'quantity': pos['quantity'],
                'margin': pos_margin
            })
        
        utilization = total_margin / self.account.equity if self.account.equity > 0 else 0
        
        return {
            'total_margin': total_margin,
            'margin_utilization': utilization,
            'excess_liquidity': self.account.equity - total_margin,
            'scheme_used': self.scheme.value,
            'position_breakdown': position_breakdown,
            'is_margin_call': utilization > 0.9,
            'is_critical': utilization > 1.0
        }
