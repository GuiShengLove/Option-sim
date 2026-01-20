"""
Margin API
==========
API endpoints for margin calculations and analysis.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from ..engines.risk import (
    MarginAccount, RiskEngine, MarginScheme,
    SSEMarginCalculator, SPANCalculator, PortfolioMarginCalculator,
    get_multiplier
)

router = APIRouter(prefix="/api/margin", tags=["margin"])


class PositionItem(BaseModel):
    """Single option position for margin calculation."""
    type: str  # 'C' or 'P'
    strike: float
    quantity: int  # Negative for short
    current_price: float
    days_to_expiry: Optional[float] = 30


class MarginRequest(BaseModel):
    """Request for margin calculation."""
    positions: List[PositionItem]
    underlying_price: float
    asset_code: str = "510050"
    margin_scheme: str = "SSE"  # 'FIXED', 'SSE', 'SPAN', 'PM'
    current_vol: Optional[float] = 0.25
    initial_capital: Optional[float] = 1000000


class MarginResponse(BaseModel):
    """Response with margin calculation results."""
    success: bool
    total_margin: float
    margin_utilization: float
    excess_liquidity: float
    scheme_used: str
    position_breakdown: List[Dict]
    is_margin_call: bool
    is_critical: bool


class ScenarioItem(BaseModel):
    """Single scenario result for SPAN analysis."""
    scenario_id: int
    spot_shock: float
    vol_shock: float
    portfolio_pnl: float


class SPANResponse(BaseModel):
    """Response with SPAN scenario analysis."""
    success: bool
    total_margin: float
    worst_scenario: Optional[int]
    scenarios: List[ScenarioItem]


@router.post("/calculate", response_model=MarginResponse)
async def calculate_margin(request: MarginRequest):
    """
    Calculate margin requirement for a portfolio.
    
    Supports multiple margin schemes:
    - FIXED: Simple fixed percentage (12%)
    - SSE: Shanghai Stock Exchange standard formulas
    - SPAN: Risk-based scenario analysis
    - PM: Portfolio margin with spread recognition
    """
    try:
        # Convert positions to dict format
        positions = [
            {
                'type': p.type,
                'strike': p.strike,
                'quantity': p.quantity,
                'current_price': p.current_price,
                'days_to_expiry': p.days_to_expiry
            }
            for p in request.positions
        ]
        
        multiplier = get_multiplier(request.asset_code)
        
        # Create temporary account for calculation
        account = MarginAccount(
            initial_capital=request.initial_capital or 1000000,
            margin_scheme=request.margin_scheme,
            asset_code=request.asset_code
        )
        
        engine = RiskEngine(account)
        
        # Get comprehensive summary
        summary = engine.get_margin_summary(positions, request.underlying_price)
        
        return MarginResponse(
            success=True,
            total_margin=summary['total_margin'],
            margin_utilization=summary['margin_utilization'],
            excess_liquidity=summary['excess_liquidity'],
            scheme_used=summary['scheme_used'],
            position_breakdown=summary['position_breakdown'],
            is_margin_call=summary['is_margin_call'],
            is_critical=summary['is_critical']
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Margin calculation failed: {str(e)}")


@router.post("/span-analysis", response_model=SPANResponse)
async def span_analysis(request: MarginRequest):
    """
    Run full SPAN scenario analysis.
    
    Returns detailed results for all 16 risk scenarios,
    showing potential P&L under various price and volatility shocks.
    """
    try:
        positions = [
            {
                'type': p.type,
                'strike': p.strike,
                'quantity': p.quantity,
                'current_price': p.current_price,
                'days_to_expiry': p.days_to_expiry
            }
            for p in request.positions
        ]
        
        multiplier = get_multiplier(request.asset_code)
        
        margin, scenario_results = SPANCalculator.calculate_portfolio_margin(
            positions,
            request.underlying_price,
            request.current_vol or 0.25,
            multiplier
        )
        
        # Find worst scenario
        worst_idx = None
        worst_loss = 0
        for i, s in enumerate(scenario_results):
            if s['portfolio_pnl'] < 0 and abs(s['portfolio_pnl']) > worst_loss:
                worst_loss = abs(s['portfolio_pnl'])
                worst_idx = i
        
        return SPANResponse(
            success=True,
            total_margin=margin,
            worst_scenario=worst_idx,
            scenarios=[
                ScenarioItem(
                    scenario_id=s['scenario_id'],
                    spot_shock=s['spot_shock'],
                    vol_shock=s['vol_shock'],
                    portfolio_pnl=s['portfolio_pnl']
                )
                for s in scenario_results
            ]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPAN analysis failed: {str(e)}")


@router.get("/schemes")
async def list_margin_schemes():
    """List available margin calculation schemes."""
    return {
        "schemes": [
            {
                "id": "FIXED",
                "name": "固定比例",
                "description": "简单固定比例保证金(12%)"
            },
            {
                "id": "SSE",
                "name": "上交所标准",
                "description": "上海证券交易所标准保证金公式"
            },
            {
                "id": "SPAN",
                "name": "SPAN风险分析",
                "description": "标准组合风险分析系统(16情景)"
            },
            {
                "id": "PM",
                "name": "组合保证金",
                "description": "识别价差组合,降低保证金要求"
            }
        ]
    }


@router.get("/asset-multipliers")
async def list_asset_multipliers():
    """List contract multipliers for supported assets."""
    from ..engines.risk import ASSET_MULTIPLIERS
    
    return {
        "multipliers": {
            code: mult for code, mult in ASSET_MULTIPLIERS.items()
        }
    }
