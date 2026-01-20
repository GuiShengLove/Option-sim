"""
Enhanced AI API Routes
======================
Upgraded with DeepSeek integration and strategy templates.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Literal
import asyncio

from ..engines.ai_service import get_ai_service
from ..engines.strategy_templates import (
    STRATEGY_TEMPLATES, 
    get_strategy_by_outlook, 
    get_strategy_by_type,
    StrategyTemplate,
    MarketOutlook,
    StrategyType
)

router = APIRouter(prefix="/api/ai", tags=["ai"])


# ============================================================
# Request/Response Models
# ============================================================

class ChatRequest(BaseModel):
    """Simple chat request."""
    message: str
    context: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response."""
    response: str
    model: str = "deepseek-chat"


class PortfolioAnalysisRequest(BaseModel):
    """Portfolio analysis request."""
    holdings: List[Dict[str, Any]]
    total_delta: float = 0
    total_gamma: float = 0
    total_vega: float = 0
    total_theta: float = 0


class PortfolioAnalysisResponse(BaseModel):
    """Portfolio analysis response."""
    analysis: str
    risk_score: int
    recommendations: List[str]


class StrategyRecommendRequest(BaseModel):
    """Strategy recommendation request."""
    market_view: Literal['bullish', 'bearish', 'neutral', 'volatile', 'range_bound']
    risk_tolerance: Literal['conservative', 'moderate', 'aggressive']
    capital: float
    underlying_price: float


class StrategyInfo(BaseModel):
    """Strategy information."""
    name: str
    chinese_name: str
    type: str
    description: str
    max_profit: str
    max_loss: str
    breakeven: str
    time_decay: str
    ideal_iv: str
    risk_level: int
    legs: List[Dict[str, Any]]


class StrategyRecommendResponse(BaseModel):
    """Strategy recommendation response."""
    recommended_strategies: List[StrategyInfo]
    ai_analysis: str


class CodeGenRequest(BaseModel):
    """Code generation request."""
    instruction: str


class CodeGenResponse(BaseModel):
    """Code generation response."""
    generated_code: str
    explanation: str


class RiskAnalysisRequest(BaseModel):
    """Risk analysis request for backtest results."""
    pnl_data: Dict[str, Any]  # Changed from float to Any to handle complex context
    sharpe: float
    max_drawdown: float
    win_rate: Optional[float] = None
    strategy_name: Optional[str] = None


class RiskAnalysisResponse(BaseModel):
    """Risk analysis response."""
    report_markdown: str
    risk_score: int


# ============================================================
# API Endpoints
# ============================================================

@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """
    Chat with AI assistant about options trading.
    """
    ai = get_ai_service()
    
    system_prompt = """你是一位专业的期权量化分析助手。请用专业但易懂的中文回答用户关于期权交易、希腊字母、策略构建等问题。

如果用户询问策略，请给出具体的构建方法和风险分析。使用Markdown格式让回答更清晰。"""
    
    from ..engines.ai_service import Message
    messages = [
        Message(role="system", content=system_prompt),
    ]
    
    if request.context:
        messages.append(Message(role="user", content=f"背景信息:\n{request.context}"))
    
    messages.append(Message(role="user", content=request.message))
    
    try:
        response = await ai.chat(messages)
        return ChatResponse(response=response)
    except Exception as e:
        return ChatResponse(response=f"AI服务暂时不可用: {str(e)}")


@router.post("/analyze-portfolio", response_model=PortfolioAnalysisResponse)
async def analyze_portfolio(request: PortfolioAnalysisRequest):
    """
    AI-powered portfolio risk analysis.
    """
    ai = get_ai_service()
    
    holdings_str = "\n".join([
        f"- {h.get('type', 'option').upper()} K={h.get('strike', 'N/A')} "
        f"到期={h.get('expiry', 'N/A')} 数量={h.get('quantity', 1)}"
        for h in request.holdings
    ])
    
    greeks_summary = f"""
Greeks汇总:
- Total Delta: {request.total_delta}
- Total Gamma: {request.total_gamma}
- Total Vega: {request.total_vega}
- Total Theta: {request.total_theta}
"""
    
    analysis = await ai.analyze_portfolio({
        "holdings": holdings_str,
        "greeks": greeks_summary
    })
    
    # Extract risk score (simplified)
    risk_score = 5
    if abs(request.total_delta) > 100:
        risk_score += 2
    if request.total_theta < -100:
        risk_score += 1
    risk_score = min(10, max(1, risk_score))
    
    recommendations = []
    if request.total_delta > 50:
        recommendations.append("考虑卖出认购期权或买入认沽期权降低Delta敞口")
    if request.total_delta < -50:
        recommendations.append("考虑买入认购期权或卖出认沽期权增加Delta敞口")
    if request.total_theta < -50:
        recommendations.append("时间价值损耗较大，考虑减少买方头寸或增加卖方头寸")
    
    return PortfolioAnalysisResponse(
        analysis=analysis,
        risk_score=risk_score,
        recommendations=recommendations
    )


@router.post("/recommend-strategy", response_model=StrategyRecommendResponse)
async def recommend_strategy(request: StrategyRecommendRequest):
    """
    Get AI-powered strategy recommendations based on market view.
    """
    ai = get_ai_service()
    
    # Map market view to outlook
    outlook_map = {
        'bullish': MarketOutlook.BULLISH,
        'bearish': MarketOutlook.BEARISH,
        'neutral': MarketOutlook.NEUTRAL,
        'volatile': MarketOutlook.HIGH_VOL,
        'range_bound': MarketOutlook.LOW_VOL,
    }
    
    outlook = outlook_map.get(request.market_view, MarketOutlook.NEUTRAL)
    
    # Get matching strategies from template library
    matching = get_strategy_by_outlook(outlook)
    
    # Filter by risk tolerance
    risk_max = {'conservative': 2, 'moderate': 3, 'aggressive': 5}
    max_risk = risk_max.get(request.risk_tolerance, 3)
    matching = [s for s in matching if s.risk_level <= max_risk][:5]
    
    # Build strategy info list
    strategies = []
    for s in matching:
        strategies.append(StrategyInfo(
            name=s.name,
            chinese_name=s.chinese_name,
            type=s.type.value,
            description=s.description,
            max_profit=s.max_profit,
            max_loss=s.max_loss,
            breakeven=s.breakeven,
            time_decay=s.time_decay,
            ideal_iv=s.ideal_iv,
            risk_level=s.risk_level,
            legs=[{
                'type': leg.type,
                'action': leg.action,
                'strike_offset': leg.strike_offset,
                'expiry_days': leg.expiry_offset,
                'quantity': leg.quantity
            } for leg in s.legs]
        ))
    
    # Get AI analysis
    ai_analysis = await ai.recommend_strategy(
        market_view=request.market_view,
        risk_tolerance=request.risk_tolerance,
        capital=request.capital,
        underlying_price=request.underlying_price
    )
    
    return StrategyRecommendResponse(
        recommended_strategies=strategies,
        ai_analysis=ai_analysis
    )


@router.get("/strategies", response_model=List[StrategyInfo])
async def list_all_strategies():
    """
    List all available strategy templates.
    """
    strategies = []
    for s in STRATEGY_TEMPLATES.values():
        strategies.append(StrategyInfo(
            name=s.name,
            chinese_name=s.chinese_name,
            type=s.type.value,
            description=s.description,
            max_profit=s.max_profit,
            max_loss=s.max_loss,
            breakeven=s.breakeven,
            time_decay=s.time_decay,
            ideal_iv=s.ideal_iv,
            risk_level=s.risk_level,
            legs=[{
                'type': leg.type,
                'action': leg.action,
                'strike_offset': leg.strike_offset,
                'expiry_days': leg.expiry_offset,
                'quantity': leg.quantity
            } for leg in s.legs]
        ))
    return strategies


@router.post("/generate-code", response_model=CodeGenResponse)
async def generate_strategy_code(request: CodeGenRequest):
    """
    Generate strategy code using AI.
    """
    ai = get_ai_service()
    
    result = await ai.generate_strategy_code(request.instruction)
    
    # Extract code block if present
    code = result
    explanation = "AI生成的策略代码"
    
    if "```python" in result:
        parts = result.split("```python")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0]
            code = code_part.strip()
            explanation = parts[0] + (parts[1].split("```")[1] if len(parts[1].split("```")) > 1 else "")
    
    return CodeGenResponse(
        generated_code=code,
        explanation=explanation.strip()
    )


@router.post("/analyze-risk", response_model=RiskAnalysisResponse)
async def analyze_risk(request: RiskAnalysisRequest):
    """
    Generate AI-powered risk analysis report for backtest results.
    """
    ai = get_ai_service()
    
    # Build context for AI
    # If pnl_data contains 'context' string, parse it for better formatting
    pnl_info = request.pnl_data
    if 'context' in pnl_info and isinstance(pnl_info['context'], str):
        try:
            import json
            pnl_info = json.loads(pnl_info['context'])
        except:
            pass

    import json
    pnl_str = json.dumps(pnl_info, ensure_ascii=False, indent=2)

    context = f"""
回测结果:
- 策略名称: {request.strategy_name or '未命名策略'}
- 夏普比率: {request.sharpe:.2f}
- 最大回撤: {request.max_drawdown:.2%}
- 胜率: {request.win_rate:.2%} (如有)

详情数据:
{pnl_str}
"""
    
    from ..engines.ai_service import Message
    messages = [
        Message(role="system", content="""你是专业的量化策略分析师。请基于回测数据生成风险分析报告。

报告格式:
1. 使用Markdown格式
2. 包含风险评估、收益分析、改进建议
3. 使用GitHub风格的提示框 ([!NOTE], [!WARNING], [!CAUTION])
4. 给出1-10的风险评分"""),
        Message(role="user", content=context)
    ]
    
    try:
        report = await ai.chat(messages)
        
        # Check if report itself indicates an error
        if report.startswith("API Error") or report.startswith("Error:"):
            raise Exception(report)
            
    except Exception as e:
        print(f"AI Generation Failed: {str(e)}")
        # Fallback to basic analysis
        report = "### 🛡️ 风险分析报告 (本地降级模式)\n\n> [!WARNING]\n> AI服务暂时不可用，以下为基础规则分析。\n\n"
        
        if request.max_drawdown < -0.20:
            report += "> [!CAUTION]\n> **高风险警告**: 最大回撤超过20%\n\n"
            risk_score = 8
        elif request.max_drawdown < -0.10:
            report += "> [!WARNING]\n> **中等风险**: 回撤在10-20%之间\n\n"
            risk_score = 5
        else:
            report += "> [!NOTE]\n> **风险可控**: 回撤低于10%\n\n"
            risk_score = 3
        
        if request.sharpe > 2.0:
            report += "**收益评估**: 夏普比率优秀 ✅\n"
        elif request.sharpe > 1.0:
            report += "**收益评估**: 夏普比率良好 📊\n"
        else:
            report += "**收益评估**: 夏普比率需要改进 ⚠️\n"
            risk_score += 1
        
        return RiskAnalysisResponse(
            report_markdown=report,
            risk_score=min(10, max(1, risk_score))
        )
    
    # Extract risk score from report (look for patterns like "风险评分: 7")
    risk_score = 5
    import re
    score_match = re.search(r'风险评分[：:]\s*(\d+)', report)
    if score_match:
        risk_score = int(score_match.group(1))
    
    return RiskAnalysisResponse(
        report_markdown=report,
        risk_score=min(10, max(1, risk_score))
    )
