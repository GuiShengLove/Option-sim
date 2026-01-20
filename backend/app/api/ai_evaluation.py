"""
AI Strategy Evaluation API
===========================
Calls DeepSeek API to generate intelligent strategy evaluations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import httpx
import os

router = APIRouter(prefix="/api/ai", tags=["ai"])

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-17849f84151a4ab689cbf23f5e86b3f9")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class EvaluationRequest(BaseModel):
    """Request for AI strategy evaluation."""
    strategy_name: str
    metrics: Dict[str, Any]  # All backtest metrics
    greeks: Optional[Dict[str, float]] = None
    trades_count: int = 0


class EvaluationResponse(BaseModel):
    """AI evaluation response."""
    success: bool
    evaluation: str  # AI generated evaluation text
    risk_level: str  # 低/中/高
    suggestions: list[str]  # Key suggestions


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_strategy(request: EvaluationRequest):
    """
    Call DeepSeek API to evaluate strategy performance.
    Returns a concise, professional evaluation.
    """
    try:
        # Build prompt
        metrics = request.metrics
        prompt = f"""你是一位专业的期权量化分析师。请根据以下回测结果，用2-3句话简洁评价该策略的表现，并给出1-2条改进建议。

策略名称: {request.strategy_name}
回测指标:
- 总收益率: {metrics.get('total_return', 0):.2f}%
- 最大回撤: {metrics.get('max_drawdown', 0):.2f}%
- 夏普比率: {metrics.get('sharpe_ratio', 0):.2f}
- 胜率: {metrics.get('win_rate', 0)*100:.1f}%
- 盈亏比: {metrics.get('profit_factor', 0):.2f}
- 交易次数: {request.trades_count}

请用JSON格式返回:
{{"evaluation": "简短评价", "risk_level": "低/中/高", "suggestions": ["建议1", "建议2"]}}"""

        # Call DeepSeek API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是专业的期权量化分析师，给出简洁专业的策略评价。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"}
                }
            )
            
            if response.status_code != 200:
                print(f"DeepSeek API error: {response.status_code} - {response.text}")
                # Fallback to default evaluation
                return generate_fallback_evaluation(request)
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Parse JSON response
            import json
            try:
                parsed = json.loads(content)
                return EvaluationResponse(
                    success=True,
                    evaluation=parsed.get("evaluation", "策略表现分析中..."),
                    risk_level=parsed.get("risk_level", "中"),
                    suggestions=parsed.get("suggestions", [])
                )
            except json.JSONDecodeError:
                # If JSON parsing fails, use the raw content
                return EvaluationResponse(
                    success=True,
                    evaluation=content[:200] if len(content) > 200 else content,
                    risk_level="中",
                    suggestions=[]
                )
                
    except Exception as e:
        print(f"AI evaluation error: {e}")
        return generate_fallback_evaluation(request)


def generate_fallback_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    """Generate fallback evaluation when API fails."""
    metrics = request.metrics
    total_return = metrics.get('total_return', 0)
    sharpe = metrics.get('sharpe_ratio', 0)
    max_dd = metrics.get('max_drawdown', 0)
    win_rate = metrics.get('win_rate', 0)
    
    # Determine risk level
    if max_dd > 20 or sharpe < 0:
        risk_level = "高"
    elif max_dd > 10 or sharpe < 0.5:
        risk_level = "中"
    else:
        risk_level = "低"
    
    # Generate evaluation
    if total_return > 10 and sharpe > 1:
        evaluation = f"策略表现优异，收益{total_return:.1f}%，夏普比率{sharpe:.2f}，风险调整后收益良好。"
    elif total_return > 0:
        evaluation = f"策略实现正收益{total_return:.1f}%，但需关注{max_dd:.1f}%的回撤控制。"
    else:
        evaluation = f"策略录得{total_return:.1f}%亏损，建议重新审视入场逻辑和风控体系。"
    
    # Generate suggestions
    suggestions = []
    if win_rate < 0.4:
        suggestions.append("胜率偏低，建议优化入场时机")
    if max_dd > 15:
        suggestions.append("回撤较大，建议加强止损机制")
    if sharpe < 0.5:
        suggestions.append("收益波动大，考虑降低仓位")
    
    if not suggestions:
        suggestions.append("继续保持当前策略配置")
    
    return EvaluationResponse(
        success=False,  # Indicate fallback was used
        evaluation=evaluation,
        risk_level=risk_level,
        suggestions=suggestions[:2]
    )
