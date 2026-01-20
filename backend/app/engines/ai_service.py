"""
AI Service for DeepSeek Integration
====================================
Professional AI-powered analysis and strategy recommendations.
"""

import os
import httpx
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-17849f84151a4ab689cbf23f5e86b3f9")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class DeepSeekService:
    """
    Service for interacting with DeepSeek AI API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base_url = DEEPSEEK_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat(
        self, 
        messages: List[Message], 
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Send a chat request to DeepSeek API.
        
        Args:
            messages: List of conversation messages
            model: Model to use (deepseek-chat, deepseek-coder)
            temperature: Randomness (0.0-1.0)
            max_tokens: Maximum response length
            
        Returns:
            AI response text
        """
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                return f"API Error: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                return f"Error: {str(e)}"
    
    async def analyze_portfolio(self, holdings: Dict[str, Any]) -> str:
        """
        Analyze portfolio risk using AI.
        """
        system_prompt = """你是一位专业的期权量化分析师。请分析用户的期权持仓组合，给出:
1. 风险评估 (方向性风险、波动率风险、时间价值损耗)
2. Greeks敞口分析
3. 改进建议
4. 风险评分 (1-10)

请用专业但易懂的语言，使用Markdown格式输出。"""
        
        user_prompt = f"""请分析以下期权组合:

持仓详情:
{holdings}

请给出详细分析报告。"""
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt)
        ]
        
        return await self.chat(messages)
    
    async def recommend_strategy(
        self, 
        market_view: str, 
        risk_tolerance: str,
        capital: float,
        underlying_price: float
    ) -> str:
        """
        Get AI-powered strategy recommendations.
        """
        system_prompt = """你是一位专业的期权策略顾问。基于用户的市场观点、风险偏好和资金情况，推荐最合适的期权策略。

请提供:
1. 推荐策略名称
2. 策略结构 (买/卖哪些期权)
3. 建议行权价和到期日选择
4. 预期收益/风险分析
5. 入场时机和注意事项

使用Markdown格式，条理清晰。"""
        
        user_prompt = f"""我的情况:
- 市场观点: {market_view}
- 风险偏好: {risk_tolerance}
- 可用资金: ¥{capital:,.0f}
- 标的现价: ¥{underlying_price:.2f}

请推荐合适的期权策略。"""
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt)
        ]
        
        return await self.chat(messages)
    
    async def generate_strategy_code(self, description: str) -> str:
        """
        Generate strategy code based on natural language description.
        """
        system_prompt = """你是一位量化策略开发专家。根据用户的策略描述，生成Python代码。

代码应该:
1. 继承BaseStrategy类
2. 实现on_bar和execute方法
3. 包含必要的风控逻辑
4. 有清晰的注释

输出格式:
```python
# 策略代码
```

之后解释代码逻辑。"""
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"请实现以下策略:\n{description}")
        ]
        
        return await self.chat(messages, model="deepseek-coder")


# Singleton instance
_ai_service: Optional[DeepSeekService] = None

def get_ai_service() -> DeepSeekService:
    """Get or create AI service instance."""
    global _ai_service
    if _ai_service is None:
        _ai_service = DeepSeekService()
    return _ai_service
