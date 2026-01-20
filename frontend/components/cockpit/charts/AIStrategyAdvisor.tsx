"use client";

import { useState, useEffect } from 'react';
import { 
  Lightbulb, TrendingUp, TrendingDown, Minus, Activity, Zap, 
  ChevronRight, AlertCircle, CheckCircle, Loader2, Send
} from 'lucide-react';

interface StrategyLeg {
  type: 'call' | 'put';
  action: 'buy' | 'sell';
  strike_offset: number;
  expiry_days: number;
  quantity: number;
}

interface Strategy {
  name: string;
  chinese_name: string;
  type: string;
  description: string;
  max_profit: string;
  max_loss: string;
  breakeven: string;
  time_decay: string;
  ideal_iv: string;
  risk_level: number;
  legs: StrategyLeg[];
}

interface AIStrategyAdvisorProps {
  spotPrice?: number;
  onStrategySelect?: (strategy: Strategy) => void;
}

export default function AIStrategyAdvisor({ spotPrice = 3.0, onStrategySelect }: AIStrategyAdvisorProps) {
  // Form state
  const [marketView, setMarketView] = useState<'bullish' | 'bearish' | 'neutral' | 'volatile' | 'range_bound'>('bullish');
  const [riskTolerance, setRiskTolerance] = useState<'conservative' | 'moderate' | 'aggressive'>('moderate');
  const [capital, setCapital] = useState(100000);
  
  // Results state
  const [recommendations, setRecommendations] = useState<Strategy[]>([]);
  const [aiAnalysis, setAIAnalysis] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  
  // Chat state
  const [chatMessage, setChatMessage] = useState('');
  const [chatResponse, setChatResponse] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  const getMarketViewIcon = (view: string) => {
    switch (view) {
      case 'bullish': return <TrendingUp className="w-4 h-4 text-[var(--accent-success)]" />;
      case 'bearish': return <TrendingDown className="w-4 h-4 text-[var(--accent-danger)]" />;
      case 'volatile': return <Activity className="w-4 h-4 text-[var(--accent-warning)]" />;
      default: return <Minus className="w-4 h-4 text-[var(--text-muted)]" />;
    }
  };

  const getRiskColor = (level: number) => {
    if (level <= 2) return 'text-[var(--accent-success)]';
    if (level <= 3) return 'text-[var(--accent-warning)]';
    return 'text-[var(--accent-danger)]';
  };

  const handleGetRecommendations = async () => {
    setLoading(true);
    setRecommendations([]);
    setAIAnalysis('');
    
    try {
      const response = await fetch('http://localhost:8000/api/ai/recommend-strategy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          market_view: marketView,
          risk_tolerance: riskTolerance,
          capital,
          underlying_price: spotPrice,
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setRecommendations(data.recommended_strategies || []);
        setAIAnalysis(data.ai_analysis || '');
      } else {
        // Fallback to local recommendations
        setRecommendations([
          {
            name: "Bull Call Spread",
            chinese_name: "牛市看涨价差",
            type: "directional",
            description: "买入较低行权价认购，卖出较高行权价认购。成本更低的看涨策略。",
            max_profit: "高行权价 - 低行权价 - 净权利金",
            max_loss: "净权利金",
            breakeven: "低行权价 + 净权利金",
            time_decay: "接近平值时负向",
            ideal_iv: "中等",
            risk_level: 2,
            legs: [
              { type: 'call', action: 'buy', strike_offset: 0, expiry_days: 30, quantity: 1 },
              { type: 'call', action: 'sell', strike_offset: 2, expiry_days: 30, quantity: 1 },
            ],
          },
        ]);
        setAIAnalysis("基于您的市场观点，推荐使用牛市策略。请根据实际情况调整。");
      }
    } catch (error) {
      console.error('Failed to get recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleChat = async () => {
    if (!chatMessage.trim()) return;
    
    setChatLoading(true);
    
    try {
      const response = await fetch('http://localhost:8000/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: chatMessage,
          context: selectedStrategy ? `当前选中策略: ${selectedStrategy.chinese_name}` : undefined,
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setChatResponse(data.response);
      } else {
        setChatResponse("AI服务暂时不可用，请稍后再试。");
      }
    } catch (error) {
      setChatResponse("连接AI服务失败，请检查网络。");
    } finally {
      setChatLoading(false);
      setChatMessage('');
    }
  };

  const handleSelectStrategy = (strategy: Strategy) => {
    setSelectedStrategy(strategy);
    onStrategySelect?.(strategy);
  };

  return (
    <div className="space-y-6">
      {/* Configuration Panel */}
      <div className="glass-card p-4">
        <div className="section-header">
          <Lightbulb className="w-4 h-4 text-[var(--accent-warning)]" />
          <h3 className="section-title">AI 策略顾问</h3>
        </div>
        
        <div className="grid grid-cols-4 gap-4 mt-4">
          <div>
            <label className="label-mono block mb-2">市场观点</label>
            <div className="grid grid-cols-2 gap-2">
              {[
                { value: 'bullish', label: '🐂 看涨', color: 'success' },
                { value: 'bearish', label: '🐻 看跌', color: 'danger' },
                { value: 'neutral', label: '📊 中性', color: 'secondary' },
                { value: 'volatile', label: '⚡ 高波动', color: 'warning' },
              ].map(opt => (
                <button
                  key={opt.value}
                  onClick={() => setMarketView(opt.value as any)}
                  className={`px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                    marketView === opt.value 
                      ? `bg-[var(--accent-${opt.color})] text-white` 
                      : 'bg-[var(--bg-card)] text-[var(--text-secondary)] hover:bg-[var(--bg-card-hover)]'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <label className="label-mono block mb-2">风险偏好</label>
            <select 
              value={riskTolerance}
              onChange={(e) => setRiskTolerance(e.target.value as any)}
              className="select-field w-full"
            >
              <option value="conservative">🛡️ 保守型</option>
              <option value="moderate">⚖️ 平衡型</option>
              <option value="aggressive">🚀 进取型</option>
            </select>
          </div>
          
          <div>
            <label className="label-mono block mb-2">可用资金 (¥)</label>
            <input 
              type="number" 
              value={capital}
              onChange={(e) => setCapital(parseInt(e.target.value) || 0)}
              className="input-field w-full"
            />
          </div>
          
          <div className="flex items-end">
            <button
              onClick={handleGetRecommendations}
              disabled={loading}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Zap className="w-4 h-4" />
              )}
              获取推荐
            </button>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="glass-card p-4">
          <div className="section-header">
            <CheckCircle className="w-4 h-4 text-[var(--accent-success)]" />
            <h3 className="section-title">推荐策略</h3>
            <span className="text-xs text-[var(--text-muted)] ml-auto">{recommendations.length} 个匹配</span>
          </div>
          
          <div className="grid grid-cols-1 gap-3 mt-4">
            {recommendations.map((strategy, idx) => (
              <div 
                key={idx}
                onClick={() => handleSelectStrategy(strategy)}
                className={`p-4 rounded-lg cursor-pointer transition-all ${
                  selectedStrategy?.name === strategy.name 
                    ? 'bg-[var(--accent-primary)]/20 border border-[var(--accent-primary)]' 
                    : 'bg-[var(--bg-card)] border border-[var(--border-primary)] hover:border-[var(--accent-primary)]'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="font-bold text-[var(--text-primary)]">{strategy.chinese_name}</span>
                      <span className="text-xs text-[var(--text-muted)]">({strategy.name})</span>
                      <span className={`text-xs font-mono ${getRiskColor(strategy.risk_level)}`}>
                        风险: {strategy.risk_level}/5
                      </span>
                    </div>
                    <p className="text-sm text-[var(--text-secondary)] mb-3">{strategy.description}</p>
                    
                    {/* Strategy Legs */}
                    <div className="flex flex-wrap gap-2">
                      {strategy.legs.map((leg, legIdx) => (
                        <span 
                          key={legIdx}
                          className={`px-2 py-1 rounded text-xs ${
                            leg.action === 'buy' 
                              ? 'bg-[var(--accent-success)]/20 text-[var(--accent-success)]' 
                              : 'bg-[var(--accent-danger)]/20 text-[var(--accent-danger)]'
                          }`}
                        >
                          {leg.action === 'buy' ? '买' : '卖'} {leg.type === 'call' ? '认购' : '认沽'} 
                          {leg.strike_offset >= 0 ? '+' : ''}{leg.strike_offset}档
                        </span>
                      ))}
                    </div>
                  </div>
                  <ChevronRight className="w-5 h-5 text-[var(--text-muted)]" />
                </div>
                
                {selectedStrategy?.name === strategy.name && (
                  <div className="mt-4 pt-4 border-t border-[var(--border-primary)] grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-[var(--text-muted)]">最大收益:</span>
                      <span className="ml-2 text-[var(--accent-success)]">{strategy.max_profit}</span>
                    </div>
                    <div>
                      <span className="text-[var(--text-muted)]">最大亏损:</span>
                      <span className="ml-2 text-[var(--accent-danger)]">{strategy.max_loss}</span>
                    </div>
                    <div>
                      <span className="text-[var(--text-muted)]">时间价值:</span>
                      <span className="ml-2">{strategy.time_decay}</span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* AI Analysis */}
      {aiAnalysis && (
        <div className="glass-card p-4">
          <div className="section-header">
            <AlertCircle className="w-4 h-4 text-[var(--accent-primary)]" />
            <h3 className="section-title">AI 分析</h3>
          </div>
          <div className="mt-4 prose prose-invert prose-sm max-w-none">
            <div className="whitespace-pre-wrap text-sm text-[var(--text-secondary)]">
              {aiAnalysis}
            </div>
          </div>
        </div>
      )}

      {/* AI Chat */}
      <div className="glass-card p-4">
        <div className="section-header">
          <Lightbulb className="w-4 h-4 text-[var(--accent-secondary)]" />
          <h3 className="section-title">AI 助手对话</h3>
        </div>
        
        <div className="mt-4 space-y-4">
          {chatResponse && (
            <div className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-primary)]">
              <div className="text-sm text-[var(--text-secondary)] whitespace-pre-wrap">
                {chatResponse}
              </div>
            </div>
          )}
          
          <div className="flex gap-2">
            <input
              type="text"
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleChat()}
              placeholder="询问有关期权策略的问题..."
              className="input-field flex-1"
            />
            <button
              onClick={handleChat}
              disabled={chatLoading || !chatMessage.trim()}
              className="btn-primary px-4"
            >
              {chatLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
