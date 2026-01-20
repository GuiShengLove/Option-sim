"use client";

import { useState, useEffect, useRef } from 'react';
import { 
  Brain, Minus, Plus, Sparkles, RefreshCw, AlertCircle, CheckCircle
} from 'lucide-react';

interface AIMonitorPanelProps {
  onOpenReport: (type: 'backtest' | 'simulation') => void;
  
  // 是否已运行回测
  hasRun?: boolean;
  
  // 回测时间戳 - 用于触发新的AI评价请求
  runTimestamp?: number | null;
  
  // 策略名称
  strategyName?: string;
  
  // 回测指标 (用于调用AI评价)
  metrics?: {
    total_return?: number;
    max_drawdown?: number;
    sharpe_ratio?: number;
    win_rate?: number;
    profit_factor?: number;
    trade_count?: number;
  };
  
  isBacktestRunning?: boolean;
  isSimulationRunning?: boolean;
}

interface AIEvaluation {
  success: boolean;
  evaluation: string;
  risk_level: string;
  suggestions: string[];
}

/**
 * AI Strategy Monitor Panel - Displays AI-generated strategy evaluation.
 * Uses runTimestamp to trigger evaluation only after backtest completion.
 */
export default function AIMonitorPanel({ 
  onOpenReport,
  hasRun = false,
  runTimestamp,
  strategyName,
  metrics,
  isBacktestRunning = false,
  isSimulationRunning = false,
}: AIMonitorPanelProps) {
  const [isMinimized, setIsMinimized] = useState(false);
  const [evaluation, setEvaluation] = useState<AIEvaluation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Track the last processed timestamp to avoid duplicate requests
  const lastProcessedTimestamp = useRef<number | null>(null);
  // Track current request to cancel stale responses
  const currentRequestId = useRef<number>(0);

  // Fetch AI evaluation only when runTimestamp changes (new backtest completed)
  useEffect(() => {
    // Only trigger when:
    // 1. We have a new runTimestamp different from the last processed one
    // 2. We have all required data
    if (runTimestamp && 
        runTimestamp !== lastProcessedTimestamp.current && 
        hasRun && 
        metrics && 
        strategyName) {
      lastProcessedTimestamp.current = runTimestamp;
      fetchAIEvaluation();
    }
  }, [runTimestamp, hasRun, metrics, strategyName]);

  const fetchAIEvaluation = async () => {
    if (!metrics || !strategyName) return;
    
    // Increment request ID to track this specific request
    const requestId = ++currentRequestId.current;
    
    setIsLoading(true);
    setError(null);
    // Clear previous evaluation while loading new one
    setEvaluation(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/ai/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_name: strategyName,
          metrics: {
            total_return: metrics.total_return ?? 0,
            max_drawdown: metrics.max_drawdown ?? 0,
            sharpe_ratio: metrics.sharpe_ratio ?? 0,
            win_rate: metrics.win_rate ?? 0,
            profit_factor: metrics.profit_factor ?? 0,
          },
          trades_count: metrics.trade_count ?? 0,
        }),
      });
      
      // Check if this request is still the current one (not stale)
      if (requestId !== currentRequestId.current) {
        console.log('Ignoring stale AI evaluation response');
        return;
      }
      
      if (!response.ok) {
        throw new Error('API请求失败');
      }
      
      const data = await response.json();
      
      // Double check before setting state
      if (requestId === currentRequestId.current) {
        setEvaluation(data);
      }
    } catch (err) {
      console.error('AI evaluation error:', err);
      // Only set error if this is still the current request
      if (requestId === currentRequestId.current) {
        setError('AI评价获取失败，使用本地分析');
        setEvaluation(generateLocalEvaluation());
      }
    } finally {
      // Only update loading state if this is still the current request
      if (requestId === currentRequestId.current) {
        setIsLoading(false);
      }
    }
  };

  const generateLocalEvaluation = (): AIEvaluation => {
    const totalReturn = metrics?.total_return ?? 0;
    const sharpe = metrics?.sharpe_ratio ?? 0;
    const maxDD = metrics?.max_drawdown ?? 0;
    const winRate = metrics?.win_rate ?? 0;
    
    let riskLevel = "中";
    if (maxDD > 20 || sharpe < 0) riskLevel = "高";
    else if (maxDD < 10 && sharpe > 0.5) riskLevel = "低";
    
    let evaluation = "";
    if (totalReturn > 10 && sharpe > 1) {
      evaluation = `策略表现优异，收益${totalReturn.toFixed(1)}%，夏普比率${sharpe.toFixed(2)}。`;
    } else if (totalReturn > 0) {
      evaluation = `策略实现正收益${totalReturn.toFixed(1)}%，最大回撤${maxDD.toFixed(1)}%在可控范围。`;
    } else {
      evaluation = `策略录得${totalReturn.toFixed(1)}%亏损，建议重新审视策略逻辑。`;
    }
    
    const suggestions: string[] = [];
    if (winRate < 0.4) suggestions.push("优化入场时机");
    if (maxDD > 15) suggestions.push("加强止损机制");
    if (!suggestions.length) suggestions.push("保持当前策略配置");
    
    return { success: false, evaluation, risk_level: riskLevel, suggestions };
  };

  const getRiskBg = (level: string) => {
    if (level === '低') return 'bg-[var(--accent-success)]';
    if (level === '中') return 'bg-[var(--accent-warning)]';
    return 'bg-[var(--accent-danger)]';
  };

  // Minimized view
  if (isMinimized) {
    return (
      <div className="bg-[var(--bg-secondary)] border-t border-[var(--border-primary)] px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-[var(--accent-primary)]" />
            <span className="text-xs font-semibold text-[var(--text-primary)]">AI 策略评价</span>
          </div>
          
          {hasRun && evaluation && (
            <>
              <div className="h-4 w-px bg-[var(--border-primary)]"></div>
              <div className={`px-2 py-0.5 rounded text-xs font-bold ${getRiskBg(evaluation.risk_level)} text-white`}>
                {evaluation.risk_level}风险
              </div>
              <span className="text-xs text-[var(--text-muted)] truncate max-w-[300px]">
                {evaluation.evaluation}
              </span>
            </>
          )}
          
          {isLoading && (
            <>
              <div className="h-4 w-px bg-[var(--border-primary)]"></div>
              <RefreshCw className="w-3 h-3 text-[var(--accent-primary)] animate-spin" />
              <span className="text-xs text-[var(--text-muted)]">分析中...</span>
            </>
          )}
        </div>
        
        <button
          onClick={() => setIsMinimized(false)}
          className="p-1 hover:bg-[var(--bg-card)] rounded transition-colors"
        >
          <Plus className="w-4 h-4 text-[var(--text-muted)]" />
        </button>
      </div>
    );
  }

  // Expanded view
  return (
    <div className="bg-[var(--bg-secondary)] border-t border-[var(--border-primary)]">
      {/* Header */}
      <div className="px-4 py-2 border-b border-[var(--border-primary)]/50 flex items-center justify-between bg-[var(--bg-primary)]/50">
        <div className="flex items-center gap-3">
          <Brain className="w-5 h-5 text-[var(--accent-primary)]" />
          <span className="text-sm font-bold text-[var(--text-primary)]">AI 策略评价中心</span>
          {evaluation && !isLoading && (
            <div className={`px-2 py-0.5 rounded text-xs font-bold ${getRiskBg(evaluation.risk_level)} text-white`}>
              {evaluation.risk_level}风险
            </div>
          )}
          {isLoading && (
            <span className="text-xs text-[var(--accent-primary)] flex items-center gap-1">
              <RefreshCw className="w-3 h-3 animate-spin" />
              分析中
            </span>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {hasRun && !isLoading && (
            <button
              onClick={fetchAIEvaluation}
              className="px-3 py-1 bg-[var(--accent-primary)] hover:bg-[var(--accent-primary)]/80 text-white text-xs rounded font-semibold transition-colors flex items-center gap-1"
            >
              <RefreshCw className="w-3 h-3" />
              重新分析
            </button>
          )}
          <button
            onClick={() => onOpenReport('backtest')}
            disabled={!hasRun}
            className={`px-3 py-1 text-xs rounded font-semibold transition-colors flex items-center gap-1 ${
              hasRun
                ? 'bg-[var(--bg-card)] hover:bg-[var(--bg-card)]/80 text-[var(--text-primary)] border border-[var(--border-primary)]'
                : 'bg-[var(--bg-card)] text-[var(--text-muted)] cursor-not-allowed'
            }`}
          >
            <Sparkles className="w-3 h-3" />
            详细报告
          </button>
          <button
            onClick={() => setIsMinimized(true)}
            className="p-1 hover:bg-[var(--bg-card)] rounded transition-colors"
          >
            <Minus className="w-4 h-4 text-[var(--text-muted)]" />
          </button>
        </div>
      </div>
      
      {/* Content */}
      <div className="px-4 py-4">
        {!hasRun ? (
          <div className="flex flex-col items-center justify-center text-[var(--text-muted)] py-4">
            <Brain className="w-8 h-8 opacity-30 mb-2" />
            <p className="text-sm">请先运行回测以获取AI策略评价</p>
          </div>
        ) : isLoading ? (
          <div className="flex items-center justify-center py-6">
            <RefreshCw className="w-6 h-6 text-[var(--accent-primary)] animate-spin mr-3" />
            <span className="text-sm text-[var(--text-secondary)]">AI正在分析策略表现...</span>
          </div>
        ) : evaluation ? (
          <div className="space-y-4">
            {/* Main Evaluation */}
            <div className="flex items-start gap-3">
              <div className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                evaluation.success ? 'bg-[var(--accent-primary)]/20' : 'bg-[var(--accent-warning)]/20'
              }`}>
                {evaluation.success ? (
                  <Sparkles className="w-4 h-4 text-[var(--accent-primary)]" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-[var(--accent-warning)]" />
                )}
              </div>
              <div className="flex-1">
                <p className="text-sm text-[var(--text-primary)] leading-relaxed">
                  {evaluation.evaluation}
                </p>
              </div>
            </div>
            
            {/* Suggestions */}
            {evaluation.suggestions.length > 0 && (
              <div className="border-t border-[var(--border-primary)]/50 pt-3">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="w-4 h-4 text-[var(--accent-success)]" />
                  <span className="text-xs font-bold text-[var(--text-primary)]">改进建议</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {evaluation.suggestions.map((suggestion, idx) => (
                    <span 
                      key={idx} 
                      className="px-3 py-1 bg-[var(--bg-card)] text-xs text-[var(--text-secondary)] rounded-full border border-[var(--border-primary)]"
                    >
                      {suggestion}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Quick Stats */}
            <div className="border-t border-[var(--border-primary)]/50 pt-3 grid grid-cols-5 gap-4 text-center">
              <div>
                <div className="text-[10px] text-[var(--text-muted)]">收益率</div>
                <div className={`text-sm font-bold ${(metrics?.total_return ?? 0) >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'}`}>
                  {(metrics?.total_return ?? 0) >= 0 ? '+' : ''}{(metrics?.total_return ?? 0).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-[10px] text-[var(--text-muted)]">最大回撤</div>
                <div className="text-sm font-bold text-[var(--accent-danger)]">
                  -{(metrics?.max_drawdown ?? 0).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-[10px] text-[var(--text-muted)]">夏普比率</div>
                <div className={`text-sm font-bold ${(metrics?.sharpe_ratio ?? 0) >= 1 ? 'text-[var(--accent-success)]' : 'text-[var(--text-secondary)]'}`}>
                  {(metrics?.sharpe_ratio ?? 0).toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-[var(--text-muted)]">胜率</div>
                <div className={`text-sm font-bold ${(metrics?.win_rate ?? 0) >= 0.5 ? 'text-[var(--accent-success)]' : 'text-[var(--text-secondary)]'}`}>
                  {((metrics?.win_rate ?? 0) * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-[10px] text-[var(--text-muted)]">交易次数</div>
                <div className="text-sm font-bold text-[var(--text-primary)]">
                  {metrics?.trade_count ?? 0}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center py-4 text-[var(--text-muted)]">
            <AlertCircle className="w-5 h-5 mr-2" />
            <span className="text-sm">{error || '等待回测完成后分析'}</span>
          </div>
        )}
      </div>
      
      {/* Running Status */}
      {(isBacktestRunning || isSimulationRunning) && (
        <div className="px-4 py-2 border-t border-[var(--border-primary)]/50 bg-[var(--accent-warning)]/5 flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-[var(--accent-warning)] animate-pulse"></div>
          <span className="text-xs text-[var(--accent-warning)]">
            {isBacktestRunning ? '回测进行中...' : '模拟进行中...'}
          </span>
        </div>
      )}
    </div>
  );
}
