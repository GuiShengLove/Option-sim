"use client";

import { useState } from 'react';
import { 
  Brain, Minus, Plus, CheckCircle, AlertTriangle, TrendingUp, 
  TrendingDown, Activity, Sparkles, Shield, PieChart, BarChart3, Settings
} from 'lucide-react';
import AIReportModal from './AIReportModal';

interface AIHintBarProps {
  // Real-time metrics from parent
  totalDelta?: number;
  totalGamma?: number;
  totalTheta?: number;
  totalVega?: number;
  marginUsage?: number;
  unrealizedPnL?: number;
  riskScore?: number;
  isBacktestRunning?: boolean;
  isSimulationRunning?: boolean;
  backtestResults?: any;  // Pass backtest results for report generation
}

/**
 * Enhanced AI hint bar with rich content when expanded.
 * Shows Greeks exposure, risk metrics, and quick analysis.
 */
export default function AIHintBar({ 
  totalDelta = 0.35,
  totalGamma = 0.08,
  totalTheta = -12.5,
  totalVega = 45.2,
  marginUsage = 0.42,
  unrealizedPnL = 2340,
  riskScore = 3,
  isBacktestRunning = false,
  isSimulationRunning = false,
  backtestResults
}: AIHintBarProps) {
  const [isMinimized, setIsMinimized] = useState(false);
  const [showReport, setShowReport] = useState(false);

  const getRiskColor = () => {
    if (riskScore <= 3) return 'text-[var(--accent-success)]';
    if (riskScore <= 6) return 'text-[var(--accent-warning)]';
    return 'text-[var(--accent-danger)]';
  };

  const getRiskLabel = () => {
    if (riskScore <= 3) return '低风险';
    if (riskScore <= 6) return '中风险';
    return '高风险';
  };

  const getRiskIcon = () => {
    if (riskScore <= 3) return <CheckCircle className="w-4 h-4 text-[var(--accent-success)]" />;
    if (riskScore <= 6) return <AlertTriangle className="w-4 h-4 text-[var(--accent-warning)]" />;
    return <AlertTriangle className="w-4 h-4 text-[var(--accent-danger)]" />;
  };

  // Generate quick analysis based on current state
  const getQuickAnalysis = () => {
    const insights: string[] = [];
    
    if (Math.abs(totalDelta) > 0.5) {
      insights.push(`方向敞口偏${totalDelta > 0 ? '多' : '空'}`);
    }
    if (Math.abs(totalTheta) > 20) {
      insights.push('Theta衰减较快');
    }
    if (totalVega > 50) {
      insights.push('波动率敏感度高');
    }
    if (marginUsage > 0.6) {
      insights.push('保证金使用率偏高');
    }
    
    return insights.length > 0 ? insights.join('，') : '组合风险平衡';
  };

  // Minimized view - just status bar
  if (isMinimized) {
    return (
      <div className="bg-[var(--bg-secondary)] border-t border-[var(--border-primary)] px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-[var(--accent-primary)]" />
            <span className="text-xs font-semibold text-[var(--text-primary)]">AI 策略监控</span>
          </div>
          
          <div className="h-4 w-px bg-[var(--border-primary)]"></div>
          
          {getRiskIcon()}
          <span className={`text-xs font-mono ${getRiskColor()}`}>{getRiskLabel()}</span>
          
          <div className="h-4 w-px bg-[var(--border-primary)]"></div>
          
          <span className="text-xs text-[var(--text-muted)]">{getQuickAnalysis()}</span>
          
          {(isBacktestRunning || isSimulationRunning) && (
            <>
              <div className="h-4 w-px bg-[var(--border-primary)]"></div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-[var(--accent-warning)] animate-pulse"></div>
                <span className="text-xs text-[var(--accent-warning)]">
                  {isBacktestRunning ? '回测中' : '模拟中'}
                </span>
              </div>
            </>
          )}
        </div>
        
        <button
          onClick={() => setIsMinimized(false)}
          className="p-1 hover:bg-[var(--bg-card)] rounded transition-colors"
          title="展开"
        >
          <Plus className="w-4 h-4 text-[var(--text-muted)]" />
        </button>
      </div>
    );
  }

  // Expanded view - full details
  return (
    <div className="bg-[var(--bg-secondary)] border-t border-[var(--border-primary)]">
      {/* Header Row */}
      <div className="px-4 py-2 border-b border-[var(--border-primary)]/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Brain className="w-4 h-4 text-[var(--accent-primary)]" />
          <span className="text-sm font-semibold text-[var(--text-primary)]">AI 策略监控中心</span>
          <span className="text-xs text-[var(--text-muted)]">实时分析</span>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowReport(true)}
            className="px-3 py-1 bg-[var(--accent-primary)]/10 hover:bg-[var(--accent-primary)]/20 text-[var(--accent-primary)] text-xs rounded border border-[var(--accent-primary)]/20 transition-colors flex items-center gap-1"
          >
            <Sparkles className="w-3 h-3" />
            生成详细报告
          </button>
          <button
            onClick={() => setIsMinimized(true)}
            className="p-1 hover:bg-[var(--bg-card)] rounded transition-colors"
            title="最小化"
          >
            <Minus className="w-4 h-4 text-[var(--text-muted)]" />
          </button>
        </div>
      </div>
      
      {/* Content Grid */}
      <div className="px-4 py-3 grid grid-cols-12 gap-4">
        {/* Greeks Exposure - 4 cols */}
        <div className="col-span-4 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <PieChart className="w-4 h-4 text-[var(--accent-secondary)]" />
            <span className="text-xs font-semibold text-[var(--text-secondary)]">Greeks敞口</span>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="text-center">
              <div className="text-xs text-[var(--text-muted)]">Δ</div>
              <div className={`text-sm font-mono font-bold ${totalDelta >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'}`}>
                {totalDelta >= 0 ? '+' : ''}{totalDelta.toFixed(2)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-[var(--text-muted)]">Γ</div>
              <div className="text-sm font-mono font-bold text-[var(--accent-warning)]">
                {totalGamma.toFixed(3)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-[var(--text-muted)]">Θ/日</div>
              <div className="text-sm font-mono font-bold text-[var(--accent-danger)]">
                {totalTheta.toFixed(1)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-[var(--text-muted)]">ν</div>
              <div className="text-sm font-mono font-bold text-[var(--accent-primary)]">
                {totalVega.toFixed(1)}
              </div>
            </div>
          </div>
        </div>
        
        {/* Risk Metrics - 3 cols */}
        <div className="col-span-3 flex items-center gap-4 border-l border-[var(--border-primary)]/50 pl-4">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-[var(--accent-danger)]" />
            <span className="text-xs font-semibold text-[var(--text-secondary)]">风险</span>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="text-center">
              <div className="text-xs text-[var(--text-muted)]">评分</div>
              <div className={`text-sm font-mono font-bold ${getRiskColor()}`}>
                {riskScore}/10
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-[var(--text-muted)]">保证金</div>
              <div className={`text-sm font-mono font-bold ${marginUsage > 0.7 ? 'text-[var(--accent-danger)]' : 'text-[var(--text-primary)]'}`}>
                {(marginUsage * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
        
        {/* P&L - 2 cols */}
        <div className="col-span-2 flex items-center gap-3 border-l border-[var(--border-primary)]/50 pl-4">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-[var(--accent-success)]" />
            <span className="text-xs font-semibold text-[var(--text-secondary)]">盈亏</span>
          </div>
          <div className={`text-lg font-mono font-bold ${unrealizedPnL >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'}`}>
            {unrealizedPnL >= 0 ? '+' : ''}¥{unrealizedPnL.toLocaleString()}
          </div>
        </div>
        
        {/* AI Insight - 3 cols */}
        <div className="col-span-3 flex items-center gap-3 border-l border-[var(--border-primary)]/50 pl-4">
          <div className="flex items-center gap-2">
            {getRiskIcon()}
            <span className="text-xs text-[var(--text-muted)]">{getRiskLabel()}</span>
          </div>
          <span className="text-xs text-[var(--text-secondary)]">
            {getQuickAnalysis()}
          </span>
        </div>
      </div>
      
      {/* Running Status Bar */}
      {(isBacktestRunning || isSimulationRunning) && (
        <div className="px-4 py-2 border-t border-[var(--border-primary)]/50 bg-[var(--accent-warning)]/5 flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-[var(--accent-warning)] animate-pulse"></div>
          <span className="text-xs text-[var(--accent-warning)]">
            {isBacktestRunning ? '回测进行中...' : '蒙特卡洛模拟进行中...'}
          </span>
          <div className="flex-1 h-1 bg-[var(--bg-card)] rounded-full overflow-hidden">
            <div className="h-full bg-[var(--accent-warning)] animate-pulse" style={{ width: '60%' }}></div>
          </div>
        </div>
      )}

      {/* AI Report Modal */}
      <AIReportModal
        isOpen={showReport}
        onClose={() => setShowReport(false)}
        reportType="backtest"
        backtestData={backtestResults || {
          strategyName: '当前持仓组合',
          startDate: '2024-01-01',
          endDate: '2024-12-31',
          initialCapital: 1000000,
          finalEquity: 1124000,
          totalReturn: 0.124,
          sharpeRatio: 1.85,
          maxDrawdown: -0.082,
          winRate: 0.58,
          totalTrades: 156,
        }}
      />
    </div>
  );
}
