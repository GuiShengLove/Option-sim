"use client";

import { useState, useEffect } from 'react';
import { 
  X, Brain, Sparkles, FileText, AlertTriangle, 
  CheckCircle, TrendingUp, TrendingDown, Loader2,
  Download, Copy, Check
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

type ReportType = 'backtest' | 'simulation';

interface BacktestData {
  strategyName: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  finalEquity: number;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  profitFactor?: number;
  avgWin?: number;
  avgLoss?: number;
}

interface SimulationData {
  model: string;
  numPaths: number;
  numDays: number;
  meanReturn: number;
  stdDev: number;
  var95: number;
  cvar95: number;
  maxDrawdown: number;
  sharpe: number;
  strategyWinRate?: number;
  strategyAvgPnL?: number;
}

interface AIReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  reportType: ReportType;
  backtestData?: BacktestData;
  simulationData?: SimulationData;
}

export default function AIReportModal({
  isOpen,
  onClose,
  reportType,
  backtestData,
  simulationData
}: AIReportModalProps) {
  const [report, setReport] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [reportSource, setReportSource] = useState<'ai' | 'local' | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    if (isOpen && !report) {
      generateReport();
    }
  }, [isOpen]);

  const generateReport = async (isRetry = false) => {
    setLoading(true);
    setReport('');
    setReportSource(null);
    if (!isRetry) setRetryCount(0);
    
    try {
      // Build context based on report type
      let context = '';
      let endpoint = '';
      
      if (reportType === 'backtest' && backtestData) {
        context = JSON.stringify({
          type: 'backtest',
          strategy_name: backtestData.strategyName,
          period: `${backtestData.startDate} 至 ${backtestData.endDate}`,
          initial_capital: backtestData.initialCapital,
          final_equity: backtestData.finalEquity,
          total_return: backtestData.totalReturn,
          sharpe_ratio: backtestData.sharpeRatio,
          max_drawdown: backtestData.maxDrawdown,
          win_rate: backtestData.winRate,
          total_trades: backtestData.totalTrades,
          profit_factor: backtestData.profitFactor,
          avg_win: backtestData.avgWin,
          avg_loss: backtestData.avgLoss,
        });
        endpoint = '/api/ai/analyze-risk';
      } else if (reportType === 'simulation' && simulationData) {
        context = JSON.stringify({
          type: 'simulation',
          model: simulationData.model,
          num_paths: simulationData.numPaths,
          num_days: simulationData.numDays,
          mean_return: simulationData.meanReturn,
          std_dev: simulationData.stdDev,
          var_95: simulationData.var95,
          cvar_95: simulationData.cvar95,
          max_drawdown: simulationData.maxDrawdown,
          sharpe: simulationData.sharpe,
          strategy_win_rate: simulationData.strategyWinRate,
          strategy_avg_pnl: simulationData.strategyAvgPnL,
        });
        endpoint = '/api/ai/analyze-risk';
      }
      
      // Call AI API
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pnl_data: { context },
          sharpe: reportType === 'backtest' ? backtestData?.sharpeRatio || 0 : simulationData?.sharpe || 0,
          max_drawdown: reportType === 'backtest' ? backtestData?.maxDrawdown || 0 : simulationData?.maxDrawdown || 0,
          win_rate: reportType === 'backtest' ? backtestData?.winRate : simulationData?.strategyWinRate,
          strategy_name: reportType === 'backtest' ? backtestData?.strategyName : `${simulationData?.model} 蒙特卡洛模拟`
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setReport(data.report_markdown);
        setReportSource('ai');
      } else {
        // Fallback to local generation
        setReport(generateLocalReport());
        setReportSource('local');
      }
    } catch (error) {
      console.error('AI Report generation failed:', error);
      setReport(generateLocalReport());
      setReportSource('local');
      setRetryCount(prev => prev + 1);
    } finally {
      setLoading(false);
    }
  };

  const generateLocalReport = (): string => {
    if (reportType === 'backtest' && backtestData) {
      return generateBacktestReport(backtestData);
    } else if (reportType === 'simulation' && simulationData) {
      return generateSimulationReport(simulationData);
    }
    return '无法生成报告：缺少数据';
  };

  const generateBacktestReport = (data: BacktestData): string => {
    const returnPct = (data.totalReturn * 100).toFixed(2);
    const ddPct = (Math.abs(data.maxDrawdown) * 100).toFixed(2);
    const winPct = (data.winRate * 100).toFixed(1);
    
    let riskLevel = '低';
    let riskAlert = '> [!NOTE]\n> **风险可控** - 回撤在合理范围内';
    
    if (Math.abs(data.maxDrawdown) > 0.20) {
      riskLevel = '高';
      riskAlert = '> [!CAUTION]\n> **高风险警告** - 最大回撤超过20%，需要立即检查杠杆和止损设置';
    } else if (Math.abs(data.maxDrawdown) > 0.10) {
      riskLevel = '中';
      riskAlert = '> [!WARNING]\n> **中等风险** - 建议增加止损保护或降低头寸规模';
    }
    
    let performanceVerdict = '🔧 需要优化';
    if (data.sharpeRatio > 2.0 && data.winRate > 0.55) {
      performanceVerdict = '🌟 专业级表现';
    } else if (data.sharpeRatio > 1.0) {
      performanceVerdict = '📊 良好表现';
    }
    
    return `# 📊 回测分析报告

## 策略概览
- **策略名称**: ${data.strategyName}
- **回测周期**: ${data.startDate} 至 ${data.endDate}
- **初始资金**: ¥${data.initialCapital.toLocaleString()}
- **最终权益**: ¥${data.finalEquity.toLocaleString()}

---

## 绩效指标

| 指标 | 数值 | 评级 |
|------|------|------|
| 总收益率 | ${returnPct}% | ${data.totalReturn > 0 ? '📈' : '📉'} |
| 夏普比率 | ${data.sharpeRatio.toFixed(2)} | ${data.sharpeRatio > 1.5 ? '优秀' : data.sharpeRatio > 1 ? '良好' : '一般'} |
| 最大回撤 | -${ddPct}% | ${riskLevel}风险 |
| 胜率 | ${winPct}% | ${data.winRate > 0.55 ? '偏高' : '正常'} |
| 交易次数 | ${data.totalTrades} | - |

---

## 风险评估

${riskAlert}

### 风险分解
- **方向性风险**: ${Math.abs(data.maxDrawdown) > 0.15 ? '较高，策略对趋势依赖强' : '可控'}
- **时间风险**: ${data.totalTrades < 50 ? '样本量偏少，统计显著性不足' : '样本量充足'}
- **流动性风险**: 需关注高换手率时段的滑点影响

---

## 改进建议

${data.sharpeRatio < 1.5 ? `1. **提升风险调整收益**: 当前夏普比率${data.sharpeRatio.toFixed(2)}偏低，考虑:
   - 增加多空对冲降低方向敞口
   - 优化入场时机减少逆势交易` : '1. **收益质量**: 夏普比率优秀，继续保持当前风控水平'}

${Math.abs(data.maxDrawdown) > 0.15 ? `2. **回撤控制**: 最大回撤${ddPct}%偏高，建议:
   - 设置硬性止损线 (如-10%)
   - 在高波动期降低杠杆` : '2. **回撤表现**: 回撤控制良好'}

${data.winRate < 0.50 ? `3. **提高胜率**: 当前胜率${winPct}%偏低，需确保盈亏比足够高` : '3. **胜率稳定**: 保持当前选股/择时逻辑'}

---

## 综合评定

**${performanceVerdict}**

${data.sharpeRatio > 1.5 && Math.abs(data.maxDrawdown) < 0.15 
  ? '该策略展现出良好的风险调整收益，可考虑逐步增加资金配置。' 
  : '建议继续优化后再投入实盘验证。'}
`;
  };

  const generateSimulationReport = (data: SimulationData): string => {
    const meanPct = (data.meanReturn * 100).toFixed(2);
    const stdPct = (data.stdDev * 100).toFixed(2);
    const varPct = (data.var95 * 100).toFixed(2);
    const cvarPct = (data.cvar95 * 100).toFixed(2);
    const ddPct = (data.maxDrawdown * 100).toFixed(2);
    
    let riskAlert = '> [!NOTE]\n> **风险可控**';
    if (Math.abs(data.var95) > 0.10) {
      riskAlert = '> [!CAUTION]\n> **尾部风险显著** - 95% VaR超过10%，极端情况下可能遭受重大损失';
    } else if (Math.abs(data.var95) > 0.05) {
      riskAlert = '> [!WARNING]\n> **需关注尾部风险** - 建议增加对冲头寸';
    }
    
    return `# 🎲 蒙特卡洛模拟分析报告

## 模拟配置
- **模型**: ${data.model}
- **路径数量**: ${data.numPaths.toLocaleString()} 条
- **模拟天数**: ${data.numDays} 天

---

## 收益分布统计

| 指标 | 数值 | 说明 |
|------|------|------|
| 平均收益 | ${meanPct}% | 期望收益率 |
| 标准差 | ${stdPct}% | 波动率 |
| VaR (95%) | ${varPct}% | 最坏5%情况的损失 |
| CVaR (95%) | ${cvarPct}% | 极端情况平均损失 |
| 最大回撤 | ${ddPct}% | 路径平均最大回撤 |
| 夏普比率 | ${data.sharpe.toFixed(2)} | 风险调整收益 |

---

## 风险评估

${riskAlert}

### 尾部风险分析
- **VaR解读**: 有5%的概率在${data.numDays}天内损失超过${Math.abs(parseFloat(varPct))}%
- **CVaR解读**: 在最坏5%的情况下，平均损失为${Math.abs(parseFloat(cvarPct))}%
- **压力情景**: 若发生${data.model === 'MJD' || data.model === 'GARCH' ? '跳跃/厚尾' : '3σ'}事件，损失可能更大

${data.strategyWinRate !== undefined ? `
---

## 策略预演结果

| 指标 | 数值 |
|------|------|
| 胜率 | ${(data.strategyWinRate * 100).toFixed(1)}% |
| 平均盈亏 | ¥${data.strategyAvgPnL?.toFixed(0) || 'N/A'} |

` : ''}

---

## 改进建议

1. **对冲尾部风险**: ${Math.abs(data.var95) > 0.05 
   ? '考虑买入虚值Put作为尾部保护' 
   : '当前风险水平可接受'}

2. **波动率管理**: ${data.stdDev > 0.15 
   ? '波动率偏高，建议增加日历价差或卖出期权降低波动敞口' 
   : '波动率适中'}

3. **情景扩展**: 建议使用多种模型(GBM/Heston/MJD)对比，验证结果稳健性

---

## 结论

基于${data.numPaths}条模拟路径的分析，该配置在${data.numDays}天周期内展现出${data.sharpe > 1 ? '可接受' : '需优化'}的风险收益特征。

${Math.abs(data.cvar95) < 0.10 
  ? '✅ 适合作为实盘策略的风险预评估参考' 
  : '⚠️ 建议增加风控措施后再用于实盘'}
`;
  };

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(report);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[var(--bg-primary)] rounded-xl shadow-2xl border border-[var(--border-primary)] w-[800px] max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[var(--border-primary)]">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[var(--accent-primary)]/20">
              <Brain className="w-5 h-5 text-[var(--accent-primary)]" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-[var(--text-primary)]">
                AI 分析报告
              </h2>
              <p className="text-xs text-[var(--text-muted)]">
                {reportType === 'backtest' ? '回测绩效分析' : '蒙特卡洛模拟分析'}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={copyToClipboard}
              className="p-2 hover:bg-[var(--bg-card)] rounded-lg transition-colors"
              title="复制报告"
            >
              {copied ? (
                <Check className="w-5 h-5 text-[var(--accent-success)]" />
              ) : (
                <Copy className="w-5 h-5 text-[var(--text-muted)]" />
              )}
            </button>
            <button
              onClick={generateReport}
              disabled={loading}
              className="p-2 hover:bg-[var(--bg-card)] rounded-lg transition-colors"
              title="重新生成"
            >
              <Sparkles className={`w-5 h-5 ${loading ? 'text-[var(--accent-warning)] animate-pulse' : 'text-[var(--text-muted)]'}`} />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-[var(--bg-card)] rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-[var(--text-muted)]" />
            </button>
          </div>
        </div>
        
        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-64 space-y-4">
              <Loader2 className="w-10 h-10 text-[var(--accent-primary)] animate-spin" />
              <p className="text-[var(--text-secondary)]">正在生成AI分析报告...</p>
              <p className="text-xs text-[var(--text-muted)]">分析绩效指标、计算风险敞口、生成改进建议</p>
            </div>
          ) : (
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown
                components={{
                  h1: ({children}) => (
                    <h1 className="text-xl font-bold text-[var(--text-primary)] mb-4 pb-2 border-b border-[var(--border-primary)]">{children}</h1>
                  ),
                  h2: ({children}) => (
                    <h2 className="text-lg font-semibold text-[var(--accent-primary)] mt-6 mb-3">{children}</h2>
                  ),
                  h3: ({children}) => (
                    <h3 className="text-base font-medium text-[var(--text-secondary)] mt-4 mb-2">{children}</h3>
                  ),
                  table: ({children}) => (
                    <table className="w-full border-collapse my-4">{children}</table>
                  ),
                  th: ({children}) => (
                    <th className="text-left py-2 px-3 bg-[var(--bg-card)] border border-[var(--border-primary)] text-xs font-semibold text-[var(--text-muted)]">{children}</th>
                  ),
                  td: ({children}) => (
                    <td className="py-2 px-3 border border-[var(--border-primary)] text-sm">{children}</td>
                  ),
                  blockquote: ({children}) => (
                    <div className="border-l-4 border-[var(--accent-primary)] pl-4 py-2 my-4 bg-[var(--bg-card)] rounded-r-lg">
                      {children}
                    </div>
                  ),
                  ul: ({children}) => (
                    <ul className="list-disc list-inside space-y-1 my-3">{children}</ul>
                  ),
                  ol: ({children}) => (
                    <ol className="list-decimal list-inside space-y-1 my-3">{children}</ol>
                  ),
                  strong: ({children}) => (
                    <strong className="font-semibold text-[var(--text-primary)]">{children}</strong>
                  ),
                  hr: () => (
                    <hr className="my-6 border-[var(--border-primary)]" />
                  ),
                }}
              >
                {report}
              </ReactMarkdown>
            </div>
          )}
        </div>
        
        {/* Footer */}
        <div className="p-4 border-t border-[var(--border-primary)] bg-[var(--bg-secondary)]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <p className="text-xs text-[var(--text-muted)]">
                {reportSource === 'ai' ? (
                  <span className="text-[var(--accent-success)]">✨ AI模型分析</span>
                ) : reportSource === 'local' ? (
                  <span className="text-[var(--accent-warning)]">📋 本地模板生成</span>
                ) : (
                  <span>🤖 由 DeepSeek AI 提供分析支持</span>
                )}
              </p>
              {reportSource === 'local' && retryCount < 3 && (
                <button
                  onClick={() => generateReport(true)}
                  className="text-xs text-[var(--accent-primary)] hover:underline"
                  disabled={loading}
                >
                  重试AI分析
                </button>
              )}
            </div>
            <button
              onClick={onClose}
              className="btn-primary px-4 py-2"
            >
              关闭
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
