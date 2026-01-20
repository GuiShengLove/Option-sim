"use client";

import { useState, useMemo, useEffect } from 'react';
import { 
  Activity, RefreshCw, TrendingUp, TrendingDown, 
  BarChart2, AlertTriangle, Target, Brain, Sparkles, Play
} from 'lucide-react';
import dynamic from 'next/dynamic';
import AIReportModal from '../AIReportModal';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

// Strategy library - synced with backend strategy_templates.py
const STRATEGY_LIBRARY = [
  { id: 'current', name: '📌 当前持仓策略', type: 'current' },
  { id: 'covered_call', name: '备兑开仓 (Covered Call)', type: 'income' },
  { id: 'cash_secured_put', name: '现金担保卖沽 (Cash-Secured Put)', type: 'income' },
  { id: 'iron_condor', name: '铁鹰式 (Iron Condor)', type: 'income' },
  { id: 'long_call', name: '买入看涨 (Long Call)', type: 'directional' },
  { id: 'long_put', name: '买入看跌 (Long Put)', type: 'directional' },
  { id: 'bull_call_spread', name: '牛市看涨价差 (Bull Call Spread)', type: 'directional' },
  { id: 'bear_put_spread', name: '熊市看跌价差 (Bear Put Spread)', type: 'directional' },
  { id: 'straddle', name: '跨式 (Straddle)', type: 'volatility' },
  { id: 'strangle', name: '宽跨式 (Strangle)', type: 'volatility' },
  { id: 'calendar_spread', name: '日历价差 (Calendar Spread)', type: 'volatility' },
  { id: 'protective_put', name: '保护性看跌 (Protective Put)', type: 'hedge' },
  { id: 'collar', name: '领口策略 (Collar)', type: 'hedge' },
  { id: 'delta_hedge', name: 'Delta对冲', type: 'hedge' },
];

interface SimulationPath {
  dates: string[];
  prices: number[];
  pnl?: number[];
}

interface SimulationResult {
  paths: SimulationPath[];
  statistics: {
    meanReturn: number;
    stdDev: number;
    var95: number;
    cvar95: number;
    maxDrawdown: number;
    sharpe: number;
  };
  strategyResults?: {
    avgPnL: number;
    winRate: number;
    maxProfit: number;
    maxLoss: number;
  };
}

interface SimConfig {
  initialCapital: number;
  model: 'GBM' | 'HESTON' | 'MJD' | 'GARCH' | 'AI_GENERATIVE';
  simulationDays: number;
  numPaths: number;
  marketRegime: 'BULL' | 'BEAR' | 'SIDEWAYS' | 'CRASH';
  panicFactor: number;
  runStrategy: boolean;
  strategyId: string;
  // Additional parameters
  calibrationMode?: 'manual' | 'historical';
  initialPrice?: number;
  mu?: number;
  sigma?: number;
}

interface MonteCarloSimulatorProps {
  simConfig?: SimConfig;
  runTimestamp?: number | null;
}

export default function MonteCarloSimulator({ simConfig, runTimestamp }: MonteCarloSimulatorProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [showPercentiles, setShowPercentiles] = useState(true);
  const [showAIReport, setShowAIReport] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRunTimestamp, setLastRunTimestamp] = useState<number | null>(null);
  
  // Get values from config with defaults
  const numPaths = simConfig?.numPaths ?? 500;
  const numDays = simConfig?.simulationDays ?? 30;
  const initialPrice = simConfig?.initialPrice ?? 3.0;
  const mu = simConfig?.mu ?? 0.05;
  const sigma = simConfig?.sigma ?? 0.20;
  const selectedModel = simConfig?.model ?? 'GBM';
  const calibrationMode = simConfig?.calibrationMode ?? 'manual';
  const selectedStrategy = simConfig?.strategyId ?? 'none';
  const panicFactor = simConfig?.panicFactor ?? 0;

  // Run simulation when runTimestamp changes
  useEffect(() => {
    if (runTimestamp && runTimestamp !== lastRunTimestamp) {
      setLastRunTimestamp(runTimestamp);
      runSimulation();
    }
  }, [runTimestamp]);

  const runSimulation = async () => {
    setIsRunning(true);
    setResult(null);
    setError(null);
    
    try {
      // Call backend simulation API
      const simResponse = await fetch('http://localhost:8000/api/simulation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          S0: initialPrice,
          T_days: numDays,
          n_paths: numPaths,
          model: selectedModel,
          calibration_mode: calibrationMode,
          dataset_id: calibrationMode === 'historical' ? '510050_SH' : undefined,
          mu,
          sigma,
          panic_factor: panicFactor
        })
      });

      if (!simResponse.ok) {
        const errData = await simResponse.json();
        throw new Error(errData.detail || 'Simulation failed');
      }

      const simData = await simResponse.json();
      
      // Transform paths to our format
      const paths: SimulationPath[] = simData.paths.slice(0, 100).map((pathPrices: number[], idx: number) => ({
        dates: simData.dates || pathPrices.map((_: number, i: number) => {
          const d = new Date();
          d.setDate(d.getDate() + i);
          return d.toISOString().split('T')[0];
        }),
        prices: pathPrices
      }));

      // Calculate statistics from backend data
      const allFinalPrices = simData.paths.map((p: number[]) => p[p.length - 1]);
      const returns = allFinalPrices.map((p: number) => (p - initialPrice) / initialPrice);
      returns.sort((a: number, b: number) => a - b);
      
      const mean = returns.reduce((a: number, b: number) => a + b, 0) / returns.length;
      const variance = returns.reduce((a: number, b: number) => a + (b - mean) ** 2, 0) / returns.length;
      const stdDev = Math.sqrt(variance);
      
      const var95Idx = Math.floor(0.05 * returns.length);
      const var95 = returns[var95Idx];
      const cvar95 = returns.slice(0, var95Idx + 1).reduce((a: number, b: number) => a + b, 0) / (var95Idx + 1);
      
      // Calculate max drawdown from paths
      let totalMaxDD = 0;
      for (const pathPrices of simData.paths) {
        let peak = pathPrices[0];
        let maxDD = 0;
        for (const price of pathPrices) {
          if (price > peak) peak = price;
          const dd = (peak - price) / peak;
          if (dd > maxDD) maxDD = dd;
        }
        totalMaxDD += maxDD;
      }
      const avgMaxDD = totalMaxDD / simData.paths.length;
      
      const annualizedReturn = mean * (252 / numDays);
      const annualizedVolatility = stdDev * Math.sqrt(252 / numDays);
      const sharpe = annualizedVolatility > 0 ? annualizedReturn / annualizedVolatility : 0;

      // If strategy selected, call strategy evaluation API
      let strategyResults = undefined;
      if (selectedStrategy !== 'none' && selectedStrategy !== 'current' && selectedStrategy !== '') {
        try {
          const evalResponse = await fetch('http://localhost:8000/api/simulation/strategy-evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              paths: simData.paths,
              strategy_id: selectedStrategy,
              spot: initialPrice,
              expiry_days: numDays,
              initial_iv: sigma,
              risk_free_rate: 0.03
            })
          });

          if (evalResponse.ok) {
            const evalData = await evalResponse.json();
            strategyResults = {
              avgPnL: evalData.avg_pnl,
              winRate: evalData.win_rate,
              maxProfit: evalData.max_profit,
              maxLoss: evalData.max_loss,
            };
          }
        } catch (err) {
          console.error('Strategy evaluation failed:', err);
          // Fall back to simple calculation
          strategyResults = {
            avgPnL: mean * initialPrice * 10000,
            winRate: returns.filter((r: number) => r > 0).length / returns.length,
            maxProfit: Math.max(...returns) * initialPrice * 10000,
            maxLoss: Math.min(...returns) * initialPrice * 10000,
          };
        }
      } else if (selectedStrategy === 'current' || selectedStrategy === '') {
        // Simple calculation for "current" or empty
        strategyResults = {
          avgPnL: mean * initialPrice * 10000,
          winRate: returns.filter((r: number) => r > 0).length / returns.length,
          maxProfit: Math.max(...returns) * initialPrice * 10000,
          maxLoss: Math.min(...returns) * initialPrice * 10000,
        };
      }

      setResult({
        paths,
        statistics: {
          meanReturn: mean,
          stdDev,
          var95,
          cvar95,
          maxDrawdown: avgMaxDD,
          sharpe,
        },
        strategyResults,
      });

    } catch (err) {
      console.error('Simulation error:', err);
      setError(err instanceof Error ? err.message : 'Simulation failed');
    } finally {
      setIsRunning(false);
    }
  };

  // Chart data for price paths
  const chartData = useMemo(() => {
    if (!result) return [];
    
    const traces = result.paths.map((path, idx) => ({
      type: 'scatter' as const,
      x: path.dates,
      y: path.prices,
      mode: 'lines' as const,
      line: { color: `rgba(139, 92, 246, ${0.1 + 0.05 * (idx % 10)})`, width: 1 },
      showlegend: false,
      hoverinfo: 'skip' as const,
    }));
    
    if (showPercentiles && result.paths.length > 0) {
      const numDates = result.paths[0].dates.length;
      const p5: number[] = [];
      const p50: number[] = [];
      const p95: number[] = [];
      
      for (let d = 0; d < numDates; d++) {
        const pricesAtD = result.paths.map(p => p.prices[d]).sort((a, b) => a - b);
        p5.push(pricesAtD[Math.floor(0.05 * pricesAtD.length)]);
        p50.push(pricesAtD[Math.floor(0.50 * pricesAtD.length)]);
        p95.push(pricesAtD[Math.floor(0.95 * pricesAtD.length)]);
      }
      
      traces.push({
        type: 'scatter',
        x: result.paths[0].dates,
        y: p95,
        mode: 'lines',
        line: { color: '#10b981', width: 2, dash: 'dash' as any },
        name: '95th Percentile',
        showlegend: true,
        hoverinfo: 'y+name' as const,
      } as any);
      
      traces.push({
        type: 'scatter',
        x: result.paths[0].dates,
        y: p50,
        mode: 'lines',
        line: { color: '#f59e0b', width: 3 },
        name: 'Median',
        showlegend: true,
        hoverinfo: 'y+name' as const,
      } as any);
      
      traces.push({
        type: 'scatter',
        x: result.paths[0].dates,
        y: p5,
        mode: 'lines',
        line: { color: '#ef4444', width: 2, dash: 'dash' as any },
        name: '5th Percentile',
        showlegend: true,
        hoverinfo: 'y+name' as const,
      } as any);
    }
    
    return traces;
  }, [result, showPercentiles]);

  const histogramData = useMemo(() => {
    if (!result) return [];
    
    const finalPrices = result.paths.map(p => p.prices[p.prices.length - 1]);
    
    return [{
      type: 'histogram' as const,
      x: finalPrices,
      nbinsx: 30,
      marker: { 
        color: 'rgba(139, 92, 246, 0.6)',
        line: { color: 'rgba(139, 92, 246, 1)', width: 1 }
      },
      name: '终值分布',
    }];
  }, [result]);

  const selectedStrategyName = STRATEGY_LIBRARY.find(s => s.id === selectedStrategy)?.name || 
    (selectedStrategy === '' ? '仅模拟价格路径' : '未选择');

  // Model display name
  const modelNames: Record<string, string> = {
    'GBM': '几何布朗运动',
    'HESTON': 'Heston随机波动率',
    'MJD': 'Merton跳跃扩散',
    'GARCH': 'GARCH',
    'AI_GENERATIVE': 'AI生成式'
  };

  return (
    <div className="space-y-4">
      {/* Header with current config summary */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-[var(--accent-secondary)]" />
            <h3 className="section-title">蒙特卡洛模拟与策略预演</h3>
          </div>
          
          <div className="flex items-center gap-2">
            {result && (
              <button
                onClick={() => setShowAIReport(true)}
                className="btn-secondary flex items-center gap-2"
              >
                <Brain className="w-4 h-4" />
                生成AI报告
              </button>
            )}
            {isRunning && (
              <span className="badge badge-warning flex items-center gap-1">
                <RefreshCw className="w-3 h-3 animate-spin" />
                模拟运行中...
              </span>
            )}
          </div>
        </div>
        
        {/* Current Parameters Summary */}
        {simConfig && (
          <div className="p-3 bg-[var(--bg-card)] rounded-lg border border-[var(--border-primary)]">
            <div className="grid grid-cols-6 gap-4 text-xs">
              <div>
                <span className="text-[var(--text-muted)]">模型</span>
                <div className="font-semibold text-[var(--text-primary)]">{modelNames[selectedModel] || selectedModel}</div>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">路径数</span>
                <div className="font-mono font-semibold text-[var(--accent-primary)]">{numPaths}</div>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">天数</span>
                <div className="font-mono font-semibold text-[var(--accent-primary)]">{numDays}</div>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">初始价格</span>
                <div className="font-mono font-semibold text-[var(--text-primary)]">¥{initialPrice.toFixed(3)}</div>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">波动率 σ</span>
                <div className="font-mono font-semibold text-[var(--accent-warning)]">{(sigma * 100).toFixed(1)}%</div>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">漂移率 μ</span>
                <div className="font-mono font-semibold text-[var(--text-secondary)]">{(mu * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
            <div className="flex items-center gap-2 text-red-400">
              <AlertTriangle className="w-4 h-4" />
              <span className="text-sm">{error}</span>
            </div>
          </div>
        )}
      </div>

      {result && (
        <>
          {/* Statistics Summary */}
          <div className="grid grid-cols-6 gap-4">
            <div className="glass-card p-3 text-center">
              <div className="label-mono text-xs mb-1">平均收益</div>
              <div className={`text-lg font-mono font-bold ${result.statistics.meanReturn >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'}`}>
                {(result.statistics.meanReturn * 100).toFixed(2)}%
              </div>
            </div>
            <div className="glass-card p-3 text-center">
              <div className="label-mono text-xs mb-1">波动率</div>
              <div className="text-lg font-mono font-bold text-[var(--accent-warning)]">
                {(result.statistics.stdDev * 100).toFixed(2)}%
              </div>
            </div>
            <div className="glass-card p-3 text-center">
              <div className="label-mono text-xs mb-1">VaR (95%)</div>
              <div className="text-lg font-mono font-bold text-[var(--accent-danger)]">
                {(result.statistics.var95 * 100).toFixed(2)}%
              </div>
            </div>
            <div className="glass-card p-3 text-center">
              <div className="label-mono text-xs mb-1">CVaR (95%)</div>
              <div className="text-lg font-mono font-bold text-[var(--accent-danger)]">
                {(result.statistics.cvar95 * 100).toFixed(2)}%
              </div>
            </div>
            <div className="glass-card p-3 text-center">
              <div className="label-mono text-xs mb-1">最大回撤</div>
              <div className="text-lg font-mono font-bold text-[var(--accent-danger)]">
                {(result.statistics.maxDrawdown * 100).toFixed(2)}%
              </div>
            </div>
            <div className="glass-card p-3 text-center">
              <div className="label-mono text-xs mb-1">夏普比率</div>
              <div className={`text-lg font-mono font-bold ${result.statistics.sharpe >= 1 ? 'text-[var(--accent-success)]' : 'text-[var(--text-secondary)]'}`}>
                {result.statistics.sharpe.toFixed(2)}
              </div>
            </div>
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-2 gap-4">
            {/* Price Paths */}
            <div className="glass-card p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="section-header">
                  <TrendingUp className="w-4 h-4 text-[var(--accent-primary)]" />
                  <h3 className="section-title">价格路径 ({result.paths.length}条)</h3>
                </div>
                <label className="flex items-center gap-2 text-xs cursor-pointer">
                  <input 
                    type="checkbox" 
                    checked={showPercentiles}
                    onChange={(e) => setShowPercentiles(e.target.checked)}
                    className="accent-[var(--accent-primary)]"
                  />
                  <span className="text-[var(--text-muted)]">显示百分位</span>
                </label>
              </div>
              <div style={{ height: '350px' }}>
                <Plot
                  data={chartData}
                  layout={{
                    autosize: true,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    margin: { l: 50, r: 20, t: 20, b: 40 },
                    xaxis: {
                      gridcolor: '#2a2d35',
                      tickfont: { color: '#6b7280', size: 10 },
                    },
                    yaxis: {
                      title: '价格',
                      titlefont: { color: '#9ca3af', size: 11 },
                      gridcolor: '#2a2d35',
                      tickfont: { color: '#6b7280', size: 10 },
                    },
                    legend: {
                      x: 0,
                      y: 1,
                      bgcolor: 'transparent',
                      font: { color: '#9ca3af', size: 10 },
                    },
                    showlegend: true,
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler={true}
                />
              </div>
            </div>

            {/* Distribution Histogram */}
            <div className="glass-card p-4">
              <div className="section-header mb-2">
                <BarChart2 className="w-4 h-4 text-[var(--accent-warning)]" />
                <h3 className="section-title">终值分布</h3>
              </div>
              <div style={{ height: '350px' }}>
                <Plot
                  data={histogramData}
                  layout={{
                    autosize: true,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    margin: { l: 50, r: 20, t: 20, b: 40 },
                    xaxis: {
                      title: '终值价格',
                      titlefont: { color: '#9ca3af', size: 11 },
                      gridcolor: '#2a2d35',
                      tickfont: { color: '#6b7280', size: 10 },
                    },
                    yaxis: {
                      title: '频次',
                      titlefont: { color: '#9ca3af', size: 11 },
                      gridcolor: '#2a2d35',
                      tickfont: { color: '#6b7280', size: 10 },
                    },
                    showlegend: false,
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler={true}
                />
              </div>
            </div>
          </div>

          {/* Strategy Preview Results */}
          {result.strategyResults && (
            <div className="glass-card p-4 border-[var(--accent-primary)]/30">
              <div className="section-header mb-4">
                <Target className="w-4 h-4 text-[var(--accent-primary)]" />
                <h3 className="section-title">策略预演结果 - {selectedStrategyName}</h3>
                <span className="text-xs text-[var(--text-muted)] ml-auto">基于 {numPaths} 条模拟路径</span>
              </div>
              
              <div className="grid grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="label-mono text-xs mb-1">平均盈亏</div>
                  <div className={`text-xl font-mono font-bold ${result.strategyResults.avgPnL >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'}`}>
                    ¥{result.strategyResults.avgPnL.toFixed(0)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="label-mono text-xs mb-1">胜率</div>
                  <div className="text-xl font-mono font-bold text-[var(--accent-primary)]">
                    {(result.strategyResults.winRate * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="label-mono text-xs mb-1">最大盈利</div>
                  <div className="text-xl font-mono font-bold text-[var(--accent-success)]">
                    ¥{result.strategyResults.maxProfit.toFixed(0)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="label-mono text-xs mb-1">最大亏损</div>
                  <div className="text-xl font-mono font-bold text-[var(--accent-danger)]">
                    ¥{result.strategyResults.maxLoss.toFixed(0)}
                  </div>
                </div>
              </div>
              
              {/* Generate Report Button */}
              <div className="mt-4 flex items-center justify-between p-3 bg-[var(--bg-secondary)] rounded-lg">
                <p className="text-xs text-[var(--text-muted)]">
                  💡 <strong>提示:</strong> 点击"生成AI报告"获取详细分析和改进建议
                </p>
                <button
                  onClick={() => setShowAIReport(true)}
                  className="btn-secondary flex items-center gap-2 text-sm"
                >
                  <Sparkles className="w-4 h-4" />
                  生成AI分析报告
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {!result && !isRunning && (
        <div className="glass-card p-8 text-center">
          <Activity className="w-12 h-12 text-[var(--text-muted)] mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-[var(--text-secondary)] mb-2">蒙特卡洛模拟</h3>
          <p className="text-sm text-[var(--text-muted)] mb-4">
            请在左侧面板设置参数，然后点击"启动蒙特卡洛模拟"按钮运行模拟
          </p>
          <div className="flex items-center justify-center gap-2 text-[var(--text-muted)]">
            <Play className="w-4 h-4" />
            <span className="text-xs">等待启动...</span>
          </div>
        </div>
      )}

      {isRunning && (
        <div className="glass-card p-8 text-center">
          <RefreshCw className="w-12 h-12 text-[var(--accent-secondary)] mx-auto mb-4 animate-spin" />
          <h3 className="text-lg font-semibold text-[var(--text-secondary)] mb-2">正在运行模拟...</h3>
          <p className="text-sm text-[var(--text-muted)]">
            生成 {numPaths} 条价格路径，模拟 {numDays} 天
          </p>
        </div>
      )}

      {/* AI Report Modal */}
      <AIReportModal
        isOpen={showAIReport}
        onClose={() => setShowAIReport(false)}
        reportType="simulation"
        simulationData={result ? {
          model: selectedModel,
          numPaths,
          numDays,
          meanReturn: result.statistics.meanReturn,
          stdDev: result.statistics.stdDev,
          var95: result.statistics.var95,
          cvar95: result.statistics.cvar95,
          maxDrawdown: result.statistics.maxDrawdown,
          sharpe: result.statistics.sharpe,
          strategyWinRate: result.strategyResults?.winRate,
          strategyAvgPnL: result.strategyResults?.avgPnL,
        } : undefined}
      />
    </div>
  );
}
