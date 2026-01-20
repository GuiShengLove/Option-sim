"use client";

import { useState, useEffect, useRef } from 'react';
import { 
  TrendingUp, Layers, BarChart3, Grid3X3, Activity, 
  Brain, FileText, Settings, Bell, Play
} from 'lucide-react';
import EquityCurveChart from './charts/EquityCurveChart';
import MarginRiskAnalysis from './charts/MarginRiskAnalysis';
import VolatilitySurface3D from './charts/VolatilitySurface3D';
import AssetExplorerTab from './charts/AssetExplorerTab';
import OptionChainWithGreeks from './charts/OptionChainWithGreeks';
import MonteCarloSimulator from './charts/MonteCarloSimulator';
import AIMonitorPanel from './AIMonitorPanel';
import AIReportModal from './AIReportModal';
import TradeLogTable from './charts/TradeLogTable';

type Tab = 'capital' | 'chain' | 'assets' | 'surface' | 'simulation' | 'trade';

interface Metrics {
  strategyValue: string;
  strategyChange: string;
  maxDrawdown: string;
  sharpeRatio: string;
  marginUsage: string;
  isLoading: boolean;
}

interface HolographicDashboardProps {
  backtestConfig?: {
    initialCapital: number;
    startDate: string;
    endDate: string;
    marginMode: string;
    marginRatio: number;
    leverage: number;
    maintenanceMargin?: number;
    strategyId?: string;
    datasetId?: string;
  };
  simConfig?: {
    initialCapital: number;
    model: string;
    simulationDays: number;
    numPaths: number;
    marketRegime: string;
    panicFactor: number;
    strategyId: string;
    calibrationMode?: string;
    initialPrice?: number;
    mu?: number;
    sigma?: number;
  };
  runTimestamp?: number | null;
  backtestResult?: {
    success: boolean;
    message: string;
    metrics?: {
      total_return: number;
      max_drawdown: number;
      sharpe_ratio: number;
      final_equity: number;
      trade_count: number;
      // 新增策略表现指标
      win_rate?: number;
      profit_factor?: number;
      avg_trade_pnl?: number;
      max_consecutive_losses?: number;
      trading_days?: number;
      daily_pnl?: number;
      realized_pnl?: number;
    };
    equity_curve?: Array<{
      date: string;
      equity: number;
      cash: number;
      margin_utilization: number;
      position_count?: number;
      total_delta?: number;
      total_gamma?: number;
      total_vega?: number;
      total_theta?: number;
    }>;
    trades?: Array<any>;
    strategy_name: string;
    dataset_id?: string;
  } | null;
}

export default function HolographicDashboard({ backtestConfig, simConfig, runTimestamp, backtestResult }: HolographicDashboardProps) {
  const [activeTab, setActiveTab] = useState<Tab>('capital');
  const [showAIReport, setShowAIReport] = useState(false);
  const [reportType, setReportType] = useState<'backtest' | 'simulation'>('backtest');
  
  // Track valid run to show results
  const [hasRun, setHasRun] = useState(false);

  const [metrics, setMetrics] = useState<Metrics>({
    strategyValue: '¥100万',
    strategyChange: '0.0%',
    maxDrawdown: '0.0%',
    sharpeRatio: '0.00',
    marginUsage: '0%',
    isLoading: false
  });

  const leverage = backtestConfig?.leverage ?? 1.0;
  const marginRatio = backtestConfig?.marginRatio ?? 0.12;

  // Effect: Use backtestResult when available
  useEffect(() => {
    if (!runTimestamp) return;

    setHasRun(true);
    
    // If we have real backtest results from API, use them
    if (backtestResult?.success && backtestResult.metrics) {
      const m = backtestResult.metrics;
      const initialCap = backtestConfig?.initialCapital ?? 1000000;
      
      setMetrics({
        strategyValue: `¥${(m.final_equity / 10000).toFixed(1)}万`,
        strategyChange: `${m.total_return >= 0 ? '+' : ''}${m.total_return.toFixed(1)}%`,
        maxDrawdown: `-${m.max_drawdown.toFixed(1)}%`,
        sharpeRatio: m.sharpe_ratio.toFixed(2),
        marginUsage: `${(marginRatio * 100).toFixed(0)}%`,
        isLoading: false
      });
      return;
    }
    
    // Fallback: fetch from ETF data if no backtest result
    setMetrics(prev => ({ ...prev, isLoading: true }));

    const fetchMetrics = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/data/etf-candle?days=60');
        if (response.ok) {
          const data = await response.json();
          const candles = data.candles || [];
          
          if (candles.length > 1) {
            const firstPrice = candles[0].close;
            const lastPrice = candles[candles.length - 1].close;
            const totalReturn = ((lastPrice - firstPrice) / firstPrice * 100);
            
            let peak = firstPrice;
            let maxDD = 0;
            candles.forEach((c: any) => {
              if (c.close > peak) peak = c.close;
              const dd = (peak - c.close) / peak * 100;
              if (dd > maxDD) maxDD = dd;
            });
            
            const returns = candles.slice(1).map((c: any, i: number) => 
              (c.close - candles[i].close) / candles[i].close
            );
            const avgReturn = returns.reduce((a: number, b: number) => a + b, 0) / returns.length;
            const variance = returns.reduce((a: number, b: number) => a + Math.pow(b - avgReturn, 2), 0) / returns.length;
            const dailyVol = Math.sqrt(variance);
            const sharpe = (avgReturn / dailyVol) * Math.sqrt(252) * leverage;
            
            const initialCap = backtestConfig?.initialCapital ?? 1000000;
            const value = initialCap * (1 + totalReturn / 100 * leverage);
            const marginUsagePercent = Math.min(95, marginRatio * 100 * leverage);
            
            setMetrics({
              strategyValue: `¥${(value / 10000).toFixed(1)}万`,
              strategyChange: `${totalReturn >= 0 ? '+' : ''}${(totalReturn * leverage).toFixed(1)}%`,
              maxDrawdown: `-${(maxDD * leverage).toFixed(1)}%`,
              sharpeRatio: sharpe.toFixed(2),
              marginUsage: `${marginUsagePercent.toFixed(0)}%`,
              isLoading: false
            });
          }
        }
      } catch (err) {
        console.error('Failed to fetch metrics:', err);
        setMetrics(prev => ({ ...prev, isLoading: false }));
      }
    };
    
    fetchMetrics();
  }, [runTimestamp, backtestResult, leverage, marginRatio, backtestConfig?.initialCapital]);

  const backtestData = {
    strategyName: '50ETF期权策略',
    startDate: '2020-01-02',
    endDate: '2020-12-31',
    initialCapital: 1000000,
    finalEquity: parseFloat(metrics.strategyValue.replace(/[^0-9.]/g, '')) * 10000,
    totalReturn: parseFloat(metrics.strategyChange) / 100,
    sharpeRatio: parseFloat(metrics.sharpeRatio),
    maxDrawdown: parseFloat(metrics.maxDrawdown) / 100,
    winRate: 0.55,
    totalTrades: 120,
    profitFactor: 1.65,
  };

  const tabs = [
    { id: 'capital' as Tab, label: '资金与风控', icon: TrendingUp },
    { id: 'chain' as Tab, label: '期权链', icon: Grid3X3 },
    { id: 'assets' as Tab, label: '标的浏览', icon: BarChart3 },
    { id: 'surface' as Tab, label: '波动率曲面', icon: Layers },
    { id: 'simulation' as Tab, label: '蒙特卡洛模拟', icon: Activity },
    { id: 'trade' as Tab, label: '交易明细', icon: FileText },
  ];

  const handleOpenAIReport = (type: 'backtest' | 'simulation') => {
    setReportType(type);
    setShowAIReport(true);
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)] overflow-hidden">
      {/* Header */}
      <div className="shrink-0 px-6 pt-6 pb-4 border-b border-[var(--border-primary)]">
        <div className="flex items-center justify-between mb-4">
          <h1 className="heading-gradient text-2xl font-bold">全息数据仪表盘</h1>
          
          <div className="flex items-center gap-1">
            <button
              onClick={() => handleOpenAIReport('backtest')}
              disabled={!hasRun}
              className={`p-2 rounded-lg transition-colors group relative ${
                hasRun ? 'hover:bg-[var(--accent-primary)]/10' : 'opacity-50 cursor-not-allowed'
              }`}
              title="生成AI分析报告"
            >
              <Brain className="w-5 h-5 text-[var(--accent-primary)] group-hover:scale-110 transition-transform" />
              {hasRun && <span className="absolute -top-1 -right-1 w-2 h-2 bg-[var(--accent-success)] rounded-full animate-pulse"></span>}
            </button>
            <button className="p-2 rounded-lg hover:bg-[var(--bg-card)] transition-colors">
              <FileText className="w-5 h-5 text-[var(--text-muted)] hover:text-[var(--text-primary)]" />
            </button>
            <button className="p-2 rounded-lg hover:bg-[var(--bg-card)] transition-colors relative">
              <Bell className="w-5 h-5 text-[var(--text-muted)] hover:text-[var(--text-primary)]" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-[var(--accent-danger)] rounded-full"></span>
            </button>
            <button className="p-2 rounded-lg hover:bg-[var(--bg-card)] transition-colors">
              <Settings className="w-5 h-5 text-[var(--text-muted)] hover:text-[var(--text-primary)]" />
            </button>
            <div className="w-px h-6 bg-[var(--border-primary)] mx-2"></div>
            
            {hasRun ? (
                <span className="badge badge-success">
                  <div className="w-2 h-2 rounded-full bg-[var(--accent-success)]"></div>
                  运行完成
                </span>
            ) : (
                <span className="badge bg-[var(--bg-card)] text-[var(--text-muted)] border border-[var(--border-primary)]">
                  <div className="w-2 h-2 rounded-full bg-[var(--text-muted)]"></div>
                  待运行
                </span>
            )}
          </div>
        </div>

        <div className="tab-nav">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`tab-item ${activeTab === tab.id ? 'active' : ''} flex items-center gap-2`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-6 scroll-smooth bg-[#0d1016]">
        
        {/* Capital Tab - KeepAlive */}
        <div style={{ display: activeTab === 'capital' ? 'block' : 'none', height: '100%' }}>
            {hasRun ? (
              backtestResult?.success === false ? (
                // 显示错误信息
                <div className="h-full flex flex-col items-center justify-center text-[var(--text-muted)] space-y-4">
                    <div className="w-16 h-16 rounded-full bg-[var(--accent-danger)]/10 flex items-center justify-center">
                        <span className="text-4xl">⚠️</span>
                    </div>
                    <p className="text-[var(--accent-danger)] text-lg font-semibold">{backtestResult.message}</p>
                    <p className="text-sm">请在左侧面板选择一个策略后重新运行回测</p>
                </div>
              ) : (
              <div className="space-y-4">
                {/* Metrics */}
                <div className="grid grid-cols-4 gap-4">
                    <div className="metric-card bg-gradient-to-br from-[var(--accent-primary)]/10 to-transparent">
                    <div className="label-mono mb-2">策略净值</div>
                    <div className="metric-value text-[var(--accent-primary)]">{metrics.strategyValue}</div>
                    <div className={`metric-change ${parseFloat(metrics.strategyChange) >= 0 ? 'positive' : 'negative'}`}>
                        {metrics.strategyChange}
                    </div>
                    </div>
                    <div className="metric-card bg-gradient-to-br from-[var(--accent-danger)]/10 to-transparent">
                    <div className="label-mono mb-2">最大回撤</div>
                    <div className="metric-value text-[var(--accent-danger)]">{metrics.maxDrawdown}</div>
                    <div className="label-mono mt-1">风险可控</div>
                    </div>
                    <div className="metric-card bg-gradient-to-br from-[var(--accent-success)]/10 to-transparent">
                    <div className="label-mono mb-2">夏普比率</div>
                    <div className="metric-value text-[var(--accent-success)]">{metrics.sharpeRatio}</div>
                    <div className="label-mono mt-1">良好</div>
                    </div>
                    <div className="metric-card bg-gradient-to-br from-[var(--accent-warning)]/10 to-transparent">
                    <div className="label-mono mb-2">保证金占用率</div>
                    <div className="metric-value text-[var(--accent-warning)]">{metrics.marginUsage}</div>
                    <div className="label-mono mt-1">安全</div>
                    </div>
                </div>

                {/* Charts */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="glass-card p-4" style={{ height: '350px' }}>
                    <div className="section-header">
                        <TrendingUp className="w-4 h-4 text-[var(--accent-primary)]" />
                        <h3 className="section-title">策略净值曲线</h3>
                    </div>
                    <EquityCurveChart equityData={backtestResult?.equity_curve} />
                    </div>
                    <div className="glass-card p-4" style={{ height: '350px' }}>
                    <div className="section-header">
                        <Activity className="w-4 h-4 text-[var(--accent-secondary)]" />
                        <h3 className="section-title">保证金风险分析</h3>
                    </div>
                    <MarginRiskAnalysis 
                      strategyId={simConfig?.strategyId} 
                      marginMode={backtestConfig?.marginMode}
                      equityCurve={backtestResult?.equity_curve}
                      startDate={backtestConfig?.startDate}
                      endDate={backtestConfig?.endDate}
                      marginRatio={backtestConfig?.marginRatio}
                      leverage={backtestConfig?.leverage}
                      maintenanceMargin={backtestConfig?.maintenanceMargin}
                    />
                    </div>
                </div>
              </div>
              )
            ) : (
                <div className="h-full flex flex-col items-center justify-center text-[var(--text-muted)] space-y-4">
                    <Play className="w-16 h-16 opacity-20" />
                    <p>请点击左下角"启动回测"按钮查看结果</p>
                </div>
            )}
        </div>

        {/* Chain - KeepAlive */}
        <div style={{ display: activeTab === 'chain' ? 'block' : 'none', height: '100%' }}>
            {hasRun ? (
              <OptionChainWithGreeks 
                selectedDate={backtestConfig?.startDate} 
                dateRange={{ start: backtestConfig?.startDate || '', end: backtestConfig?.endDate || '' }} 
                datasetId={backtestResult?.dataset_id || backtestConfig?.datasetId}
              />
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-[var(--text-muted)] space-y-4">
                <Grid3X3 className="w-16 h-16 opacity-20" />
                <p>请点击左下角"启动回测"按钮查看期权链数据</p>
              </div>
            )}
        </div>

        {/* Assets - KeepAlive */}
        <div style={{ display: activeTab === 'assets' ? 'block' : 'none', height: '100%' }}>
            {hasRun ? (
              <AssetExplorerTab 
                selectedDate={backtestConfig?.startDate} 
                dateRange={{ start: backtestConfig?.startDate || '', end: backtestConfig?.endDate || '' }}
                datasetId={backtestResult?.dataset_id || backtestConfig?.datasetId}
              />
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-[var(--text-muted)] space-y-4">
                <BarChart3 className="w-16 h-16 opacity-20" />
                <p>请点击左下角"启动回测"按钮查看标的浏览数据</p>
              </div>
            )}
        </div>

        {/* Surface - KeepAlive */}
        <div style={{ display: activeTab === 'surface' ? 'block' : 'none', height: '100%' }}>
            {hasRun ? (
              <div className="space-y-4 h-full flex flex-col">
                <div className="glass-card p-4 flex-1 min-h-[500px]">
                  <div className="section-header">
                    <Layers className="w-4 h-4 text-[var(--accent-primary)]" />
                    <h3 className="section-title">期权隐含波动率曲面 - 3D 视图</h3>
                  </div>
                  <VolatilitySurface3D 
                    selectedDate={backtestConfig?.startDate} 
                    dateRange={{ start: backtestConfig?.startDate || '', end: backtestConfig?.endDate || '' }}
                    datasetId={backtestResult?.dataset_id || backtestConfig?.datasetId}
                  />
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-[var(--text-muted)] space-y-4">
                <Layers className="w-16 h-16 opacity-20" />
                <p>请点击左下角"启动回测"按钮查看波动率曲面</p>
              </div>
            )}
        </div>

        {/* Simulator - KeepAlive */}
        <div style={{ display: activeTab === 'simulation' ? 'block' : 'none', height: '100%' }}>
             <MonteCarloSimulator 
               simConfig={simConfig ? {
                 ...simConfig,
                 // Use backtestConfig's strategyId as the primary strategy for simulation
                 strategyId: backtestConfig?.strategyId || simConfig.strategyId || ''
               } : undefined} 
               runTimestamp={runTimestamp} 
             />
        </div>

        {/* Trade Log - KeepAlive */}
        <div style={{ display: activeTab === 'trade' ? 'block' : 'none', height: '100%' }}>
            {hasRun ? (
                <div className="glass-card p-4 h-full overflow-hidden flex flex-col">
                    <div className="section-header mb-4">
                        <FileText className="w-4 h-4 text-[var(--accent-secondary)]" />
                        <h3 className="section-title">交易执行明细</h3>
                        <div className="ml-auto text-xs text-[var(--text-muted)] font-mono">
                           数据集: {backtestResult?.dataset_id || backtestConfig?.datasetId || '510050_SH'}
                        </div>
                    </div>
                    <div className="flex-1 min-h-0">
                        <TradeLogTable trades={backtestResult?.trades} />
                    </div>
                </div>
            ) : (
                <div className="h-full flex flex-col items-center justify-center text-[var(--text-muted)] space-y-4">
                    <FileText className="w-16 h-16 opacity-20" />
                    <p>请点击左下角"启动回测"按钮查看交易明细</p>
                </div>
            )}
        </div>
      </div>

      <AIMonitorPanel 
        onOpenReport={handleOpenAIReport}
        hasRun={hasRun && backtestResult?.success === true}
        runTimestamp={runTimestamp}
        strategyName={backtestResult?.strategy_name ?? '未选择策略'}
        metrics={backtestResult?.metrics ? {
          total_return: backtestResult.metrics.total_return,
          max_drawdown: backtestResult.metrics.max_drawdown,
          sharpe_ratio: backtestResult.metrics.sharpe_ratio,
          win_rate: backtestResult.metrics.win_rate,
          profit_factor: backtestResult.metrics.profit_factor,
          trade_count: backtestResult.metrics.trade_count,
        } : undefined}
        isBacktestRunning={false}
      />
      
      <AIReportModal
        isOpen={showAIReport}
        onClose={() => setShowAIReport(false)}
        reportType={reportType}
        backtestData={backtestData}
      />
    </div>
  );
}
