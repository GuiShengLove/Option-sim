"use client";

import { useState, useEffect } from 'react';
import { 
  Settings, Play, RefreshCw, ShieldAlert, Wallet, Percent, 
  TrendingDown, Layers, Clock, BarChart2, Zap, Target, Code
} from 'lucide-react';
import { clsx } from 'clsx';

// Strategy type from API
interface StrategyItem {
  id: string;
  name: string;
  description?: string;
}

interface DatasetItem {
  id: string;
  name: string;
  type: 'PLATFORM' | 'USER';
  path: string;
  date_range?: { start: string; end: string };
  file_count?: number;
}

type Mode = 'BACKTEST' | 'SIMULATION';
type MarginMode = 'FIXED' | 'SPAN' | 'PORTFOLIO';
type CommissionMode = 'FIXED' | 'PERCENT' | 'TIERED';

interface BacktestConfig {
  // Basic
  initialCapital: number;
  startDate: string;
  endDate: string;
  
  // Margin & Leverage
  marginMode: MarginMode;
  marginRatio: number;       // e.g., 0.12 = 12%
  leverage: number;          // e.g., 2.0 = 2x
  maintenanceMargin: number; // e.g., 0.08 = 8%
  
  // Commissions
  commissionMode: CommissionMode;
  commissionPerContract: number;  // Fixed: ¥ per contract
  commissionPercent: number;      // Percent: % of trade value
  
  // Slippage
  slippageMode: 'FIXED' | 'PERCENT' | 'VOLUME_IMPACT';
  slippageTicks: number;
  slippagePercent: number;
  
  // Risk Controls
  maxPositionSize: number;     // Max contracts per position
  maxDrawdown: number;         // Trigger liquidation at this %
  liquidationThreshold: number; // Margin call level
  dailyLossLimit: number;      // Stop trading if daily loss exceeds
  
  // Execution
  fillModel: 'MIDPOINT' | 'WORST' | 'VWAP';
  partialFillAllowed: boolean;
  
  // Strategy selection
  // Strategy selection
  strategyId: string;
  datasetId: string;
}

interface SimulationConfig {
  initialCapital: number;
  model: 'GBM' | 'HESTON' | 'MJD' | 'GARCH';
  simulationDays: number;
  numPaths: number;
  marketRegime: 'BULL' | 'BEAR' | 'SIDEWAYS' | 'CRASH';
  panicFactor: number;
  
  // Strategy Preview
  runStrategy: boolean;
  strategyId: string;
  
  // Parameter calibration
  calibrationMode: 'manual' | 'historical';
  initialPrice: number;
  mu: number;  // Drift rate (annual)
  sigma: number;  // Volatility (annual)
  
  // Heston Model Parameters
  v0: number;      // Initial variance (e.g., 0.04 = 20%^2)
  kappa: number;   // Mean reversion speed
  theta: number;   // Long-run variance
  xi: number;      // Vol of vol
  rho: number;     // Correlation between price and vol (-1 to 1)
  
  // MJD (Jump Diffusion) Parameters
  lambda: number;  // Jump intensity (jumps per year)
  jumpMean: number; // Mean of log jump size
  jumpVol: number;  // Std of log jump size
  
  // GARCH Parameters
  omega: number;   // Constant term
  alpha: number;   // ARCH coefficient
  beta: number;    // GARCH coefficient
}

interface MissionControlProps {
  onConfigChange?: (type: 'backtest' | 'simulation', config: any) => void;
  onRun?: () => void;
}

export default function MissionControl({ onConfigChange, onRun }: MissionControlProps) {
  const [mode, setMode] = useState<Mode>('BACKTEST');
  const [isRunning, setIsRunning] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [strategies, setStrategies] = useState<StrategyItem[]>([]);
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [loadingStrategies, setLoadingStrategies] = useState(true);
  const [configModified, setConfigModified] = useState(false);  // 参数变更追踪
  
  // Fetch strategies and datasets on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [stratRes, dataRes] = await Promise.all([
            fetch('http://localhost:8000/api/strategies/'),
            fetch('http://localhost:8000/api/files/datasets')
        ]);
        
        if (stratRes.ok) {
          const data = await stratRes.json();
          setStrategies(data.items || []);
        }
        
        if (dataRes.ok) {
            const data = await dataRes.json();
            setDatasets(data.datasets || []);
        }
      } catch (err) {
        console.error('Failed to fetch initial data:', err);
      } finally {
        setLoadingStrategies(false);
      }
    };
    fetchData();
  }, []);
  
  // Backtest configuration
  const [backtestConfig, setBacktestConfig] = useState<BacktestConfig>({
    initialCapital: 1000000,
    startDate: '2020-01-02',
    endDate: '2020-12-31',
    
    marginMode: 'FIXED',
    marginRatio: 0.12,
    leverage: 1.0,
    maintenanceMargin: 0.08,
    
    commissionMode: 'FIXED',
    commissionPerContract: 1.5,
    commissionPercent: 0.0003,
    
    slippageMode: 'FIXED',
    slippageTicks: 1,
    slippagePercent: 0.001,
    
    maxPositionSize: 100,
    maxDrawdown: 0.20,
    liquidationThreshold: 0.05,
    dailyLossLimit: 0.05,
    
    fillModel: 'MIDPOINT',
    partialFillAllowed: true,
    strategyId: '',
    datasetId: '510050_SH',
  });
  
  // Simulation configuration
  const [simConfig, setSimConfig] = useState<SimulationConfig>({
    initialCapital: 1000000,
    model: 'GBM',
    simulationDays: 30,
    numPaths: 500,
    marketRegime: 'SIDEWAYS',
    panicFactor: 0,
    runStrategy: true,
    strategyId: '',
    calibrationMode: 'manual',
    initialPrice: 3.0,
    mu: 0.05,
    sigma: 0.20,
    // Heston defaults (typical equity parameters)
    v0: 0.04,      // Initial variance = 20%^2
    kappa: 2.0,    // Mean reversion speed
    theta: 0.04,   // Long-run variance = 20%^2
    xi: 0.3,       // Vol of vol
    rho: -0.7,     // Negative correlation (leverage effect)
    // MJD defaults
    lambda: 0.75,  // ~0.75 jumps per year
    jumpMean: -0.02, // Slight negative mean jump
    jumpVol: 0.1,  // Jump volatility
    // GARCH defaults
    omega: 0.000001,
    alpha: 0.1,
    beta: 0.85,
  });

  const updateBacktest = (field: keyof BacktestConfig, value: any) => {
    const newConfig = { ...backtestConfig, [field]: value };
    setBacktestConfig(newConfig);
    setConfigModified(true);  // 标记配置已修改
    onConfigChange?.('backtest', newConfig);
  };
  
  const updateSim = (field: keyof SimulationConfig, value: any) => {
    const newConfig = { ...simConfig, [field]: value };
    setSimConfig(newConfig);
    onConfigChange?.('simulation', newConfig);
  };

  // Market regime presets - apply when regime changes
  const applyMarketRegimePreset = (regime: string) => {
    const presets: Record<string, { mu: number; sigma: number; panicFactor: number }> = {
      'BULL': { mu: 0.15, sigma: 0.15, panicFactor: 0 },     // High return, low vol
      'BEAR': { mu: -0.10, sigma: 0.30, panicFactor: 0.2 },  // Negative return, medium vol
      'SIDEWAYS': { mu: 0.03, sigma: 0.20, panicFactor: 0 }, // Near zero return, normal vol
      'CRASH': { mu: -0.40, sigma: 0.60, panicFactor: 0.8 }, // Large negative, very high vol
    };
    
    const preset = presets[regime];
    if (preset) {
      const newConfig = {
        ...simConfig,
        marketRegime: regime as SimulationConfig['marketRegime'],
        mu: preset.mu,
        sigma: preset.sigma,
        panicFactor: preset.panicFactor,
      };
      setSimConfig(newConfig);
      onConfigChange?.('simulation', newConfig);
    }
  };

  const handleRun = async () => {
    setIsRunning(true);
    setConfigModified(false);  // 运行后清除修改标记
    
    // Notify parent to refresh dashboard
    onRun?.();
    
    // Also trigger config change to ensure latest values
    if (mode === 'BACKTEST') {
      onConfigChange?.('backtest', backtestConfig);
    } else {
      onConfigChange?.('simulation', simConfig);
    }
    
    setTimeout(() => setIsRunning(false), 1000);
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-secondary)] border-r border-[var(--border-primary)] w-80 overflow-y-auto">
      {/* Header */}
      <div className="p-4 border-b border-[var(--border-primary)] glass-card-elevated sticky top-0 z-10">
        <div className="flex items-center gap-2 text-[var(--accent-primary)] font-bold tracking-wider">
          <Settings className="w-5 h-5" />
          <span className="section-title">任务控制中心</span>
        </div>
      </div>

      <div className="p-4 space-y-5 flex-1">
        {/* Mode Switcher */}
        <div className="flex bg-[var(--bg-primary)] p-1 rounded-lg border border-[var(--border-primary)]">
          <button 
            onClick={() => setMode('BACKTEST')}
            className={clsx(
              "flex-1 py-2 text-xs font-semibold rounded-md transition-all",
              mode === 'BACKTEST' 
                ? "bg-[var(--accent-primary)] text-white shadow-lg" 
                : "text-[var(--text-muted)] hover:text-white"
            )}
          >
            ⏪ 历史回测
          </button>
          <button 
            onClick={() => setMode('SIMULATION')}
            className={clsx(
              "flex-1 py-2 text-xs font-semibold rounded-md transition-all",
              mode === 'SIMULATION' 
                ? "bg-[var(--accent-secondary)] text-white shadow-lg" 
                : "text-[var(--text-muted)] hover:text-white"
            )}
          >
            🔮 蒙特卡洛模拟
          </button>
        </div>

        {/* ==================== BACKTEST MODE ==================== */}
        {mode === 'BACKTEST' && (
          <div className="space-y-4">
            {/* Basic Settings */}
            <div className="glass-card p-3 space-y-3">
              <div className="flex items-center gap-2 text-xs font-bold text-[var(--accent-primary)]">
                <Wallet className="w-4 h-4" />
                基础设置
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div className="col-span-2 space-y-1">
                  <label className="label-mono text-xs">初始资金 (¥)</label>
                  <input 
                    type="number" 
                    value={backtestConfig.initialCapital}
                    onChange={(e) => updateBacktest('initialCapital', parseInt(e.target.value))}
                    className="input-field text-sm" 
                  />
                </div>
                <div className="space-y-1">
                  <label className="label-mono text-xs">起始日期</label>
                  <input 
                    type="date" 
                    value={backtestConfig.startDate}
                    onChange={(e) => updateBacktest('startDate', e.target.value)}
                    className="input-field text-sm" 
                  />
                </div>
                <div className="space-y-1">
                  <label className="label-mono text-xs">结束日期</label>
                  <input 
                    type="date" 
                    value={backtestConfig.endDate}
                    onChange={(e) => updateBacktest('endDate', e.target.value)}
                    className="input-field text-sm" 
                  />
                </div>
              </div>
            </div>

            {/* Strategy & Data Selection */}
            <div className="glass-card p-3 space-y-3 border-l-2 border-[var(--accent-primary)]">
              <div className="flex items-center gap-2 text-xs font-bold text-[var(--accent-primary)]">
                <Code className="w-4 h-4" />
                策略与数据
              </div>
              
              <div className="space-y-1">
                <label className="label-mono text-xs">选择策略</label>
                <select 
                  value={backtestConfig.strategyId}
                  onChange={(e) => updateBacktest('strategyId', e.target.value)}
                  className="select-field text-sm"
                  disabled={loadingStrategies}
                >
                  <option value="">-- {loadingStrategies ? '加载中...' : '选择策略'} --</option>
                  {strategies.map(s => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))}
                </select>
              </div>

               <div className="space-y-1">
                <label className="label-mono text-xs">市场数据</label>
                <select 
                  value={backtestConfig.datasetId}
                  onChange={(e) => updateBacktest('datasetId', e.target.value)}
                  className="select-field text-sm"
                >
                  <optgroup label="平台数据 (Platform Data)">
                    {datasets.filter(d => d.type === 'PLATFORM').map(d => (
                        <option key={d.id} value={d.id}>{d.name}</option>
                    ))}
                  </optgroup>
                  {datasets.some(d => d.type === 'USER') && (
                    <optgroup label="用户数据 (User Data)">
                        {datasets.filter(d => d.type === 'USER').map(d => (
                            <option key={d.id} value={d.id}>{d.name}</option>
                        ))}
                    </optgroup>
                  )}
                </select>
              </div>
              
              {backtestConfig.strategyId && (
                <p className="text-xs text-[var(--text-muted)]">
                  ✓ 策略将驱动回测交易逻辑
                </p>
              )}
              
              {!backtestConfig.strategyId && (
                <p className="text-xs text-[var(--accent-danger)]">
                  ⚠ 请选择一个策略才能运行回测
                </p>
              )}
            </div>

            {/* Margin Settings */}
            <div className="glass-card p-3 space-y-3">
              <div className="flex items-center gap-2 text-xs font-bold text-[var(--accent-warning)]">
                <Layers className="w-4 h-4" />
                保证金设置
              </div>
              
              <div className="space-y-3">
                <div className="space-y-1">
                  <label className="label-mono text-xs">保证金模式</label>
                  <select 
                    value={backtestConfig.marginMode}
                    onChange={(e) => updateBacktest('marginMode', e.target.value)}
                    className="select-field text-sm"
                  >
                    <option value="FIXED">固定比例 (12%)</option>
                    <option value="SSE">上交所标准</option>
                    <option value="SPAN">SPAN风险分析</option>
                    <option value="PM">组合保证金</option>
                  </select>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  {backtestConfig.marginMode === 'FIXED' && (
                    <div className="space-y-1">
                      <label className="label-mono text-xs">保证金率 (%)</label>
                      <input 
                        type="number" 
                        value={backtestConfig.marginRatio * 100}
                        onChange={(e) => updateBacktest('marginRatio', parseFloat(e.target.value) / 100)}
                        step="1"
                        className="input-field text-sm" 
                      />
                    </div>
                  )}
                  
                  {['FIXED', 'SSE'].includes(backtestConfig.marginMode) && (
                    <div className="space-y-1">
                      <label className="label-mono text-xs">杠杆倍数</label>
                      <input 
                        type="number" 
                        value={backtestConfig.leverage}
                        onChange={(e) => updateBacktest('leverage', parseFloat(e.target.value))}
                        step="0.5"
                        min="1"
                        max="10"
                        className="input-field text-sm" 
                      />
                    </div>
                  )}
                  
                  {['SPAN', 'PM'].includes(backtestConfig.marginMode) && (
                    <div className="col-span-2 text-xs text-[var(--text-muted)] italic py-2">
                       *此模式下由系统自动计算组合风险
                    </div>
                  )}
                </div>
                
                <div className="space-y-1">
                  <label className="label-mono text-xs">维持保证金 (%)</label>
                  <input 
                    type="number" 
                    value={backtestConfig.maintenanceMargin * 100}
                    onChange={(e) => updateBacktest('maintenanceMargin', parseFloat(e.target.value) / 100)}
                    step="1"
                    className="input-field text-sm" 
                  />
                </div>
              </div>
            </div>

            {/* Commission & Slippage */}
            <div className="glass-card p-3 space-y-3">
              <div className="flex items-center gap-2 text-xs font-bold text-[var(--accent-secondary)]">
                <Percent className="w-4 h-4" />
                手续费与滑点
              </div>
              
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="label-mono text-xs">手续费模式</label>
                    <select 
                      value={backtestConfig.commissionMode}
                      onChange={(e) => updateBacktest('commissionMode', e.target.value)}
                      className="select-field text-sm"
                    >
                      <option value="FIXED">固定金额</option>
                      <option value="PERCENT">百分比</option>
                      <option value="TIERED">阶梯费率</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    {backtestConfig.commissionMode === 'FIXED' ? (
                      <>
                        <label className="label-mono text-xs">每张费用 (¥)</label>
                        <input 
                          type="number" 
                          value={backtestConfig.commissionPerContract}
                          onChange={(e) => updateBacktest('commissionPerContract', parseFloat(e.target.value))}
                          step="0.1"
                          className="input-field text-sm" 
                        />
                      </>
                    ) : (
                      <>
                        <label className="label-mono text-xs">费率 (%)</label>
                        <input 
                          type="number" 
                          value={backtestConfig.commissionPercent * 100}
                          onChange={(e) => updateBacktest('commissionPercent', parseFloat(e.target.value) / 100)}
                          step="0.001"
                          className="input-field text-sm" 
                        />
                      </>
                    )}
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="label-mono text-xs">滑点模式</label>
                    <select 
                      value={backtestConfig.slippageMode}
                      onChange={(e) => updateBacktest('slippageMode', e.target.value)}
                      className="select-field text-sm"
                    >
                      <option value="FIXED">固定Tick</option>
                      <option value="PERCENT">百分比</option>
                      <option value="VOLUME_IMPACT">成交量冲击</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    {backtestConfig.slippageMode === 'FIXED' ? (
                      <>
                        <label className="label-mono text-xs">滑点Tick数</label>
                        <input 
                          type="number" 
                          value={backtestConfig.slippageTicks}
                          onChange={(e) => updateBacktest('slippageTicks', parseInt(e.target.value))}
                          min="0"
                          className="input-field text-sm" 
                        />
                      </>
                    ) : (
                      <>
                        <label className="label-mono text-xs">滑点 (%)</label>
                        <input 
                          type="number" 
                          value={backtestConfig.slippagePercent * 100}
                          onChange={(e) => updateBacktest('slippagePercent', parseFloat(e.target.value) / 100)}
                          step="0.01"
                          className="input-field text-sm" 
                        />
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Risk Controls */}
            <div className="glass-card p-3 space-y-3 border-[var(--accent-danger)]/30">
              <div className="flex items-center gap-2 text-xs font-bold text-[var(--accent-danger)]">
                <ShieldAlert className="w-4 h-4" />
                风险控制
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="label-mono text-xs">最大持仓 (张)</label>
                  <input 
                    type="number" 
                    value={backtestConfig.maxPositionSize}
                    onChange={(e) => updateBacktest('maxPositionSize', parseInt(e.target.value))}
                    className="input-field text-sm" 
                  />
                </div>
                <div className="space-y-1">
                  <label className="label-mono text-xs">最大回撤 (%)</label>
                  <input 
                    type="number" 
                    value={backtestConfig.maxDrawdown * 100}
                    onChange={(e) => updateBacktest('maxDrawdown', parseFloat(e.target.value) / 100)}
                    className="input-field text-sm" 
                  />
                </div>
                <div className="space-y-1">
                  <label className="label-mono text-xs">强平阈值 (%)</label>
                  <input 
                    type="number" 
                    value={backtestConfig.liquidationThreshold * 100}
                    onChange={(e) => updateBacktest('liquidationThreshold', parseFloat(e.target.value) / 100)}
                    className="input-field text-sm" 
                  />
                </div>
                <div className="space-y-1">
                  <label className="label-mono text-xs">日亏损限额 (%)</label>
                  <input 
                    type="number" 
                    value={backtestConfig.dailyLossLimit * 100}
                    onChange={(e) => updateBacktest('dailyLossLimit', parseFloat(e.target.value) / 100)}
                    className="input-field text-sm" 
                  />
                </div>
              </div>
            </div>

            {/* Execution Model */}
            <div className="glass-card p-3 space-y-3">
              <div className="flex items-center gap-2 text-xs font-bold text-[var(--text-secondary)]">
                <Target className="w-4 h-4" />
                执行模型
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="label-mono text-xs">成交价模型</label>
                  <select 
                    value={backtestConfig.fillModel}
                    onChange={(e) => updateBacktest('fillModel', e.target.value)}
                    className="select-field text-sm"
                  >
                    <option value="MIDPOINT">中间价</option>
                    <option value="WORST">最差价</option>
                    <option value="VWAP">VWAP</option>
                  </select>
                </div>
                <div className="space-y-1 flex items-end">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input 
                      type="checkbox" 
                      checked={backtestConfig.partialFillAllowed}
                      onChange={(e) => updateBacktest('partialFillAllowed', e.target.checked)}
                      className="accent-[var(--accent-primary)]"
                    />
                    <span className="text-xs text-[var(--text-secondary)]">允许部分成交</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================== SIMULATION MODE ==================== */}
        {mode === 'SIMULATION' && (
          <div className="space-y-4">
            {/* Notice about shared config */}
            <div className="p-3 bg-[var(--accent-primary)]/10 rounded-lg border border-[var(--accent-primary)]/30">
              <p className="text-xs text-[var(--text-secondary)]">
                📋 <strong>共用配置：</strong>模拟将使用回测模式的策略、数据、保证金、手续费、风控等参数
              </p>
              {!backtestConfig.strategyId && (
                <p className="text-xs text-[var(--accent-warning)] mt-1">
                  ⚠️ 请先在回测模式中选择策略
                </p>
              )}
            </div>

            {/* Monte Carlo Simulation Settings */}
            <div className="glass-card p-3 space-y-3">
              <div className="flex items-center gap-2 text-xs font-bold text-[var(--accent-secondary)]">
                <Zap className="w-4 h-4" />
                蒙特卡洛模拟设置
              </div>
              
              <div className="space-y-3">
                <div className="space-y-1">
                  <label className="label-mono text-xs">价格生成模型</label>
                  <select 
                    value={simConfig.model}
                    onChange={(e) => updateSim('model', e.target.value)}
                    className="select-field text-sm"
                  >
                    <option value="GBM">GBM (几何布朗运动)</option>
                    <option value="HESTON">Heston (随机波动率)</option>
                    <option value="MJD">Merton Jump Diffusion</option>
                    <option value="GARCH">GARCH</option>
                  </select>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="label-mono text-xs">模拟天数</label>
                    <input 
                      type="number" 
                      value={simConfig.simulationDays}
                      onChange={(e) => updateSim('simulationDays', parseInt(e.target.value))}
                      className="input-field text-sm" 
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="label-mono text-xs">模拟路径数</label>
                    <input 
                      type="number" 
                      value={simConfig.numPaths}
                      onChange={(e) => updateSim('numPaths', parseInt(e.target.value))}
                      className="input-field text-sm" 
                    />
                  </div>
                </div>
                
                <div className="space-y-1">
                  <label className="label-mono text-xs">市场情景 (自动设置 μ/σ)</label>
                  <select 
                    value={simConfig.marketRegime}
                    onChange={(e) => applyMarketRegimePreset(e.target.value)}
                    className="select-field text-sm"
                  >
                    <option value="BULL">🐂 牛市 (μ=15%, σ=15%)</option>
                    <option value="BEAR">🐻 熊市 (μ=-10%, σ=30%)</option>
                    <option value="SIDEWAYS">📊 震荡 (μ=3%, σ=20%)</option>
                    <option value="CRASH">💥 崩盘 (μ=-40%, σ=60%)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Base Model Parameters (μ, σ) */}
            <div className="glass-card p-3 space-y-3">
              <div className="flex items-center gap-2 text-xs font-bold text-[var(--accent-warning)]">
                <Layers className="w-4 h-4" />
                基础参数
              </div>
              
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="label-mono text-xs">参数来源</label>
                    <select 
                      value={simConfig.calibrationMode}
                      onChange={(e) => updateSim('calibrationMode', e.target.value)}
                      className="select-field text-sm"
                    >
                      <option value="manual">手动设置</option>
                      <option value="historical">从历史数据校准</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="label-mono text-xs">初始价格 (¥)</label>
                    <input 
                      type="number" 
                      value={simConfig.initialPrice}
                      onChange={(e) => updateSim('initialPrice', parseFloat(e.target.value))}
                      step="0.001"
                      className="input-field text-sm" 
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="label-mono text-xs">年化漂移率 μ (%)</label>
                    <input 
                      type="number" 
                      value={(simConfig.mu * 100).toFixed(1)}
                      onChange={(e) => updateSim('mu', parseFloat(e.target.value) / 100)}
                      step="1"
                      className="input-field text-sm" 
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="label-mono text-xs">年化波动率 σ (%)</label>
                    <input 
                      type="number" 
                      value={(simConfig.sigma * 100).toFixed(1)}
                      onChange={(e) => updateSim('sigma', parseFloat(e.target.value) / 100)}
                      step="1"
                      className="input-field text-sm" 
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Heston Model Parameters */}
            {simConfig.model === 'HESTON' && (
              <div className="glass-card p-3 space-y-3 border-l-2 border-purple-500">
                <div className="flex items-center gap-2 text-xs font-bold text-purple-400">
                  <Layers className="w-4 h-4" />
                  Heston 随机波动率参数
                </div>
                
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <label className="label-mono text-xs">初始方差 v₀ (%²)</label>
                      <input 
                        type="number" 
                        value={(simConfig.v0 * 100).toFixed(2)}
                        onChange={(e) => updateSim('v0', parseFloat(e.target.value) / 100)}
                        step="0.1"
                        className="input-field text-sm" 
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="label-mono text-xs">长期方差 θ (%²)</label>
                      <input 
                        type="number" 
                        value={(simConfig.theta * 100).toFixed(2)}
                        onChange={(e) => updateSim('theta', parseFloat(e.target.value) / 100)}
                        step="0.1"
                        className="input-field text-sm" 
                      />
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-2">
                    <div className="space-y-1">
                      <label className="label-mono text-xs">均值回复 κ</label>
                      <input 
                        type="number" 
                        value={simConfig.kappa.toFixed(2)}
                        onChange={(e) => updateSim('kappa', parseFloat(e.target.value))}
                        step="0.1"
                        className="input-field text-sm" 
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="label-mono text-xs">波动率之波动率 ξ</label>
                      <input 
                        type="number" 
                        value={simConfig.xi.toFixed(2)}
                        onChange={(e) => updateSim('xi', parseFloat(e.target.value))}
                        step="0.05"
                        className="input-field text-sm" 
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="label-mono text-xs">相关性 ρ</label>
                      <input 
                        type="number" 
                        value={simConfig.rho.toFixed(2)}
                        onChange={(e) => updateSim('rho', parseFloat(e.target.value))}
                        step="0.1"
                        min="-1"
                        max="1"
                        className="input-field text-sm" 
                      />
                    </div>
                  </div>
                  <p className="text-xs text-[var(--text-muted)]">
                    💡 ρ &lt; 0 表示杠杆效应（价格下跌时波动率上升）
                  </p>
                </div>
              </div>
            )}

            {/* MJD (Jump Diffusion) Parameters */}
            {simConfig.model === 'MJD' && (
              <div className="glass-card p-3 space-y-3 border-l-2 border-orange-500">
                <div className="flex items-center gap-2 text-xs font-bold text-orange-400">
                  <Layers className="w-4 h-4" />
                  Merton 跳跃扩散参数
                </div>
                
                <div className="grid grid-cols-3 gap-2">
                  <div className="space-y-1">
                    <label className="label-mono text-xs">跳跃强度 λ (次/年)</label>
                    <input 
                      type="number" 
                      value={simConfig.lambda.toFixed(2)}
                      onChange={(e) => updateSim('lambda', parseFloat(e.target.value))}
                      step="0.1"
                      min="0"
                      className="input-field text-sm" 
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="label-mono text-xs">跳跃均值 m (%)</label>
                    <input 
                      type="number" 
                      value={(simConfig.jumpMean * 100).toFixed(1)}
                      onChange={(e) => updateSim('jumpMean', parseFloat(e.target.value) / 100)}
                      step="0.5"
                      className="input-field text-sm" 
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="label-mono text-xs">跳跃波动 v (%)</label>
                    <input 
                      type="number" 
                      value={(simConfig.jumpVol * 100).toFixed(1)}
                      onChange={(e) => updateSim('jumpVol', parseFloat(e.target.value) / 100)}
                      step="1"
                      min="0"
                      className="input-field text-sm" 
                    />
                  </div>
                </div>
                <p className="text-xs text-[var(--text-muted)]">
                  💡 λ=1 表示平均每年1次跳跃，m &lt; 0 表示向下跳跃
                </p>
              </div>
            )}

            {/* GARCH Parameters */}
            {simConfig.model === 'GARCH' && (
              <div className="glass-card p-3 space-y-3 border-l-2 border-cyan-500">
                <div className="flex items-center gap-2 text-xs font-bold text-cyan-400">
                  <Layers className="w-4 h-4" />
                  GARCH(1,1) 参数
                </div>
                
                <div className="grid grid-cols-3 gap-2">
                  <div className="space-y-1">
                    <label className="label-mono text-xs">常数项 ω</label>
                    <input 
                      type="number" 
                      value={simConfig.omega.toExponential(2)}
                      onChange={(e) => updateSim('omega', parseFloat(e.target.value))}
                      step="0.000001"
                      className="input-field text-sm" 
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="label-mono text-xs">ARCH α</label>
                    <input 
                      type="number" 
                      value={simConfig.alpha.toFixed(3)}
                      onChange={(e) => updateSim('alpha', parseFloat(e.target.value))}
                      step="0.01"
                      min="0"
                      max="1"
                      className="input-field text-sm" 
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="label-mono text-xs">GARCH β</label>
                    <input 
                      type="number" 
                      value={simConfig.beta.toFixed(3)}
                      onChange={(e) => updateSim('beta', parseFloat(e.target.value))}
                      step="0.01"
                      min="0"
                      max="1"
                      className="input-field text-sm" 
                    />
                  </div>
                </div>
                <p className="text-xs text-[var(--text-muted)]">
                  💡 α + β &lt; 1 确保平稳性，当前: {(simConfig.alpha + simConfig.beta).toFixed(3)}
                </p>
              </div>
            )}

            {/* Stress Testing / Panic Factor */}
            <div className="glass-card p-3 space-y-3">
              <div className="flex items-center gap-2 text-xs font-bold text-[var(--accent-danger)]">
                <TrendingDown className="w-4 h-4" />
                压力测试
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-[var(--text-secondary)]">
                  <span>恐慌因子</span>
                  <span className="text-[var(--accent-danger)] font-mono font-bold">
                    {(simConfig.panicFactor * 100).toFixed(0)}%
                  </span>
                </div>
                <input 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.1" 
                  value={simConfig.panicFactor}
                  onChange={(e) => updateSim('panicFactor', parseFloat(e.target.value))}
                  className="slider-premium w-full"
                />
                <p className="text-xs text-[var(--text-muted)]">
                  提高恐慌因子会增加波动率偏度和跳跃频率
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Action Button - Fixed at bottom */}
      <div className="p-4 border-t border-[var(--border-primary)] bg-[var(--bg-secondary)]">
        {/* 配置修改提示 */}
        {configModified && (
          <div className="mb-3 p-2 rounded-lg bg-[var(--accent-warning)]/10 border border-[var(--accent-warning)]/30 flex items-center gap-2">
            <span className="text-[var(--accent-warning)] text-sm">⚠️</span>
            <span className="text-xs text-[var(--accent-warning)]">配置已修改，请重新运行回测</span>
          </div>
        )}
        <button 
          onClick={handleRun}
          disabled={isRunning}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {isRunning ? (
            <RefreshCw className="w-5 h-5 animate-spin" />
          ) : (
            <Play className="w-5 h-5 fill-current" />
          )}
          <span>启动 {mode === 'BACKTEST' ? '回测' : '蒙特卡洛模拟'}</span>
        </button>
      </div>
    </div>
  );
}
