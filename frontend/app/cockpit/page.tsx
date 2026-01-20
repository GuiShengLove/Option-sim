"use client";

import { useState, useCallback, useRef } from 'react';
import MissionControl from '@/components/cockpit/MissionControl';
import HolographicDashboard from '@/components/cockpit/HolographicDashboard';

export interface BacktestConfig {
  initialCapital: number;
  startDate: string;
  endDate: string;
  marginMode: string;
  marginRatio: number;
  leverage: number;
  maintenanceMargin?: number;
  strategyId?: string;
  datasetId?: string;
}

export interface SimConfig {
  initialCapital: number;
  model: string;
  simulationDays: number;
  numPaths: number;
  marketRegime: string;
  panicFactor: number;
}

export interface BacktestResult {
  success: boolean;
  message: string;
  metrics?: {
    total_return: number;
    max_drawdown: number;
    sharpe_ratio: number;
    final_equity: number;
    trade_count: number;
  };
  equity_curve?: Array<{
    date: string;
    equity: number;
    cash: number;
    margin_utilization: number;
  }>;
  strategy_name: string;
}

export default function CockpitPage() {
  const [backtestConfig, setBacktestConfig] = useState<BacktestConfig>({
    initialCapital: 1000000,
    startDate: '2020-01-02',
    endDate: '2020-12-31',
    marginMode: 'FIXED',
    marginRatio: 0.12,
    leverage: 1.0,
  });
  
  const [simConfig, setSimConfig] = useState<SimConfig>({
    initialCapital: 1000000,
    model: 'GBM',
    simulationDays: 30,
    numPaths: 1000,
    marketRegime: 'SIDEWAYS',
    panicFactor: 0,
  });
  
  const [runTimestamp, setRunTimestamp] = useState<number | null>(null);
  const [progress, setProgress] = useState(0); 
  const [isSimulating, setIsSimulating] = useState(false);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const handleConfigChange = useCallback((type: 'backtest' | 'simulation', config: any) => {
    if (type === 'backtest') {
      setBacktestConfig(config);
    } else {
      setSimConfig(config);
    }
  }, []);

  const handleRun = useCallback(async () => {
    // 检查是否选择了策略
    const currentStrategyId = backtestConfig.strategyId;
    console.log('Running backtest with Strategy ID:', currentStrategyId);
    
    if (!currentStrategyId || currentStrategyId.length === 0) {
      // 未选择策略，显示提示
      setBacktestResult({
        success: false,
        message: '请先选择一个回测策略',
        strategy_name: ''
      });
      setRunTimestamp(Date.now()); // 触发 UI 更新显示错误
      return;
    }
    
    if (timerRef.current) clearInterval(timerRef.current);
    
    setIsSimulating(true);
    setProgress(0);
    
    // Simulate initial progress while waiting for API
    const startTime = Date.now();
    timerRef.current = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const p = Math.min(90, (elapsed / 5000) * 90); // Cap at 90% until API returns
      setProgress(p);
    }, 100);

    try {
      const response = await fetch('http://localhost:8000/api/backtest/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_id: currentStrategyId,
          dataset_id: backtestConfig.datasetId || '510050_SH',
          start_date: backtestConfig.startDate,
          end_date: backtestConfig.endDate,
          initial_capital: backtestConfig.initialCapital,
          margin_scheme: backtestConfig.marginMode,
          margin_ratio: backtestConfig.marginRatio,
          maintenance_margin: backtestConfig.maintenanceMargin || 0.08,
          leverage: backtestConfig.leverage
        })
      });
      
      const result: BacktestResult = await response.json();
      setBacktestResult(result);
      
      console.log('Backtest result:', result);
      
    } catch (error) {
      console.error('Backtest error:', error);
      setBacktestResult({
        success: false,
        message: `Backtest failed: ${error}`,
        strategy_name: 'Error'
      });
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
      setProgress(100);
      setRunTimestamp(Date.now());
      setTimeout(() => {
        setIsSimulating(false);
        setProgress(0);
      }, 500);
    }
  }, [backtestConfig]);


  return (
    <div className="flex h-screen w-full">
        {/* Left: Mission Control */}
        <div className="shrink-0 h-full">
            <MissionControl 
              onConfigChange={handleConfigChange}
              onRun={handleRun}
            />
        </div>
        
        {/* Right: Dashboard */}
        <div className="flex-1 h-full min-w-0 relative">
            <HolographicDashboard 
              backtestConfig={backtestConfig}
              simConfig={simConfig}
              runTimestamp={runTimestamp}
              backtestResult={backtestResult}
            />
            
            {/* Progress Overlay */}
            {isSimulating && (
                <div className="absolute inset-0 bg-black/60 backdrop-blur-sm z-50 flex flex-col items-center justify-center">
                    <div className="w-96 space-y-4">
                        <div className="flex justify-between text-sm text-[var(--accent-primary)] font-mono">
                            <span>正在执行量化回测...</span>
                            <span>{progress.toFixed(0)}%</span>
                        </div>
                        <div className="h-2 bg-[var(--bg-card)] rounded-full overflow-hidden border border-[var(--border-primary)]">
                            <div 
                                className="h-full bg-[var(--accent-primary)] transition-all duration-100 ease-out shadow-[0_0_10px_var(--accent-primary)]"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                        <div className="text-center text-xs text-[var(--text-muted)] animate-pulse">
                            {backtestConfig.strategyId 
                              ? `运行策略回测: ${backtestConfig.strategyId}...`
                              : '正在计算 Greeks, 更新保证金曲线...'}
                        </div>
                    </div>
                </div>
            )}
        </div>
    </div>
  );
}

