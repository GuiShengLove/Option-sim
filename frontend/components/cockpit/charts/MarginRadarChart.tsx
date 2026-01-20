"use client";

import { useState, useEffect, useMemo } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Area, AreaChart } from 'recharts';
import { Loader2, AlertTriangle, Calendar, Info, Shield } from 'lucide-react';

interface MarginData {
  day: string;
  fullDate: string;
  margin: number;
  threshold: number;
}

interface EquityCurveItem {
  date: string;
  equity: number;
  cash: number;
  margin_utilization: number;
  position_count?: number;
}

interface MarginRadarChartProps {
  startDate?: string;
  endDate?: string;
  strategyId?: string;
  marginMode?: string;
  equityCurve?: EquityCurveItem[];  // Real data from backtest
  dataSource?: 'backtest' | 'estimate';  // Indicate data source
  // 新增：保证金参数显示
  marginRatio?: number;
  leverage?: number;
  maintenanceMargin?: number;
}

export default function MarginRadarChart({ 
  startDate, 
  endDate, 
  strategyId = 'default',
  marginMode = 'SSE',
  equityCurve,
  dataSource = 'backtest',
  marginRatio = 0.12,
  leverage = 1.0,
  maintenanceMargin = 0.08
}: MarginRadarChartProps) {
  const [loading, setLoading] = useState(false);
  
  // Process real equity curve data into margin chart format
  const marginData: MarginData[] = useMemo(() => {
    if (!equityCurve || equityCurve.length === 0) {
      return [];
    }
    
    return equityCurve.map(item => ({
      day: item.date.slice(5),  // MM-DD format
      fullDate: item.date,
      margin: Math.min(100, Math.max(0, (item.margin_utilization || 0) * 100)),  // Convert to percentage
      threshold: 80
    }));
  }, [equityCurve]);

  // Filter by date range if specified
  const viewData = useMemo(() => {
    if (!marginData.length) return [];
    
    let filtered = marginData;
    if (startDate) {
      filtered = filtered.filter(d => d.fullDate >= startDate);
    }
    if (endDate) {
      filtered = filtered.filter(d => d.fullDate <= endDate);
    }
    return filtered;
  }, [marginData, startDate, endDate]);

  // Calculate stats
  const currentMargin = viewData.length > 0 ? viewData[viewData.length - 1].margin : 0;
  const maxMargin = viewData.length > 0 ? Math.max(...viewData.map(d => d.margin)) : 0;
  const avgMargin = viewData.length > 0 
    ? viewData.reduce((sum, d) => sum + d.margin, 0) / viewData.length 
    : 0;
  
  const isWarning = currentMargin > 70;
  const isDanger = currentMargin > 85;
  const hasRealData = equityCurve && equityCurve.length > 0;

  // Margin mode display names
  const marginModeNames: Record<string, string> = {
    'FIXED': '固定比例',
    'SSE': '上交所标准',
    'SPAN': 'SPAN风险分析',
    'PM': '组合保证金',
    'PORTFOLIO': '组合保证金'
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full flex-col gap-2">
        <Loader2 className="w-6 h-6 animate-spin text-[var(--accent-primary)]" />
        <span className="text-xs text-[var(--text-muted)]">加载保证金数据...</span>
      </div>
    );
  }

  // No data state
  if (!hasRealData) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-[var(--text-muted)] gap-3">
        <Shield className="w-12 h-12 opacity-30" />
        <p className="text-sm">运行回测后显示保证金使用情况</p>
        <div className="flex items-center gap-2 text-xs bg-[var(--bg-card)] px-3 py-2 rounded-lg">
          <Info className="w-3 h-3" />
          <span>当前模式: <strong className="text-[var(--accent-primary)]">{marginModeNames[marginMode] || marginMode}</strong></span>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header with Mode and Stats */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 px-2 py-1 bg-[var(--accent-primary)]/10 rounded text-xs">
            <Shield className="w-3 h-3 text-[var(--accent-primary)]" />
            <span className="text-[var(--accent-primary)] font-medium">
              {marginModeNames[marginMode] || marginMode}
            </span>
          </div>
          
          <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
            <span>
              当前: 
              <span className={`ml-1 font-mono font-bold ${
                isDanger ? 'text-[var(--accent-danger)]' : 
                isWarning ? 'text-[var(--accent-warning)]' : 
                'text-[var(--accent-success)]'
              }`}>
                {currentMargin.toFixed(1)}%
              </span>
            </span>
            <span className="text-[var(--border-primary)]">|</span>
            <span>
              峰值: <span className="font-mono text-[var(--text-primary)]">{maxMargin.toFixed(1)}%</span>
            </span>
            <span className="text-[var(--border-primary)]">|</span>
            <span>
              均值: <span className="font-mono text-[var(--text-secondary)]">{avgMargin.toFixed(1)}%</span>
            </span>
          </div>
        </div>

        {isDanger && (
          <span className="flex items-center gap-1 text-xs text-[var(--accent-danger)] bg-[var(--accent-danger)]/10 px-2 py-1 rounded">
            <AlertTriangle className="w-3 h-3" />
            高风险
          </span>
        )}
      </div>
      
      {/* 参数指标条 */}
      <div className="flex items-center gap-4 mb-2 px-2 py-1.5 bg-[var(--bg-card)]/50 rounded text-[10px] text-[var(--text-muted)]">
        <span>保证金率: <strong className="text-[var(--text-secondary)] font-mono">{(marginRatio * 100).toFixed(0)}%</strong></span>
        <span>杠杆: <strong className="text-[var(--text-secondary)] font-mono">{leverage.toFixed(1)}x</strong></span>
        <span>维保: <strong className="text-[var(--text-secondary)] font-mono">{(maintenanceMargin * 100).toFixed(0)}%</strong></span>
      </div>
      
      {/* Chart */}
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={viewData}>
            <defs>
              <linearGradient id="marginGradientReal" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.05}/>
              </linearGradient>
            </defs>
            
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2d35" vertical={false} />
            
            <XAxis 
              dataKey="day" 
              stroke="#6b7280" 
              fontSize={10} 
              tickLine={false} 
              axisLine={false}
              interval="preserveStartEnd"
              minTickGap={40}
            />
            
            <YAxis 
              stroke="#6b7280" 
              fontSize={10} 
              tickLine={false} 
              axisLine={false}
              domain={[0, 100]}
              tickFormatter={(v) => `${v}%`}
              width={35}
            />
            
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#161920', 
                border: '1px solid #2a2d35',
                borderRadius: '8px',
                padding: '8px 12px'
              }}
              labelStyle={{ color: '#9ca3af', fontSize: 11 }}
              formatter={(value: any) => [
                `${typeof value === 'number' ? value.toFixed(1) : value}%`, 
                '保证金占用'
              ]}
              labelFormatter={(label) => `日期: ${label}`}
            />
            
            {/* Danger zone */}
            <ReferenceLine 
              y={90} 
              stroke="#dc2626" 
              strokeDasharray="3 3"
              label={{ value: '强平线', position: 'right', fill: '#dc2626', fontSize: 8 }}
            />
            
            {/* Warning threshold */}
            <ReferenceLine 
              y={80} 
              stroke="#ef4444" 
              strokeDasharray="5 5"
              label={{ value: '警戒线', position: 'right', fill: '#ef4444', fontSize: 9 }}
            />
            
            {/* Safe zone */}
            <ReferenceLine 
              y={50} 
              stroke="#10b981" 
              strokeDasharray="3 3"
              label={{ value: '安全', position: 'right', fill: '#10b981', fontSize: 9 }}
            />
            
            <Area 
              type="monotone" 
              dataKey="margin" 
              stroke="#f59e0b" 
              strokeWidth={2}
              fill="url(#marginGradientReal)"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      
      {/* Footer with data source indicator */}
      <div className="mt-2 flex items-center justify-between text-[10px] text-[var(--text-muted)]">
        <span className="flex items-center gap-1">
          <div className={`w-1.5 h-1.5 rounded-full ${hasRealData ? 'bg-[var(--accent-success)]' : 'bg-[var(--text-muted)]'}`} />
          {hasRealData ? '真实回测数据' : '预估数据'}
        </span>
        <span>
          数据点: {viewData.length}
        </span>
      </div>
    </div>
  );
}

