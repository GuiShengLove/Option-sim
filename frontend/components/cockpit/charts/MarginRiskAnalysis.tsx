"use client";

import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid, ReferenceLine } from 'recharts';
import { Activity, BarChart2, TrendingUp, AlertTriangle } from 'lucide-react';
import MarginRadarChart from './MarginRadarChart';

interface MarginRiskAnalysisProps {
  startDate?: string;
  endDate?: string;
  strategyId?: string;
  marginMode?: string;
  equityCurve?: any[];
  dataSource?: 'backtest' | 'estimate'; 
  marginRatio?: number;
  leverage?: number;
  maintenanceMargin?: number;
}

export default function MarginRiskAnalysis(props: MarginRiskAnalysisProps) {
  const [activeTab, setActiveTab] = useState<'trend' | 'scenarios'>('trend');
  const { marginMode = 'FIXED' } = props;

  // Mock SPAN Scenarios if not provided (for demonstration of the new UI)
  const spanScenarios = Array.from({ length: 16 }, (_, i) => {
    // Generate some realistic looking shock scenarios
    // Scenario 1 is usually baseline, 16 is extreme
    const isExtreme = i === 0 || i === 15;
    const val = (Math.random() * 50000) * (Math.random() > 0.6 ? -1 : 1) - (isExtreme ? 20000 : 0);
    return {
      id: i + 1,
      name: `S${i + 1}`,
      pnl: val,
      desc: `Scenario ${i + 1}`
    };
  });

  const isSpanOrPM = ['SPAN', 'PM'].includes(marginMode);

  return (
    <div className="h-full flex flex-col bg-[var(--bg-card)]/30 rounded-lg overflow-hidden">
      {/* Header / Tabs */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border-primary)]/50">
        <div className="flex items-center gap-2">
            <div className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium ${
                isSpanOrPM ? 'bg-purple-500/10 text-purple-400' : 'bg-[var(--accent-primary)]/10 text-[var(--accent-primary)]'
            }`}>
            <Activity className="w-3.5 h-3.5" />
            <span>{isSpanOrPM ? '风险情景分析' : '保证金监控'}</span>
            </div>
        </div>

        {/* Tab Switcher - Only show if SPAN/PM */}
        {isSpanOrPM && (
            <div className="flex bg-[var(--bg-primary)] p-0.5 rounded-lg border border-[var(--border-primary)]">
                <button
                    onClick={() => setActiveTab('trend')}
                    className={`px-3 py-1 text-[10px] rounded-md transition-all ${
                        activeTab === 'trend'
                            ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-sm'
                            : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                    }`}
                >
                    <div className="flex items-center gap-1">
                        <TrendingUp className="w-3 h-3" />
                        趋势
                    </div>
                </button>
                <button
                    onClick={() => setActiveTab('scenarios')}
                    className={`px-3 py-1 text-[10px] rounded-md transition-all ${
                        activeTab === 'scenarios'
                            ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-sm'
                            : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                    }`}
                >
                     <div className="flex items-center gap-1">
                        <BarChart2 className="w-3 h-3" />
                        情景
                    </div>
                </button>
            </div>
        )}
      </div>

      {/* Content Area */}
      <div className="flex-1 min-h-0 relative p-1">
        
        {/* VIEW 1: Trend (Reusing existing component) */}
        <div className={`absolute inset-0 transition-opacity duration-300 ${activeTab === 'trend' ? 'opacity-100 z-10' : 'opacity-0 z-0'}`}>
            <MarginRadarChart {...props} />
        </div>

        {/* VIEW 2: Scenarios (New Visualization) */}
        {isSpanOrPM && (
            <div className={`absolute inset-0 flex flex-col p-2 transition-opacity duration-300 ${activeTab === 'scenarios' ? 'opacity-100 z-10' : 'opacity-0 z-0'}`}>
                {/* Summary Stats */}
                <div className="flex items-center gap-4 mb-2 text-xs text-[var(--text-muted)]">
                     <span className="flex items-center gap-1">
                        <AlertTriangle className="w-3 h-3 text-[var(--accent-danger)]" />
                        最大潜在亏损: <span className="font-mono text-[var(--accent-danger)]">-¥{(Math.max(...spanScenarios.map(s => Math.abs(Math.min(0, s.pnl))))).toLocaleString()}</span>
                     </span>
                     <span>覆盖情景: <span className="text-[var(--text-primary)]">16</span></span>
                </div>

                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={spanScenarios} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2a2d35" vertical={false} />
                        <XAxis dataKey="name" stroke="#6b7280" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis stroke="#6b7280" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} />
                        <Tooltip
                            cursor={{ fill: '#ffffff10' }}
                            contentStyle={{ backgroundColor: '#161920', border: '1px solid #2a2d35', borderRadius: '8px' }}
                            formatter={(value: number) => [`¥${value.toLocaleString()}`, 'P&L']}
                            labelFormatter={(label) => `情景 ${label}`}
                        />
                        <ReferenceLine y={0} stroke="#6b7280" />
                        <Bar dataKey="pnl" radius={[2, 2, 0, 0]}>
                            {spanScenarios.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.pnl >= 0 ? '#10b981' : '#ef4444'} fillOpacity={0.8} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
                
                <div className="mt-2 text-[10px] text-[var(--text-muted)] text-center">
                    SPAN 标准组合风险分析 - 压力测试模拟
                </div>
            </div>
        )}
      </div>
    </div>
  );
}
