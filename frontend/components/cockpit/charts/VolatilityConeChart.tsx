"use client";

import { useMemo } from 'react';
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  LineChart,
  Line,
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid,
  ReferenceDot,
  Legend
} from 'recharts';
import { Activity, TrendingUp, TrendingDown, AlertTriangle, CheckCircle } from 'lucide-react';

interface VolatilityConeData {
  dte: number;
  current_iv: number;
  percentile_rank: number;
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
}

interface VolatilityConeChartProps {
  data: VolatilityConeData[] | null;
  loading?: boolean;
}

/**
 * Volatility Cone Chart Component
 * 
 * Displays historical IV percentiles (P10-P90) as a cone shape,
 * with the current IV plotted against it to show if IV is rich or cheap.
 */
export default function VolatilityConeChart({ data, loading }: VolatilityConeChartProps) {
  // Transform data for area chart - need to calculate band heights
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    // Sort by DTE for proper display
    const sorted = [...data].sort((a, b) => a.dte - b.dte);
    
    return sorted.map(d => ({
      dte: d.dte,
      dteLabel: `${d.dte}天`,
      // Convert from percentage format to decimal for display
      currentIV: d.current_iv,
      p10: d.p10,
      p25: d.p25,
      p50: d.p50,
      p75: d.p75,
      p90: d.p90,
      percentileRank: d.percentile_rank,
      // For stacked areas, we need differential values
      // But for simplicity, we'll use overlapping areas with different opacities
    }));
  }, [data]);

  // Calculate overall IV status
  const ivStatus = useMemo(() => {
    if (!data || data.length === 0) return null;
    
    // Use shortest DTE (typically 7 days) for overall status
    const shortTerm = data.find(d => d.dte <= 14) || data[0];
    const rank = shortTerm.percentile_rank;
    
    if (rank >= 75) {
      return { 
        label: 'IV偏高', 
        color: 'var(--accent-danger)', 
        bgColor: 'rgba(239, 68, 68, 0.15)',
        icon: TrendingUp,
        description: '当前IV处于历史高位，可考虑卖方策略'
      };
    } else if (rank <= 25) {
      return { 
        label: 'IV偏低', 
        color: 'var(--accent-success)', 
        bgColor: 'rgba(16, 185, 129, 0.15)',
        icon: TrendingDown,
        description: '当前IV处于历史低位，可考虑买方策略'
      };
    } else {
      return { 
        label: 'IV正常', 
        color: 'var(--accent-warning)', 
        bgColor: 'rgba(245, 158, 11, 0.15)',
        icon: Activity,
        description: '当前IV处于历史中位水平'
      };
    }
  }, [data]);

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-sm text-[var(--text-muted)]">加载波动率锥数据...</div>
      </div>
    );
  }

  // No data state
  if (!data || data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-2">
        <AlertTriangle className="w-6 h-6 text-[var(--text-muted)]" />
        <div className="text-sm text-[var(--text-muted)]">暂无波动率锥数据</div>
      </div>
    );
  }

  const StatusIcon = ivStatus?.icon || Activity;

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Header with IV Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-[var(--accent-secondary)]" />
          <span className="text-xs font-semibold text-[var(--text-secondary)]">
            波动率锥 (Volatility Cone)
          </span>
        </div>
        
        {ivStatus && (
          <div 
            className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium"
            style={{ backgroundColor: ivStatus.bgColor, color: ivStatus.color }}
          >
            <StatusIcon className="w-3 h-3" />
            {ivStatus.label}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: 'rgba(99, 102, 241, 0.2)' }} />
          <span className="text-[var(--text-muted)]">P10-P90</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: 'rgba(99, 102, 241, 0.4)' }} />
          <span className="text-[var(--text-muted)]">P25-P75</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5 rounded-full bg-[var(--accent-warning)]" />
          <span className="text-[var(--text-muted)]">中位数P50</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5 rounded-full bg-[var(--accent-primary)]" />
          <span className="text-[var(--text-muted)]">当前IV</span>
        </div>
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
            <defs>
              {/* Gradient for P10-P90 band */}
              <linearGradient id="bandOuter" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.15} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0.05} />
              </linearGradient>
              {/* Gradient for P25-P75 band */}
              <linearGradient id="bandInner" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0.15} />
              </linearGradient>
            </defs>
            
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2d35" vertical={false} />
            
            <XAxis 
              dataKey="dte" 
              stroke="#6b7280" 
              fontSize={10} 
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `${v}天`}
            />
            
            <YAxis 
              stroke="#6b7280" 
              fontSize={10} 
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `${v}%`}
              domain={['auto', 'auto']}
              width={40}
            />
            
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#161920', 
                border: '1px solid #2a2d35',
                borderRadius: '8px',
                padding: '10px'
              }}
              labelStyle={{ color: '#9ca3af', fontSize: 11, marginBottom: 6 }}
              labelFormatter={(label) => `到期: ${label}天`}
              formatter={(value: number, name: string) => {
                const labels: Record<string, string> = {
                  'p90': 'P90',
                  'p75': 'P75',
                  'p50': 'P50 (中位数)',
                  'p25': 'P25',
                  'p10': 'P10',
                  'currentIV': '当前IV'
                };
                return [`${value.toFixed(1)}%`, labels[name] || name];
              }}
            />

            {/* P10-P90 outer band (requires area between) */}
            <Area 
              type="monotone" 
              dataKey="p90" 
              stroke="none"
              fill="url(#bandOuter)"
              fillOpacity={1}
            />
            <Area 
              type="monotone" 
              dataKey="p10" 
              stroke="none"
              fill="#161920"
              fillOpacity={1}
            />
            
            {/* P25-P75 inner band */}
            <Area 
              type="monotone" 
              dataKey="p75" 
              stroke="none"
              fill="url(#bandInner)"
              fillOpacity={1}
            />
            <Area 
              type="monotone" 
              dataKey="p25" 
              stroke="none"
              fill="#161920"
              fillOpacity={1}
            />

            {/* P50 median line */}
            <Line 
              type="monotone" 
              dataKey="p50" 
              stroke="#f59e0b" 
              strokeWidth={1.5}
              strokeDasharray="4 4"
              dot={false}
            />

            {/* Current IV line */}
            <Line 
              type="monotone" 
              dataKey="currentIV" 
              stroke="#8b5cf6" 
              strokeWidth={2.5}
              dot={{ r: 4, fill: '#8b5cf6', stroke: '#fff', strokeWidth: 1 }}
              activeDot={{ r: 6, fill: '#a78bfa' }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Percentile Details */}
      <div className="grid grid-cols-5 gap-2 pt-2 border-t border-[var(--border-subtle)]">
        {chartData.map((d) => (
          <div key={d.dte} className="text-center">
            <div className="text-xs text-[var(--text-muted)] mb-1">{d.dte}天</div>
            <div 
              className={`text-sm font-bold ${
                d.percentileRank >= 75 ? 'text-[var(--accent-danger)]' :
                d.percentileRank <= 25 ? 'text-[var(--accent-success)]' :
                'text-[var(--accent-warning)]'
              }`}
            >
              P{Math.round(d.percentileRank)}
            </div>
            <div className="text-xs text-[var(--text-muted)]">{d.currentIV.toFixed(1)}%</div>
          </div>
        ))}
      </div>

      {/* Strategy Suggestion */}
      {ivStatus && (
        <div 
          className="flex items-start gap-2 p-2 rounded-lg text-xs"
          style={{ backgroundColor: ivStatus.bgColor }}
        >
          <CheckCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: ivStatus.color }} />
          <span style={{ color: ivStatus.color }}>{ivStatus.description}</span>
        </div>
      )}
    </div>
  );
}
