"use client";

import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { 
  Grid3X3, TrendingUp, TrendingDown, Calendar, RefreshCw, 
  Calculator, ChevronDown, ChevronUp, Activity, Loader2,
  Filter, Search, Eye, EyeOff, Info, Settings, BarChart3,
  Target, Zap, Clock, ArrowUpDown
} from 'lucide-react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface OptionContract {
  id: string;
  type: 'call' | 'put';
  strike: number;
  expiry: string;
  dte: number;
  bid: number;
  ask: number;
  last: number;
  change: number;
  changePercent: number;
  volume: number;
  openInterest: number;
  iv: number;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  vanna: number;
  volga: number;
}

interface ExpiryGroup {
  expiry: string;
  dte: number;
  calls: OptionContract[];
  puts: OptionContract[];
  strikes: number[];
}

interface OptionChainWithGreeksProps {
  selectedDate?: string;
  dateRange?: { start: string; end: string };
  datasetId?: string;
}

// Greeks Tooltip Component
const GreekTooltip = ({ symbol, children }: { symbol: string; children: React.ReactNode }) => {
  const [show, setShow] = useState(false);
  
  const tooltipContent: Record<string, { name: string; desc: string }> = {
    'Δ': { name: 'Delta', desc: '价格敏感度：标的价格变动1元时，期权价格变动量' },
    'Γ': { name: 'Gamma', desc: 'Delta变化率：标的价格变动1元时，Delta的变化量' },
    'Θ': { name: 'Theta', desc: '时间衰减：每过一天，期权价值的损耗量（通常为负）' },
    'ν': { name: 'Vega', desc: '波动率敏感度：隐含波动率变动1%时，期权价格变动量' },
    'IV': { name: '隐含波动率', desc: '市场对未来波动的预期，越高代表期权越贵' },
    'Vanna': { name: 'Vanna', desc: '二阶Greeks：Delta对波动率的敏感度' },
    'Volga': { name: 'Volga', desc: '二阶Greeks：Vega对波动率的敏感度' },
  };
  
  const info = tooltipContent[symbol] || { name: symbol, desc: '' };
  
  return (
    <div className="relative inline-block" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded-lg shadow-lg z-50 whitespace-nowrap">
          <div className="font-bold text-[var(--accent-primary)] text-sm">{info.name}</div>
          <div className="text-xs text-[var(--text-muted)] max-w-[200px] whitespace-normal">{info.desc}</div>
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-[var(--border-primary)]" />
        </div>
      )}
    </div>
  );
};

// Mini Bar Chart for IV visualization
const MiniIVBar = ({ value, max = 1 }: { value: number; max?: number }) => {
  const width = Math.min((value / max) * 100, 100);
  const color = value > 0.5 ? 'var(--accent-danger)' : value > 0.3 ? 'var(--accent-warning)' : 'var(--accent-success)';
  
  return (
    <div className="flex items-center gap-1">
      <div className="w-12 h-2 bg-[var(--bg-elevated)] rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${width}%`, backgroundColor: color }} />
      </div>
      <span className="font-mono text-xs">{(value * 100).toFixed(1)}%</span>
    </div>
  );
};

// Greeks Radar Chart Component
const GreeksRadarChart = ({ option }: { option: OptionContract | null }) => {
  if (!option) return null;
  
  const normalizedData = {
    delta: Math.abs(option.delta),
    gamma: Math.min(option.gamma * 20, 1),
    theta: Math.min(Math.abs(option.theta) * 50, 1),
    vega: Math.min(option.vega * 5, 1),
    iv: option.iv,
  };
  
  const radarData = [{
    type: 'scatterpolar' as const,
    r: [normalizedData.delta, normalizedData.gamma, normalizedData.theta, normalizedData.vega, normalizedData.iv, normalizedData.delta],
    theta: ['Delta', 'Gamma', 'Theta', 'Vega', 'IV', 'Delta'],
    fill: 'toself',
    fillcolor: 'rgba(139, 92, 246, 0.3)',
    line: { color: 'rgb(139, 92, 246)', width: 2 },
    marker: { size: 6, color: 'rgb(139, 92, 246)' },
  }];
  
  const layout = {
    polar: {
      radialaxis: { visible: true, range: [0, 1], tickfont: { size: 8, color: '#6b7280' }, gridcolor: '#374151' },
      angularaxis: { tickfont: { size: 10, color: '#9ca3af' }, gridcolor: '#374151' },
      bgcolor: 'transparent',
    },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 40, r: 40, t: 20, b: 20 },
    showlegend: false,
  };
  
  return (
    <div className="glass-card-elevated p-3">
      <div className="flex items-center gap-2 mb-2">
        <Target className="w-4 h-4 text-[var(--accent-primary)]" />
        <span className="text-sm font-medium">Greeks 雷达图</span>
        <span className="text-xs text-[var(--text-muted)]">
          {option.type === 'call' ? '认购' : '认沽'} K={option.strike}
        </span>
      </div>
      <Plot
        data={radarData}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '180px' }}
        useResizeHandler={true}
      />
    </div>
  );
};

// Filter Bar Component
interface FilterState {
  moneyness: 'all' | 'itm' | 'atm' | 'otm';
  deltaMin: number;
  deltaMax: number;
  ivMin: number;
  ivMax: number;
  strikeSearch: string;
  volumeMin: number;
}

const FilterBar = ({ 
  filters, 
  onFilterChange, 
  spotPrice 
}: { 
  filters: FilterState; 
  onFilterChange: (filters: FilterState) => void;
  spotPrice: number;
}) => {
  const [expanded, setExpanded] = useState(false);
  
  return (
    <div className="glass-card p-3 space-y-3">
      {/* Quick Filters */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-[var(--accent-primary)]" />
          <span className="text-sm text-[var(--text-muted)]">快速筛选:</span>
        </div>
        
        {/* Moneyness Filter */}
        <div className="flex gap-1 bg-[var(--bg-elevated)] p-1 rounded-lg">
          {(['all', 'itm', 'atm', 'otm'] as const).map((m) => (
            <button
              key={m}
              onClick={() => onFilterChange({ ...filters, moneyness: m })}
              className={`px-3 py-1 text-xs rounded transition-all ${
                filters.moneyness === m 
                  ? 'bg-[var(--accent-primary)] text-white' 
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              }`}
            >
              {m === 'all' ? '全部' : m === 'itm' ? '价内ITM' : m === 'atm' ? '平值ATM' : '价外OTM'}
            </button>
          ))}
        </div>
        
        {/* Strike Search */}
        <div className="flex items-center gap-2 bg-[var(--bg-elevated)] rounded-lg px-3 py-1.5">
          <Search className="w-4 h-4 text-[var(--text-muted)]" />
          <input
            type="text"
            placeholder="搜索行权价..."
            value={filters.strikeSearch}
            onChange={(e) => onFilterChange({ ...filters, strikeSearch: e.target.value })}
            className="bg-transparent text-sm text-[var(--text-primary)] w-24 focus:outline-none placeholder:text-[var(--text-muted)]"
          />
        </div>
        
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-xs text-[var(--accent-primary)] hover:underline"
        >
          <Settings className="w-3 h-3" />
          {expanded ? '收起高级筛选' : '高级筛选'}
          {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </button>
      </div>
      
      {/* Advanced Filters */}
      {expanded && (
        <div className="grid grid-cols-4 gap-4 pt-3 border-t border-[var(--border-primary)]">
          {/* Delta Range */}
          <div className="space-y-1">
            <label className="text-xs text-[var(--text-muted)]">Delta 范围</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                value={filters.deltaMin}
                onChange={(e) => onFilterChange({ ...filters, deltaMin: parseFloat(e.target.value) || 0 })}
                className="w-16 px-2 py-1 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
              />
              <span className="text-xs text-[var(--text-muted)]">~</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                value={filters.deltaMax}
                onChange={(e) => onFilterChange({ ...filters, deltaMax: parseFloat(e.target.value) || 1 })}
                className="w-16 px-2 py-1 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
              />
            </div>
          </div>
          
          {/* IV Range */}
          <div className="space-y-1">
            <label className="text-xs text-[var(--text-muted)]">IV 范围 (%)</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min="0"
                max="200"
                step="5"
                value={filters.ivMin}
                onChange={(e) => onFilterChange({ ...filters, ivMin: parseFloat(e.target.value) || 0 })}
                className="w-16 px-2 py-1 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
              />
              <span className="text-xs text-[var(--text-muted)]">~</span>
              <input
                type="number"
                min="0"
                max="200"
                step="5"
                value={filters.ivMax}
                onChange={(e) => onFilterChange({ ...filters, ivMax: parseFloat(e.target.value) || 200 })}
                className="w-16 px-2 py-1 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
              />
            </div>
          </div>
          
          {/* Volume Min */}
          <div className="space-y-1">
            <label className="text-xs text-[var(--text-muted)]">最小成交量</label>
            <input
              type="number"
              min="0"
              step="100"
              value={filters.volumeMin}
              onChange={(e) => onFilterChange({ ...filters, volumeMin: parseInt(e.target.value) || 0 })}
              className="w-full px-2 py-1 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
            />
          </div>
          
          {/* Reset Button */}
          <div className="flex items-end">
            <button
              onClick={() => onFilterChange({
                moneyness: 'all',
                deltaMin: 0,
                deltaMax: 1,
                ivMin: 0,
                ivMax: 200,
                strikeSearch: '',
                volumeMin: 0,
              })}
              className="px-3 py-1 text-xs bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] rounded transition-colors"
            >
              重置筛选
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// Column Visibility Control
interface ColumnVisibility {
  delta: boolean;
  gamma: boolean;
  theta: boolean;
  vega: boolean;
  iv: boolean;
  volume: boolean;
  vanna: boolean;
  volga: boolean;
}

const ColumnSettings = ({ 
  visibility, 
  onVisibilityChange 
}: { 
  visibility: ColumnVisibility; 
  onVisibilityChange: (v: ColumnVisibility) => void;
}) => {
  const [open, setOpen] = useState(false);
  
  const columns = [
    { key: 'delta', label: 'Delta (Δ)' },
    { key: 'gamma', label: 'Gamma (Γ)' },
    { key: 'theta', label: 'Theta (Θ)' },
    { key: 'vega', label: 'Vega (ν)' },
    { key: 'iv', label: 'IV' },
    { key: 'volume', label: '成交量' },
    { key: 'vanna', label: 'Vanna' },
    { key: 'volga', label: 'Volga' },
  ];
  
  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 px-3 py-1.5 text-xs bg-[var(--bg-elevated)] hover:bg-[var(--bg-card-hover)] rounded-lg transition-colors"
      >
        <Eye className="w-3 h-3" />
        列设置
      </button>
      
      {open && (
        <div className="absolute right-0 top-full mt-2 p-3 bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded-lg shadow-lg z-50 min-w-[160px]">
          <div className="text-xs font-medium text-[var(--text-muted)] mb-2">显示列</div>
          {columns.map(col => (
            <label key={col.key} className="flex items-center gap-2 py-1 text-sm cursor-pointer hover:text-[var(--accent-primary)]">
              <input
                type="checkbox"
                checked={visibility[col.key as keyof ColumnVisibility]}
                onChange={(e) => onVisibilityChange({ ...visibility, [col.key]: e.target.checked })}
                className="accent-[var(--accent-primary)]"
              />
              {col.label}
            </label>
          ))}
        </div>
      )}
    </div>
  );
};

export default function OptionChainWithGreeks({ selectedDate, dateRange, datasetId = '510050_SH' }: OptionChainWithGreeksProps) {
  const [spotPrice, setSpotPrice] = useState(3.0);
  const [selectedExpiry, setSelectedExpiry] = useState<string | null>(null);
  const [expiryGroups, setExpiryGroups] = useState<ExpiryGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAdvancedGreeks, setShowAdvancedGreeks] = useState(false);
  const [selectedOption, setSelectedOption] = useState<OptionContract | null>(null);
  const [highlightATM, setHighlightATM] = useState(true);
  const [currentDate, setCurrentDate] = useState(selectedDate || '');
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [selectedRowIndex, setSelectedRowIndex] = useState<number>(-1);
  const tableRef = useRef<HTMLDivElement>(null);
  
  // Filter state
  const [filters, setFilters] = useState<FilterState>({
    moneyness: 'all',
    deltaMin: 0,
    deltaMax: 1,
    ivMin: 0,
    ivMax: 200,
    strikeSearch: '',
    volumeMin: 0,
  });
  
  // Column visibility state
  const [columnVisibility, setColumnVisibility] = useState<ColumnVisibility>({
    delta: true,
    gamma: true,
    theta: true,
    vega: true,
    iv: true,
    volume: false,
    vanna: true,
    volga: true,
  });

  // Keyboard navigation
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!tableRef.current) return;
    
    const currentGroup = expiryGroups.find(g => g.expiry === selectedExpiry);
    if (!currentGroup) return;
    
    const maxIndex = currentGroup.strikes.length - 1;
    
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedRowIndex(prev => Math.min(prev + 1, maxIndex));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedRowIndex(prev => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter' && selectedRowIndex >= 0) {
      const call = currentGroup.calls[selectedRowIndex];
      setSelectedOption(call);
    }
  }, [expiryGroups, selectedExpiry, selectedRowIndex]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Sync with external selectedDate prop
  useEffect(() => {
    if (selectedDate && selectedDate !== currentDate) {
      setCurrentDate(selectedDate);
    }
  }, [selectedDate]);

  // Fetch available dates and filter by dateRange
  useEffect(() => {
    const fetchDates = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/data/dates?dataset_id=${datasetId}`);
        if (response.ok) {
          const data = await response.json();
          let dates = data.dates || [];
          // Filter by dateRange if provided
          if (dateRange?.start && dateRange?.end) {
            dates = dates.filter((d: string) => d >= dateRange.start && d <= dateRange.end);
          }
          setAvailableDates(dates);
          // Only set default if no external date provided
          if (!currentDate && !selectedDate && dates.length > 0) {
            setCurrentDate(dates[0]);
          }
        }
      } catch (err) {
        console.error('Failed to fetch dates:', err);
      }
    };
    fetchDates();
  }, [dateRange?.start, dateRange?.end, datasetId]);

  // Fetch and process option chain data with real Greeks from backend
  useEffect(() => {
    if (!currentDate) return;

    const fetchOptionChain = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Step 1: Get basic asset data
        const assetsResponse = await fetch(`http://localhost:8000/api/data/assets?date=${currentDate}&limit=200&dataset_id=${datasetId}`);
        
        if (!assetsResponse.ok) {
          throw new Error(`HTTP ${assetsResponse.status}: ${assetsResponse.statusText}`);
        }
        
        const assetsData = await assetsResponse.json();
        const assets = assetsData.assets || [];
        
        if (assets.length === 0) {
          setExpiryGroups([]);
          setLoading(false);
          return;
        }
        
        // Estimate spot price from mid strike
        const strikes = assets.map((a: any) => a.strike).filter((s: number) => s > 0);
        const estimatedSpot = strikes.length > 0 
          ? strikes[Math.floor(strikes.length / 2)] 
          : 3.0;
        setSpotPrice(estimatedSpot);
        
        // Step 2: Fetch accurate Greeks from backend API
        let greeksMap = new Map<string, any>();
        try {
          const greeksResponse = await fetch('http://localhost:8000/api/greeks/chain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              trade_date: currentDate,
              spot_price: estimatedSpot,
              use_market_iv: true,  // Use market IV when available
              dataset_id: datasetId
            })
          });
          
          if (greeksResponse.ok) {
            const greeksData = await greeksResponse.json();
            // Build a map of id -> greeks for quick lookup
            (greeksData.options || []).forEach((opt: any) => {
              greeksMap.set(opt.id, opt);
            });
            console.log(`Loaded ${greeksMap.size} options with accurate Greeks`);
          }
        } catch (greeksErr) {
          console.warn('Greeks API not available, using fallback:', greeksErr);
        }
        
        // Step 3: Group by expiry date and merge Greeks
        const expiryMap = new Map<string, { calls: OptionContract[], puts: OptionContract[], strikes: Set<number> }>();
        const today = new Date(currentDate);
        
        assets.forEach((asset: any) => {
          const expiry = asset.expiry;
          if (!expiryMap.has(expiry)) {
            expiryMap.set(expiry, { calls: [], puts: [], strikes: new Set() });
          }
          
          const group = expiryMap.get(expiry)!;
          group.strikes.add(asset.strike);
          
          const expiryDate = new Date(expiry);
          const dte = Math.ceil((expiryDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
          
          // Look up Greeks from backend response
          const greeksInfo = greeksMap.get(asset.id);
          
          // Use backend Greeks if available, otherwise use asset data or fallback
          const iv = greeksInfo?.iv ?? asset.iv ?? 0.2;
          const delta = greeksInfo?.delta ?? (asset.type === 'call' ? 0.5 : -0.5);
          const gamma = greeksInfo?.gamma ?? 0.05;
          const theta = greeksInfo?.theta ?? -0.01;
          const vega = greeksInfo?.vega ?? 0.1;
          const vanna = greeksInfo?.vanna ?? 0;
          const volga = greeksInfo?.volga ?? 0;
          
          const contract: OptionContract = {
            id: asset.id,
            type: asset.type,
            strike: asset.strike,
            expiry: expiry,
            dte: dte,
            bid: asset.close * 0.98,
            ask: asset.close * 1.02,
            last: asset.close,
            change: asset.change || 0,
            changePercent: asset.change_percent || 0,
            volume: asset.volume || 0,
            openInterest: Math.floor(Math.random() * 10000),
            iv: iv,
            delta: delta,
            gamma: gamma,
            theta: theta,
            vega: vega,
            vanna: vanna,
            volga: volga,
          };
          
          if (asset.type === 'call') {
            group.calls.push(contract);
          } else {
            group.puts.push(contract);
          }
        });
        
        // Convert to ExpiryGroup array
        const groups: ExpiryGroup[] = Array.from(expiryMap.entries())
          .map(([expiry, group]) => {
            const strikes = Array.from(group.strikes).sort((a, b) => a - b);
            const expiryDate = new Date(expiry);
            const dte = Math.ceil((expiryDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
            
            const alignedCalls = strikes.map(k => group.calls.find(c => c.strike === k) || createEmptyContract(k, 'call', expiry, dte));
            const alignedPuts = strikes.map(k => group.puts.find(p => p.strike === k) || createEmptyContract(k, 'put', expiry, dte));
            
            return { expiry, dte, strikes, calls: alignedCalls, puts: alignedPuts };
          })
          .filter(g => g.dte > 0)
          .sort((a, b) => a.dte - b.dte);
        
        setExpiryGroups(groups);
        if (groups.length > 0 && !selectedExpiry) {
          setSelectedExpiry(groups[0].expiry);
        }
      } catch (err) {
        console.error('Failed to fetch option chain:', err);
        setError(err instanceof Error ? err.message : '加载期权链失败');
      } finally {
        setLoading(false);
      }
    };
    
    fetchOptionChain();
  }, [currentDate, datasetId]);


  const createEmptyContract = (strike: number, type: 'call' | 'put', expiry: string, dte: number): OptionContract => ({
    id: `EMPTY-${strike}-${type}`, type, strike, expiry, dte,
    bid: 0, ask: 0, last: 0, change: 0, changePercent: 0,
    volume: 0, openInterest: 0, iv: 0.2,
    delta: 0, gamma: 0, theta: 0, vega: 0, vanna: 0, volga: 0,
  });

  const currentGroup = useMemo(() => {
    return expiryGroups.find(g => g.expiry === selectedExpiry);
  }, [expiryGroups, selectedExpiry]);

  // Apply filters to current group
  const filteredStrikes = useMemo(() => {
    if (!currentGroup) return [];
    
    return currentGroup.strikes.filter((strike, idx) => {
      const call = currentGroup.calls[idx];
      const put = currentGroup.puts[idx];
      
      // Moneyness filter
      const isITMCall = strike < spotPrice;
      const isITMPut = strike > spotPrice;
      const isATM = Math.abs(strike - spotPrice) < 0.05;
      
      if (filters.moneyness === 'itm' && !isITMCall && !isITMPut) return false;
      if (filters.moneyness === 'otm' && (isITMCall || isITMPut)) return false;
      if (filters.moneyness === 'atm' && !isATM) return false;
      
      // Strike search
      if (filters.strikeSearch && !strike.toString().includes(filters.strikeSearch)) return false;
      
      // Delta range (use absolute value for comparison)
      const maxDelta = Math.max(Math.abs(call.delta), Math.abs(put.delta));
      if (maxDelta < filters.deltaMin || maxDelta > filters.deltaMax) return false;
      
      // IV range
      const maxIV = Math.max(call.iv, put.iv) * 100;
      if (maxIV < filters.ivMin || maxIV > filters.ivMax) return false;
      
      // Volume filter
      const totalVolume = call.volume + put.volume;
      if (totalVolume < filters.volumeMin) return false;
      
      return true;
    });
  }, [currentGroup, filters, spotPrice]);

  // Get max IV for normalization
  const maxIV = useMemo(() => {
    if (!currentGroup) return 1;
    const allIVs = [...currentGroup.calls.map(c => c.iv), ...currentGroup.puts.map(p => p.iv)];
    return Math.max(...allIVs, 0.5);
  }, [currentGroup]);

  // Aggregate Greeks for current expiry
  const aggregatedGreeks = useMemo(() => {
    if (!currentGroup) return null;
    const allOptions = [...currentGroup.calls, ...currentGroup.puts];
    return {
      totalDelta: allOptions.reduce((sum, o) => sum + o.delta, 0),
      totalGamma: allOptions.reduce((sum, o) => sum + o.gamma, 0),
      totalVega: allOptions.reduce((sum, o) => sum + o.vega, 0),
      totalTheta: allOptions.reduce((sum, o) => sum + o.theta, 0),
    };
  }, [currentGroup]);

  const formatPrice = (p: number) => p.toFixed(4);
  const formatPercent = (p: number) => `${p >= 0 ? '+' : ''}${p.toFixed(2)}%`;
  const formatGreek = (g: number) => g.toFixed(4);
  
  // Volume heatmap color
  const getVolumeColor = (volume: number, maxVol: number) => {
    const intensity = Math.min(volume / (maxVol || 1), 1);
    return `rgba(139, 92, 246, ${intensity * 0.3})`;
  };

  // Get max volume for heatmap
  const maxVolume = useMemo(() => {
    if (!currentGroup) return 1;
    const allVolumes = [...currentGroup.calls.map(c => c.volume), ...currentGroup.puts.map(p => p.volume)];
    return Math.max(...allVolumes, 1);
  }, [currentGroup]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-[var(--accent-primary)]" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-96 space-y-4">
        <div className="text-red-500 text-lg">⚠️ 加载失败</div>
        <div className="text-gray-400 text-sm">{error}</div>
        <button 
          onClick={() => setCurrentDate(currentDate)}
          className="px-4 py-2 bg-[var(--accent-primary)] text-white rounded-lg hover:opacity-90"
        >
          重试
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Date Selector */}
      <div className="glass-card p-3 flex items-center gap-4">
        <Calendar className="w-5 h-5 text-[var(--accent-primary)]" />
        <span className="text-sm text-[var(--text-muted)]">交易日期:</span>
        <select
          value={currentDate}
          onChange={(e) => setCurrentDate(e.target.value)}
          className="bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded px-3 py-1.5 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)]"
        >
          {availableDates.map(date => (
            <option key={date} value={date}>{date}</option>
          ))}
        </select>
      </div>

      {/* Filter Bar */}
      <FilterBar filters={filters} onFilterChange={setFilters} spotPrice={spotPrice} />

      {/* Header Controls */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Grid3X3 className="w-5 h-5 text-[var(--accent-primary)]" />
              <h3 className="section-title">期权链 (集成Greeks)</h3>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-[var(--text-muted)]">标的:</span>
              <span className="font-mono font-bold text-[var(--accent-success)]">¥{spotPrice.toFixed(2)}</span>
            </div>
            <div className="text-xs text-[var(--text-muted)]">
              显示 {filteredStrikes.length} / {currentGroup?.strikes.length || 0} 行权价
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <ColumnSettings visibility={columnVisibility} onVisibilityChange={setColumnVisibility} />
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input 
                type="checkbox" 
                checked={showAdvancedGreeks} 
                onChange={(e) => setShowAdvancedGreeks(e.target.checked)}
                className="accent-[var(--accent-primary)]"
              />
              <span className="text-[var(--text-secondary)]">二阶Greeks</span>
            </label>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input 
                type="checkbox" 
                checked={highlightATM} 
                onChange={(e) => setHighlightATM(e.target.checked)}
                className="accent-[var(--accent-primary)]"
              />
              <span className="text-[var(--text-secondary)]">高亮ATM</span>
            </label>
          </div>
        </div>

        {/* Expiry Tabs */}
        <div className="flex gap-2 mt-4 overflow-x-auto pb-2">
          {expiryGroups.map((group) => (
            <button
              key={group.expiry}
              onClick={() => setSelectedExpiry(group.expiry)}
              className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all ${
                selectedExpiry === group.expiry
                  ? 'bg-[var(--accent-primary)] text-white'
                  : 'bg-[var(--bg-card)] text-[var(--text-secondary)] hover:bg-[var(--bg-card-hover)]'
              }`}
            >
              <div className="flex items-center gap-2">
                <Calendar className="w-4 h-4" />
                <span>{group.expiry}</span>
                <span className="text-xs opacity-70">({group.dte}天)</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Aggregated Greeks Summary */}
      {aggregatedGreeks && (
        <div className="grid grid-cols-4 gap-4">
          <div className="glass-card-elevated p-3 text-center">
            <GreekTooltip symbol="Δ">
              <div className="label-mono text-xs mb-1 cursor-help flex items-center justify-center gap-1">
                Σ Delta <Info className="w-3 h-3 opacity-50" />
              </div>
            </GreekTooltip>
            <div className={`text-lg font-mono font-bold ${aggregatedGreeks.totalDelta >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'}`}>
              {aggregatedGreeks.totalDelta >= 0 ? '+' : ''}{aggregatedGreeks.totalDelta.toFixed(2)}
            </div>
          </div>
          <div className="glass-card-elevated p-3 text-center">
            <GreekTooltip symbol="Γ">
              <div className="label-mono text-xs mb-1 cursor-help flex items-center justify-center gap-1">
                Σ Gamma <Info className="w-3 h-3 opacity-50" />
              </div>
            </GreekTooltip>
            <div className="text-lg font-mono font-bold text-[var(--accent-warning)]">
              {aggregatedGreeks.totalGamma.toFixed(4)}
            </div>
          </div>
          <div className="glass-card-elevated p-3 text-center">
            <GreekTooltip symbol="ν">
              <div className="label-mono text-xs mb-1 cursor-help flex items-center justify-center gap-1">
                Σ Vega <Info className="w-3 h-3 opacity-50" />
              </div>
            </GreekTooltip>
            <div className="text-lg font-mono font-bold text-[var(--accent-primary)]">
              {aggregatedGreeks.totalVega.toFixed(2)}
            </div>
          </div>
          <div className="glass-card-elevated p-3 text-center">
            <GreekTooltip symbol="Θ">
              <div className="label-mono text-xs mb-1 cursor-help flex items-center justify-center gap-1">
                Σ Theta/日 <Info className="w-3 h-3 opacity-50" />
              </div>
            </GreekTooltip>
            <div className="text-lg font-mono font-bold text-[var(--accent-danger)]">
              {aggregatedGreeks.totalTheta.toFixed(2)}
            </div>
          </div>
        </div>
      )}

      {/* Main Content: Table + Radar Chart */}
      <div className="grid grid-cols-4 gap-4">
        {/* T-Quote Table */}
        {currentGroup && (
          <div className="col-span-3 glass-card overflow-hidden" ref={tableRef}>
            <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 z-10 bg-[var(--bg-card)]">
                  <tr className="border-b border-[var(--border-primary)]">
                    <th colSpan={
                      (showAdvancedGreeks && columnVisibility.vanna ? 1 : 0) + 
                      (showAdvancedGreeks && columnVisibility.volga ? 1 : 0) +
                      (columnVisibility.delta ? 1 : 0) +
                      (columnVisibility.gamma ? 1 : 0) +
                      (columnVisibility.theta ? 1 : 0) +
                      (columnVisibility.vega ? 1 : 0) +
                      (columnVisibility.iv ? 1 : 0) +
                      1 // price column always visible
                    } className="bg-[var(--accent-success)]/10 text-[var(--accent-success)] py-3 text-center font-bold">
                      📈 认购期权 (Call)
                    </th>
                    <th className="bg-[var(--bg-elevated)] text-[var(--text-primary)] py-3 px-4 text-center font-bold">
                      行权价
                    </th>
                    <th colSpan={
                      (showAdvancedGreeks && columnVisibility.vanna ? 1 : 0) + 
                      (showAdvancedGreeks && columnVisibility.volga ? 1 : 0) +
                      (columnVisibility.delta ? 1 : 0) +
                      (columnVisibility.gamma ? 1 : 0) +
                      (columnVisibility.theta ? 1 : 0) +
                      (columnVisibility.vega ? 1 : 0) +
                      (columnVisibility.iv ? 1 : 0) +
                      1 // price column always visible
                    } className="bg-[var(--accent-danger)]/10 text-[var(--accent-danger)] py-3 text-center font-bold">
                      📉 认沽期权 (Put)
                    </th>
                  </tr>
                  <tr className="border-b border-[var(--border-primary)] text-xs text-[var(--text-muted)]">
                    {/* Call columns */}
                    {showAdvancedGreeks && columnVisibility.vanna && (
                      <th className="py-2 px-2 text-right">
                        <GreekTooltip symbol="Vanna"><span className="cursor-help">Vanna</span></GreekTooltip>
                      </th>
                    )}
                    {showAdvancedGreeks && columnVisibility.volga && (
                      <th className="py-2 px-2 text-right">
                        <GreekTooltip symbol="Volga"><span className="cursor-help">Volga</span></GreekTooltip>
                      </th>
                    )}
                    {columnVisibility.delta && <th className="py-2 px-2 text-right"><GreekTooltip symbol="Δ"><span className="cursor-help">Δ</span></GreekTooltip></th>}
                    {columnVisibility.gamma && <th className="py-2 px-2 text-right"><GreekTooltip symbol="Γ"><span className="cursor-help">Γ</span></GreekTooltip></th>}
                    {columnVisibility.theta && <th className="py-2 px-2 text-right"><GreekTooltip symbol="Θ"><span className="cursor-help">Θ</span></GreekTooltip></th>}
                    {columnVisibility.vega && <th className="py-2 px-2 text-right"><GreekTooltip symbol="ν"><span className="cursor-help">ν</span></GreekTooltip></th>}
                    {columnVisibility.iv && <th className="py-2 px-2 text-right"><GreekTooltip symbol="IV"><span className="cursor-help">IV</span></GreekTooltip></th>}
                    <th className="py-2 px-2 text-right bg-[var(--accent-success)]/5">卖价</th>
                    
                    <th className="py-2 px-4 text-center bg-[var(--bg-elevated)]">K</th>
                    
                    <th className="py-2 px-2 text-left bg-[var(--accent-danger)]/5">买价</th>
                    {columnVisibility.iv && <th className="py-2 px-2 text-left"><GreekTooltip symbol="IV"><span className="cursor-help">IV</span></GreekTooltip></th>}
                    {columnVisibility.delta && <th className="py-2 px-2 text-left"><GreekTooltip symbol="Δ"><span className="cursor-help">Δ</span></GreekTooltip></th>}
                    {columnVisibility.gamma && <th className="py-2 px-2 text-left"><GreekTooltip symbol="Γ"><span className="cursor-help">Γ</span></GreekTooltip></th>}
                    {columnVisibility.theta && <th className="py-2 px-2 text-left"><GreekTooltip symbol="Θ"><span className="cursor-help">Θ</span></GreekTooltip></th>}
                    {columnVisibility.vega && <th className="py-2 px-2 text-left"><GreekTooltip symbol="ν"><span className="cursor-help">ν</span></GreekTooltip></th>}
                    {showAdvancedGreeks && columnVisibility.vanna && (
                      <th className="py-2 px-2 text-left">
                        <GreekTooltip symbol="Vanna"><span className="cursor-help">Vanna</span></GreekTooltip>
                      </th>
                    )}
                    {showAdvancedGreeks && columnVisibility.volga && (
                      <th className="py-2 px-2 text-left">
                        <GreekTooltip symbol="Volga"><span className="cursor-help">Volga</span></GreekTooltip>
                      </th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {filteredStrikes.map((strike, idx) => {
                    const originalIdx = currentGroup.strikes.indexOf(strike);
                    const call = currentGroup.calls[originalIdx];
                    const put = currentGroup.puts[originalIdx];
                    const isATM = highlightATM && Math.abs(strike - spotPrice) < 0.05;
                    const isITMCall = strike < spotPrice;
                    const isITMPut = strike > spotPrice;
                    const isSelected = selectedRowIndex === idx;
                    
                    return (
                      <tr 
                        key={`${strike}-${idx}`}
                        onClick={() => {
                          setSelectedRowIndex(idx);
                          setSelectedOption(call);
                        }}
                        className={`border-b border-[var(--border-primary)]/30 hover:bg-[var(--bg-card-hover)] transition-colors cursor-pointer ${
                          isATM ? 'bg-[var(--accent-primary)]/10' : ''
                        } ${isSelected ? 'ring-1 ring-[var(--accent-primary)]' : ''}`}
                        style={{ backgroundColor: isSelected ? 'rgba(139, 92, 246, 0.15)' : undefined }}
                      >
                        {/* Call side */}
                        {showAdvancedGreeks && columnVisibility.vanna && (
                          <td className={`py-2 px-2 text-right font-mono text-xs ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                            {formatGreek(call.vanna)}
                          </td>
                        )}
                        {showAdvancedGreeks && columnVisibility.volga && (
                          <td className={`py-2 px-2 text-right font-mono text-xs ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                            {formatGreek(call.volga)}
                          </td>
                        )}
                        {columnVisibility.delta && (
                          <td className={`py-2 px-2 text-right font-mono text-xs ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                            {formatGreek(call.delta)}
                          </td>
                        )}
                        {columnVisibility.gamma && (
                          <td className={`py-2 px-2 text-right font-mono text-xs ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                            {formatGreek(call.gamma)}
                          </td>
                        )}
                        {columnVisibility.theta && (
                          <td className={`py-2 px-2 text-right font-mono text-xs text-[var(--accent-danger)] ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                            {formatGreek(call.theta)}
                          </td>
                        )}
                        {columnVisibility.vega && (
                          <td className={`py-2 px-2 text-right font-mono text-xs ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                            {formatGreek(call.vega)}
                          </td>
                        )}
                        {columnVisibility.iv && (
                          <td className={`py-2 px-2 text-right ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                            <MiniIVBar value={call.iv} max={maxIV} />
                          </td>
                        )}
                        <td 
                          className={`py-2 px-2 text-right font-mono font-bold text-[var(--accent-success)] ${isITMCall ? 'bg-[var(--accent-success)]/10' : 'bg-[var(--accent-success)]/5'}`}
                          style={{ backgroundColor: getVolumeColor(call.volume, maxVolume) }}
                        >
                          {formatPrice(call.ask)}
                        </td>
                        
                        {/* Strike column */}
                        <td className={`py-2 px-4 text-center font-mono font-bold text-lg ${
                          isATM 
                            ? 'bg-[var(--accent-primary)] text-white' 
                            : 'bg-[var(--bg-elevated)] text-[var(--text-primary)]'
                        }`}>
                          {strike.toFixed(2)}
                        </td>
                        
                        {/* Put side */}
                        <td 
                          className={`py-2 px-2 text-left font-mono font-bold text-[var(--accent-danger)] ${isITMPut ? 'bg-[var(--accent-danger)]/10' : 'bg-[var(--accent-danger)]/5'}`}
                          style={{ backgroundColor: getVolumeColor(put.volume, maxVolume) }}
                        >
                          {formatPrice(put.bid)}
                        </td>
                        {columnVisibility.iv && (
                          <td className={`py-2 px-2 text-left ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                            <MiniIVBar value={put.iv} max={maxIV} />
                          </td>
                        )}
                        {columnVisibility.delta && (
                          <td className={`py-2 px-2 text-left font-mono text-xs ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                            {formatGreek(put.delta)}
                          </td>
                        )}
                        {columnVisibility.gamma && (
                          <td className={`py-2 px-2 text-left font-mono text-xs ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                            {formatGreek(put.gamma)}
                          </td>
                        )}
                        {columnVisibility.theta && (
                          <td className={`py-2 px-2 text-left font-mono text-xs text-[var(--accent-danger)] ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                            {formatGreek(put.theta)}
                          </td>
                        )}
                        {columnVisibility.vega && (
                          <td className={`py-2 px-2 text-left font-mono text-xs ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                            {formatGreek(put.vega)}
                          </td>
                        )}
                        {showAdvancedGreeks && columnVisibility.vanna && (
                          <td className={`py-2 px-2 text-left font-mono text-xs ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                            {formatGreek(put.vanna)}
                          </td>
                        )}
                        {showAdvancedGreeks && columnVisibility.volga && (
                          <td className={`py-2 px-2 text-left font-mono text-xs ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                            {formatGreek(put.volga)}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Right Panel: Radar Chart + Actions */}
        <div className="space-y-4">
          {/* Greeks Radar Chart */}
          <GreeksRadarChart option={selectedOption} />
          
          {/* Selected Option Actions */}
          {selectedOption && (
            <div className="glass-card-elevated p-3 space-y-3">
              <div className="text-sm font-medium flex items-center gap-2">
                <Zap className="w-4 h-4 text-[var(--accent-warning)]" />
                操作
              </div>
              <div className="space-y-2">
                <button className="w-full px-3 py-2 text-xs bg-[var(--accent-primary)]/20 hover:bg-[var(--accent-primary)]/30 text-[var(--accent-primary)] rounded-lg transition-colors flex items-center justify-center gap-2">
                  <Target className="w-3 h-3" />
                  添加到模拟组合
                </button>
                <button className="w-full px-3 py-2 text-xs bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] rounded-lg transition-colors flex items-center justify-center gap-2">
                  <BarChart3 className="w-3 h-3" />
                  查看历史K线
                </button>
                <button className="w-full px-3 py-2 text-xs bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] rounded-lg transition-colors flex items-center justify-center gap-2">
                  <ArrowUpDown className="w-3 h-3" />
                  PCR分析
                </button>
              </div>
            </div>
          )}
          
          {/* Keyboard Hint */}
          <div className="glass-card p-3 text-xs text-[var(--text-muted)]">
            <div className="flex items-center gap-2 mb-1">
              <Info className="w-3 h-3" />
              <span className="font-medium">键盘操作</span>
            </div>
            <div className="space-y-1">
              <div>↑/↓ 切换行</div>
              <div>Enter 选择合约</div>
            </div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-[var(--accent-success)]/20"></div>
              <span>价内认购 (ITM Call)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-[var(--accent-danger)]/20"></div>
              <span>价内认沽 (ITM Put)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-[var(--accent-primary)]"></div>
              <span>平值 (ATM)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-8 rounded" style={{ background: 'linear-gradient(to right, transparent, rgba(139, 92, 246, 0.3))' }}></div>
              <span>成交量热力图</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span>悬停列标题查看 Greeks 说明</span>
          </div>
        </div>
      </div>
    </div>
  );
}
