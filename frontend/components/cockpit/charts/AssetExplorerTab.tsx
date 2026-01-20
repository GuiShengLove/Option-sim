"use client";

import { useState, useEffect, useMemo } from 'react';
import { 
  TrendingUp, TrendingDown, Activity, Clock, Target, Calendar, RefreshCw, 
  BarChart2, Filter, Search, ChevronDown, ChevronUp, ArrowUpDown, 
  Layers, PieChart, LineChart, Settings, Info, Maximize2, Minimize2
} from 'lucide-react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface Asset {
  id: string;
  type: 'call' | 'put';
  strike: number;
  expiry: string;
  close: number;
  change: number;
  changePercent: number;
  iv: number;
  volume: number;
  delta?: number;
  gamma?: number;
  theta?: number;
  vega?: number;
  ohlc?: { date: string; open: number; high: number; low: number; close: number; volume?: number }[];
}

interface EtfData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  avg_iv: number;
}

interface AssetExplorerTabProps {
  selectedDate?: string;
  onDateChange?: (date: string) => void;
  dateRange?: { start: string; end: string };
  datasetId?: string;
}

// Filter state interface
interface FilterState {
  expiry: string;
  strikeMin: number;
  strikeMax: number;
  type: 'all' | 'call' | 'put';
  moneyness: 'all' | 'itm' | 'atm' | 'otm';
  ivMin: number;
  ivMax: number;
  volumeMin: number;
}

// Sort state
type SortField = 'strike' | 'expiry' | 'close' | 'changePercent' | 'iv' | 'volume';
type SortDirection = 'asc' | 'desc';

// IV Smile Curve Component
const IVSmileCurve = ({ assets, spotPrice }: { assets: Asset[]; spotPrice: number }) => {
  const callData = assets.filter(a => a.type === 'call').sort((a, b) => a.strike - b.strike);
  const putData = assets.filter(a => a.type === 'put').sort((a, b) => a.strike - b.strike);
  
  if (callData.length === 0 && putData.length === 0) return null;
  
  const traces = [
    {
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: 'Call IV',
      x: callData.map(a => a.strike),
      y: callData.map(a => a.iv * 100),
      line: { color: '#10b981', width: 2 },
      marker: { size: 6 },
    },
    {
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: 'Put IV',
      x: putData.map(a => a.strike),
      y: putData.map(a => a.iv * 100),
      line: { color: '#ef4444', width: 2 },
      marker: { size: 6 },
    },
  ];
  
  // Add ATM line
  if (spotPrice > 0) {
    traces.push({
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'ATM',
      x: [spotPrice, spotPrice],
      y: [0, 100],
      line: { color: '#8b5cf6', width: 1, dash: 'dash' } as any,
      marker: { size: 0 },
    });
  }
  
  return (
    <div className="glass-card-elevated p-3">
      <div className="flex items-center gap-2 mb-2">
        <LineChart className="w-4 h-4 text-[var(--accent-primary)]" />
        <span className="text-sm font-medium">IV 微笑曲线</span>
      </div>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 180,
          margin: { l: 40, r: 20, t: 10, b: 30 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: { title: { text: '行权价', font: { size: 10, color: '#6b7280' } }, gridcolor: '#374151', tickfont: { size: 9, color: '#6b7280' } },
          yaxis: { title: { text: 'IV (%)', font: { size: 10, color: '#6b7280' } }, gridcolor: '#374151', tickfont: { size: 9, color: '#6b7280' } },
          legend: { orientation: 'h', y: 1.1, font: { size: 9, color: '#9ca3af' } },
          showlegend: true,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '180px' }}
        useResizeHandler={true}
      />
    </div>
  );
};

// PCR Indicator Component
const PCRIndicator = ({ assets }: { assets: Asset[] }) => {
  const calls = assets.filter(a => a.type === 'call');
  const puts = assets.filter(a => a.type === 'put');
  
  const callVolume = calls.reduce((sum, a) => sum + a.volume, 0);
  const putVolume = puts.reduce((sum, a) => sum + a.volume, 0);
  const pcr = callVolume > 0 ? putVolume / callVolume : 0;
  
  const callOI = calls.length;
  const putOI = puts.length;
  const pcrOI = callOI > 0 ? putOI / callOI : 0;
  
  const getSentiment = (pcr: number) => {
    if (pcr > 1.2) return { text: '看涨', color: 'var(--accent-success)' };
    if (pcr < 0.8) return { text: '看跌', color: 'var(--accent-danger)' };
    return { text: '中性', color: 'var(--accent-warning)' };
  };
  
  const sentiment = getSentiment(pcr);
  
  return (
    <div className="glass-card-elevated p-3">
      <div className="flex items-center gap-2 mb-3">
        <PieChart className="w-4 h-4 text-[var(--accent-primary)]" />
        <span className="text-sm font-medium">PCR 指标</span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="text-center p-2 bg-[var(--bg-elevated)] rounded-lg">
          <div className="text-xs text-[var(--text-muted)] mb-1">成交量 PCR</div>
          <div className="text-lg font-mono font-bold" style={{ color: sentiment.color }}>
            {pcr.toFixed(2)}
          </div>
        </div>
        <div className="text-center p-2 bg-[var(--bg-elevated)] rounded-lg">
          <div className="text-xs text-[var(--text-muted)] mb-1">市场情绪</div>
          <div className="text-lg font-bold" style={{ color: sentiment.color }}>
            {sentiment.text}
          </div>
        </div>
      </div>
      <div className="mt-3 text-xs text-[var(--text-muted)]">
        <div className="flex justify-between">
          <span>Call成交量:</span>
          <span className="text-[var(--accent-success)]">{callVolume.toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span>Put成交量:</span>
          <span className="text-[var(--accent-danger)]">{putVolume.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
};

// Contract Hover Card
const ContractHoverCard = ({ asset, show, position }: { asset: Asset | null; show: boolean; position: { x: number; y: number } }) => {
  if (!show || !asset) return null;
  
  return (
    <div 
      className="fixed z-50 p-4 bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded-xl shadow-2xl min-w-[240px]"
      style={{ left: position.x + 10, top: position.y + 10 }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span className={`px-2 py-0.5 rounded text-xs font-bold ${
          asset.type === 'call' 
            ? 'bg-[var(--accent-success)]/20 text-[var(--accent-success)]' 
            : 'bg-[var(--accent-danger)]/20 text-[var(--accent-danger)]'
        }`}>
          {asset.type === 'call' ? '认购' : '认沽'}
        </span>
        <span className="font-mono font-bold">K={asset.strike}</span>
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-[var(--text-muted)]">当前价格</span>
          <span className="font-mono">¥{asset.close.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[var(--text-muted)]">涨跌幅</span>
          <span className={`font-mono ${asset.changePercent >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'}`}>
            {asset.changePercent >= 0 ? '+' : ''}{asset.changePercent.toFixed(2)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-[var(--text-muted)]">隐含波动率</span>
          <span className="font-mono text-[var(--accent-primary)]">{(asset.iv * 100).toFixed(1)}%</span>
        </div>
        {asset.delta !== undefined && (
          <div className="pt-2 border-t border-[var(--border-primary)]">
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div><span className="text-[var(--text-muted)]">Δ:</span> <span className="font-mono">{asset.delta.toFixed(4)}</span></div>
              <div><span className="text-[var(--text-muted)]">Γ:</span> <span className="font-mono">{asset.gamma?.toFixed(4)}</span></div>
              <div><span className="text-[var(--text-muted)]">Θ:</span> <span className="font-mono">{asset.theta?.toFixed(4)}</span></div>
              <div><span className="text-[var(--text-muted)]">ν:</span> <span className="font-mono">{asset.vega?.toFixed(4)}</span></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default function AssetExplorerTab({ selectedDate, onDateChange, dateRange, datasetId = '510050_SH' }: AssetExplorerTabProps) {
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [etfCandles, setEtfCandles] = useState<EtfData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [currentDate, setCurrentDate] = useState(selectedDate || '');
  const [viewMode, setViewMode] = useState<'etf' | 'options' | 'split'>('etf');
  const [showFilters, setShowFilters] = useState(false);
  const [spotPrice, setSpotPrice] = useState(3.0);
  
  // Technical indicators
  const [showMA5, setShowMA5] = useState(true);
  const [showMA10, setShowMA10] = useState(true);
  const [showMA20, setShowMA20] = useState(false);
  const [showBoll, setShowBoll] = useState(false);
  
  // Filters
  const [filters, setFilters] = useState<FilterState>({
    expiry: 'all',
    strikeMin: 0,
    strikeMax: 10,
    type: 'all',
    moneyness: 'all',
    ivMin: 0,
    ivMax: 200,
    volumeMin: 0,
  });
  
  // Sorting
  const [sortField, setSortField] = useState<SortField>('strike');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  
  // Hover card
  const [hoverAsset, setHoverAsset] = useState<Asset | null>(null);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const [showHoverCard, setShowHoverCard] = useState(false);

  // Sync with external selectedDate prop
  useEffect(() => {
    if (selectedDate && selectedDate !== currentDate) {
      setCurrentDate(selectedDate);
    }
  }, [selectedDate]);

  // Fetch available dates on mount and filter by dateRange
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

  // Fetch ETF data and assets when date changes
  useEffect(() => {
    if (!currentDate) return;

    const loadData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Fetch ETF candle data
        const etfResponse = await fetch(`http://localhost:8000/api/data/etf-candle?dataset_id=${datasetId}`);
        if (etfResponse.ok) {
          const etfData = await etfResponse.json();
          setEtfCandles(etfData.candles || []);
          
          // Get latest spot price
          const candles = etfData.candles || [];
          if (candles.length > 0) {
            setSpotPrice(candles[candles.length - 1].close);
          }
        }
        
        // Fetch option assets with real Greeks
        const assetsResponse = await fetch(`http://localhost:8000/api/data/assets?date=${currentDate}&limit=100&dataset_id=${datasetId}`);
        
        if (!assetsResponse.ok) {
          throw new Error(`HTTP ${assetsResponse.status}: ${assetsResponse.statusText}`);
        }
        
        const assetsData = await assetsResponse.json();
        
        // Get Greeks for better data
        const strikes = (assetsData.assets || []).map((a: any) => a.strike);
        const estimatedSpot = strikes.length > 0 ? strikes[Math.floor(strikes.length / 2)] : 3.0;
        
        let greeksMap = new Map<string, any>();
        try {
          const greeksResponse = await fetch('http://localhost:8000/api/greeks/chain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              trade_date: currentDate,
              spot_price: estimatedSpot,
              use_market_iv: true,
              dataset_id: datasetId
            })
          });
          if (greeksResponse.ok) {
            const greeksData = await greeksResponse.json();
            (greeksData.options || []).forEach((opt: any) => {
              greeksMap.set(opt.id, opt);
            });
          }
        } catch (err) {
          console.warn('Greeks API not available');
        }
        
        // Transform API response to Asset format with real IV
        const transformedAssets: Asset[] = (assetsData.assets || []).map((asset: any) => {
          const greeksInfo = greeksMap.get(asset.id);
          return {
            id: asset.id,
            type: asset.type as 'call' | 'put',
            strike: asset.strike,
            expiry: asset.expiry,
            close: asset.close,
            change: asset.change || 0,
            changePercent: asset.change_percent || 0,
            iv: greeksInfo?.iv ?? asset.iv ?? 0.2,
            volume: asset.volume,
            delta: greeksInfo?.delta,
            gamma: greeksInfo?.gamma,
            theta: greeksInfo?.theta,
            vega: greeksInfo?.vega,
          };
        });
        
        setAssets(transformedAssets);
        if (transformedAssets.length > 0) {
          setSelectedAsset(transformedAssets[0]);
          fetchCandleData(transformedAssets[0].id);
        }
      } catch (err) {
        console.error('Failed to fetch data:', err);
        setError(err instanceof Error ? err.message : '加载数据失败');
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [currentDate, datasetId]);

  // Get unique expiry dates for filter
  const expiryDates = useMemo(() => {
    const dates = [...new Set(assets.map(a => a.expiry))];
    return dates.sort();
  }, [assets]);

  // Apply filters and sorting
  const filteredAssets = useMemo(() => {
    let result = assets.filter(asset => {
      // Expiry filter
      if (filters.expiry !== 'all' && asset.expiry !== filters.expiry) return false;
      
      // Strike range
      if (asset.strike < filters.strikeMin || asset.strike > filters.strikeMax) return false;
      
      // Type filter
      if (filters.type !== 'all' && asset.type !== filters.type) return false;
      
      // Moneyness filter
      const isITM = (asset.type === 'call' && asset.strike < spotPrice) || 
                    (asset.type === 'put' && asset.strike > spotPrice);
      const isATM = Math.abs(asset.strike - spotPrice) < 0.05;
      
      if (filters.moneyness === 'itm' && !isITM) return false;
      if (filters.moneyness === 'otm' && (isITM || isATM)) return false;
      if (filters.moneyness === 'atm' && !isATM) return false;
      
      // IV range
      const ivPercent = asset.iv * 100;
      if (ivPercent < filters.ivMin || ivPercent > filters.ivMax) return false;
      
      // Volume filter
      if (asset.volume < filters.volumeMin) return false;
      
      return true;
    });
    
    // Apply sorting
    result.sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'strike': comparison = a.strike - b.strike; break;
        case 'expiry': comparison = a.expiry.localeCompare(b.expiry); break;
        case 'close': comparison = a.close - b.close; break;
        case 'changePercent': comparison = a.changePercent - b.changePercent; break;
        case 'iv': comparison = a.iv - b.iv; break;
        case 'volume': comparison = a.volume - b.volume; break;
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });
    
    return result;
  }, [assets, filters, sortField, sortDirection, spotPrice]);

  // Fetch candle data for selected asset
  const fetchCandleData = async (assetId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/data/candle?asset_id=${assetId}&dataset_id=${datasetId}`);
      if (response.ok) {
        const data = await response.json();
        setAssets(prev => prev.map(asset => 
          asset.id === assetId 
            ? { ...asset, ohlc: data.candles }
            : asset
        ));
        setSelectedAsset(prev => prev?.id === assetId ? { ...prev, ohlc: data.candles } : prev);
      }
    } catch (err) {
      console.error('Failed to fetch candle data:', err);
    }
  };

  const handleAssetSelect = (asset: Asset) => {
    setSelectedAsset(asset);
    if (!asset.ohlc) {
      fetchCandleData(asset.id);
    }
  };

  const handleDateChange = (newDate: string) => {
    setCurrentDate(newDate);
    onDateChange?.(newDate);
  };

  // Calculate MA
  const calculateMA = (data: number[], period: number) => {
    const result: (number | null)[] = [];
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        result.push(null);
      } else {
        const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
        result.push(sum / period);
      }
    }
    return result;
  };

  // Calculate Bollinger Bands
  const calculateBoll = (data: number[], period: number = 20, mult: number = 2) => {
    const ma = calculateMA(data, period);
    const upper: (number | null)[] = [];
    const lower: (number | null)[] = [];
    
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        upper.push(null);
        lower.push(null);
      } else {
        const slice = data.slice(i - period + 1, i + 1);
        const mean = slice.reduce((a, b) => a + b, 0) / period;
        const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period;
        const std = Math.sqrt(variance);
        upper.push(mean + mult * std);
        lower.push(mean - mult * std);
      }
    }
    return { ma, upper, lower };
  };

  // Build ETF chart traces
  const etfChartData = useMemo(() => {
    if (etfCandles.length === 0) return [];
    
    const dates = etfCandles.map(c => c.date);
    const closes = etfCandles.map(c => c.close);
    const volumes = etfCandles.map(c => c.volume);
    
    const traces: any[] = [{
      type: 'candlestick',
      x: dates,
      open: etfCandles.map(c => c.open),
      high: etfCandles.map(c => c.high),
      low: etfCandles.map(c => c.low),
      close: closes,
      increasing: { line: { color: '#10b981' }, fillcolor: '#10b981' },
      decreasing: { line: { color: '#ef4444' }, fillcolor: '#ef4444' },
      yaxis: 'y',
    }];
    
    // Add MA lines
    if (showMA5) {
      traces.push({
        type: 'scatter', mode: 'lines', name: 'MA5',
        x: dates, y: calculateMA(closes, 5),
        line: { color: '#f59e0b', width: 1 }, yaxis: 'y',
      });
    }
    if (showMA10) {
      traces.push({
        type: 'scatter', mode: 'lines', name: 'MA10',
        x: dates, y: calculateMA(closes, 10),
        line: { color: '#3b82f6', width: 1 }, yaxis: 'y',
      });
    }
    if (showMA20) {
      traces.push({
        type: 'scatter', mode: 'lines', name: 'MA20',
        x: dates, y: calculateMA(closes, 20),
        line: { color: '#8b5cf6', width: 1 }, yaxis: 'y',
      });
    }
    
    // Add Bollinger Bands
    if (showBoll) {
      const boll = calculateBoll(closes, 20, 2);
      traces.push({
        type: 'scatter', mode: 'lines', name: 'BOLL中轨',
        x: dates, y: boll.ma, line: { color: '#ec4899', width: 1 }, yaxis: 'y',
      });
      traces.push({
        type: 'scatter', mode: 'lines', name: 'BOLL上轨',
        x: dates, y: boll.upper, line: { color: '#ec4899', width: 1, dash: 'dot' }, yaxis: 'y',
      });
      traces.push({
        type: 'scatter', mode: 'lines', name: 'BOLL下轨',
        x: dates, y: boll.lower, line: { color: '#ec4899', width: 1, dash: 'dot' }, yaxis: 'y', fill: 'tonexty', fillcolor: 'rgba(236, 72, 153, 0.1)',
      });
    }
    
    return traces;
  }, [etfCandles, showMA5, showMA10, showMA20, showBoll]);

  // Option candle data
  const optionChartData = useMemo(() => {
    if (!selectedAsset?.ohlc || selectedAsset.ohlc.length === 0) return [];
    
    const dates = selectedAsset.ohlc.map(c => c.date);
    const closes = selectedAsset.ohlc.map(c => c.close);
    
    const traces: any[] = [{
      type: 'candlestick',
      x: dates,
      open: selectedAsset.ohlc.map(c => c.open),
      high: selectedAsset.ohlc.map(c => c.high),
      low: selectedAsset.ohlc.map(c => c.low),
      close: closes,
      increasing: { line: { color: '#10b981' }, fillcolor: '#10b981' },
      decreasing: { line: { color: '#ef4444' }, fillcolor: '#ef4444' },
    }];
    
    if (showMA5) {
      traces.push({ type: 'scatter', mode: 'lines', name: 'MA5', x: dates, y: calculateMA(closes, 5), line: { color: '#f59e0b', width: 1 } });
    }
    if (showMA10) {
      traces.push({ type: 'scatter', mode: 'lines', name: 'MA10', x: dates, y: calculateMA(closes, 10), line: { color: '#3b82f6', width: 1 } });
    }
    
    return traces;
  }, [selectedAsset?.ohlc, showMA5, showMA10]);

  // 简化的K线图布局 - 支持滚轮缩放
  const candleLayout = useMemo(() => ({
    autosize: true,
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 10, r: 70, t: 30, b: 50 },
    xaxis: {
      type: 'category' as const,
      gridcolor: 'rgba(75, 85, 99, 0.3)',
      tickfont: { color: '#9ca3af', size: 9 },
      rangeslider: { visible: false },
      showgrid: true,
      zeroline: false,
      nticks: 12,
      fixedrange: false, // 允许缩放
    },
    yaxis: {
      title: { text: '价格', font: { size: 10, color: '#9ca3af' }, standoff: 5 },
      gridcolor: 'rgba(75, 85, 99, 0.3)',
      tickfont: { color: '#10b981', size: 9 },
      side: 'right' as const,
      showgrid: true,
      zeroline: false,
      tickformat: '.3f',
      fixedrange: false, // 允许缩放
    },
    showlegend: true,
    legend: { orientation: 'h' as const, y: 1.05, x: 0.5, xanchor: 'center', font: { size: 9, color: '#d1d5db' }, bgcolor: 'transparent' },
    hovermode: 'x unified' as const,
    dragmode: 'pan' as const, // 拖拽平移，滚轮缩放
  }), []);

  // Get latest ETF data for metrics display
  const latestEtf = etfCandles.length > 0 ? etfCandles[etfCandles.length - 1] : null;
  const prevEtf = etfCandles.length > 1 ? etfCandles[etfCandles.length - 2] : null;
  const etfChange = latestEtf && prevEtf ? ((latestEtf.close - prevEtf.close) / prevEtf.close) * 100 : 0;

  // Sort handler
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Hover handlers
  const handleRowHover = (asset: Asset, e: React.MouseEvent) => {
    setHoverAsset(asset);
    setHoverPosition({ x: e.clientX, y: e.clientY });
    setShowHoverCard(true);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[var(--accent-primary)]"></div>
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
      {/* Hover Card */}
      <ContractHoverCard asset={hoverAsset} show={showHoverCard} position={hoverPosition} />
      
      {/* Date Selector and View Mode Toggle */}
      <div className="glass-card p-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Calendar className="w-5 h-5 text-[var(--accent-primary)]" />
          <span className="text-sm text-[var(--text-muted)]">交易日期:</span>
          <select
            value={currentDate}
            onChange={(e) => handleDateChange(e.target.value)}
            className="bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded px-3 py-1.5 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)]"
          >
            {availableDates.map(date => (
              <option key={date} value={date}>{date}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-4">
          {/* View Mode Toggle */}
          <div className="flex gap-1 bg-[var(--bg-elevated)] p-1 rounded-lg">
            <button
              onClick={() => setViewMode('etf')}
              className={`px-3 py-1 text-xs rounded flex items-center gap-1 transition-all ${
                viewMode === 'etf' 
                  ? 'bg-[var(--accent-primary)] text-white' 
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              }`}
            >
              <BarChart2 className="w-3 h-3" />
              50ETF
            </button>
            <button
              onClick={() => setViewMode('options')}
              className={`px-3 py-1 text-xs rounded flex items-center gap-1 transition-all ${
                viewMode === 'options' 
                  ? 'bg-[var(--accent-primary)] text-white' 
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              }`}
            >
              <Activity className="w-3 h-3" />
              期权
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`px-3 py-1 text-xs rounded flex items-center gap-1 transition-all ${
                viewMode === 'split' 
                  ? 'bg-[var(--accent-primary)] text-white' 
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              }`}
            >
              <Layers className="w-3 h-3" />
              分屏
            </button>
          </div>
          
          {/* Technical Indicators */}
          <div className="flex items-center gap-2 text-xs">
            <span className="text-[var(--text-muted)]">指标:</span>
            <label className="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" checked={showMA5} onChange={(e) => setShowMA5(e.target.checked)} className="accent-[var(--accent-warning)] w-3 h-3" />
              <span className="text-[var(--accent-warning)]">MA5</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" checked={showMA10} onChange={(e) => setShowMA10(e.target.checked)} className="accent-[var(--accent-info)] w-3 h-3" />
              <span className="text-blue-400">MA10</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" checked={showMA20} onChange={(e) => setShowMA20(e.target.checked)} className="accent-[var(--accent-primary)] w-3 h-3" />
              <span className="text-[var(--accent-primary)]">MA20</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" checked={showBoll} onChange={(e) => setShowBoll(e.target.checked)} className="accent-pink-500 w-3 h-3" />
              <span className="text-pink-400">BOLL</span>
            </label>
          </div>
          
          <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
            <span>共 {filteredAssets.length} / {assets.length} 个合约</span>
            <button 
              onClick={() => setCurrentDate(currentDate)} 
              className="p-1 hover:bg-[var(--bg-card)] rounded transition-colors"
              title="刷新数据"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Chart Area */}
      <div className={`grid gap-4 ${viewMode === 'split' ? 'grid-cols-2' : 'grid-cols-3'}`}>
        {/* ETF Candlestick Chart */}
        {(viewMode === 'etf' || viewMode === 'split') && (
          <div className={`glass-card p-4 ${viewMode === 'etf' ? 'col-span-2' : ''}`} style={{ height: '420px' }}>
            <div className="section-header">
              <BarChart2 className="w-4 h-4 text-[var(--accent-primary)]" />
              <h3 className="section-title">标的资产 K线</h3>
            </div>
            <div style={{ height: 'calc(100% - 40px)' }}>
              {etfChartData.length > 0 ? (
                <Plot
                  data={etfChartData}
                  layout={candleLayout}
                  config={{ displayModeBar: false, responsive: true, scrollZoom: true }}
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler={true}
                />
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-[var(--text-muted)] space-y-2">
                  <BarChart2 className="w-10 h-10 opacity-30" />
                  <p className="text-sm">正在加载 50ETF K线数据...</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Option Candlestick Chart */}
        {(viewMode === 'options' || viewMode === 'split') && (
          <div className={`glass-card p-4 ${viewMode === 'options' ? 'col-span-2' : ''}`} style={{ height: '420px' }}>
            <div className="section-header">
              <Activity className="w-4 h-4 text-[var(--accent-secondary)]" />
              <h3 className="section-title">
                {selectedAsset ? `${selectedAsset.type.toUpperCase()} K=${selectedAsset.strike} @ ${selectedAsset.expiry}` : '选择期权查看K线'}
              </h3>
            </div>
            <div style={{ height: 'calc(100% - 40px)' }}>
              {optionChartData.length > 0 ? (
                <Plot
                  data={optionChartData}
                  layout={{
                    autosize: true,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    margin: { l: 10, r: 70, t: 30, b: 50 },
                    xaxis: {
                      type: 'category' as const,
                      gridcolor: 'rgba(75, 85, 99, 0.3)',
                      tickfont: { color: '#9ca3af', size: 9 },
                      rangeslider: { visible: false },
                      fixedrange: false,
                    },
                    yaxis: {
                      title: { text: '期权价格', font: { size: 10, color: '#9ca3af' } },
                      gridcolor: 'rgba(75, 85, 99, 0.3)',
                      tickfont: { color: '#10b981', size: 9 },
                      side: 'right' as const,
                      tickformat: '.4f',
                      fixedrange: false,
                    },
                    showlegend: true,
                    legend: { orientation: 'h' as const, y: 1.05, x: 0.5, xanchor: 'center', font: { size: 9, color: '#d1d5db' } },
                    hovermode: 'x unified' as const,
                    dragmode: 'pan' as const,
                  }}
                  config={{ displayModeBar: false, responsive: true, scrollZoom: true }}
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler={true}
                />
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-[var(--text-muted)] space-y-2">
                  <Activity className="w-10 h-10 opacity-30" />
                  <p className="text-sm">{selectedAsset ? '该合约无历史K线数据' : '请选择合约查看K线'}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Right Panel: Metrics + Analysis */}
        <div className="space-y-4">
          {/* Market Data Card */}
          {latestEtf && (
            <div className="glass-card p-4 space-y-3">
              <div className="section-header">
                <Target className="w-4 h-4 text-[var(--accent-secondary)]" />
                <h3 className="section-title">标的数据</h3>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center p-2 glass-card-elevated rounded-lg">
                  <span className="text-xs text-[var(--text-muted)]">最新价格</span>
                  <span className="font-mono font-bold text-[var(--text-primary)]">¥{latestEtf.close.toFixed(4)}</span>
                </div>
                <div className="flex justify-between items-center p-2 glass-card-elevated rounded-lg">
                  <span className="text-xs text-[var(--text-muted)]">日涨跌幅</span>
                  <span className={`font-mono font-bold flex items-center gap-1 ${etfChange >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'}`}>
                    {etfChange >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                    {etfChange >= 0 ? '+' : ''}{etfChange.toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-2 glass-card-elevated rounded-lg">
                  <span className="text-xs text-[var(--text-muted)]">平均IV</span>
                  <span className="font-mono font-bold text-[var(--accent-primary)]">{(latestEtf.avg_iv * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          )}
          
          {/* PCR Indicator */}
          <PCRIndicator assets={assets} />
          
          {/* IV Smile Curve */}
          <IVSmileCurve assets={filteredAssets} spotPrice={spotPrice} />
        </div>
      </div>

      {/* Advanced Filter Panel */}
      <div className="glass-card p-3">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center gap-2 text-sm"
          >
            <Filter className="w-4 h-4 text-[var(--accent-primary)]" />
            <span className="font-medium">高级筛选</span>
            {showFilters ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          
          {/* Quick Type Filter */}
          <div className="flex gap-1 bg-[var(--bg-elevated)] p-1 rounded-lg">
            {(['all', 'call', 'put'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setFilters({ ...filters, type: t })}
                className={`px-3 py-1 text-xs rounded transition-all ${
                  filters.type === t 
                    ? 'bg-[var(--accent-primary)] text-white' 
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                }`}
              >
                {t === 'all' ? '全部' : t === 'call' ? '认购' : '认沽'}
              </button>
            ))}
          </div>
        </div>
        
        {showFilters && (
          <div className="grid grid-cols-6 gap-4 mt-4 pt-4 border-t border-[var(--border-primary)]">
            {/* Expiry */}
            <div className="space-y-1">
              <label className="text-xs text-[var(--text-muted)]">到期日</label>
              <select
                value={filters.expiry}
                onChange={(e) => setFilters({ ...filters, expiry: e.target.value })}
                className="w-full px-2 py-1.5 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
              >
                <option value="all">全部</option>
                {expiryDates.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
            
            {/* Strike Range */}
            <div className="space-y-1">
              <label className="text-xs text-[var(--text-muted)]">行权价范围</label>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  step="0.1"
                  value={filters.strikeMin}
                  onChange={(e) => setFilters({ ...filters, strikeMin: parseFloat(e.target.value) || 0 })}
                  className="w-14 px-1 py-1 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
                />
                <span className="text-xs">~</span>
                <input
                  type="number"
                  step="0.1"
                  value={filters.strikeMax}
                  onChange={(e) => setFilters({ ...filters, strikeMax: parseFloat(e.target.value) || 10 })}
                  className="w-14 px-1 py-1 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
                />
              </div>
            </div>
            
            {/* Moneyness */}
            <div className="space-y-1">
              <label className="text-xs text-[var(--text-muted)]">价值状态</label>
              <select
                value={filters.moneyness}
                onChange={(e) => setFilters({ ...filters, moneyness: e.target.value as any })}
                className="w-full px-2 py-1.5 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
              >
                <option value="all">全部</option>
                <option value="itm">价内ITM</option>
                <option value="atm">平值ATM</option>
                <option value="otm">价外OTM</option>
              </select>
            </div>
            
            {/* IV Range */}
            <div className="space-y-1">
              <label className="text-xs text-[var(--text-muted)]">IV范围 (%)</label>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  step="5"
                  value={filters.ivMin}
                  onChange={(e) => setFilters({ ...filters, ivMin: parseFloat(e.target.value) || 0 })}
                  className="w-14 px-1 py-1 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
                />
                <span className="text-xs">~</span>
                <input
                  type="number"
                  step="5"
                  value={filters.ivMax}
                  onChange={(e) => setFilters({ ...filters, ivMax: parseFloat(e.target.value) || 200 })}
                  className="w-14 px-1 py-1 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
                />
              </div>
            </div>
            
            {/* Volume Min */}
            <div className="space-y-1">
              <label className="text-xs text-[var(--text-muted)]">最小成交量</label>
              <input
                type="number"
                step="100"
                value={filters.volumeMin}
                onChange={(e) => setFilters({ ...filters, volumeMin: parseInt(e.target.value) || 0 })}
                className="w-full px-2 py-1.5 text-xs bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded"
              />
            </div>
            
            {/* Reset */}
            <div className="flex items-end">
              <button
                onClick={() => setFilters({
                  expiry: 'all', strikeMin: 0, strikeMax: 10, type: 'all',
                  moneyness: 'all', ivMin: 0, ivMax: 200, volumeMin: 0,
                })}
                className="px-3 py-1.5 text-xs bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] rounded transition-colors"
              >
                重置
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Asset List with Sorting */}
      <div className="glass-card p-4">
        <div className="section-header">
          <Activity className="w-4 h-4 text-[var(--accent-warning)]" />
          <h3 className="section-title">期权合约列表</h3>
          <span className="text-xs text-[var(--text-muted)] ml-auto">{filteredAssets.length} 个合约</span>
        </div>
        
        {/* Table Header with Sort */}
        <div className="grid grid-cols-8 gap-2 px-3 py-2 text-xs font-semibold text-[var(--text-muted)] border-b border-[var(--border-primary)]">
          <div>合约ID</div>
          <div>类型</div>
          <div className="text-right cursor-pointer hover:text-[var(--accent-primary)] flex items-center justify-end gap-1" onClick={() => handleSort('strike')}>
            行权价
            {sortField === 'strike' && <ArrowUpDown className="w-3 h-3" />}
          </div>
          <div className="cursor-pointer hover:text-[var(--accent-primary)] flex items-center gap-1" onClick={() => handleSort('expiry')}>
            到期日
            {sortField === 'expiry' && <ArrowUpDown className="w-3 h-3" />}
          </div>
          <div className="text-right cursor-pointer hover:text-[var(--accent-primary)] flex items-center justify-end gap-1" onClick={() => handleSort('close')}>
            收盘价
            {sortField === 'close' && <ArrowUpDown className="w-3 h-3" />}
          </div>
          <div className="text-right cursor-pointer hover:text-[var(--accent-primary)] flex items-center justify-end gap-1" onClick={() => handleSort('changePercent')}>
            涨跌幅
            {sortField === 'changePercent' && <ArrowUpDown className="w-3 h-3" />}
          </div>
          <div className="text-right cursor-pointer hover:text-[var(--accent-primary)] flex items-center justify-end gap-1" onClick={() => handleSort('iv')}>
            IV
            {sortField === 'iv' && <ArrowUpDown className="w-3 h-3" />}
          </div>
          <div className="text-right cursor-pointer hover:text-[var(--accent-primary)] flex items-center justify-end gap-1" onClick={() => handleSort('volume')}>
            成交量
            {sortField === 'volume' && <ArrowUpDown className="w-3 h-3" />}
          </div>
        </div>
        
        {/* Scrollable Asset List */}
        <div className="max-h-80 overflow-y-auto">
          {filteredAssets.map((asset) => (
            <div
              key={asset.id}
              onClick={() => handleAssetSelect(asset)}
              onMouseEnter={(e) => handleRowHover(asset, e)}
              onMouseLeave={() => setShowHoverCard(false)}
              className={`grid grid-cols-8 gap-2 px-3 py-3 cursor-pointer transition-all hover:bg-[var(--bg-card-hover)] border-b border-[var(--border-primary)]/30 ${
                selectedAsset?.id === asset.id ? 'bg-[var(--accent-primary)]/10 border-l-2 border-l-[var(--accent-primary)]' : ''
              }`}
            >
              <div className="font-mono text-xs text-[var(--text-secondary)] truncate">{asset.id}</div>
              <div>
                <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
                  asset.type === 'call' 
                    ? 'bg-[var(--accent-success)]/20 text-[var(--accent-success)]' 
                    : 'bg-[var(--accent-danger)]/20 text-[var(--accent-danger)]'
                }`}>
                  {asset.type === 'call' ? 'C' : 'P'}
                </span>
              </div>
              <div className="text-right font-mono text-sm">¥{asset.strike.toFixed(2)}</div>
              <div className="text-xs text-[var(--text-muted)]">{asset.expiry}</div>
              <div className="text-right font-mono text-sm">¥{asset.close.toFixed(4)}</div>
              <div className={`text-right font-mono text-sm ${asset.changePercent >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'}`}>
                {asset.changePercent >= 0 ? '+' : ''}{asset.changePercent.toFixed(2)}%
              </div>
              <div className="text-right font-mono text-sm text-[var(--accent-primary)]">{(asset.iv * 100).toFixed(1)}%</div>
              <div className="text-right font-mono text-xs text-[var(--text-muted)]">{asset.volume.toLocaleString()}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
