"use client";

import dynamic from 'next/dynamic';
import { useState, useMemo, useEffect, useCallback } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Legend } from 'recharts';
import { 
  Layers, Target, Clock, Calendar, RefreshCw, Loader2,
  TrendingUp, TrendingDown, Activity, BarChart2, Grid3X3, Eye, Cone,
  Triangle, Zap, Timer, Waves
} from 'lucide-react';
import VolatilityConeChart from './VolatilityConeChart';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface VolatilitySurface3DProps {
  selectedDate?: string;
  dateRange?: { start: string; end: string };
  datasetId?: string;
}

type ViewMode = '3d' | 'heatmap' | 'contour';

interface SurfaceData {
  strikes: number[];
  dtes: number[];
  z: number[][];
}

interface SurfaceStats {
  atmIV: number;
  skew25Delta: number;
  termSlope: number;
  curvature: number;
  spotPrice: number;
}

type GreekOverlay = 'none' | 'delta' | 'gamma' | 'vega' | 'theta';

interface GreeksHeatmapData {
  strikes: number[];
  dtes: number[];
  z: number[][];
  greek_type: string;
  spot_price: number;
  trade_date: string;
  min_value: number;
  max_value: number;
}

export default function VolatilitySurface3D({ selectedDate, dateRange, datasetId = '510050_SH' }: VolatilitySurface3DProps) {
  // View state
  const [viewMode, setViewMode] = useState<ViewMode>('3d');
  
  // Side panel mode: 'sections' for term structure & smile, 'cone' for volatility cone
  const [sidePanelMode, setSidePanelMode] = useState<'sections' | 'cone'>('sections');
  
  // Greeks overlay state
  const [greekOverlay, setGreekOverlay] = useState<GreekOverlay>('none');
  const [greeksHeatmapData, setGreeksHeatmapData] = useState<GreeksHeatmapData | null>(null);
  const [greeksLoading, setGreeksLoading] = useState(false);
  
  // IV Change threshold (in percentage points)
  const [ivChangeThreshold, setIvChangeThreshold] = useState(2.0);
  
  // Cross-section selectors - slider indices
  const [selectedStrikeIdx, setSelectedStrikeIdx] = useState(0);
  const [selectedDTEIdx, setSelectedDTEIdx] = useState(0);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentDate, setCurrentDate] = useState(selectedDate || '');
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  
  const [surfaceData, setSurfaceData] = useState<SurfaceData>({ strikes: [], dtes: [], z: [] });
  const [dataQuality, setDataQuality] = useState<{
    real_percent: number;
    calculated_percent: number;
    simulated_percent: number;
    spot_price: number;
  } | null>(null);
  
  // Volatility Cone data
  const [volCone, setVolCone] = useState<{
    dte: number;
    current_iv: number;
    percentile_rank: number;
    p10: number;
    p25: number;
    p50: number;
    p75: number;
    p90: number;
  }[] | null>(null);
  
  // IV Change data with Strike-level details
  const [ivChange, setIvChange] = useState<{
    atm_iv_change: number | null;
    current_atm_iv: number;
    prev_atm_iv: number;
    significant_strikes?: {
      strike: number;
      dte: number;
      current_iv: number;
      prev_iv: number;
      change: number;
      change_pct: number;
      is_significant: boolean;
    }[];
    summary?: {
      total_strikes: number;
      significant_count: number;
      max_increase: number;
      max_decrease: number;
    };
  } | null>(null);

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
          // Only set default date if no external date provided
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

  // Fetch IV surface when date changes
  useEffect(() => {
    if (!currentDate) return;

    const fetchIVSurface = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`http://localhost:8000/api/data/iv-surface?date=${currentDate}&dataset_id=${datasetId}`);
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        setSurfaceData({
          strikes: data.strikes || [],
          dtes: data.dtes || [],
          z: data.iv_matrix || []
        });
        
        // Set data quality metrics
        if (data.data_quality) {
          setDataQuality(data.data_quality);
        }
        
        // Initialize selectors to middle values
        if (data.strikes && data.strikes.length > 0) {
          setSelectedStrikeIdx(Math.floor(data.strikes.length / 2));
        }
        if (data.dtes && data.dtes.length > 0) {
          setSelectedDTEIdx(Math.floor(data.dtes.length / 2));
        }
      } catch (err) {
        console.error('Failed to fetch IV surface:', err);
        setError(err instanceof Error ? err.message : '加载IV曲面失败');
      } finally {
        setLoading(false);
      }
    };
    
    fetchIVSurface();
  }, [currentDate, datasetId]);

  // Fetch volatility cone and IV change when date changes
  useEffect(() => {
    if (!currentDate) return;
    
    // Fetch Volatility Cone
    const fetchVolCone = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/data/volatility-cone?current_date=${currentDate}&lookback_days=60&dataset_id=${datasetId}`);
        if (response.ok) {
          const data = await response.json();
          setVolCone(data.cone || null);
        }
      } catch (err) {
        console.error('Failed to fetch volatility cone:', err);
      }
    };
    
    // Fetch IV Change with Strike-level data
    const fetchIvChange = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/data/iv-change?date=${currentDate}&threshold=${ivChangeThreshold}&dataset_id=${datasetId}`);
        if (response.ok) {
          const data = await response.json();
          setIvChange(data);
        }
      } catch (err) {
        console.error('Failed to fetch IV change:', err);
      }
    };
    
    fetchVolCone();
    fetchIvChange();
  }, [currentDate, ivChangeThreshold, datasetId]);

  // Fetch Greeks heatmap when overlay changes
  useEffect(() => {
    if (greekOverlay === 'none' || !currentDate || !dataQuality?.spot_price) {
      setGreeksHeatmapData(null);
      return;
    }
    
    const fetchGreeksHeatmap = async () => {
      setGreeksLoading(true);
      try {
        const response = await fetch('http://localhost:8000/api/greeks/heatmap', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            trade_date: currentDate,
            spot_price: dataQuality.spot_price,
            volatility: 0.20,
            use_market_iv: true,
            greek_type: greekOverlay,
            option_type: 'call',
            dataset_id: datasetId
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          setGreeksHeatmapData(data);
        }
      } catch (err) {
        console.error('Failed to fetch Greeks heatmap:', err);
      } finally {
        setGreeksLoading(false);
      }
    };
    
    fetchGreeksHeatmap();
  }, [greekOverlay, currentDate, dataQuality?.spot_price, datasetId]);

  // Calculate surface statistics
  const stats = useMemo<SurfaceStats>(() => {
    if (surfaceData.strikes.length === 0 || surfaceData.z.length === 0) {
      return { atmIV: 0, skew25Delta: 0, termSlope: 0, curvature: 0, spotPrice: 0 };
    }

    const strikes = surfaceData.strikes;
    const dtes = surfaceData.dtes;
    const z = surfaceData.z;
    
    const spotIdx = Math.floor(strikes.length / 2);
    const spotPrice = strikes[spotIdx];
    const atmIV = z[0]?.[spotIdx] || 0;
    
    const put25DeltaIdx = Math.max(0, Math.floor(spotIdx * 0.9));
    const put25IV = z[0]?.[put25DeltaIdx] || 0;
    const skew25Delta = (put25IV - atmIV) * 100;
    
    const nearIV = z[0]?.[spotIdx] || 0;
    const farIdx = Math.min(dtes.length - 1, Math.floor(dtes.length * 0.7));
    const farIV = z[farIdx]?.[spotIdx] || 0;
    const termSlope = dtes.length > 1 ? ((farIV - nearIV) / (dtes[farIdx] - dtes[0])) * 100 * 30 : 0;
    
    const callIdx = Math.min(strikes.length - 1, Math.floor(spotIdx * 1.1));
    const callIV = z[0]?.[callIdx] || 0;
    const curvature = ((put25IV + callIV) / 2 - atmIV) * 100;
    
    return { atmIV, skew25Delta, termSlope, curvature, spotPrice };
  }, [surfaceData]);

  // Professional colorscale for IV
  const professionalColorscale = [
    [0, '#0f172a'],
    [0.15, '#1e3a5f'],
    [0.3, '#3b82f6'],
    [0.45, '#6366f1'],
    [0.6, '#8b5cf6'],
    [0.75, '#a855f7'],
    [0.85, '#ec4899'],
    [1, '#ef4444']
  ];

  // Greeks-specific colorscales
  const greeksColorscales: Record<string, [number, string][]> = {
    delta: [
      [0, '#1e40af'],     // Deep blue (put, negative)
      [0.3, '#3b82f6'],   // Blue
      [0.5, '#94a3b8'],   // Neutral gray  
      [0.7, '#ef4444'],   // Red
      [1, '#b91c1c']      // Deep red (call, positive)
    ],
    gamma: [
      [0, '#0f172a'],
      [0.25, '#4c1d95'],
      [0.5, '#7c3aed'],
      [0.75, '#a78bfa'],
      [1, '#ddd6fe']
    ],
    vega: [
      [0, '#052e16'],
      [0.25, '#166534'],
      [0.5, '#22c55e'],
      [0.75, '#86efac'],
      [1, '#dcfce7']
    ],
    theta: [
      [0, '#fef3c7'],
      [0.25, '#fbbf24'],
      [0.5, '#f97316'],
      [0.75, '#ea580c'],
      [1, '#9a3412']
    ]
  };

  // Determine current display data (IV or Greeks overlay)
  const displayData = useMemo(() => {
    if (greekOverlay !== 'none' && greeksHeatmapData) {
      return {
        strikes: greeksHeatmapData.strikes,
        dtes: greeksHeatmapData.dtes,
        z: greeksHeatmapData.z,
        colorscale: greeksColorscales[greekOverlay] || professionalColorscale,
        colorbarTitle: greekOverlay.charAt(0).toUpperCase() + greekOverlay.slice(1),
        tickformat: greekOverlay === 'delta' ? '.2f' : '.4f',
        hoverLabel: greekOverlay.charAt(0).toUpperCase() + greekOverlay.slice(1),
        hoverFormat: greekOverlay === 'delta' ? '%{z:.3f}' : '%{z:.5f}'
      };
    }
    return {
      strikes: surfaceData.strikes,
      dtes: surfaceData.dtes,
      z: surfaceData.z,
      colorscale: professionalColorscale,
      colorbarTitle: 'IV',
      tickformat: '.0%',
      hoverLabel: 'IV',
      hoverFormat: '%{z:.1%}'
    };
  }, [greekOverlay, greeksHeatmapData, surfaceData]);

  // IV Change markers for significant strikes (must be before surface3DData)
  const ivChangeMarkersData = useMemo(() => {
    // Only proceed if we have significant strikes data
    if (!ivChange?.significant_strikes || ivChange.significant_strikes.length === 0 || 
        surfaceData.strikes.length === 0 || surfaceData.z.length === 0) {
      return null;
    }
    
    // Build markers for significant IV changes
    const markers = {
      increases: { x: [] as number[], y: [] as number[], z: [] as number[], text: [] as string[] },
      decreases: { x: [] as number[], y: [] as number[], z: [] as number[], text: [] as string[] }
    };
    
    for (const sc of ivChange.significant_strikes) {
      const strikeIdx = surfaceData.strikes.findIndex(s => Math.abs(s - sc.strike) < 0.01);
      const dteIdx = surfaceData.dtes.findIndex(d => d === sc.dte);
      
      if (strikeIdx >= 0 && dteIdx >= 0 && surfaceData.z[dteIdx]?.[strikeIdx] != null) {
        const zVal = surfaceData.z[dteIdx][strikeIdx];
        const hoverText = `K=${sc.strike.toFixed(2)}, DTE=${sc.dte}\nIV变动: ${sc.change > 0 ? '+' : ''}${sc.change.toFixed(2)}%`;
        
        if (sc.change > 0) {
          markers.increases.x.push(sc.strike);
          markers.increases.y.push(sc.dte);
          markers.increases.z.push(zVal + 0.02);
          markers.increases.text.push(hoverText);
        } else {
          markers.decreases.x.push(sc.strike);
          markers.decreases.y.push(sc.dte);
          markers.decreases.z.push(zVal + 0.02);
          markers.decreases.text.push(hoverText);
        }
      }
    }
    
    if (markers.increases.x.length === 0 && markers.decreases.x.length === 0) {
      return null;
    }
    
    return markers;
  }, [ivChange, surfaceData]);

  // 3D Surface plot data
  const surface3DData = useMemo(() => {
    const surfaceTrace = {
      type: 'surface' as const,
      x: displayData.strikes,
      y: displayData.dtes,
      z: displayData.z,
      colorscale: displayData.colorscale,
      showscale: true,
      colorbar: {
        title: { text: displayData.colorbarTitle, font: { color: '#f0f2f5', size: 11 } },
        titleside: 'right' as const,
        tickformat: displayData.tickformat,
        tickfont: { color: '#9ca3af', size: 10 },
        len: 0.8,
        thickness: 12,
        x: 1.02,
        borderwidth: 0,
        outlinewidth: 0,
      },
      contours: {
        x: { show: true, color: 'rgba(255,255,255,0.1)', width: 1 },
        y: { show: true, color: 'rgba(255,255,255,0.1)', width: 1 },
        z: { show: true, usecolormap: true, highlightcolor: "#fff", project: { z: true }, width: 1 }
      },
      lighting: {
        ambient: 0.7,
        diffuse: 0.85,
        specular: 0.25,
        roughness: 0.35,
        fresnel: 0.3
      },
      lightposition: { x: 100, y: 200, z: 1000 },
      opacity: 0.97,
      hovertemplate: 
        '<b>行权价:</b> ¥%{x:.2f}<br>' +
        '<b>到期:</b> %{y}天<br>' +
        `<b>${displayData.hoverLabel}:</b> ${displayData.hoverFormat}<extra></extra>`
    };
  
    // Build data array with optional IV change markers
    const result: any[] = [surfaceTrace];
  
    if (ivChangeMarkersData && greekOverlay === 'none') {
      // Add increase markers (red triangles)
      if (ivChangeMarkersData.increases.x.length > 0) {
        result.push({
          type: 'scatter3d',
          mode: 'markers',
          x: ivChangeMarkersData.increases.x,
          y: ivChangeMarkersData.increases.y,
          z: ivChangeMarkersData.increases.z,
          text: ivChangeMarkersData.increases.text,
          hoverinfo: 'text',
          marker: {
            symbol: 'diamond',
            size: 6,
            color: '#ef4444',
            line: { color: '#ffffff', width: 1 }
          },
          name: 'IV↑'
        });
      }
      // Add decrease markers (blue triangles)
      if (ivChangeMarkersData.decreases.x.length > 0) {
        result.push({
          type: 'scatter3d',
          mode: 'markers',
          x: ivChangeMarkersData.decreases.x,
          y: ivChangeMarkersData.decreases.y,
          z: ivChangeMarkersData.decreases.z,
          text: ivChangeMarkersData.decreases.text,
          hoverinfo: 'text',
          marker: {
            symbol: 'diamond',
            size: 6,
            color: '#3b82f6',
            line: { color: '#ffffff', width: 1 }
          },
          name: 'IV↓'
        });
      }
    }
    
    return result;
  }, [displayData, ivChangeMarkersData, greekOverlay]);

  
  // Heatmap data
  const heatmapData = useMemo(() => [{
    type: 'heatmap' as const,
    x: displayData.strikes,
    y: displayData.dtes,
    z: displayData.z,
    colorscale: displayData.colorscale,
    showscale: true,
    colorbar: {
      title: { text: displayData.colorbarTitle, font: { color: '#f0f2f5', size: 11 } },
      tickformat: displayData.tickformat,
      tickfont: { color: '#9ca3af', size: 10 },
      len: 0.9,
      thickness: 12,
    },
    hovertemplate: 
      '<b>行权价:</b> ¥%{x:.2f}<br>' +
      '<b>到期:</b> %{y}天<br>' +
      `<b>${displayData.hoverLabel}:</b> ${displayData.hoverFormat}<extra></extra>`,
    xgap: 1,
    ygap: 1,
  }], [displayData]);
  
  // Contour data
  const contourData = useMemo(() => [{
    type: 'contour' as const,
    x: displayData.strikes,
    y: displayData.dtes,
    z: displayData.z,
    colorscale: displayData.colorscale,
    showscale: true,
    colorbar: {
      title: { text: displayData.colorbarTitle, font: { color: '#f0f2f5', size: 11 } },
      tickformat: displayData.tickformat,
      tickfont: { color: '#9ca3af', size: 10 },
      len: 0.9,
      thickness: 12,
    },
    contours: {
      coloring: 'heatmap' as const,
      showlabels: true,
      labelfont: { size: 10, color: 'white' }
    },
    line: { width: 1.5, color: 'rgba(255,255,255,0.5)' },
    hovertemplate: 
      '<b>行权价:</b> ¥%{x:.2f}<br>' +
      '<b>到期:</b> %{y}天<br>' +
      `<b>${displayData.hoverLabel}:</b> ${displayData.hoverFormat}<extra></extra>`
  }], [displayData]);

  // Term structure data for Recharts (IV/Greek vs DTE for selected strike)
  const termStructureChartData = useMemo(() => {
    if (displayData.strikes.length === 0 || displayData.z.length === 0) return [];
    
    const strikeIdx = Math.min(selectedStrikeIdx, displayData.strikes.length - 1);
    
    return displayData.dtes.map((dte, i) => ({
      dte: dte,
      label: `${dte}天`,
      value: displayData.z[i]?.[strikeIdx] || 0,
    }));
  }, [displayData, selectedStrikeIdx]);

  // Volatility smile data for Recharts (IV/Greek vs Moneyness for selected DTE)
  const volatilitySmileChartData = useMemo(() => {
    if (displayData.strikes.length === 0 || displayData.z.length === 0) return [];
    
    const dteIdx = Math.min(selectedDTEIdx, displayData.dtes.length - 1);
    const spot = stats.spotPrice || displayData.strikes[Math.floor(displayData.strikes.length / 2)] || 1;
    
    const valueRow = displayData.z[dteIdx] || [];
    
    return displayData.strikes.map((strike, i) => ({
      strike: strike,
      moneyness: strike / spot,
      moneynessLabel: (strike / spot).toFixed(2),
      value: valueRow[i] || 0,
    }));
  }, [displayData, selectedDTEIdx, stats.spotPrice]);

  // Layout configurations
  const layout3D = useMemo(() => ({
    autosize: true,
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 0, r: 50, t: 10, b: 0 },
    scene: {
      xaxis: { 
        title: { text: '行权价', font: { color: '#9ca3af', size: 11 } },
        tickfont: { color: '#6b7280', size: 10 },
        gridcolor: '#1f2937',
        showbackground: true,
        backgroundcolor: '#0c0f14'
      },
      yaxis: { 
        title: { text: 'DTE (天)', font: { color: '#9ca3af', size: 11 } },
        tickfont: { color: '#6b7280', size: 10 },
        gridcolor: '#1f2937',
        showbackground: true,
        backgroundcolor: '#0c0f14'
      },
      zaxis: { 
        title: { text: 'IV', font: { color: '#9ca3af', size: 11 } },
        tickfont: { color: '#6b7280', size: 10 },
        gridcolor: '#1f2937',
        showbackground: true,
        backgroundcolor: '#0c0f14',
        tickformat: '.0%'
      },
      camera: { eye: { x: 1.6, y: 1.6, z: 1.0 } },
      aspectratio: { x: 1.2, y: 1, z: 0.7 }
    }
  }), []);

  const layout2D = useMemo(() => ({
    autosize: true,
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 60, r: 80, t: 20, b: 50 },
    xaxis: {
      gridcolor: '#1f2937',
      tickfont: { color: '#6b7280', size: 10 },
      zerolinecolor: '#1f2937',
    },
    yaxis: {
      gridcolor: '#1f2937',
      tickfont: { color: '#6b7280', size: 10 },
      zerolinecolor: '#1f2937',
    }
  }), []);

  const config = { displayModeBar: false, responsive: true };
  const config3D = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'autoScale2d', 'hoverClosest3d'],
    responsive: true,
    scrollZoom: true,
  };

  // Loading state
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <Loader2 className="w-10 h-10 text-[var(--accent-primary)] animate-spin" />
        <div className="text-sm text-[var(--text-muted)]">加载IV曲面数据...</div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
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

  const selectedStrike = surfaceData.strikes[selectedStrikeIdx] || 0;
  const selectedDTE = surfaceData.dtes[selectedDTEIdx] || 0;

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Stats Header */}
      <div className="grid grid-cols-5 gap-3">
        {/* Date Selector */}
        <div className="glass-card-elevated px-3 py-2 flex items-center gap-2 rounded-lg">
          <Calendar className="w-4 h-4 text-[var(--accent-primary)]" />
          <select
            value={currentDate}
            onChange={(e) => setCurrentDate(e.target.value)}
            className="bg-transparent border-none text-sm text-[var(--text-primary)] focus:outline-none cursor-pointer flex-1"
          >
            {availableDates.map(date => (
              <option key={date} value={date} className="bg-[var(--bg-elevated)]">{date}</option>
            ))}
          </select>
          <button 
            onClick={() => setCurrentDate(currentDate)} 
            className="p-1 hover:bg-[var(--bg-card)] rounded transition-colors"
            title="刷新数据"
          >
            <RefreshCw className="w-3.5 h-3.5 text-[var(--text-muted)]" />
          </button>
        </div>

        {/* ATM IV with Change */}
        <div className="glass-card-elevated px-3 py-2 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <Target className="w-3.5 h-3.5 text-[var(--accent-primary)]" />
            <span className="text-xs text-[var(--text-muted)]">ATM IV</span>
            {ivChange?.atm_iv_change !== null && ivChange?.atm_iv_change !== undefined && (
              <span className={`text-xs font-medium ${ivChange.atm_iv_change >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'}`}>
                {ivChange.atm_iv_change >= 0 ? '↑' : '↓'}{Math.abs(ivChange.atm_iv_change).toFixed(1)}
              </span>
            )}
          </div>
          <div className="flex items-baseline gap-2">
            <div className="text-lg font-bold text-[var(--accent-primary)]">
              {(stats.atmIV * 100).toFixed(1)}%
            </div>
            {volCone && volCone[0] && (
              <span className={`text-xs px-1.5 py-0.5 rounded ${
                volCone[0].percentile_rank > 75 ? 'bg-[var(--accent-danger)]/20 text-[var(--accent-danger)]' :
                volCone[0].percentile_rank < 25 ? 'bg-[var(--accent-success)]/20 text-[var(--accent-success)]' :
                'bg-[var(--accent-warning)]/20 text-[var(--accent-warning)]'
              }`}>
                P{Math.round(volCone[0].percentile_rank)}
              </span>
            )}
          </div>
        </div>

        {/* 25Δ Put Skew */}
        <div className="glass-card-elevated px-3 py-2 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            {stats.skew25Delta < 0 ? 
              <TrendingDown className="w-3.5 h-3.5 text-[var(--accent-danger)]" /> :
              <TrendingUp className="w-3.5 h-3.5 text-[var(--accent-success)]" />
            }
            <span className="text-xs text-[var(--text-muted)]">25Δ Skew</span>
          </div>
          <div className={`text-lg font-bold ${stats.skew25Delta < 0 ? 'text-[var(--accent-danger)]' : 'text-[var(--accent-success)]'}`}>
            {stats.skew25Delta >= 0 ? '+' : ''}{stats.skew25Delta.toFixed(1)}%
          </div>
        </div>

        {/* Term Structure */}
        <div className="glass-card-elevated px-3 py-2 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <Clock className="w-3.5 h-3.5 text-[var(--accent-warning)]" />
            <span className="text-xs text-[var(--text-muted)]">期限斜率</span>
          </div>
          <div className={`text-lg font-bold ${stats.termSlope < 0 ? 'text-[var(--accent-danger)]' : 'text-[var(--accent-success)]'}`}>
            {stats.termSlope >= 0 ? '+' : ''}{stats.termSlope.toFixed(2)}
          </div>
        </div>

        {/* Curvature */}
        <div className="glass-card-elevated px-3 py-2 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <Activity className="w-3.5 h-3.5 text-[var(--accent-secondary)]" />
            <span className="text-xs text-[var(--text-muted)]">微笑曲率</span>
          </div>
          <div className="text-lg font-bold text-[var(--accent-secondary)]">
            {stats.curvature.toFixed(2)}
          </div>
        </div>
      </div>

      {/* View Mode Toggle */}
      <div className="flex items-center gap-2">
        <div className="glass-card-elevated p-1 rounded-lg flex gap-1">
          <button
            onClick={() => setViewMode('3d')}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-all flex items-center gap-1.5 ${
              viewMode === '3d' 
                ? 'bg-[var(--accent-primary)] text-white' 
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]'
            }`}
          >
            <Layers className="w-3.5 h-3.5" /> 3D曲面
          </button>
          <button
            onClick={() => setViewMode('heatmap')}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-all flex items-center gap-1.5 ${
              viewMode === 'heatmap' 
                ? 'bg-[var(--accent-primary)] text-white' 
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]'
            }`}
          >
            <Grid3X3 className="w-3.5 h-3.5" /> 热力图
          </button>
          <button
            onClick={() => setViewMode('contour')}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-all flex items-center gap-1.5 ${
              viewMode === 'contour' 
                ? 'bg-[var(--accent-primary)] text-white' 
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]'
            }`}
          >
            <Eye className="w-3.5 h-3.5" /> 等高线
          </button>
        </div>
        
        {/* Greeks Overlay Toggle */}
        <div className="glass-card-elevated p-1 rounded-lg flex gap-1">
          <span className="px-2 py-1.5 text-xs text-[var(--text-muted)]">Greeks:</span>
          <button
            onClick={() => setGreekOverlay('none')}
            className={`px-2 py-1.5 rounded text-xs font-medium transition-all ${
              greekOverlay === 'none' 
                ? 'bg-[var(--accent-secondary)] text-white' 
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]'
            }`}
          >
            IV
          </button>
          <button
            onClick={() => setGreekOverlay('delta')}
            className={`px-2 py-1.5 rounded text-xs font-medium transition-all flex items-center gap-1 ${
              greekOverlay === 'delta' 
                ? 'bg-blue-600 text-white' 
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]'
            }`}
          >
            <Triangle className="w-3 h-3" /> Δ
          </button>
          <button
            onClick={() => setGreekOverlay('gamma')}
            className={`px-2 py-1.5 rounded text-xs font-medium transition-all flex items-center gap-1 ${
              greekOverlay === 'gamma' 
                ? 'bg-purple-600 text-white' 
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]'
            }`}
          >
            <Zap className="w-3 h-3" /> Γ
          </button>
          <button
            onClick={() => setGreekOverlay('vega')}
            className={`px-2 py-1.5 rounded text-xs font-medium transition-all flex items-center gap-1 ${
              greekOverlay === 'vega' 
                ? 'bg-green-600 text-white' 
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]'
            }`}
          >
            <Waves className="w-3 h-3" /> ν
          </button>
          <button
            onClick={() => setGreekOverlay('theta')}
            className={`px-2 py-1.5 rounded text-xs font-medium transition-all flex items-center gap-1 ${
              greekOverlay === 'theta' 
                ? 'bg-orange-600 text-white' 
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]'
            }`}
          >
            <Timer className="w-3 h-3" /> Θ
          </button>
          {greeksLoading && <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--text-muted)] ml-1" />}
        </div>
        
        {/* IV Change Threshold Control */}
        {greekOverlay === 'none' && (
          <div className="glass-card-elevated p-1.5 px-3 rounded-lg flex items-center gap-2">
            <Zap className="w-3.5 h-3.5 text-[var(--accent-warning)]" />
            <span className="text-xs text-[var(--text-muted)]">IV变动≥</span>
            <input
              type="number"
              min={0}
              step={0.1}
              value={ivChangeThreshold}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                if (!isNaN(val) && val > 0) setIvChangeThreshold(val);
              }}
              className="w-14 h-6 px-1.5 text-xs text-center bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded text-[var(--accent-warning)] font-mono focus:outline-none focus:border-[var(--accent-warning)]"
            />
            <span className="text-xs text-[var(--text-muted)]">%</span>
            {ivChange?.summary && (
              <span className="text-xs text-[var(--text-muted)] border-l border-[var(--border-subtle)] pl-2">
                <span className="text-[var(--accent-danger)]">↑{ivChange.significant_strikes?.filter(s => s.change > 0).length || 0}</span>
                {' / '}
                <span className="text-[var(--accent-primary)]">↓{ivChange.significant_strikes?.filter(s => s.change < 0).length || 0}</span>
              </span>
            )}
          </div>
        )}
        
        <div className="flex-1" />

        
        {/* Data Quality Indicator */}
        {dataQuality && (
          <div className="flex items-center gap-3 mr-4">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-[var(--accent-success)]" />
              <span className="text-xs text-[var(--text-muted)]">真实 <span className="text-[var(--accent-success)] font-medium">{dataQuality.real_percent}%</span></span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-[var(--accent-primary)]" />
              <span className="text-xs text-[var(--text-muted)]">计算 <span className="text-[var(--accent-primary)] font-medium">{dataQuality.calculated_percent}%</span></span>
            </div>
            {dataQuality.simulated_percent > 0 && (
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-[var(--accent-warning)]" />
                <span className="text-xs text-[var(--text-muted)]">模拟 <span className="text-[var(--accent-warning)] font-medium">{dataQuality.simulated_percent}%</span></span>
              </div>
            )}
          </div>
        )}
        
        <div className="text-xs text-[var(--text-muted)]">
          现货: <span className="text-[var(--text-primary)] font-medium">¥{dataQuality?.spot_price?.toFixed(3) || stats.spotPrice.toFixed(3)}</span>
        </div>
      </div>

      {/* Main Content: Surface + Cross-sections side by side */}
      <div className="flex-1 grid grid-cols-3 gap-3 min-h-0">
        {/* Main View Area (2/3 width) */}
        <div className="col-span-2 glass-card-elevated rounded-lg overflow-hidden">
          {viewMode === '3d' && (
            <Plot 
              data={surface3DData} 
              layout={layout3D} 
              config={config3D}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler={true}
            />
          )}
          {viewMode === 'heatmap' && (
            <Plot 
              data={heatmapData} 
              layout={{
                ...layout2D,
                xaxis: { ...layout2D.xaxis, title: { text: '行权价 (Strike)', font: { color: '#9ca3af', size: 11 } } },
                yaxis: { ...layout2D.yaxis, title: { text: 'DTE (天)', font: { color: '#9ca3af', size: 11 } } }
              }} 
              config={config}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler={true}
            />
          )}
          {viewMode === 'contour' && (
            <Plot 
              data={contourData} 
              layout={{
                ...layout2D,
                xaxis: { ...layout2D.xaxis, title: { text: '行权价 (Strike)', font: { color: '#9ca3af', size: 11 } } },
                yaxis: { ...layout2D.yaxis, title: { text: 'DTE (天)', font: { color: '#9ca3af', size: 11 } } }
              }} 
              config={config}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler={true}
            />
          )}
        </div>

        {/* Side Panels (1/3 width) */}
        <div className="flex flex-col gap-3">
          {/* Side Panel Mode Toggle */}
          <div className="glass-card-elevated p-1 rounded-lg flex gap-1">
            <button
              onClick={() => setSidePanelMode('sections')}
              className={`flex-1 px-2 py-1.5 rounded text-xs font-medium transition-all flex items-center justify-center gap-1 ${
                sidePanelMode === 'sections' 
                  ? 'bg-[var(--accent-primary)] text-white' 
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]'
              }`}
            >
              <BarChart2 className="w-3 h-3" /> 切面分析
            </button>
            <button
              onClick={() => setSidePanelMode('cone')}
              className={`flex-1 px-2 py-1.5 rounded text-xs font-medium transition-all flex items-center justify-center gap-1 ${
                sidePanelMode === 'cone' 
                  ? 'bg-[var(--accent-secondary)] text-white' 
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]'
              }`}
            >
              <TrendingUp className="w-3 h-3" /> 波动率锥
            </button>
          </div>

          {/* Conditional Panel Content */}
          {sidePanelMode === 'cone' ? (
            /* Volatility Cone Panel */
            <div className="flex-1 glass-card-elevated p-3 rounded-lg flex flex-col min-h-0">
              <VolatilityConeChart data={volCone} loading={loading} />
            </div>
          ) : (
            /* Cross-Section Panels */
            <>
              {/* Term Structure Chart */}
              <div className="flex-1 glass-card-elevated p-3 rounded-lg flex flex-col min-h-0">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart2 className="w-4 h-4 text-[var(--accent-primary)]" />
                  <span className="text-xs font-semibold text-[var(--text-secondary)]">
                    {greekOverlay === 'none' ? '期限结构' : `${displayData.colorbarTitle} 期限结构`}
                  </span>
                  <span className="text-xs text-[var(--text-muted)] ml-auto">
                    K = {selectedStrike.toFixed(2)}
                  </span>
                </div>
            
                {/* Strike Slider */}
                <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-[var(--text-muted)] whitespace-nowrap">行权价:</span>
              <input
                type="range"
                min={0}
                max={Math.max(0, displayData.strikes.length - 1)}
                value={selectedStrikeIdx}
                onChange={(e) => setSelectedStrikeIdx(parseInt(e.target.value))}
                className="flex-1 h-1.5 bg-[var(--bg-card)] rounded-lg appearance-none cursor-pointer accent-[var(--accent-primary)]"
              />
              <span className="text-xs text-[var(--text-primary)] font-mono w-12 text-right">
                {selectedStrike.toFixed(2)}
              </span>
            </div>
            
            {/* Term Structure Line Chart */}
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={termStructureChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <defs>
                    <linearGradient id="termGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2d35" vertical={false} />
                  <XAxis 
                    dataKey="dte" 
                    stroke="#6b7280" 
                    fontSize={9} 
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `${v}天`}
                  />
                  <YAxis 
                    stroke="#6b7280" 
                    fontSize={9} 
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => greekOverlay === 'none' ? `${(v * 100).toFixed(0)}%` : v.toFixed(3)}
                    domain={['auto', 'auto']}
                    width={40}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#161920', 
                      border: '1px solid #2a2d35',
                      borderRadius: '8px',
                      padding: '8px'
                    }}
                    labelStyle={{ color: '#9ca3af', fontSize: 10 }}
                    formatter={(value: any) => [
                      greekOverlay === 'none' ? `${(value * 100).toFixed(1)}%` : value.toFixed(4), 
                      displayData.colorbarTitle
                    ]}
                    labelFormatter={(label) => `${label}天到期`}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#8b5cf6" 
                    strokeWidth={2}
                    dot={{ r: 3, fill: '#8b5cf6' }}
                    activeDot={{ r: 5, fill: '#a855f7' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Volatility Smile Chart */}
          <div className="flex-1 glass-card-elevated p-3 rounded-lg flex flex-col min-h-0">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-[var(--accent-success)]" />
              <span className="text-xs font-semibold text-[var(--text-secondary)]">
                {greekOverlay === 'none' ? '波动率微笑' : `${displayData.colorbarTitle} 分布`}
              </span>
              <span className="text-xs text-[var(--text-muted)] ml-auto">
                DTE = {selectedDTE}天
              </span>
            </div>
            
            {/* DTE Slider */}
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-[var(--text-muted)] whitespace-nowrap">到期日:</span>
              <input
                type="range"
                min={0}
                max={Math.max(0, displayData.dtes.length - 1)}
                value={selectedDTEIdx}
                onChange={(e) => setSelectedDTEIdx(parseInt(e.target.value))}
                className="flex-1 h-1.5 bg-[var(--bg-card)] rounded-lg appearance-none cursor-pointer accent-[var(--accent-success)]"
              />
              <span className="text-xs text-[var(--text-primary)] font-mono w-12 text-right">
                {selectedDTE}天
              </span>
            </div>
            
            {/* Smile Line Chart */}
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={volatilitySmileChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <defs>
                    <linearGradient id="smileGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2d35" vertical={false} />
                  <XAxis 
                    dataKey="moneyness" 
                    stroke="#6b7280" 
                    fontSize={9} 
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => v.toFixed(2)}
                  />
                  <YAxis 
                    stroke="#6b7280" 
                    fontSize={9} 
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => greekOverlay === 'none' ? `${(v * 100).toFixed(0)}%` : v.toFixed(3)}
                    domain={['auto', 'auto']}
                    width={40}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#161920', 
                      border: '1px solid #2a2d35',
                      borderRadius: '8px',
                      padding: '8px'
                    }}
                    labelStyle={{ color: '#9ca3af', fontSize: 10 }}
                    formatter={(value: any) => [
                      greekOverlay === 'none' ? `${(value * 100).toFixed(1)}%` : value.toFixed(4), 
                      displayData.colorbarTitle
                    ]}
                    labelFormatter={(label) => `Moneyness: ${parseFloat(label).toFixed(2)}`}
                  />
                  <ReferenceLine 
                    x={1} 
                    stroke="#6366f1" 
                    strokeDasharray="3 3" 
                    label={{ value: 'ATM', position: 'top', fill: '#6366f1', fontSize: 9 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    dot={{ r: 3, fill: '#10b981' }}
                    activeDot={{ r: 5, fill: '#34d399' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
