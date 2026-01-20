"use client";

import { useState, useEffect, useMemo } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { Loader2, RefreshCw } from 'lucide-react';

interface SimulationPreviewProps {
  numPaths?: number;
  numDays?: number;
}

export default function SimulationPreview({ numPaths = 10, numDays = 30 }: SimulationPreviewProps) {
  const [basePrice, setBasePrice] = useState(3.0);
  const [historicalVol, setHistoricalVol] = useState(0.25);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch real 50ETF data to get base price and historical volatility
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await fetch('http://localhost:8000/api/data/etf-candle?days=60');
        if (response.ok) {
          const data = await response.json();
          const candles = data.candles || [];
          
          if (candles.length > 0) {
            // Get the latest price
            const lastPrice = candles[candles.length - 1].close;
            setBasePrice(lastPrice);
            
            // Calculate historical volatility from returns
            if (candles.length > 1) {
              const returns = candles.slice(1).map((c: any, i: number) => 
                Math.log(c.close / candles[i].close)
              );
              const mean = returns.reduce((a: number, b: number) => a + b, 0) / returns.length;
              const variance = returns.reduce((a: number, b: number) => a + Math.pow(b - mean, 2), 0) / returns.length;
              const dailyVol = Math.sqrt(variance);
              setHistoricalVol(dailyVol * Math.sqrt(252)); // Annualize
            }
          }
        }
      } catch (err) {
        console.error('Failed to fetch ETF data:', err);
        setError('Using default parameters');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);

  // Generate Monte Carlo paths based on real parameters
  const { data, paths } = useMemo(() => {
    const dt = 1 / 252; // Daily time step
    const mu = 0.05; // Expected annual return
    const sigma = historicalVol;
    
    const generatedPaths = Array.from({ length: numPaths }, (_, pathIdx) => {
      let price = basePrice;
      return Array.from({ length: numDays }, (_, day) => {
        // Geometric Brownian Motion
        const z = gaussianRandom();
        const drift = (mu - 0.5 * sigma * sigma) * dt;
        const diffusion = sigma * Math.sqrt(dt) * z;
        price = price * Math.exp(drift + diffusion);
        
        return {
          day: day + 1,
          [`path${pathIdx}`]: price,
        };
      });
    });
    
    // Merge all paths into single data array
    const mergedData = Array.from({ length: numDays }, (_, day) => {
      const point: any = { day: day + 1 };
      generatedPaths.forEach((path, idx) => {
        point[`path${idx}`] = path[day][`path${idx}`];
      });
      return point;
    });
    
    return { data: mergedData, paths: generatedPaths };
  }, [basePrice, historicalVol, numPaths, numDays]);

  // Standard normal random number (Box-Muller transform)
  function gaussianRandom(): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // Colors for paths
  const colors = [
    '#8b5cf6', '#ec4899', '#06b6d4', '#10b981', '#f59e0b',
    '#ef4444', '#6366f1', '#14b8a6', '#f97316', '#84cc16'
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-6 h-6 animate-spin text-[var(--accent-primary)]" />
      </div>
    );
  }

  // Calculate statistics
  const finalPrices = paths.map(p => p[numDays - 1][`path${paths.indexOf(p)}`]);
  const meanPrice = finalPrices.reduce((a, b) => a + b, 0) / finalPrices.length;
  const minPrice = Math.min(...finalPrices);
  const maxPrice = Math.max(...finalPrices);

  return (
    <div className="h-full flex flex-col">
      {/* Stats */}
      <div className="flex items-center gap-4 mb-2 text-xs">
        <span className="text-[var(--text-muted)]">
          起始价: <span className="text-[var(--text-primary)]">¥{basePrice.toFixed(3)}</span>
        </span>
        <span className="text-[var(--text-muted)]">
          波动率: <span className="text-[var(--accent-warning)]">{(historicalVol * 100).toFixed(1)}%</span>
        </span>
        <span className="text-[var(--text-muted)]">
          均值: <span className="text-[var(--accent-success)]">¥{meanPrice.toFixed(3)}</span>
        </span>
        {error && <span className="text-[var(--accent-warning)]">⚠</span>}
      </div>
      
      <div className="flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <defs>
              {colors.map((color, idx) => (
                <linearGradient key={idx} id={`pathGrad${idx}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={color} stopOpacity={0.2}/>
                </linearGradient>
              ))}
            </defs>
            
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2d35" vertical={false} />
            
            <XAxis 
              dataKey="day" 
              stroke="#6b7280" 
              fontSize={10} 
              tickLine={false} 
              axisLine={false}
              label={{ value: '交易日', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 10 }}
            />
            
            <YAxis 
              stroke="#6b7280" 
              fontSize={10} 
              tickLine={false} 
              axisLine={false}
              domain={['auto', 'auto']}
              tickFormatter={(v) => `¥${v.toFixed(2)}`}
            />
            
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#161920', 
                border: '1px solid #2a2d35',
                borderRadius: '8px',
                padding: '8px 12px'
              }}
              labelStyle={{ color: '#9ca3af', fontSize: 11 }}
              formatter={(value: any) => [`¥${value.toFixed(4)}`, '']}
            />
            
            {Array.from({ length: numPaths }).map((_, idx) => (
              <Line 
                key={idx}
                type="monotone" 
                dataKey={`path${idx}`} 
                stroke={colors[idx % colors.length]} 
                strokeWidth={1.5}
                dot={false}
                strokeOpacity={0.7}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
