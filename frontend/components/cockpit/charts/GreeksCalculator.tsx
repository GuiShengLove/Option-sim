"use client";

import { useState, useEffect, useMemo } from 'react';
import { Activity, TrendingUp, TrendingDown, Zap, Clock, Target, AlertTriangle, RefreshCw } from 'lucide-react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface GreeksData {
  price: number;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
  vanna: number;
  volga: number;
  charm: number;
  speed: number;
  color: number;
}

interface GreeksCalculatorProps {
  onCalculate?: (greeks: GreeksData) => void;
}

export default function GreeksCalculator({ onCalculate }: GreeksCalculatorProps) {
  // Input parameters
  const [spot, setSpot] = useState(3.0);
  const [strike, setStrike] = useState(3.0);
  const [tte, setTte] = useState(30); // Days
  const [vol, setVol] = useState(20); // Percentage
  const [isCall, setIsCall] = useState(true);
  const [riskFree, setRiskFree] = useState(3); // Percentage
  
  // Results
  const [greeks, setGreeks] = useState<GreeksData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Auto-calculate on parameter change
  useEffect(() => {
    const calculateGreeks = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch('http://localhost:8000/api/greeks/calculate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            spot,
            strike,
            time_to_expiry: tte / 365,
            volatility: vol / 100,
            is_call: isCall,
            risk_free_rate: riskFree / 100,
            dividend_yield: 0.01,
          }),
        });
        
        if (!response.ok) {
          throw new Error('API request failed');
        }
        
        const data = await response.json();
        setGreeks(data);
        onCalculate?.(data);
      } catch (err) {
        // Use local calculation fallback
        const T = tte / 365;
        const r = riskFree / 100;
        const sigma = vol / 100;
        
        // Simplified BSM approximation
        const d1 = (Math.log(spot / strike) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        const d2 = d1 - sigma * Math.sqrt(T);
        
        const N = (x: number) => 0.5 * (1 + erf(x / Math.sqrt(2)));
        const n = (x: number) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
        
        function erf(x: number): number {
          const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
          const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
          const sign = x < 0 ? -1 : 1;
          x = Math.abs(x);
          const t = 1.0 / (1.0 + p * x);
          const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
          return sign * y;
        }
        
        const price = isCall
          ? spot * N(d1) - strike * Math.exp(-r * T) * N(d2)
          : strike * Math.exp(-r * T) * N(-d2) - spot * N(-d1);
        
        const delta = isCall ? N(d1) : N(d1) - 1;
        const gamma = n(d1) / (spot * sigma * Math.sqrt(T));
        const vega = spot * n(d1) * Math.sqrt(T) / 100;
        const theta = isCall
          ? (-spot * n(d1) * sigma / (2 * Math.sqrt(T)) - r * strike * Math.exp(-r * T) * N(d2)) / 365
          : (-spot * n(d1) * sigma / (2 * Math.sqrt(T)) + r * strike * Math.exp(-r * T) * N(-d2)) / 365;
        const rho = isCall
          ? strike * T * Math.exp(-r * T) * N(d2) / 100
          : -strike * T * Math.exp(-r * T) * N(-d2) / 100;
        
        const vanna = -n(d1) * d2 / sigma / 100;
        const volga = vega * d1 * d2 / sigma;
        
        setGreeks({
          price: parseFloat(price.toFixed(6)),
          delta: parseFloat(delta.toFixed(4)),
          gamma: parseFloat(gamma.toFixed(6)),
          vega: parseFloat(vega.toFixed(4)),
          theta: parseFloat(theta.toFixed(4)),
          rho: parseFloat(rho.toFixed(4)),
          vanna: parseFloat(vanna.toFixed(6)),
          volga: parseFloat(volga.toFixed(6)),
          charm: 0,
          speed: 0,
          color: 0,
        });
      } finally {
        setLoading(false);
      }
    };
    
    const debounce = setTimeout(calculateGreeks, 300);
    return () => clearTimeout(debounce);
  }, [spot, strike, tte, vol, isCall, riskFree, onCalculate]);

  // P&L Surface data
  const plSurfaceData = useMemo(() => {
    if (!greeks) return [];
    
    const spotRange = Array.from({ length: 21 }, (_, i) => spot * (0.9 + i * 0.01));
    const volRange = Array.from({ length: 11 }, (_, i) => vol * (0.5 + i * 0.1));
    
    const z = volRange.map(v => 
      spotRange.map(s => {
        // Quick P&L approximation using Greeks
        const dS = s - spot;
        const dVol = (v - vol) / 100;
        return greeks.delta * dS + 
               0.5 * greeks.gamma * dS * dS + 
               greeks.vega * 100 * dVol +
               0.5 * greeks.volga * dVol * dVol * 10000 +
               greeks.vanna * dS * dVol * 100;
      })
    );
    
    return [{
      type: 'surface' as const,
      x: spotRange,
      y: volRange,
      z: z,
      colorscale: [
        [0, '#ef4444'],
        [0.4, '#f59e0b'],
        [0.5, '#1a1c22'],
        [0.6, '#10b981'],
        [1, '#22c55e']
      ],
      showscale: true,
      colorbar: {
        title: 'P&L',
        titleside: 'right',
        tickfont: { color: '#9ca3af', size: 10 },
        titlefont: { color: '#f0f2f5', size: 11 },
      },
    }];
  }, [greeks, spot, vol]);

  const GreekCard = ({ 
    label, 
    value, 
    unit = '', 
    icon: Icon, 
    color,
    description 
  }: { 
    label: string; 
    value: number; 
    unit?: string; 
    icon: any; 
    color: string;
    description: string;
  }) => (
    <div className="glass-card-elevated p-3 rounded-lg hover:border-[var(--border-secondary)] transition-all group">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-[var(--text-muted)] uppercase tracking-wider">{label}</span>
        <Icon className={`w-4 h-4 ${color}`} />
      </div>
      <div className={`text-xl font-mono font-bold ${color}`}>
        {value >= 0 ? '+' : ''}{value.toFixed(4)}{unit}
      </div>
      <div className="text-xs text-[var(--text-muted)] mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
        {description}
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Parameter Inputs */}
      <div className="glass-card p-4">
        <div className="section-header">
          <Zap className="w-4 h-4 text-[var(--accent-warning)]" />
          <h3 className="section-title">Greeks 计算器</h3>
          {loading && <RefreshCw className="w-4 h-4 animate-spin text-[var(--accent-primary)] ml-auto" />}
        </div>
        
        <div className="grid grid-cols-6 gap-4 mt-4">
          <div>
            <label className="label-mono block mb-1">标的价格</label>
            <input 
              type="number" 
              value={spot} 
              onChange={(e) => setSpot(parseFloat(e.target.value) || 0)}
              step="0.01"
              className="input-field w-full"
            />
          </div>
          <div>
            <label className="label-mono block mb-1">行权价</label>
            <input 
              type="number" 
              value={strike} 
              onChange={(e) => setStrike(parseFloat(e.target.value) || 0)}
              step="0.01"
              className="input-field w-full"
            />
          </div>
          <div>
            <label className="label-mono block mb-1">到期天数</label>
            <input 
              type="number" 
              value={tte} 
              onChange={(e) => setTte(parseInt(e.target.value) || 0)}
              className="input-field w-full"
            />
          </div>
          <div>
            <label className="label-mono block mb-1">波动率 (%)</label>
            <input 
              type="number" 
              value={vol} 
              onChange={(e) => setVol(parseFloat(e.target.value) || 0)}
              step="1"
              className="input-field w-full"
            />
          </div>
          <div>
            <label className="label-mono block mb-1">无风险利率 (%)</label>
            <input 
              type="number" 
              value={riskFree} 
              onChange={(e) => setRiskFree(parseFloat(e.target.value) || 0)}
              step="0.1"
              className="input-field w-full"
            />
          </div>
          <div>
            <label className="label-mono block mb-1">期权类型</label>
            <div className="flex bg-[var(--bg-primary)] p-1 rounded-lg border border-[var(--border-primary)]">
              <button
                onClick={() => setIsCall(true)}
                className={`flex-1 py-2 text-xs font-semibold rounded-md transition-all ${
                  isCall ? 'bg-[var(--accent-success)] text-white' : 'text-[var(--text-muted)]'
                }`}
              >
                📈 Call
              </button>
              <button
                onClick={() => setIsCall(false)}
                className={`flex-1 py-2 text-xs font-semibold rounded-md transition-all ${
                  !isCall ? 'bg-[var(--accent-danger)] text-white' : 'text-[var(--text-muted)]'
                }`}
              >
                📉 Put
              </button>
            </div>
          </div>
        </div>
      </div>

      {greeks && (
        <>
          {/* First Order Greeks */}
          <div className="glass-card p-4">
            <div className="section-header">
              <Activity className="w-4 h-4 text-[var(--accent-primary)]" />
              <h3 className="section-title">一阶希腊字母</h3>
              <span className="text-sm font-mono text-[var(--accent-primary)] ml-auto">
                理论价格: ¥{greeks.price.toFixed(4)}
              </span>
            </div>
            
            <div className="grid grid-cols-5 gap-3 mt-4">
              <GreekCard 
                label="Delta (Δ)" 
                value={greeks.delta}
                icon={TrendingUp}
                color={greeks.delta >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'}
                description="标的价格变动1单位的期权价格变化"
              />
              <GreekCard 
                label="Gamma (Γ)" 
                value={greeks.gamma}
                icon={Activity}
                color="text-[var(--accent-warning)]"
                description="Delta对标的价格的敏感度"
              />
              <GreekCard 
                label="Vega (ν)" 
                value={greeks.vega}
                icon={Zap}
                color="text-[var(--accent-primary)]"
                description="波动率变动1%的期权价格变化"
              />
              <GreekCard 
                label="Theta (Θ)" 
                value={greeks.theta}
                icon={Clock}
                color="text-[var(--accent-danger)]"
                description="每日时间价值衰减"
              />
              <GreekCard 
                label="Rho (ρ)" 
                value={greeks.rho}
                icon={Target}
                color="text-[var(--accent-secondary)]"
                description="利率变动1%的期权价格变化"
              />
            </div>
          </div>

          {/* Second Order Greeks */}
          <div className="glass-card p-4">
            <div className="section-header">
              <AlertTriangle className="w-4 h-4 text-[var(--accent-warning)]" />
              <h3 className="section-title">二阶希腊字母</h3>
            </div>
            
            <div className="grid grid-cols-5 gap-3 mt-4">
              <GreekCard 
                label="Vanna" 
                value={greeks.vanna}
                icon={Activity}
                color="text-[var(--accent-pink)]"
                description="Delta对波动率的敏感度"
              />
              <GreekCard 
                label="Volga" 
                value={greeks.volga}
                icon={Zap}
                color="text-[var(--accent-warning)]"
                description="Vega对波动率的敏感度"
              />
              <GreekCard 
                label="Charm" 
                value={greeks.charm}
                icon={Clock}
                color="text-[var(--accent-secondary)]"
                description="Delta对时间的敏感度 (每日)"
              />
              <GreekCard 
                label="Speed" 
                value={greeks.speed}
                icon={TrendingUp}
                color="text-[var(--accent-primary)]"
                description="Gamma对标的价格的敏感度"
              />
              <GreekCard 
                label="Color" 
                value={greeks.color}
                icon={Clock}
                color="text-[var(--accent-danger)]"
                description="Gamma对时间的敏感度 (每日)"
              />
            </div>
          </div>

          {/* P&L Surface */}
          <div className="glass-card p-4">
            <div className="section-header">
              <Activity className="w-4 h-4 text-[var(--accent-success)]" />
              <h3 className="section-title">P&L 敏感度曲面 (Greeks 近似)</h3>
            </div>
            <div style={{ height: '400px' }}>
              <Plot
                data={plSurfaceData}
                layout={{
                  autosize: true,
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  margin: { l: 50, r: 20, t: 30, b: 50 },
                  scene: {
                    xaxis: { 
                      title: '标的价格',
                      titlefont: { color: '#9ca3af', size: 11 },
                      tickfont: { color: '#6b7280', size: 10 },
                      gridcolor: '#2a2d35',
                    },
                    yaxis: { 
                      title: '波动率 (%)',
                      titlefont: { color: '#9ca3af', size: 11 },
                      tickfont: { color: '#6b7280', size: 10 },
                      gridcolor: '#2a2d35',
                    },
                    zaxis: { 
                      title: 'P&L',
                      titlefont: { color: '#9ca3af', size: 11 },
                      tickfont: { color: '#6b7280', size: 10 },
                      gridcolor: '#2a2d35',
                    },
                    camera: { eye: { x: 1.5, y: 1.5, z: 1.0 } }
                  }
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
