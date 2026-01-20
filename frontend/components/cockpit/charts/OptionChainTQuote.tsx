"use client";

import { useState, useEffect, useMemo } from 'react';
import { Grid3X3, TrendingUp, TrendingDown, Calendar, RefreshCw, Filter, Loader2 } from 'lucide-react';

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
}

interface ExpiryGroup {
  expiry: string;
  dte: number;
  calls: OptionContract[];
  puts: OptionContract[];
  strikes: number[];
}

interface OptionChainTQuoteProps {
  selectedDate?: string;
}

export default function OptionChainTQuote({ selectedDate }: OptionChainTQuoteProps) {
  const [spotPrice, setSpotPrice] = useState(3.0);
  const [selectedExpiry, setSelectedExpiry] = useState<string | null>(null);
  const [expiryGroups, setExpiryGroups] = useState<ExpiryGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showGreeks, setShowGreeks] = useState(true);
  const [highlightATM, setHighlightATM] = useState(true);
  const [currentDate, setCurrentDate] = useState(selectedDate || '');
  const [availableDates, setAvailableDates] = useState<string[]>([]);

  // Fetch available dates
  useEffect(() => {
    const fetchDates = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/data/dates');
        if (response.ok) {
          const data = await response.json();
          setAvailableDates(data.dates || []);
          if (!currentDate && data.dates && data.dates.length > 0) {
            setCurrentDate(data.dates[0]);
          }
        }
      } catch (err) {
        console.error('Failed to fetch dates:', err);
      }
    };
    fetchDates();
  }, []);

  // Fetch and process option chain data
  useEffect(() => {
    if (!currentDate) return;

    const fetchOptionChain = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`http://localhost:8000/api/data/assets?date=${currentDate}&limit=100`);
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        const assets = data.assets || [];
        
        // Group by expiry date
        const expiryMap = new Map<string, { calls: OptionContract[], puts: OptionContract[], strikes: Set<number> }>();
        const today = new Date(currentDate);
        
        assets.forEach((asset: any) => {
          const expiry = asset.expiry;
          if (!expiryMap.has(expiry)) {
            expiryMap.set(expiry, { calls: [], puts: [], strikes: new Set() });
          }
          
          const group = expiryMap.get(expiry)!;
          group.strikes.add(asset.strike);
          
          // Calculate DTE
          const expiryDate = new Date(expiry);
          const dte = Math.ceil((expiryDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
          
          // Transform to OptionContract format with estimated Greeks
          const sigma = asset.iv || 0.2;
          const moneyness = asset.type === 'call' 
            ? (spotPrice - asset.strike) / spotPrice 
            : (asset.strike - spotPrice) / spotPrice;
          const T = Math.max(dte, 1) / 365;
          
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
            iv: sigma,
            delta: asset.type === 'call' ? 0.3 + moneyness * 0.5 : -0.3 - moneyness * 0.5,
            gamma: 0.05 / (1 + Math.abs(moneyness) * 2),
            theta: -asset.close * 0.05 / Math.max(dte, 1),
            vega: asset.close * 0.1 * Math.sqrt(T),
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
            
            // Align calls and puts by strike
            const alignedCalls = strikes.map(k => group.calls.find(c => c.strike === k) || createEmptyContract(k, 'call', expiry, dte));
            const alignedPuts = strikes.map(k => group.puts.find(p => p.strike === k) || createEmptyContract(k, 'put', expiry, dte));
            
            return {
              expiry,
              dte,
              strikes,
              calls: alignedCalls,
              puts: alignedPuts,
            };
          })
          .filter(g => g.dte > -30)  // Include past expiries for historical data
          .sort((a, b) => a.dte - b.dte);
        
        setExpiryGroups(groups);
        if (groups.length > 0 && !selectedExpiry) {
          setSelectedExpiry(groups[0].expiry);
        }
        
        // Estimate spot price from ATM strikes
        if (groups.length > 0 && groups[0].strikes.length > 0) {
          const midStrike = groups[0].strikes[Math.floor(groups[0].strikes.length / 2)];
          setSpotPrice(midStrike);
        }
      } catch (err) {
        console.error('Failed to fetch option chain:', err);
        setError(err instanceof Error ? err.message : '加载期权链失败');
      } finally {
        setLoading(false);
      }
    };
    
    fetchOptionChain();
  }, [currentDate]);

  // Helper to create empty contract placeholder
  const createEmptyContract = (strike: number, type: 'call' | 'put', expiry: string, dte: number): OptionContract => ({
    id: `EMPTY-${strike}-${type}`,
    type,
    strike,
    expiry,
    dte,
    bid: 0,
    ask: 0,
    last: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    openInterest: 0,
    iv: 0.2,
    delta: 0,
    gamma: 0,
    theta: 0,
    vega: 0,
  });

  const currentGroup = useMemo(() => {
    return expiryGroups.find(g => g.expiry === selectedExpiry);
  }, [expiryGroups, selectedExpiry]);

  const formatPrice = (p: number) => p.toFixed(4);
  const formatPercent = (p: number) => `${p >= 0 ? '+' : ''}${p.toFixed(2)}%`;
  const formatGreek = (g: number) => g.toFixed(4);

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

      {/* Header Controls */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Grid3X3 className="w-5 h-5 text-[var(--accent-primary)]" />
              <h3 className="section-title">期权链 T型报价</h3>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-[var(--text-muted)]">标的价格:</span>
              <span className="font-mono font-bold text-[var(--accent-success)]">¥{spotPrice.toFixed(2)}</span>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input 
                type="checkbox" 
                checked={showGreeks} 
                onChange={(e) => setShowGreeks(e.target.checked)}
                className="accent-[var(--accent-primary)]"
              />
              <span className="text-[var(--text-secondary)]">显示Greeks</span>
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

      {/* T-Quote Table */}
      {currentGroup && (
        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--border-primary)]">
                  {/* Call side headers */}
                  <th colSpan={showGreeks ? 8 : 4} className="bg-[var(--accent-success)]/10 text-[var(--accent-success)] py-3 text-center font-bold">
                    📈 认购期权 (Call)
                  </th>
                  {/* Strike column */}
                  <th className="bg-[var(--bg-elevated)] text-[var(--text-primary)] py-3 px-4 text-center font-bold">
                    行权价
                  </th>
                  {/* Put side headers */}
                  <th colSpan={showGreeks ? 8 : 4} className="bg-[var(--accent-danger)]/10 text-[var(--accent-danger)] py-3 text-center font-bold">
                    📉 认沽期权 (Put)
                  </th>
                </tr>
                <tr className="border-b border-[var(--border-primary)] text-xs text-[var(--text-muted)]">
                  {/* Call columns */}
                  {showGreeks && (
                    <>
                      <th className="py-2 px-2 text-right">Delta</th>
                      <th className="py-2 px-2 text-right">Gamma</th>
                      <th className="py-2 px-2 text-right">Theta</th>
                      <th className="py-2 px-2 text-right">Vega</th>
                    </>
                  )}
                  <th className="py-2 px-2 text-right">IV</th>
                  <th className="py-2 px-2 text-right">涨跌</th>
                  <th className="py-2 px-2 text-right">买价</th>
                  <th className="py-2 px-2 text-right bg-[var(--accent-success)]/5">卖价</th>
                  
                  {/* Strike */}
                  <th className="py-2 px-4 text-center bg-[var(--bg-elevated)]">K</th>
                  
                  {/* Put columns */}
                  <th className="py-2 px-2 text-left bg-[var(--accent-danger)]/5">买价</th>
                  <th className="py-2 px-2 text-left">卖价</th>
                  <th className="py-2 px-2 text-left">涨跌</th>
                  <th className="py-2 px-2 text-left">IV</th>
                  {showGreeks && (
                    <>
                      <th className="py-2 px-2 text-left">Delta</th>
                      <th className="py-2 px-2 text-left">Gamma</th>
                      <th className="py-2 px-2 text-left">Theta</th>
                      <th className="py-2 px-2 text-left">Vega</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {currentGroup.strikes.map((strike, idx) => {
                  const call = currentGroup.calls[idx];
                  const put = currentGroup.puts[idx];
                  const isATM = highlightATM && Math.abs(strike - spotPrice) < 0.05;
                  const isITMCall = strike < spotPrice;
                  const isITMPut = strike > spotPrice;
                  
                  return (
                    <tr 
                      key={strike}
                      className={`border-b border-[var(--border-primary)]/30 hover:bg-[var(--bg-card-hover)] transition-colors ${
                        isATM ? 'bg-[var(--accent-primary)]/10' : ''
                      }`}
                    >
                      {/* Call side */}
                      {showGreeks && (
                        <>
                          <td className={`py-2 px-2 text-right font-mono text-xs ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                            {formatGreek(call.delta)}
                          </td>
                          <td className={`py-2 px-2 text-right font-mono text-xs ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                            {formatGreek(call.gamma)}
                          </td>
                          <td className={`py-2 px-2 text-right font-mono text-xs text-[var(--accent-danger)] ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                            {formatGreek(call.theta)}
                          </td>
                          <td className={`py-2 px-2 text-right font-mono text-xs ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                            {formatGreek(call.vega)}
                          </td>
                        </>
                      )}
                      <td className={`py-2 px-2 text-right font-mono text-xs text-[var(--accent-primary)] ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                        {(call.iv * 100).toFixed(1)}%
                      </td>
                      <td className={`py-2 px-2 text-right font-mono text-xs ${call.changePercent >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'} ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                        {formatPercent(call.changePercent)}
                      </td>
                      <td className={`py-2 px-2 text-right font-mono ${isITMCall ? 'bg-[var(--accent-success)]/5' : ''}`}>
                        {formatPrice(call.bid)}
                      </td>
                      <td className={`py-2 px-2 text-right font-mono font-bold text-[var(--accent-success)] ${isITMCall ? 'bg-[var(--accent-success)]/10' : 'bg-[var(--accent-success)]/5'}`}>
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
                      <td className={`py-2 px-2 text-left font-mono font-bold text-[var(--accent-danger)] ${isITMPut ? 'bg-[var(--accent-danger)]/10' : 'bg-[var(--accent-danger)]/5'}`}>
                        {formatPrice(put.bid)}
                      </td>
                      <td className={`py-2 px-2 text-left font-mono ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                        {formatPrice(put.ask)}
                      </td>
                      <td className={`py-2 px-2 text-left font-mono text-xs ${put.changePercent >= 0 ? 'text-[var(--accent-success)]' : 'text-[var(--accent-danger)]'} ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                        {formatPercent(put.changePercent)}
                      </td>
                      <td className={`py-2 px-2 text-left font-mono text-xs text-[var(--accent-primary)] ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                        {(put.iv * 100).toFixed(1)}%
                      </td>
                      {showGreeks && (
                        <>
                          <td className={`py-2 px-2 text-left font-mono text-xs ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                            {formatGreek(put.delta)}
                          </td>
                          <td className={`py-2 px-2 text-left font-mono text-xs ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                            {formatGreek(put.gamma)}
                          </td>
                          <td className={`py-2 px-2 text-left font-mono text-xs text-[var(--accent-danger)] ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                            {formatGreek(put.theta)}
                          </td>
                          <td className={`py-2 px-2 text-left font-mono text-xs ${isITMPut ? 'bg-[var(--accent-danger)]/5' : ''}`}>
                            {formatGreek(put.vega)}
                          </td>
                        </>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

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
          </div>
          <div>
            💡 点击期权合约查看详情和交易
          </div>
        </div>
      </div>
    </div>
  );
}
