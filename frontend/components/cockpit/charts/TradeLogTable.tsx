import React, { useState } from 'react';
import { ArrowUpRight, ArrowDownLeft, Filter, Download } from 'lucide-react';

interface TradeRecord {
  date: string;
  symbol: string;
  action: string;
  quantity: number;
  price: number;
  fee: number;
  realized_pnl: number;
}

interface TradeLogTableProps {
  trades?: TradeRecord[];
}

export default function TradeLogTable({ trades = [] }: TradeLogTableProps) {
  const [filter, setFilter] = useState('');
  const [sortField, setSortField] = useState<keyof TradeRecord>('date');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  if (!trades || trades.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-[var(--text-muted)] space-y-2">
        <p>暂无交易记录</p>
      </div>
    );
  }

  // Filter & Sort
  const filteredTrades = trades.filter(t => 
    t.symbol.toLowerCase().includes(filter.toLowerCase()) || 
    t.date.includes(filter)
  ).sort((a, b) => {
    const va = a[sortField];
    const vb = b[sortField];
    if (va < vb) return sortDir === 'asc' ? -1 : 1;
    if (va > vb) return sortDir === 'asc' ? 1 : -1;
    return 0;
  });

  const handleSort = (field: keyof TradeRecord) => {
    if (sortField === field) {
      setSortDir(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDir('desc');
    }
  };

  const downloadCSV = () => {
    if (!trades.length) return;
    const headers = Object.keys(trades[0]).join(',');
    const rows = trades.map(t => Object.values(t).join(','));
    const csvContent = [headers, ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `trade_log_${new Date().toISOString().slice(0,10)}.csv`;
    link.click();
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Toolbar */}
      <div className="flex justify-between items-center">
        <div className="relative">
          <Filter className="absolute left-2 top-2.5 w-4 h-4 text-[var(--text-muted)]" />
          <input 
            type="text" 
            placeholder="过滤代码/日期..." 
            value={filter}
            onChange={e => setFilter(e.target.value)}
            className="input-field pl-8 w-64 text-sm"
          />
        </div>
        <button 
          onClick={downloadCSV}
          className="btn-ghost flex items-center gap-2 text-xs"
        >
          <Download className="w-4 h-4" />
          导出CSV
        </button>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto rounded-lg border border-[var(--border-primary)] bg-[var(--bg-card)]">
        <table className="w-full text-sm text-left">
          <thead className="bg-[var(--bg-secondary)] sticky top-0 z-10 text-[var(--text-secondary)]">
            <tr>
              {['date', 'symbol', 'action', 'quantity', 'price', 'fee', 'realized_pnl'].map(field => (
                <th 
                  key={field} 
                  onClick={() => handleSort(field as keyof TradeRecord)}
                  className="px-4 py-3 cursor-pointer hover:bg-[var(--bg-primary)] transition-colors font-mono"
                >
                  <div className="flex items-center gap-1">
                    {field === 'date' ? '日期' : 
                     field === 'symbol' ? '合约代码' :
                     field === 'action' ? '方向' :
                     field === 'quantity' ? '数量' :
                     field === 'price' ? '价格' :
                     field === 'fee' ? '手续费' : '实现盈亏'}
                    {sortField === field && (
                      <span className="text-[var(--accent-primary)]">{sortDir === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-primary)]">
            {filteredTrades.map((trade, idx) => (
              <tr key={idx} className="hover:bg-[var(--bg-secondary)]/50 transition-colors">
                <td className="px-4 py-2 font-mono text-[var(--text-muted)]">{trade.date}</td>
                <td className="px-4 py-2 font-mono text-[var(--accent-primary)]">{trade.symbol}</td>
                <td className="px-4 py-2">
                  <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                    trade.action.includes('BUY') || trade.action === 'LONG' 
                    ? 'bg-[var(--accent-success)]/20 text-[var(--accent-success)]' 
                    : 'bg-[var(--accent-danger)]/20 text-[var(--accent-danger)]'
                  }`}>
                    {trade.action}
                  </span>
                </td>
                <td className="px-4 py-2 font-mono text-right">{trade.quantity}</td>
                <td className="px-4 py-2 font-mono text-right text-[var(--text-primary)]">¥{trade.price.toFixed(4)}</td>
                <td className="px-4 py-2 font-mono text-right text-[var(--text-muted)]">¥{trade.fee.toFixed(2)}</td>
                <td className={`px-4 py-2 font-mono text-right font-bold ${
                  trade.realized_pnl > 0 ? 'text-[var(--accent-success)]' : 
                  trade.realized_pnl < 0 ? 'text-[var(--accent-danger)]' : 'text-[var(--text-muted)]'
                }`}>
                  {trade.realized_pnl !== 0 ? `¥${trade.realized_pnl.toFixed(2)}` : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="text-right text-xs text-[var(--text-muted)]">
        共显示 {filteredTrades.length} 条记录
      </div>
    </div>
  );
}
