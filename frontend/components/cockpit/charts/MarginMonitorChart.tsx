"use client";

import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, Legend } from 'recharts';

const data = Array.from({ length: 30 }, (_, i) => ({
    date: `Day ${i+1}`,
    utilization: 30 + Math.random() * 20 + (i > 25 ? i * 2 : 0), // Spike at end
    leverage: 1.5 + Math.random() * 0.5,
    limit: 100
}));

export default function MarginMonitorChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data}>
        <defs>
          <linearGradient id="colorUtil" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8}/>
            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2e37" vertical={false} />
        <XAxis dataKey="date" stroke="#6b7280" fontSize={10} tickLine={false} axisLine={false} />
        <YAxis yAxisId="left" stroke="#f59e0b" fontSize={10} tickLine={false} axisLine={false} unit="%" />
        <YAxis yAxisId="right" orientation="right" stroke="#10b981" fontSize={10} tickLine={false} axisLine={false} />
        
        <Tooltip 
            contentStyle={{ backgroundColor: '#161920', border: '1px solid #2a2e37' }}
        />
        
        <ReferenceLine yAxisId="left" y={80} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'DANGER', fill: '#ef4444', fontSize: 10 }} />
        
        <Area yAxisId="left" type="monotone" dataKey="utilization" stroke="#f59e0b" fill="url(#colorUtil)" name="Margin Util %" />
        <Area yAxisId="right" type="step" dataKey="leverage" stroke="#10b981" fill="transparent" strokeWidth={2} name="Leverage (x)" />
        
        <Legend />
      </AreaChart>
    </ResponsiveContainer>
  );
}
