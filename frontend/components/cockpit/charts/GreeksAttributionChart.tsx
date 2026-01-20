"use client";

import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

// Mock data: Greeks exposure over time
const data = Array.from({ length: 30 }, (_, i) => {
  return {
    day: i + 1,
    delta: 5000 + Math.sin(i / 4) * 3000 + Math.random() * 1000,
    gamma: 2000 + Math.cos(i / 3) * 1000 + Math.random() * 500,
    vega: 3500 + Math.sin(i / 5) * 1500 + Math.random() * 800,
    theta: 1200 + Math.random() * 300,
  };
});

export default function GreeksAttributionChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data}>
        <defs>
          <linearGradient id="deltaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#10b981" stopOpacity={0.6}/>
            <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
          </linearGradient>
          <linearGradient id="gammaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.6}/>
            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.1}/>
          </linearGradient>
          <linearGradient id="vegaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.6}/>
            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.1}/>
          </linearGradient>
          <linearGradient id="thetaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6}/>
            <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1}/>
          </linearGradient>
        </defs>
        
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2d35" vertical={false} />
        
        <XAxis 
          dataKey="day" 
          stroke="#6b7280" 
          fontSize={11} 
          tickLine={false} 
          axisLine={false}
          label={{ value: 'Day', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 11 }}
        />
        
        <YAxis 
          stroke="#6b7280" 
          fontSize={11} 
          tickLine={false} 
          axisLine={false}
          tickFormatter={(v) => `$${(v/1000).toFixed(0)}K`}
        />
        
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#161920', 
            border: '1px solid #2a2d35',
            borderRadius: '8px',
            padding: '8px 12px'
          }}
          labelStyle={{ color: '#9ca3af', fontSize: 11 }}
          formatter={(value: any) => [`$${(value/1000).toFixed(1)}K`, '']}
        />
        
        <Legend 
          wrapperStyle={{ fontSize: 12, color: '#9ca3af' }}
          iconType="circle"
        />
        
        <Area 
          type="monotone" 
          dataKey="theta" 
          stackId="1"
          stroke="#ef4444" 
          fill="url(#thetaGrad)" 
          name="Theta (时间价值损耗)"
        />
        
        <Area 
          type="monotone" 
          dataKey="vega" 
          stackId="1"
          stroke="#8b5cf6" 
          fill="url(#vegaGrad)" 
          name="Vega (波动率敏感度)"
        />
        
        <Area 
          type="monotone" 
          dataKey="gamma" 
          stackId="1"
          stroke="#f59e0b" 
          fill="url(#gammaGrad)" 
          name="Gamma (Delta变化率)"
        />
        
        <Area 
          type="monotone" 
          dataKey="delta" 
          stackId="1"
          stroke="#10b981" 
          fill="url(#deltaGrad)" 
          name="Delta (方向敞口)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
