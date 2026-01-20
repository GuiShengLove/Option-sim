"use client";

import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';

const data = [
  { name: 'Equity Options', value: 45000, color: '#6366f1' },
  { name: 'Index Options', value: 25000, color: '#d946ef' },
  { name: 'Cash', value: 30000, color: '#10b981' },
  { name: 'Margin Req', value: 15000, color: '#f59e0b' },
];

export default function MultiAssetAllocation() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={80}
          paddingAngle={5}
          dataKey="value"
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} stroke="none" />
          ))}
        </Pie>
        <Tooltip 
             contentStyle={{ backgroundColor: '#161920', border: '1px solid #2a2e37' }}
             itemStyle={{ color: '#fff' }}
             formatter={(value) => `$${value}`}
        />
        <Legend verticalAlign="middle" align="right" layout="vertical" iconType="circle" />
      </PieChart>
    </ResponsiveContainer>
  );
}
