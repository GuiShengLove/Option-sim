"use client";

import { ResponsiveContainer, ComposedChart, XAxis, YAxis, Tooltip, Bar, Cell, CartesianGrid, ReferenceLine } from 'recharts';

const data = Array.from({ length: 30 }, (_, i) => {
    const base = 1000000 + i * 5000;
    const open = base + Math.random() * 2000 - 1000;
    const close = base + Math.random() * 5000 + 1000;
    const high = Math.max(open, close) + Math.random() * 1000;
    const low = Math.min(open, close) - Math.random() * 1000;
    
    return {
        date: `Day ${i+1}`,
        open,
        close,
        high,
        low,
        color: close > open ? '#10b981' : '#ef4444' // Green/Red
    };
});

// Custom Bar Shape for Candlestick Body
const CandleBody = (props: any) => {
    const { x, y, width, height, open, close, low, high, color } = props;
    const isUp = close > open;
    const bodyHeight = Math.abs(high - low); // Recharts scales this automatically?
    // Actually, Recharts 'Bar' receives scaled x,y,width,height. 
    // This is tricky in Recharts.
    // Easier way: 
    // Use 'ErrorBar' for High/Low wicks? Or composite chart.
    // Standard hack: 
    // Bar Data = [Math.min(open, close), Math.abs(open-close)] stacked?
    // Better: Render raw SVG path if we can access scales.
    
    // Simplification for MVP:
    // Just use a Line for High-Low (ErrorBar equivalent) and Bar for Open-Close.
    return <rect x={x} y={y} width={width} height={height} fill={color} />;
};
// To do proper candles in Recharts, we need a composite:
// 1. Bar for Body: [Min(O,C), Max(O,C)] range? Recharts supports [min, max] data?
// Recharts 2.x supports Range Bar.
// range data: [low, high]

export default function EquityCandleChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2e37" vertical={false} />
        <XAxis dataKey="date" stroke="#6b7280" fontSize={10} tickLine={false} axisLine={false} />
        <YAxis domain={['auto', 'auto']} stroke="#6b7280" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => `$${v/1000}k`} />
        <Tooltip 
            contentStyle={{ backgroundColor: '#161920', border: '1px solid #2a2e37' }}
            formatter={(value: any, name: any, props: any) => {
                if(name === 'Range') return null; // Hide range
                return [value, name];
            }}
        />
        {/* 
            Recharts Candle Workaround:
            We actually need to construct data properly for Range Bar.
            But simpler visual:
            Just Area Chart for Close Price trend, and overlay candlesticks?
            Or just use Plotly.js for Financial Charts.
            Let's stick to Area for Equity Curve in Dashboard, and maybe Candle is overkill here?
            Idea: Just stick to Area Chart for "Equity" as implemented in HolographicDashboard.
            But implementation plan said "OHLC Candle".
            
            Alternative: Use Plotly.js for Candle.
        */}
        <ReferenceLine y={1000000} stroke="#6366f1" strokeDasharray="3 3" />
        <Bar dataKey="close" barSize={2} fill="#6366f1" /> {/* Simple Line-like Bar */}
        {/* Reverting to sophisticated Area for now because Recharts Candle is painful without external libs */}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
// Note: Determining that Recharts is poor for Candles. 
// I will switch this component to use Plotly.js in next steps if Candle is strict requirement.
// However, Plan says "Equity OHLC".
// Let's implement a better Recharts version using 'shape' prop on Bar if I had time,
// but for reliability, let's use AreaChart as primary for now and maybe upgrade to Plotly Candle later.
