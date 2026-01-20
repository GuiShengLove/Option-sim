"use client";

import { useState } from 'react';
import { Sparkles, Brain, ShieldCheck, AlertTriangle, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

// Initial placeholder - replaced after first API call
const INITIAL_MESSAGE = `### 🛡️ AI 风险评估

点击 **GENERATE REPORT** 按钮进行实时AI分析。

分析内容包括：
- 策略风险评估
- Greeks敞口分析
- 改进建议`;

export default function AIInsightPanel() {
  const [report, setReport] = useState(INITIAL_MESSAGE);
  const [loading, setLoading] = useState(false);
  const [riskScore, setRiskScore] = useState<number | null>(null);

  const generateReport = async () => {
    setLoading(true);
    
    try {
      // Call real AI risk analysis API
      const response = await fetch('http://localhost:8000/api/ai/analyze-risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pnl_data: { 
            delta_pnl: 0.15,
            gamma_pnl: 0.08,
            theta_pnl: -0.05,
            vega_pnl: 0.02
          },
          sharpe: 1.5,
          max_drawdown: -0.12,
          win_rate: 0.58,
          strategy_name: '当前组合'
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setReport(data.report_markdown);
        setRiskScore(data.risk_score);
      } else {
        throw new Error('API request failed');
      }
    } catch (error) {
      console.error('AI Risk Analysis Error:', error);
      setReport(`### ⚠️ 分析失败

无法连接到AI服务。请确保后端服务正在运行。

错误信息: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full bg-[#111318] border-t border-[#2a2e37] flex flex-col">
        <div className="p-3 border-b border-[#2a2e37] flex justify-between items-center">
             <div className="flex items-center gap-2 text-accent font-bold">
                <Brain className="w-4 h-4" />
                <span>AI STRATEGY AUDIT</span>
                {riskScore !== null && (
                  <span className={`ml-2 px-2 py-0.5 rounded text-xs ${
                    riskScore <= 3 ? 'bg-green-500/20 text-green-400' :
                    riskScore <= 6 ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-red-500/20 text-red-400'
                  }`}>
                    风险评分: {riskScore}/10
                  </span>
                )}
             </div>
             <button 
                onClick={generateReport}
                disabled={loading}
                className="px-3 py-1 bg-accent/10 hover:bg-accent/20 text-accent text-xs rounded border border-accent/20 transition-colors flex items-center gap-1 disabled:opacity-50"
             >
                {loading ? (
                  <>
                    <Loader2 className="w-3 h-3 animate-spin" />
                    ANALYZING...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3 h-3" />
                    GENERATE REPORT
                  </>
                )}
             </button>
        </div>
        
        <div className="flex-1 p-4 overflow-y-auto font-mono text-sm text-gray-300">
            {loading ? (
                <div className="flex flex-col items-center justify-center h-full space-y-3 opacity-50">
                    <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                    <div className="text-xs">正在调用DeepSeek AI分析...</div>
                </div>
            ) : (
                <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown
                        components={{
                            blockquote: ({node, ...props}) => (
                                <div className="border-l-4 border-primary pl-4 py-1 my-2 bg-primary/5 rounded-r">
                                    {props.children}
                                </div>
                            )
                        }}
                    >
                        {report}
                    </ReactMarkdown>
                </div>
            )}
        </div>
    </div>
  );
}

