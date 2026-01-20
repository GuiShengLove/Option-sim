"use client";

import { useState, useEffect } from 'react';
import { Send, Sparkles, User, Loader2, Code, Eye, EyeOff } from 'lucide-react';

type Message = {
  role: 'user' | 'ai';
  content: string;
};

interface AICopilotProps {
  currentCode?: string;
  currentFileName?: string;
}

export default function AICopilot({ currentCode, currentFileName }: AICopilotProps) {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'ai', content: '你好！我是 AI 策略助手。我可以帮你编写期权策略代码、优化参数，或者解释量化模型。有什么可以帮助你的吗？' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [includeCode, setIncludeCode] = useState(true);

  // Notify when code context is available
  useEffect(() => {
    if (currentCode && currentFileName) {
      setMessages(prev => {
        // Check if we already notified about this file
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.content.includes(currentFileName)) return prev;
        
        return [...prev, {
          role: 'ai',
          content: `📝 已加载代码: **${currentFileName}** (${currentCode.length}字符)\n\n现在我可以看到你的代码了！你可以问我关于这个策略的问题，比如：\n- 帮我分析策略逻辑\n- 建议优化方案\n- 解释某个函数`
        }];
      });
    }
  }, [currentFileName]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    
    const userMessage = input.trim();
    const newUserMessage: Message = { role: 'user', content: userMessage };
    setMessages(prev => [...prev, newUserMessage]);
    setInput('');
    setLoading(true);
    
    try {
      // Build context with current code if available and enabled
      let context = '用户正在策略工坊中开发期权量化策略。';
      
      if (includeCode && currentCode && currentFileName) {
        context = `用户正在编辑策略文件: ${currentFileName}

当前代码内容:
\`\`\`python
${currentCode.slice(0, 3000)}${currentCode.length > 3000 ? '\n... (代码过长已截断)' : ''}
\`\`\`

请根据上述代码上下文回答用户的问题。`;
      }
      
      // Call real AI API with code context
      const response = await fetch('http://localhost:8000/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          context: context
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        const aiResponse: Message = {
          role: 'ai',
          content: data.response || '抱歉，我暂时无法处理这个请求。'
        };
        setMessages(prev => [...prev, aiResponse]);
      } else {
        throw new Error('API request failed');
      }
    } catch (error) {
      console.error('AI Chat Error:', error);
      const errorResponse: Message = {
        role: 'ai',
        content: '抱歉，AI服务暂时不可用，请稍后重试。'
      };
      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-secondary)] border-l border-[var(--border-primary)] w-80">
      {/* Header */}
      <div className="p-4 border-b border-[var(--border-primary)] glass-card-elevated">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[var(--accent-primary)]" />
            <h3 className="section-title">AI 智能助手</h3>
          </div>
          {/* Code Context Toggle */}
          <button
            onClick={() => setIncludeCode(!includeCode)}
            className={`p-1.5 rounded transition-colors ${
              includeCode 
                ? 'bg-[var(--accent-primary)]/20 text-[var(--accent-primary)]' 
                : 'bg-[var(--bg-card)] text-[var(--text-muted)]'
            }`}
            title={includeCode ? '已启用代码上下文' : '未启用代码上下文'}
          >
            {includeCode ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
        </div>
        
        {/* Status */}
        <div className="flex items-center gap-2 mt-2">
          <Code className="w-3 h-3 text-[var(--text-muted)]" />
          {currentFileName ? (
            <span className="text-xs text-[var(--accent-success)]">
              {currentFileName}
            </span>
          ) : (
            <span className="text-xs text-[var(--text-muted)]">
              未选择文件
            </span>
          )}
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className="flex gap-3">
            <div className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
              msg.role === 'ai' 
                ? 'bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-pink)]' 
                : 'bg-[var(--bg-elevated)]'
            }`}>
              {msg.role === 'ai' ? (
                <Sparkles className="w-4 h-4 text-white" />
              ) : (
                <User className="w-4 h-4 text-[var(--text-muted)]" />
              )}
            </div>

            <div className={`flex-1 ${
              msg.role === 'ai' 
                ? 'glass-card p-3' 
                : 'bg-[var(--bg-elevated)] p-3 rounded-lg border border-[var(--border-primary)]'
            }`}>
              <p className="text-sm text-[var(--text-primary)] leading-relaxed whitespace-pre-wrap">
                {msg.content}
              </p>
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="flex gap-3">
            <div className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-pink)]">
              <Loader2 className="w-4 h-4 text-white animate-spin" />
            </div>
            <div className="glass-card p-3 flex-1">
              <p className="text-sm text-[var(--text-muted)]">AI正在思考...</p>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-[var(--border-primary)] bg-[var(--bg-primary)]">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder={currentCode ? "问我关于这段代码的问题..." : "咨询 AI 助手..."}
            className="input-field flex-1"
          />
          <button
            onClick={handleSend}
            disabled={loading}
            className="btn-primary px-4 py-2 shrink-0 disabled:opacity-50"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </div>
  );
}
