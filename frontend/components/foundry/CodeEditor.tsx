"use client";

import Editor, { loader } from '@monaco-editor/react';
import { useState, useEffect, useRef } from 'react';
import { Save, Play, Check, AlertCircle, Loader2 } from 'lucide-react';

// Pre-load theme definition
loader.init().then(monaco => {
    monaco.editor.defineTheme('cyberpunk', {
        base: 'vs-dark',
        inherit: true,
        rules: [
            { token: 'comment', foreground: '6272a4' },
            { token: 'keyword', foreground: 'ff79c6' },
            { token: 'identifier', foreground: '8be9fd' },
            { token: 'string', foreground: 'f1fa8c' },
            { token: 'number', foreground: 'bd93f9' },
            { token: 'type', foreground: '50fa7b' },
        ],
        colors: {
            'editor.background': '#161920',
            'editor.lineHighlightBackground': '#1f242e',
            'editor.selectionBackground': '#44475a',
            'editorCursor.foreground': '#f1fa8c',
        }
    });
});

interface StrategyFile {
    id: string;
    name: string;
    content: string;
}

interface CodeEditorProps {
    selectedFile?: StrategyFile;
    onSave?: (file: StrategyFile) => void;
    onCodeChange?: (code: string) => void;
}

const DEFAULT_CODE = `"""
欢迎使用策略工坊
================

请从左侧文件浏览器选择一个策略文件，
或点击"新建策略"创建新的策略。

策略模板包含完整的代码结构和注释，
可以帮助您快速开始开发。
"""

# 从左侧选择策略开始编辑...
`;

export default function CodeEditor({ selectedFile, onSave, onCodeChange }: CodeEditorProps) {
    const [code, setCode] = useState(DEFAULT_CODE);
    const [fileName, setFileName] = useState('');
    const [saving, setSaving] = useState(false);
    const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');
    const [hasChanges, setHasChanges] = useState(false);
    const editorRef = useRef<any>(null);
    
    // Update code when file changes
    useEffect(() => {
        if (selectedFile) {
            setCode(selectedFile.content);
            setFileName(selectedFile.name);
            setHasChanges(false);
            setSaveStatus('idle');
        }
    }, [selectedFile]);
    
    // Handle code change
    const handleEditorChange = (value: string | undefined) => {
        if (value !== undefined) {
            setCode(value);
            setHasChanges(true);
            setSaveStatus('idle');
            onCodeChange?.(value);  // Notify parent of code changes
        }
    };
    
    // Handle editor mount
    const handleEditorDidMount = (editor: any) => {
        editorRef.current = editor;
        
        // Add keyboard shortcut for save (Ctrl+S)
        editor.addCommand(2097 /* Ctrl+S */, () => {
            handleSave();
        });
    };
    
    // Save file
    const handleSave = async () => {
        if (!selectedFile || !hasChanges) return;
        
        setSaving(true);
        setSaveStatus('idle');
        
        try {
            const response = await fetch(`http://localhost:8000/api/strategies/${selectedFile.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    name: selectedFile.name, 
                    content: code 
                })
            });
            
            if (response.ok) {
                setSaveStatus('success');
                setHasChanges(false);
                onSave?.({ ...selectedFile, content: code });
                
                // Reset status after 2s
                setTimeout(() => setSaveStatus('idle'), 2000);
            } else {
                setSaveStatus('error');
            }
        } catch (error) {
            console.error('Failed to save:', error);
            setSaveStatus('error');
        } finally {
            setSaving(false);
        }
    };
    
    return (
        <div className="h-full w-full flex flex-col bg-[#161920]">
            {/* Toolbar */}
            <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--border-primary)] bg-[var(--bg-secondary)]">
                <div className="flex items-center gap-3">
                    <span className="text-sm font-mono text-[var(--text-primary)]">
                        {fileName || 'No file selected'}
                    </span>
                    {hasChanges && (
                        <span className="text-xs text-[var(--accent-warning)]">• 未保存</span>
                    )}
                </div>
                
                <div className="flex items-center gap-2">
                    {/* Save Status */}
                    {saveStatus === 'success' && (
                        <span className="flex items-center gap-1 text-xs text-[var(--accent-success)]">
                            <Check className="w-3 h-3" />
                            已保存
                        </span>
                    )}
                    {saveStatus === 'error' && (
                        <span className="flex items-center gap-1 text-xs text-red-500">
                            <AlertCircle className="w-3 h-3" />
                            保存失败
                        </span>
                    )}
                    
                    {/* Save Button */}
                    <button
                        onClick={handleSave}
                        disabled={!selectedFile || !hasChanges || saving}
                        className={`flex items-center gap-1 px-3 py-1.5 rounded text-sm transition-colors ${
                            hasChanges && selectedFile
                                ? 'bg-[var(--accent-primary)] text-white hover:opacity-90'
                                : 'bg-[var(--bg-card)] text-[var(--text-muted)] cursor-not-allowed'
                        }`}
                    >
                        {saving ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <Save className="w-4 h-4" />
                        )}
                        保存
                    </button>
                    
                    {/* Run Button (placeholder) */}
                    <button
                        disabled={!selectedFile}
                        className="flex items-center gap-1 px-3 py-1.5 rounded text-sm bg-[var(--accent-success)] text-white hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
                        title="运行策略 (开发中)"
                    >
                        <Play className="w-4 h-4" />
                        运行
                    </button>
                </div>
            </div>
            
            {/* Editor */}
            <div className="flex-1">
                <Editor 
                    height="100%"
                    language="python"
                    value={code}
                    onChange={handleEditorChange}
                    onMount={handleEditorDidMount}
                    theme="cyberpunk"
                    options={{
                        minimap: { enabled: true },
                        fontSize: 14,
                        fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
                        scrollBeyondLastLine: false,
                        padding: { top: 16 },
                        lineNumbers: 'on',
                        renderLineHighlight: 'all',
                        bracketPairColorization: { enabled: true },
                        formatOnPaste: true,
                        tabSize: 4,
                        insertSpaces: true,
                        wordWrap: 'on'
                    }}
                />
            </div>
            
            {/* Status Bar */}
            <div className="flex items-center justify-between px-4 py-1 border-t border-[var(--border-primary)] bg-[var(--bg-secondary)] text-xs text-[var(--text-muted)]">
                <div className="flex items-center gap-4">
                    <span>Python</span>
                    <span>UTF-8</span>
                </div>
                <div className="flex items-center gap-4">
                    <span>Ctrl+S 保存</span>
                    {selectedFile && (
                        <span>策略ID: {selectedFile.id}</span>
                    )}
                </div>
            </div>
        </div>
    );
}
