"use client";

import { useState } from 'react';
import FileExplorer from '@/components/foundry/FileExplorer';
import CodeEditor from '@/components/foundry/CodeEditor';
import AICopilot from '@/components/foundry/AICopilot';
import ModelDataManager from '@/components/foundry/ModelDataManager';
import { Code2, Database } from 'lucide-react';

interface StrategyFile {
  id: string;
  name: string;
  content: string;
  last_modified?: string;
}

type LeftPanelMode = 'strategies' | 'models';

export default function FoundryPage() {
  const [selectedFile, setSelectedFile] = useState<StrategyFile | undefined>(undefined);
  const [currentCode, setCurrentCode] = useState<string>('');
  const [leftPanelMode, setLeftPanelMode] = useState<LeftPanelMode>('strategies');

  const handleFileSelect = (file: StrategyFile) => {
    setSelectedFile(file);
    setCurrentCode(file.content);
  };

  const handleFileSave = (file: StrategyFile) => {
    setSelectedFile(file);
    setCurrentCode(file.content);
  };

  const handleCodeChange = (code: string) => {
    setCurrentCode(code);
  };

  return (
    <div className="flex h-screen w-full">
        {/* Left Panel Tabs + Content */}
        <div className="w-64 h-full shrink-0 flex flex-col">
            {/* Panel Switcher */}
            <div className="flex border-b border-[var(--border-primary)] bg-[var(--bg-secondary)]">
                <button
                    onClick={() => setLeftPanelMode('strategies')}
                    className={`flex-1 py-2 px-3 text-xs font-medium flex items-center justify-center gap-1 transition-colors ${
                        leftPanelMode === 'strategies'
                            ? 'bg-[var(--accent-primary)]/10 text-[var(--accent-primary)] border-b-2 border-[var(--accent-primary)]'
                            : 'text-[var(--text-muted)] hover:bg-[var(--bg-card)]'
                    }`}
                >
                    <Code2 className="w-3 h-3" />
                    策略
                </button>
                <button
                    onClick={() => setLeftPanelMode('models')}
                    className={`flex-1 py-2 px-3 text-xs font-medium flex items-center justify-center gap-1 transition-colors ${
                        leftPanelMode === 'models'
                            ? 'bg-[var(--accent-primary)]/10 text-[var(--accent-primary)] border-b-2 border-[var(--accent-primary)]'
                            : 'text-[var(--text-muted)] hover:bg-[var(--bg-card)]'
                    }`}
                >
                    <Database className="w-3 h-3" />
                    模型/数据
                </button>
            </div>
            
            {/* Panel Content */}
            <div className="flex-1 overflow-hidden">
                {leftPanelMode === 'strategies' ? (
                    <FileExplorer 
                        onFileSelect={handleFileSelect}
                        selectedFileId={selectedFile?.id}
                    />
                ) : (
                    <ModelDataManager />
                )}
            </div>
        </div>
        
        {/* Center: Editor */}
        <div className="flex-1 h-full min-w-0">
            <CodeEditor 
              selectedFile={selectedFile}
              onSave={handleFileSave}
              onCodeChange={handleCodeChange}
            />
        </div>
        
        {/* Right: Copilot - Now receives current code */}
        <div className="shrink-0 h-full">
            <AICopilot
              currentCode={currentCode}
              currentFileName={selectedFile?.name}
            />
        </div>
    </div>
  );
}
