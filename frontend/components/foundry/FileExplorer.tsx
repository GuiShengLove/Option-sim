"use client";

import { useState, useEffect } from 'react';
import { Folder, File, ChevronRight, ChevronDown, Code2, Database, Brain, Plus, Upload, Loader2, Trash2, RefreshCw, FolderPlus, MoreVertical } from 'lucide-react';

type StrategyFile = {
  id: string;
  name: string;
  content: string;
  last_modified: string;
};

type Template = {
  id: string;
  name: string;
  chinese_name: string;
  type: string;
  risk_level: number;
  description: string;
};

type FileNode = {
  id: string;
  name: string;
  type: 'file' | 'folder';
  path?: string; // Full relative path for folders
  content?: string;
  children?: FileNode[];
};

interface FileExplorerProps {
  onFileSelect?: (file: StrategyFile) => void;
  selectedFileId?: string;
}

export default function FileExplorer({ onFileSelect, selectedFileId }: FileExplorerProps) {
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNewDialog, setShowNewDialog] = useState(false);
  const [showNewFolderDialog, setShowNewFolderDialog] = useState(false);
  const [showTemplateDialog, setShowTemplateDialog] = useState(false);
  const [newFileName, setNewFileName] = useState('');
  const [newFolderName, setNewFolderName] = useState('');
  const [targetFolderPath, setTargetFolderPath] = useState(''); // For creating inside a folder
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['strategies']));
  const [contextMenu, setContextMenu] = useState<{x: number, y: number, node: FileNode} | null>(null);

  // Fetch strategies from API
  const fetchStrategies = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/strategies');
      if (response.ok) {
        const data = await response.json();
        const strategies = data.items || [];
        
      // Build file tree from relative paths
      const buildFileTree = (items: StrategyFile[]): FileNode[] => {
        const root: FileNode[] = [];
        
        items.forEach(item => {
          const parts = item.name.split('/');
          let currentLevel = root;
          let currentPath = '';
          
          parts.forEach((part, index) => {
            const isFile = index === parts.length - 1;
            currentPath = currentPath ? `${currentPath}/${part}` : part;
            
            // Check if node exists at current level
            let node = currentLevel.find(n => n.name === part);
            
            if (!node) {
              if (isFile) {
                node = {
                  id: item.id,
                  name: part,
                  type: 'file',
                  path: currentPath,
                  content: item.content
                };
                currentLevel.push(node);
              } else {
                node = {
                  id: `folder-${currentPath}`,
                  name: part,
                  type: 'folder',
                  path: currentPath,
                  children: []
                };
                currentLevel.push(node);
              }
            }
            
            if (!isFile && node.children) {
              currentLevel = node.children;
            }
          });
        });
        
        // Sort: Folders first, then files
        const sortNodes = (nodes: FileNode[]) => {
          nodes.sort((a, b) => {
            if (a.type === b.type) return a.name.localeCompare(b.name);
            return a.type === 'folder' ? -1 : 1;
          });
          nodes.forEach(n => {
            if (n.children) sortNodes(n.children);
          });
        };
        
        sortNodes(root);
        return root;
      };

      const strategyNodes = buildFileTree(strategies);

        // Build file tree
        const tree: FileNode[] = [
          {
            id: 'strategies',
            name: '策略文件',
            type: 'folder',
            path: '',
            children: strategyNodes
          },
          {
            id: 'templates',
            name: '策略模板',
            type: 'folder',
            path: '',
            children: [] // Will be populated separately
          }
        ];
        
        setFileTree(tree);
      }
    } catch (error) {
      console.error('Failed to fetch strategies:', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch templates
  const fetchTemplates = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/strategies/templates');
      if (response.ok) {
        const data = await response.json();
        setTemplates(data.templates || []);
      }
    } catch (error) {
      console.error('Failed to fetch templates:', error);
    }
  };

  useEffect(() => {
    fetchStrategies();
    fetchTemplates();
  }, []);

  // Close context menu when clicking outside
  useEffect(() => {
    const handleClick = () => setContextMenu(null);
    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, []);

  // Create new strategy
  const createNewStrategy = async () => {
    if (!newFileName.trim()) return;
    
    let filename = newFileName.endsWith('.py') ? newFileName : `${newFileName}.py`;
    // If creating in a subfolder, prepend the path
    if (targetFolderPath) {
      filename = `${targetFolderPath}/${filename}`;
    }
    
    const defaultContent = `"""
${newFileName}
自定义策略
"""

from typing import Dict, Any


class MyStrategy:
    """自定义策略类"""
    
    def __init__(self):
        self.positions = []
        
    def on_init(self, context: Dict[str, Any]):
        """策略初始化"""
        print("策略初始化完成")
        
    def on_bar(self, context: Dict[str, Any], data: Dict[str, Any]):
        """每根K线触发"""
        spot_price = data.get('spot_price', 0)
        # TODO: 添加交易逻辑
        pass
        
    def get_greeks(self) -> Dict[str, float]:
        """获取组合Greeks"""
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}


STRATEGY_META = {
    'name': '${newFileName.replace('.py', '')}',
    'display_name': '自定义策略',
    'type': 'custom',
    'risk_level': 3,
    'description': '自定义策略描述'
}
`;

    try {
      const response = await fetch('http://localhost:8000/api/strategies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: filename, content: defaultContent })
      });
      
      if (response.ok) {
        setShowNewDialog(false);
        setNewFileName('');
        setTargetFolderPath('');
        fetchStrategies();
      }
    } catch (error) {
      console.error('Failed to create strategy:', error);
    }
  };

  // Create new folder
  const createNewFolder = async () => {
    if (!newFolderName.trim()) return;
    
    let folderPath = newFolderName.trim();
    // If creating inside another folder
    if (targetFolderPath) {
      folderPath = `${targetFolderPath}/${folderPath}`;
    }
    
    try {
      const response = await fetch(`http://localhost:8000/api/strategies/folders?folder_name=${encodeURIComponent(folderPath)}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setShowNewFolderDialog(false);
        setNewFolderName('');
        setTargetFolderPath('');
        fetchStrategies();
      } else {
        const error = await response.json();
        alert(error.detail || '创建文件夹失败');
      }
    } catch (error) {
      console.error('Failed to create folder:', error);
    }
  };

  // Create from template
  const createFromTemplate = async (templateId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/strategies/from-template/${templateId}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setShowTemplateDialog(false);
        fetchStrategies();
      }
    } catch (error) {
      console.error('Failed to create from template:', error);
    }
  };

  // Delete strategy
  const deleteStrategy = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('确定删除此策略？')) return;
    
    try {
      const response = await fetch(`http://localhost:8000/api/strategies/${id}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        fetchStrategies();
      }
    } catch (error) {
      console.error('Failed to delete strategy:', error);
    }
  };

  // Delete folder
  const deleteFolder = async (folderPath: string) => {
    if (!confirm(`确定删除文件夹 "${folderPath}" 及其所有内容？`)) return;
    
    try {
      const response = await fetch(`http://localhost:8000/api/strategies/folders/${encodeURIComponent(folderPath)}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        fetchStrategies();
      } else {
        const error = await response.json();
        alert(error.detail || '删除文件夹失败');
      }
    } catch (error) {
      console.error('Failed to delete folder:', error);
    }
  };

  // Handle file click
  const handleFileClick = (node: FileNode) => {
    if (node.type === 'file' && onFileSelect) {
      onFileSelect({
        id: node.id,
        name: node.name,
        content: node.content || '',
        last_modified: ''
      });
    }
  };

  // Toggle folder
  const toggleFolder = (id: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // Right click context menu
  const handleContextMenu = (e: React.MouseEvent, node: FileNode) => {
    e.preventDefault();
    e.stopPropagation();
    setContextMenu({ x: e.clientX, y: e.clientY, node });
  };

  const getIcon = (node: FileNode) => {
    if (node.name.includes('策略')) return <Code2 className="w-4 h-4" />;
    if (node.name.includes('模板')) return <Brain className="w-4 h-4" />;
    if (node.type === 'folder') return <Folder className="w-4 h-4" />;
    return <File className="w-4 h-4" />;
  };

  const renderNode = (node: FileNode, depth: number = 0) => {
    const isFolder = node.type === 'folder';
    const isExpanded = expandedFolders.has(node.id);
    const isSelected = node.id === selectedFileId;
    const isRootFolder = node.id === 'strategies' || node.id === 'templates';

    return (
      <div key={node.id} className="group">
        <div
          className={`flex items-center gap-2 px-3 py-2 cursor-pointer rounded-lg transition-colors ${
            isSelected ? 'bg-[var(--accent-primary)]/20' : 'hover:bg-[var(--bg-card-hover)]'
          }`}
          style={{ paddingLeft: `${depth * 12 + 12}px` }}
          onClick={() => isFolder ? toggleFolder(node.id) : handleFileClick(node)}
          onContextMenu={(e) => handleContextMenu(e, node)}
        >
          {isFolder && (
            <span className="text-[var(--text-muted)]">
              {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            </span>
          )}
          <span className={isFolder ? "text-[var(--accent-primary)]" : "text-[var(--text-muted)]"}>
            {getIcon(node)}
          </span>
          <span className={`text-sm flex-1 ${isFolder ? 'font-semibold text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}`}>
            {node.name}
          </span>
          
          {/* Action buttons */}
          {isFolder && !isRootFolder && node.id !== 'templates' && (
            <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  setTargetFolderPath(node.path || '');
                  setShowNewDialog(true);
                }}
                className="hover:text-[var(--accent-primary)] p-1"
                title="在此文件夹中新建文件"
              >
                <Plus className="w-3 h-3" />
              </button>
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  setTargetFolderPath(node.path || '');
                  setShowNewFolderDialog(true);
                }}
                className="hover:text-[var(--accent-primary)] p-1"
                title="在此文件夹中新建子文件夹"
              >
                <FolderPlus className="w-3 h-3" />
              </button>
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  if (node.path) deleteFolder(node.path);
                }}
                className="hover:text-red-500 p-1"
                title="删除文件夹"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          )}
          
          {/* Root strategies folder special actions */}
          {node.id === 'strategies' && (
            <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  setTargetFolderPath('');
                  setShowNewFolderDialog(true);
                }}
                className="hover:text-[var(--accent-primary)] p-1"
                title="新建文件夹"
              >
                <FolderPlus className="w-3 h-3" />
              </button>
            </div>
          )}
          
          {!isFolder && (
            <button 
              onClick={(e) => deleteStrategy(node.id, e)}
              className="opacity-0 group-hover:opacity-100 hover:text-red-500 transition-opacity p-1"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          )}
        </div>

        {isFolder && isExpanded && node.children && (
          <div>
            {node.children.map(child => renderNode(child, depth + 1))}
            {node.children.length === 0 && (
              <div className="text-xs text-[var(--text-muted)] py-2" style={{ paddingLeft: `${(depth + 1) * 12 + 24}px` }}>
                (空)
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-[var(--bg-secondary)]">
        <Loader2 className="w-6 h-6 animate-spin text-[var(--accent-primary)]" />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-[var(--bg-secondary)] border-r border-[var(--border-primary)]">
      {/* Header */}
      <div className="p-4 border-b border-[var(--border-primary)] glass-card-elevated">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Code2 className="w-4 h-4 text-[var(--accent-primary)]" />
            <h3 className="section-title">策略工坊</h3>
          </div>
          <button 
            onClick={() => fetchStrategies()} 
            className="p-1 hover:bg-[var(--bg-card)] rounded transition-colors"
            title="刷新"
          >
            <RefreshCw className="w-4 h-4 text-[var(--text-muted)]" />
          </button>
        </div>
      </div>

      {/* File Tree */}
      <div className="flex-1 overflow-y-auto p-2">
        {fileTree.map(node => renderNode(node))}
      </div>

      {/* Actions */}
      <div className="p-3 border-t border-[var(--border-primary)] space-y-2">
        <div className="flex gap-2">
          <button 
            onClick={() => {
              setTargetFolderPath('');
              setShowNewDialog(true);
            }}
            className="btn-secondary flex-1 text-sm py-2 flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" />
            新建策略
          </button>
          <button 
            onClick={() => {
              setTargetFolderPath('');
              setShowNewFolderDialog(true);
            }}
            className="btn-secondary text-sm py-2 px-3 flex items-center justify-center"
            title="新建文件夹"
          >
            <FolderPlus className="w-4 h-4" />
          </button>
        </div>
        <button 
          onClick={() => setShowTemplateDialog(true)}
          className="btn-secondary w-full text-sm py-2 flex items-center justify-center gap-2"
        >
          <Brain className="w-4 h-4" />
          从模板创建
        </button>
      </div>

      {/* New Strategy Dialog */}
      {showNewDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="glass-card p-6 w-96">
            <h3 className="text-lg font-bold mb-4">新建策略</h3>
            {targetFolderPath && (
              <p className="text-xs text-[var(--text-muted)] mb-2">
                位置: /{targetFolderPath}/
              </p>
            )}
            <input
              type="text"
              value={newFileName}
              onChange={(e) => setNewFileName(e.target.value)}
              placeholder="输入策略名称..."
              className="w-full px-3 py-2 bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded mb-4 focus:outline-none focus:border-[var(--accent-primary)]"
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && createNewStrategy()}
            />
            <div className="flex gap-2 justify-end">
              <button 
                onClick={() => {
                  setShowNewDialog(false);
                  setTargetFolderPath('');
                }}
                className="btn-secondary px-4 py-2"
              >
                取消
              </button>
              <button 
                onClick={createNewStrategy}
                className="btn-primary px-4 py-2"
              >
                创建
              </button>
            </div>
          </div>
        </div>
      )}

      {/* New Folder Dialog */}
      {showNewFolderDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="glass-card p-6 w-96">
            <h3 className="text-lg font-bold mb-4">新建文件夹</h3>
            {targetFolderPath && (
              <p className="text-xs text-[var(--text-muted)] mb-2">
                位置: /{targetFolderPath}/
              </p>
            )}
            <input
              type="text"
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              placeholder="输入文件夹名称..."
              className="w-full px-3 py-2 bg-[var(--bg-elevated)] border border-[var(--border-primary)] rounded mb-4 focus:outline-none focus:border-[var(--accent-primary)]"
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && createNewFolder()}
            />
            <div className="flex gap-2 justify-end">
              <button 
                onClick={() => {
                  setShowNewFolderDialog(false);
                  setTargetFolderPath('');
                }}
                className="btn-secondary px-4 py-2"
              >
                取消
              </button>
              <button 
                onClick={createNewFolder}
                className="btn-primary px-4 py-2"
              >
                创建
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Template Dialog */}
      {showTemplateDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="glass-card p-6 w-[500px] max-h-[80vh] overflow-auto">
            <h3 className="text-lg font-bold mb-4">选择策略模板</h3>
            <div className="space-y-2">
              {templates.map(template => (
                <div 
                  key={template.id}
                  onClick={() => createFromTemplate(template.id)}
                  className="p-3 bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] rounded-lg cursor-pointer transition-colors"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold">{template.chinese_name}</span>
                    <span className="text-xs text-[var(--accent-warning)]">
                      {'★'.repeat(template.risk_level)}{'☆'.repeat(5 - template.risk_level)}
                    </span>
                  </div>
                  <div className="text-xs text-[var(--text-muted)]">{template.description}</div>
                </div>
              ))}
            </div>
            <div className="flex justify-end mt-4">
              <button 
                onClick={() => setShowTemplateDialog(false)}
                className="btn-secondary px-4 py-2"
              >
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Context Menu */}
      {contextMenu && (
        <div 
          className="fixed bg-[var(--bg-card)] border border-[var(--border-primary)] rounded shadow-lg py-1 z-50"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          {contextMenu.node.type === 'folder' && contextMenu.node.id !== 'templates' && (
            <>
              <button 
                className="w-full px-4 py-2 text-sm text-left hover:bg-[var(--bg-card-hover)] flex items-center gap-2"
                onClick={() => {
                  setTargetFolderPath(contextMenu.node.path || '');
                  setShowNewDialog(true);
                  setContextMenu(null);
                }}
              >
                <Plus className="w-3 h-3" /> 新建策略
              </button>
              <button 
                className="w-full px-4 py-2 text-sm text-left hover:bg-[var(--bg-card-hover)] flex items-center gap-2"
                onClick={() => {
                  setTargetFolderPath(contextMenu.node.path || '');
                  setShowNewFolderDialog(true);
                  setContextMenu(null);
                }}
              >
                <FolderPlus className="w-3 h-3" /> 新建子文件夹
              </button>
              {contextMenu.node.id !== 'strategies' && (
                <button 
                  className="w-full px-4 py-2 text-sm text-left hover:bg-[var(--bg-card-hover)] flex items-center gap-2 text-red-500"
                  onClick={() => {
                    if (contextMenu.node.path) deleteFolder(contextMenu.node.path);
                    setContextMenu(null);
                  }}
                >
                  <Trash2 className="w-3 h-3" /> 删除文件夹
                </button>
              )}
            </>
          )}
          {contextMenu.node.type === 'file' && (
            <button 
              className="w-full px-4 py-2 text-sm text-left hover:bg-[var(--bg-card-hover)] flex items-center gap-2 text-red-500"
              onClick={(e) => {
                deleteStrategy(contextMenu.node.id, e);
                setContextMenu(null);
              }}
            >
              <Trash2 className="w-3 h-3" /> 删除策略
            </button>
          )}
        </div>
      )}
    </div>
  );
}
