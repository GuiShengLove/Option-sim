"use client";

import { useState, useEffect, useRef } from 'react';
import { 
  Brain, Database, Upload, Trash2, RefreshCw, Loader2, 
  ChevronDown, ChevronRight, File, Eye, X, FileText, Table, Folder, FolderPlus, Plus
} from 'lucide-react';

interface FileInfo {
  id: string;
  name: string;
  type: string;
  extension: string;
  size: number;
  size_human: string;
  modified: string;
  relative_path?: string;
}

interface YearGroup {
  year: string;
  files: FileInfo[];
  total_size: number;
  total_size_human: string;
}

interface DatasetInfo {
  id: string;
  name: string;
  type: 'PLATFORM' | 'USER';
  path: string;
  years: YearGroup[];
}

interface DataPreview {
  filename: string;
  columns: string[];
  preview: any[];
  total_rows: number;
  showing: number;
}

type TabType = 'models' | 'user_data' | 'platform_data';

export default function ModelDataManager() {
  const [activeTab, setActiveTab] = useState<TabType>('models');
  const [models, setModels] = useState<FileInfo[]>([]);
  const [modelFolders, setModelFolders] = useState<{id: string, name: string, path: string, files: FileInfo[], count: number}[]>([]);
  const [userData, setUserData] = useState<FileInfo[]>([]);
  const [userDatasets, setUserDatasets] = useState<DatasetInfo[]>([]);
  const [platformDatasets, setPlatformDatasets] = useState<DatasetInfo[]>([]);
  const [platformGroups, setPlatformGroups] = useState<YearGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [previewData, setPreviewData] = useState<DataPreview | null>(null);
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [showCreateFolder, setShowCreateFolder] = useState(false);
  const [newFolderPath, setNewFolderPath] = useState('');
  const [parentFolderPath, setParentFolderPath] = useState(''); // 父文件夹路径，为空表示根目录
  const [selectedUploadPath, setSelectedUploadPath] = useState('');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [folderType, setFolderType] = useState<'models' | 'data'>('data');
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // 右键菜单状态
  const [contextMenu, setContextMenu] = useState<{
    visible: boolean;
    x: number;
    y: number;
    type: 'folder' | 'subfolder' | 'file' | 'empty';
    path: string;
    name: string;
  } | null>(null);

  // Fetch all data
  const fetchData = async () => {
    setLoading(true);
    try {
      const [modelsRes, modelsTreeRes, dataRes, platformRes, treeRes] = await Promise.all([
        fetch('http://localhost:8000/api/files/models'),
        fetch('http://localhost:8000/api/files/models/tree'),
        fetch('http://localhost:8000/api/files/data'),
        fetch('http://localhost:8000/api/files/platform-data'),
        fetch('http://localhost:8000/api/files/data/tree')
      ]);

      if (modelsRes.ok) {
        const data = await modelsRes.json();
        setModels(data.models || []);
      }
      if (modelsTreeRes.ok) {
        const data = await modelsTreeRes.json();
        setModelFolders(data.folders || []);
        // Use tree files for root models if tree API available
        if (data.files) setModels(data.files);
      }
      if (dataRes.ok) {
        const data = await dataRes.json();
        setUserData(data.data || []);
      }
      if (platformRes.ok) {
        const data = await platformRes.json();
        setPlatformGroups(data.groups || []);
        setPlatformDatasets(data.datasets || []);
      }
      if (treeRes.ok) {
        const data = await treeRes.json();
        setUserDatasets(data.datasets || []);
      }
    } catch (error) {
      console.error('Failed to fetch files:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Toggle item expansion (dataset or year)
  const toggleItem = (key: string) => {
    setExpandedItems(prev => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  // Create folder (for both models and data)
  const handleCreateFolder = async () => {
    if (!newFolderPath.trim()) return;
    
    // 如果有父文件夹路径，拼接完整路径
    const fullPath = parentFolderPath 
      ? `${parentFolderPath}/${newFolderPath.trim()}`
      : newFolderPath.trim();
    
    const endpoint = folderType === 'models' 
      ? `http://localhost:8000/api/files/models/folder?path=${encodeURIComponent(fullPath)}`
      : `http://localhost:8000/api/files/data/folder?path=${encodeURIComponent(fullPath)}`;
    
    try {
      const response = await fetch(endpoint, { method: 'POST' });

      if (response.ok) {
        const savedPath = fullPath; // 保存用于显示
        setNewFolderPath('');
        setParentFolderPath('');
        setShowCreateFolder(false);
        fetchData();
        alert(`文件夹 ${savedPath} 创建成功！`);
      } else {
        const error = await response.json();
        alert(`创建失败: ${error.detail}`);
      }
    } catch (error) {
      console.error('Create folder failed:', error);
      alert('创建失败，请检查网络连接');
    }
  };

  // 开始在指定文件夹下创建子文件夹
  const startCreateSubfolder = (parentPath: string, type: 'models' | 'data') => {
    setFolderType(type);
    setParentFolderPath(parentPath);
    setNewFolderPath('');
    setShowCreateFolder(true);
    setContextMenu(null);
  };

  // 显示右键菜单
  const showContextMenuHandler = (
    e: React.MouseEvent, 
    type: 'folder' | 'subfolder' | 'file' | 'empty', 
    path: string = '', 
    name: string = ''
  ) => {
    e.preventDefault();
    e.stopPropagation();
    setContextMenu({
      visible: true,
      x: e.clientX,
      y: e.clientY,
      type,
      path,
      name
    });
  };

  // 关闭右键菜单
  const closeContextMenu = () => {
    setContextMenu(null);
  };

  // 直接删除文件夹（不通过 handleContextMenuAction）
  const deleteFolder = async (path: string) => {
    // 先关闭菜单
    setContextMenu(null);
    
    console.log('[deleteFolder] Path received:', path);
    
    // 稍后显示确认对话框
    requestAnimationFrame(() => {
      if (!window.confirm(`确定删除文件夹 "${path}" 及其所有内容？`)) {
        console.log('[deleteFolder] User cancelled');
        return;
      }
      
      const endpoint = `http://localhost:8000/api/files/data/folder?path=${encodeURIComponent(path)}`;
      console.log('[deleteFolder] Calling endpoint:', endpoint);
      
      fetch(endpoint, { method: 'DELETE' })
        .then(response => {
          console.log('[deleteFolder] Response status:', response.status);
          if (response.ok) {
            console.log('[deleteFolder] Success!');
            fetchData();
          } else {
            return response.json().then(error => {
              console.log('[deleteFolder] Error:', error);
              alert(`删除失败: ${error.detail}`);
            });
          }
        })
        .catch(error => {
          console.error('Delete folder failed:', error);
          alert('删除请求失败，请检查网络');
        });
    });
  };

  // 直接删除文件
  const deleteFile = async (filename: string, folderPath: string) => {
    setContextMenu(null);
    
    console.log('[deleteFile] Filename:', filename, 'FolderPath:', folderPath);
    
    requestAnimationFrame(() => {
      if (!window.confirm(`确定删除文件 "${filename}"？`)) {
        console.log('[deleteFile] User cancelled');
        return;
      }
      
      // 删除指定路径下的文件
      const fullPath = folderPath ? `${folderPath}/${filename}` : filename;
      const endpoint = `http://localhost:8000/api/files/data/${encodeURIComponent(fullPath)}`;
      console.log('[deleteFile] FullPath:', fullPath, 'Endpoint:', endpoint);
      
      fetch(endpoint, { method: 'DELETE' })
        .then(response => {
          console.log('[deleteFile] Response status:', response.status);
          if (response.ok) {
            console.log('[deleteFile] Success!');
            fetchData();
          } else {
            return response.json().then(error => {
              console.log('[deleteFile] Error:', error);
              alert(`删除失败: ${error.detail}`);
            });
          }
        })
        .catch(error => {
          console.error('Delete file failed:', error);
          alert('删除请求失败，请检查网络');
        });
    });
  };

  // 处理右键菜单操作（用于非删除操作）
  const handleContextMenuAction = (action: string) => {
    if (!contextMenu) return;
    
    const contextType = activeTab === 'models' ? 'models' : 'data';
    const menuPath = contextMenu.path;
    const menuName = contextMenu.name;
    
    closeContextMenu();

    switch (action) {
      case 'createSubfolder':
        startCreateSubfolder(menuPath, contextType);
        break;
      case 'uploadHere':
        setSelectedUploadPath(menuPath);
        setShowUploadModal(true);
        break;
      case 'createRootFolder':
        setFolderType(contextType);
        setParentFolderPath('');
        setNewFolderPath('');
        setShowCreateFolder(true);
        break;
    }
  };

  // 点击其他地方关闭右键菜单
  useEffect(() => {
    const handleClick = () => closeContextMenu();
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, []);

  // Delete folder (for models - legacy)
  const handleDeleteFolder = async (path: string, type: 'models' | 'data' = 'data') => {
    if (!confirm(`确定删除文件夹 ${path} 及其所有内容？`)) {
      return;
    }

    const endpoint = type === 'models'
      ? `http://localhost:8000/api/files/models/folder?path=${encodeURIComponent(path)}`
      : `http://localhost:8000/api/files/data/folder?path=${encodeURIComponent(path)}`;

    try {
      const response = await fetch(endpoint, { method: 'DELETE' });

      if (response.ok) {
        fetchData();
      } else {
        const error = await response.json();
        alert(`删除失败: ${error.detail}`);
      }
    } catch (error) {
      console.error('Delete folder failed:', error);
      alert('删除请求失败，请检查网络');
    }
  };

  // Upload file with target path
  const handleUploadWithPath = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_path', selectedUploadPath);
    formData.append('description', '');

    const endpoint = activeTab === 'models' 
      ? 'http://localhost:8000/api/files/models/upload'
      : 'http://localhost:8000/api/files/data/upload-to';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        fetchData();
        setShowUploadModal(false);
        setSelectedUploadPath('');
        alert(`${file.name} 上传成功！`);
      } else {
        const error = await response.json();
        alert(`上传失败: ${error.detail}`);
      }
    } catch (error) {
      console.error('Upload failed:', error);
      alert('上传失败，请检查网络连接');
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Legacy upload handler (for models)
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('description', '');

    try {
      const response = await fetch('http://localhost:8000/api/files/models/upload', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        fetchData();
        alert(`${file.name} 上传成功！`);
      } else {
        const error = await response.json();
        alert(`上传失败: ${error.detail}`);
      }
    } catch (error) {
      console.error('Upload failed:', error);
      alert('上传失败，请检查网络连接');
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Delete file
  const handleDelete = async (filename: string, type: 'models' | 'data') => {
    if (!confirm(`确定删除 ${filename}？`)) return;

    try {
      const response = await fetch(`http://localhost:8000/api/files/${type}/${filename}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        fetchData();
      }
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  // Preview data
  const handlePreview = async (filename: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/files/data/preview/${filename}?rows=10`);
      if (response.ok) {
        const data = await response.json();
        setPreviewData(data);
      }
    } catch (error) {
      console.error('Preview failed:', error);
    }
  };

  const tabs = [
    { id: 'models' as TabType, label: '模型', icon: Brain },
    { id: 'user_data' as TabType, label: '我的数据', icon: Database },
    { id: 'platform_data' as TabType, label: '平台数据', icon: FileText },
  ];

  const currentFiles = activeTab === 'models' ? models : userData;

  const acceptedFormats = activeTab === 'models' 
    ? '.pkl,.h5,.pt,.pth,.onnx,.joblib'
    : '.parquet,.csv,.xlsx,.json';

  // Render file item
  const renderFileItem = (file: FileInfo, showActions: boolean = true) => (
    <div 
      key={file.id}
      className="flex items-center gap-2 p-2 rounded-lg hover:bg-[var(--bg-card)] transition-colors group"
    >
      <File className="w-4 h-4 text-[var(--text-muted)] flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="text-sm text-[var(--text-primary)] truncate" title={file.name}>
          {file.name}
        </div>
        <div className="text-xs text-[var(--text-muted)]">
          {file.size_human}
        </div>
      </div>
      {showActions && (
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {activeTab === 'user_data' && (
            <button
              onClick={() => handlePreview(file.name)}
              className="p-1 hover:bg-[var(--bg-elevated)] rounded"
              title="预览"
            >
              <Eye className="w-3 h-3 text-[var(--text-muted)]" />
            </button>
          )}
          {activeTab !== 'platform_data' && (
            <button
              onClick={() => handleDelete(file.name, activeTab === 'models' ? 'models' : 'data')}
              className="p-1 hover:bg-[var(--bg-elevated)] rounded"
              title="删除"
            >
              <Trash2 className="w-3 h-3 text-red-500" />
            </button>
          )}
        </div>
      )}
    </div>
  );

  // Render models tree with folders
  const renderModelsTree = () => {
    if (modelFolders.length === 0 && models.length === 0) {
      return (
        <div className="text-center py-8 text-[var(--text-muted)] text-sm">
          暂无模型，点击下方上传或新建文件夹
        </div>
      );
    }

    return (
      <div className="space-y-1">
        {/* Render folders */}
        {modelFolders.map(folder => (
          <div key={folder.id}>
            <button
              onClick={() => toggleItem(`model-folder-${folder.id}`)}
              className="w-full flex items-center gap-2 p-2 rounded-lg hover:bg-[var(--bg-card)] transition-colors text-left group"
            >
              {expandedItems.has(`model-folder-${folder.id}`) ? (
                <ChevronDown className="w-4 h-4 text-[var(--accent-primary)]" />
              ) : (
                <ChevronRight className="w-4 h-4 text-[var(--accent-primary)]" />
              )}
              <Folder className="w-4 h-4 text-[var(--accent-warning)]" />
              <span className="font-semibold text-[var(--text-primary)]">{folder.name}</span>
              <span className="text-xs text-[var(--text-muted)] ml-auto">
                {folder.count}个
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); handleDeleteFolder(folder.path, 'models'); }}
                className="p-1 opacity-0 group-hover:opacity-100 hover:bg-[var(--bg-elevated)] rounded"
                title="删除文件夹"
              >
                <Trash2 className="w-3 h-3 text-red-500" />
              </button>
            </button>
            
            {expandedItems.has(`model-folder-${folder.id}`) && (
              <div className="ml-6 space-y-0.5">
                {folder.files.map(file => renderFileItem(file, true))}
                {folder.files.length === 0 && (
                  <div className="text-xs text-[var(--text-muted)] py-1 px-2">(空)</div>
                )}
              </div>
            )}
          </div>
        ))}

        {/* Render root files */}
        {models.length > 0 && modelFolders.length > 0 && (
          <div className="text-xs text-[var(--text-muted)] px-2 pt-2">根目录文件:</div>
        )}
        {models.map(file => renderFileItem(file))}
      </div>
    );
  };

  // Render user data tree (标的 -> 年份/子文件夹 -> 文件)
  const renderUserDataTree = () => {
    if (userDatasets.length === 0 && userData.length === 0) {
      return (
        <div 
          className="text-center py-8 text-[var(--text-muted)] text-sm"
          onContextMenu={(e) => showContextMenuHandler(e, 'empty', '', '')}
        >
          暂无数据，请新建标的文件夹
        </div>
      );
    }

    return (
      <div 
        className="space-y-1"
        onContextMenu={(e) => {
          // 只有点击空白处才触发
          if (e.target === e.currentTarget) {
            showContextMenuHandler(e, 'empty', '', '');
          }
        }}
      >
        {/* Render datasets (标的层级) */}
        {userDatasets.map((dataset: any) => (
          <div key={dataset.id}>
            {/* Dataset header (标的) - 右键显示菜单 */}
            <button
              onClick={() => toggleItem(`dataset-${dataset.id}`)}
              onContextMenu={(e) => showContextMenuHandler(e, 'folder', dataset.id, dataset.name)} // Level 1 Context Menu
              className="w-full flex items-center gap-2 p-2 rounded-lg hover:bg-[var(--bg-card)] transition-colors text-left"
            >
              {expandedItems.has(`dataset-${dataset.id}`) ? (
                <ChevronDown className="w-4 h-4 text-[var(--accent-primary)]" />
              ) : (
                <ChevronRight className="w-4 h-4 text-[var(--accent-primary)]" />
              )}
              <Folder className="w-4 h-4 text-[var(--accent-warning)]" />
              <span className="font-semibold text-[var(--text-primary)]">{dataset.name}</span>
              <span className="text-xs text-[var(--text-muted)] ml-auto">
                {dataset.years?.length || 0}个年份
              </span>
            </button>
            
            {/* Expanded dataset content */}
            {expandedItems.has(`dataset-${dataset.id}`) && (
              <div className="ml-4">
                {/* 子文件夹（年份等） */}
                {dataset.years?.map((year: any) => (
                  <div key={`${dataset.id}-${year.year}`}>
                    <button
                      onClick={() => toggleItem(`year-${dataset.id}-${year.year}`)}
                      onContextMenu={(e) => showContextMenuHandler(e, 'subfolder', `${dataset.id}/${year.year}`, year.year)} // Level 2 Context Menu
                      className="w-full flex items-center gap-2 p-2 rounded-lg hover:bg-[var(--bg-card)] transition-colors text-left"
                    >
                      {expandedItems.has(`year-${dataset.id}-${year.year}`) ? (
                        <ChevronDown className="w-3 h-3 text-[var(--text-muted)]" />
                      ) : (
                        <ChevronRight className="w-3 h-3 text-[var(--text-muted)]" />
                      )}
                      <Folder className="w-3 h-3 text-[var(--accent-tertiary)]" />
                      <span className="text-sm text-[var(--text-primary)]">{year.year}</span>
                      <span className="text-xs text-[var(--text-muted)] ml-auto">
                        {year.files?.length || 0}个文件
                      </span>
                    </button>
                    
                    {expandedItems.has(`year-${dataset.id}-${year.year}`) && (
                      <div className="ml-6 space-y-0.5">
                        {year.files?.length > 0 ? (
                          year.files.slice(0, 20).map((file: FileInfo) => (
                            <div 
                              key={file.id}
                              onContextMenu={(e) => showContextMenuHandler(e, 'file', `${dataset.id}/${year.year}`, file.name)}
                              className="flex items-center gap-2 p-2 rounded-lg hover:bg-[var(--bg-card)] transition-colors"
                            >
                              <File className="w-4 h-4 text-[var(--text-muted)] flex-shrink-0" />
                              <div className="flex-1 min-w-0">
                                <div className="text-sm text-[var(--text-primary)] truncate">{file.name}</div>
                                <div className="text-xs text-[var(--text-muted)]">{file.size_human}</div>
                              </div>
                            </div>
                          ))
                        ) : (
                          <div className="text-xs text-[var(--text-muted)] py-1 px-2">(空 - 右键上传数据)</div>
                        )}
                        {year.files?.length > 20 && (
                          <div className="text-xs text-[var(--text-muted)] py-1 px-2">
                            还有 {year.files.length - 20} 个文件...
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
                
                {/* 空文件夹提示 */}
                {(!dataset.years || dataset.years.length === 0) && (
                  <div className="text-xs text-[var(--text-muted)] py-2 px-6">(空标的 - 右键添加年份)</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  // Render platform data with full 3-level structure (标的 -> 年份 -> 文件)
  const renderPlatformData = () => {
    if (platformDatasets.length === 0) {
      return (
        <div className="text-center py-8 text-[var(--text-muted)] text-sm">
          暂无平台数据
        </div>
      );
    }

    return (
      <div className="space-y-1">
        {/* Render datasets (标的层级) */}
        {platformDatasets.map(dataset => (
          <div key={dataset.id}>
            {/* Dataset header */}
            <button
              onClick={() => toggleItem(`platform-dataset-${dataset.id}`)}
              className="w-full flex items-center gap-2 p-2 rounded-lg hover:bg-[var(--bg-card)] transition-colors text-left"
            >
              {expandedItems.has(`platform-dataset-${dataset.id}`) ? (
                <ChevronDown className="w-4 h-4 text-[var(--accent-primary)]" />
              ) : (
                <ChevronRight className="w-4 h-4 text-[var(--accent-primary)]" />
              )}
              <Database className="w-4 h-4 text-[var(--accent-primary)]" />
              <span className="font-semibold text-[var(--text-primary)]">{dataset.name}</span>
              <span className="text-xs text-[var(--text-muted)] ml-auto">
                {dataset.years.length}年 · {(dataset as any).total_size_human}
              </span>
            </button>
            
            {/* Expanded years */}
            {expandedItems.has(`platform-dataset-${dataset.id}`) && (
              <div className="ml-4">
                {dataset.years.map(year => (
                  <div key={`${dataset.id}-${year.year}`}>
                    {/* Year header */}
                    <button
                      onClick={() => toggleItem(`platform-year-${dataset.id}-${year.year}`)}
                      className="w-full flex items-center gap-2 p-2 rounded-lg hover:bg-[var(--bg-card)] transition-colors text-left"
                    >
                      {expandedItems.has(`platform-year-${dataset.id}-${year.year}`) ? (
                        <ChevronDown className="w-3 h-3 text-[var(--text-muted)]" />
                      ) : (
                        <ChevronRight className="w-3 h-3 text-[var(--text-muted)]" />
                      )}
                      <Folder className="w-3 h-3 text-[var(--accent-warning)]" />
                      <span className="text-sm text-[var(--text-primary)]">{year.year}年</span>
                      <span className="text-xs text-[var(--text-muted)] ml-auto">
                        {year.total_size_human}
                      </span>
                    </button>
                    
                    {/* Files */}
                    {expandedItems.has(`platform-year-${dataset.id}-${year.year}`) && (
                      <div className="ml-6 space-y-0.5">
                        {year.files.slice(0, 10).map(file => renderFileItem(file, false))}
                        {year.files.length > 10 && (
                          <div className="text-xs text-[var(--text-muted)] py-1 px-2">
                            还有 {year.files.length - 10} 个文件...
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  // Get available upload paths based on current user datasets
  const getUploadPathOptions = (): string[] => {
    const paths: string[] = [''];  // Root (根目录)
    userDatasets.forEach((dataset: any) => {
      paths.push(dataset.id);  // 标的文件夹
      dataset.years?.forEach((year: any) => {
        paths.push(`${dataset.id}/${year.year}`);  // 子文件夹
      });
    });
    return paths;
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-secondary)] border-r border-[var(--border-primary)]">
      {/* Tabs */}
      <div className="flex border-b border-[var(--border-primary)]">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-3 px-2 text-xs font-medium transition-colors flex items-center justify-center gap-1 ${
              activeTab === tab.id
                ? 'bg-[var(--accent-primary)]/10 text-[var(--accent-primary)] border-b-2 border-[var(--accent-primary)]'
                : 'text-[var(--text-muted)] hover:bg-[var(--bg-card)]'
            }`}
          >
            <tab.icon className="w-3 h-3" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* File List */}
      <div className="flex-1 overflow-y-auto p-2">
        {loading ? (
          <div className="flex items-center justify-center h-32">
            <Loader2 className="w-6 h-6 animate-spin text-[var(--accent-primary)]" />
          </div>
        ) : activeTab === 'platform_data' ? (
          renderPlatformData()
        ) : activeTab === 'user_data' ? (
          renderUserDataTree()
        ) : activeTab === 'models' ? (
          renderModelsTree()
        ) : (
          <div className="text-center py-8 text-[var(--text-muted)] text-sm">
            暂无文件
          </div>
        )}
      </div>

      {/* Actions */}
      {activeTab !== 'platform_data' && (
        <div className="p-3 border-t border-[var(--border-primary)] space-y-2">
          {/* For user_data: ONLY show new folder (Level 1) */}
          {activeTab === 'user_data' && (
            <div className="flex gap-2">
              <button
                onClick={() => {
                  setFolderType('data');
                  setParentFolderPath(''); // Create Root Folder
                  setShowCreateFolder(true);
                }}
                className="btn-secondary flex-1 text-xs py-2 flex items-center justify-center gap-1"
              >
                <FolderPlus className="w-3 h-3" />
                新建文件夹
              </button>
            </div>
          )}

          {/* For models: new folder and upload */}
          {activeTab === 'models' && (
            <div className="flex gap-2">
              <button
                onClick={() => {
                  setFolderType('models');
                  setShowCreateFolder(true);
                }}
                className="btn-secondary flex-1 text-xs py-2 flex items-center justify-center gap-1"
              >
                <FolderPlus className="w-3 h-3" />
                新建文件夹
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept={acceptedFormats}
                onChange={handleUpload}

                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className="btn-secondary flex-1 text-xs py-2 flex items-center justify-center gap-1"
              >
                {uploading ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <Upload className="w-3 h-3" />
                )}
                上传模型
              </button>
            </div>
          )}
          <div className="text-xs text-[var(--text-muted)] text-center">
            支持: {acceptedFormats.replace(/\./g, '').split(',').join(', ')}
          </div>
        </div>
      )}

      {/* Refresh Button */}
      <div className="p-2 border-t border-[var(--border-primary)]">
        <button
          onClick={fetchData}
          disabled={loading}
          className="w-full text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] flex items-center justify-center gap-1 py-1"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          刷新
        </button>
      </div>

      {/* Create Folder Modal */}
      {showCreateFolder && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="glass-card p-4 w-80">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold flex items-center gap-2">
                <FolderPlus className="w-5 h-5 text-[var(--accent-primary)]" />
                新建{folderType === 'models' ? '模型' : '数据'}文件夹
              </h3>
              <button 
                onClick={() => {
                  setShowCreateFolder(false);
                  setNewFolderPath('');
                }} 
                className="p-1 hover:bg-[var(--bg-card)] rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="space-y-3">
              <input
                type="text"
                value={newFolderPath}
                onChange={(e) => setNewFolderPath(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreateFolder()}
                placeholder={folderType === 'models' ? "例如: LSTM" : "例如: BTC_Option/2024"}
                className="w-full px-3 py-2 bg-[var(--bg-card)] border border-[var(--border-primary)] rounded-lg text-sm"
                autoFocus
              />
              <p className="text-xs text-[var(--text-muted)]">
                {folderType === 'models' 
                  ? '输入模型分类文件夹名称'
                  : '使用 / 创建嵌套文件夹（如：标的/年份）'
                }
              </p>
              <button
                onClick={handleCreateFolder}
                disabled={!newFolderPath.trim()}
                className="btn-primary w-full text-sm py-2 disabled:opacity-50"
              >
                创建
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="glass-card p-4 w-80">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold flex items-center gap-2">
                <Upload className="w-5 h-5 text-[var(--accent-primary)]" />
                上传数据
              </h3>
              <button onClick={() => setShowUploadModal(false)} className="p-1 hover:bg-[var(--bg-card)] rounded">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-[var(--text-muted)] mb-1 block">目标文件夹</label>
                <select
                  value={selectedUploadPath}
                  onChange={(e) => setSelectedUploadPath(e.target.value)}
                  className="w-full px-3 py-2 bg-[var(--bg-card)] border border-[var(--border-primary)] rounded-lg text-sm"
                >
                  <option value="">根目录</option>
                  {getUploadPathOptions().filter(p => p !== '').map(path => (
                    <option key={path} value={path}>{path}</option>
                  ))}
                </select>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept={acceptedFormats}
                onChange={handleUploadWithPath}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className="btn-primary w-full text-sm py-2 flex items-center justify-center gap-2"
              >
                {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                选择文件上传
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Preview Modal */}
      {previewData && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="glass-card p-6 w-[800px] max-h-[80vh] overflow-auto">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Table className="w-5 h-5 text-[var(--accent-primary)]" />
                <h3 className="text-lg font-bold">{previewData.filename}</h3>
                <span className="text-sm text-[var(--text-muted)]">
                  (显示 {previewData.showing} / {previewData.total_rows} 行)
                </span>
              </div>
              <button
                onClick={() => setPreviewData(null)}
                className="p-1 hover:bg-[var(--bg-card)] rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[var(--border-primary)]">
                    {previewData.columns.map(col => (
                      <th key={col} className="px-3 py-2 text-left text-[var(--text-muted)] font-medium">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.preview.map((row, i) => (
                    <tr key={i} className="border-b border-[var(--border-primary)]/30 hover:bg-[var(--bg-card)]">
                      {previewData.columns.map(col => (
                        <td key={col} className="px-3 py-2 text-[var(--text-secondary)]">
                          {typeof row[col] === 'number' 
                            ? row[col].toFixed(4) 
                            : String(row[col] ?? '-')
                          }
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* 右键菜单 */}
      {contextMenu && contextMenu.visible && (
        <div
          className="fixed z-50 bg-[var(--bg-card)] border border-[var(--border-primary)] rounded-lg shadow-xl py-1 min-w-[160px]"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* 空白处右键菜单 (Root) */}
          {contextMenu.type === 'empty' && (
            <>
              {activeTab === 'user_data' ? (
                // For user data, root only allows creating Subject folders
                <button
                   onClick={() => {
                    setFolderType('data');
                    setParentFolderPath('');
                    setShowCreateFolder(true);
                    closeContextMenu();
                   }}
                   className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--bg-elevated)] flex items-center gap-2"
                >
                  <FolderPlus className="w-4 h-4" />
                  新建标的文件夹
                </button>
              ) : (
                // For models, allow generic folder
                 <button
                  onClick={() => handleContextMenuAction('createRootFolder')}
                  className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--bg-elevated)] flex items-center gap-2"
                >
                  <FolderPlus className="w-4 h-4" />
                  新建文件夹
                </button>
              )}
              
              {/* No upload at root for user data */}
              {activeTab !== 'user_data' && (
                 <button
                  onClick={() => {
                    setSelectedUploadPath('');
                    setShowUploadModal(true);
                    closeContextMenu();
                  }}
                  className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--bg-elevated)] flex items-center gap-2"
                >
                  <Upload className="w-4 h-4" />
                  上传文件
                </button>
              )}
            </>
          )}

          {/* 文件夹右键菜单 (Level 1 - Subject) */}
          {contextMenu.type === 'folder' && (
            <>
              <button
                onClick={() => handleContextMenuAction('createSubfolder')}
                className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--bg-elevated)] flex items-center gap-2"
              >
                <FolderPlus className="w-4 h-4" />
                新建年份文件夹
              </button>
              <div className="border-t border-[var(--border-primary)] my-1" />
              <button
                onClick={() => deleteFolder(contextMenu.path)}
                className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--bg-elevated)] flex items-center gap-2 text-red-500"
              >
                <Trash2 className="w-4 h-4" />
                删除文件夹
              </button>
            </>
          )}

          {/* 子文件夹右键菜单 (Level 2 - Year) */}
          {contextMenu.type === 'subfolder' && (
            <>
              <button
                onClick={() => handleContextMenuAction('uploadHere')}
                className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--bg-elevated)] flex items-center gap-2"
              >
                <Upload className="w-4 h-4" />
                上传数据
              </button>
              <div className="border-t border-[var(--border-primary)] my-1" />
              <button
                onClick={() => deleteFolder(contextMenu.path)}
                className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--bg-elevated)] flex items-center gap-2 text-red-500"
              >
                <Trash2 className="w-4 h-4" />
                删除文件夹
              </button>
            </>
          )}

          {/* 文件右键菜单 */}
          {contextMenu.type === 'file' && (
            <>
              <button
                onClick={() => deleteFile(contextMenu.name, contextMenu.path)}
                className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--bg-elevated)] flex items-center gap-2 text-red-500"
              >
                <Trash2 className="w-4 h-4" />
                删除文件
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
