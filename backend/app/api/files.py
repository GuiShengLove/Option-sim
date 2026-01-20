"""
Models and Data Upload API
Handles user model files (.pkl, .h5, .pt, .onnx) and data files (.parquet, .csv)
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
from datetime import datetime
import os
import shutil
import json
import hashlib

router = APIRouter(prefix="/api/files", tags=["files"])

# Storage directories
# From: backend/app/api/files.py -> go up to project root (Option-sim/)
_CURRENT_FILE = os.path.abspath(__file__)  # .../backend/app/api/files.py
_API_DIR = os.path.dirname(_CURRENT_FILE)   # .../backend/app/api
_APP_DIR = os.path.dirname(_API_DIR)        # .../backend/app
_BACKEND_DIR = os.path.dirname(_APP_DIR)    # .../backend
BASE_DIR = os.path.dirname(_BACKEND_DIR)    # .../Option-sim (project root)

USER_MODELS_DIR = os.path.join(_BACKEND_DIR, "user_models")
USER_DATA_DIR = os.path.join(_BACKEND_DIR, "user_data")
PLATFORM_DATA_DIR = os.path.join(BASE_DIR, "data")  # Project root /data folder

# Ensure directories exist
os.makedirs(USER_MODELS_DIR, exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Allowed extensions
MODEL_EXTENSIONS = {'.pkl', '.h5', '.pt', '.pth', '.onnx', '.joblib'}
DATA_EXTENSIONS = {'.parquet', '.csv', '.xlsx', '.json'}


def generate_file_id(filename: str) -> str:
    """Generate unique file ID."""
    return hashlib.md5(f"{filename}{datetime.now().isoformat()}".encode()).hexdigest()[:12]


def get_file_info(filepath: str, file_type: str) -> dict:
    """Get file metadata."""
    stat = os.stat(filepath)
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    
    return {
        'id': generate_file_id(filename),
        'name': filename,
        'type': file_type,
        'extension': ext,
        'size': stat.st_size,
        'size_human': format_size(stat.st_size),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'path': filepath
    }


def format_size(size: int) -> str:
    """Format file size to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ============== Models API ==============

@router.get("/models")
async def list_models():
    """List all user uploaded models."""
    models = []
    
    if os.path.exists(USER_MODELS_DIR):
        for filename in os.listdir(USER_MODELS_DIR):
            ext = os.path.splitext(filename)[1].lower()
            if ext in MODEL_EXTENSIONS:
                filepath = os.path.join(USER_MODELS_DIR, filename)
                models.append(get_file_info(filepath, 'model'))
    
    return {
        'models': models,
        'count': len(models),
        'storage_path': USER_MODELS_DIR
    }


@router.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    description: str = Form(default="")
):
    """Upload a model file."""
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in MODEL_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的模型格式: {ext}。支持: {', '.join(MODEL_EXTENSIONS)}"
        )
    
    # Save file
    filepath = os.path.join(USER_MODELS_DIR, file.filename)
    
    try:
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Save metadata
        meta_path = filepath + '.meta.json'
        meta = {
            'description': description,
            'upload_time': datetime.now().isoformat(),
            'original_name': file.filename
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        return {
            'status': 'success',
            'message': f'模型 {file.filename} 上传成功',
            'file': get_file_info(filepath, 'model')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{filename}")
async def delete_model(filename: str):
    """Delete a model file."""
    filepath = os.path.join(USER_MODELS_DIR, filename)
    meta_path = filepath + '.meta.json'
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="模型文件不存在")
    
    try:
        os.remove(filepath)
        if os.path.exists(meta_path):
            os.remove(meta_path)
        return {'status': 'success', 'message': f'模型 {filename} 已删除'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/folder")
async def create_model_folder(path: str):
    """Create a folder in user models directory."""
    clean_path = path.replace("\\", "/").strip("/")
    if ".." in clean_path or not clean_path:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    target_path = os.path.join(USER_MODELS_DIR, clean_path)
    
    if os.path.exists(target_path):
        raise HTTPException(status_code=400, detail=f"文件夹已存在: {clean_path}")
    
    try:
        os.makedirs(target_path, exist_ok=True)
        return {
            'status': 'success',
            'message': f'文件夹 {clean_path} 创建成功',
            'path': clean_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/folder")
async def delete_model_folder(path: str):
    """Delete a folder in user models directory (recursively)."""
    clean_path = path.replace("\\", "/").strip("/")
    if ".." in clean_path or not clean_path:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    target_path = os.path.join(USER_MODELS_DIR, clean_path)
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"文件夹不存在: {clean_path}")
    
    if not os.path.isdir(target_path):
        raise HTTPException(status_code=400, detail=f"不是文件夹: {clean_path}")
    
    try:
        shutil.rmtree(target_path)
        return {'status': 'success', 'message': f'文件夹 {clean_path} 及其内容已删除'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/tree")
async def list_models_tree():
    """List models as tree structure with folders."""
    folders = []
    root_files = []
    
    if not os.path.exists(USER_MODELS_DIR):
        return {'folders': [], 'files': [], 'total': 0}
    
    # Walk through directory
    for item in sorted(os.listdir(USER_MODELS_DIR)):
        item_path = os.path.join(USER_MODELS_DIR, item)
        
        if os.path.isdir(item_path):
            # It's a folder
            folder_files = []
            for f in sorted(os.listdir(item_path)):
                ext = os.path.splitext(f)[1].lower()
                if ext in MODEL_EXTENSIONS:
                    filepath = os.path.join(item_path, f)
                    folder_files.append(get_file_info(filepath, 'model'))
            
            folders.append({
                'id': item,
                'name': item,
                'path': item,
                'files': folder_files,
                'count': len(folder_files)
            })
        else:
            # It's a file
            ext = os.path.splitext(item)[1].lower()
            if ext in MODEL_EXTENSIONS:
                root_files.append(get_file_info(item_path, 'model'))
    
    return {
        'folders': folders,
        'files': root_files,
        'total': len(root_files) + sum(f['count'] for f in folders)
    }


@router.post("/models/upload-to")
async def upload_model_to(
    file: UploadFile = File(...),
    target_path: str = Form(default=""),
    description: str = Form(default="")
):
    """Upload a model file to a specific subdirectory."""
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in MODEL_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的模型格式: {ext}。支持: {', '.join(MODEL_EXTENSIONS)}"
        )
    
    clean_target = target_path.replace("\\", "/").strip("/") if target_path else ""
    if ".." in clean_target:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    target_dir = os.path.join(USER_MODELS_DIR, clean_target) if clean_target else USER_MODELS_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    filepath = os.path.join(target_dir, file.filename)
    
    try:
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        meta_path = filepath + '.meta.json'
        meta = {
            'description': description,
            'upload_time': datetime.now().isoformat(),
            'original_name': file.filename,
            'target_path': clean_target
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        return {
            'status': 'success',
            'message': f'模型 {file.filename} 上传到 {clean_target or "根目录"} 成功',
            'file': get_file_info(filepath, 'model')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Data API ==============

@router.get("/data")
async def list_data():
    """List all user uploaded data files."""
    data_files = []
    
    if os.path.exists(USER_DATA_DIR):
        for filename in os.listdir(USER_DATA_DIR):
            ext = os.path.splitext(filename)[1].lower()
            if ext in DATA_EXTENSIONS:
                filepath = os.path.join(USER_DATA_DIR, filename)
                data_files.append(get_file_info(filepath, 'data'))
    
    return {
        'data': data_files,
        'count': len(data_files),
        'storage_path': USER_DATA_DIR
    }


@router.post("/data/upload")
async def upload_data(
    file: UploadFile = File(...),
    description: str = Form(default="")
):
    """Upload a data file."""
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in DATA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的数据格式: {ext}。支持: {', '.join(DATA_EXTENSIONS)}"
        )
    
    filepath = os.path.join(USER_DATA_DIR, file.filename)
    
    try:
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Save metadata
        meta_path = filepath + '.meta.json'
        meta = {
            'description': description,
            'upload_time': datetime.now().isoformat(),
            'original_name': file.filename
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        return {
            'status': 'success',
            'message': f'数据文件 {file.filename} 上传成功',
            'file': get_file_info(filepath, 'data')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/preview/{filename}")
async def preview_data(filename: str, rows: int = 10):
    """Preview data file contents."""
    filepath = os.path.join(USER_DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="数据文件不存在")
    
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        if ext == '.parquet':
            import pandas as pd
            df = pd.read_parquet(filepath)
            preview = df.head(rows).to_dict(orient='records')
            columns = list(df.columns)
            total_rows = len(df)
        elif ext == '.csv':
            import pandas as pd
            df = pd.read_csv(filepath)
            preview = df.head(rows).to_dict(orient='records')
            columns = list(df.columns)
            total_rows = len(df)
        elif ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                preview = data[:rows]
                columns = list(data[0].keys()) if data else []
                total_rows = len(data)
            else:
                preview = [data]
                columns = list(data.keys())
                total_rows = 1
        else:
            raise HTTPException(status_code=400, detail=f"暂不支持预览 {ext} 格式")
        
        return {
            'filename': filename,
            'columns': columns,
            'preview': preview,
            'total_rows': total_rows,
            'showing': min(rows, total_rows)
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="需要安装pandas库来预览数据")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== User Data Folder Delete API (MUST be before /data/{filepath:path}) ==============

@router.delete("/data/folder")
async def delete_folder(path: str):
    """
    Delete a folder in user data directory (recursively).
    NOTE: This route MUST be defined BEFORE /data/{filepath:path} to avoid route conflict.
    """
    clean_path = path.replace("\\", "/").strip("/")
    if ".." in clean_path or not clean_path:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    target_path = os.path.join(USER_DATA_DIR, clean_path)
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"文件夹不存在: {clean_path}")
    
    if not os.path.isdir(target_path):
        raise HTTPException(status_code=400, detail=f"不是文件夹: {clean_path}")
    
    try:
        shutil.rmtree(target_path)
        return {'status': 'success', 'message': f'文件夹 {clean_path} 及其内容已删除'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/data/{filepath:path}")
async def delete_data(filepath: str):
    """Delete a data file. Supports nested paths like 'subject/year/file.parquet'."""
    # Sanitize path
    clean_path = filepath.replace("\\", "/").strip("/")
    if ".." in clean_path or not clean_path:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    full_path = os.path.join(USER_DATA_DIR, clean_path)
    meta_path = full_path + '.meta.json'
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"数据文件不存在: {clean_path}")
    
    if os.path.isdir(full_path):
        raise HTTPException(status_code=400, detail=f"这是文件夹，请使用文件夹删除接口: {clean_path}")
    
    try:
        os.remove(full_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)
        return {'status': 'success', 'message': f'数据文件 {clean_path} 已删除'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ============== Platform Data API ==============

@router.get("/platform-data")
async def list_platform_data():
    """
    List platform built-in data with full 3-level structure:
    datasets (标的) -> years (年份) -> files (文件)
    """
    datasets = []
    all_files = []
    
    if not os.path.exists(PLATFORM_DATA_DIR):
        return {'datasets': [], 'groups': [], 'data': [], 'count': 0}
    
    # Scan top-level directories (dataset/asset level)
    for asset_dir in sorted(os.listdir(PLATFORM_DATA_DIR)):
        asset_path = os.path.join(PLATFORM_DATA_DIR, asset_dir)
        if not os.path.isdir(asset_path):
            continue
        if asset_dir.startswith('.'):
            continue
        
        dataset = {
            'id': asset_dir,
            'name': asset_dir,
            'type': 'PLATFORM',
            'path': asset_path,
            'years': [],
            'total_files': 0,
            'total_size': 0
        }
        
        # Scan year directories
        year_groups = {}
        for year_dir in sorted(os.listdir(asset_path)):
            year_path = os.path.join(asset_path, year_dir)
            if not os.path.isdir(year_path):
                continue
            if not (year_dir.isdigit() and len(year_dir) == 4):
                continue
            
            year = year_dir
            if year not in year_groups:
                year_groups[year] = {
                    'year': year,
                    'files': [],
                    'total_size': 0
                }
            
            # Scan files in year folder
            for filename in sorted(os.listdir(year_path)):
                ext = os.path.splitext(filename)[1].lower()
                if ext in DATA_EXTENSIONS:
                    filepath = os.path.join(year_path, filename)
                    info = get_file_info(filepath, 'platform_data')
                    info['relative_path'] = f"{asset_dir}/{year}/{filename}"
                    year_groups[year]['files'].append(info)
                    year_groups[year]['total_size'] += info['size']
                    all_files.append(info)
                    dataset['total_files'] += 1
                    dataset['total_size'] += info['size']
        
        # Format year groups and add to dataset
        for year in sorted(year_groups.keys()):
            group = year_groups[year]
            group['total_size_human'] = format_size(group['total_size'])
            dataset['years'].append(group)
        
        dataset['total_size_human'] = format_size(dataset['total_size'])
        
        if dataset['years']:  # Only include datasets with data
            datasets.append(dataset)
    
    # Also return legacy flat groups for backward compatibility
    legacy_groups = []
    for ds in datasets:
        for yr in ds['years']:
            legacy_groups.append(yr)
    
    return {
        'datasets': datasets,  # New: full 3-level structure
        'groups': legacy_groups,  # Legacy: flat year groups
        'data': all_files,
        'count': len(all_files),
        'storage_path': PLATFORM_DATA_DIR
    }


# ============== User Data Folder Management API ==============

@router.post("/data/folder")
async def create_folder(path: str):
    """
    Create a folder in user data directory.
    Supports nested paths like 'BTC_Option/2024'.
    """
    # Sanitize path to prevent directory traversal
    clean_path = path.replace("\\", "/").strip("/")
    if ".." in clean_path:
        raise HTTPException(status_code=400, detail="Invalid path: '..' not allowed")
    
    target_path = os.path.join(USER_DATA_DIR, clean_path)
    
    if os.path.exists(target_path):
        raise HTTPException(status_code=400, detail=f"文件夹已存在: {clean_path}")
    
    try:
        os.makedirs(target_path, exist_ok=True)
        return {
            'status': 'success',
            'message': f'文件夹 {clean_path} 创建成功',
            'path': clean_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/data/upload-to")
async def upload_data_to(
    file: UploadFile = File(...),
    target_path: str = Form(default=""),
    description: str = Form(default="")
):
    """
    Upload a data file to a specific subdirectory.
    target_path: e.g., 'BTC_Option/2024' or '' for root.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in DATA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的数据格式: {ext}。支持: {', '.join(DATA_EXTENSIONS)}"
        )
    
    # Sanitize target path
    clean_target = target_path.replace("\\", "/").strip("/") if target_path else ""
    if ".." in clean_target:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    # Create target directory if needed
    target_dir = os.path.join(USER_DATA_DIR, clean_target) if clean_target else USER_DATA_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    filepath = os.path.join(target_dir, file.filename)
    
    try:
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Save metadata
        meta_path = filepath + '.meta.json'
        meta = {
            'description': description,
            'upload_time': datetime.now().isoformat(),
            'original_name': file.filename,
            'target_path': clean_target
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        return {
            'status': 'success',
            'message': f'数据文件 {file.filename} 上传到 {clean_target or "根目录"} 成功',
            'file': get_file_info(filepath, 'data'),
            'path': os.path.join(clean_target, file.filename).replace("\\", "/")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/tree")
async def list_data_tree():
    """
    List user data as a tree structure (same format as platform data).
    Returns structure: { datasets: [...], total_files: int }
    Each dataset: { id, name, type:'USER', years: [...], path }
    Each year: { year, files: [...], total_size, total_size_human }
    
    Now also includes:
    - Empty folders (so users can see folders they just created)
    - Folders without year subfolders (files directly in folder)
    """
    datasets = []
    
    if not os.path.exists(USER_DATA_DIR):
        return {'datasets': [], 'total_files': 0}
    
    # Scan top-level directories (dataset/asset level)
    for asset_dir in sorted(os.listdir(USER_DATA_DIR)):
        asset_path = os.path.join(USER_DATA_DIR, asset_dir)
        if not os.path.isdir(asset_path):
            continue
        
        # Skip hidden folders and __pycache__
        if asset_dir.startswith('.') or asset_dir == '__pycache__':
            continue
        
        dataset = {
            'id': asset_dir,
            'name': asset_dir,
            'type': 'USER',
            'path': asset_path,
            'years': [],
            'root_files': []  # Files directly in the folder (not in year subdirs)
        }
        
        # Track if folder has any content at all
        has_content = False
        
        # Scan items in asset folder
        year_groups = {}
        for item in sorted(os.listdir(asset_path)):
            item_path = os.path.join(asset_path, item)
            
            if os.path.isdir(item_path):
                # It's a subdirectory - could be year folder or generic folder
                subdir_name = item
                subdir_files = []
                subdir_size = 0
                
                # Scan files in subdirectory
                for filename in sorted(os.listdir(item_path)):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in DATA_EXTENSIONS:
                        filepath = os.path.join(item_path, filename)
                        info = get_file_info(filepath, 'user_data')
                        info['relative_path'] = f"{asset_dir}/{subdir_name}/{filename}"
                        subdir_files.append(info)
                        subdir_size += info['size']
                
                # Add as a "year" group (works for any subfolder)
                year_groups[subdir_name] = {
                    'year': subdir_name,
                    'files': subdir_files,
                    'total_size': subdir_size,
                    'total_size_human': format_size(subdir_size)
                }
                has_content = True
            else:
                # It's a file directly in the asset folder
                ext = os.path.splitext(item)[1].lower()
                if ext in DATA_EXTENSIONS:
                    filepath = item_path
                    info = get_file_info(filepath, 'user_data')
                    info['relative_path'] = f"{asset_dir}/{item}"
                    dataset['root_files'].append(info)
                    has_content = True
        
        # Format year groups and add to dataset
        for year in sorted(year_groups.keys()):
            dataset['years'].append(year_groups[year])
        
        # Include dataset even if empty (so user can see their created folders)
        # But mark it so UI can display differently if needed
        dataset['is_empty'] = not has_content and len(year_groups) == 0 and len(dataset['root_files']) == 0
        datasets.append(dataset)
    
    total_files = sum(
        len(f['files']) 
        for d in datasets 
        for f in d['years']
    ) + sum(len(d.get('root_files', [])) for d in datasets)
    
    return {
        'datasets': datasets,
        'total_files': total_files
    }


# ============== Datasets List API ==============

@router.get("/datasets")
async def list_all_datasets():
    """
    List all available datasets (both platform and user).
    Used for backtest data source selector.
    Returns list of { id, name, type:'PLATFORM'|'USER', path, date_range, file_count }
    """
    datasets = []
    
    # 1. Platform datasets (scan data/ folder)
    if os.path.exists(PLATFORM_DATA_DIR):
        for asset_dir in sorted(os.listdir(PLATFORM_DATA_DIR)):
            asset_path = os.path.join(PLATFORM_DATA_DIR, asset_dir)
            if not os.path.isdir(asset_path):
                continue
            if asset_dir.startswith('.'):
                continue
            
            # Count files and find date range
            dates = []
            file_count = 0
            for year_dir in os.listdir(asset_path):
                year_path = os.path.join(asset_path, year_dir)
                if os.path.isdir(year_path):
                    for f in os.listdir(year_path):
                        if f.endswith('.parquet') and f.startswith('options_'):
                            file_count += 1
                            # Extract date from filename
                            date_str = f.replace('options_', '').replace('.parquet', '')
                            dates.append(date_str)
            
            if dates:
                dates.sort()
                datasets.append({
                    'id': asset_dir,
                    'name': asset_dir,
                    'type': 'PLATFORM',
                    'path': asset_path,
                    'date_range': {
                        'start': dates[0],
                        'end': dates[-1]
                    },
                    'file_count': file_count
                })
    
    # 2. User datasets (scan user_data/ folder)
    if os.path.exists(USER_DATA_DIR):
        for asset_dir in sorted(os.listdir(USER_DATA_DIR)):
            asset_path = os.path.join(USER_DATA_DIR, asset_dir)
            if not os.path.isdir(asset_path):
                continue
            if asset_dir.startswith('.') or asset_dir == '__pycache__':
                continue
            
            dates = []
            file_count = 0
            for year_dir in os.listdir(asset_path):
                year_path = os.path.join(asset_path, year_dir)
                if os.path.isdir(year_path) and year_dir.isdigit():
                    for f in os.listdir(year_path):
                        if f.endswith('.parquet'):
                            file_count += 1
                            if f.startswith('options_'):
                                date_str = f.replace('options_', '').replace('.parquet', '')
                                dates.append(date_str)
            
            if file_count > 0:
                date_range = None
                if dates:
                    dates.sort()
                    date_range = {'start': dates[0], 'end': dates[-1]}
                
                datasets.append({
                    'id': asset_dir,
                    'name': f"{asset_dir} (用户)",
                    'type': 'USER',
                    'path': asset_path,
                    'date_range': date_range,
                    'file_count': file_count
                })
    
    return {
        'datasets': datasets,
        'total': len(datasets)
    }
