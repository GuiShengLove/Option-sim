from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime
import os
import hashlib
from ..models.strategy import StrategyFile, StrategyListResponse, SaveStrategyRequest
from ..engines.strategy_templates import STRATEGY_TEMPLATES

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

# Strategy storage directory
STRATEGY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "strategies")
if not os.path.exists(STRATEGY_DIR):
    os.makedirs(STRATEGY_DIR)


def generate_id(name: str) -> str:
    """Generate unique ID from filename."""
    return hashlib.md5(name.encode()).hexdigest()[:8]


@router.get("/", response_model=StrategyListResponse)
async def list_strategies():
    """List all saved strategies from disk (recursive)."""
    items = []
    
    try:
        # Walk directory recursively
        for root, dirs, files in os.walk(STRATEGY_DIR):
            for filename in files:
                if filename.endswith('.py'):
                    # Get full path
                    filepath = os.path.join(root, filename)
                    # Get relative path from STRATEGY_DIR (e.g. "trend/super_trend.py")
                    rel_path = os.path.relpath(filepath, STRATEGY_DIR)
                    # Use forward slashes for consistency
                    rel_path = rel_path.replace("\\", "/")
                    
                    stat = os.stat(filepath)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    items.append(StrategyFile(
                        id=generate_id(rel_path), # ID based on relative path
                        name=rel_path,            # Name is the relative path
                        content=content,
                        last_modified=datetime.fromtimestamp(stat.st_mtime)
                    ))
    except Exception as e:
        print(f"Error listing strategies: {e}")
        
    return StrategyListResponse(items=items)


@router.get("/templates")
async def get_strategy_templates():
    """Get available strategy templates."""
    templates = []
    for key, template in STRATEGY_TEMPLATES.items():
        templates.append({
            'id': key,
            'name': template.name,
            'chinese_name': template.chinese_name,
            'type': template.type.value,
            'risk_level': template.risk_level,
            'description': template.description,
            'max_profit': template.max_profit,
            'max_loss': template.max_loss,
            'ideal_iv': template.ideal_iv,
            'time_decay': template.time_decay,
            'legs_count': len(template.legs)
        })
    return {'templates': templates}


@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Get strategy content by ID (recursive lookup)."""
    # Recursive search
    for root, dirs, files in os.walk(STRATEGY_DIR):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, STRATEGY_DIR).replace("\\", "/")
                
                if generate_id(rel_path) == strategy_id:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    stat = os.stat(filepath)
                    return {
                        'id': strategy_id,
                        'name': rel_path,
                        'content': content,
                        'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
    
    raise HTTPException(status_code=404, detail="Strategy not found")


@router.post("/")
async def save_strategy(req: SaveStrategyRequest):
    """Save a new strategy or update existing (supports folders)."""
    # Name can be "folder/file.py"
    rel_path = req.name
    if not rel_path.endswith('.py'):
        rel_path += '.py'
    
    # Ensure usage of forward slashes
    rel_path = rel_path.replace("\\", "/")
    
    filepath = os.path.join(STRATEGY_DIR, rel_path)
    
    # Create subdirectories if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(req.content)
        
        return {
            'status': 'success',
            'id': generate_id(rel_path),
            'name': rel_path,
            'message': f'策略 {rel_path} 保存成功'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{strategy_id}")
async def update_strategy(strategy_id: str, req: SaveStrategyRequest):
    """Update existing strategy."""
    # Find file first
    found_path = None
    found_rel = None
    
    for root, dirs, files in os.walk(STRATEGY_DIR):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, STRATEGY_DIR).replace("\\", "/")
                
                if generate_id(rel_path) == strategy_id:
                    found_path = filepath
                    found_rel = rel_path
                    break
        if found_path:
            break
            
    if found_path:
        # If user changed name (moved file), handle it?
        # Current API only sends content. Req.name might be absent or same.
        # Assuming in-place update for PUT.
        
        with open(found_path, 'w', encoding='utf-8') as f:
            f.write(req.content)
        
        return {
            'status': 'success',
            'id': strategy_id,
            'message': f'策略 {found_rel} 更新成功'
        }
    
    raise HTTPException(status_code=404, detail="Strategy not found")


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Delete a strategy."""
    for root, dirs, files in os.walk(STRATEGY_DIR):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, STRATEGY_DIR).replace("\\", "/")
                
                if generate_id(rel_path) == strategy_id:
                    os.remove(filepath)
                    # Optional: Remove empty parent directories?
                    return {
                        'status': 'success',
                        'message': f'策略 {rel_path} 已删除'
                    }
    
    raise HTTPException(status_code=404, detail="Strategy not found")


@router.post("/folders")
async def create_folder(folder_name: str):
    """Create a new folder in strategies directory."""
    # Sanitize folder name
    folder_name = folder_name.strip().replace("\\", "/")
    if not folder_name:
        raise HTTPException(status_code=400, detail="文件夹名称不能为空")
    
    folder_path = os.path.join(STRATEGY_DIR, folder_name)
    
    if os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail="文件夹已存在")
    
    try:
        os.makedirs(folder_path)
        return {
            'status': 'success',
            'folder': folder_name,
            'message': f'文件夹 {folder_name} 创建成功'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/folders/{folder_path:path}")
async def delete_folder(folder_path: str):
    """Delete a folder (must be empty or force delete)."""
    import shutil
    
    # Sanitize path
    folder_path = folder_path.strip().replace("\\", "/")
    full_path = os.path.join(STRATEGY_DIR, folder_path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="文件夹不存在")
    
    if not os.path.isdir(full_path):
        raise HTTPException(status_code=400, detail="指定路径不是文件夹")
    
    # Check if folder is empty
    if os.listdir(full_path):
        # Force delete with all contents
        try:
            shutil.rmtree(full_path)
            return {
                'status': 'success',
                'message': f'文件夹 {folder_path} 及其内容已删除'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Empty folder, just remove
        try:
            os.rmdir(full_path)
            return {
                'status': 'success',
                'message': f'空文件夹 {folder_path} 已删除'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/from-template/{template_id}")
async def create_from_template(template_id: str, name: Optional[str] = None):
    """Create a new strategy from template."""
    template = STRATEGY_TEMPLATES.get(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    filename = name or f"{template_id}_strategy.py"
    if not filename.endswith('.py'):
        filename += '.py'
    
    # Generate strategy code from template
    code = generate_template_code(template)
    
    filepath = os.path.join(STRATEGY_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)
    
    return {
        'status': 'success',
        'id': generate_id(filename),
        'name': filename,
        'message': f'从模板 {template.chinese_name} 创建策略成功'
    }


def generate_template_code(template) -> str:
    """Generate Python strategy code from template."""
    legs_code = []
    for i, leg in enumerate(template.legs):
        legs_code.append(f"""        # Leg {i+1}: {leg.action.upper()} {leg.type.upper()}
        self.add_leg(
            type='{leg.type}',
            action='{leg.action}',
            strike_offset={leg.strike_offset},
            expiry_days={leg.expiry_offset},
            quantity={leg.quantity}
        )""")
    
    legs_str = '\n'.join(legs_code)
    
    return f'''"""
{template.chinese_name} ({template.name})
{'=' * (len(template.chinese_name) + len(template.name) + 3)}
{template.description}

风险等级: {'★' * template.risk_level}{'☆' * (5 - template.risk_level)}
类型: {template.type.value}
最大收益: {template.max_profit}
最大亏损: {template.max_loss}
盈亏平衡: {template.breakeven}
理想IV: {template.ideal_iv}
时间衰减: {template.time_decay}
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """策略配置参数"""
    capital: float = 100000
    position_size: int = 1


class {template.name.replace(" ", "").replace("-", "")}Strategy:
    """
    {template.chinese_name}策略实现
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.legs = []
        
    def on_init(self, context: Dict[str, Any]):
        """策略初始化 - 设置策略腿"""
{legs_str}
        
    def add_leg(self, type: str, action: str, strike_offset: float, 
                expiry_days: int, quantity: int):
        """添加策略腿"""
        self.legs.append({{
            'type': type,
            'action': action,
            'strike_offset': strike_offset,
            'expiry_days': expiry_days,
            'quantity': quantity
        }})
        
    def on_bar(self, context: Dict[str, Any], data: Dict[str, Any]):
        """每根K线触发"""
        spot_price = data.get('spot_price', 0)
        option_chain = data.get('option_chain', [])
        
        # TODO: 实现具体交易逻辑
        pass
        
    def get_greeks(self) -> Dict[str, float]:
        """获取组合Greeks"""
        return {{'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}}


# 策略元数据
STRATEGY_META = {{
    'name': '{template.name.lower().replace(" ", "_")}',
    'display_name': '{template.chinese_name}',
    'type': '{template.type.value}',
    'risk_level': {template.risk_level},
    'description': '{template.description}'
}}
'''
