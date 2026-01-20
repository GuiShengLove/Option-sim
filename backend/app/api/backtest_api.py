"""
Backtest API Endpoint
=====================
Exposes BacktestEngine functionality via REST API.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import importlib.util
import sys
from datetime import datetime

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# Path to strategies storage
STRATEGY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "strategies")


class BacktestRequest(BaseModel):
    """Request model for backtest execution."""
    strategy_id: str
    dataset_id: str = "510050_SH"  # Platform or User dataset ID
    start_date: str
    end_date: str
    initial_capital: float = 1_000_000
    margin_scheme: str = "SSE"  # 'FIXED', 'SSE', 'SPAN', 'PM'
    # 新增：自定义保证金参数
    margin_ratio: Optional[float] = None  # 保证金率，覆盖默认 12%
    maintenance_margin: Optional[float] = None  # 维持保证金率，覆盖默认 7%
    leverage: Optional[float] = 1.0  # 杠杆倍数


class BacktestMetrics(BaseModel):
    """Response metrics from backtest."""
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    final_equity: float
    trade_count: int
    # 新增策略表现指标
    win_rate: float = 0.0  # 胜率 (0-1)
    profit_factor: float = 0.0  # 盈亏比 (总盈利/总亏损)
    avg_trade_pnl: float = 0.0  # 平均单笔盈亏
    max_consecutive_losses: int = 0  # 最大连续亏损次数
    trading_days: int = 0  # 实际交易天数
    daily_pnl: float = 0.0  # 最后一日盈亏
    realized_pnl: float = 0.0  # 已实现盈亏


class EquityPoint(BaseModel):
    """Single point on equity curve."""
    date: str
    equity: float
    cash: float
    margin_utilization: float
    position_count: Optional[int] = 0
    total_delta: Optional[float] = 0.0
    total_gamma: Optional[float] = 0.0
    total_vega: Optional[float] = 0.0
    total_theta: Optional[float] = 0.0


class TradeRecord(BaseModel):
    """Single trade record."""
    date: str
    symbol: str
    action: str
    quantity: float
    price: float
    fee: float
    realized_pnl: float


class BacktestResponse(BaseModel):
    """Response from backtest execution."""
    success: bool
    message: str
    metrics: Optional[BacktestMetrics] = None
    equity_curve: Optional[List[EquityPoint]] = None
    trades: Optional[List[TradeRecord]] = None
    strategy_name: str = ""
    dataset_id: str = ""



def load_strategy_class(strategy_id: str):
    """
    Load strategy class from file by ID.
    Returns (strategy_class, strategy_name).
    """
    import hashlib
    from abc import ABC
    import inspect

    def _generate_id(name: str) -> str:
        return hashlib.md5(name.encode()).hexdigest()[:8]
    
    def _is_concrete_strategy(cls) -> bool:
        """Check if class is a concrete (non-abstract) strategy with required methods."""
        if not isinstance(cls, type):
            return False
        # Skip abstract classes
        if inspect.isabstract(cls):
            return False
        # Skip if class name is 'BaseStrategy' or 'ABC'
        if cls.__name__ in ('BaseStrategy', 'ABC', 'ABCMeta'):
            return False
        # Must have on_bar and on_init methods
        if not (hasattr(cls, 'on_bar') and hasattr(cls, 'on_init')):
            return False
        return True
    
    # Search for strategy file recursively
    found_path = None
    found_name = None
    
    for root, dirs, files in os.walk(STRATEGY_DIR):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, STRATEGY_DIR).replace("\\", "/")
                
                if _generate_id(rel_path) == strategy_id or rel_path == strategy_id or filename == strategy_id:
                    found_path = filepath
                    found_name = rel_path
                    break
        if found_path:
            break
    
    if not found_path:
        raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_id}")
    
    # Dynamically load the module
    spec = importlib.util.spec_from_file_location("strategy_module", found_path)
    module = importlib.util.module_from_spec(spec)
    
    # Ensure parent package is in path
    strategy_parent = os.path.dirname(found_path)
    if strategy_parent not in sys.path:
        sys.path.insert(0, strategy_parent)
    
    # Also add project root to path for backend imports
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(found_path)))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load strategy: {str(e)}")
    
    strategy_class = None
    
    # Method 1: Check for STRATEGY_META with 'class' key
    if hasattr(module, 'STRATEGY_META') and isinstance(module.STRATEGY_META, dict):
        meta_class = module.STRATEGY_META.get('class')
        if meta_class and _is_concrete_strategy(meta_class):
            strategy_class = meta_class
    
    # Method 2: Look for concrete strategy classes (not abstract)
    if strategy_class is None:
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
            attr = getattr(module, attr_name)
            if _is_concrete_strategy(attr):
                strategy_class = attr
                break
    
    # Method 3: Try common names but exclude abstract ones
    if strategy_class is None:
        for name in ['Strategy', 'DemoStrategy', 'TestStrategy']:
            if hasattr(module, name):
                candidate = getattr(module, name)
                if _is_concrete_strategy(candidate):
                    strategy_class = candidate
                    break
    
    if strategy_class is None:
        raise HTTPException(status_code=400, detail=f"No valid concrete strategy class found in {found_name}. Make sure the strategy class is not abstract.")
    
    return strategy_class, found_name



@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    Execute a backtest with the specified strategy and parameters.
    """
    try:
        # Import backtest engine
        from ..engines.backtest import BacktestEngine
        from ..engines.strategy import BaseStrategy
        
        # Load strategy
        strategy_class, strategy_name = load_strategy_class(request.strategy_id)
        
        print(f"🚀 Starting backtest: {strategy_name}")
        print(f"   Dataset: {request.dataset_id}")
        print(f"   Date range: {request.start_date} to {request.end_date}")
        print(f"   Initial capital: ¥{request.initial_capital:,.0f}")
        if request.margin_ratio:
            print(f"   Custom margin ratio: {request.margin_ratio*100:.1f}%")
        if request.maintenance_margin:
            print(f"   Custom maintenance margin: {request.maintenance_margin*100:.1f}%")
        
        # Create engine with custom margin parameters
        engine = BacktestEngine(
            dataset_id=request.dataset_id,
            initial_capital=request.initial_capital,
            margin_scheme=request.margin_scheme,
            margin_ratio=request.margin_ratio,
            maintenance_margin=request.maintenance_margin,
            leverage=request.leverage or 1.0
        )
        
        # Run backtest
        results_df, trade_log = engine.run(
            strategy_cls=strategy_class,
            strategy_config={},
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if results_df.empty:
            return BacktestResponse(
                success=False,
                message="No trading data found for the specified date range",
                strategy_name=strategy_name,
                dataset_id=request.dataset_id
            )
        
        # Calculate metrics
        equity = results_df['equity'].values
        initial_equity = request.initial_capital
        final_equity = equity[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Max drawdown
        running_max = equity[0]
        max_drawdown = 0
        for e in equity:
            if e > running_max:
                running_max = e
            dd = (running_max - e) / running_max
            if dd > max_drawdown:
                max_drawdown = dd
        
        # Sharpe ratio (simplified)
        import numpy as np
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Build equity curve (handle NaN values)
        import math
        
        def safe_float(val, default=0.0):
            """Convert to float, replacing NaN/inf with default."""
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return default
                return f
            except (TypeError, ValueError):
                return default
        
        def safe_int(val, default=0):
            """Convert to int, handling NaN."""
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return default
                return int(f)
            except (TypeError, ValueError):
                return default
        
        equity_curve = []
        for idx, row in results_df.iterrows():
            equity_curve.append(EquityPoint(
                date=idx.strftime("%Y-%m-%d"),
                equity=safe_float(row['equity']),
                cash=safe_float(row['cash']),
                margin_utilization=safe_float(row['margin_utilization']),
                position_count=safe_int(row.get('position_count', 0)),
                total_delta=safe_float(row.get('total_delta', 0.0)),
                total_gamma=safe_float(row.get('total_gamma', 0.0)),
                total_vega=safe_float(row.get('total_vega', 0.0)),
                total_theta=safe_float(row.get('total_theta', 0.0))
            ))
        
        # 计算策略表现指标
        win_count = 0
        total_profit = 0.0
        total_loss = 0.0
        consecutive_losses = 0
        max_consecutive_losses = 0
        realized_pnl = 0.0
        
        for trade in trade_log:
            pnl = trade.get('realized_pnl', 0.0)
            realized_pnl += pnl
            if pnl > 0:
                win_count += 1
                total_profit += pnl
                consecutive_losses = 0
            elif pnl < 0:
                total_loss += abs(pnl)
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        trade_count = len(trade_log)
        win_rate = win_count / trade_count if trade_count > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)
        avg_trade_pnl = realized_pnl / trade_count if trade_count > 0 else 0.0
        trading_days = len(results_df)
        
        # 计算最后一日盈亏
        daily_pnl = 0.0
        if len(equity) >= 2:
            daily_pnl = equity[-1] - equity[-2]
        
        metrics = BacktestMetrics(
            total_return=round(total_return * 100, 2),
            max_drawdown=round(max_drawdown * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            final_equity=round(final_equity, 2),
            trade_count=trade_count,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
            avg_trade_pnl=round(avg_trade_pnl, 2),
            max_consecutive_losses=max_consecutive_losses,
            trading_days=trading_days,
            daily_pnl=round(daily_pnl, 2),
            realized_pnl=round(realized_pnl, 2)
        )
        
        print(f"✅ Backtest complete: Return={metrics.total_return}%, MaxDD={metrics.max_drawdown}%")
        
        return BacktestResponse(
            success=True,
            message=f"Backtest completed successfully for {strategy_name}",
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trade_log,
            strategy_name=strategy_name,
            dataset_id=request.dataset_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return BacktestResponse(
            success=False,
            message=f"Backtest failed: {str(e)}",
            strategy_name=request.strategy_id
        )


@router.get("/demo")
async def demo_backtest():
    """
    Run a demo backtest with default parameters.
    Returns simulated results if no strategy is selected.
    """
    import numpy as np
    
    # Generate demo equity curve
    dates = []
    equity = []
    base = 1000000
    
    from datetime import datetime, timedelta
    current = datetime(2020, 1, 2)
    
    for i in range(252):  # Trading days in a year
        dates.append(current.strftime("%Y-%m-%d"))
        # Simulated random walk with slight upward drift
        change = np.random.normal(0.0003, 0.015)
        base = base * (1 + change)
        equity.append(round(base, 2))
        current += timedelta(days=1)
        while current.weekday() >= 5:  # Skip weekends
            current += timedelta(days=1)
    
    final = equity[-1]
    initial = 1000000
    total_return = (final - initial) / initial * 100
    
    # Calculate max drawdown
    running_max = equity[0]
    max_dd = 0
    for e in equity:
        if e > running_max:
            running_max = e
        dd = (running_max - e) / running_max * 100
        if dd > max_dd:
            max_dd = dd
    
    equity_curve = [
        EquityPoint(date=d, equity=e, cash=e*0.8, margin_utilization=0.12)
        for d, e in zip(dates, equity)
    ]
    
    return BacktestResponse(
        success=True,
        message="Demo backtest (simulated data)",
        metrics=BacktestMetrics(
            total_return=round(total_return, 2),
            max_drawdown=round(max_dd, 2),
            sharpe_ratio=round(np.random.uniform(0.5, 2.0), 2),
            final_equity=round(final, 2),
            trade_count=120
        ),
        equity_curve=equity_curve,
        strategy_name="Demo Strategy"
    )
