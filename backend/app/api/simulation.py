"""
Simulation API
==============
Monte Carlo simulation and strategy evaluation endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from ..models.simulation import (
    SimulationConfig, SimulationResponse,
    CalibrationRequest, CalibrationResponse,
    StrategyEvaluationRequest, StrategyEvaluationResponse
)
from ..engines.simulator import Simulator
from ..engines.strategy_evaluator import StrategyEvaluator
from ..engines.pricing import PricingEngine
import numpy as np
import os
import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/simulation", tags=["simulation"])

simulator = Simulator()
pricing_engine = PricingEngine()
strategy_evaluator = StrategyEvaluator(pricing_engine)

# Data directory for calibration
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")


def _get_historical_prices(dataset_id: str, lookback_days: int = 60) -> np.ndarray:
    """
    Load historical underlying prices from parquet files.
    Returns array of close prices (oldest to newest).
    """
    dataset_dir = os.path.join(DATA_DIR, dataset_id)
    if not os.path.exists(dataset_dir):
        # Try user data
        user_data_dir = os.path.join(_PROJECT_ROOT, "user_data", dataset_id)
        if os.path.exists(user_data_dir):
            dataset_dir = user_data_dir
        else:
            raise ValueError(f"Dataset not found: {dataset_id}")
    
    # Find available date files
    date_files = []
    for f in os.listdir(dataset_dir):
        if f.endswith('.parquet'):
            date_str = f.replace('.parquet', '')
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                date_files.append((date_str, os.path.join(dataset_dir, f)))
            except ValueError:
                continue
    
    if not date_files:
        raise ValueError(f"No data files found in {dataset_id}")
    
    # Sort by date and take latest N days
    date_files.sort(key=lambda x: x[0], reverse=True)
    date_files = date_files[:lookback_days]
    date_files.reverse()  # Oldest to newest
    
    prices = []
    for date_str, file_path in date_files:
        try:
            df = pd.read_parquet(file_path)
            # Try to get underlying price from first row (assuming it's stored there)
            # Or calculate from ATM options
            if 'us_close' in df.columns:
                # Underlying close price
                price = df['us_close'].iloc[0]
            elif 'underlying_price' in df.columns:
                price = df['underlying_price'].iloc[0]
            else:
                # Fallback: estimate from ATM option data
                if 'strike' in df.columns and 'close' in df.columns:
                    # Use average of ATM call and put
                    atm_options = df[df['strike'].between(df['strike'].median() - 0.1, 
                                                          df['strike'].median() + 0.1)]
                    if not atm_options.empty:
                        price = atm_options['strike'].median()
                    else:
                        continue
                else:
                    continue
            
            if price and price > 0:
                prices.append(float(price))
        except Exception as e:
            continue
    
    return np.array(prices)


@router.post("/run", response_model=SimulationResponse)
async def run_simulation(config: SimulationConfig):
    """
    Run Monte Carlo Simulation.
    
    Supports:
    - Manual parameter specification
    - Historical data calibration
    - Multiple models (GBM, Heston, MJD)
    """
    try:
        params = config.model_dump()
        calibration_info = None
        
        # Historical calibration mode
        if config.calibration_mode == 'historical' and config.dataset_id:
            try:
                prices = _get_historical_prices(config.dataset_id, lookback_days=60)
                mu, sigma = Simulator.calibrate_from_history(prices)
                params['mu'] = mu
                params['sigma'] = sigma
                calibration_info = {
                    'calibrated_mu': mu,
                    'calibrated_sigma': sigma,
                    'data_points': len(prices),
                    'last_price': float(prices[-1]) if len(prices) > 0 else None
                }
            except Exception as e:
                # Fall back to manual params if calibration fails
                calibration_info = {'error': str(e)}
        
        # Apply Panic Factor Logic
        if config.panic_factor > 0:
            if config.model == 'HESTON':
                params['xi'] += config.panic_factor * 2.0
                params['theta'] += config.panic_factor * 0.2
            elif config.model == 'GBM':
                params['sigma'] += config.panic_factor * 0.5
            elif config.model == 'MJD':
                params['lam'] *= (1 + config.panic_factor * 5)
                params['m'] -= config.panic_factor * 0.1

        # Set random seed if provided
        if config.seed is not None:
            params['seed'] = config.seed

        paths = simulator.generate_paths(
            model=config.model,
            S0=config.S0,
            T_days=config.T_days,
            n_paths=config.n_paths,
            params=params
        )
        
        # Convert to list for JSON
        paths_list = paths.tolist()
        
        # Calculate statistics
        final_prices = paths[:, -1]
        returns = (final_prices - config.S0) / config.S0
        sorted_returns = np.sort(returns)
        var_95_idx = int(0.05 * len(sorted_returns))
        
        # CVaR95 (Expected Shortfall) - average of returns below VaR95
        cvar_95 = float(np.mean(sorted_returns[:var_95_idx])) if var_95_idx > 0 else 0
        
        stats = {
            "mean_terminal": float(np.mean(final_prices)),
            "std_terminal": float(np.std(final_prices)),
            "min_terminal": float(np.min(final_prices)),
            "max_terminal": float(np.max(final_prices)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "var_95": float(sorted_returns[var_95_idx]) if var_95_idx < len(sorted_returns) else 0,
            "cvar_95": cvar_95,  # NEW: Expected Shortfall
            "model": config.model,
            "n_paths": config.n_paths,
            "gpu_used": simulator.gpu_available and config.n_paths >= 1000
        }

        # Generate date labels
        dates = []
        start_date = datetime.now()
        for i in range(config.T_days + 1):
            dates.append((start_date + timedelta(days=i)).strftime('%Y-%m-%d'))

        return SimulationResponse(
            paths=paths_list,
            dates=dates,
            stats=stats,
            calibration_info=calibration_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate_parameters(request: CalibrationRequest):
    """
    Calibrate mu and sigma from historical data.
    """
    try:
        prices = _get_historical_prices(request.dataset_id, request.lookback_days)
        
        if len(prices) < 2:
            raise HTTPException(status_code=400, detail="Not enough historical data for calibration")
        
        mu, sigma = Simulator.calibrate_from_history(prices)
        
        return CalibrationResponse(
            sigma=sigma,
            mu=mu,
            last_price=float(prices[-1]),
            data_points=len(prices)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy-evaluate", response_model=StrategyEvaluationResponse)
async def evaluate_strategy(request: StrategyEvaluationRequest):
    """
    Evaluate an option strategy on simulation paths.
    
    Calculates real option PnL using BSM pricing at entry
    and intrinsic value at expiry.
    """
    try:
        paths = np.array(request.paths)
        
        result = strategy_evaluator.evaluate_strategy(
            paths=paths,
            strategy_id=request.strategy_id,
            spot=request.spot,
            expiry_days=request.expiry_days,
            initial_iv=request.initial_iv,
            risk_free_rate=request.risk_free_rate,
            custom_strikes=request.strikes
        )
        
        return StrategyEvaluationResponse(
            pnl_distribution=result.pnl_distribution,
            avg_pnl=result.avg_pnl,
            win_rate=result.win_rate,
            max_profit=result.max_profit,
            max_loss=result.max_loss,
            var_95=result.var_95,
            cvar_95=result.cvar_95,
            strategy_info={
                "name": result.strategy_name,
                "type": result.strategy_type
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_simulation_info():
    """
    Get simulation engine capabilities and status.
    """
    return {
        "gpu_available": simulator.gpu_available,
        "supported_models": ["GBM", "HESTON", "MJD"],
        "calibration_modes": ["manual", "historical"],
        "max_paths_recommended": 10000 if simulator.gpu_available else 2000,
        "version": "2.0.0"
    }

