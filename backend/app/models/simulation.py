from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, List, Any

class SimulationConfig(BaseModel):
    # Market Params
    S0: float = Field(..., description="Initial Underlying Price")
    T_days: int = Field(..., description="Simulation Duration in Days")
    n_paths: int = Field(100, description="Number of Monte Carlo paths")
    
    # Model Selection
    model: Literal['GBM', 'HESTON', 'MJD', 'GARCH'] = 'GBM'
    
    # Calibration Mode
    calibration_mode: Literal['manual', 'historical'] = 'manual'
    dataset_id: Optional[str] = Field(None, description="Dataset ID for historical calibration")
    
    # Random Seed for reproducibility
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    # Model Params (Optional based on selection)
    mu: float = 0.05
    sigma: float = 0.20
    
    # Heston specific
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    
    # Jump specific (MJD)
    lam: float = 0.75
    m: float = -0.02
    v: float = 0.1
    
    # GARCH specific
    omega: float = 0.000001  # Constant term
    alpha: float = 0.1      # ARCH coefficient
    beta: float = 0.85      # GARCH coefficient
    
    # Panic Factor (0.0 - 1.0)
    panic_factor: float = 0.0

class SimulationResponse(BaseModel):
    paths: list[list[float]]  # [path_idx][step_idx]
    dates: list[str] = []
    stats: Dict[str, Any]  # Mixed types: float, int, str, bool
    calibration_info: Optional[Dict[str, Any]] = None  # Contains calibrated params

class CalibrationRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID")
    lookback_days: int = Field(60, description="Days of history for volatility calculation")

class CalibrationResponse(BaseModel):
    sigma: float
    mu: float
    last_price: float
    data_points: int

class StrategyEvaluationRequest(BaseModel):
    paths: list[list[float]]  # Simulation paths
    strategy_id: str = Field(..., description="Strategy template ID")
    spot: float = Field(..., description="Initial spot price")
    expiry_days: int = Field(..., description="Days to expiry")
    initial_iv: float = Field(0.20, description="Initial implied volatility")
    risk_free_rate: float = Field(0.03, description="Risk-free rate")
    strikes: Optional[List[float]] = Field(None, description="Custom strikes (optional)")

class StrategyEvaluationResponse(BaseModel):
    pnl_distribution: List[float]  # PnL for each path
    avg_pnl: float
    win_rate: float
    max_profit: float
    max_loss: float
    var_95: float
    cvar_95: float
    strategy_info: Dict[str, str]

