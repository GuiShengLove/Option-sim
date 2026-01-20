"""
Greeks API Routes
=================
API endpoints for Greeks calculation, IV solving, and SVI fitting.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Literal
import numpy as np
import pandas as pd

from ..engines.pricing import PricingEngine, quick_greeks, quick_iv

router = APIRouter(prefix="/api/greeks", tags=["greeks"])

# Initialize pricing engine
engine = PricingEngine(risk_free_rate=0.03, dividend_yield=0.01)


# ============================================================
# Request/Response Models
# ============================================================

class SingleGreeksRequest(BaseModel):
    """Request for single option Greeks calculation."""
    spot: float
    strike: float
    time_to_expiry: float  # In years
    volatility: float
    is_call: bool = True
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.0


class GreeksResponse(BaseModel):
    """Full Greeks response."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    # Second order
    vanna: float
    volga: float
    charm: float
    speed: float
    color: float


class IVRequest(BaseModel):
    """Request for IV calculation."""
    price: float
    spot: float
    strike: float
    time_to_expiry: float
    is_call: bool = True
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.0


class IVResponse(BaseModel):
    """IV calculation response."""
    implied_volatility: float
    is_valid: bool


class OptionChainGreeksRequest(BaseModel):
    """Request for option chain Greeks."""
    trade_date: str
    spot_price: float
    volatility: float = 0.20
class OptionChainGreeksRequest(BaseModel):
    """Request for option chain Greeks."""
    trade_date: str
    spot_price: float
    volatility: float = 0.20
    use_market_iv: bool = False
    dataset_id: str = "510050_SH"


class OptionWithGreeks(BaseModel):
    """Single option with Greeks."""
    id: str
    type: Literal['call', 'put']
    strike: float
    expiry: str
    dte: int
    price: float
    iv: float
    # First order Greeks
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    # Second order Greeks
    vanna: float
    volga: float
    charm: float


class OptionChainGreeksResponse(BaseModel):
    """Response with full option chain Greeks."""
    trade_date: str
    spot_price: float
    count: int
    options: List[OptionWithGreeks]
    # Portfolio aggregates
    total_delta: float
    total_gamma: float
    total_vega: float
    total_theta: float


class SVIFitRequest(BaseModel):
    """Request for SVI surface fitting."""
    strikes: List[float]
    ivs: List[float]
    forward: float
    time_to_expiry: float


class SVIFitResponse(BaseModel):
    """SVI fit response."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    iv_fitted: List[float]
    rmse: float
    success: bool


class GreeksHeatmapRequest(BaseModel):
    """Request for Greeks heatmap data (Strike x DTE matrix)."""
    trade_date: str
    spot_price: float
    volatility: float = 0.20
    use_market_iv: bool = True
    trade_date: str
    spot_price: float
    volatility: float = 0.20
    use_market_iv: bool = True
    greek_type: Literal['delta', 'gamma', 'vega', 'theta'] = 'delta'
    option_type: Literal['call', 'put', 'both'] = 'call'
    dataset_id: str = "510050_SH"


class GreeksHeatmapResponse(BaseModel):
    """Greeks heatmap data in matrix format for 3D visualization."""
    strikes: List[float]
    dtes: List[int]
    z: List[List[float]]  # Greeks matrix [strike_idx][dte_idx]
    greek_type: str
    spot_price: float
    trade_date: str
    min_value: float
    max_value: float

# ============================================================
# API Endpoints
# ============================================================

@router.post("/calculate", response_model=GreeksResponse)
async def calculate_greeks(request: SingleGreeksRequest):
    """
    Calculate all Greeks for a single option.
    
    Includes first-order (Delta, Gamma, Vega, Theta, Rho) and 
    second-order (Vanna, Volga, Charm, Speed, Color) Greeks.
    """
    try:
        local_engine = PricingEngine(request.risk_free_rate, request.dividend_yield)
        result = local_engine.calculate_greeks(
            S=request.spot,
            K=request.strike,
            T=request.time_to_expiry,
            sigma=request.volatility,
            is_call=request.is_call
        )
        
        return GreeksResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/iv", response_model=IVResponse)
async def calculate_implied_volatility(request: IVRequest):
    """
    Calculate implied volatility from option price.
    
    Uses Newton-Raphson with Bisection fallback for robust convergence.
    """
    try:
        iv = quick_iv(
            price=request.price,
            S=request.spot,
            K=request.strike,
            T=request.time_to_expiry,
            r=request.risk_free_rate,
            q=request.dividend_yield,
            is_call=request.is_call
        )
        
        is_valid = not np.isnan(iv) and iv > 0
        
        return IVResponse(
            implied_volatility=round(iv, 6) if is_valid else 0.0,
            is_valid=is_valid
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chain", response_model=OptionChainGreeksResponse)
async def calculate_chain_greeks(request: OptionChainGreeksRequest):
    """
    Calculate Greeks for entire option chain on a given date.
    """
    from ..api.data import _load_date_data
    from datetime import datetime
    
    try:
        df = _load_date_data(request.trade_date, request.dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Prepare data
    trade_date = pd.to_datetime(request.trade_date)
    
    # Find expiry column
    expiry_col = None
    for col in ['expiry_date', 'maturity_date', 'expire_date']:
        if col in df.columns:
            expiry_col = col
            break
    
    if not expiry_col:
        raise HTTPException(status_code=500, detail="Expiry column not found")
    
    df['expiry_parsed'] = pd.to_datetime(df[expiry_col])
    df['dte'] = (df['expiry_parsed'] - trade_date).dt.days
    df = df[df['dte'] > 0].head(100)  # Limit to 100 options
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="No options found")
    
    # Prepare arrays
    n = len(df)
    S = np.full(n, request.spot_price, dtype=np.float64)
    
    strike_col = 'strike' if 'strike' in df.columns else 'strike_price'
    K = df[strike_col].values.astype(np.float64)
    T = (df['dte'].values / 365.0).astype(np.float64)
    
    r = np.full(n, engine.r, dtype=np.float64)
    q = np.full(n, engine.q, dtype=np.float64)
    
    # Option type
    type_col = None
    for col in ['type', 'call_put', 'cp_flag']:
        if col in df.columns:
            type_col = col
            break
    
    if type_col:
        is_call = df[type_col].astype(str).str.upper().isin(['C', 'CALL']).values
    else:
        is_call = np.ones(n, dtype=np.bool_)
    
    # Volatility
    if request.use_market_iv and 'close' in df.columns:
        from ..engines.pricing import _vectorized_implied_volatility
        prices = df['close'].values.astype(np.float64)
        sigma = _vectorized_implied_volatility(prices, S, K, T, r, q, is_call)
        sigma = np.where(np.isnan(sigma), request.volatility, sigma)
    else:
        sigma = np.full(n, request.volatility, dtype=np.float64)
    
    # Calculate Greeks
    from ..engines.pricing import _vectorized_all_greeks
    results = _vectorized_all_greeks(S, K, T, r, q, sigma, is_call)
    
    price, delta, gamma, vega, theta, rho, vanna, volga, charm, speed, color = results
    
    # Build response
    options = []
    # Use same ID column as data.py for consistency
    id_col = None
    for col in ['ts_code', 'symbol', 'order_book_id']:
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        id_col = df.columns[0]
    
    for i, (_, row) in enumerate(df.iterrows()):
        opt = OptionWithGreeks(
            id=str(row.get(id_col, i)),
            type='call' if is_call[i] else 'put',
            strike=float(K[i]),
            expiry=row['expiry_parsed'].strftime('%Y-%m-%d'),
            dte=int(row['dte']),
            price=round(float(price[i]), 6),
            iv=round(float(sigma[i]), 4),
            delta=round(float(delta[i]), 4),
            gamma=round(float(gamma[i]), 6),
            vega=round(float(vega[i]), 4),
            theta=round(float(theta[i]), 4),
            rho=round(float(rho[i]), 4),
            vanna=round(float(vanna[i]), 6),
            volga=round(float(volga[i]), 6),
            charm=round(float(charm[i]), 6),
        )
        options.append(opt)
    
    return OptionChainGreeksResponse(
        trade_date=request.trade_date,
        spot_price=request.spot_price,
        count=len(options),
        options=options,
        total_delta=round(float(np.sum(delta)), 4),
        total_gamma=round(float(np.sum(gamma)), 6),
        total_vega=round(float(np.sum(vega)), 4),
        total_theta=round(float(np.sum(theta)), 4),
    )


@router.post("/svi-fit", response_model=SVIFitResponse)
async def fit_svi_surface(request: SVIFitRequest):
    """
    Fit SVI (Stochastic Volatility Inspired) model to IV smile data.
    """
    try:
        strikes = np.array(request.strikes)
        ivs = np.array(request.ivs)
        
        result = engine.fit_iv_surface(
            strikes=strikes,
            ivs=ivs,
            forward=request.forward,
            T=request.time_to_expiry
        )
        
        return SVIFitResponse(
            a=round(result['a'], 6),
            b=round(result['b'], 6),
            rho=round(result['rho'], 6),
            m=round(result['m'], 6),
            sigma=round(result['sigma'], 6),
            iv_fitted=[round(float(iv), 6) for iv in result['iv_fitted']],
            rmse=round(result['rmse'], 6),
            success=result['success']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick")
async def quick_greeks_calc(
    spot: float = Query(..., description="Spot price"),
    strike: float = Query(..., description="Strike price"),
    tte: float = Query(..., description="Time to expiry in years"),
    vol: float = Query(..., description="Volatility"),
    is_call: bool = Query(True, description="Is call option"),
):
    """
    Quick Greeks calculation via GET request.
    Convenient for testing and simple integrations.
    """
    result = quick_greeks(spot, strike, tte, vol, is_call=is_call)
    return result


@router.post("/heatmap", response_model=GreeksHeatmapResponse)
async def get_greeks_heatmap(request: GreeksHeatmapRequest):
    """
    Get Greeks data in matrix format for 3D heatmap visualization.
    Returns Strike x DTE grid with selected Greek values.
    """
    from ..api.data import _load_date_data
    
    try:
        df = _load_date_data(request.trade_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    trade_date = pd.to_datetime(request.trade_date)
    
    # Find expiry column
    expiry_col = None
    for col in ['expiry_date', 'maturity_date', 'expire_date']:
        if col in df.columns:
            expiry_col = col
            break
    
    if not expiry_col:
        raise HTTPException(status_code=500, detail="Expiry column not found")
    
    df['expiry_parsed'] = pd.to_datetime(df[expiry_col])
    df['dte'] = (df['expiry_parsed'] - trade_date).dt.days
    df = df[df['dte'] > 0]
    
    # Filter by option type
    type_col = None
    for col in ['type', 'call_put', 'cp_flag']:
        if col in df.columns:
            type_col = col
            break
    
    if type_col and request.option_type != 'both':
        if request.option_type == 'call':
            df = df[df[type_col].astype(str).str.upper().isin(['C', 'CALL'])]
        else:
            df = df[df[type_col].astype(str).str.upper().isin(['P', 'PUT'])]
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="No options found")
    
    # Get unique strikes and DTEs
    strike_col = 'strike' if 'strike' in df.columns else 'strike_price'
    unique_strikes = sorted(df[strike_col].unique())
    unique_dtes = sorted(df['dte'].unique())
    
    # Limit grid size
    if len(unique_strikes) > 25:
        step = len(unique_strikes) // 25
        unique_strikes = unique_strikes[::step]
    if len(unique_dtes) > 15:
        step = len(unique_dtes) // 15
        unique_dtes = unique_dtes[::step]
    
    # Create Strike x DTE grid
    n_strikes = len(unique_strikes)
    n_dtes = len(unique_dtes)
    
    # Prepare arrays for vectorized calculation
    grid_size = n_strikes * n_dtes
    S = np.full(grid_size, request.spot_price, dtype=np.float64)
    K = np.zeros(grid_size, dtype=np.float64)
    T = np.zeros(grid_size, dtype=np.float64)
    
    idx = 0
    for strike in unique_strikes:
        for dte in unique_dtes:
            K[idx] = strike
            T[idx] = dte / 365.0
            idx += 1
    
    r = np.full(grid_size, engine.r, dtype=np.float64)
    q = np.full(grid_size, engine.q, dtype=np.float64)
    is_call = np.ones(grid_size, dtype=np.bool_) if request.option_type != 'put' else np.zeros(grid_size, dtype=np.bool_)
    
    # Get IV from data or use default
    if request.use_market_iv:
        sigma = np.full(grid_size, request.volatility, dtype=np.float64)
        # Try to interpolate from actual data
        for i, (strike, dte) in enumerate([(s, d) for s in unique_strikes for d in unique_dtes]):
            match = df[(df[strike_col] == strike) & (df['dte'] == dte)]
            if len(match) > 0 and 'close' in match.columns:
                from ..engines.pricing import quick_iv
                price = match['close'].values[0]
                iv = quick_iv(price, request.spot_price, strike, dte/365.0, engine.r, engine.q, is_call[i])
                if not np.isnan(iv) and iv > 0:
                    sigma[i] = iv
    else:
        sigma = np.full(grid_size, request.volatility, dtype=np.float64)
    
    # Calculate Greeks
    from ..engines.pricing import _vectorized_all_greeks
    results = _vectorized_all_greeks(S, K, T, r, q, sigma, is_call)
    
    price, delta, gamma, vega, theta, rho, vanna, volga, charm, speed, color = results
    
    # Select requested Greek
    greek_map = {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta
    }
    greek_values = greek_map[request.greek_type]
    
    # Reshape to matrix [strike_idx][dte_idx]
    z_matrix = []
    idx = 0
    for i in range(n_strikes):
        row = []
        for j in range(n_dtes):
            val = float(greek_values[idx])
            row.append(round(val, 6) if not np.isnan(val) else 0.0)
            idx += 1
        z_matrix.append(row)
    
    # Calculate min/max for colorscale
    flat_values = [v for row in z_matrix for v in row]
    min_val = min(flat_values) if flat_values else 0
    max_val = max(flat_values) if flat_values else 1
    
    return GreeksHeatmapResponse(
        strikes=[round(float(s), 4) for s in unique_strikes],
        dtes=[int(d) for d in unique_dtes],
        z=z_matrix,
        greek_type=request.greek_type,
        spot_price=request.spot_price,
        trade_date=request.trade_date,
        min_value=round(min_val, 6),
        max_value=round(max_val, 6)
    )
