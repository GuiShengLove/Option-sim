"""
Professional Pricing Engine for Options
========================================
Numba-accelerated Vectorized BSM Pricing & Greeks Module.

Core Features:
1. Complete Greeks: Delta, Gamma, Vega, Theta, Rho
2. Second-Order Greeks: Vanna, Volga (Vomma), Charm, Speed, Color
3. Implied Volatility Solver (Newton-Raphson with Bisection fallback)
4. IV Surface Fitting (SVI Parameterization)

Performance:
Target < 10ms for 1000+ contracts pricing.
"""

import numpy as np
import numba
from numba import float64, boolean
import pandas as pd
from typing import Union, Tuple, Optional
import math

# Constants
N_PRIME = 0.3989422804014327   # 1/sqrt(2*pi)
SQRT_2 = 1.4142135623730951
DAYS_PER_YEAR = 365.0

# ============================================================
# Numba Optimized Core Functions
# ============================================================

@numba.jit(nopython=True, fastmath=True, cache=True)
def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / SQRT_2))

@numba.jit(nopython=True, fastmath=True, cache=True)
def _norm_pdf(x: float) -> float:
    """Probability density function for the standard normal distribution."""
    return N_PRIME * math.exp(-0.5 * x * x)

@numba.jit(nopython=True, fastmath=True, cache=True)
def _bsm_d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
    """
    Calculate d1 and d2 for BSM with dividend yield.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
    """
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0, 0.0
    
    sqrt_T = math.sqrt(T)
    log_SK = math.log(S / K)
    
    d1 = (log_SK + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2

@numba.jit(nopython=True, fastmath=True, cache=True)
def _bsm_price_single(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    """Calculate BSM price for a single option with dividend yield."""
    if T <= 0:
        if is_call:
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
    
    d1, d2 = _bsm_d1_d2(S, K, T, r, q, sigma)
    
    exp_qT = math.exp(-q * T)
    exp_rT = math.exp(-r * T)
    
    if is_call:
        return S * exp_qT * _norm_cdf(d1) - K * exp_rT * _norm_cdf(d2)
    else:
        return K * exp_rT * _norm_cdf(-d2) - S * exp_qT * _norm_cdf(-d1)

@numba.jit(nopython=True, fastmath=True, cache=True)
def _vega_single(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Calculate Vega for Newton-Raphson IV solver."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    
    d1, _ = _bsm_d1_d2(S, K, T, r, q, sigma)
    sqrt_T = math.sqrt(T)
    exp_qT = math.exp(-q * T)
    
    return S * exp_qT * _norm_pdf(d1) * sqrt_T

# ============================================================
# Implied Volatility Solver
# ============================================================

@numba.jit(nopython=True, fastmath=True, cache=True)
def _implied_volatility_nr(
    price: float, S: float, K: float, T: float, r: float, q: float, 
    is_call: bool, max_iter: int = 100, tol: float = 1e-8
) -> float:
    """
    Newton-Raphson method for IV calculation with Bisection fallback.
    
    Returns sigma or NaN if failed.
    """
    if T <= 0.0 or price <= 0.0:
        return math.nan
    
    # Initial guess using Brenner-Subrahmanyam approximation
    # sigma_approx = sqrt(2*pi/T) * price / S
    sigma = math.sqrt(2 * math.pi / T) * price / S
    sigma = max(0.01, min(sigma, 5.0))  # Clamp to reasonable range
    
    # Intrinsic value check
    if is_call:
        intrinsic = max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
    else:
        intrinsic = max(0.0, K * math.exp(-r * T) - S * math.exp(-q * T))
    
    if price < intrinsic - 1e-10:
        return math.nan  # Price below intrinsic
    
    # Newton-Raphson iteration
    for i in range(max_iter):
        bs_price = _bsm_price_single(S, K, T, r, q, sigma, is_call)
        vega = _vega_single(S, K, T, r, q, sigma)
        
        diff = bs_price - price
        
        if abs(diff) < tol:
            return sigma
        
        if vega < 1e-12:
            break  # Switch to bisection
        
        sigma_new = sigma - diff / vega
        
        # Keep sigma in valid range
        if sigma_new <= 0.001 or sigma_new > 10.0:
            break
        
        sigma = sigma_new
    
    # Fallback to Bisection method
    sigma_low = 0.001
    sigma_high = 5.0
    
    for i in range(100):
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        bs_price = _bsm_price_single(S, K, T, r, q, sigma_mid, is_call)
        
        if abs(bs_price - price) < tol:
            return sigma_mid
        
        if bs_price > price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
        
        if sigma_high - sigma_low < 1e-10:
            break
    
    return 0.5 * (sigma_low + sigma_high)

# ============================================================
# Vectorized Greeks Calculation
# ============================================================

@numba.jit(nopython=True, fastmath=True, cache=True, parallel=True)
def _vectorized_all_greeks(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, 
    r: np.ndarray, q: np.ndarray, sigma: np.ndarray, is_call: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """
    Calculate all Greeks including second-order.
    
    Returns:
        price, delta, gamma, vega, theta, rho,
        vanna, volga, charm, speed, color
    """
    n = len(S)
    
    # First-order Greeks
    price = np.zeros(n, dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)
    gamma = np.zeros(n, dtype=np.float64)
    vega = np.zeros(n, dtype=np.float64)
    theta = np.zeros(n, dtype=np.float64)
    rho = np.zeros(n, dtype=np.float64)
    
    # Second-order Greeks
    vanna = np.zeros(n, dtype=np.float64)       # dDelta/dVol = dVega/dS
    volga = np.zeros(n, dtype=np.float64)       # dVega/dVol (Vomma)
    charm = np.zeros(n, dtype=np.float64)       # dDelta/dT (Delta decay)
    speed = np.zeros(n, dtype=np.float64)       # dGamma/dS
    color = np.zeros(n, dtype=np.float64)       # dGamma/dT
    
    for i in numba.prange(n):
        if T[i] <= 0:
            # Expired: intrinsic value only
            if is_call[i]:
                price[i] = max(0.0, S[i] - K[i])
                delta[i] = 1.0 if S[i] > K[i] else 0.0
            else:
                price[i] = max(0.0, K[i] - S[i])
                delta[i] = -1.0 if S[i] < K[i] else 0.0
            continue
        
        d1, d2 = _bsm_d1_d2(S[i], K[i], T[i], r[i], q[i], sigma[i])
        
        sqrt_T = math.sqrt(T[i])
        exp_qT = math.exp(-q[i] * T[i])
        exp_rT = math.exp(-r[i] * T[i])
        
        pdf_d1 = _norm_pdf(d1)
        cdf_d1 = _norm_cdf(d1)
        cdf_d2 = _norm_cdf(d2)
        cdf_neg_d1 = _norm_cdf(-d1)
        cdf_neg_d2 = _norm_cdf(-d2)
        
        sigma_sqrt_T = sigma[i] * sqrt_T
        
        # === Price ===
        if is_call[i]:
            price[i] = S[i] * exp_qT * cdf_d1 - K[i] * exp_rT * cdf_d2
        else:
            price[i] = K[i] * exp_rT * cdf_neg_d2 - S[i] * exp_qT * cdf_neg_d1
        
        # === First-Order Greeks ===
        # Delta
        if is_call[i]:
            delta[i] = exp_qT * cdf_d1
        else:
            delta[i] = -exp_qT * cdf_neg_d1
        
        # Gamma (same for call and put)
        gamma[i] = exp_qT * pdf_d1 / (S[i] * sigma_sqrt_T)
        
        # Vega (per 1% vol change)
        vega[i] = S[i] * exp_qT * pdf_d1 * sqrt_T / 100.0
        
        # Theta (daily)
        theta_common = -S[i] * exp_qT * pdf_d1 * sigma[i] / (2 * sqrt_T)
        if is_call[i]:
            theta[i] = (theta_common 
                       + q[i] * S[i] * exp_qT * cdf_d1 
                       - r[i] * K[i] * exp_rT * cdf_d2) / DAYS_PER_YEAR
        else:
            theta[i] = (theta_common 
                       - q[i] * S[i] * exp_qT * cdf_neg_d1 
                       + r[i] * K[i] * exp_rT * cdf_neg_d2) / DAYS_PER_YEAR
        
        # Rho (per 1% rate change)
        if is_call[i]:
            rho[i] = K[i] * T[i] * exp_rT * cdf_d2 / 100.0
        else:
            rho[i] = -K[i] * T[i] * exp_rT * cdf_neg_d2 / 100.0
        
        # === Second-Order Greeks ===
        # Vanna = dDelta/dVol = dVega/dSpot
        vanna[i] = -exp_qT * pdf_d1 * d2 / sigma[i] / 100.0
        
        # Volga (Vomma) = dVega/dVol
        volga[i] = vega[i] * d1 * d2 / sigma[i]
        
        # Charm = dDelta/dT (Delta decay per day)
        charm_term = 2 * (r[i] - q[i]) * T[i] - d2 * sigma_sqrt_T
        if is_call[i]:
            charm[i] = (-exp_qT * pdf_d1 * charm_term / (2 * T[i] * sigma_sqrt_T)
                       + q[i] * exp_qT * cdf_d1) / DAYS_PER_YEAR
        else:
            charm[i] = (-exp_qT * pdf_d1 * charm_term / (2 * T[i] * sigma_sqrt_T)
                       - q[i] * exp_qT * cdf_neg_d1) / DAYS_PER_YEAR
        
        # Speed = dGamma/dS
        speed[i] = -gamma[i] * (1 + d1 / sigma_sqrt_T) / S[i]
        
        # Color = dGamma/dT (per day)
        color_term = 2 * q[i] * T[i] + 1 + d1 * (2 * (r[i] - q[i]) * T[i] - d2 * sigma_sqrt_T) / sigma_sqrt_T
        color[i] = -exp_qT * pdf_d1 / (2 * S[i] * T[i] * sigma_sqrt_T) * color_term / DAYS_PER_YEAR
    
    return price, delta, gamma, vega, theta, rho, vanna, volga, charm, speed, color

# ============================================================
# Vectorized IV Calculation
# ============================================================

@numba.jit(nopython=True, fastmath=True, cache=True)
def _vectorized_implied_volatility(
    prices: np.ndarray, S: np.ndarray, K: np.ndarray, T: np.ndarray,
    r: np.ndarray, q: np.ndarray, is_call: np.ndarray
) -> np.ndarray:
    """Calculate IV for multiple options."""
    n = len(prices)
    iv = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        iv[i] = _implied_volatility_nr(
            prices[i], S[i], K[i], T[i], r[i], q[i], is_call[i]
        )
    
    return iv

# ============================================================
# SVI (Stochastic Volatility Inspired) Surface Fitting
# ============================================================

def svi_raw(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """
    SVI Raw parameterization for variance (sigma^2 * T).
    
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    
    Args:
        k: Log-moneyness ln(K/F)
        a, b, rho, m, sigma: SVI parameters
    
    Returns:
        Total variance w = sigma^2 * T
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi_surface(strikes: np.ndarray, iv: np.ndarray, forward: float, 
                    T: float, weights: Optional[np.ndarray] = None) -> dict:
    """
    Fit SVI model to IV smile data.
    
    Args:
        strikes: Array of strike prices
        iv: Array of implied volatilities
        forward: Forward price
        T: Time to expiry
        weights: Optional weights for fitting
    
    Returns:
        Dictionary with SVI parameters
    """
    from scipy.optimize import minimize
    
    # Log-moneyness
    k = np.log(strikes / forward)
    
    # Total variance
    w_market = iv ** 2 * T
    
    if weights is None:
        weights = np.ones_like(iv)
    
    # Initial guess
    a0 = np.mean(w_market)
    b0 = 0.1
    rho0 = -0.3
    m0 = 0.0
    sigma0 = 0.1
    
    def objective(params):
        a, b, rho, m, sig = params
        if b <= 0 or abs(rho) >= 1 or sig <= 0:
            return 1e10
        w_model = svi_raw(k, a, b, rho, m, sig)
        if np.any(w_model <= 0):
            return 1e10
        return np.sum(weights * (w_model - w_market) ** 2)
    
    # Bounds
    bounds = [
        (-0.5, 1.0),      # a
        (0.001, 1.0),     # b
        (-0.999, 0.999),  # rho
        (-0.5, 0.5),      # m
        (0.001, 1.0),     # sigma
    ]
    
    result = minimize(objective, [a0, b0, rho0, m0, sigma0], 
                     method='L-BFGS-B', bounds=bounds)
    
    a, b, rho, m, sig = result.x
    
    # Calculate fitted IV
    w_fitted = svi_raw(k, a, b, rho, m, sig)
    iv_fitted = np.sqrt(w_fitted / T)
    
    return {
        'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sig,
        'iv_fitted': iv_fitted,
        'rmse': np.sqrt(np.mean((iv_fitted - iv) ** 2)),
        'success': result.success
    }


# ============================================================
# Python Wrapper Class
# ============================================================

class PricingEngine:
    """
    Professional Option Pricing Engine with complete Greeks and IV calculation.
    
    Features:
    - BSM pricing with dividend yield
    - Complete first-order Greeks: Delta, Gamma, Vega, Theta, Rho
    - Second-order Greeks: Vanna, Volga, Charm, Speed, Color
    - IV solver with Newton-Raphson and Bisection fallback
    - SVI surface fitting
    """
    
    def __init__(self, risk_free_rate: float = 0.03, dividend_yield: float = 0.0):
        self.r = risk_free_rate
        self.q = dividend_yield
    
    def price(self, S: float, K: float, T: float, sigma: float, is_call: bool = True) -> float:
        """Calculate single option price."""
        return _bsm_price_single(S, K, T, self.r, self.q, sigma, is_call)
    
    def implied_volatility(self, price: float, S: float, K: float, T: float, is_call: bool = True) -> float:
        """Calculate implied volatility for a single option."""
        return _implied_volatility_nr(price, S, K, T, self.r, self.q, is_call)
    
    def calculate_greeks(self, S: float, K: float, T: float, sigma: float, is_call: bool = True) -> dict:
        """Calculate all Greeks for a single option."""
        S_arr = np.array([S], dtype=np.float64)
        K_arr = np.array([K], dtype=np.float64)
        T_arr = np.array([T], dtype=np.float64)
        r_arr = np.array([self.r], dtype=np.float64)
        q_arr = np.array([self.q], dtype=np.float64)
        sigma_arr = np.array([sigma], dtype=np.float64)
        is_call_arr = np.array([is_call], dtype=np.bool_)
        
        results = _vectorized_all_greeks(S_arr, K_arr, T_arr, r_arr, q_arr, sigma_arr, is_call_arr)
        
        return {
            'price': results[0][0],
            'delta': results[1][0],
            'gamma': results[2][0],
            'vega': results[3][0],
            'theta': results[4][0],
            'rho': results[5][0],
            'vanna': results[6][0],
            'volga': results[7][0],
            'charm': results[8][0],
            'speed': results[9][0],
            'color': results[10][0],
        }
    
    def calculate_all(self, df: pd.DataFrame, underlying_price: float, 
                      volatility: float = 0.20, use_iv: bool = False) -> pd.DataFrame:
        """
        Calculate Prices and Greeks for a DataFrame of options.
        
        Args:
            df: DataFrame with columns: strike, expiry_date, trade_date, type, [close for IV]
            underlying_price: Current spot price
            volatility: Default volatility to use (ignored if use_iv=True)
            use_iv: If True, calculate IV from market prices first
        
        Returns:
            DataFrame with all Greeks columns added
        """
        n = len(df)
        S = np.full(n, underlying_price, dtype=np.float64)
        K = df['strike'].values.astype(np.float64)
        
        # Time to expiry
        if 'trade_date' in df.columns and 'expiry_date' in df.columns:
            T_days = (pd.to_datetime(df['expiry_date']) - pd.to_datetime(df['trade_date'])).dt.days.values
        else:
            raise ValueError("DataFrame must contain 'trade_date' and 'expiry_date' columns")
        
        T = np.maximum(T_days / DAYS_PER_YEAR, 0.0).astype(np.float64)
        
        r = np.full(n, self.r, dtype=np.float64)
        q = np.full(n, self.q, dtype=np.float64)
        
        # Determine call/put
        type_col = df['type'].astype(str).str.upper()
        is_call = type_col.isin(['C', 'CALL']).values.astype(np.bool_)
        
        # Get volatility
        if use_iv and 'close' in df.columns:
            prices = df['close'].values.astype(np.float64)
            sigma = _vectorized_implied_volatility(prices, S, K, T, r, q, is_call)
            # Replace NaN with default
            sigma = np.where(np.isnan(sigma), volatility, sigma)
        else:
            sigma = np.full(n, volatility, dtype=np.float64)
        
        # Calculate all Greeks
        results = _vectorized_all_greeks(S, K, T, r, q, sigma, is_call)
        
        # Build result DataFrame
        result = df.copy()
        result['theoretical_price'] = results[0]
        result['iv'] = sigma
        result['delta'] = results[1]
        result['gamma'] = results[2]
        result['vega'] = results[3]
        result['theta'] = results[4]
        result['rho'] = results[5]
        result['vanna'] = results[6]
        result['volga'] = results[7]
        result['charm'] = results[8]
        result['speed'] = results[9]
        result['color'] = results[10]
        
        return result
    
    def fit_iv_surface(self, strikes: np.ndarray, ivs: np.ndarray, 
                       forward: float, T: float) -> dict:
        """Fit SVI model to IV smile data."""
        return fit_svi_surface(strikes, ivs, forward, T)


# ============================================================
# Convenience Functions
# ============================================================

def quick_greeks(S: float, K: float, T: float, sigma: float, 
                 r: float = 0.03, q: float = 0.0, is_call: bool = True) -> dict:
    """Quick standalone Greeks calculation."""
    engine = PricingEngine(r, q)
    return engine.calculate_greeks(S, K, T, sigma, is_call)


def quick_iv(price: float, S: float, K: float, T: float, 
             r: float = 0.03, q: float = 0.0, is_call: bool = True) -> float:
    """Quick standalone IV calculation."""
    return _implied_volatility_nr(price, S, K, T, r, q, is_call)
