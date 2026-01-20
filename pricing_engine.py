"""
Pricing Engine for Options
==========================
Numba-accelerated Vectorized BSM Pricing & Greeks Module.

Core Features:
1. Vectorized BSM Pricing (d1, d2, Call, Put)
2. Vectorized Greeks (Delta, Gamma, Theta, Vega, Rho)
3. Implied Volatility Solver (Newton-Raphson)

Performance:
Target < 10ms for 1000+ contracts pricing.
"""

import numpy as np
import numba
from numba import float64, boolean, types
import pandas as pd
from typing import Union, Tuple

# Constants
N_PRIME = 0.3989422804014327   # 1/sqrt(2*pi)

# ============================================================
# Numba Optimized Core Functions (No Python Objects)
# ============================================================

@numba.jit(nopython=True, fastmath=True, cache=True)
def _norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1.0 + numba.cuda.erf(x / np.sqrt(2.0))) if False else 0.5 * (1.0 + np.math.erf(x / 1.4142135623730951))

@numba.jit(nopython=True, fastmath=True, cache=True)
def _norm_pdf(x):
    """Probability density function for the standard normal distribution."""
    return N_PRIME * np.exp(-0.5 * x * x)

@numba.jit(nopython=True, fastmath=True, cache=True)
def _bsm_d1_d2(S, K, T, r, sigma):
    """Calculate d1 and d2 for BSM."""
    # Avoid division by zero and log of zero
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0, 0.0
    
    sqrt_T = np.sqrt(T)
    log_SK = np.log(S / K)
    
    d1 = (log_SK + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2

@numba.jit(nopython=True, fastmath=True, cache=True)
def _vectorized_bsm_price(S, K, T, r, sigma, is_call):
    """
    Vectorized BSM pricing calculation.
    arrays must be same length numpy arrays.
    """
    n = len(S)
    prices = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if T[i] <= 0:
            # Expired: max(S-K, 0) for call, max(K-S, 0) for put
            if is_call[i]:
                prices[i] = max(0.0, S[i] - K[i])
            else:
                prices[i] = max(0.0, K[i] - S[i])
            continue
            
        d1, d2 = _bsm_d1_d2(S[i], K[i], T[i], r[i], sigma[i])
        
        # d1/d2 zero return check handled inside _bsm_d1_d2 but if T>0 and we get 0,0 check logic?
        # Actually simplest is just calc:
        
        norm_d1 = _norm_cdf(d1)
        norm_d2 = _norm_cdf(d2)
        
        if is_call[i]:
            prices[i] = S[i] * norm_d1 - K[i] * np.exp(-r[i] * T[i]) * norm_d2
        else:
            prices[i] = K[i] * np.exp(-r[i] * T[i]) * _norm_cdf(-d2) - S[i] * _norm_cdf(-d1)
            
    return prices

@numba.jit(nopython=True, fastmath=True, cache=True)
def _vectorized_greeks(S, K, T, r, sigma, is_call):
    """
    Calculate Greeks: Delta, Gamma, Vega, Theta, Rho.
    Returns tuple of arrays.
    """
    n = len(S)
    delta = np.zeros(n, dtype=np.float64)
    gamma = np.zeros(n, dtype=np.float64)
    vega = np.zeros(n, dtype=np.float64)
    theta = np.zeros(n, dtype=np.float64)
    rho = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if T[i] <= 0:
            # Expired options have no optionality
            # Delta is 0 or 1/-1 depending on ITM, Gamma/Vega/Theta/Rho = 0
            if is_call[i]:
                delta[i] = 1.0 if S[i] > K[i] else 0.0
            else:
                delta[i] = -1.0 if S[i] < K[i] else 0.0
            continue

        d1, d2 = _bsm_d1_d2(S[i], K[i], T[i], r[i], sigma[i])
        
        pdf_d1 = _norm_pdf(d1)
        cdf_d1 = _norm_cdf(d1)
        cdf_neg_d1 = _norm_cdf(-d1)
        cdf_d2 = _norm_cdf(d2)
        cdf_neg_d2 = _norm_cdf(-d2)
        
        sqrt_T = np.sqrt(T[i])
        exp_rT = np.exp(-r[i] * T[i])
        
        # Common Gamma / Vega
        gamma[i] = pdf_d1 / (S[i] * sigma[i] * sqrt_T)
        vega[i] = S[i] * pdf_d1 * sqrt_T / 100.0  # Divided by 100 for percentage change
        
        if is_call[i]:
            delta[i] = cdf_d1
            theta[i] = (- (S[i] * pdf_d1 * sigma[i]) / (2 * sqrt_T) 
                        - r[i] * K[i] * exp_rT * cdf_d2) / 365.0 # Daily Theta
            rho[i] = (K[i] * T[i] * exp_rT * cdf_d2) / 100.0
        else:
            delta[i] = cdf_d1 - 1.0
            theta[i] = (- (S[i] * pdf_d1 * sigma[i]) / (2 * sqrt_T) 
                        + r[i] * K[i] * exp_rT * cdf_neg_d2) / 365.0 # Daily Theta
            rho[i] = (-K[i] * T[i] * exp_rT * cdf_neg_d2) / 100.0
            
    return delta, gamma, vega, theta, rho

# ============================================================
# Python Wrapper Class
# ============================================================

class PricingEngine:
    """
    High-level API for Option Pricing and Greeks calculation.
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        self.r = risk_free_rate

    def calculate_all(self, df: pd.DataFrame, underlying_price: float, volatility: float = 0.20) -> pd.DataFrame:
        """
        Calculate Prices and Greeks for a DataFrame of options.
        
        Expects df to have columns:
        - strike
        - expiry_date
        - type (C/P)
        
        Args:
            df: Input DataFrame
            underlying_price: Current price of the underlying asset
            volatility: Annualized volatility (e.g. 0.20 for 20%)
            
        Returns:
            DataFrame with added columns: theoretical_price, delta, gamma, vega, theta, rho
        """
        # Prepare numpy arrays for Numba
        n = len(df)
        S = np.full(n, underlying_price, dtype=np.float64)
        K = df['strike'].values.astype(np.float64)
        
        # Time to expiry in years
        # Use simple day count for specialized calc, assuming current date is not provided
        # Ideal: T = (expiry - current) / 365
        # Since usage might be inside a backtest loop where we know current date, we expect it passed or calc'd.
        # But here we only have expiry_date. We need 'current_date' or 'trade_date' in df.
        
        if 'trade_date' in df.columns:
            T_days = (df['expiry_date'] - df['trade_date']).dt.days.values
        else:
            # Fallback or error? Let's check if we can infer or pass current_date.
            # For now assume trade_date is in df as per schema.
            raise ValueError("DataFrame must contain 'trade_date' column for Time to Expiry calculation.")

        # Ensure non-negative T
        T = np.maximum(T_days / 365.0, 0.0).astype(np.float64)
        
        r = np.full(n, self.r, dtype=np.float64)
        sigma = np.full(n, volatility, dtype=np.float64)
        
        # Convert type to boolean (True=Call, False=Put)
        # Handle 'C'/'P' categories or strings
        is_call = (df['type'].astype(str).str.upper() == 'C').values
        
        # Calculate Prices
        prices = _vectorized_bsm_price(S, K, T, r, sigma, is_call)
        
        # Calculate Greeks
        delta, gamma, vega, theta, rho = _vectorized_greeks(S, K, T, r, sigma, is_call)
        
        # Assign back to DataFrame
        result = df.copy()
        result['theoretical_price'] = prices
        result['delta'] = delta
        result['gamma'] = gamma
        result['vega'] = vega
        result['theta'] = theta
        result['rho'] = rho
        
        return result

    def calculate_iv(self, df: pd.DataFrame, underlying_price: float) -> np.ndarray:
        """
        Calculate Implied Volatility.
        (Placeholder for Newton-Raphson implementation)
        """
        # TODO: Implement Newton-Raphson solver kernel
        return np.zeros(len(df))

# ============================================================
# Benchmark / Test
# ============================================================

if __name__ == "__main__":
    import time
    
    print("Initializing PricingEngine...")
    engine = PricingEngine()
    
    # Mock data
    N = 100_000
    print(f"Generating {N} mock options...")
    df = pd.DataFrame({
        'trade_date': [pd.Timestamp('2022-01-01')] * N,
        'expiry_date': [pd.Timestamp('2022-02-01')] * N,
        'strike': np.random.uniform(2.0, 3.0, N),
        'type': np.random.choice(['C', 'P'], N)
    })
    S0 = 2.5
    vol = 0.25
    
    print("Testing JIT compilation & Pricing Speed...")
    start = time.time()
    res = engine.calculate_all(df, S0, vol)
    end = time.time()
    
    print(f"Time taken: {(end - start)*1000:.2f} ms")
    print("\nSample Output:")
    print(res[['strike', 'type', 'theoretical_price', 'delta', 'gamma']].head())
