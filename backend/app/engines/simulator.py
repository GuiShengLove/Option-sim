"""
Simulation Engine
=================
Monte Carlo Path Generator with advanced stochastic models.
Supported: GBM, Heston, Merton Jump Diffusion.

Features:
- Numba JIT acceleration for CPU
- Optional CuPy GPU acceleration
- Historical volatility calibration
- Reproducible random seeds
"""
import numpy as np
import numba
import os
from typing import Optional, Dict, Tuple

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

@numba.jit(nopython=True, cache=True)
def _generate_gbm(S0, mu, sigma, T, dt, n_paths):
    """Geometric Brownian Motion."""
    n_steps = int(T / dt)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for i in range(n_paths):
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal()
            paths[i, t] = paths[i, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            
    return paths

@numba.jit(nopython=True, cache=True, parallel=True)
def _generate_gbm_vectorized(S0, mu, sigma, T, dt, n_paths):
    """Optimized GBM using vectorized operations (parallel)."""
    n_steps = int(T / dt)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate all random numbers at once
    # Using parallel for outer loop
    for i in numba.prange(n_paths):
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal()
            paths[i, t] = paths[i, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            
    return paths

@numba.jit(nopython=True, cache=True)
def _generate_heston(S0, v0, mu, kappa, theta, xi, rho, T, dt, n_paths):
    """
    Heston Stochastic Volatility Model.
    dS = mu*S*dt + sqrt(v)*S*dW1
    dv = kappa*(theta-v)*dt + xi*sqrt(v)*dW2
    dW1*dW2 = rho*dt
    """
    n_steps = int(T / dt)
    paths = np.zeros((n_paths, n_steps + 1))
    vols = np.zeros((n_paths, n_steps + 1)) # Variance process
    paths[:, 0] = S0
    vols[:, 0] = v0
    
    for i in range(n_paths):
        for t in range(1, n_steps + 1):
            # Correlated Brownian Motions
            z1 = np.random.standard_normal()
            z2 = np.random.standard_normal()
            z2 = rho * z1 + np.sqrt(1 - rho**2) * z2
            
            # Update Variance (Full Truncation Scheme to keep v > 0)
            vt = vols[i, t-1]
            vt_plus = vt + kappa * (theta - vt) * dt + xi * np.sqrt(max(0, vt)) * np.sqrt(dt) * z2
            vols[i, t] = max(0, vt_plus)
            
            # Update Price
            paths[i, t] = paths[i, t-1] * np.exp(
                (mu - 0.5 * vt) * dt + np.sqrt(max(0, vt)) * np.sqrt(dt) * z1
            )
            
    return paths

@numba.jit(nopython=True, cache=True)
def _generate_mjd(S0, mu, sigma, lam, m, v, T, dt, n_paths):
    """
    Merton Jump Diffusion.
    dS/S = (mu - lam*k)*dt + sigma*dW + (Y-1)dN
    """
    n_steps = int(T / dt)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    # Jump parameters
    # lam: jump intensity (number of jumps per year)
    # m: mean of log jump size
    # v: std of log jump size
    # k = E[Y-1] = exp(m + v^2/2) - 1
    
    k = np.exp(m + 0.5 * v**2) - 1
    
    for i in range(n_paths):
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal()
            
            # Diffusion component
            diffusion = (mu - 0.5 * sigma**2 - lam * k) * dt + sigma * np.sqrt(dt) * z
            
            # Jump component
            # Poisson process for number of jumps in dt
            n_jumps = np.random.poisson(lam * dt)
            jump_factor = 0.0
            if n_jumps > 0:
                # Sum of log-normal jumps
                # J = sum(N(m, v))
                jump_sum = 0.0
                for _ in range(n_jumps):
                    jump_sum += np.random.normal(m, v)
                jump_factor = jump_sum
            
            paths[i, t] = paths[i, t-1] * np.exp(diffusion + jump_factor)
            
    return paths


@numba.jit(nopython=True, cache=True)
def _generate_garch(S0, mu, omega, alpha, beta, T, dt, n_paths):
    """
    GARCH(1,1) Model.
    r_t = mu * dt + sqrt(h_t) * z_t
    h_t = omega + alpha * epsilon_{t-1}^2 + beta * h_{t-1}
    
    Args:
        S0: Initial price
        mu: Drift rate (annual)
        omega: Constant term in variance equation
        alpha: ARCH coefficient
        beta: GARCH coefficient
        T: Time in years
        dt: Time step
        n_paths: Number of paths
    """
    n_steps = int(T / dt)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    # Initial variance (long-run variance under GARCH)
    # h_bar = omega / (1 - alpha - beta)
    persistence = alpha + beta
    if persistence < 1.0:
        h_bar = omega / (1 - persistence)
    else:
        h_bar = 0.04  # Default 20% annualized vol squared
    
    for i in range(n_paths):
        h_t = h_bar  # Start at long-run variance
        epsilon_prev = 0.0
        
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal()
            
            # Update variance
            h_t = omega + alpha * epsilon_prev**2 + beta * h_t
            h_t = max(h_t, 1e-10)  # Ensure positive variance
            
            # Daily return
            epsilon = np.sqrt(h_t) * z
            daily_return = mu * dt + epsilon
            
            # Update price
            paths[i, t] = paths[i, t-1] * np.exp(daily_return)
            
            epsilon_prev = epsilon
            
    return paths


def _generate_gbm_gpu(S0, mu, sigma, T, dt, n_paths):
    """GPU-accelerated GBM using CuPy."""
    if not HAS_GPU:
        raise RuntimeError("CuPy not available for GPU acceleration")
    
    n_steps = int(T / dt)
    
    # Generate all random numbers on GPU
    z = cp.random.standard_normal((n_paths, n_steps))
    
    # Compute log returns
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion
    
    # Cumulative sum for paths
    log_paths = cp.zeros((n_paths, n_steps + 1))
    log_paths[:, 0] = np.log(S0)
    log_paths[:, 1:] = np.log(S0) + cp.cumsum(log_returns, axis=1)
    
    paths = cp.exp(log_paths)
    
    return cp.asnumpy(paths)


class Simulator:
    """
    Simulation Engine Wrapper.
    
    Features:
    - Multiple stochastic models (GBM, Heston, MJD)
    - Historical volatility calibration
    - GPU acceleration (optional)
    - Reproducible random seeds
    """
    def __init__(self):
        self._seed = None
        self._use_gpu = False
        
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._seed = seed
        np.random.seed(seed)
        if HAS_GPU:
            cp.random.seed(seed)
    
    def enable_gpu(self, enable: bool = True):
        """Enable or disable GPU acceleration."""
        if enable and not HAS_GPU:
            print("Warning: CuPy not installed. GPU acceleration not available.")
            self._use_gpu = False
        else:
            self._use_gpu = enable
    
    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available."""
        return HAS_GPU
        
    def generate_paths(self, model: str, S0: float, T_days: int, n_paths: int, params: dict):
        """
        Generate Asset Paths.
        
        Args:
            model: 'GBM', 'HESTON', or 'MJD'
            S0: Initial price
            T_days: Number of trading days
            n_paths: Number of paths to generate
            params: Model-specific parameters
            
        Returns:
            np.ndarray of shape (n_paths, T_days + 1)
        """
        dt = 1/252.0
        T_years = T_days / 252.0
        
        # Set seed if provided
        if params.get('seed') is not None:
            self.set_random_seed(params['seed'])
        
        if model == 'GBM':
            mu = params.get('mu', 0.05)
            sigma = params.get('sigma', 0.20)
            
            # Use GPU if enabled and large simulation
            if self._use_gpu and HAS_GPU and n_paths >= 1000:
                return _generate_gbm_gpu(S0, mu, sigma, T_years, dt, n_paths)
            elif n_paths >= 100:
                # Use parallel version for larger simulations
                return _generate_gbm_vectorized(S0, mu, sigma, T_years, dt, n_paths)
            else:
                return _generate_gbm(S0, mu, sigma, T_years, dt, n_paths)
            
        elif model == 'HESTON':
            v0 = params.get('v0', 0.04) # Initial variance (0.2^2)
            mu = params.get('mu', 0.05)
            kappa = params.get('kappa', 2.0) # Mean reversion speed
            theta = params.get('theta', 0.04) # Long run variance
            xi = params.get('xi', 0.3) # Vol of Vol
            rho = params.get('rho', -0.7) # Correlation
            return _generate_heston(S0, v0, mu, kappa, theta, xi, rho, T_years, dt, n_paths)
            
        elif model == 'MJD':
            mu = params.get('mu', 0.05)
            sigma = params.get('sigma', 0.20)
            lam = params.get('lam', 0.75) # Jumps per year
            m = params.get('m', -0.02)
            v = params.get('v', 0.1)
            return _generate_mjd(S0, mu, sigma, lam, m, v, T_years, dt, n_paths)
        
        elif model == 'GARCH':
            mu = params.get('mu', 0.05)
            omega = params.get('omega', 0.000001)
            alpha = params.get('alpha', 0.1)
            beta = params.get('beta', 0.85)
            return _generate_garch(S0, mu, omega, alpha, beta, T_years, dt, n_paths)
            
        else:
            raise ValueError(f"Unknown model: {model}")
    
    @staticmethod
    def calibrate_from_history(prices: np.ndarray, annualize: bool = True) -> Tuple[float, float]:
        """
        Calibrate mu and sigma from historical prices.
        
        Args:
            prices: Array of historical prices (oldest to newest)
            annualize: Whether to annualize the parameters
            
        Returns:
            (mu, sigma) tuple
        """
        if len(prices) < 2:
            return 0.05, 0.20  # Default values
        
        # Calculate log returns
        log_returns = np.diff(np.log(prices))
        
        # Daily statistics
        daily_mu = np.mean(log_returns)
        daily_sigma = np.std(log_returns, ddof=1)
        
        if annualize:
            # Annualize using 252 trading days
            mu = daily_mu * 252
            sigma = daily_sigma * np.sqrt(252)
        else:
            mu = daily_mu
            sigma = daily_sigma
        
        return float(mu), float(sigma)

