"""
Market Path Simulator
=====================
Generates future underlying price paths using Monte Carlo methods.

Models:
1. Geometric Brownian Motion (GBM)
2. Merton Jump Diffusion (MJD)
"""

import numpy as np
import pandas as pd
import numba
from typing import Tuple, Optional

@numba.jit(nopython=True, cache=True)
def _gbm_paths(S0, mu, sigma, dt, n_steps, n_paths):
    """
    Generate GBM paths via Numba.
    S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
    """
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for i in range(n_paths):
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal()
            # Euler-Maruyama discretization
            # ln S_t = ln S_{t-1} + (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
            
            paths[i, t] = paths[i, t-1] * np.exp((mu - 0.5 * sigma * sigma) * dt + 
                                                 sigma * np.sqrt(dt) * z)
    return paths

@numba.jit(nopython=True, cache=True)
def _mjd_paths(S0, mu, sigma, dt, n_steps, n_paths, lambda_j, mu_j, sigma_j):
    """
    Generate Jump Diffusion paths.
    lambda_j: Jump intensity (expected jumps per year)
    mu_j: Mean jump size (log return)
    sigma_j: Jump size volatility
    """
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for i in range(n_paths):
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal()
            
            # Jump component
            # Number of jumps in dt (Poisson process)
            # Approximation: small dt, mostly 0 or 1 jump. 
            # Poisson(lambda * dt)
            n_jumps = np.random.poisson(lambda_j * dt)
            jump_factor = 1.0
            
            if n_jumps > 0:
                # Sum of random jumps
                log_jump_sum = 0.0
                for _ in range(n_jumps):
                    log_jump_sum += np.random.normal(mu_j, sigma_j)
                jump_factor = np.exp(log_jump_sum)
            
            # Combine GBM + Jump
            # S_t = S_{t-1} * exp(drift + diffusion) * Jump
            # Drift adjustment for jumps usually needed if mu is total drift, 
            # here mu is diffusion drift.
            
            gbm_term = np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z)
            paths[i, t] = paths[i, t-1] * gbm_term * jump_factor
            
    return paths

class PathGenerator:
    """
    Monte Carlo Path Generator for Underlying Asset.
    """
    
    def __init__(self, S0: float, mu: float, sigma: float, T_days: int):
        """
        Args:
            S0: Initial price
            mu: Annualized drift (e.g. 0.05 for 5%)
            sigma: Annualized volatility (e.g. 0.20 for 20%)
            T_days: Number of days to simulate
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T_days = T_days
        self.dt = 1.0 / 252.0  # Daily steps assuming 252 trading days
        
    def generate_gbm(self, n_paths: int) -> np.ndarray:
        """
        Generate Geometric Brownian Motion paths.
        Returns: (n_paths, T_days + 1) array
        """
        return _gbm_paths(self.S0, self.mu, self.sigma, self.dt, self.T_days, n_paths)
    
    def generate_mjd(
        self, 
        n_paths: int, 
        lambda_j: float = 0.5, 
        mu_j: float = -0.05, 
        sigma_j: float = 0.1
    ) -> np.ndarray:
        """
        Generate Merton Jump Diffusion paths.
        
        Args:
            n_paths: Number of paths
            lambda_j: Jumps per year
            mu_j: Mean log-jump size (e.g. -0.05 means -5% drop on average)
            sigma_j: Volatility of jump size
        """
        return _mjd_paths(
            self.S0, self.mu, self.sigma, self.dt, self.T_days, n_paths,
            lambda_j, mu_j, sigma_j
        )

# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print("Testing PathGenerator...")
    gen = PathGenerator(3.0, 0.05, 0.20, T_days=20)
    
    print("Generating 5 GBM paths...")
    paths = gen.generate_gbm(5)
    print("Shape:", paths.shape)
    print("End prices:", paths[:, -1])
    
    print("\nGenerating 5 MJD paths (crash heavy)...")
    # High crash probability for testing
    j_paths = gen.generate_mjd(5, lambda_j=10.0, mu_j=-0.1, sigma_j=0.05)
    print("End prices:", j_paths[:, -1])
