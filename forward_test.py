"""
Forward Test Orchestrator
=========================
Runs Monte Carlo simulations for Option Strategies.

Process:
1. Generate N paths (Simulator)
2. Generate N Option Chains (SyntheticGenerator)
3. Run Backtest on each path (BacktestEngine)
4. Aggregate Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Type
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# Note: ProcessPoolExecutor might be hard with Numba/Pickle. Using ThreadPool for IO/CPU mix or sequential for safety first.
# Numba releases GIL, so threading might work.

from simulator import PathGenerator
from synthetic_generator import SyntheticGenerator, MemoryDataLoader
from data_loader import DataLoader
from pricing_engine import PricingEngine
from backtest_engine import BacktestEngine
from strategies.base_strategy import BaseStrategy

class ForwardTestEngine:
    """
    Monte Carlo Forward Test Engine.
    """
    
    def __init__(self, data_dir: str = "data/510050_SH"):
        self.data_dir = data_dir
        
        # Init components
        self.disk_loader = DataLoader(data_dir)
        self.pricer = PricingEngine()
        self.path_gen = None # Init per run
        self.syn_gen = SyntheticGenerator(self.disk_loader, self.pricer)
        
    def run_simulation(
        self,
        strategy_cls: Type[BaseStrategy],
        strategy_config: Dict,
        S0: float,
        mu: float,
        sigma: float,
        days: int,
        n_paths: int = 10,
        model: str = 'GBM',
        base_date: str = '2020-01-02'
    ) -> pd.DataFrame:
        """
        Run Forward Test simulation.
        
        Args:
            strategy_cls: Strategy class
            strategy_config: Strategy config
            S0: Initial Underlying Price
            mu: Drift
            sigma: Volatility
            days: Simulation duration
            n_paths: Number of MC paths
            model: 'GBM' or 'MJD'
            base_date: Template date for option chain structure
            
        Returns:
            DataFrame containing PnL stats for each path
        """
        print(f"\n🌪️ Starting Forward Test ({n_paths} paths, {days} days)")
        
        # 1. Generate Paths
        self.path_gen = PathGenerator(S0, mu, sigma, days)
        if model == 'MJD':
            paths = self.path_gen.generate_mjd(n_paths)
        else:
            paths = self.path_gen.generate_gbm(n_paths)
            
        print(f"   ✓ Paths Generated. Shape: {paths.shape}")
        
        # 2. Sequential Execution (MVP)
        # To avoid complexity with Pickling Numba functions in Multiprocessing on Windows
        results = []
        
        start_sim_date = '2025-01-01' # Hypothetical future start
        
        for i in range(n_paths):
            print(f"   ► Simulating Path {i+1}/{n_paths}...", end='\r')
            
            # A. Generate Synthetic Option Data
            S_path = paths[i]
            daily_dfs = self.syn_gen.generate_chain_for_path(S_path, base_date, start_sim_date)
            
            if not daily_dfs:
                print(f"   ⚠️ Path {i+1} failed to generate data")
                continue
                
            # Create Memory Loader
            data_map = {}
            for df in daily_dfs:
                if not df.empty:
                    d_str = df.iloc[0]['trade_date'].strftime('%Y-%m-%d')
                    data_map[d_str] = df
            
            if not data_map:
                continue
                
            mem_loader = MemoryDataLoader(data_map)
            
            # B. Run Backtest
            # Inject mem_loader into a new BacktestEngine
            # We need to subclass or modify BacktestEngine to accept loader?
            # BacktestEngine currently init loader in __init__.
            # Let's Modify BacktestEngine to iterate over dates or accept loader instance.
            # Workaround: Subclass or monkey-patch. 
            # Or better: Instantiate BacktestEngine and REPLACE its loader.
            
            bt_engine = BacktestEngine(self.data_dir) 
            bt_engine.loader = mem_loader # Dependency Injection
            
            # Silence output
            import sys, io
            # saved_stdout = sys.stdout
            # sys.stdout = io.StringIO()
            
            try:
                available_dates = mem_loader.get_available_dates()
                res_df = bt_engine.run(
                    strategy_cls, 
                    strategy_config, 
                    available_dates[0], 
                    available_dates[-1]
                )
                
                final_equity = res_df['equity'].iloc[-1]
                pnl = final_equity - bt_engine.initial_cash
                ret = pnl / bt_engine.initial_cash
                
                results.append({
                    'path_id': i,
                    'final_equity': final_equity,
                    'pnl': pnl,
                    'return': ret,
                    'max_drawdown': self._calc_max_dd(res_df['equity']),
                    'final_S': S_path[-1]
                })
                
            except Exception as e:
                print(f"\n   ❌ Path {i+1} Error: {e}")
            finally:
                # sys.stdout = saved_stdout
                pass
                
        print(f"\n   ✅ Simulation Complete. Processed {len(results)} paths.")
        return pd.DataFrame(results)

    def _calc_max_dd(self, equity_curve):
        """Calculate Max Drawdown."""
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak
        return dd.min()

# ============================================================
# Demo Run
# ============================================================
if __name__ == "__main__":
    from strategies.base_strategy import DemoStrategy
    
    engine = ForwardTestEngine()
    
    # Run small simulation
    results = engine.run_simulation(
        strategy_cls=DemoStrategy,
        strategy_config={'iv_threshold': 0.15},
        S0=3.0,
        mu=0.05,
        sigma=0.20,
        days=10,
        n_paths=5,
        base_date='2020-01-02'
    )
    
    print("\nSummary Stats:")
    print(results.describe())
