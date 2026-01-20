"""
Synthetic Option Data Generator
===============================
Generates consistent option chains for simulated underlying paths.

Logic:
1. Takes a "Template" Option Chain from a real date (to preserve strike structure & open interest distribution).
2. For each step in simulated path:
    - Update underlying price (S)
    - Update Time to Expiry (T)
    - Recalculate Prices using BSM (PricingEngine)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

from data_loader import DataLoader
from pricing_engine import PricingEngine

class SyntheticGenerator:
    """
    Generates synthetic option market data.
    """
    
    def __init__(self, data_loader: DataLoader, pricing_engine: PricingEngine):
        self.loader = data_loader
        self.pricer = pricing_engine
        
    def generate_chain_for_path(
        self, 
        S_path: np.ndarray, 
        base_date: str, 
        start_date: str
    ) -> List[pd.DataFrame]:
        """
        Generate a list of daily DataFrames for a single price path.
        
        Args:
            S_path: Array of underlying prices (length N)
            base_date: Date to fetch template structure from (YYYY-MM-DD)
            start_date: Start date for the simulation (YYYY-MM-DD)
            
        Returns:
            List of DataFrames (one per trading day)
        """
        # 1. Load Template
        template_df = self.loader.load_single_date(base_date)
        
        if template_df.empty:
            raise ValueError(f"No template data found for {base_date}")
            
        # Calc logic relative to base_date
        base_dt = pd.to_datetime(base_date)
        
        # Clean template: keep only static info
        # We need expiry_date to calc DTE
        static_cols = ['symbol', 'strike', 'type', 'expiry_date', 'contract_multiplier']
        available_cols = [c for c in static_cols if c in template_df.columns]
        template = template_df[available_cols].copy()
        
        # Calculate Days to Expiry (DTE) for template
        template['dte'] = (template['expiry_date'] - base_dt).dt.days
        
        # 2. Simulation Loop
        synthetic_days = []
        current_dt = pd.to_datetime(start_date)
        
        for t, S_t in enumerate(S_path):
            sim_date = current_dt + timedelta(days=t)
            
            # Create daily dataframe
            daily_df = template.copy()
            daily_df['trade_date'] = sim_date
            
            # Shift Expiry Date: new_expiry = sim_date + dte
            # Note: This updates the expiry date to be relative to the SIMULATION date
            # effectively preserving the "structure" of the chain (e.g. 20 days out, 50 days out...)
            daily_df['expiry_date'] = sim_date + pd.to_timedelta(daily_df['dte'], unit='D')
            
            # Remove DTE helper
            # daily_df.drop(columns=['dte'], inplace=True) 
            
            # Filter out expired options or invalid DTE <= 0
            daily_df = daily_df[daily_df['dte'] > 0] # Filter by original positive DTE logic
            
            if daily_df.empty:
                synthetic_days.append(daily_df)
                continue
            
            # Recalculate Prices
            # PricingEngine handles T calculation internally from trade_date & expiry_date
            
            # Assuming constant volatility for MVP
            # In a real engine, we'd map S_t & T to a Vol Surface
            implied_vol = 0.20 
            
            priced_df = self.pricer.calculate_all(daily_df, S_t, implied_vol)
            
            # Map theoretical price to 'close', 'open', 'high', 'low'
            # For backtesting, we assume Open = Close = Theoretical (No intraday volatility in daily step)
            p = priced_df['theoretical_price']
            priced_df['close'] = p
            priced_df['open'] = p
            priced_df['high'] = p
            priced_df['low'] = p
            priced_df['volume'] = 0 # Synthetic volume? Or copy?
            
            synthetic_days.append(priced_df)
            
        return synthetic_days

class MemoryDataLoader(DataLoader):
    """
    Adapter to feed synthetic data into BacktestEngine.
    Overrides load_single_date to return in-memory data.
    """
    def __init__(self, data_map: Dict[str, pd.DataFrame]):
        """
        Args:
            data_map: Dictionary { 'YYYY-MM-DD': DataFrame }
        """
        # Do not call super init which checks disk
        self._data_map = data_map
        self._available_dates = sorted(list(data_map.keys()))
        
    def load_single_date(self, date: str) -> pd.DataFrame:
        if date not in self._data_map:
            raise ValueError(f"Synthetic data not found for {date}")
        return self._data_map[date]
    
    def get_available_dates(self) -> List[str]:
        return self._available_dates

# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    from simulator import PathGenerator
    
    print("Testing SyntheticGenerator...")
    
    # Init dependencies
    disk_loader = DataLoader("data/510050_SH")
    pricer = PricingEngine()
    gen = SyntheticGenerator(disk_loader, pricer)
    
    # 1. Generate Path
    path_gen = PathGenerator(3.0, 0.0, 0.2, 5) # 5 days
    S_path = path_gen.generate_gbm(1)[0] # 1 path
    print(f"Simulated Path: {S_path}")
    
    # 2. Generate Options
    # Use real data date as template: '2020-01-02'
    base_date = '2020-01-02'
    start_sim = '2023-01-01' # Future date
    
    try:
        chain_list = gen.generate_chain_for_path(S_path, base_date, start_sim)
        print(f"Generated {len(chain_list)} days of synthetic options")
        if chain_list:
            print(chain_list[0][['symbol', 'strike', 'close', 'trade_date']].head())
            
            # Test Memory Loader
            data_map = {df.iloc[0]['trade_date'].strftime('%Y-%m-%d'): df for df in chain_list if not df.empty}
            mem_loader = MemoryDataLoader(data_map)
            print("Memory Loader Dates:", mem_loader.get_available_dates())
            
    except Exception as e:
        print(f"Test Failed: {e}")
