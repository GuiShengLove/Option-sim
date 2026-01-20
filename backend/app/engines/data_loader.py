"""
Data Loader for 50ETF Options
==============================
Lazy loading from partitioned Parquet files with date-range filtering.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Union
import warnings

import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')

class DataLoader:
    """
    Parquet data loader with lazy loading and date-range filtering.
    
    Optimized for the directory structure:
        data/510050_SH/{Year}/options_{Date}.parquet
    """
    
    def __init__(self, dataset_id: str = "510050_SH"):
        """
        Initialize the data loader with a specific dataset.
        
        Args:
            dataset_id: Folder name of the dataset (e.g. "510050_SH", "BTC_Option")
        """
        self.dataset_id = dataset_id
        self.data_dir = self._resolve_data_dir(dataset_id)
        
        if not self.data_dir.exists():
            print(f"Warning: Data directory not found at {self.data_dir}")
        
        # Cache available dates on init
        self._available_dates: Optional[List[str]] = None
        self._date_to_path: dict = {}
        self._build_date_index()
        
    def _resolve_data_dir(self, dataset_id: str) -> Path:
        """
        Resolve data directory looking in:
        1. data/{dataset_id} (Platform Data)
        2. user_data/{dataset_id} (User Data)
        """
        # Base paths relative to backend execution
        # Assuming we are in backend/app/engines, so root is ../../../
        # But safest is to use absolute or relative to CWD options
        
        # Try finding project root
        # If CWD is project_root (e:\Quant_code\Option-sim), then just use data/
        cwd = Path.cwd()
        
        # Candidate paths
        candidates = [
            cwd / "data" / dataset_id,
            cwd / "user_data" / dataset_id,
            # Fallback for when running from backend/ subdir
            cwd.parent / "data" / dataset_id, 
            cwd.parent / "user_data" / dataset_id,
             # Hardcoded absolute paths for robustness (based on user info)
            Path(r"e:\Quant_code\Option-sim\data") / dataset_id,
            Path(r"e:\Quant_code\Option-sim\user_data") / dataset_id
        ]
        
        for path in candidates:
            if path.exists():
                print(f"✅ Loaded Dataset: {dataset_id} from {path}")
                return path
                
        # Default fallback to create a valid path object even if not exists
        return cwd / "data" / dataset_id
    
    def _build_date_index(self) -> None:
        """Build an index of available dates and their file paths."""
        self._available_dates = []
        self._date_to_path = {}
        
        if not self.data_dir.exists():
            return

        # Scan all year directories
        for year_dir in sorted(self.data_dir.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                # Scan parquet files in year directory
                for parquet_file in sorted(year_dir.glob("options_*.parquet")):
                    # Extract date from filename: options_2020-01-02.parquet
                    date_str = parquet_file.stem.replace("options_", "")
                    self._available_dates.append(date_str)
                    self._date_to_path[date_str] = parquet_file
        
        # print(f"📊 DataLoader initialized: {len(self._available_dates)} trading days available")

    def get_available_dates(self) -> List[str]:
        """
        Get list of all available trading dates.
        """
        if self._available_dates is None:
             self._build_date_index()
        return self._available_dates.copy()
    
    def load_single_date(self, date: str) -> pd.DataFrame:
        """
        Load data for a single trading date.
        """
        if date not in self._date_to_path:
            raise ValueError(f"Date {date} not available.")
        
        return pd.read_parquet(self._date_to_path[date])
    
    def load_date_range(self, start: str, end: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load data for a date range (inclusive).
        """
        # Filter dates in range
        dates_in_range = [
            d for d in self._available_dates
            if start <= d <= end
        ]
        
        if not dates_in_range:
            raise ValueError(f"No data available for range {start} to {end}")
        
        # Load and concatenate
        dfs = []
        for date in dates_in_range:
            df = pd.read_parquet(
                self._date_to_path[date],
                columns=columns
            )
            dfs.append(df)
        
        result = pd.concat(dfs, ignore_index=True)
        return result
    
    def get_option_chain(self, trade_date: str) -> pd.DataFrame:
        """
        Get the full option chain for a specific trading date.
        """
        df = self.load_single_date(trade_date)
        return df.sort_values(["expiry_date", "type", "strike"]).reset_index(drop=True)
    
    def get_unique_expiries(self, trade_date: str) -> List[str]:
        """
        Get unique expiry dates available for a trading date.
        """
        df = self.load_single_date(trade_date)
        expiries = df["expiry_date"].dropna().unique()
        return sorted([pd.Timestamp(e).strftime("%Y-%m-%d") for e in expiries])
