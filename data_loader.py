"""
Data Loader for 50ETF Options
==============================
Lazy loading from partitioned Parquet files with date-range filtering.

Usage:
    loader = DataLoader("data/510050_SH")
    df = loader.load_date_range("2020-01-02", "2020-01-31")
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
    
    def __init__(self, data_dir: Union[str, Path] = "data/510050_SH"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the root data directory
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Cache available dates on init
        self._available_dates: Optional[List[str]] = None
        self._date_to_path: dict = {}
        self._build_date_index()
    
    def _build_date_index(self) -> None:
        """Build an index of available dates and their file paths."""
        self._available_dates = []
        self._date_to_path = {}
        
        # Scan all year directories
        for year_dir in sorted(self.data_dir.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                # Scan parquet files in year directory
                for parquet_file in sorted(year_dir.glob("options_*.parquet")):
                    # Extract date from filename: options_2020-01-02.parquet
                    date_str = parquet_file.stem.replace("options_", "")
                    self._available_dates.append(date_str)
                    self._date_to_path[date_str] = parquet_file
        
        print(f"📊 DataLoader initialized: {len(self._available_dates)} trading days available")
        if self._available_dates:
            print(f"   Date range: {self._available_dates[0]} to {self._available_dates[-1]}")
    
    def get_available_dates(self) -> List[str]:
        """
        Get list of all available trading dates.
        
        Returns:
            Sorted list of date strings (YYYY-MM-DD format)
        """
        return self._available_dates.copy()
    
    def load_single_date(self, date: str) -> pd.DataFrame:
        """
        Load data for a single trading date.
        
        Args:
            date: Date string in YYYY-MM-DD format
            
        Returns:
            DataFrame with option data for the specified date
            
        Raises:
            ValueError: If date is not available
        """
        if date not in self._date_to_path:
            raise ValueError(f"Date {date} not available. Use get_available_dates() to see options.")
        
        return pd.read_parquet(self._date_to_path[date])
    
    def load_date_range(
        self, 
        start: str, 
        end: str,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load data for a date range (inclusive).
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            columns: Optional list of columns to load (for memory optimization)
            
        Returns:
            Concatenated DataFrame with all data in the range
        """
        # Filter dates in range
        dates_in_range = [
            d for d in self._available_dates
            if start <= d <= end
        ]
        
        if not dates_in_range:
            raise ValueError(f"No data available for range {start} to {end}")
        
        print(f"📖 Loading {len(dates_in_range)} days: {dates_in_range[0]} to {dates_in_range[-1]}")
        
        # Load and concatenate
        dfs = []
        for date in dates_in_range:
            df = pd.read_parquet(
                self._date_to_path[date],
                columns=columns
            )
            dfs.append(df)
        
        result = pd.concat(dfs, ignore_index=True)
        
        # Memory report
        mem_mb = result.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"   ✓ Loaded {len(result):,} rows ({mem_mb:.2f} MB)")
        
        return result
    
    def load_by_expiry(
        self, 
        expiry_date: str,
        trade_date_start: Optional[str] = None,
        trade_date_end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load options with a specific expiry date.
        
        Args:
            expiry_date: Expiry date (YYYY-MM-DD)
            trade_date_start: Optional start of trade date range
            trade_date_end: Optional end of trade date range
            
        Returns:
            DataFrame filtered by expiry date
        """
        # Determine date range
        if trade_date_start is None:
            trade_date_start = self._available_dates[0]
        if trade_date_end is None:
            trade_date_end = expiry_date  # Can't trade after expiry
        
        # Load full range
        df = self.load_date_range(trade_date_start, min(trade_date_end, expiry_date))
        
        # Filter by expiry
        expiry_dt = pd.to_datetime(expiry_date)
        df_filtered = df[df["expiry_date"] == expiry_dt]
        
        print(f"   Filtered to expiry {expiry_date}: {len(df_filtered):,} rows")
        
        return df_filtered
    
    def get_option_chain(self, trade_date: str) -> pd.DataFrame:
        """
        Get the full option chain for a specific trading date.
        
        Convenience method that returns data sorted by expiry and strike.
        
        Args:
            trade_date: Trading date (YYYY-MM-DD)
            
        Returns:
            DataFrame sorted by expiry_date, type, strike
        """
        df = self.load_single_date(trade_date)
        return df.sort_values(["expiry_date", "type", "strike"]).reset_index(drop=True)
    
    def get_unique_expiries(self, trade_date: str) -> List[str]:
        """
        Get unique expiry dates available for a trading date.
        
        Args:
            trade_date: Trading date (YYYY-MM-DD)
            
        Returns:
            Sorted list of expiry date strings
        """
        df = self.load_single_date(trade_date)
        expiries = df["expiry_date"].dropna().unique()
        return sorted([pd.Timestamp(e).strftime("%Y-%m-%d") for e in expiries])


# ============================================================
# Convenience Functions
# ============================================================

def load_options(
    start: str, 
    end: str, 
    data_dir: str = "data/510050_SH"
) -> pd.DataFrame:
    """
    Convenience function to load options data.
    
    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        data_dir: Data directory path
        
    Returns:
        DataFrame with options data
    """
    loader = DataLoader(data_dir)
    return loader.load_date_range(start, end)


# ============================================================
# Test / Demo
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DataLoader Demo")
    print("=" * 60)
    
    # Initialize
    loader = DataLoader()
    
    # Show available dates
    dates = loader.get_available_dates()
    print(f"\nTotal trading days: {len(dates)}")
    print(f"First 5: {dates[:5]}")
    print(f"Last 5: {dates[-5:]}")
    
    # Load single date
    print("\n--- Single Date Load ---")
    df = loader.load_single_date(dates[0])
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    # Load date range
    print("\n--- Date Range Load (first week) ---")
    df_week = loader.load_date_range(dates[0], dates[4])
    print(f"Shape: {df_week.shape}")
    
    # Get option chain
    print("\n--- Option Chain ---")
    chain = loader.get_option_chain(dates[0])
    print(f"Expiries available: {loader.get_unique_expiries(dates[0])}")
    print(chain.head())
