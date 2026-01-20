"""
ETL Pipeline for 50ETF Options Data
====================================
Phase 1: Data Infrastructure

Transforms raw .pkl and .feather files into standardized Parquet format
with Year/Date partitioning for efficient lazy loading.

Target Schema:
- trade_date, expiry_date, strike, type (C/P), symbol
- open, high, low, close, volume, underlying_close

Author: AI Quant Platform
"""

import os
import sys
from pathlib import Path
from typing import Tuple
import warnings

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')


# ============================================================
# Configuration
# ============================================================

class Config:
    """ETL Pipeline Configuration"""
    
    # Source paths (relative to data/50ETF/)
    SOURCE_DIR = Path("data/50ETF")
    INSTRUMENTS_FILE = "Filtered_OptionInstruments_510050.pkl"
    PRICE_FILE = "Filtered_OptionPrice_2020_2022.feather"
    
    # Output path
    OUTPUT_DIR = Path("data/510050_SH")
    
    # Parquet settings
    COMPRESSION = "snappy"
    
    # Column mappings: source -> target
    INSTRUMENT_COLS = {
        "maturity_date": "expiry_date",
        "strike_price": "strike",
        "option_type": "type",
        "symbol": "symbol",
        "order_book_id": "order_book_id",  # join key
    }
    
    PRICE_COLS = {
        "date": "trade_date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "order_book_id": "order_book_id",  # join key
    }
    
    # Target schema with dtypes
    TARGET_SCHEMA = {
        "trade_date": "datetime64[ns]",
        "expiry_date": "datetime64[ns]",
        "strike": "float32",
        "type": "category",          # C or P
        "symbol": "string",
        "open": "float32",
        "high": "float32",
        "low": "float32",
        "close": "float32",
        "volume": "float32",
        "underlying_close": "float32",  # placeholder, NaN for now
        "order_book_id": "string",       # keep for reference
    }


# ============================================================
# ETL Functions
# ============================================================

def load_source_data(base_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load source data files.
    
    Returns:
        Tuple of (instruments_df, price_df)
    """
    print("\n" + "=" * 60)
    print("STEP 1: Loading Source Data")
    print("=" * 60)
    
    source_dir = base_path / Config.SOURCE_DIR
    
    # Load instruments
    inst_path = source_dir / Config.INSTRUMENTS_FILE
    print(f"\n📂 Loading instruments: {inst_path}")
    instruments = pd.read_pickle(inst_path)
    print(f"   ✓ Shape: {instruments.shape}")
    
    # Load price data
    price_path = source_dir / Config.PRICE_FILE
    print(f"\n📂 Loading price data: {price_path}")
    price = pd.read_feather(price_path)
    print(f"   ✓ Shape: {price.shape}")
    
    return instruments, price


def transform_data(instruments: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """
    Transform and merge source data into target schema.
    
    Returns:
        Merged DataFrame with standardized schema
    """
    print("\n" + "=" * 60)
    print("STEP 2: Transforming Data")
    print("=" * 60)
    
    # Select and rename instrument columns
    inst_cols = list(Config.INSTRUMENT_COLS.keys())
    inst_subset = instruments[inst_cols].copy()
    inst_subset = inst_subset.rename(columns=Config.INSTRUMENT_COLS)
    print(f"\n📋 Instrument columns selected: {len(inst_cols)}")
    
    # Select and rename price columns
    price_cols = list(Config.PRICE_COLS.keys())
    price_subset = price[price_cols].copy()
    price_subset = price_subset.rename(columns=Config.PRICE_COLS)
    print(f"📋 Price columns selected: {len(price_cols)}")
    
    # Standardize join key type
    inst_subset["order_book_id"] = inst_subset["order_book_id"].astype(str)
    price_subset["order_book_id"] = price_subset["order_book_id"].astype(str)
    
    # Merge: LEFT JOIN on price (keep all price records)
    print("\n🔗 Merging datasets on order_book_id...")
    merged = price_subset.merge(
        inst_subset,
        on="order_book_id",
        how="left"
    )
    print(f"   ✓ Merged shape: {merged.shape}")
    
    # Check for unmatched records
    unmatched = merged["expiry_date"].isna().sum()
    if unmatched > 0:
        print(f"   ⚠ Warning: {unmatched} price records have no matching instrument")
    
    # Add placeholder for underlying_close
    merged["underlying_close"] = np.nan
    print("   ✓ Added underlying_close placeholder (NaN)")
    
    # Apply data types
    print("\n🔧 Applying target schema types...")
    for col, dtype in Config.TARGET_SCHEMA.items():
        if col in merged.columns:
            try:
                if dtype == "datetime64[ns]":
                    merged[col] = pd.to_datetime(merged[col])
                elif dtype == "category":
                    merged[col] = merged[col].astype("category")
                elif dtype == "string":
                    merged[col] = merged[col].astype("string")
                elif dtype == "float32":
                    merged[col] = merged[col].astype("float32")
            except Exception as e:
                print(f"   ⚠ Column {col}: {e}")
    
    # Reorder columns to match target schema
    target_cols = list(Config.TARGET_SCHEMA.keys())
    merged = merged[target_cols]
    
    print(f"   ✓ Final columns: {list(merged.columns)}")
    
    return merged


def partition_and_save(df: pd.DataFrame, base_path: Path) -> dict:
    """
    Partition data by Year/Date and save as Parquet files.
    
    Returns:
        Statistics dict with file counts and sizes
    """
    print("\n" + "=" * 60)
    print("STEP 3: Partitioning and Saving")
    print("=" * 60)
    
    output_dir = base_path / Config.OUTPUT_DIR
    
    # Extract year for partitioning
    df["_year"] = df["trade_date"].dt.year
    df["_date_str"] = df["trade_date"].dt.strftime("%Y-%m-%d")
    
    stats = {
        "total_rows": len(df),
        "years": {},
        "files_created": 0,
        "total_bytes": 0
    }
    
    # Group by year
    for year, year_group in df.groupby("_year"):
        year_dir = output_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        
        year_stats = {"dates": 0, "rows": 0}
        
        # Group by date within year
        for date_str, date_group in year_group.groupby("_date_str"):
            # Remove helper columns before saving
            save_df = date_group.drop(columns=["_year", "_date_str"])
            
            # Convert to PyArrow table for better type control
            table = pa.Table.from_pandas(save_df, preserve_index=False)
            
            # Save as Parquet
            file_path = year_dir / f"options_{date_str}.parquet"
            pq.write_table(
                table, 
                file_path,
                compression=Config.COMPRESSION
            )
            
            # Update stats
            file_size = file_path.stat().st_size
            year_stats["dates"] += 1
            year_stats["rows"] += len(date_group)
            stats["files_created"] += 1
            stats["total_bytes"] += file_size
        
        stats["years"][year] = year_stats
        print(f"\n📁 {year}/")
        print(f"   ├── Files: {year_stats['dates']}")
        print(f"   └── Rows: {year_stats['rows']:,}")
    
    return stats


def verify_output(base_path: Path) -> bool:
    """
    Verify the output Parquet files are valid.
    
    Returns:
        True if verification passed
    """
    print("\n" + "=" * 60)
    print("STEP 4: Verification")
    print("=" * 60)
    
    output_dir = base_path / Config.OUTPUT_DIR
    
    # Find all parquet files
    parquet_files = list(output_dir.rglob("*.parquet"))
    print(f"\n🔍 Found {len(parquet_files)} Parquet files")
    
    # Random sample verification
    import random
    sample_files = random.sample(parquet_files, min(3, len(parquet_files)))
    
    print("\n📊 Sample file verification:")
    all_valid = True
    
    for file_path in sample_files:
        try:
            df = pd.read_parquet(file_path)
            target_cols = list(Config.TARGET_SCHEMA.keys())
            missing = set(target_cols) - set(df.columns)
            
            if missing:
                print(f"   ✗ {file_path.name}: Missing columns {missing}")
                all_valid = False
            else:
                print(f"   ✓ {file_path.name}: {len(df)} rows, schema OK")
        except Exception as e:
            print(f"   ✗ {file_path.name}: Read error - {e}")
            all_valid = False
    
    return all_valid


def print_summary(stats: dict, verified: bool):
    """Print final ETL summary."""
    print("\n" + "=" * 60)
    print("ETL PIPELINE SUMMARY")
    print("=" * 60)
    
    print(f"""
📈 Data Statistics:
   • Total rows processed: {stats['total_rows']:,}
   • Files created: {stats['files_created']}
   • Total size: {stats['total_bytes'] / 1024 / 1024:.2f} MB
   • Compression: {Config.COMPRESSION}

📅 Year Breakdown:""")
    
    for year, year_stats in sorted(stats["years"].items()):
        print(f"   • {year}: {year_stats['dates']} days, {year_stats['rows']:,} rows")
    
    if verified:
        print("\n✅ Verification: PASSED")
    else:
        print("\n❌ Verification: FAILED - Please check output files")
    
    print("\n" + "=" * 60)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Run the ETL pipeline."""
    print("""
╔══════════════════════════════════════════════════════════╗
║     50ETF Option Data ETL Pipeline                       ║
║     Phase 1: Data Infrastructure                         ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Determine base path (script should be run from project root)
    base_path = Path(__file__).parent
    print(f"📍 Base path: {base_path.absolute()}")
    
    try:
        # Step 1: Load
        instruments, price = load_source_data(base_path)
        
        # Step 2: Transform
        merged_df = transform_data(instruments, price)
        
        # Memory usage report
        mem_usage = merged_df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"\n💾 In-memory DataFrame size: {mem_usage:.2f} MB")
        
        # Step 3: Partition & Save
        stats = partition_and_save(merged_df, base_path)
        
        # Step 4: Verify
        verified = verify_output(base_path)
        
        # Summary
        print_summary(stats, verified)
        
        return 0 if verified else 1
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Source file not found - {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
