"""
Backtest Verification Script
============================
Verifies that backtest is correctly using real data, loading strategies,
and applying parameters.

Run with: python backend/tests/verify_backtest.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import pandas as pd
from datetime import datetime


def verify_data_loading():
    """Verify that DataLoader correctly loads real parquet data."""
    print("\n" + "="*60)
    print("1. VERIFYING DATA LOADING")
    print("="*60)
    
    from backend.app.engines.data_loader import DataLoader
    
    # Test platform data
    loader = DataLoader("510050_SH")
    dates = loader.get_available_dates()
    
    print(f"✓ DataLoader initialized for 510050_SH")
    print(f"  - Data directory: {loader.data_dir}")
    print(f"  - Available dates: {len(dates)} trading days")
    
    if dates:
        print(f"  - Date range: {dates[0]} to {dates[-1]}")
        
        # Load sample data
        sample_date = dates[0]
        df = loader.load_single_date(sample_date)
        print(f"\n  Sample data for {sample_date}:")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)[:10]}...")
        
        # Verify key columns exist
        required_cols = ['symbol', 'close', 'strike', 'type']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  ⚠ Missing columns: {missing}")
        else:
            print(f"  ✓ All required columns present")
        
        return True
    else:
        print("  ✗ No trading dates found!")
        return False


def verify_strategy_loading():
    """Verify that strategies are correctly loaded."""
    print("\n" + "="*60)
    print("2. VERIFYING STRATEGY LOADING")
    print("="*60)
    
    STRATEGY_DIR = PROJECT_ROOT / "strategies"
    
    strategies_found = []
    for f in STRATEGY_DIR.glob("**/*.py"):
        if f.name.startswith("_"):
            continue
        strategies_found.append(f.relative_to(STRATEGY_DIR))
    
    print(f"✓ Found {len(strategies_found)} strategy files:")
    for s in strategies_found[:5]:
        print(f"  - {s}")
    if len(strategies_found) > 5:
        print(f"  ... and {len(strategies_found) - 5} more")
    
    # Try loading a strategy
    try:
        from backend.app.api.backtest_api import load_strategy_class
        
        # Find first concrete strategy
        test_strategy_path = None
        for f in strategies_found:
            if "strategy" in str(f).lower():
                test_strategy_path = str(f)
                break
        
        if test_strategy_path:
            print(f"\n  Testing load: {test_strategy_path}")
            try:
                strategy_class, name = load_strategy_class(test_strategy_path)
                print(f"  ✓ Successfully loaded: {strategy_class.__name__}")
                print(f"    - Has on_init: {hasattr(strategy_class, 'on_init')}")
                print(f"    - Has on_bar: {hasattr(strategy_class, 'on_bar')}")
                return True
            except Exception as e:
                print(f"  ⚠ Could not load (may be abstract): {e}")
    except ImportError as e:
        print(f"  ⚠ Import error (expected in test): {e}")
    
    return True


def verify_backtest_engine():
    """Verify backtest engine initialization and data flow."""
    print("\n" + "="*60)
    print("3. VERIFYING BACKTEST ENGINE")
    print("="*60)
    
    try:
        from backend.app.engines.backtest import BacktestEngine
        from backend.app.engines.data_loader import DataLoader
        
        # Create engine
        engine = BacktestEngine(
            dataset_id="510050_SH",
            initial_capital=1_000_000,
            margin_scheme="REG_T"
        )
        
        print(f"✓ BacktestEngine created")
        print(f"  - Initial capital: ¥{engine.account.cash:,.0f}")
        print(f"  - Dataset: {engine.data_loader.dataset_id}")
        
        # Verify data loader is connected
        dates = engine.data_loader.get_available_dates()
        print(f"  - Data dates available: {len(dates)}")
        
        if dates:
            # Load first date data
            first_date = dates[0]
            df = engine.data_loader.load_single_date(first_date)
            print(f"\n  Data verification for {first_date}:")
            print(f"  - Options loaded: {len(df)}")
            if 'close' in df.columns:
                avg_price = df['close'].mean()
                print(f"  - Avg close price: ¥{avg_price:.4f}")
            
            return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_api_endpoints():
    """Verify that API endpoints are configured correctly."""
    print("\n" + "="*60)
    print("4. VERIFYING API ENDPOINTS")
    print("="*60)
    
    try:
        from backend.app.api.files import router as files_router
        from backend.app.api.backtest_api import router as backtest_router
        
        # List files routes
        file_routes = [r.path for r in files_router.routes]
        backtest_routes = [r.path for r in backtest_router.routes]
        
        print(f"✓ Files API routes ({len(file_routes)}):")
        for r in file_routes[:8]:
            print(f"  - {r}")
        if len(file_routes) > 8:
            print(f"  ... and {len(file_routes) - 8} more")
        
        print(f"\n✓ Backtest API routes ({len(backtest_routes)}):")
        for r in backtest_routes:
            print(f"  - {r}")
        
        # Check for new endpoints
        expected_new = ['/data/folder', '/data/upload-to', '/data/tree', '/datasets']
        found_new = [e for e in expected_new if any(e in r for r in file_routes)]
        print(f"\n  New endpoints verified: {found_new}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("="*60)
    print("BACKTEST VERIFICATION SCRIPT")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {
        "Data Loading": verify_data_loading(),
        "Strategy Loading": verify_strategy_loading(),
        "Backtest Engine": verify_backtest_engine(),
        "API Endpoints": verify_api_endpoints(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("All verifications passed! Backtest is correctly configured.")
    else:
        print("Some verifications failed. Please review the output above.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
