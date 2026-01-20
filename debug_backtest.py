import sys
import os
import pandas as pd
from datetime import datetime

# Add 'backend' directory to sys.path so we can import 'app'
backend_path = os.path.join(os.getcwd(), 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

try:
    # Try importing as if we are in backend environment
    from app.engines.backtest import BacktestEngine
    # Strategies are in root/strategies
    sys.path.append(os.getcwd()) 
    from strategies.collar_strategy import CollarStrategy, StrategyConfig
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback debug print
    import traceback
    traceback.print_exc()
    sys.exit(1)

def run_debug():
    print("Initializing Backtest Engine...")
    try:
        engine = BacktestEngine(
            data_dir="e:/Quant_code/Option-sim/backend/data/510050_SH", # Adjust path if needed
            initial_capital=1000000
        )
        
        print("Initializing Strategy...")
        # config = StrategyConfig()
        
        print("Running Backtest...")
        df = engine.run(
            strategy_cls=CollarStrategy,
            strategy_config={},
            start_date="2020-01-02",
            end_date="2020-02-01"
        )
        
        print("\nBacktest Result Summary:")
        print(f"Rows: {len(df)}")
        if not df.empty:
            print(f"Final Equity: {df['equity'].iloc[-1]}")
            print(f"Total Return: {(df['equity'].iloc[-1] - 1000000)/10000:.2f}%")
            print("\nFirst 5 rows:")
            print(df.head())
        else:
            print("Result DataFrame is empty!")
            
    except Exception as e:
        print("\nCRITICAL ERROR DURING BACKTEST:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()
