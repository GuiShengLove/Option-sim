"""Verify margin fix - force reload module."""
import sys
import importlib

# Remove cached modules
for mod in list(sys.modules.keys()):
    if 'backtest' in mod or 'risk' in mod:
        del sys.modules[mod]

sys.path.insert(0, 'e:/Quant_code/Option-sim/backend')

from app.engines.backtest import BacktestEngine
from app.api.backtest_api import load_strategy_class

strategy_class, _ = load_strategy_class('641f1b06')
engine = BacktestEngine('510050_SH', 1000000, 0.03, 'SSE')

# Suppress strategy prints
import io
from contextlib import redirect_stdout

with redirect_stdout(io.StringIO()):
    results_df, trade_log = engine.run(
        strategy_cls=strategy_class,
        strategy_config={},
        start_date='2020-01-02',
        end_date='2020-01-15'
    )

print("=== Margin Verification ===")
print(f"Trades: {len(trade_log)}")
print(f"\nmargin_utilization values:")
for idx, val in results_df['margin_utilization'].items():
    pos_cnt = results_df.loc[idx, 'position_count']
    print(f"  {idx.strftime('%Y-%m-%d')}: margin_util={val:.4f}, positions={pos_cnt}")
