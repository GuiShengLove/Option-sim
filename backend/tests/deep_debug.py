"""Deep debug margin - check all account values."""
import sys
for mod in list(sys.modules.keys()):
    if 'backtest' in mod or 'risk' in mod:
        del sys.modules[mod]

sys.path.insert(0, 'e:/Quant_code/Option-sim/backend')

from app.engines.backtest import BacktestEngine
from app.api.backtest_api import load_strategy_class
import pandas as pd

strategy_class, _ = load_strategy_class('641f1b06')
engine = BacktestEngine('510050_SH', 1000000, 0.03, 'SSE')

# Patch to capture all values
original_update = engine._update_portfolio_state
call_count = [0]

def debug_update(market_data, underlying_price):
    call_count[0] += 1
    original_update(market_data, underlying_price)
    
    positions = engine.context.positions
    if len(positions) > 0 or call_count[0] > 12:  # Only print when we have positions
        print(f"\n[Call #{call_count[0]}] _update_portfolio_state:")
        print(f"  Positions: {len(positions)}")
        for sym, pos in positions.items():
            print(f"    {sym}: qty={pos.quantity}")
        print(f"  Account Cash: {engine.account.cash:,.0f}")
        print(f"  Account Equity: {engine.account.equity:,.0f}")
        print(f"  Maintenance Margin: {engine.account.maintenance_margin:,.0f}")
        margin_util = engine.account.maintenance_margin / engine.account.equity if engine.account.equity > 0 else 0
        print(f"  Calculated margin_util: {margin_util:.6f}")

engine._update_portfolio_state = debug_update

# Run backtest - suppress strategy prints 
import io
from contextlib import redirect_stdout

class PartialRedirect:
    def write(self, s):
        if not any(x in s for x in ['📝', '├─', '└─', '🚀', '📅']):
            sys.__stdout__.write(s)
    def flush(self):
        sys.__stdout__.flush()

with redirect_stdout(PartialRedirect()):
    results_df, trade_log = engine.run(
        strategy_cls=strategy_class,
        strategy_config={},
        start_date='2020-01-02',
        end_date='2020-01-10'  # Just few days
    )

print("\n=== Final Results ===")
print(results_df[['equity', 'cash', 'margin_utilization', 'position_count']])
