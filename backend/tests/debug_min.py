"""Debug margin - minimal output."""
import sys
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

print("=== Key Debug Info ===")
print(f"Trades: {len(trade_log)}")

# Check positions after run
print(f"Final positions count: {len(engine.context.positions)}")
for sym, pos in engine.context.positions.items():
    print(f"  {sym}: qty={pos.quantity}, price={pos.current_price:.4f}")

print(f"\nFinal account state:")
print(f"  Cash: {engine.account.cash:,.0f}")
print(f"  Equity: {engine.account.equity:,.0f}")
print(f"  Maintenance Margin: {engine.account.maintenance_margin:,.0f}")

print(f"\nResults margin_utilization column:")
print(results_df['margin_utilization'].values)
print(f"\nResults position_count column:")
print(results_df['position_count'].values)
