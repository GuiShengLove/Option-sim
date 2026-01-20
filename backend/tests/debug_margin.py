"""Debug trade log and margin."""
from app.engines.backtest import BacktestEngine
from app.api.backtest_api import load_strategy_class

strategy_class, _ = load_strategy_class('641f1b06')
engine = BacktestEngine('510050_SH', 1000000, 0.03, 'SSE')

results_df, trade_log = engine.run(
    strategy_cls=strategy_class,
    strategy_config={},
    start_date='2020-01-02',
    end_date='2020-01-15'
)

print('Trade log:')
for t in trade_log:
    print(f"  {t.get('date')}: {t.get('action')} {t.get('quantity')} x {t.get('symbol')}")

print()
print('Account details after run:')
print('  Cash:', engine.account.cash)
print('  Equity:', engine.account.equity)
print('  Maintenance Margin:', engine.account.maintenance_margin)
print()
print('Margin utilization in results:')
print(results_df[['margin_utilization', 'position_count']].tail(10))
