"""Debug positions in backtest."""
from app.engines.backtest import BacktestEngine
from app.api.backtest_api import load_strategy_class

strategy_class, _ = load_strategy_class('641f1b06')
engine = BacktestEngine('510050_SH', 1000000, 0.03, 'SSE')

# Patch to debug positions
original_update = engine._update_portfolio_state

def debug_update(market_data, underlying_price):
    """Wrapper to debug."""
    print(f"\n  Debug _update_portfolio_state:")
    print(f"    Positions count: {len(engine.context.positions)}")
    for symbol, pos in engine.context.positions.items():
        print(f"    - {symbol}: quantity={pos.quantity}, price={pos.current_price:.4f}")
    
    original_update(market_data, underlying_price)
    
    print(f"    Maintenance margin: {engine.account.maintenance_margin:,.0f}")
    print(f"    Equity: {engine.account.equity:,.0f}")
    margin_util = engine.account.maintenance_margin / engine.account.equity if engine.account.equity > 0 else 0
    print(f"    Margin utilization: {margin_util:.4f}")

engine._update_portfolio_state = debug_update

# Run backtest
results_df, trade_log = engine.run(
    strategy_cls=strategy_class,
    strategy_config={},
    start_date='2020-01-02',
    end_date='2020-01-10'  # Just a few days
)
