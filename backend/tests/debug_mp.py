"""Debug margin_positions structure."""
import sys
for mod in list(sys.modules.keys()):
    if 'backtest' in mod or 'risk' in mod:
        del sys.modules[mod]

sys.path.insert(0, 'e:/Quant_code/Option-sim/backend')

from app.engines.backtest import BacktestEngine
from app.api.backtest_api import load_strategy_class

strategy_class, _ = load_strategy_class('641f1b06')
engine = BacktestEngine('510050_SH', 1000000, 0.03, 'SSE')

# Patch _update_portfolio_state to print margin_positions
original_update = engine._update_portfolio_state

def debug_update(market_data, underlying_price):
    """Debug wrapper."""
    price_map = dict(zip(market_data['symbol'], market_data['close']))
    
    margin_positions = []
    for symbol, pos in engine.context.positions.items():
        if symbol in price_map:
            pos.current_price = price_map[symbol]
            row = market_data[market_data['symbol'] == symbol]
            if not row.empty:
                mp = {
                    'type': row.iloc[0]['type'],
                    'strike': row.iloc[0]['strike'],
                    'quantity': pos.quantity,
                    'current_price': pos.current_price
                }
                margin_positions.append(mp)
    
    if margin_positions:
        print(f"\n=== margin_positions ===")
        for mp in margin_positions:
            print(f"  {mp}")
        
        # Call calculate_portfolio_margin directly
        margin = engine.risk.calculate_portfolio_margin(margin_positions, underlying_price)
        print(f"  -> calculate_portfolio_margin returned: {margin:,.0f}")
    
    # Call original
    original_update(market_data, underlying_price)

engine._update_portfolio_state = debug_update

# Run backtest - suppress most output
import io
from contextlib import redirect_stdout

results_df, trade_log = engine.run(
    strategy_cls=strategy_class,
    strategy_config={},
    start_date='2020-01-02',
    end_date='2020-01-10'
)
