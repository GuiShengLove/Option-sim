"""Test SSE margin calculation directly."""
from app.engines.risk import SSEMarginCalculator, RiskEngine, MarginAccount

# Test SSE margin calculator directly
print("=== SSE Margin Calculator Test ===")

underlying_price = 3.0
strike = 2.95
premium = 0.05
quantity_long = 2   # Buy
quantity_short = -2  # Sell
multiplier = 10000

margin_long = SSEMarginCalculator.calculate_position_margin(
    underlying_price, strike, 'P', premium, quantity_long, multiplier
)
print(f"Long Put margin (qty=+2): {margin_long:,.0f}")

margin_short = SSEMarginCalculator.calculate_position_margin(
    underlying_price, strike, 'P', premium, quantity_short, multiplier
)
print(f"Short Put margin (qty=-2): {margin_short:,.0f}")

# Test RiskEngine.calculate_portfolio_margin
print("\n=== RiskEngine Portfolio Margin Test ===")

account = MarginAccount(initial_capital=1000000, margin_scheme='SSE', asset_code='510050')
engine = RiskEngine(account)

positions = [
    {
        'type': 'P',
        'strike': 2.95,
        'quantity': -2,  # Short
        'current_price': 0.05
    }
]

margin = engine.calculate_portfolio_margin(positions, 3.0)
print(f"Portfolio margin for 2 short puts: {margin:,.0f}")
