"""Debug why SSE calculator returns NaN."""
import sys
sys.path.insert(0, 'e:/Quant_code/Option-sim/backend')

from app.engines.risk import SSEMarginCalculator, RiskEngine, MarginAccount

# Test with realistic values from the backtest
underlying_price = 3.0
strike = 2.95
premium = 0.0382  # Example close price from data
quantity = -2  # Short
multiplier = 10000

print("=== Testing SSE Margin Calculator ===")
print(f"Inputs: S={underlying_price}, K={strike}, premium={premium}, qty={quantity}")

margin = SSEMarginCalculator.calculate_position_margin(
    underlying_price, strike, 'P', premium, quantity, multiplier
)
print(f"Result: {margin}")
print(f"Is NaN? {margin != margin}")  # NaN != NaN is True

# Check intermediate calculations
print("\n=== Intermediate Calculations ===")
otm_amount = max(0, underlying_price - strike)
print(f"OTM amount: {otm_amount}")

margin_method1 = premium + underlying_price * 0.12 - otm_amount
margin_method2 = premium + strike * 0.07
print(f"Method 1: {premium} + {underlying_price} * 0.12 - {otm_amount} = {margin_method1}")
print(f"Method 2: {premium} + {strike} * 0.07 = {margin_method2}")

margin_per_unit = max(margin_method1, margin_method2)
print(f"Margin per unit: {margin_per_unit}")
print(f"Total margin: {margin_per_unit} * {multiplier} * {abs(quantity)} = {margin_per_unit * multiplier * abs(quantity)}")

# Now test with possible NaN inputs
print("\n=== Testing with potential NaN inputs ===")
test_cases = [
    (3.0, 2.95, 0.0382, -2),
    (3.0, 2.95, float('nan'), -2),
    (float('nan'), 2.95, 0.0382, -2),
]

for (s, k, p, q) in test_cases:
    result = SSEMarginCalculator.calculate_position_margin(s, k, 'P', p, q, multiplier)
    print(f"  S={s}, K={k}, P={p}, Q={q} => {result}")
