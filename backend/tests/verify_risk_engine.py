
import sys
import os
import pandas as pd
import numpy as np

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engines.risk import RiskEngine, MarginAccount

def test_portfolio_margin():
    print("Testing RiskEngine.calculate_portfolio_margin...")
    # Initialize with a dummy account
    account = MarginAccount(initial_capital=1000000.0)
    engine = RiskEngine(account)
    
    # Mock data
    spot_price = 3.0
    
    # Case 1: Naked Call (High Risk)
    positions_naked = [
        {'type': 'C', 'strike': 3.0, 'quantity': 1, 'current_price': 0.1, 'delta': 0.5, 'gamma': 0.1, 'vega': 0.05, 'theta': -0.01}
    ]
    margin_naked = engine.calculate_portfolio_margin(positions_naked, spot_price)
    print(f"Naked Call Margin: {margin_naked:.2f}")
    
    # Case 2: Bull Call Spread (Lower Risk)
    # Long 3.0 Call, Short 3.1 Call
    positions_spread = [
        {'type': 'C', 'strike': 3.0, 'quantity': 1, 'current_price': 0.1, 'delta': 0.5, 'gamma': 0.1},
        {'type': 'C', 'strike': 3.1, 'quantity': -1, 'current_price': 0.05, 'delta': 0.4, 'gamma': 0.08}
    ]
    margin_spread = engine.calculate_portfolio_margin(positions_spread, spot_price)
    print(f"Bull Spread Margin: {margin_spread:.2f}")
    
    # Verification
    if margin_spread < margin_naked:
        print("[PASS] Spread margin is lower than naked margin.")
    else:
        print("[FAIL] Spread margin should be lower!")

    # Case 3: Naked Put
    positions_put = [
        {'type': 'P', 'strike': 3.0, 'quantity': 1, 'current_price': 0.1, 'delta': -0.5, 'gamma': 0.1}
    ]
    margin_put = engine.calculate_portfolio_margin(positions_put, spot_price)
    print(f"Naked Put Margin: {margin_put:.2f}")

if __name__ == "__main__":
    try:
        test_portfolio_margin()
        print("\nTest Complete.")
    except Exception as e:
        print(f"\nTest Failed with error: {e}")
