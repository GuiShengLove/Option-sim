"""
Backtest Engine Pro
===================
Event-driven Backtesting Kernel for Option Strategies.
Integrates Risk Engine (Margin) and Pricing Engine.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Type
import warnings

from .data_loader import DataLoader
from .pricing import PricingEngine
from .risk import RiskEngine, MarginAccount
from .strategy import BaseStrategy, BacktestContext, Position, Order

warnings.filterwarnings('ignore')

class BacktestEngine:
    """
    Core Backtest Engine with Margin Support.
    """
    
    def __init__(
        self, 
        dataset_id: str = "510050_SH",
        initial_capital: float = 1_000_000,
        risk_free_rate: float = 0.03,
        margin_scheme: str = 'SSE',  # 'FIXED', 'SSE', 'SPAN', 'PM'
        margin_ratio: float = None,  # 自定义保证金率，覆盖默认
        maintenance_margin: float = None,  # 自定义维持保证金率
        leverage: float = 1.0  # 杠杆倍数
    ):
        self.loader = DataLoader(dataset_id)
        self.pricer = PricingEngine(risk_free_rate)
        
        # Extract asset code from dataset_id for multiplier lookup
        asset_code = dataset_id.split('_')[0] if '_' in dataset_id else dataset_id
        
        # Initialize Margin Account with asset-specific multiplier and custom margins
        self.account = MarginAccount(
            initial_capital=initial_capital, 
            margin_scheme=margin_scheme,
            asset_code=asset_code,
            custom_margin_ratio=margin_ratio,
            custom_maintenance_margin=maintenance_margin,
            leverage=leverage
        )
        self.risk = RiskEngine(self.account)
        
        self.rf = risk_free_rate
        
        # State
        self.context = None
        self.trade_log = []  # Trade history


        
    def run(
        self, 
        strategy_cls: Type[BaseStrategy], 
        strategy_config: Dict,
        start_date: str, 
        end_date: str
    ) -> (pd.DataFrame, List[Dict]):
        """
        Run backtest.
        """
        print(f"🚀 Starting Backtest: {start_date} to {end_date}")
        
        # 1. Initialize Context & Strategy
        self.context = BacktestContext(
            current_date=pd.to_datetime(start_date),
            cash=self.account.cash,
            portfolio_value=self.account.equity
        )
        
        strategy = strategy_cls(**strategy_config)
        # Inject Risk and Account into Strategy
        strategy.risk = self.risk
        strategy.account = self.account
        
        strategy.on_init(self.context)
        
        # 2. Get Trading Days
        dates = [d for d in self.loader.get_available_dates() if start_date <= d <= end_date]
        if not dates:
            raise ValueError("No trading dates found in range.")
            
        print(f"   📅 Processing {len(dates)} trading days...")
        
        # 3. Main Loop
        results = []
        
        for date_str in dates:
            current_dt = pd.to_datetime(date_str)
            self.context.current_date = current_dt
            
            # --- A. Load Data ---
            daily_options = self.loader.load_single_date(date_str)
            daily_options['trade_date'] = current_dt
            
            # Underlying Price S Inference
            # Try to get from data columns first, else infer from ATM Parity
            S = None
            if 'underlying_close' in daily_options.columns:
                val = daily_options['underlying_close'].iloc[0]
                if pd.notna(val) and val > 0:
                    S = float(val)
            
            if S is None and 'etf_close' in daily_options.columns:
                val = daily_options['etf_close'].iloc[0]
                if pd.notna(val) and val > 0:
                    S = float(val)
            
            if S is None:
                # Fallback: Try to infer from strike prices (median of available strikes)
                if 'strike' in daily_options.columns:
                    median_strike = daily_options['strike'].median()
                    if pd.notna(median_strike) and median_strike > 0:
                        S = float(median_strike)
                
                # Ultimate fallback
                if S is None:
                    S = 3.0  # Reasonable default for 50ETF
            
            # Calculate Greeks/Prices
            enriched_options = self.pricer.calculate_all(daily_options, S, 0.20)
            
            # --- B. Update Portfolio Value & Margin Logic ---
            # Mark-to-Market
            self._update_portfolio_state(enriched_options, S)
            
            # Check Margin Call / Liquidation (Portfolio Scan)
            if self.risk.check_liquidation():
                print(f"🔥 MARGIN CALL at {date_str}! Equity: {self.account.equity:.2f} < Maint: {self.account.maintenance_margin:.2f}")
                self._liquidate_positions(enriched_options)
            
            # --- C. Strategy Signal ---
            data_feed = {
                'options': enriched_options,
                'underlying_price': S
            }
            strategy.on_bar(self.context, data_feed)
            
            # --- D. Order Execution ---
            self._process_orders(enriched_options, S)
            
            # --- D2. Re-calculate Margin After Order Execution ---
            # This is crucial: orders executed above may have created new positions
            # We need to recalculate margin AFTER orders are filled
            self._update_portfolio_state(enriched_options, S)
            
            # --- E. Record Result ---
            # Aggregate Greeks
            agg_greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
            for pos in self.context.positions.values():
                 # Need to lookup Greeks for position. 
                 # In _update_portfolio_state we iterate but don't save per pos currently unless we store in Position.
                 # Let's assume we can approximate or we should strictly store it.
                 # For MVP, we'll zero it or need to fetch from enriched_options again.
                 row = enriched_options[enriched_options['symbol'] == pos.symbol]
                 if not row.empty:
                     qty = pos.quantity * 10000 # Contract Multiplier? 
                     # Wait, delta is usually per share or per contract? 
                     # BSM output delta is 0-1 (per share).
                     # So Position Delta = delta * quantity * multiplier
                     multiplier = 10000
                     agg_greeks['delta'] += row.iloc[0]['delta'] * pos.quantity * multiplier
                     agg_greeks['gamma'] += row.iloc[0]['gamma'] * pos.quantity * multiplier
                     agg_greeks['vega'] += row.iloc[0]['vega'] * pos.quantity * multiplier
                     agg_greeks['theta'] += row.iloc[0]['theta'] * pos.quantity * multiplier

            results.append({
                'date': current_dt,
                'equity': self.account.equity,
                'cash': self.account.cash,
                'margin_utilization': self.account.maintenance_margin / self.account.equity if self.account.equity > 0 else 0,
                'position_count': len(self.context.positions),
                'total_delta': agg_greeks['delta'],
                'total_gamma': agg_greeks['gamma'],
                'total_vega': agg_greeks['vega'],
                'total_theta': agg_greeks['theta'],
                'underlying_price': S
            })
            
        return pd.DataFrame(results).set_index('date'), self.trade_log

    def _update_portfolio_state(self, market_data: pd.DataFrame, underlying_price: float):
        """Update Equity and Margin Requirements based on new market prices."""
        price_map = dict(zip(market_data['symbol'], market_data['close']))
        # In a real engine, we also need Strike/Type for Margin Calc.
        # We can build a map: symbol -> record
        # For Optimization, we iterate over positions.
        
        total_market_value = 0.0
        total_maint_margin = 0.0
        
        # Create a lookup for full option record
        # market_records = market_data.set_index('symbol').to_dict('index')
        # This is slow every step. Better: Only query held positions.
        
        # Prepare positions list for Portfolio Margin
        margin_positions = []
        
        for symbol, pos in self.context.positions.items():
            if symbol in price_map:
                pos.current_price = price_map[symbol]
                
                # Fetch contract details for Risk Engine
                row = market_data[market_data['symbol'] == symbol]
                if not row.empty:
                    margin_positions.append({
                        'type': row.iloc[0]['type'],
                        'strike': row.iloc[0]['strike'],
                        'quantity': pos.quantity,
                        'current_price': pos.current_price
                    })
            
            # Update Market Value
            pos_mv = pos.quantity * pos.current_price * 10000
            total_market_value += pos_mv

        # Calculate Portfolio Margin (Scenario Analysis)
        total_maint_margin = self.risk.calculate_portfolio_margin(margin_positions, underlying_price)

        self.account.equity = self.account.cash + total_market_value 
        self.account.maintenance_margin = total_maint_margin
        self.account.update_balances()
        
        # Sync context
        self.context.portfolio_value = self.account.equity
        self.context.cash = self.account.cash

    def _process_orders(self, market_data: pd.DataFrame, underlying_price: float):
        """Match and Execute orders with Pre-trade Margin Check."""
        price_map = dict(zip(market_data['symbol'], market_data['close']))
        
        for order in self.context.orders:
            if order.status != "PENDING": continue
            
            if order.symbol in price_map:
                price = price_map[order.symbol]
                
                # 1. Pre-trade Margin Check
                # Need token detail for margin calc
                row = market_data[market_data['symbol'] == order.symbol]
                if row.empty: continue
                
                strike = row.iloc[0]['strike']
                otype = row.iloc[0]['type']
                
                # Provisional calculation
                # We need to knowing if it increases risk.
                # Simple check: Calculate Margin Impact of this NEW order
                impact = self.risk.calculate_margin_impact(
                    underlying_price, strike, otype, price, order.quantity, is_short=(order.quantity < 0)
                )
                
                # Check Liquidity
                # Long: need Cash > Cost
                # Short: need Excess Liq > Margin Impact
                
                cost = order.quantity * price * 10000
                
                if order.quantity > 0: # BUY
                    if self.account.cash < cost:
                        # print(f"❌ REJECT BUY {order.symbol}: Insufficient Cash")
                        order.status = "REJECTED"
                        continue
                else: # SELL (Short)
                    if self.account.excess_liquidity < impact:
                         print(f"❌ REJECT SELL {order.symbol}: Insufficient Margin (Req: {impact:.2f}, Avail: {self.account.excess_liquidity:.2f})")
                         order.status = "REJECTED"
                         continue

                # Execute
                self._execute_trade(order, price, self.context.current_date)
                
    def _execute_trade(self, order: Order, price: float, date: datetime):
        cost = order.quantity * price * 10000
        # Commission (Simple fixed)
        fee = max(5.0, abs(order.quantity) * 2.0) # 2 RMB per contract
        
        self.account.cash -= (cost + fee)
        
        # Calculate Realized PnL for closing trades
        realized_pnl = 0.0
        symbol = order.symbol
        
        if symbol in self.context.positions:
            pos = self.context.positions[symbol]
            # Check if this is a closing trade (opposite direction)
            if (pos.quantity > 0 and order.quantity < 0) or (pos.quantity < 0 and order.quantity > 0):
                # Closing trade - calculate realized PnL
                close_qty = min(abs(order.quantity), abs(pos.quantity))
                if pos.quantity > 0:
                    # Long position being closed: (sell price - entry price) * qty * multiplier
                    realized_pnl = (price - pos.entry_price) * close_qty * 10000
                else:
                    # Short position being closed: (entry price - buy price) * qty * multiplier
                    realized_pnl = (pos.entry_price - price) * close_qty * 10000
        
        # Log Trade
        self.trade_log.append({
            "date": date.strftime("%Y-%m-%d"),
            "symbol": order.symbol,
            "action": "BUY" if order.quantity > 0 else "SELL",
            "quantity": abs(order.quantity),
            "price": price,
            "fee": fee,
            "realized_pnl": round(realized_pnl, 2)
        })
        
        # Update Position
        if symbol not in self.context.positions:
            self.context.positions[symbol] = Position(symbol, 0.0, 0.0)
        
        pos = self.context.positions[symbol]
        
        # Avg Price Logic
        new_quantity = pos.quantity + order.quantity
        if new_quantity == 0:
            del self.context.positions[symbol]
        else:
            # If increasing pos, re-avg entry
            if (pos.quantity >= 0 and order.quantity > 0) or \
               (pos.quantity <= 0 and order.quantity < 0):
                total_val = (pos.quantity * pos.entry_price) + (order.quantity * price)
                pos.entry_price = total_val / new_quantity
            pos.quantity = new_quantity
            pos.current_price = price
            
        order.status = "FILLED"
        self.account.update_balances()

    def _liquidate_positions(self, market_data: pd.DataFrame):
        """Force close positions to restore margin."""
        # Simple LIFO: Close last added positions? 
        # Or Random? Or Largest Margin User?
        # MVP: Close all.
        print("   ☠️ LIQUIDATING ALL POSITIONS")
        self.context.close_all_positions()
        # Process immediately
        # self._process_orders(market_data, ...) # Need recursively call or next tick
        # Simpler: just clear for MVP
        self.context.positions.clear()
        self.account.update_balances()
