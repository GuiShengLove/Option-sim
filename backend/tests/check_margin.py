import requests

body = {
    'strategy_id': '641f1b06',
    'dataset_id': '510050_SH',
    'start_date': '2020-01-02',
    'end_date': '2020-01-31',
    'initial_capital': 1000000,
    'margin_scheme': 'SSE'
}

response = requests.post('http://localhost:8000/api/backtest/run', json=body)
data = response.json()

equity_curve = data.get('equity_curve', [])
print(f'Equity curve points: {len(equity_curve)}')

print('\nFirst 5 points:')
for p in equity_curve[:5]:
    print(f"  {p['date']}: margin_utilization={p['margin_utilization']:.4f}, position_count={p.get('position_count', 0)}")

print('\nLast 3 points:')
for p in equity_curve[-3:]:
    print(f"  {p['date']}: margin_utilization={p['margin_utilization']:.4f}, position_count={p.get('position_count', 0)}")

# Check if all margin_utilization values are 0
all_zero = all(p['margin_utilization'] == 0 for p in equity_curve)
print(f'\nAll margin_utilization zero: {all_zero}')
