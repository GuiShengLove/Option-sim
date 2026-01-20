"""Verify increased margin utilization."""
import sys
sys.path.insert(0, 'e:/Quant_code/Option-sim/backend')

import requests
body = {
    'strategy_id': '641f1b06',  # weekly_put_selling
    'dataset_id': '510050_SH',
    'start_date': '2020-01-02',
    'end_date': '2020-02-28',
    'initial_capital': 1000000,
    'margin_scheme': 'SSE'
}

print("Testing increased position size (20 contracts)...")
response = requests.post('http://localhost:8000/api/backtest/run', json=body)
data = response.json()

if data.get('success'):
    equity_curve = data.get('equity_curve', [])
    
    # Find max margin utilization
    max_margin = max(p['margin_utilization'] for p in equity_curve)
    avg_margin = sum(p['margin_utilization'] for p in equity_curve) / len(equity_curve)
    
    # Find days with positions
    position_days = [p for p in equity_curve if p.get('position_count', 0) > 0]
    
    print(f"\n=== 保证金测试结果 ===")
    print(f"回测天数: {len(equity_curve)}")
    print(f"有持仓天数: {len(position_days)}")
    print(f"")
    print(f"保证金占用率统计:")
    print(f"  最大值: {max_margin * 100:.2f}%")
    print(f"  平均值: {avg_margin * 100:.2f}%")
    print(f"")
    
    # Show some sample days with positions
    if position_days:
        print("样本数据 (有持仓的日期):")
        for p in position_days[:5]:
            print(f"  {p['date']}: 保证金占用 {p['margin_utilization']*100:.2f}%, 持仓 {p['position_count']} 张")
else:
    print("Error:", data.get('message'))
