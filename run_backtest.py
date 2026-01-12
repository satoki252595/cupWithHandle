#!/usr/bin/env python3
"""
簡易バックテスト実行スクリプト
パラメータを絞って高速に実行
"""

import os
import sys
from datetime import datetime

from backtest import (
    select_diversified_stocks,
    CupHandleBacktester,
    BacktestResult,
    generate_report
)

def main():
    print("=" * 70)
    print("Cup with Handle Pattern Backtest (Optimized)")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 100銘柄を選定
    print("\n[1/4] Selecting 100 diversified stocks...")
    symbols = select_diversified_stocks(100)
    print(f"Selected {len(symbols)} stocks")

    # パラメータ組み合わせ（絞り込み版: 2×2×2×2 = 16通り）
    param_combinations = [
        # 基本パラメータ
        {'cup_min_depth_pct': 12.0, 'cup_max_depth_pct': 33.0, 'handle_max_depth_pct': 15.0, 'peak_order': 7},
        # カップ深さを調整
        {'cup_min_depth_pct': 15.0, 'cup_max_depth_pct': 30.0, 'handle_max_depth_pct': 15.0, 'peak_order': 7},
        {'cup_min_depth_pct': 10.0, 'cup_max_depth_pct': 35.0, 'handle_max_depth_pct': 15.0, 'peak_order': 7},
        # ハンドル深さを調整
        {'cup_min_depth_pct': 12.0, 'cup_max_depth_pct': 33.0, 'handle_max_depth_pct': 12.0, 'peak_order': 7},
        {'cup_min_depth_pct': 12.0, 'cup_max_depth_pct': 33.0, 'handle_max_depth_pct': 18.0, 'peak_order': 7},
        # ピーク検出を調整
        {'cup_min_depth_pct': 12.0, 'cup_max_depth_pct': 33.0, 'handle_max_depth_pct': 15.0, 'peak_order': 5},
        {'cup_min_depth_pct': 12.0, 'cup_max_depth_pct': 33.0, 'handle_max_depth_pct': 15.0, 'peak_order': 10},
        # 複合調整
        {'cup_min_depth_pct': 15.0, 'cup_max_depth_pct': 30.0, 'handle_max_depth_pct': 12.0, 'peak_order': 7},
        {'cup_min_depth_pct': 15.0, 'cup_max_depth_pct': 30.0, 'handle_max_depth_pct': 12.0, 'peak_order': 5},
        {'cup_min_depth_pct': 10.0, 'cup_max_depth_pct': 35.0, 'handle_max_depth_pct': 18.0, 'peak_order': 10},
        # 厳格版
        {'cup_min_depth_pct': 15.0, 'cup_max_depth_pct': 25.0, 'handle_max_depth_pct': 10.0, 'peak_order': 7},
        {'cup_min_depth_pct': 18.0, 'cup_max_depth_pct': 28.0, 'handle_max_depth_pct': 10.0, 'peak_order': 7},
        # 緩和版
        {'cup_min_depth_pct': 8.0, 'cup_max_depth_pct': 40.0, 'handle_max_depth_pct': 20.0, 'peak_order': 5},
        {'cup_min_depth_pct': 10.0, 'cup_max_depth_pct': 38.0, 'handle_max_depth_pct': 18.0, 'peak_order': 6},
        # カップ期間を調整
        {'cup_min_depth_pct': 12.0, 'cup_max_depth_pct': 33.0, 'handle_max_depth_pct': 15.0, 'peak_order': 7, 'cup_min_days': 30, 'cup_max_days': 250},
        {'cup_min_depth_pct': 12.0, 'cup_max_depth_pct': 33.0, 'handle_max_depth_pct': 15.0, 'peak_order': 7, 'cup_min_days': 40, 'cup_max_days': 300},
    ]

    print(f"\n[2/4] Running {len(param_combinations)} parameter combinations...")

    backtester = CupHandleBacktester(
        holding_period=20,
        stop_loss_pct=-8.0,
        take_profit_pct=20.0,
    )

    results = []

    for i, params in enumerate(param_combinations):
        print(f"\nCombination {i + 1}/{len(param_combinations)}")
        print(f"  Params: {params}")

        result = backtester.run_backtest(
            symbols,
            params,
            min_quality_score=50.0,
            show_progress=True
        )

        results.append(result)

        print(f"  => Trades: {result.total_trades} | "
              f"Win Rate: {result.win_rate:.1f}% | "
              f"Avg Return: {result.avg_return:+.2f}% | "
              f"Sharpe: {result.sharpe_ratio:.2f}")

    # スコアでソート
    print("\n[3/4] Analyzing results...")
    results.sort(key=lambda x: (x.sharpe_ratio * x.win_rate / 100) if x.total_trades > 5 else -999, reverse=True)

    # 上位結果を表示
    print("\n" + "=" * 70)
    print("TOP 5 Parameter Sets")
    print("=" * 70)

    for i, r in enumerate(results[:5]):
        print(f"\n#{i + 1}:")
        print(f"  Params: {r.params}")
        print(f"  Trades: {r.total_trades} | Win Rate: {r.win_rate:.1f}% | "
              f"Avg Return: {r.avg_return:+.2f}% | Sharpe: {r.sharpe_ratio:.2f}")

    # レポート生成
    print("\n[4/4] Generating report...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'backtest_report_{timestamp}.md')

    generate_report(results, symbols, report_path)

    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("Backtest complete!")

    return results


if __name__ == '__main__':
    main()
