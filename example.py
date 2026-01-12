"""
Cup with Handle Detection - Complete Example
=============================================
全機能を使った統合的なサンプルスクリプト
"""

import pandas as pd
import matplotlib.pyplot as plt

from cup_handle_detector import CupHandleDetector, create_sample_data
from visualizer import CupHandleVisualizer, plot_cup_handle
from screener import (
    CupHandleScreener,
    StockDataFetcher,
    screen_stocks,
    US_GROWTH_SAMPLE,
    NIKKEI225_SAMPLE,
    YFINANCE_AVAILABLE
)


def example_basic_detection():
    """基本的なパターン検出の例"""
    print("\n" + "="*60)
    print("Example 1: Basic Pattern Detection with Sample Data")
    print("="*60)

    # サンプルデータでテスト
    df = create_sample_data()
    detector = CupHandleDetector()
    result = detector.detect(df)

    print("\nDetection Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    return df, result


def example_real_stock_detection(symbol: str = 'AAPL'):
    """実際の株価データでの検出例"""
    print("\n" + "="*60)
    print(f"Example 2: Real Stock Detection ({symbol})")
    print("="*60)

    if not YFINANCE_AVAILABLE:
        print("yfinance is not available. Skipping this example.")
        return None, None

    fetcher = StockDataFetcher()
    df, name = fetcher.fetch(symbol, period='2y')

    if df.empty:
        print(f"Failed to fetch data for {symbol}")
        return None, None

    print(f"\nStock: {name} ({symbol})")
    print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Current price: ${df['Close'].iloc[-1]:.2f}")

    detector = CupHandleDetector()
    result = detector.detect(df)

    print("\nDetection Result:")
    print(f"  Pattern Detected: {result['is_match']}")
    print(f"  Trend Template Passed: {result['trend_template_passed']}")

    if result['is_match']:
        print(f"  Cup Depth: {result['cup_depth']}%")
        print(f"  Handle Depth: {result['handle_depth']}%")
        print(f"  Pivot Price: ${result['pivot_price']}")
        print(f"  Quality Score: {result['pattern_quality_score']}")
        print(f"  Volume Valid: {result.get('volume_is_valid', 'N/A')}")

    return df, result


def example_visualization(df, result, title="Cup with Handle Pattern"):
    """可視化の例"""
    print("\n" + "="*60)
    print("Example 3: Pattern Visualization")
    print("="*60)

    if df is None or result is None:
        print("No data to visualize.")
        return

    visualizer = CupHandleVisualizer()
    fig = visualizer.plot(
        df, result,
        show_volume=True,
        show_sma=True,
        title=title
    )

    print("Chart created. Close the window to continue...")
    plt.show()


def example_screening():
    """複数銘柄のスクリーニング例"""
    print("\n" + "="*60)
    print("Example 4: Multi-Stock Screening")
    print("="*60)

    if not YFINANCE_AVAILABLE:
        print("yfinance is not available. Skipping this example.")
        return

    # 米国テック株のサンプルでスクリーニング
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AMD', 'CRM']
    print(f"\nScreening {len(symbols)} stocks...")

    df_results = screen_stocks(
        symbols,
        min_quality_score=50.0,
        period='2y',
        show_results=True
    )

    return df_results


def example_custom_parameters():
    """カスタムパラメータでの検出例"""
    print("\n" + "="*60)
    print("Example 5: Detection with Custom Parameters")
    print("="*60)

    # より厳格な条件でのカスタム設定
    strict_detector = CupHandleDetector(
        # カップの条件を厳格に
        cup_min_depth_pct=15.0,  # 最低15%の深さ
        cup_max_depth_pct=30.0,  # 最大30%の深さ
        cup_lip_balance_min=0.90,  # 左右バランス90-110%
        cup_lip_balance_max=1.10,

        # ハンドルの条件を厳格に
        handle_max_depth_pct=12.0,  # 最大12%の深さ

        # ピーク検出を調整
        peak_order=10,  # より大きなピークのみ検出
    )

    # 緩い条件でのカスタム設定
    loose_detector = CupHandleDetector(
        cup_min_depth_pct=10.0,
        cup_max_depth_pct=40.0,
        cup_lip_balance_min=0.80,
        cup_lip_balance_max=1.20,
        handle_max_depth_pct=18.0,
        peak_order=5,
    )

    df = create_sample_data()

    print("\nStrict Parameters:")
    result_strict = strict_detector.detect(df)
    print(f"  Pattern Detected: {result_strict['is_match']}")
    if result_strict['is_match']:
        print(f"  Quality Score: {result_strict['pattern_quality_score']}")

    print("\nLoose Parameters:")
    result_loose = loose_detector.detect(df)
    print(f"  Pattern Detected: {result_loose['is_match']}")
    if result_loose['is_match']:
        print(f"  Quality Score: {result_loose['pattern_quality_score']}")

    return strict_detector, loose_detector


def example_volume_analysis():
    """ボリューム分析の詳細例"""
    print("\n" + "="*60)
    print("Example 6: Volume Analysis Details")
    print("="*60)

    df = create_sample_data()
    detector = CupHandleDetector()
    result = detector.detect(df)

    if not result['is_match']:
        print("No pattern detected for volume analysis.")
        return

    print("\nVolume Analysis Results:")
    print(f"  Volume Valid: {result.get('volume_is_valid', 'N/A')}")
    print(f"  Cup Left Volume Decline: {result.get('cup_left_volume_decline', 'N/A')}")
    print(f"  Cup Bottom Dry Up: {result.get('cup_bottom_dry_up', 'N/A')}")
    print(f"  Cup Right Volume Increase: {result.get('cup_right_volume_increase', 'N/A')}")
    print(f"  Handle Volume Contraction: {result.get('handle_volume_contraction', 'N/A')}")
    print(f"  Breakout Volume Surge: {result.get('breakout_volume_surge', 'N/A')}")
    print(f"  50-day Avg Volume: {result.get('avg_volume_50d', 'N/A')}")
    print(f"  Breakout Volume Ratio: {result.get('breakout_volume_ratio', 'N/A')}x")

    return result


def main():
    """全てのサンプルを実行"""
    print("\n" + "#"*60)
    print("# Cup with Handle Pattern Detection - Complete Examples")
    print("#"*60)

    # Example 1: 基本的な検出
    df_sample, result_sample = example_basic_detection()

    # Example 2: 実データでの検出
    df_real, result_real = example_real_stock_detection('GOOGL')

    # Example 3: 可視化（最後に表示）
    # example_visualization(df_sample, result_sample, "Sample Data - Cup with Handle")

    # Example 4: スクリーニング
    df_screening = example_screening()

    # Example 5: カスタムパラメータ
    example_custom_parameters()

    # Example 6: ボリューム分析
    example_volume_analysis()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)

    # 可視化を表示（コメント解除で実行）
    print("\nTo visualize the patterns, uncomment the visualization calls in main()")
    print("or run: python -c \"from example import *; example_visualization(...)\"")


if __name__ == '__main__':
    main()
