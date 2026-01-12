#!/usr/bin/env python3
"""
Cup with Handle Pattern Scanner for Japanese Stocks
====================================================
日本株全銘柄を対象としたカップ・ウィズ・ハンドルパターン検出ツール

Usage:
    python main.py                  # 全銘柄スクリーニング
    python main.py 7203 6758 8035   # 指定銘柄のみテスト
    python main.py --sample         # サンプル銘柄でテスト
    python main.py --ml-mode        # ML予測モード
    python main.py --help           # ヘルプ表示
"""

import sys
import os
import argparse
from datetime import datetime

# 依存パッケージのチェック
def check_dependencies():
    missing = []
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    try:
        import pandas
    except ImportError:
        missing.append('pandas')
    try:
        import scipy
    except ImportError:
        missing.append('scipy')
    try:
        import yfinance
    except ImportError:
        missing.append('yfinance')
    try:
        import requests
    except ImportError:
        missing.append('requests')

    if missing:
        print("Missing required packages:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

from screener import (
    screen_japan_stocks,
    screen_stocks,
    get_all_japan_stocks,
    JapanStockListFetcher,
    CupHandleScreener,
    StockDataFetcher,
    NIKKEI225_SAMPLE,
    JAPAN_SEMICONDUCTOR,
    JAPAN_GROWTH_SAMPLE
)
from cup_handle_detector import CupHandleDetector
from visualizer import plot_cup_handle, CupHandleVisualizer

# 出力ディレクトリ
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
ML_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml', 'models')

# ML関連のインポート（オプション）
ML_AVAILABLE = False
try:
    from ml.ensemble import SimpleEnsemble, create_ensemble_from_checkpoint
    from ml.gbm_model import GBMClassifier, HAS_LIGHTGBM
    from ml.feature_extractor import FeatureExtractor
    ML_AVAILABLE = True
except ImportError:
    pass


def run_full_screening(
    min_quality_score: float = 50.0,
    period: str = '2y',
    save_csv: bool = True,
    delay: float = 0.3
):
    """
    全銘柄スクリーニングを実行（直列処理・レート制限対策済み）
    """
    print("=" * 70)
    print("Full Japanese Stock Screening for Cup with Handle Pattern")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Min quality score: {min_quality_score}")
    print(f"Data period: {period}")
    print(f"Request delay: {delay}s")
    print()

    # JPX公式の銘柄リストを取得
    fetcher = JapanStockListFetcher()
    stock_df = fetcher.fetch()

    if stock_df.empty:
        print("Error: Failed to fetch stock list")
        return []

    all_symbols = fetcher.get_ticker_symbols()
    print(f"Total Japanese stocks (from JPX): {len(all_symbols)}")
    print()

    # 直列処理でスクリーニング
    screener = CupHandleScreener(min_quality_score=min_quality_score)

    all_results = screener.screen_symbols(
        all_symbols,
        period=period,
        delay=delay,
        show_progress=True
    )

    # 結果を集計
    print()
    print("=" * 70)
    print("Screening Complete")
    print("=" * 70)

    matches = screener.get_matches(all_results, min_score=min_quality_score)

    print(f"Total screened: {len(all_results)}")
    print(f"Trend Template passed: {sum(1 for r in all_results if r.trend_template_passed)}")
    print(f"Patterns detected: {len(matches)}")
    print()

    if matches:
        print("Top Matches (sorted by quality score):")
        print("-" * 70)
        for r in matches[:20]:  # 上位20件
            name = r.name[:20] if r.name else r.symbol
            print(f"  {r.symbol:8s} {name:22s} | Score: {r.pattern_quality_score:5.1f} | "
                  f"Cup: {r.cup_depth:5.1f}% | Pivot: ¥{r.pivot_price:,.0f}")

    # CSV保存
    if save_csv and all_results:
        import pandas as pd
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 全結果
        df_all = screener.to_dataframe(all_results)
        all_file = os.path.join(OUTPUT_DIR, f'cup_handle_all_{timestamp}.csv')
        df_all.to_csv(all_file, index=False, encoding='utf-8-sig')
        print(f"\nAll results saved to: {all_file}")

        # マッチのみ
        if matches:
            df_matches = pd.DataFrame([r.to_dict() for r in matches])
            matches_file = os.path.join(OUTPUT_DIR, f'cup_handle_matches_{timestamp}.csv')
            df_matches.to_csv(matches_file, index=False, encoding='utf-8-sig')
            print(f"Matches saved to: {matches_file}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results


def run_sample_test(sample_type: str = 'nikkei'):
    """
    サンプル銘柄でテスト
    """
    samples = {
        'nikkei': NIKKEI225_SAMPLE,
        'semiconductor': JAPAN_SEMICONDUCTOR,
        'growth': JAPAN_GROWTH_SAMPLE,
    }

    symbols = samples.get(sample_type, NIKKEI225_SAMPLE)

    print(f"Running sample test with {sample_type} stocks ({len(symbols)} stocks)...")
    return screen_stocks(symbols, min_quality_score=40.0, show_results=True)


def run_symbol_test(symbols: list, visualize: bool = False, ml_mode: bool = False):
    """
    指定銘柄をテスト
    """
    # .Tサフィックスを追加
    symbols = [s if s.endswith('.T') else f"{s}.T" for s in symbols]

    print(f"Testing specified stocks: {symbols}")

    if ml_mode and ML_AVAILABLE:
        # ML予測モード
        run_ml_prediction(symbols, visualize=visualize)
    else:
        df = screen_stocks(symbols, min_quality_score=30.0, show_results=True)

        # 可視化（オプション）
        if visualize:
            fetcher = StockDataFetcher()
            detector = CupHandleDetector()

            for symbol in symbols:
                stock_df, name = fetcher.fetch(symbol, period='2y')
                if not stock_df.empty:
                    result = detector.detect(stock_df)
                    if result['is_match'] or result['trend_template_passed']:
                        print(f"\nVisualizing {symbol} ({name})...")
                        import matplotlib.pyplot as plt
                        fig = plot_cup_handle(stock_df, result, title=f"{symbol} - {name}")
                        plt.show()

        return df


def run_ml_prediction(symbols: list, visualize: bool = False):
    """
    ML予測を実行

    Args:
        symbols: 銘柄コードリスト
        visualize: 可視化するか
    """
    if not ML_AVAILABLE:
        print("ML modules not available. Install dependencies with:")
        print("  pip install torch torchvision lightgbm scikit-learn")
        return

    print("=" * 70)
    print("ML-Enhanced Cup with Handle Pattern Detection")
    print("=" * 70)

    # モデル読み込み
    ensemble = None
    gbm_path = os.path.join(ML_MODEL_DIR, 'gbm_model.meta')

    if os.path.exists(gbm_path):
        try:
            ensemble = create_ensemble_from_checkpoint(ML_MODEL_DIR, use_cnn=False)
            print("Loaded ML model from checkpoint")
        except Exception as e:
            print(f"Could not load ML model: {e}")

    if ensemble is None:
        print("Using rule-based detection (no ML model found)")
        ensemble = SimpleEnsemble(gbm_model=None)

    # 銘柄ごとに予測
    fetcher = StockDataFetcher()
    detector = CupHandleDetector()

    results = []

    for symbol in symbols:
        print(f"\n--- {symbol} ---")

        stock_df, name = fetcher.fetch(symbol, period='2y')
        if stock_df.empty:
            print(f"  Could not fetch data")
            continue

        # パターン検出
        result = detector.detect(stock_df)

        if result.get('is_match'):
            # ML予測
            prediction = ensemble.predict(stock_df, result)

            print(f"  Name: {name}")
            print(f"  Pattern Detected: Yes")
            print(f"  Quality Score: {result.get('pattern_quality_score', 0):.1f}")
            print(f"  Cup Depth: {result.get('cup_depth', 0):.1f}%")
            print(f"  Handle Depth: {result.get('handle_depth', 0):.1f}%")
            print(f"  Pivot Price: ¥{result.get('pivot_price', 0):,.0f}")
            print()
            print(f"  [ML Prediction]")
            print(f"  Success Probability: {prediction.probability:.1%}")
            print(f"  Confidence: {prediction.confidence}")
            print(f"  Recommendation: {prediction.recommendation}")

            results.append({
                'symbol': symbol,
                'name': name,
                'quality_score': result.get('pattern_quality_score', 0),
                'ml_probability': prediction.probability,
                'confidence': prediction.confidence,
                'recommendation': prediction.recommendation,
            })

            # 可視化
            if visualize:
                import matplotlib.pyplot as plt
                title = f"{symbol} - {name}\nML Score: {prediction.probability:.1%} ({prediction.confidence})"
                fig = plot_cup_handle(stock_df, result, title=title)
                plt.show()

        elif result.get('trend_template_passed'):
            print(f"  Name: {name}")
            print(f"  Trend Template: Passed (but no Cup with Handle pattern)")
        else:
            print(f"  Name: {name}")
            print(f"  Trend Template: Not passed (not in uptrend)")

    # サマリー
    if results:
        print("\n" + "=" * 70)
        print("Summary - Stocks with Cup with Handle Pattern")
        print("=" * 70)
        print(f"{'Symbol':<10} {'Name':<20} {'Quality':<10} {'ML Prob':<10} {'Confidence':<12}")
        print("-" * 70)

        for r in sorted(results, key=lambda x: x['ml_probability'], reverse=True):
            print(f"{r['symbol']:<10} {r['name'][:18]:<20} {r['quality_score']:<10.1f} "
                  f"{r['ml_probability']:<10.1%} {r['confidence']:<12}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Cup with Handle Pattern Scanner for Japanese Stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Screen all Japanese stocks
  python main.py 7203 6758 8035     # Test specific stocks
  python main.py --sample nikkei    # Test with Nikkei 225 sample
  python main.py --sample semiconductor  # Test semiconductor stocks
  python main.py 7203 --visualize   # Test with chart visualization
  python main.py 7203 --ml-mode     # Test with ML prediction
        """
    )

    parser.add_argument(
        'symbols',
        nargs='*',
        help='Stock codes to test (e.g., 7203 6758)'
    )
    parser.add_argument(
        '--sample',
        choices=['nikkei', 'semiconductor', 'growth'],
        help='Run with sample stock list'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=50.0,
        help='Minimum quality score (default: 50.0)'
    )
    parser.add_argument(
        '--period',
        default='2y',
        help='Data period (default: 2y)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to CSV'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show chart visualization for matched patterns'
    )
    parser.add_argument(
        '--ml-mode',
        action='store_true',
        help='Enable ML-enhanced prediction mode'
    )

    args = parser.parse_args()

    try:
        if args.ml_mode and args.symbols:
            # ML予測モード
            run_ml_prediction(
                [s if s.endswith('.T') else f"{s}.T" for s in args.symbols],
                visualize=args.visualize
            )

        elif args.sample:
            # サンプルテスト
            if args.ml_mode:
                samples = {
                    'nikkei': NIKKEI225_SAMPLE,
                    'semiconductor': JAPAN_SEMICONDUCTOR,
                    'growth': JAPAN_GROWTH_SAMPLE,
                }
                run_ml_prediction(samples.get(args.sample, NIKKEI225_SAMPLE)[:10])
            else:
                run_sample_test(args.sample)

        elif args.symbols:
            # 指定銘柄テスト
            run_symbol_test(args.symbols, visualize=args.visualize, ml_mode=args.ml_mode)

        else:
            # 全銘柄スクリーニング
            run_full_screening(
                min_quality_score=args.min_score,
                period=args.period,
                save_csv=not args.no_save
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
