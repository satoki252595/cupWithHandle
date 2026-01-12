"""
Stock Screener for Cup with Handle Pattern
==========================================
複数銘柄からカップ・ウィズ・ハンドルパターンを検出するスクリーニング機能。
yfinanceを使用した実データの取得に対応。
日本株（東証）に最適化。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from io import StringIO
import warnings
import requests
import time
import os
import sys
from dataclasses import dataclass

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance is not installed. Install with: pip install yfinance")

from cup_handle_detector import CupHandleDetector


class JapanStockListFetcher:
    """
    日本株の銘柄一覧を取得するクラス
    JPX（日本取引所グループ）の公式データを使用
    """

    # JPX公式の上場銘柄一覧（Excelファイル）
    JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

    def __init__(self, cache_dir: str = None):
        self._stock_list = None
        self._stock_dict = None
        self._cache_dir = cache_dir or os.path.dirname(os.path.abspath(__file__))
        self._cache_file = os.path.join(self._cache_dir, 'jpx_stock_list.csv')

    def fetch(self, force_refresh: bool = False, use_cache: bool = True) -> pd.DataFrame:
        """
        日本の上場銘柄一覧を取得（JPX公式データ）

        Returns:
            DataFrame with columns: stock_code, company_name, market, sector
        """
        if self._stock_list is not None and not force_refresh:
            return self._stock_list

        # キャッシュを確認（1日以内なら使用）
        if use_cache and os.path.exists(self._cache_file) and not force_refresh:
            cache_age = time.time() - os.path.getmtime(self._cache_file)
            if cache_age < 86400:  # 24時間
                try:
                    print("Loading stock list from cache...")
                    self._stock_list = pd.read_csv(self._cache_file)
                    self._stock_dict = dict(zip(
                        self._stock_list['stock_code'],
                        self._stock_list['company_name']
                    ))
                    print(f"Loaded {len(self._stock_list)} stocks from cache")
                    return self._stock_list
                except Exception:
                    pass

        try:
            print("Fetching Japanese stock list from JPX...")

            # JPXのExcelファイルを直接読み込み
            df = pd.read_excel(self.JPX_URL, header=0)

            # カラム名を確認して正規化
            # JPXのフォーマット: コード, 銘柄名, 市場・商品区分, 33業種コード, 33業種区分, ...
            df.columns = df.columns.str.strip()

            # 必要なカラムを抽出
            result = pd.DataFrame()
            result['stock_code'] = df['コード'].astype(str).str.zfill(4)
            result['company_name'] = df['銘柄名']
            result['market'] = df['市場・商品区分'] if '市場・商品区分' in df.columns else ''
            result['sector'] = df['33業種区分'] if '33業種区分' in df.columns else ''

            # 内国株式のみフィルタ（ETF、REITなどを除外）
            # 市場区分に「内国株式」が含まれるもののみ
            if 'market' in result.columns:
                stock_markets = ['プライム', 'スタンダード', 'グロース']
                mask = result['market'].str.contains('|'.join(stock_markets), na=False)
                result = result[mask]

            # 重複削除
            result = result.drop_duplicates(subset=['stock_code'])
            result = result.reset_index(drop=True)

            self._stock_list = result
            self._stock_dict = dict(zip(result['stock_code'], result['company_name']))

            # キャッシュに保存
            try:
                result.to_csv(self._cache_file, index=False, encoding='utf-8-sig')
            except Exception:
                pass

            print(f"Loaded {len(result)} Japanese listed stocks from JPX")
            return result

        except Exception as e:
            warnings.warn(f"Failed to fetch from JPX: {str(e)}")
            # フォールバック: キャッシュがあれば使用
            if os.path.exists(self._cache_file):
                try:
                    self._stock_list = pd.read_csv(self._cache_file)
                    self._stock_dict = dict(zip(
                        self._stock_list['stock_code'],
                        self._stock_list['company_name']
                    ))
                    print(f"Using cached stock list ({len(self._stock_list)} stocks)")
                    return self._stock_list
                except Exception:
                    pass
            return pd.DataFrame(columns=['stock_code', 'company_name', 'market', 'sector'])

    def get_all_codes(self) -> List[str]:
        """全銘柄の証券コードを取得（4桁）"""
        if self._stock_list is None:
            self.fetch()
        return self._stock_list['stock_code'].tolist()

    def get_ticker_symbols(self) -> List[str]:
        """全銘柄のYahoo Finance用ティッカーシンボルを取得（.T形式）"""
        codes = self.get_all_codes()
        return [f"{code}.T" for code in codes]

    def get_company_name(self, stock_code: str) -> Optional[str]:
        """証券コードから会社名を取得"""
        if self._stock_dict is None:
            self.fetch()
        # 4桁に正規化
        code = stock_code.replace('.T', '').zfill(4)
        return self._stock_dict.get(code)

    def code_to_ticker(self, stock_code: str) -> str:
        """証券コードをYahoo Financeティッカーに変換"""
        code = stock_code.replace('.T', '').zfill(4)
        return f"{code}.T"

    def get_by_market(self, market: str) -> List[str]:
        """市場区分で絞り込み（プライム、スタンダード、グロース）"""
        if self._stock_list is None:
            self.fetch()
        mask = self._stock_list['market'].str.contains(market, na=False)
        return self._stock_list[mask]['stock_code'].tolist()

    def get_by_sector(self, sector: str) -> List[str]:
        """業種で絞り込み"""
        if self._stock_list is None:
            self.fetch()
        mask = self._stock_list['sector'].str.contains(sector, na=False)
        return self._stock_list[mask]['stock_code'].tolist()


@dataclass
class ScreenerResult:
    """スクリーニング結果を格納"""
    symbol: str
    name: Optional[str]
    is_match: bool
    pattern_quality_score: Optional[float]
    cup_depth: Optional[float]
    handle_depth: Optional[float]
    pivot_price: Optional[float]
    current_price: Optional[float]
    distance_to_pivot_pct: Optional[float]
    volume_valid: Optional[bool]
    trend_template_passed: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'is_match': self.is_match,
            'pattern_quality_score': self.pattern_quality_score,
            'cup_depth': self.cup_depth,
            'handle_depth': self.handle_depth,
            'pivot_price': self.pivot_price,
            'current_price': self.current_price,
            'distance_to_pivot_pct': self.distance_to_pivot_pct,
            'volume_valid': self.volume_valid,
            'trend_template_passed': self.trend_template_passed,
            'error': self.error,
        }


class StockDataFetcher:
    """
    株価データ取得クラス
    """

    def __init__(self, data_source: str = 'yfinance'):
        """
        Args:
            data_source: データソース ('yfinance' or 'custom')
        """
        self.data_source = data_source

        if data_source == 'yfinance' and not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

    def fetch(
        self,
        symbol: str,
        period: str = '2y',
        interval: str = '1d'
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        株価データを取得

        Args:
            symbol: 銘柄コード
            period: 期間 ('1y', '2y', '5y', 'max' など)
            interval: 足の間隔 ('1d', '1wk' など)

        Returns:
            (DataFrame, company_name) または エラー時は空のDataFrame
        """
        if self.data_source == 'yfinance':
            return self._fetch_yfinance(symbol, period, interval)
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

    def _fetch_yfinance(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """yfinanceからデータ取得"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return pd.DataFrame(), None

            # カラム名を標準化
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })

            # 必要なカラムのみ抽出
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            # 会社名を取得
            try:
                info = ticker.info
                name = info.get('longName') or info.get('shortName') or symbol
            except Exception:
                name = symbol

            return df, name

        except Exception as e:
            warnings.warn(f"Error fetching {symbol}: {str(e)}")
            return pd.DataFrame(), None

    def fetch_multiple(
        self,
        symbols: List[str],
        period: str = '2y',
        delay: float = 0.5,
        show_progress: bool = True
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[str]]]:
        """
        複数銘柄のデータを直列取得（レート制限対策）

        Args:
            symbols: 銘柄コードのリスト
            period: 期間
            delay: 各リクエスト間の待機時間（秒）
            show_progress: 進捗表示

        Returns:
            {symbol: (df, name)} の辞書
        """
        results = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            try:
                df, name = self.fetch(symbol, period)
                results[symbol] = (df, name)

                # 進捗表示（10件ごと）
                if show_progress and (i + 1) % 10 == 0:
                    print(f"  Fetched {i + 1}/{total} symbols...", file=sys.stderr)

            except Exception as e:
                results[symbol] = (pd.DataFrame(), None)

            # レート制限対策の待機
            if i < total - 1:
                time.sleep(delay)

        return results


class CupHandleScreener:
    """
    カップ・ウィズ・ハンドル パターンスクリーナー
    """

    def __init__(
        self,
        detector: CupHandleDetector = None,
        fetcher: StockDataFetcher = None,
        min_quality_score: float = 50.0
    ):
        """
        Args:
            detector: CupHandleDetector インスタンス
            fetcher: StockDataFetcher インスタンス
            min_quality_score: 最低品質スコア閾値
        """
        self.detector = detector or CupHandleDetector()
        self.fetcher = fetcher or StockDataFetcher()
        self.min_quality_score = min_quality_score

    def screen_symbol(
        self,
        symbol: str,
        df: pd.DataFrame = None,
        name: str = None,
        period: str = '2y'
    ) -> ScreenerResult:
        """
        単一銘柄をスクリーニング

        Args:
            symbol: 銘柄コード
            df: 株価データ（Noneの場合は取得）
            name: 会社名
            period: データ取得期間

        Returns:
            ScreenerResult
        """
        try:
            # データ取得
            if df is None or df.empty:
                df, name = self.fetcher.fetch(symbol, period)

            if df is None or df.empty:
                return ScreenerResult(
                    symbol=symbol,
                    name=name,
                    is_match=False,
                    pattern_quality_score=None,
                    cup_depth=None,
                    handle_depth=None,
                    pivot_price=None,
                    current_price=None,
                    distance_to_pivot_pct=None,
                    volume_valid=None,
                    trend_template_passed=False,
                    error="Failed to fetch data"
                )

            # パターン検出
            result = self.detector.detect(df)

            # 現在価格とピボットまでの距離
            current_price = df['Close'].iloc[-1]
            pivot_price = result.get('pivot_price')

            if pivot_price and current_price:
                distance_to_pivot = ((pivot_price - current_price) / current_price) * 100
            else:
                distance_to_pivot = None

            return ScreenerResult(
                symbol=symbol,
                name=name,
                is_match=result.get('is_match', False),
                pattern_quality_score=result.get('pattern_quality_score'),
                cup_depth=result.get('cup_depth'),
                handle_depth=result.get('handle_depth'),
                pivot_price=pivot_price,
                current_price=round(current_price, 2),
                distance_to_pivot_pct=round(distance_to_pivot, 2) if distance_to_pivot else None,
                volume_valid=result.get('volume_is_valid'),
                trend_template_passed=result.get('trend_template_passed', False),
            )

        except Exception as e:
            return ScreenerResult(
                symbol=symbol,
                name=name,
                is_match=False,
                pattern_quality_score=None,
                cup_depth=None,
                handle_depth=None,
                pivot_price=None,
                current_price=None,
                distance_to_pivot_pct=None,
                volume_valid=None,
                trend_template_passed=False,
                error=str(e)
            )

    def screen_symbols(
        self,
        symbols: List[str],
        period: str = '2y',
        delay: float = 0.5,
        show_progress: bool = True
    ) -> List[ScreenerResult]:
        """
        複数銘柄をスクリーニング（直列処理・レート制限対策済み）

        Args:
            symbols: 銘柄コードのリスト
            period: データ取得期間
            delay: 各リクエスト間の待機時間（秒）
            show_progress: 進捗表示

        Returns:
            ScreenerResult のリスト
        """
        results = []
        total = len(symbols)
        matches_found = 0
        errors = 0

        if show_progress:
            print(f"Screening {total} symbols (delay: {delay}s per request)...")

        for i, symbol in enumerate(symbols):
            try:
                # データ取得
                df, name = self.fetcher.fetch(symbol, period)

                if df is None or df.empty:
                    errors += 1
                    results.append(ScreenerResult(
                        symbol=symbol,
                        name=name,
                        is_match=False,
                        pattern_quality_score=None,
                        cup_depth=None,
                        handle_depth=None,
                        pivot_price=None,
                        current_price=None,
                        distance_to_pivot_pct=None,
                        volume_valid=None,
                        trend_template_passed=False,
                        error="No data"
                    ))
                else:
                    # パターン検出
                    result = self.screen_symbol(symbol, df, name, period)
                    results.append(result)

                    if result.is_match:
                        matches_found += 1
                        if show_progress:
                            print(f"  Found: {symbol} ({name}) - Score: {result.pattern_quality_score}")

            except Exception as e:
                errors += 1
                results.append(ScreenerResult(
                    symbol=symbol,
                    name=None,
                    is_match=False,
                    pattern_quality_score=None,
                    cup_depth=None,
                    handle_depth=None,
                    pivot_price=None,
                    current_price=None,
                    distance_to_pivot_pct=None,
                    volume_valid=None,
                    trend_template_passed=False,
                    error=str(e)
                ))

            # 進捗表示（50件ごと）
            if show_progress and (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{total} | Matches: {matches_found} | Errors: {errors}")

            # レート制限対策の待機
            if i < total - 1:
                time.sleep(delay)

        if show_progress:
            print(f"  Complete: {total}/{total} | Matches: {matches_found} | Errors: {errors}")

        return results

    def get_matches(
        self,
        results: List[ScreenerResult],
        min_score: float = None,
        sort_by: str = 'pattern_quality_score'
    ) -> List[ScreenerResult]:
        """
        マッチした銘柄のみをフィルタリング

        Args:
            results: スクリーニング結果
            min_score: 最低スコア（Noneの場合はself.min_quality_score）
            sort_by: ソート基準

        Returns:
            フィルタリング・ソートされた結果
        """
        min_score = min_score or self.min_quality_score

        matches = [
            r for r in results
            if r.is_match and r.pattern_quality_score and r.pattern_quality_score >= min_score
        ]

        # ソート
        if sort_by == 'pattern_quality_score':
            matches.sort(key=lambda x: x.pattern_quality_score or 0, reverse=True)
        elif sort_by == 'distance_to_pivot':
            matches.sort(key=lambda x: x.distance_to_pivot_pct or float('inf'))

        return matches

    def to_dataframe(self, results: List[ScreenerResult]) -> pd.DataFrame:
        """結果をDataFrameに変換"""
        return pd.DataFrame([r.to_dict() for r in results])


# 日本株用のプリセット銘柄リスト
NIKKEI225_SAMPLE = [
    '7203.T',  # トヨタ
    '6758.T',  # ソニー
    '9984.T',  # ソフトバンクG
    '6861.T',  # キーエンス
    '8035.T',  # 東京エレクトロン
    '6098.T',  # リクルート
    '4063.T',  # 信越化学
    '6902.T',  # デンソー
    '7741.T',  # HOYA
    '9433.T',  # KDDI
]

# 日本成長株サンプル
JAPAN_GROWTH_SAMPLE = [
    '4385.T',  # メルカリ
    '4478.T',  # フリー
    '4443.T',  # Sansan
    '7342.T',  # ウェルスナビ
    '4176.T',  # ココナラ
    '4057.T',  # インターファクトリー
    '4051.T',  # GMOフィナンシャルG
    '7342.T',  # ウェルスナビ
    '4167.T',  # ココペリ
    '4171.T',  # グローバルインフォ
]

# 半導体関連
JAPAN_SEMICONDUCTOR = [
    '8035.T',  # 東京エレクトロン
    '6857.T',  # アドバンテスト
    '6920.T',  # レーザーテック
    '6146.T',  # ディスコ
    '7735.T',  # SCREENホールディングス
    '6728.T',  # アルバック
    '6963.T',  # ローム
    '6723.T',  # ルネサス
    '4186.T',  # 東京応化工業
    '3436.T',  # SUMCO
]

US_GROWTH_SAMPLE = [
    'AAPL',
    'MSFT',
    'GOOGL',
    'AMZN',
    'NVDA',
    'META',
    'TSLA',
    'AMD',
    'CRM',
    'ADBE',
]


def get_all_japan_stocks() -> List[str]:
    """
    日本の全上場銘柄のティッカーシンボルを取得

    Returns:
        Yahoo Finance形式のティッカーリスト（.T形式）
    """
    fetcher = JapanStockListFetcher()
    return fetcher.get_ticker_symbols()


def screen_stocks(
    symbols: List[str],
    min_quality_score: float = 50.0,
    period: str = '2y',
    show_results: bool = True,
    delay: float = 0.3
) -> pd.DataFrame:
    """
    便利関数: 銘柄リストからカップ・ウィズ・ハンドルをスクリーニング

    Args:
        symbols: 銘柄コードのリスト
        min_quality_score: 最低品質スコア
        period: データ取得期間
        show_results: 結果を表示するか
        delay: リクエスト間の待機時間（秒）

    Returns:
        結果のDataFrame
    """
    screener = CupHandleScreener(min_quality_score=min_quality_score)
    results = screener.screen_symbols(symbols, period, delay=delay, show_progress=show_results)
    matches = screener.get_matches(results)

    if show_results:
        print(f"\n{'='*70}")
        print(f"Cup with Handle Screening Results")
        print(f"{'='*70}")
        print(f"Total symbols screened: {len(symbols)}")
        print(f"Trend Template passed: {sum(1 for r in results if r.trend_template_passed)}")
        print(f"Patterns detected: {len(matches)}")
        print()

        if matches:
            print("Matches (sorted by quality score):")
            print("-" * 70)
            for r in matches:
                name = r.name[:15] if r.name else r.symbol
                print(f"  {r.symbol:8s} {name:16s} | Score: {r.pattern_quality_score:5.1f} | "
                      f"Cup: {r.cup_depth:5.1f}% | Handle: {r.handle_depth:5.1f}% | "
                      f"Pivot: ¥{r.pivot_price:,.0f}")
        else:
            print("No patterns detected matching criteria.")

    return screener.to_dataframe(results)


def screen_japan_stocks(
    symbols: List[str] = None,
    min_quality_score: float = 50.0,
    period: str = '2y',
    save_csv: bool = True,
    output_dir: str = 'output'
) -> pd.DataFrame:
    """
    日本株のカップ・ウィズ・ハンドルスクリーニング

    Args:
        symbols: 銘柄リスト（Noneの場合は全上場銘柄）
        min_quality_score: 最低品質スコア
        period: データ取得期間
        save_csv: CSV保存するか
        output_dir: 出力ディレクトリ

    Returns:
        結果のDataFrame
    """
    import os

    if symbols is None:
        # 全銘柄を取得
        symbols = get_all_japan_stocks()
        print(f"Screening all {len(symbols)} Japanese listed stocks...")
    else:
        # .Tサフィックスを追加（なければ）
        symbols = [s if s.endswith('.T') else f"{s}.T" for s in symbols]

    # スクリーニング実行
    df_results = screen_stocks(
        symbols,
        min_quality_score=min_quality_score,
        period=period,
        show_results=True,
        delay=0.5,  # レート制限対策
    )

    # CSV保存
    if save_csv and len(df_results) > 0:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}/cup_handle_japan_{timestamp}.csv"
        df_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {filename}")

    return df_results


if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Cup with Handle Pattern Screener for Japanese Stocks")
    print("=" * 60)

    if not YFINANCE_AVAILABLE:
        print("yfinance is not installed. Install with: pip install yfinance")
        sys.exit(1)

    # コマンドライン引数をチェック
    if len(sys.argv) > 1:
        # 引数がある場合は指定銘柄をテスト
        symbols = sys.argv[1:]
        print(f"\nTesting with specified stocks: {symbols}")
        df = screen_japan_stocks(symbols, min_quality_score=40.0, save_csv=False)
    else:
        # 引数がない場合はサンプルテスト（デモ用）
        print("\nNo arguments provided. Running sample test with Nikkei 225 stocks...")
        print("Usage: python screener.py [stock_code1] [stock_code2] ...")
        print("Example: python screener.py 7203 6758 8035")
        print("\nTo screen all Japanese stocks, run: python main.py\n")

        df = screen_stocks(NIKKEI225_SAMPLE[:10], min_quality_score=40.0)

    print("\nFull Results DataFrame:")
    cols = ['symbol', 'name', 'is_match', 'pattern_quality_score', 'trend_template_passed']
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].to_string())
