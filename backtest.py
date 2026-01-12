#!/usr/bin/env python3
"""
Cup with Handle Pattern Backtester
==================================
パターン検出後のリターンを検証し、最適パラメータを探索
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import time
import os
import sys
import json
import warnings
import itertools

try:
    import yfinance as yf
except ImportError:
    print("yfinance is required. Install with: pip install yfinance")
    sys.exit(1)

from cup_handle_detector import CupHandleDetector
from screener import JapanStockListFetcher


@dataclass
class BacktestTrade:
    """個別トレード結果"""
    symbol: str
    company_name: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    holding_days: int
    return_pct: float
    pattern_score: float
    cup_depth: float
    handle_depth: float


@dataclass
class BacktestResult:
    """バックテスト結果"""
    params: Dict
    trades: List[BacktestTrade]
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    max_return: float = 0.0
    min_return: float = 0.0
    std_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0

    def calculate_metrics(self):
        """メトリクスを計算"""
        if not self.trades:
            return

        returns = [t.return_pct for t in self.trades]
        self.total_trades = len(returns)
        self.winning_trades = sum(1 for r in returns if r > 0)
        self.losing_trades = sum(1 for r in returns if r <= 0)
        self.win_rate = self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0

        self.avg_return = np.mean(returns)
        self.max_return = np.max(returns)
        self.min_return = np.min(returns)
        self.std_return = np.std(returns)

        # シャープレシオ（年率換算、リスクフリーレート0%）
        if self.std_return > 0:
            self.sharpe_ratio = (self.avg_return / self.std_return) * np.sqrt(252 / 20)  # 20日保有想定
        else:
            self.sharpe_ratio = 0

        # プロフィットファクター
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # 最大ドローダウン（累積リターンベース）
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        self.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0


class CupHandleBacktester:
    """
    カップ・ウィズ・ハンドルパターンのバックテスター
    """

    def __init__(
        self,
        holding_period: int = 20,
        stop_loss_pct: float = -8.0,
        take_profit_pct: float = 20.0,
        lookback_days: int = 500,
    ):
        """
        Args:
            holding_period: 保有期間（営業日）
            stop_loss_pct: ストップロス（%）
            take_profit_pct: テイクプロフィット（%）
            lookback_days: パターン検出用のデータ期間
        """
        self.holding_period = holding_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.lookback_days = lookback_days

    def fetch_stock_data(
        self,
        symbol: str,
        period: str = '3y'
    ) -> Tuple[pd.DataFrame, str]:
        """株価データを取得"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty:
                return pd.DataFrame(), None

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            try:
                info = ticker.info
                name = info.get('longName') or info.get('shortName') or symbol
            except:
                name = symbol

            return df, name

        except Exception as e:
            return pd.DataFrame(), None

    def run_single_backtest(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_name: str,
        detector: CupHandleDetector,
        min_quality_score: float = 50.0
    ) -> List[BacktestTrade]:
        """
        単一銘柄のバックテスト

        過去データをスライディングウィンドウで分析し、
        パターン検出時点でのフォワードリターンを計算
        """
        trades = []

        if len(df) < self.lookback_days + self.holding_period + 50:
            return trades

        # スライディングウィンドウでパターンを検出
        for end_idx in range(self.lookback_days, len(df) - self.holding_period, 20):
            # 分析用データ（パターン検出時点まで）
            analysis_df = df.iloc[end_idx - self.lookback_days:end_idx].copy()

            try:
                result = detector.detect(analysis_df)

                if not result.get('is_match'):
                    continue

                score = result.get('pattern_quality_score', 0)
                if score < min_quality_score:
                    continue

                # エントリー日とピボット価格
                entry_date = df.index[end_idx]
                pivot_price = result.get('pivot_price')

                if pivot_price is None:
                    continue

                # 実際のエントリー価格（翌日の始値）
                if end_idx + 1 >= len(df):
                    continue

                entry_price = df.iloc[end_idx + 1]['Open']

                # ピボット価格を超えていない場合はスキップ
                if entry_price < pivot_price * 0.98:
                    continue

                # 保有期間中の価格推移を確認
                exit_idx = min(end_idx + 1 + self.holding_period, len(df) - 1)
                holding_df = df.iloc[end_idx + 1:exit_idx + 1]

                if len(holding_df) == 0:
                    continue

                # ストップロス・テイクプロフィットの確認
                exit_price = None
                exit_date = None
                actual_holding_days = 0

                for i, (date, row) in enumerate(holding_df.iterrows()):
                    low = row['Low']
                    high = row['High']
                    close = row['Close']

                    # ストップロスチェック
                    if (low - entry_price) / entry_price * 100 <= self.stop_loss_pct:
                        exit_price = entry_price * (1 + self.stop_loss_pct / 100)
                        exit_date = date
                        actual_holding_days = i + 1
                        break

                    # テイクプロフィットチェック
                    if (high - entry_price) / entry_price * 100 >= self.take_profit_pct:
                        exit_price = entry_price * (1 + self.take_profit_pct / 100)
                        exit_date = date
                        actual_holding_days = i + 1
                        break

                # ストップ/テイクなしの場合は保有期間終了時の終値
                if exit_price is None:
                    exit_price = holding_df.iloc[-1]['Close']
                    exit_date = holding_df.index[-1]
                    actual_holding_days = len(holding_df)

                return_pct = (exit_price - entry_price) / entry_price * 100

                trades.append(BacktestTrade(
                    symbol=symbol,
                    company_name=company_name,
                    entry_date=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_date=exit_date,
                    exit_price=round(exit_price, 2),
                    holding_days=actual_holding_days,
                    return_pct=round(return_pct, 2),
                    pattern_score=score,
                    cup_depth=result.get('cup_depth', 0),
                    handle_depth=result.get('handle_depth', 0),
                ))

            except Exception as e:
                continue

        return trades

    def run_backtest(
        self,
        symbols: List[str],
        params: Dict,
        min_quality_score: float = 50.0,
        show_progress: bool = True
    ) -> BacktestResult:
        """
        複数銘柄でバックテスト実行

        Args:
            symbols: 銘柄リスト
            params: CupHandleDetectorのパラメータ
            min_quality_score: 最低品質スコア
            show_progress: 進捗表示

        Returns:
            BacktestResult
        """
        detector = CupHandleDetector(**params)
        all_trades = []
        errors = 0

        for i, symbol in enumerate(symbols):
            try:
                df, name = self.fetch_stock_data(symbol, period='3y')

                if df.empty:
                    errors += 1
                    continue

                trades = self.run_single_backtest(
                    symbol, df, name, detector, min_quality_score
                )
                all_trades.extend(trades)

                if show_progress and (i + 1) % 10 == 0:
                    print(f"  Backtested {i + 1}/{len(symbols)} | Trades: {len(all_trades)}")

            except Exception as e:
                errors += 1
                continue

            time.sleep(0.3)  # レート制限対策

        result = BacktestResult(params=params, trades=all_trades)
        result.calculate_metrics()

        return result


def select_diversified_stocks(n: int = 100) -> List[str]:
    """
    セクター・時価総額を分散させた銘柄を選定

    Args:
        n: 選定銘柄数

    Returns:
        銘柄リスト（.T形式）
    """
    fetcher = JapanStockListFetcher()
    df = fetcher.fetch()

    if df.empty:
        print("Failed to fetch stock list")
        return []

    # セクター別に銘柄を抽出
    sectors = df['sector'].unique()
    selected = []

    # 各セクターから均等に選択
    stocks_per_sector = max(1, n // len(sectors))

    for sector in sectors:
        sector_stocks = df[df['sector'] == sector]['stock_code'].tolist()

        # 時価総額の分散のため、コードの範囲を分けて選択
        if len(sector_stocks) >= stocks_per_sector:
            # 前半、中盤、後半から選択
            step = len(sector_stocks) // stocks_per_sector
            for i in range(stocks_per_sector):
                idx = min(i * step, len(sector_stocks) - 1)
                selected.append(sector_stocks[idx])
        else:
            selected.extend(sector_stocks)

        if len(selected) >= n:
            break

    # 足りない場合は追加
    if len(selected) < n:
        remaining = df[~df['stock_code'].isin(selected)]['stock_code'].tolist()
        selected.extend(remaining[:n - len(selected)])

    # .Tサフィックスを追加
    return [f"{code}.T" for code in selected[:n]]


def run_parameter_optimization(
    symbols: List[str],
    param_grid: Dict,
    n_iterations: int = 10,
    show_progress: bool = True
) -> List[BacktestResult]:
    """
    パラメータ最適化を実行

    Args:
        symbols: 銘柄リスト
        param_grid: パラメータグリッド
        n_iterations: 上位何パターンを詳細検証するか
        show_progress: 進捗表示

    Returns:
        BacktestResultのリスト（スコア順）
    """
    backtester = CupHandleBacktester(
        holding_period=20,
        stop_loss_pct=-8.0,
        take_profit_pct=20.0,
    )

    # パラメータの組み合わせを生成
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    print(f"Testing {len(combinations)} parameter combinations...")

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(param_keys, combo))

        if show_progress:
            print(f"\nCombination {i + 1}/{len(combinations)}: {params}")

        result = backtester.run_backtest(
            symbols,
            params,
            min_quality_score=50.0,
            show_progress=False
        )

        results.append(result)

        if show_progress:
            print(f"  Trades: {result.total_trades} | "
                  f"Win Rate: {result.win_rate:.1f}% | "
                  f"Avg Return: {result.avg_return:.2f}% | "
                  f"Sharpe: {result.sharpe_ratio:.2f}")

    # スコアでソート（シャープレシオ × 勝率）
    results.sort(key=lambda x: x.sharpe_ratio * x.win_rate / 100, reverse=True)

    return results


def generate_report(
    results: List[BacktestResult],
    symbols: List[str],
    output_path: str
):
    """
    バックテストレポートをMarkdown形式で出力

    Args:
        results: BacktestResultのリスト
        symbols: テスト銘柄リスト
        output_path: 出力ファイルパス
    """
    best = results[0] if results else None

    report = f"""# Cup with Handle パターン バックテストレポート

**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. テスト概要

| 項目 | 値 |
|------|------|
| テスト銘柄数 | {len(symbols)} |
| データ期間 | 過去3年間 |
| 保有期間 | 20営業日 |
| ストップロス | -8% |
| テイクプロフィット | +20% |
| テストしたパラメータ組み合わせ | {len(results)} |

---

## 2. 最適パラメータ

"""

    if best:
        report += """### 推奨パラメータ設定

```python
detector = CupHandleDetector(
"""
        for key, value in best.params.items():
            report += f"    {key}={value},\n"
        report += """)
```

### パフォーマンス指標

| 指標 | 値 | 説明 |
|------|------|------|
| 総トレード数 | {total_trades} | パターン検出→エントリー回数 |
| 勝率 | {win_rate:.1f}% | プラスリターンの割合 |
| 平均リターン | {avg_return:+.2f}% | 全トレードの平均 |
| 最大リターン | {max_return:+.2f}% | 最も良かったトレード |
| 最小リターン | {min_return:+.2f}% | 最も悪かったトレード |
| 標準偏差 | {std_return:.2f}% | リターンのばらつき |
| シャープレシオ | {sharpe:.2f} | リスク調整後リターン |
| プロフィットファクター | {pf:.2f} | 総利益÷総損失 |
| 最大ドローダウン | {mdd:.2f}% | 累積最大下落 |

""".format(
            total_trades=best.total_trades,
            win_rate=best.win_rate,
            avg_return=best.avg_return,
            max_return=best.max_return,
            min_return=best.min_return,
            std_return=best.std_return,
            sharpe=best.sharpe_ratio,
            pf=best.profit_factor if best.profit_factor != float('inf') else 999,
            mdd=best.max_drawdown,
        )

    report += """---

## 3. パラメータ比較（上位10件）

"""

    for i, r in enumerate(results[:10]):
        report += f"""### {i+1}位
- **パラメータ**: `cup_depth={r.params.get('cup_min_depth_pct', 12)}-{r.params.get('cup_max_depth_pct', 33)}%`, `handle_max={r.params.get('handle_max_depth_pct', 15)}%`, `peak_order={r.params.get('peak_order', 7)}`
- **トレード数**: {r.total_trades} | **勝率**: {r.win_rate:.1f}% | **平均リターン**: {r.avg_return:+.2f}% | **シャープ**: {r.sharpe_ratio:.2f}

"""

    if best and best.trades:
        report += """
---

## 4. 個別トレード詳細（上位10件）

| 銘柄 | 企業名 | エントリー日 | 保有日数 | リターン | スコア |
|------|--------|-------------|---------|---------|--------|
"""
        # リターン順でソート
        sorted_trades = sorted(best.trades, key=lambda x: x.return_pct, reverse=True)

        for t in sorted_trades[:10]:
            report += f"| {t.symbol} | {t.company_name[:15]} | {t.entry_date.strftime('%Y-%m-%d')} | {t.holding_days}日 | {t.return_pct:+.1f}% | {t.pattern_score:.0f} |\n"

        report += """
### 損失トレード（下位5件）

| 銘柄 | 企業名 | エントリー日 | 保有日数 | リターン | スコア |
|------|--------|-------------|---------|---------|--------|
"""
        for t in sorted_trades[-5:]:
            report += f"| {t.symbol} | {t.company_name[:15]} | {t.entry_date.strftime('%Y-%m-%d')} | {t.holding_days}日 | {t.return_pct:+.1f}% | {t.pattern_score:.0f} |\n"

    report += """
---

## 5. 結論と推奨事項

### パラメータ設定のポイント

"""

    if best:
        # パラメータの傾向を分析
        report += f"""1. **カップ深さ**: {best.params.get('cup_min_depth_pct', 12)}% 〜 {best.params.get('cup_max_depth_pct', 33)}%
   - 深すぎるカップは回復に時間がかかり、パフォーマンスが低下する傾向

2. **ハンドル深さ**: 最大{best.params.get('handle_max_depth_pct', 15)}%
   - 浅いハンドルほど強気のサイン

3. **カップ期間**: {best.params.get('cup_min_days', 35)}日 〜 {best.params.get('cup_max_days', 325)}日
   - 短すぎると信頼性低下、長すぎるとモメンタム減衰

4. **ピーク検出感度**: order={best.params.get('peak_order', 7)}
   - ノイズを拾いすぎず、重要なピークを検出

"""

    report += """### 注意事項

- **過去のパフォーマンスは将来を保証しません**
- バックテストはあくまで参考値であり、実際の取引では手数料・スリッページが発生します
- 市場環境の変化によりパターンの有効性は変動します
- 必ず他のファンダメンタル分析と組み合わせてご利用ください

---

*このレポートは自動生成されました*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")


def save_trades_for_ml(trades: List[BacktestTrade], output_path: str):
    """
    MLトレーニング用にトレード結果を保存

    Args:
        trades: BacktestTradeのリスト
        output_path: 保存先パス
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = []
    for trade in trades:
        data.append({
            'symbol': trade.symbol,
            'company_name': trade.company_name,
            'entry_date': trade.entry_date.strftime('%Y-%m-%d'),
            'exit_date': trade.exit_date.strftime('%Y-%m-%d'),
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'return_pct': trade.return_pct,
            'pattern_score': trade.pattern_score,
            'cup_depth': trade.cup_depth,
            'handle_depth': trade.handle_depth,
            'holding_days': trade.holding_days,
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(trades)} trades for ML training to: {output_path}")


if __name__ == '__main__':
    print("=" * 70)
    print("Cup with Handle Pattern Backtest")
    print("=" * 70)

    # 100銘柄を選定
    print("\nSelecting 100 diversified stocks...")
    symbols = select_diversified_stocks(100)
    print(f"Selected {len(symbols)} stocks")

    # パラメータグリッド
    param_grid = {
        'cup_min_depth_pct': [10.0, 12.0, 15.0],
        'cup_max_depth_pct': [30.0, 33.0, 35.0],
        'handle_max_depth_pct': [12.0, 15.0, 18.0],
        'peak_order': [5, 7, 10],
    }

    # パラメータ最適化
    print("\nRunning parameter optimization...")
    results = run_parameter_optimization(symbols, param_grid)

    # レポート生成
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'backtest_report_{timestamp}.md')

    generate_report(results, symbols, report_path)

    print("\nBacktest complete!")
