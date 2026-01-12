"""
Cup with Handle Pattern Visualizer
===================================
検出したカップ・ウィズ・ハンドルパターンをチャート上で可視化する機能。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from matplotlib.lines import Line2D
from typing import Dict, Optional, List
from datetime import datetime


class CupHandleVisualizer:
    """
    カップ・ウィズ・ハンドルパターンの可視化クラス
    """

    def __init__(
        self,
        figsize: tuple = (14, 10),
        style: str = 'seaborn-v0_8-whitegrid',
        color_scheme: dict = None
    ):
        self.figsize = figsize
        self.style = style
        self.color_scheme = color_scheme or {
            'price': '#2E86AB',
            'cup_fill': '#A8DADC',
            'handle_fill': '#FFE66D',
            'pivot_line': '#E63946',
            'sma50': '#FF6B6B',
            'sma150': '#4ECDC4',
            'sma200': '#45B7D1',
            'volume_up': '#26A69A',
            'volume_down': '#EF5350',
            'annotation': '#1D3557',
        }

    def plot(
        self,
        df: pd.DataFrame,
        result: Dict,
        show_volume: bool = True,
        show_sma: bool = True,
        title: str = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        カップ・ウィズ・ハンドルパターンをチャートで表示

        Args:
            df: OHLCV DataFrame
            result: CupHandleDetector.detect() の戻り値
            show_volume: 出来高を表示するか
            show_sma: 移動平均線を表示するか
            title: チャートタイトル
            save_path: 保存先パス（Noneの場合は表示のみ）

        Returns:
            matplotlib Figure オブジェクト
        """
        try:
            plt.style.use(self.style)
        except OSError:
            plt.style.use('ggplot')

        # サブプロット構成
        if show_volume:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=self.figsize,
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=True
            )
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
            ax2 = None

        # 価格チャート
        self._plot_price(ax1, df, result, show_sma)

        # パターンが検出された場合はハイライト
        if result.get('is_match'):
            self._highlight_pattern(ax1, df, result)
            self._add_annotations(ax1, df, result)

        # 出来高チャート
        if show_volume and ax2 is not None:
            self._plot_volume(ax2, df, result)

        # タイトルと凡例
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        elif result.get('is_match'):
            score = result.get('pattern_quality_score', 'N/A')
            fig.suptitle(
                f'Cup with Handle Pattern Detected (Quality Score: {score})',
                fontsize=14, fontweight='bold'
            )
        else:
            fig.suptitle('Price Chart - No Pattern Detected', fontsize=14)

        ax1.legend(loc='upper left', fontsize=9)

        # X軸フォーマット
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")

        return fig

    def _plot_price(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        result: Dict,
        show_sma: bool
    ):
        """価格チャートを描画"""
        dates = df.index
        close = df['Close']
        high = df['High']
        low = df['Low']

        # 終値ライン
        ax.plot(
            dates, close,
            color=self.color_scheme['price'],
            linewidth=1.5,
            label='Close Price'
        )

        # 高値・安値の範囲を薄く表示
        ax.fill_between(
            dates, low, high,
            alpha=0.1,
            color=self.color_scheme['price']
        )

        # 移動平均線
        if show_sma:
            sma50 = close.rolling(50, min_periods=1).mean()
            sma150 = close.rolling(150, min_periods=1).mean()
            sma200 = close.rolling(200, min_periods=1).mean()

            ax.plot(dates, sma50, color=self.color_scheme['sma50'],
                    linewidth=1, linestyle='--', alpha=0.7, label='SMA 50')
            ax.plot(dates, sma150, color=self.color_scheme['sma150'],
                    linewidth=1, linestyle='--', alpha=0.7, label='SMA 150')
            ax.plot(dates, sma200, color=self.color_scheme['sma200'],
                    linewidth=1, linestyle='--', alpha=0.7, label='SMA 200')

        ax.set_ylabel('Price', fontsize=11)
        ax.grid(True, alpha=0.3)

    def _highlight_pattern(self, ax: plt.Axes, df: pd.DataFrame, result: Dict):
        """パターン部分をハイライト"""
        cup_start = pd.Timestamp(result['cup_start_date'])
        cup_end = pd.Timestamp(result['cup_end_date'])
        cup_bottom = pd.Timestamp(result['cup_bottom_date'])
        handle_start = pd.Timestamp(result['handle_start_date'])
        handle_end = pd.Timestamp(result.get('handle_end_date', df.index[-1]))

        # カップ期間のデータを取得
        cup_mask = (df.index >= cup_start) & (df.index <= cup_end)
        cup_df = df[cup_mask]

        if len(cup_df) > 0:
            # カップ部分を塗りつぶし
            ax.fill_between(
                cup_df.index,
                cup_df['Low'],
                cup_df['High'],
                alpha=0.3,
                color=self.color_scheme['cup_fill'],
                label='Cup Formation'
            )

        # ハンドル期間のデータを取得
        handle_mask = (df.index >= handle_start) & (df.index <= handle_end)
        handle_df = df[handle_mask]

        if len(handle_df) > 0:
            # ハンドル部分を塗りつぶし
            ax.fill_between(
                handle_df.index,
                handle_df['Low'],
                handle_df['High'],
                alpha=0.4,
                color=self.color_scheme['handle_fill'],
                label='Handle Formation'
            )

        # ピボット価格ライン
        if result.get('pivot_price'):
            ax.axhline(
                y=result['pivot_price'],
                color=self.color_scheme['pivot_line'],
                linestyle='-',
                linewidth=2,
                alpha=0.8,
                label=f"Pivot: {result['pivot_price']}"
            )

    def _add_annotations(self, ax: plt.Axes, df: pd.DataFrame, result: Dict):
        """パターンの主要ポイントにアノテーションを追加"""
        color = self.color_scheme['annotation']

        # カップ左端
        cup_start = pd.Timestamp(result['cup_start_date'])
        if cup_start in df.index:
            price = df.loc[cup_start, 'High']
            ax.annotate(
                'L',
                xy=(cup_start, price),
                xytext=(0, 15),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                color=color,
                ha='center'
            )

        # カップ底
        cup_bottom = pd.Timestamp(result['cup_bottom_date'])
        if cup_bottom in df.index:
            price = df.loc[cup_bottom, 'Low']
            ax.annotate(
                'B',
                xy=(cup_bottom, price),
                xytext=(0, -20),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                color=color,
                ha='center'
            )

        # カップ右端
        cup_end = pd.Timestamp(result['cup_end_date'])
        if cup_end in df.index:
            price = df.loc[cup_end, 'High']
            ax.annotate(
                'R',
                xy=(cup_end, price),
                xytext=(0, 15),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                color=color,
                ha='center'
            )

        # 情報ボックス
        info_text = (
            f"Cup Depth: {result['cup_depth']}%\n"
            f"Handle Depth: {result['handle_depth']}%\n"
            f"Quality Score: {result.get('pattern_quality_score', 'N/A')}"
        )

        # ボリューム情報
        if result.get('volume_is_valid') is not None:
            vol_status = "Valid" if result['volume_is_valid'] else "Invalid"
            breakout_ratio = result.get('breakout_volume_ratio', 'N/A')
            info_text += f"\nVolume: {vol_status} ({breakout_ratio}x)"

        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    def _plot_volume(self, ax: plt.Axes, df: pd.DataFrame, result: Dict):
        """出来高チャートを描画"""
        dates = df.index
        volume = df['Volume']
        close = df['Close']

        # 上昇/下落で色分け
        colors = np.where(
            close.pct_change() >= 0,
            self.color_scheme['volume_up'],
            self.color_scheme['volume_down']
        )

        ax.bar(dates, volume, color=colors, alpha=0.7, width=0.8)

        # 50日平均出来高
        avg_vol = volume.rolling(50, min_periods=1).mean()
        ax.plot(dates, avg_vol, color='black', linewidth=1,
                linestyle='--', alpha=0.7, label='50-day Avg')

        # パターン検出時はハンドル期間とブレイクアウトをマーク
        if result.get('is_match') and result.get('handle_start_date'):
            handle_start = pd.Timestamp(result['handle_start_date'])
            ax.axvline(x=handle_start, color='orange', linestyle=':', alpha=0.8)

        ax.set_ylabel('Volume', fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def plot_comparison(
        self,
        results: List[tuple],  # List of (df, result, title)
        save_path: str = None
    ) -> plt.Figure:
        """
        複数のパターンを比較表示

        Args:
            results: [(df, result, title), ...] のリスト
            save_path: 保存先パス
        """
        n = len(results)
        if n == 0:
            return None

        cols = min(2, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (df, result, title) in enumerate(results):
            ax = axes[i]
            dates = df.index
            close = df['Close']

            ax.plot(dates, close, color=self.color_scheme['price'], linewidth=1)

            if result.get('is_match'):
                # カップ期間をハイライト
                cup_start = pd.Timestamp(result['cup_start_date'])
                cup_end = pd.Timestamp(result['cup_end_date'])
                cup_mask = (df.index >= cup_start) & (df.index <= cup_end)
                cup_df = df[cup_mask]
                if len(cup_df) > 0:
                    ax.fill_between(cup_df.index, cup_df['Low'], cup_df['High'],
                                    alpha=0.3, color=self.color_scheme['cup_fill'])

                score = result.get('pattern_quality_score', 'N/A')
                ax.set_title(f"{title}\nScore: {score}", fontsize=10)
            else:
                ax.set_title(f"{title}\n(No Pattern)", fontsize=10)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        # 余ったサブプロットを非表示
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def plot_cup_handle(
    df: pd.DataFrame,
    result: Dict,
    **kwargs
) -> plt.Figure:
    """
    便利関数: カップ・ウィズ・ハンドルパターンを表示

    Args:
        df: OHLCV DataFrame
        result: 検出結果
        **kwargs: CupHandleVisualizer.plot() に渡す引数

    Returns:
        matplotlib Figure
    """
    visualizer = CupHandleVisualizer()
    return visualizer.plot(df, result, **kwargs)


if __name__ == '__main__':
    # テスト用
    from cup_handle_detector import CupHandleDetector, create_sample_data

    df = create_sample_data()
    detector = CupHandleDetector()
    result = detector.detect(df)

    print("Detection Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    fig = plot_cup_handle(df, result, title="Sample Cup with Handle Pattern")
    plt.show()
