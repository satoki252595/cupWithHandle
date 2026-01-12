"""
ML Data Generator
=================
Cup with Handle パターン検出のための訓練データ生成モジュール。
バックテスト結果からラベル付きデータを作成し、CNN用画像を生成する。
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from cup_handle_detector import CupHandleDetector


@dataclass
class TrainingExample:
    """訓練データの1サンプル"""
    symbol: str
    entry_date: str
    exit_date: str
    return_pct: float
    label: int  # 1: success (return > 0), 0: failure
    quality_score: float
    cup_depth: float
    handle_depth: float
    cup_duration: int
    handle_duration: int
    volume_valid: bool
    image_path: Optional[str] = None
    features: Optional[Dict] = None


class MLDataGenerator:
    """
    機械学習用データ生成クラス

    機能:
    1. バックテストデータからラベル付きサンプルを生成
    2. CNN用チャート画像を生成
    3. 特徴量ベクトルを抽出
    """

    def __init__(
        self,
        output_dir: str = None,
        image_size: Tuple[int, int] = (224, 224),
        lookback_days: int = 250,  # 画像に含める日数
    ):
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), 'data'
        )
        self.image_size = image_size
        self.lookback_days = lookback_days

        # 出力ディレクトリ作成
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images', 'success'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images', 'failure'), exist_ok=True)

    def generate_chart_image(
        self,
        df: pd.DataFrame,
        result: Dict,
        save_path: str,
        include_volume: bool = True,
        include_sma: bool = True,
    ) -> bool:
        """
        CNN用のチャート画像を生成

        Args:
            df: OHLCV DataFrame
            result: パターン検出結果
            save_path: 保存先パス
            include_volume: 出来高を含めるか
            include_sma: 移動平均線を含めるか

        Returns:
            成功したかどうか
        """
        try:
            # 画像サイズに合わせたfigure作成
            dpi = 100
            figsize = (self.image_size[0] / dpi, self.image_size[1] / dpi)

            if include_volume:
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=figsize,
                    gridspec_kw={'height_ratios': [3, 1]},
                    sharex=True
                )
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=figsize)
                ax2 = None

            # 直近N日のデータを使用
            df_plot = df.iloc[-self.lookback_days:] if len(df) > self.lookback_days else df
            dates = df_plot.index
            close = df_plot['Close'].values

            # 価格チャート（シンプルな線グラフ）
            ax1.plot(dates, close, color='black', linewidth=0.8)

            # 高値・安値の範囲
            ax1.fill_between(
                dates, df_plot['Low'].values, df_plot['High'].values,
                alpha=0.2, color='gray'
            )

            # 移動平均線
            if include_sma and len(df_plot) >= 50:
                sma50 = df_plot['Close'].rolling(50, min_periods=1).mean()
                sma150 = df_plot['Close'].rolling(150, min_periods=1).mean()
                ax1.plot(dates, sma50, color='red', linewidth=0.5, alpha=0.7)
                if len(df_plot) >= 150:
                    ax1.plot(dates, sma150, color='blue', linewidth=0.5, alpha=0.7)

            # パターン部分をハイライト（検出された場合）
            if result.get('is_match'):
                cup_start = pd.Timestamp(result['cup_start_date'])
                cup_end = pd.Timestamp(result['cup_end_date'])

                # カップ部分を薄い色で塗りつぶし
                cup_mask = (df_plot.index >= cup_start) & (df_plot.index <= cup_end)
                if cup_mask.any():
                    cup_df = df_plot[cup_mask]
                    ax1.fill_between(
                        cup_df.index,
                        cup_df['Low'].values,
                        cup_df['High'].values,
                        alpha=0.3, color='cyan'
                    )

                # ピボットライン
                if result.get('pivot_price'):
                    ax1.axhline(y=result['pivot_price'], color='red',
                               linewidth=0.5, linestyle='--', alpha=0.7)

            # 軸ラベル・タイトルを削除（CNNには不要）
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)

            # 出来高チャート
            if include_volume and ax2 is not None:
                volume = df_plot['Volume'].values
                colors = np.where(
                    df_plot['Close'].pct_change() >= 0, 'green', 'red'
                )
                ax2.bar(range(len(dates)), volume, color=colors, alpha=0.6, width=0.8)
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['bottom'].set_visible(False)
                ax2.spines['left'].set_visible(False)

            plt.tight_layout(pad=0)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       pad_inches=0, facecolor='white')
            plt.close(fig)

            return True

        except Exception as e:
            print(f"Error generating image: {e}")
            plt.close('all')
            return False

    def create_training_example(
        self,
        symbol: str,
        df: pd.DataFrame,
        entry_date: datetime,
        exit_date: datetime,
        return_pct: float,
        result: Dict,
        generate_image: bool = True,
    ) -> Optional[TrainingExample]:
        """
        1つの訓練サンプルを作成

        Args:
            symbol: 銘柄コード
            df: OHLCV DataFrame
            entry_date: エントリー日
            exit_date: 決済日
            return_pct: リターン（%）
            result: パターン検出結果
            generate_image: 画像を生成するか

        Returns:
            TrainingExample または None
        """
        try:
            # ラベル: リターンが正なら1（成功）、そうでなければ0（失敗）
            label = 1 if return_pct > 0 else 0
            label_str = 'success' if label == 1 else 'failure'

            # カップ・ハンドル期間を計算
            cup_start = pd.Timestamp(result.get('cup_start_date'))
            cup_end = pd.Timestamp(result.get('cup_end_date'))
            handle_start = pd.Timestamp(result.get('handle_start_date'))
            handle_end = pd.Timestamp(result.get('handle_end_date'))

            cup_duration = (cup_end - cup_start).days if cup_start and cup_end else 0
            handle_duration = (handle_end - handle_start).days if handle_start and handle_end else 0

            # 画像生成
            image_path = None
            if generate_image:
                # エントリー日時点までのデータを使用
                entry_ts = pd.Timestamp(entry_date)
                df_before_entry = df[df.index <= entry_ts]

                if len(df_before_entry) >= 50:  # 最低限のデータが必要
                    image_filename = f"{symbol}_{entry_date.strftime('%Y%m%d')}.png"
                    image_path = os.path.join(
                        self.output_dir, 'images', label_str, image_filename
                    )

                    self.generate_chart_image(df_before_entry, result, image_path)

            return TrainingExample(
                symbol=symbol,
                entry_date=entry_date.strftime('%Y-%m-%d'),
                exit_date=exit_date.strftime('%Y-%m-%d'),
                return_pct=return_pct,
                label=label,
                quality_score=result.get('pattern_quality_score', 0),
                cup_depth=result.get('cup_depth', 0),
                handle_depth=result.get('handle_depth', 0),
                cup_duration=cup_duration,
                handle_duration=handle_duration,
                volume_valid=result.get('volume_is_valid', False),
                image_path=image_path,
            )

        except Exception as e:
            print(f"Error creating training example: {e}")
            return None

    def generate_from_backtest(
        self,
        backtest_trades: List[Dict],
        fetch_data_func,
        generate_images: bool = True,
    ) -> List[TrainingExample]:
        """
        バックテスト結果から訓練データを生成

        Args:
            backtest_trades: バックテストのトレードリスト
            fetch_data_func: データ取得関数 func(symbol) -> DataFrame
            generate_images: 画像を生成するか

        Returns:
            TrainingExampleのリスト
        """
        examples = []

        for i, trade in enumerate(backtest_trades):
            print(f"\rGenerating training data: {i+1}/{len(backtest_trades)}", end='')

            symbol = trade.get('symbol')
            entry_date = trade.get('entry_date')
            exit_date = trade.get('exit_date')
            return_pct = trade.get('return_pct', 0)

            if not all([symbol, entry_date, exit_date]):
                continue

            # データ取得
            try:
                df = fetch_data_func(symbol)
                if df is None or len(df) < 100:
                    continue
            except Exception as e:
                print(f"\nError fetching data for {symbol}: {e}")
                continue

            # パターン検出（エントリー日時点で）
            entry_ts = pd.Timestamp(entry_date)
            df_at_entry = df[df.index <= entry_ts]

            detector = CupHandleDetector()
            result = detector.detect(df_at_entry)

            # 訓練サンプル作成
            example = self.create_training_example(
                symbol=symbol,
                df=df,
                entry_date=entry_date if isinstance(entry_date, datetime) else datetime.strptime(entry_date, '%Y-%m-%d'),
                exit_date=exit_date if isinstance(exit_date, datetime) else datetime.strptime(exit_date, '%Y-%m-%d'),
                return_pct=return_pct,
                result=result,
                generate_image=generate_images,
            )

            if example:
                examples.append(example)

        print(f"\nGenerated {len(examples)} training examples")
        return examples

    def save_dataset(
        self,
        examples: List[TrainingExample],
        filename: str = 'dataset.json'
    ):
        """データセットをJSONファイルに保存"""
        filepath = os.path.join(self.output_dir, filename)

        data = [asdict(ex) for ex in examples]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Dataset saved to: {filepath}")
        print(f"Total examples: {len(examples)}")
        print(f"Success: {sum(1 for ex in examples if ex.label == 1)}")
        print(f"Failure: {sum(1 for ex in examples if ex.label == 0)}")

    def load_dataset(self, filename: str = 'dataset.json') -> List[TrainingExample]:
        """データセットをJSONファイルから読み込み"""
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        examples = [TrainingExample(**item) for item in data]
        return examples

    def get_train_val_test_split(
        self,
        examples: List[TrainingExample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        sort_by_date: bool = True,
    ) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
        """
        時系列を考慮したデータ分割

        Args:
            examples: 全サンプル
            train_ratio: 訓練データの割合
            val_ratio: 検証データの割合
            sort_by_date: 日付順にソートするか

        Returns:
            (train, val, test) のタプル
        """
        if sort_by_date:
            examples = sorted(examples, key=lambda x: x.entry_date)

        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = examples[:train_end]
        val = examples[train_end:val_end]
        test = examples[val_end:]

        return train, val, test


class SyntheticDataGenerator:
    """
    合成データ生成クラス

    データ不足時に訓練データを増強するため、
    既存パターンにノイズを加えた合成データを生成する。
    """

    def __init__(self, noise_level: float = 0.02):
        self.noise_level = noise_level

    def augment_price_data(
        self,
        df: pd.DataFrame,
        n_augmentations: int = 3
    ) -> List[pd.DataFrame]:
        """
        価格データにノイズを加えて増強

        Args:
            df: 元のOHLCV DataFrame
            n_augmentations: 生成する増強データ数

        Returns:
            増強されたDataFrameのリスト
        """
        augmented = []

        for i in range(n_augmentations):
            df_aug = df.copy()

            # 価格にランダムノイズを追加
            noise = np.random.normal(1, self.noise_level, len(df))
            for col in ['Open', 'High', 'Low', 'Close']:
                df_aug[col] = df_aug[col] * noise

            # High/Lowの整合性を維持
            df_aug['High'] = df_aug[['Open', 'High', 'Low', 'Close']].max(axis=1)
            df_aug['Low'] = df_aug[['Open', 'High', 'Low', 'Close']].min(axis=1)

            # 出来高にもノイズ追加
            vol_noise = np.random.normal(1, self.noise_level * 2, len(df))
            df_aug['Volume'] = (df_aug['Volume'] * vol_noise).astype(int)

            augmented.append(df_aug)

        return augmented

    def generate_synthetic_cup_handle(
        self,
        base_price: float = 1000,
        n_days: int = 300,
        cup_depth_pct: float = 20,
        handle_depth_pct: float = 10,
        cup_duration: int = 100,
        handle_duration: int = 20,
    ) -> pd.DataFrame:
        """
        カップ・ウィズ・ハンドルパターンを持つ合成データを生成

        Args:
            base_price: 基準価格
            n_days: 生成する日数
            cup_depth_pct: カップの深さ（%）
            handle_depth_pct: ハンドルの深さ（%）
            cup_duration: カップの期間（日数）
            handle_duration: ハンドルの期間（日数）

        Returns:
            合成データのDataFrame
        """
        np.random.seed(None)  # ランダム性を確保

        # 期間の計算
        pre_pattern = n_days - cup_duration - handle_duration - 50  # 前段階
        post_pattern = 50  # パターン後

        prices = []

        # 前段階: 緩やかな上昇
        for i in range(pre_pattern):
            trend = base_price * (1 + i * 0.001)
            noise = np.random.normal(0, base_price * 0.01)
            prices.append(trend + noise)

        # カップ左端の価格
        cup_left_price = prices[-1] if prices else base_price
        cup_bottom_price = cup_left_price * (1 - cup_depth_pct / 100)

        # カップの形成（U字型）
        cup_half = cup_duration // 2

        # 左側（下落）
        for i in range(cup_half):
            progress = i / cup_half
            price = cup_left_price - (cup_left_price - cup_bottom_price) * progress
            noise = np.random.normal(0, base_price * 0.01)
            prices.append(price + noise)

        # 右側（回復）
        for i in range(cup_duration - cup_half):
            progress = i / (cup_duration - cup_half)
            price = cup_bottom_price + (cup_left_price - cup_bottom_price) * progress
            noise = np.random.normal(0, base_price * 0.01)
            prices.append(price + noise)

        # ハンドルの形成
        handle_start_price = prices[-1]
        handle_bottom_price = handle_start_price * (1 - handle_depth_pct / 100)
        handle_half = handle_duration // 2

        # ハンドル下落
        for i in range(handle_half):
            progress = i / handle_half
            price = handle_start_price - (handle_start_price - handle_bottom_price) * progress
            noise = np.random.normal(0, base_price * 0.005)
            prices.append(price + noise)

        # ハンドル回復
        for i in range(handle_duration - handle_half):
            progress = i / (handle_duration - handle_half)
            price = handle_bottom_price + (handle_start_price - handle_bottom_price) * progress
            noise = np.random.normal(0, base_price * 0.005)
            prices.append(price + noise)

        # パターン後（上昇）
        breakout_price = prices[-1]
        for i in range(post_pattern):
            trend = breakout_price * (1 + i * 0.003)
            noise = np.random.normal(0, base_price * 0.01)
            prices.append(trend + noise)

        # DataFrame作成
        prices = np.array(prices)
        dates = pd.date_range(end=datetime.now(), periods=len(prices), freq='B')

        # OHLCV生成
        df = pd.DataFrame({
            'Open': prices * np.random.uniform(0.995, 1.005, len(prices)),
            'High': prices * np.random.uniform(1.005, 1.02, len(prices)),
            'Low': prices * np.random.uniform(0.98, 0.995, len(prices)),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, len(prices)),
        }, index=dates)

        # High/Lowの整合性
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

        return df


if __name__ == '__main__':
    # テスト: 合成データ生成とパターン検出
    print("Testing MLDataGenerator...")

    synth_gen = SyntheticDataGenerator()
    df = synth_gen.generate_synthetic_cup_handle(
        base_price=1000,
        cup_depth_pct=20,
        handle_depth_pct=10,
    )

    print(f"Generated {len(df)} days of synthetic data")
    print(f"Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")

    # パターン検出テスト
    detector = CupHandleDetector()
    result = detector.detect(df)

    print(f"Pattern detected: {result.get('is_match')}")
    if result.get('is_match'):
        print(f"  Cup depth: {result.get('cup_depth')}%")
        print(f"  Handle depth: {result.get('handle_depth')}%")
        print(f"  Quality score: {result.get('pattern_quality_score')}")

    # 画像生成テスト
    data_gen = MLDataGenerator()
    test_image_path = os.path.join(data_gen.output_dir, 'test_image.png')
    success = data_gen.generate_chart_image(df, result, test_image_path)
    print(f"Image generation: {'Success' if success else 'Failed'}")
    if success:
        print(f"  Saved to: {test_image_path}")
