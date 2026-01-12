"""
Feature Extractor
=================
Cup with Handle パターンの特徴量抽出モジュール。
価格データとパターン検出結果から機械学習用の特徴量ベクトルを生成する。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class FeatureSet:
    """特徴量セット"""
    names: List[str]
    values: np.ndarray

    def to_dict(self) -> Dict[str, float]:
        return dict(zip(self.names, self.values))


class FeatureExtractor:
    """
    特徴量抽出クラス

    抽出する特徴量カテゴリ:
    1. パターン形状特徴量（カップ・ハンドルの幾何学的特性）
    2. 出来高特徴量
    3. テクニカル指標（RSI, MACD, ボリンジャーバンドなど）
    4. トレンド特徴量
    5. ボラティリティ特徴量
    """

    def __init__(self):
        self.feature_names = []

    def extract_all(
        self,
        df: pd.DataFrame,
        result: Dict,
    ) -> FeatureSet:
        """
        全ての特徴量を抽出

        Args:
            df: OHLCV DataFrame
            result: パターン検出結果

        Returns:
            FeatureSet オブジェクト
        """
        features = {}

        # 1. パターン形状特徴量
        pattern_features = self.extract_pattern_features(result)
        features.update(pattern_features)

        # 2. 出来高特徴量
        volume_features = self.extract_volume_features(df, result)
        features.update(volume_features)

        # 3. テクニカル指標
        technical_features = self.extract_technical_features(df)
        features.update(technical_features)

        # 4. トレンド特徴量
        trend_features = self.extract_trend_features(df)
        features.update(trend_features)

        # 5. ボラティリティ特徴量
        volatility_features = self.extract_volatility_features(df)
        features.update(volatility_features)

        # 6. 価格位置特徴量
        price_position_features = self.extract_price_position_features(df)
        features.update(price_position_features)

        names = list(features.keys())
        values = np.array([features[name] for name in names])

        return FeatureSet(names=names, values=values)

    def extract_pattern_features(self, result: Dict) -> Dict[str, float]:
        """パターン形状特徴量を抽出"""
        features = {}

        # カップの特徴
        features['cup_depth'] = result.get('cup_depth', 0) or 0
        features['handle_depth'] = result.get('handle_depth', 0) or 0
        features['quality_score'] = result.get('pattern_quality_score', 0) or 0

        # カップ深さとハンドル深さの比率
        if features['cup_depth'] > 0:
            features['handle_to_cup_ratio'] = features['handle_depth'] / features['cup_depth']
        else:
            features['handle_to_cup_ratio'] = 0

        # 期間の計算
        cup_start = result.get('cup_start_date')
        cup_end = result.get('cup_end_date')
        handle_start = result.get('handle_start_date')
        handle_end = result.get('handle_end_date')

        if cup_start and cup_end:
            cup_start = pd.Timestamp(cup_start)
            cup_end = pd.Timestamp(cup_end)
            features['cup_duration'] = (cup_end - cup_start).days
        else:
            features['cup_duration'] = 0

        if handle_start and handle_end:
            handle_start = pd.Timestamp(handle_start)
            handle_end = pd.Timestamp(handle_end)
            features['handle_duration'] = (handle_end - handle_start).days
        else:
            features['handle_duration'] = 0

        # 期間の比率
        if features['cup_duration'] > 0:
            features['handle_to_cup_duration_ratio'] = features['handle_duration'] / features['cup_duration']
        else:
            features['handle_to_cup_duration_ratio'] = 0

        # ピボット価格からの距離（%）
        # これは現在価格との比較で計算する必要があるため、後で補完

        return features

    def extract_volume_features(self, df: pd.DataFrame, result: Dict) -> Dict[str, float]:
        """出来高特徴量を抽出"""
        features = {}

        if 'Volume' not in df.columns:
            return {
                'volume_valid': 0,
                'volume_ratio_20d': 1.0,
                'volume_ratio_50d': 1.0,
                'volume_trend_20d': 0,
                'breakout_volume_ratio': 1.0,
            }

        volume = df['Volume'].values

        # 出来高の移動平均
        vol_20d = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        vol_50d = np.mean(volume[-50:]) if len(volume) >= 50 else np.mean(volume)

        # 直近出来高と平均の比率
        recent_vol = np.mean(volume[-5:]) if len(volume) >= 5 else volume[-1]
        features['volume_ratio_20d'] = recent_vol / vol_20d if vol_20d > 0 else 1.0
        features['volume_ratio_50d'] = recent_vol / vol_50d if vol_50d > 0 else 1.0

        # 出来高トレンド（20日間の変化率）
        if len(volume) >= 20:
            first_half = np.mean(volume[-20:-10])
            second_half = np.mean(volume[-10:])
            features['volume_trend_20d'] = (second_half - first_half) / first_half if first_half > 0 else 0
        else:
            features['volume_trend_20d'] = 0

        # パターン検出結果からの出来高情報
        features['volume_valid'] = 1 if result.get('volume_is_valid') else 0
        features['breakout_volume_ratio'] = result.get('breakout_volume_ratio', 1.0) or 1.0

        # 出来高の標準偏差（正規化）
        features['volume_std_ratio'] = np.std(volume[-20:]) / vol_20d if vol_20d > 0 else 0

        return features

    def extract_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """テクニカル指標を抽出"""
        features = {}
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values

        # RSI (14日)
        features['rsi_14'] = self._calculate_rsi(close, 14)

        # RSI (7日) - より短期
        features['rsi_7'] = self._calculate_rsi(close, 7)

        # MACD
        macd, signal, hist = self._calculate_macd(close)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = hist

        # ボリンジャーバンド位置
        bb_position = self._calculate_bb_position(close)
        features['bb_position'] = bb_position

        # ストキャスティクス
        k, d = self._calculate_stochastic(high, low, close)
        features['stochastic_k'] = k
        features['stochastic_d'] = d

        # ATR (Average True Range)
        features['atr_ratio'] = self._calculate_atr_ratio(high, low, close)

        # ADX (Average Directional Index) - 簡易版
        features['adx'] = self._calculate_adx(high, low, close)

        return features

    def extract_trend_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """トレンド特徴量を抽出"""
        features = {}
        close = df['Close'].values

        # 移動平均との位置関係
        sma_20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
        sma_200 = np.mean(close[-200:]) if len(close) >= 200 else close[-1]

        current_price = close[-1]

        features['price_to_sma20'] = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
        features['price_to_sma50'] = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0
        features['price_to_sma200'] = (current_price - sma_200) / sma_200 if sma_200 > 0 else 0

        # 移動平均の傾き
        if len(close) >= 25:
            sma_20_prev = np.mean(close[-25:-5])
            features['sma20_slope'] = (sma_20 - sma_20_prev) / sma_20_prev if sma_20_prev > 0 else 0
        else:
            features['sma20_slope'] = 0

        if len(close) >= 55:
            sma_50_prev = np.mean(close[-55:-5])
            features['sma50_slope'] = (sma_50 - sma_50_prev) / sma_50_prev if sma_50_prev > 0 else 0
        else:
            features['sma50_slope'] = 0

        # リターン（過去N日）
        for days in [5, 10, 20, 60]:
            if len(close) > days:
                features[f'return_{days}d'] = (close[-1] - close[-days]) / close[-days]
            else:
                features[f'return_{days}d'] = 0

        # 移動平均の順序（1: 正しい順序, 0: そうでない）
        features['ma_order_correct'] = 1 if (sma_20 > sma_50 > sma_200) else 0

        return features

    def extract_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """ボラティリティ特徴量を抽出"""
        features = {}
        close = df['Close'].values

        # 日次リターンの標準偏差（年率換算）
        if len(close) >= 20:
            returns = np.diff(close[-20:]) / close[-21:-1]
            features['volatility_20d'] = np.std(returns) * np.sqrt(252)
        else:
            features['volatility_20d'] = 0

        if len(close) >= 60:
            returns = np.diff(close[-60:]) / close[-61:-1]
            features['volatility_60d'] = np.std(returns) * np.sqrt(252)
        else:
            features['volatility_60d'] = 0

        # ボラティリティの変化
        if len(close) >= 40:
            returns_first = np.diff(close[-40:-20]) / close[-41:-21]
            returns_second = np.diff(close[-20:]) / close[-21:-1]
            vol_first = np.std(returns_first)
            vol_second = np.std(returns_second)
            features['volatility_change'] = (vol_second - vol_first) / vol_first if vol_first > 0 else 0
        else:
            features['volatility_change'] = 0

        # 高値・安値レンジ（正規化）
        if len(df) >= 20:
            recent_range = df['High'].iloc[-20:].max() - df['Low'].iloc[-20:].min()
            avg_price = np.mean(close[-20:])
            features['price_range_20d'] = recent_range / avg_price if avg_price > 0 else 0
        else:
            features['price_range_20d'] = 0

        return features

    def extract_price_position_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """価格位置特徴量を抽出"""
        features = {}
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values

        current_price = close[-1]

        # 52週高値・安値からの位置
        lookback = min(252, len(df))
        high_52w = high[-lookback:].max()
        low_52w = low[-lookback:].min()

        features['pct_from_52w_high'] = (high_52w - current_price) / high_52w if high_52w > 0 else 0
        features['pct_from_52w_low'] = (current_price - low_52w) / low_52w if low_52w > 0 else 0

        # 52週レンジ内の位置（0-1）
        range_52w = high_52w - low_52w
        features['position_in_52w_range'] = (current_price - low_52w) / range_52w if range_52w > 0 else 0.5

        # 直近20日の高値・安値からの位置
        if len(df) >= 20:
            high_20d = high[-20:].max()
            low_20d = low[-20:].min()
            range_20d = high_20d - low_20d
            features['position_in_20d_range'] = (current_price - low_20d) / range_20d if range_20d > 0 else 0.5
        else:
            features['position_in_20d_range'] = 0.5

        return features

    # === ヘルパーメソッド ===

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """RSIを計算"""
        if len(close) < period + 1:
            return 50.0

        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        close: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """MACDを計算"""
        if len(close) < slow + signal:
            return 0.0, 0.0, 0.0

        # 指数移動平均
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)

        macd_line = ema_fast[-1] - ema_slow[-1]
        macd_series = ema_fast - ema_slow
        signal_line = self._ema(macd_series, signal)[-1]
        histogram = macd_line - signal_line

        # 価格に対する正規化
        price = close[-1]
        return macd_line / price * 100, signal_line / price * 100, histogram / price * 100

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """指数移動平均を計算"""
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _calculate_bb_position(self, close: np.ndarray, period: int = 20) -> float:
        """ボリンジャーバンド内の位置を計算 (-1 to 1)"""
        if len(close) < period:
            return 0.0

        sma = np.mean(close[-period:])
        std = np.std(close[-period:])

        if std == 0:
            return 0.0

        upper = sma + 2 * std
        lower = sma - 2 * std

        current = close[-1]

        # -1（下限バンド）から+1（上限バンド）の範囲に正規化
        position = (current - sma) / (2 * std)
        return np.clip(position, -1, 1)

    def _calculate_stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[float, float]:
        """ストキャスティクスを計算"""
        if len(close) < k_period:
            return 50.0, 50.0

        highest_high = high[-k_period:].max()
        lowest_low = low[-k_period:].min()

        if highest_high == lowest_low:
            k = 50.0
        else:
            k = (close[-1] - lowest_low) / (highest_high - lowest_low) * 100

        # %D は %K の移動平均（簡略化）
        d = k  # 本来は過去の%Kの平均

        return k, d

    def _calculate_atr_ratio(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> float:
        """ATR（価格比）を計算"""
        if len(close) < period + 1:
            return 0.0

        tr_list = []
        for i in range(-period, 0):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
            tr_list.append(tr)

        atr = np.mean(tr_list)
        return atr / close[-1] * 100  # 価格に対する%

    def _calculate_adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> float:
        """ADX（簡易版）を計算"""
        if len(close) < period + 1:
            return 0.0

        # トレンドの強さを簡易的に計算
        price_change = abs(close[-1] - close[-period]) / close[-period]
        return price_change * 100


def extract_features_for_prediction(
    df: pd.DataFrame,
    result: Dict,
) -> np.ndarray:
    """
    予測用の特徴量ベクトルを抽出するユーティリティ関数

    Args:
        df: OHLCV DataFrame
        result: パターン検出結果

    Returns:
        特徴量の numpy 配列
    """
    extractor = FeatureExtractor()
    feature_set = extractor.extract_all(df, result)
    return feature_set.values


if __name__ == '__main__':
    # テスト
    print("Testing FeatureExtractor...")

    # サンプルデータ生成
    np.random.seed(42)
    dates = pd.date_range(end='2024-01-01', periods=300, freq='B')
    prices = 1000 + np.cumsum(np.random.randn(300) * 10)

    df = pd.DataFrame({
        'Open': prices * 0.995,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, 300),
    }, index=dates)

    # パターン検出結果（サンプル）
    result = {
        'is_match': True,
        'cup_start_date': dates[-200],
        'cup_end_date': dates[-100],
        'cup_bottom_date': dates[-150],
        'handle_start_date': dates[-100],
        'handle_end_date': dates[-80],
        'cup_depth': 20.0,
        'handle_depth': 10.0,
        'pivot_price': prices[-1],
        'pattern_quality_score': 75.0,
        'volume_is_valid': True,
        'breakout_volume_ratio': 1.5,
    }

    # 特徴量抽出
    extractor = FeatureExtractor()
    feature_set = extractor.extract_all(df, result)

    print(f"\nExtracted {len(feature_set.names)} features:")
    for name, value in zip(feature_set.names, feature_set.values):
        print(f"  {name}: {value:.4f}")
