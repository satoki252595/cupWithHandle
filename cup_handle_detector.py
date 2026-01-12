"""
Cup with Handle Pattern Detector
================================
ウィリアム・オニールの理論に基づく「カップ・ウィズ・ハンドル」パターン検出器。
Mark Minerviniの「Trend Template」を前提とした厳格なフィルタリングを実装。

Author: Quant Developer
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from dataclasses import dataclass
from typing import Optional, Tuple, List
from datetime import datetime


@dataclass
class VolumeAnalysis:
    """ボリューム分析結果を格納するデータクラス"""
    is_valid: bool = False
    cup_left_volume_decline: bool = False  # カップ左側で出来高減少
    cup_bottom_dry_up: bool = False        # 底で出来高枯渇
    cup_right_volume_increase: bool = False # カップ右側で出来高増加
    handle_volume_contraction: bool = False # ハンドルで出来高縮小
    breakout_volume_surge: bool = False     # ブレイクアウトで出来高急増
    avg_volume_50d: Optional[float] = None
    breakout_volume_ratio: Optional[float] = None  # ブレイクアウト時の出来高倍率

    def to_dict(self) -> dict:
        return {
            'volume_is_valid': self.is_valid,
            'cup_left_volume_decline': self.cup_left_volume_decline,
            'cup_bottom_dry_up': self.cup_bottom_dry_up,
            'cup_right_volume_increase': self.cup_right_volume_increase,
            'handle_volume_contraction': self.handle_volume_contraction,
            'breakout_volume_surge': self.breakout_volume_surge,
            'avg_volume_50d': self.avg_volume_50d,
            'breakout_volume_ratio': self.breakout_volume_ratio,
        }


@dataclass
class CupHandleResult:
    """検出結果を格納するデータクラス"""
    is_match: bool
    cup_start_date: Optional[datetime] = None
    cup_end_date: Optional[datetime] = None
    cup_bottom_date: Optional[datetime] = None
    handle_start_date: Optional[datetime] = None
    handle_end_date: Optional[datetime] = None
    cup_depth: Optional[float] = None
    handle_depth: Optional[float] = None
    pivot_price: Optional[float] = None
    trend_template_passed: bool = False
    volume_analysis: Optional[VolumeAnalysis] = None
    pattern_quality_score: Optional[float] = None  # パターン品質スコア (0-100)

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        result = {
            'is_match': self.is_match,
            'cup_start_date': self.cup_start_date,
            'cup_end_date': self.cup_end_date,
            'cup_bottom_date': self.cup_bottom_date,
            'handle_start_date': self.handle_start_date,
            'handle_end_date': self.handle_end_date,
            'cup_depth': self.cup_depth,
            'handle_depth': self.handle_depth,
            'pivot_price': self.pivot_price,
            'trend_template_passed': self.trend_template_passed,
            'pattern_quality_score': self.pattern_quality_score,
        }
        if self.volume_analysis:
            result.update(self.volume_analysis.to_dict())
        return result


class CupHandleDetector:
    """
    カップ・ウィズ・ハンドル パターン検出クラス

    3段階のパイプライン処理:
    1. Trend Template: ステージ2（上昇局面）の判定
    2. Cup Geometry: カップ形状の検出
    3. Handle Geometry: ハンドル形状の検証
    """

    def __init__(
        self,
        # Trend Template パラメータ
        sma_short: int = 50,
        sma_mid: int = 150,
        sma_long: int = 200,
        min_above_52w_low_pct: float = 25.0,
        max_below_52w_high_pct: float = 25.0,

        # Cup Geometry パラメータ
        cup_min_days: int = 35,
        cup_max_days: int = 325,
        cup_min_depth_pct: float = 12.0,
        cup_max_depth_pct: float = 33.0,
        cup_lip_balance_min: float = 0.85,
        cup_lip_balance_max: float = 1.15,

        # Handle Geometry パラメータ
        handle_min_days: int = 5,
        handle_max_days: int = 50,
        handle_max_depth_pct: float = 15.0,

        # Peak検出パラメータ
        peak_order: int = 7,
        trough_order: int = 7,
    ):
        # Trend Template
        self.sma_short = sma_short
        self.sma_mid = sma_mid
        self.sma_long = sma_long
        self.min_above_52w_low_pct = min_above_52w_low_pct
        self.max_below_52w_high_pct = max_below_52w_high_pct

        # Cup Geometry
        self.cup_min_days = cup_min_days
        self.cup_max_days = cup_max_days
        self.cup_min_depth_pct = cup_min_depth_pct
        self.cup_max_depth_pct = cup_max_depth_pct
        self.cup_lip_balance_min = cup_lip_balance_min
        self.cup_lip_balance_max = cup_lip_balance_max

        # Handle Geometry
        self.handle_min_days = handle_min_days
        self.handle_max_days = handle_max_days
        self.handle_max_depth_pct = handle_max_depth_pct

        # Peak検出
        self.peak_order = peak_order
        self.trough_order = trough_order

        # ボリューム分析パラメータ
        self.volume_decline_threshold = 0.7   # 左側で平均の70%以下
        self.volume_dryup_threshold = 0.5     # 底で平均の50%以下
        self.volume_increase_threshold = 1.2  # 右側で平均の120%以上
        self.handle_contraction_threshold = 0.8  # ハンドルで平均の80%以下
        self.breakout_surge_threshold = 1.5   # ブレイクアウトで平均の150%以上

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """移動平均を計算"""
        df = df.copy()
        df['SMA50'] = df['Close'].rolling(window=self.sma_short, min_periods=1).mean()
        df['SMA150'] = df['Close'].rolling(window=self.sma_mid, min_periods=1).mean()
        df['SMA200'] = df['Close'].rolling(window=self.sma_long, min_periods=1).mean()
        return df

    def _check_trend_template(self, df: pd.DataFrame) -> bool:
        """
        Phase 1: Trend Template (前提トレンド判定)

        ステージ2（上昇局面）にあるかを判定:
        1. 現在の株価 > 50日MA > 150日MA > 200日MA
        2. 200日MAが上昇中（過去1ヶ月前と比較）
        3. 現在の株価が52週安値から25%以上高い
        4. 現在の株価が52週高値から25%以内
        """
        if len(df) < self.sma_long + 22:  # 200日 + 1ヶ月分必要
            return False

        current_price = df['Close'].iloc[-1]
        sma50 = df['SMA50'].iloc[-1]
        sma150 = df['SMA150'].iloc[-1]
        sma200 = df['SMA200'].iloc[-1]
        sma200_1m_ago = df['SMA200'].iloc[-22]  # 約1ヶ月前

        # 52週（約252営業日）の高値・安値
        lookback_52w = min(252, len(df))
        high_52w = df['High'].iloc[-lookback_52w:].max()
        low_52w = df['Low'].iloc[-lookback_52w:].min()

        # 条件1: 価格 > SMA50 > SMA150 > SMA200
        cond1 = (current_price > sma50) and (sma50 > sma150) and (sma150 > sma200)

        # 条件2: 200日MAが上昇中
        cond2 = sma200 > sma200_1m_ago

        # 条件3: 52週安値から25%以上高い
        pct_above_low = ((current_price - low_52w) / low_52w) * 100
        cond3 = pct_above_low >= self.min_above_52w_low_pct

        # 条件4: 52週高値から25%以内
        pct_below_high = ((high_52w - current_price) / high_52w) * 100
        cond4 = pct_below_high <= self.max_below_52w_high_pct

        return cond1 and cond2 and cond3 and cond4

    def _find_peaks_and_troughs(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        極大値（Peaks）と極小値（Troughs）のインデックスを検出
        scipy.signal.argrelextrema を使用
        """
        close_prices = df['Close'].values

        # 極大値（Peaks）の検出
        peaks = argrelextrema(
            close_prices,
            np.greater_equal,
            order=self.peak_order
        )[0]

        # 極小値（Troughs）の検出
        troughs = argrelextrema(
            close_prices,
            np.less_equal,
            order=self.trough_order
        )[0]

        return peaks, troughs

    def _find_cup_candidates(
        self,
        df: pd.DataFrame,
        peaks: np.ndarray,
        troughs: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Phase 2: Cup Geometry (カップの検出)

        条件:
        1. 期間: 35〜325営業日
        2. 深さ: 12%〜33%
        3. 左右バランス: 右端は左端の85%〜115%

        Returns:
            List of (left_idx, bottom_idx, right_idx)
        """
        close_prices = df['Close'].values
        candidates = []

        # 全てのPeakペア（左端L, 右端R）を探索
        for i, left_idx in enumerate(peaks):
            left_price = close_prices[left_idx]

            for right_idx in peaks[i+1:]:
                # 期間チェック
                cup_duration = right_idx - left_idx
                if not (self.cup_min_days <= cup_duration <= self.cup_max_days):
                    continue

                right_price = close_prices[right_idx]

                # 左右バランスチェック
                lip_ratio = right_price / left_price
                if not (self.cup_lip_balance_min <= lip_ratio <= self.cup_lip_balance_max):
                    continue

                # カップ区間内の最安値（Bottom）を検出
                cup_segment = close_prices[left_idx:right_idx+1]
                bottom_offset = np.argmin(cup_segment)
                bottom_idx = left_idx + bottom_offset
                bottom_price = cup_segment[bottom_offset]

                # 深さチェック（左端からの下落率）
                depth_pct = ((left_price - bottom_price) / left_price) * 100
                if not (self.cup_min_depth_pct <= depth_pct <= self.cup_max_depth_pct):
                    continue

                # U字形状の簡易チェック: 底が中央付近にあるか
                # V字だと底が端に寄りすぎる
                bottom_position_ratio = bottom_offset / cup_duration
                if not (0.25 <= bottom_position_ratio <= 0.75):
                    continue

                candidates.append((left_idx, bottom_idx, right_idx))

        return candidates

    def _validate_handle(
        self,
        df: pd.DataFrame,
        cup_left_idx: int,
        cup_bottom_idx: int,
        cup_right_idx: int
    ) -> Optional[Tuple[int, float, float]]:
        """
        Phase 3: Handle Geometry (ハンドルの検出)

        条件:
        1. 期間: 5〜50営業日
        2. 上半分ルール: H_low > Bottom + 0.5 * (Left_Lip - Bottom)
        3. 深さ: 10%〜15%以内

        Returns:
            (handle_start_idx, handle_depth_pct, pivot_price) or None
        """
        close_prices = df['Close'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values

        cup_left_price = close_prices[cup_left_idx]
        cup_bottom_price = close_prices[cup_bottom_idx]
        cup_right_price = close_prices[cup_right_idx]

        # 上半分ルールの閾値
        midpoint = cup_bottom_price + 0.5 * (cup_left_price - cup_bottom_price)

        # ハンドル探索範囲
        handle_start = cup_right_idx
        handle_end_max = min(handle_start + self.handle_max_days, len(df) - 1)

        if handle_end_max - handle_start < self.handle_min_days:
            return None

        # ハンドル期間の候補を探索
        for handle_end in range(handle_start + self.handle_min_days, handle_end_max + 1):
            handle_segment_low = low_prices[handle_start:handle_end+1]
            handle_segment_high = high_prices[handle_start:handle_end+1]
            handle_segment_close = close_prices[handle_start:handle_end+1]

            handle_low = np.min(handle_segment_low)
            handle_high = np.max(handle_segment_high)

            # 上半分ルールのチェック
            if handle_low <= midpoint:
                continue

            # ハンドルの深さチェック（右端からの下落率）
            handle_depth_pct = ((cup_right_price - handle_low) / cup_right_price) * 100
            if handle_depth_pct > self.handle_max_depth_pct:
                continue

            # 現在値がハンドルの高値付近にある（ブレイクアウト寸前）
            current_price = close_prices[-1]

            # ピボット価格: ハンドルの高値 or カップ右端の高値
            pivot_price = max(handle_high, high_prices[cup_right_idx])

            # 最新の終値がピボット価格の95%以上（ブレイクアウト近傍）
            if current_price >= pivot_price * 0.95:
                return (cup_right_idx, handle_end, handle_depth_pct, pivot_price)

        return None

    def _analyze_volume(
        self,
        df: pd.DataFrame,
        cup_left_idx: int,
        cup_bottom_idx: int,
        cup_right_idx: int,
        handle_end_idx: int
    ) -> VolumeAnalysis:
        """
        Phase 4: Volume Analysis (ボリューム分析)

        オニール理論に基づくボリューム検証:
        1. カップ左側（下落時）: 出来高減少傾向
        2. カップ底: 出来高枯渇（Dry Up）
        3. カップ右側（上昇時）: 出来高増加傾向
        4. ハンドル形成時: 出来高縮小（Tight Consolidation）
        5. ブレイクアウト時: 出来高急増（平均の50%以上増加が理想）
        """
        volume = df['Volume'].values

        # 50日平均出来高を計算
        avg_vol_50d = np.mean(volume[max(0, len(volume)-50):])

        # カップ左側（左端から底まで）の出来高分析
        left_segment = volume[cup_left_idx:cup_bottom_idx+1]
        left_avg = np.mean(left_segment) if len(left_segment) > 0 else avg_vol_50d
        cup_left_decline = left_avg < avg_vol_50d * self.volume_decline_threshold

        # カップ底付近の出来高分析（底の前後5日）
        bottom_start = max(cup_left_idx, cup_bottom_idx - 5)
        bottom_end = min(cup_right_idx, cup_bottom_idx + 5)
        bottom_segment = volume[bottom_start:bottom_end+1]
        bottom_avg = np.mean(bottom_segment) if len(bottom_segment) > 0 else avg_vol_50d
        cup_bottom_dryup = bottom_avg < avg_vol_50d * self.volume_dryup_threshold

        # カップ右側（底から右端まで）の出来高分析
        right_segment = volume[cup_bottom_idx:cup_right_idx+1]
        right_avg = np.mean(right_segment) if len(right_segment) > 0 else avg_vol_50d
        cup_right_increase = right_avg > avg_vol_50d * self.volume_increase_threshold

        # ハンドル期間の出来高分析
        handle_segment = volume[cup_right_idx:handle_end_idx+1]
        handle_avg = np.mean(handle_segment) if len(handle_segment) > 0 else avg_vol_50d
        handle_contraction = handle_avg < avg_vol_50d * self.handle_contraction_threshold

        # ブレイクアウト（直近3日）の出来高分析
        breakout_segment = volume[-3:]
        breakout_avg = np.mean(breakout_segment) if len(breakout_segment) > 0 else avg_vol_50d
        breakout_surge = breakout_avg > avg_vol_50d * self.breakout_surge_threshold
        breakout_ratio = round(breakout_avg / avg_vol_50d, 2) if avg_vol_50d > 0 else 0

        # 総合判定: 少なくとも3つの条件を満たす場合は有効
        conditions_met = sum([
            cup_left_decline,
            cup_bottom_dryup,
            cup_right_increase,
            handle_contraction,
            breakout_surge
        ])
        is_valid = conditions_met >= 3

        return VolumeAnalysis(
            is_valid=is_valid,
            cup_left_volume_decline=cup_left_decline,
            cup_bottom_dry_up=cup_bottom_dryup,
            cup_right_volume_increase=cup_right_increase,
            handle_volume_contraction=handle_contraction,
            breakout_volume_surge=breakout_surge,
            avg_volume_50d=round(avg_vol_50d, 0),
            breakout_volume_ratio=breakout_ratio,
        )

    def _calculate_pattern_quality(
        self,
        cup_depth_pct: float,
        handle_depth_pct: float,
        lip_ratio: float,
        bottom_position_ratio: float,
        volume_analysis: VolumeAnalysis
    ) -> float:
        """
        パターン品質スコアを計算 (0-100)

        評価基準:
        - カップの深さ: 15-25%が理想的
        - ハンドルの深さ: 5-12%が理想的
        - 左右バランス: 1.0に近いほど良い
        - 底の位置: 0.5（中央）に近いほど良い
        - ボリューム分析: 条件を満たすほど高得点
        """
        score = 0.0

        # カップ深さスコア (最大25点)
        if 15 <= cup_depth_pct <= 25:
            score += 25
        elif 12 <= cup_depth_pct <= 33:
            score += 15
        else:
            score += 5

        # ハンドル深さスコア (最大20点)
        if 5 <= handle_depth_pct <= 12:
            score += 20
        elif handle_depth_pct <= 15:
            score += 12
        else:
            score += 5

        # 左右バランススコア (最大20点)
        balance_deviation = abs(1.0 - lip_ratio)
        if balance_deviation <= 0.05:
            score += 20
        elif balance_deviation <= 0.10:
            score += 15
        elif balance_deviation <= 0.15:
            score += 10
        else:
            score += 5

        # 底の位置スコア (最大15点)
        position_deviation = abs(0.5 - bottom_position_ratio)
        if position_deviation <= 0.10:
            score += 15
        elif position_deviation <= 0.20:
            score += 10
        else:
            score += 5

        # ボリュームスコア (最大20点)
        volume_conditions = sum([
            volume_analysis.cup_left_volume_decline,
            volume_analysis.cup_bottom_dry_up,
            volume_analysis.cup_right_volume_increase,
            volume_analysis.handle_volume_contraction,
            volume_analysis.breakout_volume_surge
        ])
        score += volume_conditions * 4  # 各条件4点

        return round(score, 1)

    def _find_best_pattern(
        self,
        df: pd.DataFrame,
        cup_candidates: List[Tuple[int, int, int]]
    ) -> Optional[CupHandleResult]:
        """
        最適なカップ・ウィズ・ハンドルパターンを選択

        優先順位:
        1. 最も直近のパターン
        2. ハンドルが有効なもの
        3. パターン品質スコアが高いもの
        """
        close_prices = df['Close'].values
        best_result = None
        best_score = -1

        # 右端が新しい順にソート
        sorted_candidates = sorted(cup_candidates, key=lambda x: x[2], reverse=True)

        for cup_left_idx, cup_bottom_idx, cup_right_idx in sorted_candidates:
            handle_result = self._validate_handle(
                df, cup_left_idx, cup_bottom_idx, cup_right_idx
            )

            if handle_result is not None:
                handle_start_idx, handle_end_idx, handle_depth_pct, pivot_price = handle_result

                # カップの深さを計算
                cup_left_price = close_prices[cup_left_idx]
                cup_right_price = close_prices[cup_right_idx]
                cup_bottom_price = close_prices[cup_bottom_idx]
                cup_depth_pct = ((cup_left_price - cup_bottom_price) / cup_left_price) * 100

                # 左右バランスと底の位置を計算
                lip_ratio = cup_right_price / cup_left_price
                cup_duration = cup_right_idx - cup_left_idx
                bottom_offset = cup_bottom_idx - cup_left_idx
                bottom_position_ratio = bottom_offset / cup_duration

                # ボリューム分析
                volume_analysis = self._analyze_volume(
                    df, cup_left_idx, cup_bottom_idx, cup_right_idx, handle_end_idx
                )

                # パターン品質スコア
                quality_score = self._calculate_pattern_quality(
                    cup_depth_pct, handle_depth_pct, lip_ratio,
                    bottom_position_ratio, volume_analysis
                )

                # より高いスコアのパターンを選択
                if quality_score > best_score:
                    best_score = quality_score
                    best_result = CupHandleResult(
                        is_match=True,
                        cup_start_date=df.index[cup_left_idx].to_pydatetime()
                            if hasattr(df.index[cup_left_idx], 'to_pydatetime')
                            else df.index[cup_left_idx],
                        cup_end_date=df.index[cup_right_idx].to_pydatetime()
                            if hasattr(df.index[cup_right_idx], 'to_pydatetime')
                            else df.index[cup_right_idx],
                        cup_bottom_date=df.index[cup_bottom_idx].to_pydatetime()
                            if hasattr(df.index[cup_bottom_idx], 'to_pydatetime')
                            else df.index[cup_bottom_idx],
                        handle_start_date=df.index[handle_start_idx].to_pydatetime()
                            if hasattr(df.index[handle_start_idx], 'to_pydatetime')
                            else df.index[handle_start_idx],
                        handle_end_date=df.index[handle_end_idx].to_pydatetime()
                            if hasattr(df.index[handle_end_idx], 'to_pydatetime')
                            else df.index[handle_end_idx],
                        cup_depth=round(cup_depth_pct, 2),
                        handle_depth=round(handle_depth_pct, 2),
                        pivot_price=round(pivot_price, 2),
                        trend_template_passed=True,
                        volume_analysis=volume_analysis,
                        pattern_quality_score=quality_score,
                    )

        return best_result

    def detect(self, df: pd.DataFrame) -> dict:
        """
        カップ・ウィズ・ハンドルパターンを検出

        Args:
            df: OHLCV形式のDataFrame (columns: Open, High, Low, Close, Volume)
                index: DatetimeIndex

        Returns:
            検出結果の辞書
        """
        # データの検証
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(df) < self.sma_long + self.cup_min_days:
            return CupHandleResult(
                is_match=False,
                trend_template_passed=False
            ).to_dict()

        # 移動平均を計算
        df = self._calculate_moving_averages(df)

        # Phase 1: Trend Template チェック
        if not self._check_trend_template(df):
            return CupHandleResult(
                is_match=False,
                trend_template_passed=False
            ).to_dict()

        # Phase 2: Peaks/Troughs検出 & Cup Geometry
        peaks, troughs = self._find_peaks_and_troughs(df)

        if len(peaks) < 2:
            return CupHandleResult(
                is_match=False,
                trend_template_passed=True
            ).to_dict()

        cup_candidates = self._find_cup_candidates(df, peaks, troughs)

        if not cup_candidates:
            return CupHandleResult(
                is_match=False,
                trend_template_passed=True
            ).to_dict()

        # Phase 3: Handle Geometry & 最適パターン選択
        result = self._find_best_pattern(df, cup_candidates)

        if result is None:
            return CupHandleResult(
                is_match=False,
                trend_template_passed=True
            ).to_dict()

        return result.to_dict()

    def detect_all_patterns(
        self,
        df: pd.DataFrame,
        max_patterns: int = 5
    ) -> List[dict]:
        """
        全てのカップ・ウィズ・ハンドルパターンを検出

        Args:
            df: OHLCV形式のDataFrame
            max_patterns: 返すパターンの最大数

        Returns:
            検出結果のリスト（新しい順）
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(df) < self.sma_long + self.cup_min_days:
            return []

        df = self._calculate_moving_averages(df)

        if not self._check_trend_template(df):
            return []

        peaks, troughs = self._find_peaks_and_troughs(df)

        if len(peaks) < 2:
            return []

        cup_candidates = self._find_cup_candidates(df, peaks, troughs)

        if not cup_candidates:
            return []

        # 全てのパターンを検証
        close_prices = df['Close'].values
        results = []
        sorted_candidates = sorted(cup_candidates, key=lambda x: x[2], reverse=True)

        for cup_left_idx, cup_bottom_idx, cup_right_idx in sorted_candidates:
            handle_result = self._validate_handle(
                df, cup_left_idx, cup_bottom_idx, cup_right_idx
            )

            if handle_result is not None:
                handle_start_idx, handle_end_idx, handle_depth_pct, pivot_price = handle_result

                cup_left_price = close_prices[cup_left_idx]
                cup_right_price = close_prices[cup_right_idx]
                cup_bottom_price = close_prices[cup_bottom_idx]
                cup_depth_pct = ((cup_left_price - cup_bottom_price) / cup_left_price) * 100

                # 左右バランスと底の位置を計算
                lip_ratio = cup_right_price / cup_left_price
                cup_duration = cup_right_idx - cup_left_idx
                bottom_offset = cup_bottom_idx - cup_left_idx
                bottom_position_ratio = bottom_offset / cup_duration

                # ボリューム分析
                volume_analysis = self._analyze_volume(
                    df, cup_left_idx, cup_bottom_idx, cup_right_idx, handle_end_idx
                )

                # パターン品質スコア
                quality_score = self._calculate_pattern_quality(
                    cup_depth_pct, handle_depth_pct, lip_ratio,
                    bottom_position_ratio, volume_analysis
                )

                result = CupHandleResult(
                    is_match=True,
                    cup_start_date=df.index[cup_left_idx].to_pydatetime()
                        if hasattr(df.index[cup_left_idx], 'to_pydatetime')
                        else df.index[cup_left_idx],
                    cup_end_date=df.index[cup_right_idx].to_pydatetime()
                        if hasattr(df.index[cup_right_idx], 'to_pydatetime')
                        else df.index[cup_right_idx],
                    cup_bottom_date=df.index[cup_bottom_idx].to_pydatetime()
                        if hasattr(df.index[cup_bottom_idx], 'to_pydatetime')
                        else df.index[cup_bottom_idx],
                    handle_start_date=df.index[handle_start_idx].to_pydatetime()
                        if hasattr(df.index[handle_start_idx], 'to_pydatetime')
                        else df.index[handle_start_idx],
                    handle_end_date=df.index[handle_end_idx].to_pydatetime()
                        if hasattr(df.index[handle_end_idx], 'to_pydatetime')
                        else df.index[handle_end_idx],
                    cup_depth=round(cup_depth_pct, 2),
                    handle_depth=round(handle_depth_pct, 2),
                    pivot_price=round(pivot_price, 2),
                    trend_template_passed=True,
                    volume_analysis=volume_analysis,
                    pattern_quality_score=quality_score,
                )
                results.append(result.to_dict())

                if len(results) >= max_patterns:
                    break

        return results


# サンプルデータでのテスト用関数
def create_sample_data() -> pd.DataFrame:
    """テスト用のサンプルデータを生成"""
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=400, freq='B')

    # 上昇トレンド + カップ形状 + ハンドル形状を模倣
    base_trend = np.linspace(100, 150, 200)  # 上昇トレンド

    # カップ形状（約100日）
    cup_days = 100
    cup_depth = 0.20  # 20%の下落
    cup = np.concatenate([
        np.linspace(150, 150 * (1 - cup_depth), cup_days // 2),  # 下落
        np.linspace(150 * (1 - cup_depth), 148, cup_days // 2),  # 回復
    ])

    # ハンドル形状（約20日）
    handle_days = 20
    handle = np.linspace(148, 145, handle_days // 2)
    handle = np.concatenate([handle, np.linspace(145, 149, handle_days // 2)])

    # 継続上昇
    continuation = np.linspace(149, 160, 80)

    # 結合
    prices = np.concatenate([base_trend, cup, handle, continuation])

    # ノイズ追加
    noise = np.random.normal(0, 1, len(prices))
    prices = prices + noise

    df = pd.DataFrame({
        'Open': prices * 0.995,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, len(prices)),
    }, index=dates)

    return df


if __name__ == '__main__':
    # サンプルデータでテスト
    df = create_sample_data()

    detector = CupHandleDetector()
    result = detector.detect(df)

    print("=== Cup with Handle Detection Result ===")
    for key, value in result.items():
        print(f"{key}: {value}")
