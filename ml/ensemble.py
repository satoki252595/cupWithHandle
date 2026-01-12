"""
Ensemble Model
==============
CNNとGBMを組み合わせたハイブリッドアンサンブルモデル。
画像特徴量と工学的特徴量の両方を活用して予測精度を向上させる。
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.cnn_model import ChartCNN, predict_from_image
from ml.gbm_model import GBMClassifier
from ml.feature_extractor import FeatureExtractor


@dataclass
class EnsemblePrediction:
    """アンサンブル予測結果"""
    label: int  # 0: failure, 1: success
    probability: float  # 成功確率
    cnn_probability: Optional[float] = None
    gbm_probability: Optional[float] = None
    confidence: str = 'low'  # 'low', 'medium', 'high'
    recommendation: str = ''


class HybridEnsemble:
    """
    ハイブリッドアンサンブルモデル

    CNNの画像ベース予測とGBMの特徴量ベース予測を
    重み付き平均で組み合わせる。

    Args:
        cnn_model: 学習済みCNNモデル
        gbm_model: 学習済みGBMモデル
        cnn_weight: CNNの重み（デフォルト0.6）
        gbm_weight: GBMの重み（デフォルト0.4）
        threshold: 成功と判定する確率の閾値
    """

    def __init__(
        self,
        cnn_model: Optional[ChartCNN] = None,
        gbm_model: Optional[GBMClassifier] = None,
        cnn_weight: float = 0.6,
        gbm_weight: float = 0.4,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.cnn_model = cnn_model
        self.gbm_model = gbm_model
        self.cnn_weight = cnn_weight
        self.gbm_weight = gbm_weight
        self.threshold = threshold
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.feature_extractor = FeatureExtractor()

        # 重みの正規化
        total_weight = cnn_weight + gbm_weight
        self.cnn_weight = cnn_weight / total_weight
        self.gbm_weight = gbm_weight / total_weight

    def predict(
        self,
        df: pd.DataFrame,
        result: Dict,
        image_path: Optional[str] = None,
    ) -> EnsemblePrediction:
        """
        アンサンブル予測を実行

        Args:
            df: OHLCV DataFrame
            result: パターン検出結果
            image_path: チャート画像パス（CNNに必要）

        Returns:
            EnsemblePrediction オブジェクト
        """
        cnn_prob = None
        gbm_prob = None
        weights_used = {'cnn': 0, 'gbm': 0}

        # CNN予測
        if self.cnn_model is not None and image_path is not None and os.path.exists(image_path):
            try:
                _, cnn_prob = predict_from_image(
                    self.cnn_model, image_path, self.device
                )
                weights_used['cnn'] = self.cnn_weight
            except Exception as e:
                print(f"CNN prediction error: {e}")

        # GBM予測
        if self.gbm_model is not None:
            try:
                features = self.feature_extractor.extract_all(df, result)
                gbm_prob = self.gbm_model.predict_proba(features.values.reshape(1, -1))[0]
                weights_used['gbm'] = self.gbm_weight
            except Exception as e:
                print(f"GBM prediction error: {e}")

        # アンサンブル確率を計算
        if cnn_prob is not None and gbm_prob is not None:
            # 両方のモデルが利用可能
            ensemble_prob = (
                weights_used['cnn'] * cnn_prob +
                weights_used['gbm'] * gbm_prob
            )
        elif cnn_prob is not None:
            # CNNのみ
            ensemble_prob = cnn_prob
        elif gbm_prob is not None:
            # GBMのみ
            ensemble_prob = gbm_prob
        else:
            # どちらも利用不可（フォールバック: 品質スコアベース）
            quality_score = result.get('pattern_quality_score', 50)
            ensemble_prob = quality_score / 100.0

        # ラベルと信頼度を決定
        label = 1 if ensemble_prob >= self.threshold else 0
        confidence = self._calculate_confidence(ensemble_prob, cnn_prob, gbm_prob)
        recommendation = self._generate_recommendation(
            label, ensemble_prob, confidence, result
        )

        return EnsemblePrediction(
            label=label,
            probability=ensemble_prob,
            cnn_probability=cnn_prob,
            gbm_probability=gbm_prob,
            confidence=confidence,
            recommendation=recommendation,
        )

    def predict_batch(
        self,
        samples: List[Tuple[pd.DataFrame, Dict, Optional[str]]],
    ) -> List[EnsemblePrediction]:
        """
        バッチ予測

        Args:
            samples: [(df, result, image_path), ...] のリスト

        Returns:
            EnsemblePrediction のリスト
        """
        predictions = []
        for df, result, image_path in samples:
            pred = self.predict(df, result, image_path)
            predictions.append(pred)
        return predictions

    def _calculate_confidence(
        self,
        ensemble_prob: float,
        cnn_prob: Optional[float],
        gbm_prob: Optional[float],
    ) -> str:
        """信頼度を計算"""
        # 確率が極端な値（0.2未満または0.8以上）なら高信頼度
        if ensemble_prob < 0.2 or ensemble_prob > 0.8:
            base_confidence = 'high'
        elif ensemble_prob < 0.35 or ensemble_prob > 0.65:
            base_confidence = 'medium'
        else:
            base_confidence = 'low'

        # 両モデルの一致度をチェック
        if cnn_prob is not None and gbm_prob is not None:
            diff = abs(cnn_prob - gbm_prob)
            if diff < 0.1:
                # モデル間で一致しているので信頼度UP
                if base_confidence == 'low':
                    return 'medium'
                elif base_confidence == 'medium':
                    return 'high'
            elif diff > 0.3:
                # モデル間で不一致なので信頼度DOWN
                if base_confidence == 'high':
                    return 'medium'
                elif base_confidence == 'medium':
                    return 'low'

        return base_confidence

    def _generate_recommendation(
        self,
        label: int,
        probability: float,
        confidence: str,
        result: Dict,
    ) -> str:
        """推奨アクションを生成"""
        quality_score = result.get('pattern_quality_score', 0)

        if label == 1:  # 成功予測
            if confidence == 'high' and quality_score >= 70:
                return "強い買いシグナル: パターン品質・ML予測ともに良好"
            elif confidence == 'high':
                return "買いシグナル: ML予測は良好だがパターン品質を確認"
            elif confidence == 'medium':
                return "弱い買いシグナル: 追加の確認を推奨"
            else:
                return "シグナル不明確: 慎重な判断を推奨"
        else:  # 失敗予測
            if confidence == 'high':
                return "見送り推奨: ML予測が否定的"
            elif confidence == 'medium':
                return "要注意: 失敗リスクあり"
            else:
                return "シグナル不明確: パターンのみでの判断を推奨"

    def calibrate_weights(
        self,
        X_val: List[Tuple[pd.DataFrame, Dict, str]],
        y_val: np.ndarray,
    ) -> Tuple[float, float]:
        """
        検証データを使って最適な重みを探索

        Args:
            X_val: [(df, result, image_path), ...]
            y_val: 真のラベル

        Returns:
            (最適なcnn_weight, 最適なgbm_weight)
        """
        best_auc = 0
        best_weights = (self.cnn_weight, self.gbm_weight)

        # グリッドサーチ
        for cnn_w in np.arange(0.1, 1.0, 0.1):
            gbm_w = 1.0 - cnn_w

            self.cnn_weight = cnn_w
            self.gbm_weight = gbm_w

            predictions = self.predict_batch(X_val)
            probs = np.array([p.probability for p in predictions])

            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_val, probs)

                if auc > best_auc:
                    best_auc = auc
                    best_weights = (cnn_w, gbm_w)
            except:
                continue

        self.cnn_weight, self.gbm_weight = best_weights
        print(f"Calibrated weights: CNN={self.cnn_weight:.2f}, GBM={self.gbm_weight:.2f}")
        print(f"Best AUC: {best_auc:.4f}")

        return best_weights

    def save(self, save_dir: str):
        """モデルを保存"""
        os.makedirs(save_dir, exist_ok=True)

        # CNNモデル保存
        if self.cnn_model is not None:
            cnn_path = os.path.join(save_dir, 'cnn_model.pt')
            torch.save(self.cnn_model.state_dict(), cnn_path)

        # GBMモデル保存
        if self.gbm_model is not None:
            gbm_path = os.path.join(save_dir, 'gbm_model')
            self.gbm_model.save(gbm_path)

        # 設定保存
        config = {
            'cnn_weight': self.cnn_weight,
            'gbm_weight': self.gbm_weight,
            'threshold': self.threshold,
        }
        config_path = os.path.join(save_dir, 'ensemble_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Ensemble model saved to: {save_dir}")

    def load(self, save_dir: str):
        """モデルを読み込み"""
        # 設定読み込み
        config_path = os.path.join(save_dir, 'ensemble_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.cnn_weight = config.get('cnn_weight', 0.6)
            self.gbm_weight = config.get('gbm_weight', 0.4)
            self.threshold = config.get('threshold', 0.5)

        # CNNモデル読み込み
        cnn_path = os.path.join(save_dir, 'cnn_model.pt')
        if os.path.exists(cnn_path) and self.cnn_model is not None:
            self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
            self.cnn_model.to(self.device)
            self.cnn_model.eval()

        # GBMモデル読み込み
        gbm_path = os.path.join(save_dir, 'gbm_model')
        if os.path.exists(gbm_path + '.meta'):
            if self.gbm_model is None:
                self.gbm_model = GBMClassifier()
            self.gbm_model.load(gbm_path)

        print(f"Ensemble model loaded from: {save_dir}")


class SimpleEnsemble:
    """
    シンプルなアンサンブルモデル

    GBMモデルのみを使用する軽量版。
    CNN訓練データが不足している場合に使用。
    """

    def __init__(
        self,
        gbm_model: Optional[GBMClassifier] = None,
        threshold: float = 0.5,
    ):
        self.gbm_model = gbm_model
        self.threshold = threshold
        self.feature_extractor = FeatureExtractor()

    def predict(
        self,
        df: pd.DataFrame,
        result: Dict,
    ) -> EnsemblePrediction:
        """予測"""
        if self.gbm_model is None:
            # GBMモデルがない場合は品質スコアベース
            quality_score = result.get('pattern_quality_score', 50)
            prob = quality_score / 100.0
        else:
            features = self.feature_extractor.extract_all(df, result)
            prob = self.gbm_model.predict_proba(features.values.reshape(1, -1))[0]

        label = 1 if prob >= self.threshold else 0

        # 信頼度
        if prob < 0.3 or prob > 0.7:
            confidence = 'high'
        elif prob < 0.4 or prob > 0.6:
            confidence = 'medium'
        else:
            confidence = 'low'

        return EnsemblePrediction(
            label=label,
            probability=prob,
            cnn_probability=None,
            gbm_probability=prob,
            confidence=confidence,
            recommendation=self._generate_recommendation(label, prob, confidence),
        )

    def _generate_recommendation(
        self,
        label: int,
        probability: float,
        confidence: str,
    ) -> str:
        """推奨アクションを生成"""
        if label == 1:
            if confidence == 'high':
                return f"買いシグナル (確率: {probability:.1%})"
            elif confidence == 'medium':
                return f"弱い買いシグナル (確率: {probability:.1%})"
            else:
                return f"シグナル不明確 (確率: {probability:.1%})"
        else:
            if confidence == 'high':
                return f"見送り推奨 (成功確率: {probability:.1%})"
            else:
                return f"要注意 (成功確率: {probability:.1%})"


def create_ensemble_from_checkpoint(
    checkpoint_dir: str,
    use_cnn: bool = True,
    device: Optional[torch.device] = None,
) -> Union[HybridEnsemble, SimpleEnsemble]:
    """
    チェックポイントからアンサンブルモデルを作成

    Args:
        checkpoint_dir: モデル保存ディレクトリ
        use_cnn: CNNモデルを使用するか
        device: デバイス

    Returns:
        アンサンブルモデル
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnn_path = os.path.join(checkpoint_dir, 'cnn_model.pt')
    gbm_path = os.path.join(checkpoint_dir, 'gbm_model')

    cnn_model = None
    gbm_model = None

    # CNN読み込み
    if use_cnn and os.path.exists(cnn_path):
        cnn_model = ChartCNN(num_classes=2)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn_model.to(device)
        cnn_model.eval()
        print(f"Loaded CNN model from {cnn_path}")

    # GBM読み込み
    if os.path.exists(gbm_path + '.meta'):
        gbm_model = GBMClassifier()
        gbm_model.load(gbm_path)
        print(f"Loaded GBM model from {gbm_path}")

    if cnn_model is not None:
        return HybridEnsemble(
            cnn_model=cnn_model,
            gbm_model=gbm_model,
            device=device,
        )
    else:
        return SimpleEnsemble(gbm_model=gbm_model)


if __name__ == '__main__':
    # テスト
    print("Testing Ensemble Models...")

    # サンプルデータ
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

    result = {
        'is_match': True,
        'cup_depth': 20.0,
        'handle_depth': 10.0,
        'pattern_quality_score': 75.0,
        'volume_is_valid': True,
        'cup_start_date': dates[-200],
        'cup_end_date': dates[-100],
        'handle_start_date': dates[-100],
        'handle_end_date': dates[-80],
    }

    # SimpleEnsembleテスト（GBMなし）
    print("\n1. SimpleEnsemble (no GBM):")
    simple = SimpleEnsemble(gbm_model=None)
    pred = simple.predict(df, result)
    print(f"  Label: {pred.label}")
    print(f"  Probability: {pred.probability:.4f}")
    print(f"  Confidence: {pred.confidence}")
    print(f"  Recommendation: {pred.recommendation}")

    # HybridEnsembleテスト（モデルなし）
    print("\n2. HybridEnsemble (no models - fallback to quality score):")
    hybrid = HybridEnsemble(cnn_model=None, gbm_model=None)
    pred = hybrid.predict(df, result, image_path=None)
    print(f"  Label: {pred.label}")
    print(f"  Probability: {pred.probability:.4f}")
    print(f"  Confidence: {pred.confidence}")
    print(f"  Recommendation: {pred.recommendation}")
