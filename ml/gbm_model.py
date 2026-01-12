"""
Gradient Boosting Model for Pattern Classification
==================================================
LightGBM/XGBoostベースの勾配ブースティングモデルで
工学的特徴量からカップ・ウィズ・ハンドルパターンの成功/失敗を予測する。
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class GBMPrediction:
    """GBMモデルの予測結果"""
    label: int
    probability: float
    feature_importance: Optional[Dict[str, float]] = None


class GBMClassifier:
    """
    勾配ブースティング分類器

    LightGBMまたはXGBoostを使用して特徴量ベースの予測を行う。
    """

    def __init__(
        self,
        backend: str = 'lightgbm',  # 'lightgbm' or 'xgboost'
        params: Optional[Dict] = None,
        random_state: int = 42,
    ):
        self.backend = backend
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

        # デフォルトパラメータ
        if params is None:
            if backend == 'lightgbm':
                self.params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 20,
                    'verbose': -1,
                    'seed': random_state,
                }
            else:  # xgboost
                self.params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 20,
                    'seed': random_state,
                }
        else:
            self.params = params

        # バックエンドのチェック
        if backend == 'lightgbm' and not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
        if backend == 'xgboost' and not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> 'GBMClassifier':
        """
        モデルを訓練

        Args:
            X: 特徴量（訓練データ）
            y: ラベル（訓練データ）
            feature_names: 特徴量名のリスト
            X_val: 特徴量（検証データ）
            y_val: ラベル（検証データ）
            num_boost_round: ブースティング回数
            early_stopping_rounds: 早期停止のラウンド数

        Returns:
            self
        """
        self.feature_names = feature_names

        # 特徴量のスケーリング
        X_scaled = self.scaler.fit_transform(X)

        if self.backend == 'lightgbm':
            train_data = lgb.Dataset(X_scaled, label=y, feature_name=feature_names)

            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
                callbacks = [
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                    lgb.log_evaluation(period=100)
                ]
                self.model = lgb.train(
                    self.params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'val'],
                    callbacks=callbacks,
                )
            else:
                self.model = lgb.train(
                    self.params,
                    train_data,
                    num_boost_round=num_boost_round,
                )

        else:  # xgboost
            dtrain = xgb.DMatrix(X_scaled, label=y, feature_names=feature_names)

            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=feature_names)
                watchlist = [(dtrain, 'train'), (dval, 'val')]
                self.model = xgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=100,
                )
            else:
                self.model = xgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=num_boost_round,
                )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測（クラスラベル）"""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """予測（確率）"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        if self.backend == 'lightgbm':
            return self.model.predict(X_scaled)
        else:  # xgboost
            dtest = xgb.DMatrix(X_scaled, feature_names=self.feature_names)
            return self.model.predict(dtest)

    def predict_single(self, X: np.ndarray) -> GBMPrediction:
        """
        単一サンプルの予測（詳細情報付き）

        Args:
            X: 特徴量ベクトル（1D or 2D）

        Returns:
            GBMPrediction オブジェクト
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        prob = self.predict_proba(X)[0]
        label = 1 if prob >= 0.5 else 0

        # 特徴量重要度
        importance = self.get_feature_importance()

        return GBMPrediction(
            label=label,
            probability=prob,
            feature_importance=importance,
        )

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """特徴量重要度を取得"""
        if self.model is None or self.feature_names is None:
            return None

        if self.backend == 'lightgbm':
            importance = self.model.feature_importance(importance_type='gain')
        else:  # xgboost
            importance = list(self.model.get_score(importance_type='gain').values())

        # 正規化
        total = sum(importance)
        if total > 0:
            importance = [imp / total for imp in importance]

        return dict(zip(self.feature_names, importance))

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """
        クロスバリデーション

        Args:
            X: 特徴量
            y: ラベル
            n_splits: 分割数

        Returns:
            評価指標の辞書
        """
        X_scaled = self.scaler.fit_transform(X)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        accuracies = []
        aucs = []

        for train_idx, val_idx in cv.split(X_scaled, y):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if self.backend == 'lightgbm':
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

                model = lgb.train(
                    self.params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )
                y_pred_proba = model.predict(X_val)
            else:  # xgboost
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)

                model = xgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=500,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )
                y_pred_proba = model.predict(dval)

            y_pred = (y_pred_proba >= 0.5).astype(int)

            accuracies.append(accuracy_score(y_val, y_pred))
            aucs.append(roc_auc_score(y_val, y_pred_proba))

        return {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
        }

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        モデルを評価

        Args:
            X: 特徴量
            y: 真のラベル

        Returns:
            評価指標の辞書
        """
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0,
        }

    def save(self, path: str):
        """モデルを保存"""
        data = {
            'backend': self.backend,
            'params': self.params,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
        }

        # モデル本体を保存
        model_path = path + '.model'
        if self.backend == 'lightgbm':
            self.model.save_model(model_path)
        else:  # xgboost
            self.model.save_model(model_path)

        # メタデータを保存
        meta_path = path + '.meta'
        with open(meta_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Model saved to: {path}")

    def load(self, path: str):
        """モデルを読み込み"""
        # メタデータを読み込み
        meta_path = path + '.meta'
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)

        self.backend = data['backend']
        self.params = data['params']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']

        # モデル本体を読み込み
        model_path = path + '.model'
        if self.backend == 'lightgbm':
            self.model = lgb.Booster(model_file=model_path)
        else:  # xgboost
            self.model = xgb.Booster()
            self.model.load_model(model_path)

        print(f"Model loaded from: {path}")


class GBMRegressor:
    """
    勾配ブースティング回帰器

    リターン値を直接予測する回帰モデル。
    """

    def __init__(
        self,
        backend: str = 'lightgbm',
        params: Optional[Dict] = None,
        random_state: int = 42,
    ):
        self.backend = backend
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

        if params is None:
            if backend == 'lightgbm':
                self.params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 20,
                    'verbose': -1,
                    'seed': random_state,
                }
            else:
                self.params = {
                    'objective': 'reg:squarederror',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'seed': random_state,
                }
        else:
            self.params = params

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> 'GBMRegressor':
        """モデルを訓練"""
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)

        if self.backend == 'lightgbm':
            train_data = lgb.Dataset(X_scaled, label=y, feature_name=feature_names)

            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
                callbacks = [
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                    lgb.log_evaluation(period=100)
                ]
                self.model = lgb.train(
                    self.params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'val'],
                    callbacks=callbacks,
                )
            else:
                self.model = lgb.train(
                    self.params,
                    train_data,
                    num_boost_round=num_boost_round,
                )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測（リターン値）"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        if self.backend == 'lightgbm':
            return self.model.predict(X_scaled)
        else:
            dtest = xgb.DMatrix(X_scaled, feature_names=self.feature_names)
            return self.model.predict(dtest)


def train_gbm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    backend: str = 'lightgbm',
) -> Tuple[GBMClassifier, Dict]:
    """
    GBM分類器を訓練するユーティリティ関数

    Returns:
        (訓練済みモデル, 評価結果) のタプル
    """
    model = GBMClassifier(backend=backend)
    model.fit(
        X_train, y_train,
        feature_names=feature_names,
        X_val=X_val,
        y_val=y_val,
    )

    metrics = model.evaluate(X_val, y_val)
    return model, metrics


if __name__ == '__main__':
    # テスト
    print("Testing GBMClassifier...")

    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 200
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.3 > 0).astype(int)

    feature_names = [f'feature_{i}' for i in range(n_features)]

    # 訓練・検証分割
    train_size = int(n_samples * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # モデル訓練
    if HAS_LIGHTGBM:
        print("\nTraining LightGBM...")
        model = GBMClassifier(backend='lightgbm')
        model.fit(
            X_train, y_train,
            feature_names=feature_names,
            X_val=X_val,
            y_val=y_val,
        )

        # 評価
        metrics = model.evaluate(X_val, y_val)
        print(f"\nEvaluation metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # 特徴量重要度
        importance = model.get_feature_importance()
        print(f"\nTop 5 important features:")
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, imp in sorted_imp:
            print(f"  {name}: {imp:.4f}")

        # クロスバリデーション
        print("\nCross-validation...")
        cv_results = model.cross_validate(X, y)
        print(f"  Accuracy: {cv_results['accuracy_mean']:.4f} (+/- {cv_results['accuracy_std']:.4f})")
        print(f"  AUC: {cv_results['auc_mean']:.4f} (+/- {cv_results['auc_std']:.4f})")
    else:
        print("LightGBM not installed. Skipping test.")
