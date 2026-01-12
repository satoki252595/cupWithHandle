#!/usr/bin/env python3
"""
Evaluation Script
=================
訓練済みモデルの評価とレポート生成。
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data_generator import MLDataGenerator, TrainingExample
from ml.feature_extractor import FeatureExtractor
from ml.gbm_model import GBMClassifier, HAS_LIGHTGBM
from ml.ensemble import SimpleEnsemble, HybridEnsemble, create_ensemble_from_checkpoint

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        precision_recall_curve, roc_curve
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ModelEvaluator:
    """
    モデル評価クラス
    """

    def __init__(self, model_dir: str, data_dir: str):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.feature_extractor = FeatureExtractor()
        self.results = {}

    def load_test_data(self) -> List[TrainingExample]:
        """テストデータを読み込み"""
        generator = MLDataGenerator(output_dir=str(self.data_dir))

        dataset_path = self.data_dir / 'dataset.json'
        if not dataset_path.exists():
            print("No dataset found. Using synthetic data for evaluation.")
            return []

        examples = generator.load_dataset('dataset.json')

        # テストデータ（最後の15%）を取得
        _, _, test_examples = generator.get_train_val_test_split(examples)

        return test_examples

    def prepare_features(
        self,
        examples: List[TrainingExample],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """特徴量を準備"""
        features_list = []
        labels = []

        for ex in examples:
            feat = [
                ex.cup_depth,
                ex.handle_depth,
                ex.quality_score,
                ex.cup_duration,
                ex.handle_duration,
                1 if ex.volume_valid else 0,
                ex.handle_depth / ex.cup_depth if ex.cup_depth > 0 else 0,
                ex.handle_duration / ex.cup_duration if ex.cup_duration > 0 else 0,
            ]
            features_list.append(feat)
            labels.append(ex.label)

        feature_names = [
            'cup_depth', 'handle_depth', 'quality_score',
            'cup_duration', 'handle_duration', 'volume_valid',
            'handle_to_cup_ratio', 'handle_to_cup_duration_ratio',
        ]

        return np.array(features_list), np.array(labels), feature_names

    def evaluate_gbm(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """GBMモデルを評価"""
        gbm_path = self.model_dir / 'gbm_model'

        if not (gbm_path.with_suffix('.meta')).exists():
            print("GBM model not found.")
            return {}

        model = GBMClassifier()
        model.load(str(gbm_path))

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }

        if len(np.unique(y_test)) > 1:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba)

        # 混同行列
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # 特徴量重要度
        importance = model.get_feature_importance()
        if importance:
            metrics['feature_importance'] = importance

        return metrics

    def evaluate_at_thresholds(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
    ) -> List[Dict]:
        """複数の閾値で評価"""
        results = []

        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)

            result = {
                'threshold': thresh,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'n_positive': int(y_pred.sum()),
            }
            results.append(result)

        return results

    def calculate_trading_metrics(
        self,
        examples: List[TrainingExample],
        predictions: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict:
        """取引指標を計算"""
        # 予測が1（成功）のもののみ取引
        trade_mask = predictions >= threshold

        if not trade_mask.any():
            return {
                'n_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
            }

        returns = [ex.return_pct for i, ex in enumerate(examples) if trade_mask[i]]

        n_trades = len(returns)
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / n_trades if n_trades > 0 else 0
        avg_return = np.mean(returns) if returns else 0
        total_return = np.sum(returns) if returns else 0

        # シャープレシオ（簡易版）
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 / 20) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return {
            'n_trades': n_trades,
            'win_rate': win_rate * 100,
            'avg_return': avg_return,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
        }

    def generate_report(
        self,
        output_path: str,
        gbm_metrics: Dict,
        threshold_results: List[Dict],
        trading_metrics: Dict,
    ):
        """評価レポートを生成"""
        report = []
        report.append("# Cup with Handle ML Model Evaluation Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Model Directory:** {self.model_dir}")
        report.append("\n---")

        # GBMモデルの評価
        report.append("\n## 1. GBM Model Performance")
        if gbm_metrics:
            report.append("\n### Classification Metrics")
            report.append("\n| Metric | Value |")
            report.append("|--------|-------|")
            for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                if key in gbm_metrics:
                    report.append(f"| {key.capitalize()} | {gbm_metrics[key]:.4f} |")

            if 'confusion_matrix' in gbm_metrics:
                cm = gbm_metrics['confusion_matrix']
                report.append("\n### Confusion Matrix")
                report.append("```")
                report.append("              Predicted")
                report.append("              Neg    Pos")
                report.append(f"Actual Neg   {cm[0][0]:4d}   {cm[0][1]:4d}")
                report.append(f"Actual Pos   {cm[1][0]:4d}   {cm[1][1]:4d}")
                report.append("```")

            if 'feature_importance' in gbm_metrics:
                report.append("\n### Feature Importance")
                report.append("\n| Feature | Importance |")
                report.append("|---------|------------|")
                sorted_imp = sorted(
                    gbm_metrics['feature_importance'].items(),
                    key=lambda x: x[1], reverse=True
                )
                for name, imp in sorted_imp[:10]:
                    report.append(f"| {name} | {imp:.4f} |")
        else:
            report.append("\nGBM model not available.")

        # 閾値別評価
        report.append("\n---")
        report.append("\n## 2. Threshold Analysis")
        report.append("\n| Threshold | Accuracy | Precision | Recall | F1 | N Trades |")
        report.append("|-----------|----------|-----------|--------|-----|----------|")
        for r in threshold_results:
            report.append(
                f"| {r['threshold']:.1f} | {r['accuracy']:.4f} | "
                f"{r['precision']:.4f} | {r['recall']:.4f} | "
                f"{r['f1']:.4f} | {r['n_positive']} |"
            )

        # 取引指標
        report.append("\n---")
        report.append("\n## 3. Trading Simulation")
        report.append("\n| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Number of Trades | {trading_metrics['n_trades']} |")
        report.append(f"| Win Rate | {trading_metrics['win_rate']:.1f}% |")
        report.append(f"| Average Return | {trading_metrics['avg_return']:+.2f}% |")
        report.append(f"| Total Return | {trading_metrics['total_return']:+.2f}% |")
        report.append(f"| Sharpe Ratio | {trading_metrics['sharpe_ratio']:.2f} |")

        # 推奨事項
        report.append("\n---")
        report.append("\n## 4. Recommendations")

        # 最適な閾値を見つける
        best_f1 = max(threshold_results, key=lambda x: x['f1'])
        report.append(f"\n- **Recommended Threshold:** {best_f1['threshold']:.1f} (best F1 score)")

        if gbm_metrics and 'auc' in gbm_metrics:
            if gbm_metrics['auc'] >= 0.7:
                report.append("- Model shows good discriminative ability (AUC >= 0.7)")
            elif gbm_metrics['auc'] >= 0.6:
                report.append("- Model shows moderate discriminative ability (AUC >= 0.6)")
            else:
                report.append("- Model needs improvement (AUC < 0.6). Consider more training data.")

        if trading_metrics['win_rate'] >= 60:
            report.append("- Trading simulation shows promising results")
        else:
            report.append("- Trading simulation suggests careful position sizing")

        report.append("\n---")
        report.append("\n*This report was automatically generated.*")

        # ファイルに書き出し
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"Report saved to: {output_path}")

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str,
    ):
        """ROCカーブをプロット"""
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            return

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"ROC curve saved to: {save_path}")

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str,
    ):
        """Precision-Recallカーブをプロット"""
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            return

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"Precision-Recall curve saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Cup with Handle ML Models')
    parser.add_argument('--model-dir', type=str, default='ml/models',
                       help='Model directory')
    parser.add_argument('--data-dir', type=str, default='ml/data',
                       help='Data directory')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for reports')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / args.model_dir
    data_dir = base_dir / args.data_dir
    output_dir = base_dir / args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Cup with Handle Pattern - ML Model Evaluation")
    print("=" * 60)

    if not HAS_SKLEARN:
        print("scikit-learn not installed. Install with: pip install scikit-learn")
        return

    evaluator = ModelEvaluator(str(model_dir), str(data_dir))

    # テストデータ読み込み
    print("\n[1/4] Loading test data...")
    examples = evaluator.load_test_data()

    if not examples:
        print("No test data available. Generating synthetic data for demo...")
        from ml.data_generator import SyntheticDataGenerator
        from cup_handle_detector import CupHandleDetector

        synth_gen = SyntheticDataGenerator()
        detector = CupHandleDetector()
        examples = []

        for i in range(50):
            df = synth_gen.generate_synthetic_cup_handle(
                base_price=np.random.uniform(500, 5000),
                cup_depth_pct=np.random.uniform(12, 30),
            )
            result = detector.detect(df)

            if result.get('is_match'):
                quality = result.get('pattern_quality_score', 50)
                is_success = np.random.random() < (quality / 100 * 0.7 + 0.15)

                examples.append(TrainingExample(
                    symbol=f'TEST{i:04d}',
                    entry_date=datetime.now().strftime('%Y-%m-%d'),
                    exit_date=datetime.now().strftime('%Y-%m-%d'),
                    return_pct=np.random.uniform(5, 20) if is_success else np.random.uniform(-8, 0),
                    label=1 if is_success else 0,
                    quality_score=quality,
                    cup_depth=result.get('cup_depth', 0),
                    handle_depth=result.get('handle_depth', 0),
                    cup_duration=100,
                    handle_duration=20,
                    volume_valid=result.get('volume_is_valid', False),
                ))

    print(f"Test samples: {len(examples)}")

    # 特徴量準備
    print("\n[2/4] Preparing features...")
    X_test, y_test, feature_names = evaluator.prepare_features(examples)
    print(f"Feature dimensions: {X_test.shape}")

    # GBMモデル評価
    print("\n[3/4] Evaluating GBM model...")
    gbm_metrics = {}

    gbm_path = model_dir / 'gbm_model.meta'
    if gbm_path.exists():
        gbm_metrics = evaluator.evaluate_gbm(X_test, y_test)
        print(f"  Accuracy: {gbm_metrics.get('accuracy', 0):.4f}")
        print(f"  AUC: {gbm_metrics.get('auc', 0):.4f}")

        # GBMの予測確率を取得
        model = GBMClassifier()
        model.load(str(model_dir / 'gbm_model'))
        y_pred_proba = model.predict_proba(X_test)

        # 閾値別評価
        threshold_results = evaluator.evaluate_at_thresholds(y_test, y_pred_proba)

        # ROCカーブ
        if len(np.unique(y_test)) > 1:
            roc_path = output_dir / 'roc_curve.png'
            evaluator.plot_roc_curve(y_test, y_pred_proba, str(roc_path))

            pr_path = output_dir / 'pr_curve.png'
            evaluator.plot_precision_recall_curve(y_test, y_pred_proba, str(pr_path))
    else:
        print("  GBM model not found.")
        threshold_results = []
        y_pred_proba = np.array([ex.quality_score / 100 for ex in examples])

    # 取引シミュレーション
    print("\n[4/4] Running trading simulation...")
    trading_metrics = evaluator.calculate_trading_metrics(examples, y_pred_proba)
    print(f"  Trades: {trading_metrics['n_trades']}")
    print(f"  Win Rate: {trading_metrics['win_rate']:.1f}%")
    print(f"  Avg Return: {trading_metrics['avg_return']:+.2f}%")

    # レポート生成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f'ml_evaluation_report_{timestamp}.md'
    evaluator.generate_report(
        str(report_path),
        gbm_metrics,
        threshold_results,
        trading_metrics,
    )

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
