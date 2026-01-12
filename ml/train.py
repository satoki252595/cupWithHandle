#!/usr/bin/env python3
"""
Training Script
===============
Cup with Handle パターン分類モデルの訓練スクリプト。
CNN画像分類モデルとGBM特徴量モデルを訓練する。
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data_generator import MLDataGenerator, TrainingExample, SyntheticDataGenerator
from ml.feature_extractor import FeatureExtractor
from ml.cnn_model import ChartCNN, ChartImageDataset, CNNTrainer, get_data_transforms
from ml.gbm_model import GBMClassifier, HAS_LIGHTGBM
from ml.ensemble import HybridEnsemble


def prepare_training_data(
    data_dir: str,
    generate_synthetic: bool = True,
    n_synthetic: int = 100,
) -> Tuple[List[TrainingExample], Dict]:
    """
    訓練データを準備

    Args:
        data_dir: データディレクトリ
        generate_synthetic: 合成データを生成するか
        n_synthetic: 合成データ数

    Returns:
        (訓練サンプルリスト, 統計情報)
    """
    generator = MLDataGenerator(output_dir=data_dir)

    # 既存データセットの読み込みを試みる
    dataset_path = os.path.join(data_dir, 'dataset.json')
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}")
        examples = generator.load_dataset('dataset.json')
    else:
        examples = []
        print("No existing dataset found.")

    # 合成データの追加
    if generate_synthetic and len(examples) < 100:
        print(f"Generating {n_synthetic} synthetic examples...")
        synth_gen = SyntheticDataGenerator()

        from cup_handle_detector import CupHandleDetector
        detector = CupHandleDetector()

        for i in range(n_synthetic):
            # ランダムなパラメータで合成データ生成
            cup_depth = np.random.uniform(12, 30)
            handle_depth = np.random.uniform(5, 15)

            df = synth_gen.generate_synthetic_cup_handle(
                base_price=np.random.uniform(500, 5000),
                cup_depth_pct=cup_depth,
                handle_depth_pct=handle_depth,
            )

            result = detector.detect(df)

            if result.get('is_match'):
                # 成功/失敗をランダムに割り当て（合成データなので）
                # 品質スコアが高いほど成功確率が高い
                quality = result.get('pattern_quality_score', 50)
                success_prob = quality / 100 * 0.7 + 0.15
                is_success = np.random.random() < success_prob
                return_pct = np.random.uniform(5, 20) if is_success else np.random.uniform(-8, 0)

                example = TrainingExample(
                    symbol=f'SYNTH{i:04d}',
                    entry_date=datetime.now().strftime('%Y-%m-%d'),
                    exit_date=datetime.now().strftime('%Y-%m-%d'),
                    return_pct=return_pct,
                    label=1 if is_success else 0,
                    quality_score=quality,
                    cup_depth=result.get('cup_depth', 0),
                    handle_depth=result.get('handle_depth', 0),
                    cup_duration=100,
                    handle_duration=20,
                    volume_valid=result.get('volume_is_valid', False),
                )
                examples.append(example)

        print(f"Generated {n_synthetic} synthetic examples")

    # 統計情報
    stats = {
        'total': len(examples),
        'success': sum(1 for ex in examples if ex.label == 1),
        'failure': sum(1 for ex in examples if ex.label == 0),
    }

    return examples, stats


def prepare_features(
    examples: List[TrainingExample],
    feature_extractor: FeatureExtractor,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    特徴量を準備

    Args:
        examples: 訓練サンプル
        feature_extractor: 特徴量抽出器

    Returns:
        (特徴量配列, ラベル配列, 特徴量名)
    """
    # 訓練サンプルから簡易的な特徴量を抽出
    features_list = []
    labels = []

    for ex in examples:
        # 基本特徴量
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
        'cup_depth',
        'handle_depth',
        'quality_score',
        'cup_duration',
        'handle_duration',
        'volume_valid',
        'handle_to_cup_ratio',
        'handle_to_cup_duration_ratio',
    ]

    X = np.array(features_list)
    y = np.array(labels)

    return X, y, feature_names


def train_gbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    save_dir: str,
) -> GBMClassifier:
    """
    GBMモデルを訓練

    Args:
        X_train, y_train: 訓練データ
        X_val, y_val: 検証データ
        feature_names: 特徴量名
        save_dir: 保存先

    Returns:
        訓練済みGBMClassifier
    """
    print("\n" + "=" * 50)
    print("Training GBM Model")
    print("=" * 50)

    if not HAS_LIGHTGBM:
        print("LightGBM not installed. Skipping GBM training.")
        return None

    model = GBMClassifier(backend='lightgbm')

    model.fit(
        X_train, y_train,
        feature_names=feature_names,
        X_val=X_val,
        y_val=y_val,
        num_boost_round=500,
        early_stopping_rounds=50,
    )

    # 評価
    metrics = model.evaluate(X_val, y_val)
    print(f"\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # 特徴量重要度
    importance = model.get_feature_importance()
    if importance:
        print(f"\nFeature Importance:")
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_imp[:10]:
            print(f"  {name}: {imp:.4f}")

    # 保存
    model_path = os.path.join(save_dir, 'gbm_model')
    model.save(model_path)

    return model


def train_cnn_model(
    data_dir: str,
    save_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> Optional[ChartCNN]:
    """
    CNNモデルを訓練

    Args:
        data_dir: データディレクトリ（画像が含まれる）
        save_dir: 保存先
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率

    Returns:
        訓練済みChartCNN または None
    """
    print("\n" + "=" * 50)
    print("Training CNN Model")
    print("=" * 50)

    # 画像ディレクトリの確認
    success_dir = os.path.join(data_dir, 'images', 'success')
    failure_dir = os.path.join(data_dir, 'images', 'failure')

    if not os.path.exists(success_dir) or not os.path.exists(failure_dir):
        print("Image directories not found. Skipping CNN training.")
        return None

    # 画像パスとラベルを収集
    image_paths = []
    labels = []

    for img_file in os.listdir(success_dir):
        if img_file.endswith('.png'):
            image_paths.append(os.path.join(success_dir, img_file))
            labels.append(1)

    for img_file in os.listdir(failure_dir):
        if img_file.endswith('.png'):
            image_paths.append(os.path.join(failure_dir, img_file))
            labels.append(0)

    print(f"Found {len(image_paths)} images")
    print(f"  Success: {sum(labels)}")
    print(f"  Failure: {len(labels) - sum(labels)}")

    if len(image_paths) < 20:
        print("Insufficient images for CNN training (need at least 20). Skipping.")
        return None

    # データ分割
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)

    train_size = int(len(indices) * 0.8)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    # データ変換
    transforms_dict = get_data_transforms(augment=True)

    train_dataset = ChartImageDataset(train_paths, train_labels, transforms_dict['train'])
    val_dataset = ChartImageDataset(val_paths, val_labels, transforms_dict['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # モデル作成
    model = ChartCNN(num_classes=2, pretrained=True, backbone='resnet18')

    # まずバックボーンを固定して訓練
    print("\nPhase 1: Training with frozen backbone...")
    model.freeze_backbone()

    trainer = CNNTrainer(model, learning_rate=learning_rate * 10)
    trainer.train(
        train_loader, val_loader,
        epochs=min(10, epochs // 2),
        early_stopping_patience=5,
    )

    # 全層を解放してファインチューニング
    print("\nPhase 2: Fine-tuning all layers...")
    model.unfreeze_backbone()

    trainer = CNNTrainer(model, learning_rate=learning_rate)
    save_path = os.path.join(save_dir, 'cnn_model.pt')
    trainer.train(
        train_loader, val_loader,
        epochs=epochs,
        early_stopping_patience=10,
        save_path=save_path,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description='Train Cup with Handle ML Models')
    parser.add_argument('--data-dir', type=str, default='ml/data',
                       help='Data directory')
    parser.add_argument('--output-dir', type=str, default='ml/models',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for CNN training')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--no-cnn', action='store_true',
                       help='Skip CNN training')
    parser.add_argument('--no-gbm', action='store_true',
                       help='Skip GBM training')
    parser.add_argument('--generate-synthetic', action='store_true',
                       help='Generate synthetic training data')
    parser.add_argument('--n-synthetic', type=int, default=200,
                       help='Number of synthetic samples to generate')
    args = parser.parse_args()

    # ディレクトリ設定
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data_dir
    output_dir = base_dir / args.output_dir

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Cup with Handle Pattern - ML Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # 訓練データの準備
    print("\n[1/4] Preparing training data...")
    examples, stats = prepare_training_data(
        str(data_dir),
        generate_synthetic=args.generate_synthetic,
        n_synthetic=args.n_synthetic,
    )
    print(f"Total samples: {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failure: {stats['failure']}")

    if len(examples) < 10:
        print("\nInsufficient training data. Please run backtest first or use --generate-synthetic.")
        return

    # データ分割
    print("\n[2/4] Splitting data...")
    generator = MLDataGenerator(output_dir=str(data_dir))
    train_examples, val_examples, test_examples = generator.get_train_val_test_split(examples)
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")

    # 特徴量準備
    print("\n[3/4] Extracting features...")
    feature_extractor = FeatureExtractor()

    X_train, y_train, feature_names = prepare_features(train_examples, feature_extractor)
    X_val, y_val, _ = prepare_features(val_examples, feature_extractor)
    X_test, y_test, _ = prepare_features(test_examples, feature_extractor)

    print(f"Feature dimensions: {X_train.shape[1]}")

    # モデル訓練
    print("\n[4/4] Training models...")
    gbm_model = None
    cnn_model = None

    # GBM訓練
    if not args.no_gbm:
        gbm_model = train_gbm_model(
            X_train, y_train,
            X_val, y_val,
            feature_names,
            str(output_dir),
        )

    # CNN訓練
    if not args.no_cnn:
        cnn_model = train_cnn_model(
            str(data_dir),
            str(output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    # アンサンブル設定の保存
    ensemble_config = {
        'cnn_weight': 0.6 if cnn_model else 0.0,
        'gbm_weight': 0.4 if gbm_model else 1.0,
        'threshold': 0.5,
        'trained_at': datetime.now().isoformat(),
        'train_samples': len(train_examples),
        'val_samples': len(val_examples),
        'test_samples': len(test_examples),
    }

    config_path = output_dir / 'ensemble_config.json'
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Models saved to: {output_dir}")
    print(f"  - GBM: {'Yes' if gbm_model else 'No'}")
    print(f"  - CNN: {'Yes' if cnn_model else 'No'}")


if __name__ == '__main__':
    main()
