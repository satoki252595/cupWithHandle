"""
CNN Model for Chart Pattern Classification
==========================================
ResNet/EfficientNetベースの転移学習モデルで
チャート画像からカップ・ウィズ・ハンドルパターンの成功/失敗を予測する。
"""

import os
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


class ChartImageDataset(Dataset):
    """
    チャート画像データセット

    Args:
        image_paths: 画像ファイルパスのリスト
        labels: ラベル（0 or 1）のリスト
        transform: 画像変換
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or self._get_default_transform()

    def _get_default_transform(self) -> transforms.Compose:
        """デフォルトの画像変換"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # 画像読み込み
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class ChartCNN(nn.Module):
    """
    チャートパターン分類用CNNモデル

    ResNet18をベースに転移学習を行う。
    最終層を2クラス分類に置き換え。
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        backbone: str = 'resnet18',  # 'resnet18', 'resnet34', 'efficientnet_b0'
    ):
        super(ChartCNN, self).__init__()

        self.backbone_name = backbone

        if backbone == 'resnet18':
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(256, num_classes)
            )

        elif backbone == 'resnet34':
            self.backbone = models.resnet34(
                weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(256, num_classes)
            )

        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(256, num_classes)
            )

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """確率を出力"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs

    def freeze_backbone(self):
        """バックボーンの重みを固定（ファインチューニング初期段階用）"""
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 最終層のみ学習可能に
        if self.backbone_name.startswith('resnet'):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self):
        """バックボーンの重みを解放（全層ファインチューニング用）"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class CNNTrainer:
    """
    CNNモデルの訓練クラス
    """

    def __init__(
        self,
        model: ChartCNN,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """検証"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        モデルを訓練

        Args:
            train_loader: 訓練データローダー
            val_loader: 検証データローダー
            epochs: エポック数
            early_stopping_patience: 早期停止のパティエンス
            save_path: モデル保存先

        Returns:
            訓練履歴
        """
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Training on {self.device}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print("-" * 50)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

            # 早期停止チェック
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                if save_path:
                    self.save_model(save_path)
                    print(f"  -> Model saved (best val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }

    def save_model(self, path: str):
        """モデルを保存"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }, path)

    def load_model(self, path: str):
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])


def get_data_transforms(augment: bool = True) -> Dict[str, transforms.Compose]:
    """
    データ変換を取得

    Args:
        augment: データ拡張を行うか

    Returns:
        {'train': ..., 'val': ...} の辞書
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    return {
        'train': train_transform,
        'val': val_transform,
    }


def predict_from_image(
    model: ChartCNN,
    image_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[int, float]:
    """
    単一画像から予測

    Args:
        model: 学習済みモデル
        image_path: 画像パス
        device: デバイス

    Returns:
        (予測クラス, 確率) のタプル
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_class].item()

    return pred_class, pred_prob


if __name__ == '__main__':
    # テスト
    print("Testing ChartCNN...")

    # モデル作成
    model = ChartCNN(num_classes=2, pretrained=True, backbone='resnet18')
    print(f"Model created: {model.backbone_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ダミー入力でフォワードパステスト
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # 確率出力テスト
    probs = model.predict_proba(dummy_input)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sum of probs: {probs.sum(dim=1)}")  # 各サンプルで1になるはず
