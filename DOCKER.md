# Docker環境セットアップガイド

Cup with Handle パターン検出ツールをDockerで実行するためのガイドです。

---

## 前提条件

### Windows (WSL2)
1. **Docker Desktop for Windows** をインストール
   - https://www.docker.com/products/docker-desktop/
   - WSL2バックエンドを有効化

2. **WSL2** を有効化
   ```powershell
   wsl --install
   ```

### Linux / macOS
- Docker と Docker Compose がインストール済みであること

---

## クイックスタート

### 1. イメージをビルド

```bash
# WSL / Linux / macOS
./docker-run.sh build

# Windows (PowerShell / cmd)
docker-run.bat build
```

または直接:
```bash
docker-compose build
```

### 2. サンプルテスト

```bash
# 日経225サンプル銘柄でテスト
./docker-run.sh sample nikkei

# Windows
docker-run.bat sample nikkei
```

### 3. 特定銘柄をスキャン

```bash
# トヨタ、ソニー、東京エレクトロン
./docker-run.sh scan 7203 6758 8035

# Windows
docker-run.bat scan 7203 6758 8035
```

---

## コマンド一覧

| コマンド | 説明 | 例 |
|---------|------|-----|
| `build` | Dockerイメージをビルド | `./docker-run.sh build` |
| `scan <symbols>` | 指定銘柄をスキャン | `./docker-run.sh scan 7203 6758` |
| `sample [type]` | サンプル銘柄でテスト | `./docker-run.sh sample nikkei` |
| `ml <symbols>` | ML予測モード | `./docker-run.sh ml 7203` |
| `full` | 全銘柄スクリーニング | `./docker-run.sh full` |
| `train` | MLモデルを訓練 | `./docker-run.sh train` |
| `backtest` | バックテスト実行 | `./docker-run.sh backtest` |
| `evaluate` | モデル評価 | `./docker-run.sh evaluate` |
| `shell` | 対話型シェル | `./docker-run.sh shell` |
| `clean` | コンテナ・イメージ削除 | `./docker-run.sh clean` |

---

## 詳細な使い方

### ML訓練パイプライン

```bash
# 1. 合成データでMLモデルを訓練
./docker-run.sh train --generate-synthetic --n-synthetic 300

# 2. モデルを評価
./docker-run.sh evaluate

# 3. ML予測モードでスキャン
./docker-run.sh ml 7203 6758 8035
```

### バックテスト

```bash
# バックテストを実行（100銘柄 × 16パラメータ）
./docker-run.sh backtest

# 結果は ./output/ に保存される
```

### 全銘柄スクリーニング

```bash
# 全日本株をスクリーニング（時間がかかります）
./docker-run.sh full

# 結果は ./output/cup_handle_*.csv に保存
```

---

## docker-compose 直接使用

```bash
# スキャナー
docker-compose run --rm scanner 7203 6758 8035
docker-compose run --rm scanner --sample nikkei
docker-compose run --rm scanner --ml-mode 7203

# ML訓練
docker-compose run --rm trainer --generate-synthetic

# バックテスト
docker-compose run --rm backtest

# 評価
docker-compose run --rm evaluator

# 対話型シェル
docker-compose run --rm shell
```

---

## ボリュームマウント

以下のディレクトリがホストとコンテナ間で共有されます：

| ホスト | コンテナ | 説明 |
|--------|---------|------|
| `./output/` | `/app/output/` | スキャン結果CSV、レポート |
| `./ml/models/` | `/app/ml/models/` | 訓練済みMLモデル |
| `./ml/data/` | `/app/ml/data/` | 訓練データ、画像 |

---

## リソース設定

`docker-compose.yml` でCPU/メモリ制限を調整できます：

```yaml
services:
  scanner:
    deploy:
      resources:
        limits:
          cpus: '8'      # 最大CPU数
          memory: 16G    # 最大メモリ
        reservations:
          cpus: '2'      # 最小CPU数
          memory: 4G     # 最小メモリ
```

AMD Ryzen AI 9 HX 370 (12コア/24スレッド) + 64GB RAM の場合、以下を推奨：

| サービス | CPU | メモリ |
|---------|-----|--------|
| scanner | 4-8 | 8-16GB |
| trainer | 8-12 | 16-32GB |
| backtest | 6-8 | 8-16GB |

---

## トラブルシューティング

### ビルドが遅い
```bash
# BuildKitを有効化（高速化）
DOCKER_BUILDKIT=1 docker-compose build
```

### メモリ不足
```bash
# Docker Desktopの設定でWSL2のメモリを増やす
# Settings > Resources > WSL Integration
```

### yfinanceでエラー
- API制限に達している可能性があります
- `--delay` オプションでリクエスト間隔を調整:
```bash
docker-compose run --rm scanner --delay 0.5 7203
```

### 権限エラー（Linux）
```bash
# 出力ディレクトリの権限を設定
chmod -R 777 ./output ./ml/models ./ml/data
```

---

## 開発者向け

### カスタムDockerfileでビルド

```bash
docker build -t cup-handle-detector:custom -f Dockerfile .
```

### イメージサイズの確認

```bash
docker images cup-handle-detector
```

### ログの確認

```bash
docker-compose logs -f scanner
```

### コンテナ内でデバッグ

```bash
docker-compose run --rm shell
# コンテナ内で
python -c "from cup_handle_detector import CupHandleDetector; print('OK')"
```

---

## WSL2での注意点

1. **ファイルシステム**: WSL2内のファイル（`/home/user/...`）を使用すると高速
2. **Windows側のファイル**: `/mnt/c/...` は遅いため避ける
3. **Docker Desktop**: WSL2統合を有効化すること

推奨ディレクトリ構成:
```
# WSL2内
~/projects/cup-handle-detector/
├── Dockerfile
├── docker-compose.yml
├── output/
├── ml/
└── ...
```

---

## 参考リンク

- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- [WSL2 インストールガイド](https://docs.microsoft.com/ja-jp/windows/wsl/install)
- [Docker Compose リファレンス](https://docs.docker.com/compose/compose-file/)
