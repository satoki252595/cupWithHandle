# Cup with Handle Pattern Detector - Docker Image
# ===============================================
# Python 3.11 + PyTorch CPU + LightGBM
#
# Build: docker build -t cup-handle-detector .
# Run:   docker run -it cup-handle-detector python main.py --sample nikkei

FROM python:3.11-slim-bookworm

LABEL maintainer="Cup Handle Detector"
LABEL description="Cup with Handle Pattern Detection Tool for Japanese Stocks with ML"

# 環境変数
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# システム依存パッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# 依存パッケージを先にインストール（キャッシュ効率化）
COPY requirements.txt .

# PyTorch CPU版をインストール（GPU不要のため軽量化）
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# その他の依存パッケージ
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 出力ディレクトリを作成
RUN mkdir -p /app/output /app/ml/data /app/ml/models

# 非rootユーザーを作成（セキュリティ）
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# デフォルトコマンド
CMD ["python", "main.py", "--help"]
