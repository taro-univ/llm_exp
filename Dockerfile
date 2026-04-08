# 1. NVIDIA公式のCUDAイメージをベースにする（RTX 5070に対応）
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 2. タイムゾーンなどの対話型プロンプトを無効化
ENV DEBIAN_FRONTEND=noninteractive

# 3. システムのセットアップ（Python3, pip, git, 各種ツール）
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel

# 4. 作業ディレクトリの設定
WORKDIR /workspace

# 5. 作成した requirements.txt をコンテナ内にコピー
COPY requirements.txt .

# 6. ライブラリのインストール
RUN pip3 install --no-cache-dir -r requirements.txt psycopg2-binary

# 7. コンテナ起動時に自動的に実行されるコマンド（任意）
CMD ["/bin/bash"]