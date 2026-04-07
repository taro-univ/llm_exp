# 実験環境セットアップガイド

LLMの推論崩壊・相転移実験（Tower of Hanoi）を再現するための環境構成と構築手順をまとめる。

---

## 1. ハードウェア構成

| 項目 | 内容 |
|------|------|
| CPU | Intel Core Ultra 7 265F |
| RAM | 8 GB（WSL2 割当） |
| GPU | NVIDIA GeForce RTX 5070（VRAM 12 GB） |
| OS（ホスト） | Windows 11 Home (Build 26200) |
| OS（実験環境） | Ubuntu 24.04.4 LTS on WSL2 |

---

## 2. ソフトウェア構成

### 2-1. WSL2 / Ubuntu

| コンポーネント | バージョン |
|---------------|-----------|
| WSL カーネル | 6.6.87.2-microsoft-standard-WSL2 |
| Ubuntu | 24.04.4 LTS (Noble) |
| Python | 3.12.3（システム python3） |

> **注意**: 仮想環境（`.venv`）は**存在しない**。  
> システムの `python3` / `pip3` を直接使用している。

### 2-2. Python パッケージ（主要なもの）

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| openai | 2.30.0 | Ollama OpenAI互換API経由でLLMを呼び出す |
| matplotlib | 3.10.8 | 相図・ポテンシャル地形の可視化 |
| numpy | 2.4.4 | 軌跡の補間・統計処理 |
| pillow | 12.2.0 | 画像処理（matplotlib の依存） |
| pydantic | 2.12.5 | openai ライブラリの依存 |
| httpx | 0.28.1 | openai ライブラリの依存 |

### 2-3. Ollama

| 項目 | 内容 |
|------|------|
| バージョン | 0.20.2 |
| インストール先 | WSL2（Ubuntu）内 |
| エンドポイント | `http://localhost:11434` |
| 使用モデル | `deepseek-r1:14b`（9.0 GB） |
| GPU推論 | RTX 5070 経由（CUDA 13.1 / Driver 591.86） |

### 2-4. Claude Code（AI アシスタント）

| 項目 | 内容 |
|------|------|
| 起動方法 | Windows 側の Git Bash から `cd C:\Users\shona\llm_exp && claude` |
| 実行モデル | claude-sonnet-4-6 |
| 作業ディレクトリ | `C:\Users\shona\llm_exp`（Windows側） |
| Python実行 | WSL2経由でコマンドを発行 |

---

## 3. ディレクトリ構成

```
C:\Users\shona\llm_exp\         ← Git Bash / Claude Code の作業ディレクトリ
│                                  WSL2 からは /mnt/c/Users/shona/llm_exp/ として見える
│
├── hanoi_phase_diagram.py      ← メイン実験スクリプト
├── plot_10trial.py             ← 可視化のみ行うスクリプト（保存済みJSONから再描画）
├── cot_simulation.py           ← CoT シミュレーション
├── llm_physics_sim.py          ← LLM×物理シミュレーション
├── percolation_jamming.py      ← パーコレーション・ジャミング実験
│
├── first_experiment.md         ← 実験仕様書（ハノイ相転移）
├── spec_llm_physics_sim.md     ← LLM物理シミュレーション仕様
├── llm_physics_sim_extended_spec.md
├── environment_setup.md        ← 本ドキュメント
│
├── experiment_results/         ← 本番実験の結果JSON（N=2〜12, trial単位）
│   ├── results_N02_trial1.json
│   ├── ...
│   └── summary.json
│
├── experiment_results_test/    ← テスト実行の結果JSON
│
├── phase_diagram.png           ← 最初期の可視化（少数trial）
├── phase_diagram_reload.png
├── phase_diagram_10trial.png   ← N=2~7: 10試行、N=8: 7試行での可視化
└── production_phase_diagram.png
```

---

## 4. 環境構築手順（ゼロから再現する場合）

### Step 1: WSL2 + Ubuntu のセットアップ

```powershell
# Windows PowerShell（管理者）で実行
wsl --install -d Ubuntu-24.04
```

インストール後、Ubuntu を起動してユーザー名・パスワードを設定する。

`/etc/wsl.conf` に以下を追記して systemd を有効化する：

```ini
[boot]
systemd=true

[user]
default=shona
```

WSL2 を再起動：

```powershell
wsl --shutdown
wsl
```

### Step 2: Python パッケージのインストール

WSL2（Ubuntu）のターミナル上で実行：

```bash
sudo apt update
sudo apt install -y python3-pip python3-dev

# 主要パッケージ
pip3 install openai matplotlib numpy pillow
```

### Step 3: Ollama のインストールとモデルの取得

```bash
# Ollama インストール（WSL2内）
curl -fsSL https://ollama.com/install.sh | sh

# デーモン起動（systemd が有効なら自動起動も可能）
ollama serve &

# deepseek-r1:14b モデルをダウンロード（約9GB）
ollama pull deepseek-r1:14b

# 動作確認
ollama list
curl http://localhost:11434/api/tags
```

> **GPU利用について**: Ollama は RTX 5070 を自動検出してGPU推論を行う。  
> NVIDIA ドライバー（Windows側 591.86以上）があれば WSL2 から `nvidia-smi` で確認できる。

### Step 4: Git リポジトリのクローン

Windows 側の Git Bash で：

```bash
cd /c/Users/shona
git clone <repository_url> llm_exp
cd llm_exp
```

### Step 5: Claude Code の起動

```bash
# Git Bash 上（Windows側）でそのまま起動
cd /c/Users/shona/llm_exp
claude
```

---

## 5. 実験の実行方法

### Pythonスクリプトの実行規則

Claude Code（Git Bash）からWSL2のPythonを呼び出す際は以下の形式を使う：

```bash
wsl -e bash -c "cd /mnt/c/Users/shona/llm_exp && python3 <スクリプト名> <引数>"
```

### テスト実行（N=2〜8、各3試行）

```bash
wsl -e bash -c "cd /mnt/c/Users/shona/llm_exp && python3 hanoi_phase_diagram.py \
  --provider ollama \
  --model deepseek-r1:14b \
  --n-min 2 --n-max 8 \
  --trials 3 \
  --output-dir experiment_results_test"
```

### 本番実験（N=2〜8、各10試行）

```bash
wsl -e bash -c "cd /mnt/c/Users/shona/llm_exp && python3 hanoi_phase_diagram.py \
  --provider ollama \
  --model deepseek-r1:14b \
  --n-min 2 --n-max 8 \
  --trials 10 \
  --save-fig production_phase_diagram.png"
```

> **所要時間の目安**:  
> - N=2〜3: 数秒〜数十秒/試行  
> - N=4〜6: 数分/試行  
> - N=7〜8: 10〜30分/試行（deepseek-r1:14b のChain-of-Thought推論のため）

### 保存済みデータから可視化のみ再実行

```bash
wsl -e bash -c "cd /mnt/c/Users/shona/llm_exp && python3 hanoi_phase_diagram.py \
  --plot-only \
  --output-dir experiment_results \
  --save-fig phase_diagram_reload.png"
```

### カスタム可視化（特定トライアル数での集計）

```bash
wsl -e bash -c "cd /mnt/c/Users/shona/llm_exp && python3 plot_10trial.py"
```

`plot_10trial.py` の `TARGET` 辞書を編集することで、Nごとに参照するtrial番号を変更できる。

---

## 6. Ollama の管理

```bash
# Ollama サービス状態確認
wsl -e bash -c "systemctl status ollama"

# 手動起動
wsl -e bash -c "ollama serve"

# モデル一覧
wsl -e bash -c "ollama list"

# モデル削除
wsl -e bash -c "ollama rm deepseek-r1:14b"

# 別モデルの追加例
wsl -e bash -c "ollama pull llama3.3:70b"
```

---

## 7. トラブルシューティング

### Ollama に接続できない

```bash
# Ollama が起動しているか確認
wsl -e bash -c "curl http://localhost:11434/api/tags"

# 起動していなければ
wsl -e bash -c "ollama serve &"
```

### タイムアウトエラー（大きいNで停止）

`hanoi_phase_diagram.py` の `_ollama_timeout` メソッドを修正する：

```python
@staticmethod
def _ollama_timeout(N: int) -> float:
    # 上限を 900s → 3600s に変更する例
    return min(3600.0, max(120.0, 30.0 * (2 ** max(0, N - 3))))
```

### WSL2 のメモリ不足

`C:\Users\shona\.wslconfig` を作成して制限を設定：

```ini
[wsl2]
memory=12GB
processors=8
```

```powershell
wsl --shutdown
```

---

## 8. 実験結果の確認

```bash
# サマリーの確認
wsl -e bash -c "cat /mnt/c/Users/shona/llm_exp/experiment_results/summary.json | python3 -m json.tool"

# 特定試行の詳細確認
wsl -e bash -c "cat /mnt/c/Users/shona/llm_exp/experiment_results/results_N05_trial1.json | python3 -m json.tool"
```

---

*最終更新: 2026-04-07*
