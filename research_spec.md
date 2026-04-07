# LLM 推論崩壊の Loss Landscape 解析 — 設計仕様書

> **論文参照:** Li et al. (2018) "Visualizing the Loss Landscape of Neural Nets"  
> **ベースコード:** `hanoi_phase_diagram.py`

---

## 1. 研究背景：推論の物理学的解釈

ニューラルネットワークの学習において、特定のアーキテクチャ（スキップ接続など）が「損失関数の地形（Loss Landscape）」を平滑化し、学習を容易にすることが知られている。Li et al. (2018) は、ネットワークが深くなるにつれて、この地形が **「ほぼ凸（Nearly Convex）」な状態から「カオス的（Chaotic）」な状態へと急激に相転移する** ことを視覚的に示した。

本研究では、この知見を **LLM の推論プロセス** へと転移させる。ハノイの塔の円盤数 $N$ をネットワークの深さ（パラメータ複雑さ）と見なし、以下の現象を定量的・定性的に検証する。

| 現象 | 内容 |
|---|---|
| **推論の相転移** | $N$ が小さいときは正解への勾配が明確な「凸」な地形。臨界点 $N_c$ を超えると「カオス」な地形へ変容し、勾配が「粉砕（Shatter）」される |
| **推論の頑健性（Flatness）** | 正解近傍の地形が平坦（Flat）であるほど、LLM は多少のノイズや迷走があっても最終的に正解へ辿り着ける（汎化性能の類似概念）|

---

## 2. 数学的定式化

### 2.1 状態空間の座標化

ハノイの塔の状態 $s$ を、高次元空間内の座標 $\mathbf{x}$ としてマッピングする。  
円盤数 $N$、ペグ数 3 のとき、各状態は以下の $3N$ 次元ベクトルで表現される。

$$\mathbf{x} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_N]^\top \in \{0, 1\}^{3N}$$

ここで $\mathbf{v}_i$ は円盤 $i$ の位置を示す one-hot ベクトルである。

```
例 (N=2):
  円盤1 が Peg A にある → v_1 = [1, 0, 0]
  円盤2 が Peg C にある → v_2 = [0, 0, 1]
  → x = [1, 0, 0,  0, 0, 1]  (6次元)
```

### 2.2 推論ポテンシャル $V(\mathbf{x})$ の定義

Li et al. (2018) の損失関数 $L(\theta)$ を、**推論ポテンシャル** $V(\mathbf{x})$ に置き換える。

$$V(\mathbf{x}) = \lambda_{\text{dist}} \cdot \hat{D}(s) + \lambda_{\text{penalty}} \cdot P(s)$$

**$\hat{D}(s)$（正規化距離項）**

状態 $s$ から目標状態 $G$ までの理論上の最短手数 $D(s, G)$ に基づく項。  
Filter Normalization の思想に基づき、問題の規模によらずポテンシャルの最大値を 1 に揃える。

$$\hat{D}(s) = \frac{D(s, G)}{2^N - 1}$$

| 状態 | $\hat{D}(s)$ | $V(\mathbf{x})$ の意味 |
|---|---|---|
| 目標状態 $G$ | $0$ | 谷底（正解）|
| 初期状態 | $1$ | 高原（出発点）|

**$P(s)$（ペナルティ項）**

ルール違反（小さい円盤の上に大きい円盤を置く等）を、物理的なエネルギー障壁として定義する。  
非凸なカオス地形における「棘（spike）」として機能し、$V > 1$ となる領域を生み出す。

### 2.3 地形の視覚化手法

2次元コンター図を作成するため、以下の射影関数をプロットする（Li et al. Figure 5/6 に相当）。

$$f(\alpha, \beta) = V\!\left(s^* + \alpha \boldsymbol{\delta} + \beta \boldsymbol{\eta}\right)$$

| 変数 | 定義 |
|---|---|
| $s^*$ | 最適な推論軌跡上の点（初期状態 $x_0$ を基準に使用）|
| $\boldsymbol{\delta}$ | 初期状態 $x_0$ から目標状態 $x_G$ へ向かう**主軸（推論方向）**ベクトル |
| $\boldsymbol{\eta}$ | $\boldsymbol{\delta}$ にグラム・シュミット直交化した**直交軸**ランダムベクトル |
| $\alpha$ | 主軸方向のスカラー係数（$\alpha=0$: 初期状態、$\alpha=1$: 目標状態）|
| $\beta$ | 直交軸方向のスカラー係数（正解ルートからの「ズレ」）|

---

## 3. 実装フェーズ

### Phase 1 — 数学的基盤の構築（`TowerOfHanoiEnv` 拡張）

#### 1a. `_get_state_coord(state)` — 座標変換

盤面状態 `dict` を $3N$ 次元 numpy 配列へ変換するメソッド。

```python
def _get_state_coord(self, state: dict) -> np.ndarray:
    # 各円盤 i について属するペグを one-hot 化し連結する
    # 返値: shape=(3*N,), dtype=float
```

#### 1b. `_min_moves_from(state)` — O(N) 再帰最短手数計算

任意の盤面状態からゴール（全円盤を Peg C へ）までの最短手数を **O(N)** の再帰で求める。

**アルゴリズム核心（`_min_moves_to_peg(state, n, target)`）:**

```
D(state, n, target):
  if n == 0: return 0

  peg_of_n = disk n の現在のペグ

  if peg_of_n == target:
    # ディスク n は既定位置 → n-1 サブ問題を同じ target で解く
    return D(state, n-1, target)
  else:
    # ディスク n を target へ移す必要がある
    aux = 残りの第3ペグ
    cost_to_clear = D(state, n-1, aux)   # 1..n-1 を aux へ集める
    cost_from_aux = 2^(n-1) - 1          # aux から target へ（標準ハノイ、定数）
    return cost_to_clear + 1 + cost_from_aux
```

- 各再帰で `n` が 1 ずつ減るため **O(N)** で計算完了
- 初期状態（A）→ ゴール（C）で `D = 2^N - 1` になることで正当性を確認できる

#### 1c. `_compute_V(state)` の切り出し — 評価の中核

`evaluate_state()` から状態評価ロジックを分離し、**`state: dict` を直接受け取る**メソッドとして実装する。  
コンター図生成時に state を直接渡せる設計とするため、この分離が重要となる。

```python
def _compute_V(self, state: dict, illegal_count: int = 0) -> float:
    d_hat = self._min_moves_from(state) / self.min_moves  # Filter Normalization
    penalty = lambda_penalty * illegal_count
    return d_hat + penalty
```

#### 1d. `evaluate_state()` の刷新

既存の「Cペグ進捗カウント」評価を廃止し、`_compute_V()` を呼び出す形へ変更する。

**スコアの意味の変化:**

| | 変更前 | 変更後 |
|---|---|---|
| 目標状態 | `+1.0` | `V = 0.0`（谷底）|
| 初期状態 | `0.0` | `V = 1.0`（高原）|
| ルール違反 | 負のスコア | `V > 1.0`（エネルギー障壁）|

> ⚠️ **後方互換注意:** `score` の意味が逆転するため、既存の保存済み JSON を `--plot-only` で読み込む場合はレガシー変換の対応が必要。

---

### Phase 2 — `PotentialMapper` の更新

`map_trajectory()` の `score` フィールドが新しい `V(x)` を参照するよう更新する。  
`normalized_step`（横軸 = `step / min_moves`）は現行通りを維持する。

---

### Phase 3 — Loss Landscape 可視化の実装（`Visualizer` 拡張）

#### 3a. 2次元コンター図 `_plot_landscape_contour()`

**座標計算の手順:**

1. `x_init = _get_state_coord(initial_state)`、`x_goal = _get_state_coord(goal_state)` を取得
2. 主軸ベクトル `δ = x_goal - x_init` を計算
3. ランダムベクトル `η` を生成し、グラム・シュミット法で `δ` 成分を除去・正規化
4. `(α, β)` グリッド（例: 30×30）上で `x = x_init + α·δ + β·η` を計算
5. 各 `x` を最近傍の合法状態にスナップし `V(state)` を評価
6. `plt.contourf()` で等高線マップを描画

**離散化戦略（one-hot空間の連続補間問題への対応）:**

one-hot 座標の連続補間は意味を持たないため、以下のいずれかを採用する。

| 方式 | 内容 | 適用場面 |
|---|---|---|
| **最近傍スナップ** | 連続座標から最も近いユークリッド距離の合法状態へ対応付け | N が小さい場合（全状態列挙が可能）|
| **経路補間** | α 軸を最短経路上の具体的な状態列にマッピングし、β 軸はペグ割り当てを摂動 | N が大きい場合のスケール対応 |

**計算コストの上限設定:**

- グリッド解像度: 30×30 = 900点（固定）
- N ≥ 8 では事前に全合法状態の座標を辞書化するプリコンピュートを実施
- 全状態数は $3^N$（N=12 で約 53 万）

#### 3b. N 別コンター図による相転移の表現

N ごとに地形の変化を並べて表示することで、相転移を直観的に可視化する。

```
N = 2, 3:   なだらかなすり鉢状（Convex basin）   → 正解への一本道の谷
N = 4, 5:   底面がわずかに荒れ始める             → 勾配が弱まり始める
N ≥ Nc:     棘（局所的ポテンシャル障壁）が随所に → カオス的迷宮地形
```

#### 3c. GridSpec 再設計案

```
┌─────────────────────────────────────────────────┐
│  [N=2 Contour] [N=4 Contour] [N=8 Contour] [N=12]  ← 新規追加（上段）
├─────────────────────────────────────────────────┤
│         軌跡プロット（Filter Normalization）         ← 既存維持（中段）
├──────────────────────┬──────────────────────────┤
│     相図（Phase）     │    トークン数スケーリング    ← 既存維持（下段）
└──────────────────────┴──────────────────────────┘
```

あるいは、コンター図は `--landscape` フラグで **別 Figure として独立保存**する選択肢も有効（通常実行への影響ゼロ）。

---

## 4. 設計判断の整理

### クラス構造の変更方針

既存の `hanoi_phase_diagram.py` のクラス構造は**維持**する。変更はメソッドの追加・オーバーライドのみとし、`ExperimentRunner` や `LLMReasoner` の外部 API は変更しない。

### λ パラメータの値設定

| パラメータ | 推奨値 | 根拠 |
|---|---|---|
| `λ_dist` | `1.0` | $\hat{D}$ が `[0, 1]` を占めるよう固定 |
| `λ_penalty` | `0.5 / 違反1回` | 2回違反で $V > 1.0$（障壁がベースラインを超える）|

### 実装の依存関係と推奨順序

```
_get_state_coord()
    └─→ _min_moves_from() / _min_moves_to_peg()
            └─→ _compute_V(state)
                    ├─→ evaluate_state()  ← PotentialMapper に波及
                    └─→ _plot_landscape_contour()  ← Visualizer 拡張
```

| ステップ | 実装対象 | 依存 |
|---|---|---|
| 1 | `_get_state_coord()` | なし |
| 2 | `_min_moves_to_peg()` / `_min_moves_from()` | なし |
| 3 | `_compute_V(state)` | ステップ 1, 2 |
| 4 | `evaluate_state()` の置き換え | ステップ 3 |
| 5 | `PotentialMapper.map_trajectory()` 更新 | ステップ 4 |
| 6 | `Visualizer._plot_landscape_contour()` | ステップ 1, 3 |
| 7 | GridSpec 再設計 + `main()` への `--landscape` フラグ追加 | ステップ 6 |
