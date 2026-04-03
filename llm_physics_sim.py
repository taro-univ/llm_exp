"""
LLM推論崩壊のポテンシャル井戸シミュレータ
Apple論文に着想を得た、LRM思考プロセスを非線形力学系でモデリングする可視化スクリプト
"""

import re
import math
import threading
import time
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from collections import deque


# ---------------------------------------------------------------------------
# ① TowerOfHanoiEnv クラス
# ---------------------------------------------------------------------------

class TowerOfHanoiEnv:
    """
    ハノイの塔の環境シミュレータ。
    円盤数 N を管理し、LLMへのプロンプト生成と状態評価を担う。
    """

    def __init__(self, N: int):
        self.N = N
        self.initial_state = {
            'A': list(range(N, 0, -1)),  # [N, N-1, ..., 1]
            'B': [],
            'C': [],
        }
        self.goal_state = {
            'A': [],
            'B': [],
            'C': list(range(N, 0, -1)),
        }
        self.min_moves = (2 ** N) - 1

    def get_prompt(self) -> str:
        """LLMに与える初期プロンプトを生成して返す。"""
        initial_str = self._state_to_str(self.initial_state)
        goal_str = self._state_to_str(self.goal_state)
        return (
            f"You are an AI solving the Tower of Hanoi puzzle.\n\n"
            f"[Rules]\n"
            f"1. There are 3 pegs (A, B, C) and {self.N} disks.\n"
            f"2. Only the top disk of a peg can be moved at a time.\n"
            f"3. A larger disk cannot be placed on a smaller disk.\n\n"
            f"[Initial State]\n{initial_str}\n\n"
            f"[Goal State]\n{goal_str}\n\n"
            f"[Output Format]\n"
            f'Output each step as "Move <disk_number> from <peg> to <peg>" on its own line.\n'
            f"Example: Move 1 from A to C\n\n"
            f"Solve in the minimum number of moves ({self.min_moves} moves). Begin:\n"
        )

    def evaluate_state(self, current_moves: list) -> float:
        """
        LLMが生成した手のリストを受け取り、目標状態への近さを -1.0〜1.0 で返す。
        """
        state = {k: list(v) for k, v in self.initial_state.items()}
        illegal_count = 0

        for move_str in current_moves:
            parsed = self._parse_move(move_str)
            if parsed is None:
                continue
            disk, src, dst = parsed
            if not self._apply_move(state, disk, src, dst):
                illegal_count += 1

        if illegal_count > 0:
            return -min(1.0, illegal_count * 0.2)

        return self._compute_progress(state)

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _state_to_str(self, state: dict) -> str:
        lines = []
        for peg in ('A', 'B', 'C'):
            disks = state[peg]
            disk_str = ', '.join(str(d) for d in disks) if disks else '(empty)'
            lines.append(f"  Peg {peg}: [{disk_str}]")
        return '\n'.join(lines)

    def _parse_move(self, move_str: str):
        m = re.search(r'Move\s+(\d+)\s+from\s+([ABC])\s+to\s+([ABC])', move_str, re.IGNORECASE)
        if m:
            return int(m.group(1)), m.group(2).upper(), m.group(3).upper()
        return None

    def _apply_move(self, state: dict, disk: int, src: str, dst: str) -> bool:
        src_stack = state.get(src)
        dst_stack = state.get(dst)
        if src_stack is None or dst_stack is None or not src_stack:
            return False
        if src_stack[-1] != disk:
            return False
        if dst_stack and dst_stack[-1] < disk:
            return False
        dst_stack.append(src_stack.pop())
        return True

    def _compute_progress(self, state: dict) -> float:
        goal_c = self.goal_state['C']
        current_c = state['C']
        correct = 0
        for i, disk in enumerate(goal_c):
            if i < len(current_c) and current_c[i] == disk:
                correct += 1
            else:
                break
        if correct == self.N:
            return 1.0
        return (correct / self.N) * 0.9


# ---------------------------------------------------------------------------
# ③ PotentialPhysics クラス
# ---------------------------------------------------------------------------

class PotentialPhysics:
    """
    亜臨界ピッチフォーク分岐モデル。

    V(x) = -(r/2)*x^2 - (x^4)/4 + (x^6)/6
    dx/dt = -dV/dx + F_LLM + noise
    """

    def __init__(self, N: int, dt: float = 0.05, noise_scale: float = 0.05):
        self.N = N
        self.dt = dt
        self.noise_scale = noise_scale
        # r: N が小さい → r > 0 → x=±1 の井戸が深い
        #    N が大きい → r < 0 → x=0 付近が安定（分岐消失）
        self.r = 1.5 - 0.25 * N
        self.x = 0.0
        self._f_llm: float = 0.0
        self.history: deque = deque(maxlen=600)
        self.time: float = 0.0

    def V(self, x: float) -> float:
        return -(self.r / 2) * x**2 - (x**4) / 4 + (x**6) / 6

    def dVdx(self, x: float) -> float:
        return -self.r * x - x**3 + x**5

    def V_array(self, xs: np.ndarray) -> np.ndarray:
        return -(self.r / 2) * xs**2 - (xs**4) / 4 + (xs**6) / 6

    def set_llm_force(self, score: float, force_scale: float = 0.3):
        self._f_llm = score * force_scale

    def step(self) -> float:
        noise = np.random.normal(0, self.noise_scale)
        dxdt = -self.dVdx(self.x) + self._f_llm + noise
        self.x = float(np.clip(self.x + dxdt * self.dt, -1.5, 1.5))
        self.time += self.dt
        self.history.append((self.time, self.x))
        return self.x

    def get_potential_curve(self, x_range=(-1.5, 1.5), n_points=300):
        xs = np.linspace(x_range[0], x_range[1], n_points)
        return xs, self.V_array(xs)

    def get_history_arrays(self):
        if not self.history:
            return np.array([]), np.array([])
        times, xs = zip(*self.history)
        return np.array(times), np.array(xs)


# ---------------------------------------------------------------------------
# ② LLMReasoner クラス
# ---------------------------------------------------------------------------

class LLMReasoner:
    """
    LLM APIと通信し、思考プロセスをストリーミングで取得するクラス。
    対応プロバイダ: anthropic / openai / demo
    バックグラウンドスレッドで動作し、スレッドセーフに状態を共有する。
    """

    _MOVE_RE = re.compile(r'Move\s+(\d+)\s+from\s+([ABC])\s+to\s+([ABC])', re.IGNORECASE)

    def __init__(
        self,
        env: TowerOfHanoiEnv,
        provider: str = 'demo',
        model: str = None,
        api_key: str = None,
    ):
        self.env = env
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key

        self._lock = threading.Lock()
        self._current_moves: list = []
        self._latest_text: str = ""
        self._done: bool = False

    def start(self):
        """バックグラウンドスレッドで推論を開始する。"""
        t = threading.Thread(target=self._run, daemon=True, name="LLMReasoner")
        t.start()

    def get_state(self) -> tuple:
        """スレッドセーフに (moves, latest_text, done) を返す。"""
        with self._lock:
            return list(self._current_moves), self._latest_text, self._done

    # ------------------------------------------------------------------
    # スレッド内部
    # ------------------------------------------------------------------

    def _run(self):
        try:
            if self.provider == 'demo':
                self._run_demo()
            elif self.provider == 'anthropic':
                self._run_anthropic()
            elif self.provider == 'openai':
                self._run_openai()
            elif self.provider == 'gemini':
                self._run_gemini()
            else:
                raise ValueError(f"Unknown provider: {self.provider!r}")
        except Exception as e:
            with self._lock:
                self._latest_text += f"\n[ERROR] {type(e).__name__}: {e}"
                self._done = True

    def _push(self, accumulated: str):
        """テキストをパースして共有状態を更新する（Lock保護）。"""
        matches = self._MOVE_RE.findall(accumulated)
        move_strs = [f"Move {d} from {s} to {t}" for d, s, t in matches]
        with self._lock:
            self._current_moves = move_strs
            self._latest_text = accumulated

    # ------------------------------------------------------------------
    # Demo モード
    # ------------------------------------------------------------------

    def _run_demo(self):
        """
        APIなしで動作するデモ。
        N > 4 の場合はエラーを注入してコラプスを演出する。
        """
        N = self.env.N
        optimal = self._solve_hanoi(N, 'A', 'C', 'B')
        total = len(optimal)

        # エラー注入数: N<=4 → 0, N=6 → 2, N=8 → 4
        n_errors = max(0, N - 4)
        # エラー挿入位置: 全体の 1/3〜1/2 付近
        error_at = {total // 3 + i * 2 for i in range(n_errors)}

        accumulated = (
            f"[Demo Mode]  N={N}  minimum moves={total}\n"
            f"r={self.env.N * -0.25 + 1.5:.2f}  "
            f"error_injections={n_errors}\n\n"
        )
        self._push(accumulated)
        time.sleep(0.6)

        injected = 0
        for i, move in enumerate(optimal):
            # エラー注入
            if i in error_at and injected < n_errors:
                bad = self._bad_move(i)
                accumulated += f"  [thinking...] {bad}\n"
                self._push(accumulated)
                time.sleep(0.5)
                injected += 1

            accumulated += f"{move}\n"
            self._push(accumulated)

            # N が大きいほど思考に時間がかかる演出
            delay = 0.25 + 0.04 * N + np.random.uniform(0, 0.15)
            time.sleep(delay)

        accumulated += "\n[Completed]"
        self._push(accumulated)
        with self._lock:
            self._done = True

    def _solve_hanoi(self, n: int, src: str, dst: str, aux: str) -> list:
        if n == 0:
            return []
        return (
            self._solve_hanoi(n - 1, src, aux, dst)
            + [f"Move {n} from {src} to {dst}"]
            + self._solve_hanoi(n - 1, aux, dst, src)
        )

    def _bad_move(self, idx: int) -> str:
        """現在の盤面で非合法となる手を生成する（大円盤の強制移動）。"""
        pegs = ['A', 'B', 'C']
        src = pegs[idx % 3]
        dst = pegs[(idx + 2) % 3]
        return f"Move {self.env.N} from {src} to {dst}"

    # ------------------------------------------------------------------
    # Anthropic モード
    # ------------------------------------------------------------------

    def _run_anthropic(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic  が必要です")

        api_key = self.api_key or os.environ.get('ANTHROPIC_API_KEY')
        model = self.model or 'claude-haiku-4-5-20251001'

        client = anthropic.Anthropic(api_key=api_key)
        accumulated = ""

        with client.messages.stream(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": self.env.get_prompt()}],
        ) as stream:
            for chunk in stream.text_stream:
                accumulated += chunk
                self._push(accumulated)

        with self._lock:
            self._done = True

    # ------------------------------------------------------------------
    # Gemini モード
    # ------------------------------------------------------------------

    def _run_gemini(self):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError("pip install google-genai  が必要です")

        api_key = self.api_key or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY 環境変数が設定されていません")
        model_name = self.model or 'gemini-2.5-flash'

        client = genai.Client(api_key=api_key)
        accumulated = ""

        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=self.env.get_prompt(),
        ):
            try:
                text = chunk.text
            except Exception:
                continue
            if text:
                accumulated += text
                self._push(accumulated)

        with self._lock:
            self._done = True

    # ------------------------------------------------------------------
    # OpenAI モード
    # ------------------------------------------------------------------

    def _run_openai(self):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai  が必要です")

        api_key = self.api_key or os.environ.get('OPENAI_API_KEY')
        model = self.model or 'gpt-4o'

        client = openai.OpenAI(api_key=api_key)
        accumulated = ""

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": self.env.get_prompt()}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                accumulated += delta
                self._push(accumulated)

        with self._lock:
            self._done = True


# ---------------------------------------------------------------------------
# ④ Visualizer クラス
# ---------------------------------------------------------------------------

class Visualizer:
    """
    matplotlib FuncAnimation を用いたリアルタイムアニメーション。

    上部パネル: ポテンシャル曲線 V(x) と粒子
    下部パネル: 時系列軌道 x(t)
    ステータス行: LLM の直近出力と現在スコア
    """

    _WINDOW_SECS = 30.0   # 軌道グラフの表示時間幅

    def __init__(self, physics: PotentialPhysics, env: TowerOfHanoiEnv, reasoner: LLMReasoner):
        self.physics = physics
        self.env = env
        self.reasoner = reasoner
        self._score = 0.0
        self._move_count = 0
        self._setup_figure()

    # ------------------------------------------------------------------
    # 図のセットアップ
    # ------------------------------------------------------------------

    def _setup_figure(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(11, 8))
        self.fig.patch.set_facecolor('#0a0a1a')

        N, r = self.env.N, self.physics.r
        self.fig.suptitle(
            f"LLM Reasoning Collapse Simulator  |  N={N} disks  |  r={r:+.2f}",
            color='#d0d0ff', fontsize=13, fontweight='bold', y=0.97
        )

        gs = gridspec.GridSpec(
            3, 1, figure=self.fig,
            height_ratios=[2.2, 1.8, 0.45],
            hspace=0.48, top=0.92, bottom=0.07
        )

        # ---- 上部: ポテンシャル景観 ----------------------------------------
        self.ax_pot = self.fig.add_subplot(gs[0])
        self.ax_pot.set_facecolor('#0a0a1a')

        xs, Vs = self.physics.get_potential_curve()
        v_min, v_max = float(Vs.min()), float(Vs.max())
        v_span = max(v_max - v_min, 0.1)
        self._pot_ylim = (v_min - 0.12 * v_span, v_max + 0.25 * v_span)

        self.ax_pot.plot(xs, Vs, color='#00cfff', lw=2.0)
        self.ax_pot.axvspan( 0.65, 1.55, alpha=0.07, color='#00ff00')
        self.ax_pot.axvspan(-1.55, -0.65, alpha=0.07, color='#ff0000')
        self.ax_pot.axvline(x=0, color='#444466', ls=':', lw=0.8)

        y_lbl = self._pot_ylim[1] - 0.06 * v_span
        self.ax_pot.text( 1.1, y_lbl, 'Overthinking\n(x ≈ +1)',
                          color='#66ff99', fontsize=8, ha='center', va='top')
        self.ax_pot.text(-1.1, y_lbl, 'Fixation\n(x ≈ −1)',
                          color='#ff6666', fontsize=8, ha='center', va='top')

        self.ax_pot.set_xlim(-1.58, 1.58)
        self.ax_pot.set_ylim(*self._pot_ylim)
        self.ax_pot.set_xlabel('x  (correctness of reasoning)', color='#9999bb', fontsize=9)
        self.ax_pot.set_ylabel('V(x)', color='#9999bb', fontsize=9)
        self.ax_pot.tick_params(colors='#9999bb', labelsize=8)
        for sp in self.ax_pot.spines.values():
            sp.set_edgecolor('#2a2a4a')

        # 粒子ドット
        px = self.physics.x
        self.particle_dot, = self.ax_pot.plot(
            [px], [self.physics.V(px)], 'o',
            color='#ffff00', markersize=14, zorder=10,
            markeredgecolor='white', markeredgewidth=1.2
        )
        # スコアテキスト
        self.score_text = self.ax_pot.text(
            0.02, 0.97, 'Score: 0.00 | Moves: 0',
            transform=self.ax_pot.transAxes,
            color='#ffff88', fontsize=9, va='top', ha='left', family='monospace'
        )

        # ---- 下部: 時系列軌道 -----------------------------------------------
        self.ax_traj = self.fig.add_subplot(gs[1])
        self.ax_traj.set_facecolor('#0a0a1a')

        self.ax_traj.axhspan( 0.65, 1.6,  alpha=0.07, color='#00ff00')
        self.ax_traj.axhspan(-1.6, -0.65, alpha=0.07, color='#ff0000')
        self.ax_traj.axhline(y= 1.0, color='#44aa55', ls='--', lw=0.9, alpha=0.6)
        self.ax_traj.axhline(y=-1.0, color='#aa4444', ls='--', lw=0.9, alpha=0.6)
        self.ax_traj.axhline(y= 0.0, color='#333355', ls=':',  lw=0.8)

        self.ax_traj.set_ylim(-1.65, 1.65)
        self.ax_traj.set_xlabel('Time', color='#9999bb', fontsize=9)
        self.ax_traj.set_ylabel('x(t)', color='#9999bb', fontsize=9)
        self.ax_traj.set_title('Reasoning Trajectory', color='#ccccee', fontsize=10, pad=4)
        self.ax_traj.tick_params(colors='#9999bb', labelsize=8)
        for sp in self.ax_traj.spines.values():
            sp.set_edgecolor('#2a2a4a')

        self.ax_traj.text(0.995, 0.96, 'Correct', transform=self.ax_traj.transAxes,
                          color='#66ff99', fontsize=8, ha='right', va='top')
        self.ax_traj.text(0.995, 0.04, 'Fixation', transform=self.ax_traj.transAxes,
                          color='#ff6666', fontsize=8, ha='right', va='bottom')

        self.traj_line, = self.ax_traj.plot([], [], color='#ffdd44', lw=1.3, alpha=0.85)
        self.traj_head, = self.ax_traj.plot([], [], 'o', color='#ffff00', markersize=6, zorder=5)

        # ---- ステータス行 ---------------------------------------------------
        self.ax_stat = self.fig.add_subplot(gs[2])
        self.ax_stat.set_facecolor('#0a0a1a')
        self.ax_stat.axis('off')
        self.stat_text = self.ax_stat.text(
            0.5, 0.5, 'LLM starting...',
            transform=self.ax_stat.transAxes,
            ha='center', va='center', color='#888899', fontsize=8,
            family='monospace'
        )

    # ------------------------------------------------------------------
    # アニメーション更新
    # ------------------------------------------------------------------

    @staticmethod
    def _particle_color(x: float) -> str:
        """x の位置から粒子の色（#rrggbb）を返す。"""
        if x >= 0.65:
            return '#00ff88'
        if x <= -0.65:
            return '#ff4444'
        # -0.65 → 0.65 を赤→黄→緑にグラデーション
        t = (x + 0.65) / 1.3        # 0.0 → 1.0
        r_ch = int(255 * (1.0 - t))
        g_ch = int(180 * t + 200 * (1 - t))
        return f'#{r_ch:02x}{g_ch:02x}44'

    def _update(self, frame: int):
        """FuncAnimation から毎フレーム呼ばれる更新関数。"""
        # LLM 最新状態取得（メインスレッドから読むだけ）
        moves, text, done = self.reasoner.get_state()

        if moves:
            score = self.env.evaluate_state(moves)
            self.physics.set_llm_force(score)
            self._score = score
            self._move_count = len(moves)

        # 物理を複数ステップ進める（1フレーム ≈ 4 × dt の時間発展）
        for _ in range(4):
            self.physics.step()

        x = self.physics.x
        color = self._particle_color(x)

        # 粒子位置の更新
        self.particle_dot.set_data([x], [self.physics.V(x)])
        self.particle_dot.set_color(color)

        # スコアテキスト
        label = 'DONE' if done else 'Running...'
        self.score_text.set_text(
            f"Score: {self._score:+.3f}  |  Moves: {self._move_count}  |  {label}"
        )

        # 軌道グラフの更新
        times, xs = self.physics.get_history_arrays()
        if len(times) > 1:
            t_now = times[-1]
            self.traj_line.set_data(times, xs)
            self.traj_head.set_data([t_now], [xs[-1]])
            self.traj_head.set_color(color)
            x_min = max(0.0, t_now - self._WINDOW_SECS)
            self.ax_traj.set_xlim(x_min, t_now + 0.5)

        # ステータステキスト（直近2行 or "waiting"）
        if text:
            lines = [ln for ln in text.strip().split('\n') if ln.strip()]
            snippet = '  |  '.join(lines[-2:])[:130]
        else:
            snippet = 'Waiting for LLM...'
        self.stat_text.set_text(snippet)

        return (self.particle_dot, self.traj_line,
                self.traj_head, self.score_text, self.stat_text)

    # ------------------------------------------------------------------
    # 実行
    # ------------------------------------------------------------------

    def run(self):
        """アニメーションを開始する（ウィンドウが閉じられるまでブロック）。"""
        self._anim = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=50,           # 50 ms ≈ 20 FPS
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


# ---------------------------------------------------------------------------
# main 関数
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  LLM Reasoning Collapse Simulator")
    print("  ポテンシャル井戸で LLM の思考崩壊を可視化する")
    print("=" * 60)

    # 円盤数の入力
    while True:
        try:
            raw = input("\n円盤の数 N を入力してください (2〜8): ").strip()
            N = int(raw)
            if 2 <= N <= 8:
                break
            print("  2〜8 の整数を入力してください。")
        except ValueError:
            print("  整数を入力してください。")

    # プロバイダ選択
    print("\nLLM プロバイダを選択してください:")
    print("  demo      : API不要・シミュレーション（推奨）")
    print("  gemini    : Gemini API   (GEMINI_API_KEY)")
    print("  anthropic : Claude API   (ANTHROPIC_API_KEY)")
    print("  openai    : OpenAI API   (OPENAI_API_KEY)")
    provider = input("プロバイダ [demo]: ").strip().lower() or 'demo'

    api_key = None
    model = None
    if provider == 'gemini':
        env_var = 'GEMINI_API_KEY'
        default_model = 'gemini-2.5-flash'
        if not os.environ.get(env_var):
            api_key = input(f"API キーを入力してください ({env_var}): ").strip() or None
        model_in = input(f"モデル名 [デフォルト: {default_model}]: ").strip()
        model = model_in or default_model
    elif provider in ('anthropic', 'openai'):
        env_var = 'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY'
        default_model = 'claude-haiku-4-5-20251001' if provider == 'anthropic' else 'gpt-4o'
        if not os.environ.get(env_var):
            api_key = input(f"API キーを入力してください ({env_var}): ").strip() or None
        model_in = input(f"モデル名 [デフォルト: {default_model}]: ").strip()
        model = model_in or default_model

    print(f"\n  N={N}  provider={provider}  r={1.5 - 0.25 * N:+.2f}")
    print("  ウィンドウを閉じるとシミュレーションを終了します。\n")

    # 各コンポーネントの初期化
    env = TowerOfHanoiEnv(N)
    physics = PotentialPhysics(N)
    reasoner = LLMReasoner(env, provider=provider, model=model, api_key=api_key)
    viz = Visualizer(physics, env, reasoner)

    # LLM をバックグラウンドスレッドで起動
    reasoner.start()

    # アニメーション開始（ブロッキング）
    viz.run()

    # --- GIFとして保存する ---
    print("GIFを生成中... (しばらく時間がかかります)")
    # interval=50 なので、200フレームで10秒分のアニメーションになります
    viz._anim = animation.FuncAnimation(
        viz.fig, viz._update, frames=200, interval=50, blit=False
    )

    # 保存実行（ImageMagickやPillowが必要です）
    viz._anim.save('llm_reasoning_collapse.gif', writer='pillow', fps=20)
    print("保存完了: llm_reasoning_collapse.gif")


if __name__ == '__main__':
    main()
