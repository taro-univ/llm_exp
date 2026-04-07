import re
import os
import random
import json
import time


# ---------------------------------------------------------------------------
# ① TowerOfHanoiEnv クラス
# ---------------------------------------------------------------------------

class TowerOfHanoiEnv:
    """
    ハノイの塔の環境シミュレータおよび推論ポテンシャルの絶対評価。

    円盤数 N で初期化し、初期状態と目標状態を管理する。
    evaluate_state は LLM の対数尤度や自信度に一切依存せず、
    Li et al. (2018) の Filter Normalization の思想に基づく
    推論ポテンシャル V(x) を用いた物理的な絶対評価でスコアを返す。

    V(x) の意味:
        V = 0.0  … 目標状態（谷底）
        V = 1.0  … 初期状態（高原）
        V > 1.0  … ルール違反によるエネルギー障壁（棘）
    """

    # Filter Normalization の係数
    LAMBDA_DIST    = 1.0   # 距離項の重み
    LAMBDA_PENALTY = 0.5   # ルール違反1回あたりのペナルティ

    def __init__(self, N: int):
        self.N = N
        self.initial_state = {
            'A': list(range(N, 0, -1)),  # [N, N-1, ..., 1] (底が大きい)
            'B': [],
            'C': [],
        }
        self.goal_state = {
            'A': [],
            'B': [],
            'C': list(range(N, 0, -1)),  # [N, N-1, ..., 1]
        }
        self.min_moves = (2 ** N) - 1

    def get_prompt(self) -> str:
        """LLM に与える初期プロンプトを生成して返す。"""
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
        LLM が生成した手のリストを受け取り、推論ポテンシャル V(x) を返す。

        手のリストを先頭から順にシミュレートし、到達した盤面状態に対して
        _compute_V() を呼び出す。ルール違反手は状態に適用せず違反数のみ
        カウントし、LAMBDA_PENALTY として V に上乗せする。

        Returns
        -------
        float
            V(x) ∈ [0.0, ∞)
            - 目標状態に到達: 0.0
            - 初期状態のまま: 1.0
            - ルール違反あり: 1.0 を超えるエネルギー障壁値

        Note
        ----
        この評価は LLM の対数尤度・自信度に一切依存せず、
        物理的な盤面状態のみに基づく絶対評価である。
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

        return self._compute_V(state, illegal_count)

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
        """
        "Move <disk> from <src> to <dst>" 形式の文字列をパースする。
        マッチしない場合は None を返す。
        """
        m = re.search(
            r'Move\s+(\d+)\s+from\s+([ABC])\s+to\s+([ABC])',
            move_str,
            re.IGNORECASE,
        )
        if m:
            return int(m.group(1)), m.group(2).upper(), m.group(3).upper()
        return None

    def _apply_move(self, state: dict, disk: int, src: str, dst: str) -> bool:
        """
        盤面 state に対してムーブを適用する。
        合法なら True、非合法なら False を返す（state は変更しない）。
        """
        src_stack = state.get(src)
        dst_stack = state.get(dst)

        # ペグ名が不正、または移動元が空
        if src_stack is None or dst_stack is None or not src_stack:
            return False
        # 移動元の最上段が指定円盤でない
        if src_stack[-1] != disk:
            return False
        # 移動先の最上段より大きい円盤は置けない
        if dst_stack and dst_stack[-1] < disk:
            return False

        dst_stack.append(src_stack.pop())
        return True

    # ------------------------------------------------------------------
    # Phase 1 拡張: 推論ポテンシャル V(x) の計算
    # ------------------------------------------------------------------

    def _get_state_coord(self, state: dict) -> 'np.ndarray':
        """
        盤面状態を 3N 次元の one-hot ベクトルへ変換する。

        各円盤 i (1..N) について、属するペグを [is_A, is_B, is_C] で表し、
        全円盤分を連結した形で返す。

            x = [v_1, v_2, ..., v_N]  ∈ {0,1}^{3N}

        Returns
        -------
        np.ndarray
            shape=(3*N,), dtype=float64
        """
        import numpy as np
        peg_index = {'A': 0, 'B': 1, 'C': 2}
        coord = np.zeros(3 * self.N, dtype=float)
        for peg, disks in state.items():
            col = peg_index[peg]
            for disk in disks:
                # 円盤 disk (1-indexed) の one-hot 開始位置: (disk-1)*3
                coord[(disk - 1) * 3 + col] = 1.0
        return coord

    def _min_moves_from(self, state: dict) -> int:
        """
        任意の盤面状態からゴール（全円盤を Peg C へ）までの
        最短手数を O(N) の再帰で計算して返す。

        初期状態からの最短手数は 2^N - 1 = self.min_moves に一致する。
        """
        return self._min_moves_to_peg(state, self.N, 'C')

    def _min_moves_to_peg(self, state: dict, n: int, target: str) -> int:
        """
        円盤 1..n を target ペグへ積み上げるための最短手数を再帰で返す。

        アルゴリズム:
          - 円盤 n が既に target にある → n-1 のサブ問題を同じ target で解く
          - 円盤 n が別のペグにある   → 残りの第3ペグ aux を中継地として
              cost = D(state, n-1, aux)    # 1..n-1 を aux へ集める
                   + 1                     # 円盤 n を target へ移動
                   + (2^(n-1) - 1)         # 1..n-1 を aux から target へ（標準ハノイ）
        """
        if n == 0:
            return 0

        # 円盤 n が現在どのペグにあるかを特定する
        peg_of_n = None
        for peg, disks in state.items():
            if n in disks:
                peg_of_n = peg
                break

        if peg_of_n == target:
            # 円盤 n は既定位置 → n-1 サブ問題を同じ target で再帰
            return self._min_moves_to_peg(state, n - 1, target)
        else:
            # 円盤 n を target へ移す必要がある
            aux = self._third_peg(peg_of_n, target)
            cost_to_clear = self._min_moves_to_peg(state, n - 1, aux)
            cost_from_aux = (2 ** (n - 1)) - 1
            return cost_to_clear + 1 + cost_from_aux

    @staticmethod
    def _third_peg(peg1: str, peg2: str) -> str:
        """3つのペグ {A, B, C} のうち peg1・peg2 以外の1つを返す。"""
        return ({'A', 'B', 'C'} - {peg1, peg2}).pop()

    def _compute_V(self, state: dict, illegal_count: int = 0) -> float:
        """
        推論ポテンシャル V(x) を計算して返す。

            V(x) = LAMBDA_DIST * D_hat(s) + LAMBDA_PENALTY * illegal_count

            D_hat(s) = D(s, G) / (2^N - 1)   ← Filter Normalization

        Parameters
        ----------
        state : dict
            現在の盤面状態 {'A': [...], 'B': [...], 'C': [...]}.
        illegal_count : int
            シミュレーション中に発生したルール違反の累計回数。

        Returns
        -------
        float
            V ∈ [0.0, ∞)  (目標状態で 0.0、初期状態で 1.0、違反時は > 1.0)
        """
        d_hat   = self._min_moves_from(state) / self.min_moves
        penalty = self.LAMBDA_PENALTY * illegal_count
        return round(self.LAMBDA_DIST * d_hat + penalty, 6)


# ---------------------------------------------------------------------------
# ② LLMReasoner クラス
# ---------------------------------------------------------------------------

class LLMReasoner:
    """
    LLM の推論プロセス（思考トークンと手）を取得するクラス。

    対応プロバイダ: gemini / dummy / openai / anthropic

    reason(env) を呼ぶと推論を同期実行し、以下の dict を返す。
        {
            'moves'      : list[str]   # パースされた手の列（順序保持）
            'token_count': int         # 消費トークン数の推定値
            'raw_text'   : str         # LLM の生テキスト出力
        }
    """

    _MOVE_RE = re.compile(
        r'Move\s+(\d+)\s+from\s+([ABC])\s+to\s+([ABC])', re.IGNORECASE
    )

    # dummy モードで N >= FIXATION_THRESHOLD になると Fixation が起きる
    FIXATION_THRESHOLD = 7

    def __init__(
        self,
        provider: str = 'dummy',
        model: str = None,
        api_key: str = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def reason(self, env: 'TowerOfHanoiEnv') -> dict:
        """
        env で定義されたハノイの塔をプロバイダに解かせ、結果を返す。

        Returns
        -------
        dict
            'moves'      : list[str]  生成された手の列
            'token_count': int        消費トークン数
            'raw_text'   : str        生テキスト
        """
        dispatch = {
            'dummy'    : self._run_dummy,
            'anthropic': self._run_anthropic,
            'openai'   : self._run_openai,
            'gemini'   : self._run_gemini,
            'ollama'   : self._run_ollama,
        }
        if self.provider not in dispatch:
            raise ValueError(
                f"Unknown provider: {self.provider!r}. "
                f"Choose from {list(dispatch.keys())}"
            )
        return dispatch[self.provider](env)

    # ------------------------------------------------------------------
    # dummy モード
    # ------------------------------------------------------------------

    def _run_dummy(self, env: 'TowerOfHanoiEnv') -> dict:
        """
        API 不要のシミュレーション。

        N < FIXATION_THRESHOLD (Overthinking フェーズ)
            正解手列に余分な手（遠回り）を混入。トークン数は N とともに増加。

        N >= FIXATION_THRESHOLD (Fixation フェーズ)
            序盤は正しく動くが途中で同一の非合法手を繰り返す。
            「推論放棄」を模倣してトークン数が減少に転じる。
        """
        N = env.N
        optimal = self._solve_hanoi(N, 'A', 'C', 'B')

        if N < self.FIXATION_THRESHOLD:
            moves, raw_text = self._overthinking_sequence(N, optimal)
        else:
            moves, raw_text = self._fixation_sequence(N, optimal)

        token_count = self._estimate_tokens(N, moves, raw_text)
        return {'moves': moves, 'token_count': token_count, 'raw_text': raw_text}

    def _overthinking_sequence(self, N: int, optimal: list) -> tuple:
        """
        正解手列の途中に「往復」する冗長手を挿入して Overthinking を再現する。
        冗長手数は N が大きいほど増える。
        """
        # 冗長往復の回数: N=2→0, N=3→1, N=4→2, N=5→3, N=6→4
        n_detours = max(0, N - 2)
        # 挿入位置候補: 最適解の 1/4 〜 3/4 の範囲
        insert_positions = sorted(
            random.sample(
                range(len(optimal) // 4, 3 * len(optimal) // 4),
                min(n_detours, len(optimal) // 2),
            )
        ) if len(optimal) >= 4 else []

        result = []
        inserted = set()
        for i, move in enumerate(optimal):
            result.append(move)
            if i in insert_positions and i not in inserted:
                inserted.add(i)
                # 今追加した手を「往復」: dst→src に戻す手を追加
                parsed = self._parse_one(move)
                if parsed:
                    disk, src, dst = parsed
                    result.append(f"Move {disk} from {dst} to {src}")  # 戻す
                    result.append(f"Move {disk} from {src} to {dst}")  # 再び進む

        lines = [f"[Dummy/Overthinking N={N}]"] + result
        return result, '\n'.join(lines)

    def _fixation_sequence(self, N: int, optimal: list) -> tuple:
        """
        正解手列の前半を実行した後、同一の非合法手を繰り返す Fixation を再現する。
        非合法手の繰り返し数は N が大きいほど少ない（推論放棄が早まる）。
        """
        # 正解手列のうち前半 1/3 だけ出力してから固着する
        cutoff = max(1, len(optimal) // 3)
        partial = optimal[:cutoff]

        # 繰り返す非合法手: 最大の円盤を A→B に強制移動（ほぼ常に非合法）
        bad_move = f"Move {N} from A to B"
        # 繰り返し数: N=7→6, N=8→5, ..., N=12→1（大きいほど早く諦める）
        repeat = max(1, 13 - N)
        fixation_block = [bad_move] * repeat

        moves = partial + fixation_block
        lines = (
            [f"[Dummy/Fixation N={N}  cutoff={cutoff}  repeat={repeat}]"]
            + partial
            + [f"[STUCK] {bad_move}"] * repeat
        )
        return moves, '\n'.join(lines)

    def _estimate_tokens(self, N: int, moves: list, raw_text: str) -> int:
        """
        dummy モード用のトークン数推定。
        - N < FIXATION_THRESHOLD: 2^N に比例して増加（Overthinking）
        - N >= FIXATION_THRESHOLD: 減少に転じる（推論放棄）
        字数ベースの概算にランダムノイズを加える。
        """
        base_chars = len(raw_text)
        char_tokens = base_chars // 4  # 英語: 約4文字/token

        if N < self.FIXATION_THRESHOLD:
            # Overthinking: N が増えるにつれ CoT が長くなる演出
            thinking_bonus = int(150 * (2 ** (N - 2)))
        else:
            # Fixation: 崩壊後は思考を放棄するので急減
            thinking_bonus = int(150 * (2 ** (self.FIXATION_THRESHOLD - 2))
                                 * (0.6 ** (N - self.FIXATION_THRESHOLD + 1)))

        noise = random.randint(-50, 50)
        return max(50, char_tokens + thinking_bonus + noise)

    # ------------------------------------------------------------------
    # Anthropic モード
    # ------------------------------------------------------------------

    def _run_anthropic(self, env: 'TowerOfHanoiEnv') -> dict:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic  が必要です")

        api_key = self.api_key or os.environ.get('ANTHROPIC_API_KEY')
        model = self.model or 'claude-haiku-4-5-20251001'
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": env.get_prompt()}],
        )
        raw_text = response.content[0].text
        token_count = (
            response.usage.input_tokens + response.usage.output_tokens
        )
        moves = self._extract_moves(raw_text)
        return {'moves': moves, 'token_count': token_count, 'raw_text': raw_text}

    # ------------------------------------------------------------------
    # OpenAI モード
    # ------------------------------------------------------------------

    def _run_openai(self, env: 'TowerOfHanoiEnv') -> dict:
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai  が必要です")

        api_key = self.api_key or os.environ.get('OPENAI_API_KEY')
        model = self.model or 'gpt-4o'
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": env.get_prompt()}],
        )
        raw_text = response.choices[0].message.content
        token_count = (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )
        moves = self._extract_moves(raw_text)
        return {'moves': moves, 'token_count': token_count, 'raw_text': raw_text}

    # ------------------------------------------------------------------
    # Ollama モード (OpenAI 互換 API 経由)
    # ------------------------------------------------------------------

    # DeepSeek-R1 が出力する <think>...</think> ブロックにマッチする正規表現
    _THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)

    # N ごとのタイムアウト秒数: 最小 300 秒、N が増えるにつれ指数的に延長（上限 1800 秒）
    @staticmethod
    def _ollama_timeout(N: int) -> float:
        return min(1800.0, max(300.0, 60.0 * (2 ** max(0, N - 3))))

    def _run_ollama(self, env: 'TowerOfHanoiEnv') -> dict:
        """
        Ollama の OpenAI 互換エンドポイント (http://localhost:11434/v1) 経由で
        LLM を呼び出す。DeepSeek-R1:14b 等の思考モデルに対応。

        - <think>...</think> 内のテキストを「思考トークン」としてカウントする。
        - moves の抽出は <think> タグを除去した回答本体のみから行う。
        - temperature=0 を固定して再現性を確保する。
        - APITimeoutError に対して指数バックオフ付きで最大 3 回リトライする。
        """
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai  が必要です")

        model = self.model or 'deepseek-r1:14b'
        # api_key は Ollama では不要だが openai ライブラリが要求するためダミーを渡す
        api_key = self.api_key or 'ollama'
        timeout_sec = self._ollama_timeout(env.N)
        client = openai.OpenAI(
            base_url='http://localhost:11434/v1',
            api_key=api_key,
            timeout=timeout_sec,
        )

        # --- リトライループ (APITimeoutError 対応) ---
        max_retries = 3
        wait = 30  # 初回リトライ待機秒
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": env.get_prompt()}],
                    temperature=0,
                )
                break  # 成功
            except openai.APITimeoutError as e:
                last_exc = e
                if attempt == max_retries:
                    raise
                print(
                    f"    [Ollama] タイムアウト (試行 {attempt}/{max_retries})。"
                    f"{wait}秒待機後リトライ... (timeout={timeout_sec:.0f}s)"
                )
                time.sleep(wait)
                wait = min(wait * 2, 120)
            except openai.APIConnectionError as e:
                last_exc = e
                if attempt == max_retries:
                    raise
                print(
                    f"    [Ollama] 接続エラー (試行 {attempt}/{max_retries})。"
                    f"{wait}秒待機後リトライ..."
                )
                time.sleep(wait)
                wait = min(wait * 2, 120)

        raw_text = response.choices[0].message.content or ''

        # --- 思考トークンのカウント ---
        # <think> ブロック内の文字数をトークン換算（約4文字/token）して加算する
        think_blocks = self._THINK_RE.findall(raw_text)
        think_chars = sum(len(b) for b in think_blocks)
        think_tokens = think_chars // 4

        # API が usage を返す場合はそちらを優先し、思考トークンを上乗せする
        try:
            api_tokens = (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
        except (AttributeError, TypeError):
            api_tokens = len(raw_text) // 4

        token_count = api_tokens + think_tokens

        # --- moves の抽出: <think> ブロックを除去した回答本体のみを対象にする ---
        answer_text = self._THINK_RE.sub('', raw_text)
        moves = self._extract_moves(answer_text)

        return {'moves': moves, 'token_count': token_count, 'raw_text': raw_text}

    # ------------------------------------------------------------------
    # Gemini モード
    # ------------------------------------------------------------------

    def _run_gemini(self, env: 'TowerOfHanoiEnv') -> dict:
        try:
            from google import genai
            from google.genai import errors as genai_errors
        except ImportError:
            raise ImportError("pip install google-genai  が必要です")

        api_key = self.api_key or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY 環境変数が設定されていません")
        model_name = self.model or 'gemini-2.5-flash'
        client = genai.Client(api_key=api_key)

        # 指数バックオフ付きリトライ (429 / RESOURCE_EXHAUSTED 対応)
        max_retries = 5
        wait = 30  # 初回待機秒数
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=env.get_prompt(),
                )
                break  # 成功したらループを抜ける
            except genai_errors.ClientError as e:
                msg = str(e)
                if '429' in msg or 'RESOURCE_EXHAUSTED' in msg:
                    if attempt == max_retries:
                        raise
                    print(
                        f"    [Rate limit] 429/RESOURCE_EXHAUSTED を検知。"
                        f"{wait}秒待機してリトライ ({attempt}/{max_retries})..."
                    )
                    time.sleep(wait)
                    wait = min(wait * 2, 120)  # 最大 120 秒まで倍増
                else:
                    raise  # レートリミット以外のエラーはそのまま再送出

        raw_text = response.text
        # usage_metadata があればトークン数を取得、なければ文字数で推定
        try:
            token_count = (
                response.usage_metadata.prompt_token_count
                + response.usage_metadata.candidates_token_count
            )
        except AttributeError:
            token_count = len(raw_text) // 4

        moves = self._extract_moves(raw_text)
        return {'moves': moves, 'token_count': token_count, 'raw_text': raw_text}

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _extract_moves(self, text: str) -> list:
        """テキストから手の列を抽出して返す。"""
        matches = self._MOVE_RE.findall(text)
        return [f"Move {d} from {s} to {t}" for d, s, t in matches]

    def _parse_one(self, move_str: str):
        """1手分の文字列をパースして (disk, src, dst) を返す。失敗時は None。"""
        m = self._MOVE_RE.search(move_str)
        if m:
            return int(m.group(1)), m.group(2).upper(), m.group(3).upper()
        return None

    def _solve_hanoi(self, n: int, src: str, dst: str, aux: str) -> list:
        """ハノイの塔の最適解を再帰で生成する。"""
        if n == 0:
            return []
        return (
            self._solve_hanoi(n - 1, src, aux, dst)
            + [f"Move {n} from {src} to {dst}"]
            + self._solve_hanoi(n - 1, aux, dst, src)
        )


# ---------------------------------------------------------------------------
# ③ PotentialMapper クラス
# ---------------------------------------------------------------------------

class PotentialMapper:
    """
    推論の軌跡データを推論ポテンシャル V(x) の空間へ変換する。

    Li et al. (2018) の Filter Normalization の思想に基づき、
    横軸を推論ステップ数で正規化し、縦軸を TowerOfHanoiEnv._compute_V()
    が返すポテンシャル値 V(x) とすることで、N が異なっても
    同一スケールでポテンシャル地形を比較できる。

    map_trajectory() が返す軌跡の各要素:
        {
            'step'           : int    生の手インデックス (0-indexed 先頭 + 1-indexed)
            'normalized_step': float  step / min_moves  (横軸; Filter Normalization)
            'score'          : float  V(x) = evaluate_state(moves[:step])  ∈ [0.0, ∞)
                                      0.0 = 目標状態（谷底）
                                      1.0 = 初期状態（高原）
                                      >1.0 = ルール違反によるエネルギー障壁
        }
    """

    def map_trajectory(
        self,
        moves: list,
        env: 'TowerOfHanoiEnv',
        step_stride: int = 1,
    ) -> list:
        """
        手の列 moves を 1 手ずつ env に適用しながらスコアを計算し、
        Filter Normalization 済みの軌跡データを返す。

        Parameters
        ----------
        moves : list[str]
            LLMReasoner.reason() が返した手の列。
        env : TowerOfHanoiEnv
            評価対象の環境（N と初期状態を保持）。
        step_stride : int
            何手おきにサンプリングするか（手数が多い場合に軽量化）。

        Returns
        -------
        list[dict]
            各要素は {'step', 'normalized_step', 'score'}。
            先頭要素は step=0（開始時点）で、score = V(initial_state) = 1.0。
        """
        min_moves: int = env.min_moves  # 2^N - 1

        # step=0 は初期状態そのもの。V(initial_state) = 1.0（高原）が正しい出発点。
        v_initial = env._compute_V(env.initial_state)
        trajectory = [
            {'step': 0, 'normalized_step': 0.0, 'score': round(v_initial, 6)}
        ]

        indices = list(range(1, len(moves) + 1, step_stride))
        # 最終ステップは必ず含める
        if not indices or indices[-1] != len(moves):
            indices.append(len(moves))

        for step in indices:
            score = env.evaluate_state(moves[:step])
            normalized = step / min_moves
            trajectory.append({
                'step'           : step,
                'normalized_step': round(normalized, 6),
                'score'          : round(score, 6),
            })

        return trajectory

    def normalize_batch(
        self,
        raw_results: list,
    ) -> list:
        """
        ExperimentRunner が収集した複数実行分の結果リストに対して
        一括で Filter Normalization を適用し直す。

        Parameters
        ----------
        raw_results : list[dict]
            ExperimentRunner が保存した結果リスト。各要素は少なくとも
            'N', 'moves', 'token_count', 'accuracy' を持つ dict。

        Returns
        -------
        list[dict]
            各要素に 'trajectory' (Filter Normalization 済み) を追加したもの。
        """
        enriched = []
        for record in raw_results:
            N = record['N']
            env = TowerOfHanoiEnv(N)
            stride = max(1, len(record['moves']) // 200)  # 最大 200 点
            trajectory = self.map_trajectory(record['moves'], env, step_stride=stride)
            enriched.append({**record, 'trajectory': trajectory})
        return enriched


# ---------------------------------------------------------------------------
# ④ ExperimentRunner クラス
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    複雑度 N ごとのバッチ実行とデータ保存。

    run() を呼ぶと N=n_min〜n_max の範囲で trials 回ずつ推論を実行し、
    各試行の結果（Accuracy・トークン数・正規化済み軌跡）を JSON で保存する。

    出力ファイル構成:
        <output_dir>/
            results_N{N:02d}_trial{trial}.json   # 試行ごとの詳細
            summary.json                          # 全試行の集計（N ごとの平均）
    """

    def __init__(
        self,
        reasoner: 'LLMReasoner',
        mapper: 'PotentialMapper',
        output_dir: str = 'experiment_results',
        n_min: int = 2,
        n_max: int = 12,
        trials: int = 3,
    ):
        self.reasoner   = reasoner
        self.mapper     = mapper
        self.output_dir = output_dir
        self.n_min      = n_min
        self.n_max      = n_max
        self.trials     = trials

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def run(self) -> list:
        """
        全 N・全試行の実験を実行し、結果を JSON に保存する。

        Returns
        -------
        list[dict]
            全試行の結果レコードのリスト（summary.json と同内容）。
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        all_records = []
        # API レートリミット / モデル冷却対策
        # gemini は 5 req/min のため 13 秒間隔
        # ollama はローカルだが連続リクエストでメモリ不足になる場合があるため 5 秒間隔
        if self.reasoner.provider == 'gemini':
            inter_trial_wait = 13
        elif self.reasoner.provider == 'ollama':
            inter_trial_wait = 5
        else:
            inter_trial_wait = 0

        for N in range(self.n_min, self.n_max + 1):
            env = TowerOfHanoiEnv(N)
            print(f"\n[N={N:2d}]  min_moves={env.min_moves}")

            for trial in range(1, self.trials + 1):
                # 2 試行目以降はレートリミット対策のウェイトを挿入
                if inter_trial_wait > 0 and trial > 1:
                    wait_sec = inter_trial_wait + random.randint(0, 2)
                    print(f"    [wait {wait_sec}s for rate limit]")
                    time.sleep(wait_sec)

                record = self._run_one(N, env, trial)
                all_records.append(record)

                # 試行ごとの詳細を保存
                path = os.path.join(
                    self.output_dir,
                    f"results_N{N:02d}_trial{trial}.json",
                )
                self._save_json(record, path)

                acc_str = "OK" if record['accuracy'] == 1 else "FAIL"
                print(
                    f"  trial {trial}/{self.trials}  "
                    f"[{acc_str}]  "
                    f"tokens={record['token_count']:5d}  "
                    f"final_score={record['final_score']:+.3f}  "
                    f"traj_pts={len(record['trajectory'])}"
                )

        # summary を保存
        summary = self._build_summary(all_records)
        self._save_json(summary, os.path.join(self.output_dir, 'summary.json'))
        print(f"\n保存完了: {self.output_dir}/")

        return all_records

    def load_results(self) -> list:
        """
        output_dir に保存された全試行 JSON を読み込んで返す。
        summary.json は除外し、試行ごとのファイルのみ対象とする。
        """
        import os, glob, json
        pattern = os.path.join(self.output_dir, 'results_N*.json')
        records = []
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding='utf-8') as f:
                records.append(json.load(f))
        return records

    # ------------------------------------------------------------------
    # 内部実装
    # ------------------------------------------------------------------

    def _run_one(self, N: int, env: 'TowerOfHanoiEnv', trial: int) -> dict:
        """1 試行分の推論を実行し、結果レコードを返す。"""
        import datetime

        result     = self.reasoner.reason(env)
        moves      = result['moves']
        token_count = result['token_count']
        final_score = env.evaluate_state(moves)
        # V(x) = 0.0 が目標状態（谷底）に到達したことを意味する
        accuracy    = 1 if final_score == 0.0 else 0

        # 軌跡: 手数が多い場合は最大 200 点にダウンサンプリング
        stride     = max(1, len(moves) // 200)
        trajectory = self.mapper.map_trajectory(moves, env, step_stride=stride)

        return {
            'N'           : N,
            'trial'       : trial,
            'provider'    : self.reasoner.provider,
            'model'       : self.reasoner.model,
            'accuracy'    : accuracy,
            'final_score' : round(final_score, 6),
            'token_count' : token_count,
            'move_count'  : len(moves),
            'min_moves'   : env.min_moves,
            'moves'       : moves,
            'trajectory'  : trajectory,
            'timestamp'   : datetime.datetime.now().isoformat(timespec='seconds'),
        }

    def _build_summary(self, records: list) -> dict:
        """
        全試行レコードから N ごとの集計（平均 Accuracy・平均トークン数）を作る。
        """
        from collections import defaultdict

        buckets = defaultdict(list)
        for r in records:
            buckets[r['N']].append(r)

        per_n = []
        for N in sorted(buckets):
            group = buckets[N]
            per_n.append({
                'N'               : N,
                'trials'          : len(group),
                'mean_accuracy'   : round(sum(r['accuracy']    for r in group) / len(group), 4),
                'mean_token_count': round(sum(r['token_count'] for r in group) / len(group), 1),
                'mean_final_score': round(sum(r['final_score'] for r in group) / len(group), 4),
            })

        return {'per_n': per_n, 'total_trials': len(records)}

    @staticmethod
    def _save_json(data: dict, path: str):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# ⑤ Visualizer クラス
# ---------------------------------------------------------------------------

class Visualizer:
    """
    収集・正規化されたデータを基に、スケール則と相転移を可視化する。

    以下の 3 サブプロットを 1 つの Figure に描画する（first_experiment.md §2-⑤）:
        1. 経験的ポテンシャル地形  ── Filter Normalization 済み軌跡を N 別に色分けプロット
        2. 相図 (Phase Diagram)   ── N vs 平均 Accuracy、臨界点での急落を可視化
        3. 推論トークン数の推移   ── N vs 平均トークン数、崩壊点を超えると減少に転じる
    """

    # N=2〜12 を寒色（Overthinking）→暖色（Fixation）にグラデーション
    _CMAP = 'plasma'

    def __init__(self, records: list):
        """
        Parameters
        ----------
        records : list[dict]
            ExperimentRunner.run() または load_results() が返す試行レコードのリスト。
        """
        self.records = records
        self._summary = self._build_summary(records)

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def plot(
        self,
        save_path: str = None,
        show: bool = True,
        ns_for_landscape: list = None,
        landscape_resolution: int = 30,
        landscape_seed: int = 42,
    ):
        """
        全サブプロットを 3 行 GridSpec で描画する（3c 再設計）。

        レイアウト:
            Row 0: ① Loss Landscape コンター図 × 4 パネル（N 別）
            Row 1: ② 経験的ポテンシャル軌跡（Filter Normalization）
            Row 2: ③ 相図（Phase Diagram） + ④ 推論トークン数スケーリング

        Parameters
        ----------
        save_path : str | None
            指定すると PNG として保存する。
        show : bool
            True なら plt.show() を呼ぶ。
        ns_for_landscape : list[int] | None
            コンター図に使う N 値（最大 4 個）。None なら自動選択。
        landscape_resolution : int
            コンター図のグリッド解像度。default=30。
        landscape_seed : int
            直交ベクトル η の乱数シード。default=42。
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # ------------------------------------------------------------------
        # 共通カラーマップと臨界点の計算
        # ------------------------------------------------------------------
        all_N = sorted({r['N'] for r in self.records})
        cmap  = plt.colormaps[self._CMAP].resampled(len(all_N))
        color_map = {N: cmap(i) for i, N in enumerate(all_N)}

        Ns_sum  = [row['N']             for row in self._summary]
        acc_sum = [row['mean_accuracy'] for row in self._summary]
        threshold_N = self._find_threshold(Ns_sum, acc_sum)

        # コンター図用 N の自動選択（臨界点を跨ぐように 4 点選ぶ）
        if ns_for_landscape is None:
            ns_for_landscape = self._auto_select_landscape_Ns(all_N, threshold_N)

        # ------------------------------------------------------------------
        # Figure / GridSpec
        # ------------------------------------------------------------------
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 14))
        fig.patch.set_facecolor('#0a0a1a')
        fig.suptitle(
            'Tower of Hanoi  —  LLM Reasoning Loss Landscape & Phase Transition',
            color='#d0d0ff', fontsize=13, fontweight='bold', y=0.995,
        )

        n_land = len(ns_for_landscape)
        gs = gridspec.GridSpec(
            3, max(n_land, 2), figure=fig,
            height_ratios=[1.05, 1.35, 1.0],
            hspace=0.55, wspace=0.42,
            top=0.96, bottom=0.07, left=0.06, right=0.98,
        )

        # Row 0: コンター図パネル
        ax_lands = [fig.add_subplot(gs[0, i]) for i in range(n_land)]
        # Row 1: ポテンシャル軌跡（全幅）
        ax_traj  = fig.add_subplot(gs[1, :])
        # Row 2: 相図（左半）+ トークン数（右半）
        mid = max(n_land, 2) // 2
        ax_phase = fig.add_subplot(gs[2, :mid])
        ax_token = fig.add_subplot(gs[2, mid:])

        # ------------------------------------------------------------------
        # 描画
        # ------------------------------------------------------------------
        # ① Loss Landscape（3a + 3b）
        self._plot_landscape_contour(
            ax_lands, ns_for_landscape,
            resolution=landscape_resolution,
            seed=landscape_seed,
            threshold_N=threshold_N,
        )
        # 上段の共通ラベル
        fig.text(
            0.5, 0.975,
            '① Loss Landscape  (α: Init→Goal,  β: orthogonal direction)',
            ha='center', va='top', color='#aaaacc', fontsize=8,
        )

        # ② 軌跡
        self._plot_trajectory(ax_traj, color_map, all_N)

        # ③ 相図  ④ トークン数
        self._plot_phase(ax_phase, color_map, all_N)
        self._plot_tokens(ax_token, color_map, all_N)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"保存: {save_path}")
        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # サブプロット①: 経験的ポテンシャル地形
    # ------------------------------------------------------------------

    def _plot_trajectory(self, ax, color_map: dict, all_N: list):
        """
        横軸: normalized_step (step / min_moves)
        縦軸: 推論ポテンシャル V(x) ∈ [0, ∞)
            V = 1.0  … 初期状態（高原・出発点）
            V = 0.0  … 目標状態（谷底・正解）
            V > 1.0  … ルール違反によるエネルギー障壁（棘）
        各 N の全試行を半透明でプロットし、平均軌跡を太線で重ねる。
        """
        import numpy as np

        ax.set_facecolor('#0a0a1a')

        # V > 1 の「エネルギー障壁ゾーン」を淡い赤で背景色付け
        ax.axhspan(1.0, 2.65, alpha=0.07, color='#ff4444',
                   label='Violation zone  (V > 1)')
        # 目標ライン V=0 と初期ライン V=1
        ax.axhline(y=0.0, color='#44aa55', ls='--', lw=1.2, alpha=0.75,
                   label='Goal state  (V = 0)')
        ax.axhline(y=1.0, color='#ffcc44', ls='--', lw=1.0, alpha=0.65,
                   label='Initial state  (V = 1)')
        ax.axvline(x=1.0, color='#556677', ls=':',  lw=0.8,
                   label='norm_step = 1  (= min_moves)')

        # 正規化後に共通グリッドへ補間して平均を取る
        grid = np.linspace(0, 3.0, 300)

        for N in all_N:
            color = color_map[N]
            trials = [r for r in self.records if r['N'] == N]
            interp_scores = []

            for rec in trials:
                traj = rec['trajectory']
                xs = np.array([p['normalized_step'] for p in traj])
                ys = np.array([p['score']           for p in traj])
                if len(xs) < 2:
                    continue
                # 軌跡の範囲外は端の値で外挿
                interp = np.interp(grid, xs, ys,
                                   left=ys[0], right=ys[-1])
                interp_scores.append(interp)
                # 個別試行: 細い半透明線
                ax.plot(xs, ys, color=color, lw=0.8, alpha=0.25)

            if interp_scores:
                mean_curve = np.mean(interp_scores, axis=0)
                ax.plot(grid, mean_curve, color=color, lw=2.0,
                        label=f'N={N}', zorder=3)

        ax.set_xlim(-0.05, 3.05)
        ax.set_ylim(-0.12, 2.65)
        ax.set_xlabel('Normalized reasoning step  (step / min_moves)',
                      color='#9999bb', fontsize=9)
        ax.set_ylabel('Reasoning Potential  V(x)',  color='#9999bb', fontsize=9)
        ax.set_title('② Empirical Potential Trajectory  (Filter Normalization)',
                     color='#ccccee', fontsize=10, pad=6)
        ax.tick_params(colors='#9999bb', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a4a')

        # 凡例: 参照ラインと N 値を分けて右上に表示
        handles, labels = ax.get_legend_handles_labels()
        n_labels   = [(h, l) for h, l in zip(handles, labels) if l.startswith('N=')]
        ref_labels = [(h, l) for h, l in zip(handles, labels) if not l.startswith('N=')]
        # 参照ライン凡例（左上）
        if ref_labels:
            ax.legend(
                [h for h, _ in ref_labels], [l for _, l in ref_labels],
                loc='upper left', fontsize=7, framealpha=0.3, labelcolor='white',
            )
        # N 値凡例（右上）
        if n_labels:
            from matplotlib.legend import Legend
            leg2 = Legend(ax,
                          [h for h, _ in n_labels], [l for _, l in n_labels],
                          loc='upper right', fontsize=7, ncol=2,
                          framealpha=0.3, labelcolor='white',
                          title='complexity N', title_fontsize=7)
            ax.add_artist(leg2)

    # ------------------------------------------------------------------
    # サブプロット②: 相図
    # ------------------------------------------------------------------

    def _plot_phase(self, ax, color_map: dict, all_N: list):
        """
        横軸: N
        縦軸: 平均 Accuracy  (0〜1)
        臨界点での急落（相転移）を可視化する。
        """
        Ns  = [row['N']             for row in self._summary]
        acc = [row['mean_accuracy'] for row in self._summary]

        # 背景: 転移前後を色分け
        threshold_N = self._find_threshold(Ns, acc)
        if threshold_N is not None:
            ax.axvspan(threshold_N - 0.5, max(Ns) + 0.5,
                       alpha=0.08, color='#ff4444',
                       label=f'Collapsed (N≥{threshold_N})')
            ax.axvline(x=threshold_N - 0.5, color='#ff6644',
                       ls='--', lw=1.2, alpha=0.7)

        # 折れ線 + 散布
        ax.plot(Ns, acc, color='#88ddff', lw=1.8, zorder=2)
        for N, a in zip(Ns, acc):
            ax.scatter(N, a, color=color_map[N], s=55, zorder=3,
                       edgecolors='white', linewidths=0.5)

        ax.set_xlim(min(Ns) - 0.5, max(Ns) + 0.5)
        ax.set_ylim(-0.08, 1.12)
        ax.set_xticks(Ns)
        ax.set_xlabel('Complexity  N',       color='#9999bb', fontsize=9)
        ax.set_ylabel('Mean Accuracy',        color='#9999bb', fontsize=9)
        ax.set_title('② Phase Diagram',      color='#ccccee', fontsize=10, pad=6)
        ax.set_facecolor('#0a0a1a')
        ax.tick_params(colors='#9999bb', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a4a')
        if threshold_N is not None:
            ax.legend(fontsize=7, framealpha=0.3, labelcolor='white')

    # ------------------------------------------------------------------
    # サブプロット③: 推論トークン数の推移
    # ------------------------------------------------------------------

    def _plot_tokens(self, ax, color_map: dict, all_N: list):
        """
        横軸: N
        縦軸: 平均トークン数
        臨界点を超えるとトークン数が減少に転じる（推論放棄）ことを可視化する。
        """
        Ns     = [row['N']                for row in self._summary]
        tokens = [row['mean_token_count'] for row in self._summary]
        acc    = [row['mean_accuracy']    for row in self._summary]

        threshold_N = self._find_threshold(Ns, acc)
        if threshold_N is not None:
            ax.axvspan(threshold_N - 0.5, max(Ns) + 0.5,
                       alpha=0.08, color='#ff4444')
            ax.axvline(x=threshold_N - 0.5, color='#ff6644',
                       ls='--', lw=1.2, alpha=0.7, label=f'Collapse point N={threshold_N}')

        ax.fill_between(Ns, tokens, alpha=0.15, color='#ffaa44')
        ax.plot(Ns, tokens, color='#ffcc55', lw=1.8, zorder=2)
        for N, t in zip(Ns, tokens):
            ax.scatter(N, t, color=color_map[N], s=55, zorder=3,
                       edgecolors='white', linewidths=0.5)

        ax.set_xlim(min(Ns) - 0.5, max(Ns) + 0.5)
        ax.set_xticks(Ns)
        ax.set_xlabel('Complexity  N',            color='#9999bb', fontsize=9)
        ax.set_ylabel('Mean Token Count',          color='#9999bb', fontsize=9)
        ax.set_title('③ Reasoning Token Scaling', color='#ccccee', fontsize=10, pad=6)
        ax.set_facecolor('#0a0a1a')
        ax.tick_params(colors='#9999bb', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a4a')
        if threshold_N is not None:
            ax.legend(fontsize=7, framealpha=0.3, labelcolor='white')

    # ------------------------------------------------------------------
    # Phase 3a: 2次元コンター図（Loss Landscape）
    # ------------------------------------------------------------------

    def plot_landscape(
        self,
        ns_to_plot: list = None,
        resolution: int = 30,
        seed: int = 42,
        save_path: str = None,
        show: bool = True,
    ):
        """
        Li et al. (2018) Figure 5/6 に倣った 2D Loss Landscape コンター図を
        独立した Figure として描画する。

        Parameters
        ----------
        ns_to_plot : list[int] | None
            描画する N 値のリスト。None のとき records 内の N を最大 4 個選択。
        resolution : int
            αβ グリッドの解像度（resolution × resolution 点）。default=30。
        seed : int
            直交ベクトル η の乱数シード。再現性のため固定推奨。
        save_path : str | None
            指定すると PNG として保存する。
        show : bool
            True なら plt.show() を呼ぶ。

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        all_N = sorted({r['N'] for r in self.records})
        # 臨界点を計算して自動選択（plot() と共通ロジック）
        Ns_sum  = [row['N']             for row in self._summary]
        acc_sum = [row['mean_accuracy'] for row in self._summary]
        threshold_N_auto = self._find_threshold(Ns_sum, acc_sum)
        if ns_to_plot is None:
            ns_to_plot = self._auto_select_landscape_Ns(all_N, threshold_N_auto)
        n_panels = len(ns_to_plot)

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(5 * n_panels, 4.5))
        fig.patch.set_facecolor('#0a0a1a')
        fig.suptitle(
            'Loss Landscape of LLM Reasoning  (Li et al. 2018 – Filter Normalization)',
            color='#d0d0ff', fontsize=12, fontweight='bold', y=1.01,
        )

        gs = gridspec.GridSpec(
            1, n_panels, figure=fig,
            wspace=0.38, left=0.06, right=0.97,
            top=0.88, bottom=0.14,
        )
        axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]

        # threshold_N_auto は上で既に計算済み（N 自動選択と共用）
        self._plot_landscape_contour(
            axes, ns_to_plot,
            resolution=resolution, seed=seed,
            threshold_N=threshold_N_auto,
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"保存: {save_path}")
        if show:
            plt.show()
        return fig

    def plot_landscape_3d(
        self,
        ns_to_plot: list = None,
        resolution: int = 30,
        seed: int = 42,
        elev: float = 30,
        azim: float = -60,
        save_path: str = None,
        show: bool = True,
    ):
        """
        3D サーフェスプロットで Loss Landscape を描画する（mpl_toolkits.mplot3d）。

        2D コンター図と同じ α-β グリッド・V(x) を使用し、高さ方向に V(x) をとる。
        谷（V=0、目標）と棘（V>1、障壁）が奥行きで直感的にわかるよう設計する。

        各パネルには 2D と同様のフェーズラベル（Convex/Critical/Chaotic）と
        タイトル色を適用する。底面に等高線の投影（影）を重ね、
        最適経路（β=0）を白線でサーフェス上に描画する。

        Parameters
        ----------
        ns_to_plot : list[int] | None
            描画する N 値リスト（最大 4 個）。None なら自動選択。
        resolution : int
            グリッド解像度。default=30。
        seed : int
            直交ベクトル η の乱数シード。default=42。
        elev : float
            3D 視点の仰角（度）。default=30。
        azim : float
            3D 視点の方位角（度）。default=-60。
        save_path : str | None
            PNG 保存先パス。
        show : bool
            True なら plt.show() を呼ぶ。

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (import side-effect)

        # ------------------------------------------------------------------
        # N 選択と臨界点計算（plot_landscape と共通ロジック）
        # ------------------------------------------------------------------
        all_N = sorted({r['N'] for r in self.records})
        Ns_sum  = [row['N']             for row in self._summary]
        acc_sum = [row['mean_accuracy'] for row in self._summary]
        threshold_N = self._find_threshold(Ns_sum, acc_sum)
        if ns_to_plot is None:
            ns_to_plot = self._auto_select_landscape_Ns(all_N, threshold_N)
        n_panels = len(ns_to_plot)

        # ------------------------------------------------------------------
        # Figure 構築
        # ------------------------------------------------------------------
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(5.8 * n_panels, 5.2))
        fig.patch.set_facecolor('#0a0a1a')
        fig.suptitle(
            '3D Loss Landscape of LLM Reasoning  —  V(x) Surface  (Li et al. 2018)',
            color='#d0d0ff', fontsize=12, fontweight='bold', y=1.01,
        )

        V_MAX    = 2.0
        N_LEVELS = 15

        for idx, N in enumerate(ns_to_plot):
            # ----------------------------------------------------------
            # グリッド計算（2D と同じ _build_landscape_grid を再利用）
            # ----------------------------------------------------------
            alpha_arr, beta_arr, Z = self._build_landscape_grid(
                N,
                resolution=resolution,
                alpha_lim=(-0.3, 1.3),
                beta_lim=(-1.0, 1.0),
                seed=seed,
            )
            Z_clipped = np.clip(Z, 0.0, V_MAX)
            A, B = np.meshgrid(alpha_arr, beta_arr)

            # ----------------------------------------------------------
            # フェーズ判定（2D コンター図と同一ロジック）
            # ----------------------------------------------------------
            if threshold_N is None:
                phase_label = ''
                phase_color = '#ccccee'
            elif N < threshold_N:
                phase_label = 'Convex'
                phase_color = '#44aaff'
            elif N == threshold_N:
                phase_label = 'Critical'
                phase_color = '#ffaa00'
            else:
                phase_label = 'Chaotic'
                phase_color = '#ff4444'

            # ----------------------------------------------------------
            # 3D サブプロット
            # ----------------------------------------------------------
            ax = fig.add_subplot(1, n_panels, idx + 1, projection='3d')

            # ---- サーフェス ----
            surf = ax.plot_surface(
                A, B, Z_clipped,
                cmap='RdYlGn_r',
                vmin=0.0, vmax=V_MAX,
                linewidth=0,
                antialiased=True,
                alpha=0.88,
            )

            # ---- 底面に等高線の影を投影（谷の形を立体的に強調）----
            ax.contourf(
                A, B, Z_clipped,
                zdir='z', offset=0.0,
                levels=N_LEVELS,
                cmap='RdYlGn_r',
                alpha=0.30,
            )

            # ---- 最適経路（β=0）をサーフェス上に白線で描画 ----
            i_zero = int(np.argmin(np.abs(beta_arr - 0.0)))
            z_path = Z_clipped[i_zero, :]
            ax.plot(
                alpha_arr,
                np.zeros_like(alpha_arr),
                z_path,
                color='white', lw=2.0, zorder=5,
                label='Optimal path (β=0)',
            )

            # ---- Init (α=0, β=0) / Goal (α=1, β=0) マーカー ----
            j_init = int(np.argmin(np.abs(alpha_arr - 0.0)))
            j_goal = int(np.argmin(np.abs(alpha_arr - 1.0)))
            ax.scatter(
                [0.0], [0.0], [Z_clipped[i_zero, j_init]],
                color='cyan', s=70, zorder=6, depthshade=False,
                edgecolors='white', linewidths=0.5,
            )
            ax.scatter(
                [1.0], [0.0], [Z_clipped[i_zero, j_goal]],
                color='lime', s=70, zorder=6, depthshade=False,
                edgecolors='white', linewidths=0.5,
            )

            # ---- 視点 ----
            ax.view_init(elev=elev, azim=azim)

            # ---- ダーク背景スタイル ----
            _pane = (0.04, 0.04, 0.10, 0.85)
            _edge = '#1a1a3a'
            ax.xaxis.pane.set_facecolor(_pane)
            ax.yaxis.pane.set_facecolor(_pane)
            ax.zaxis.pane.set_facecolor(_pane)
            ax.xaxis.pane.set_edgecolor(_edge)
            ax.yaxis.pane.set_edgecolor(_edge)
            ax.zaxis.pane.set_edgecolor(_edge)
            ax.grid(True, color='#2a2a4a', linewidth=0.4, alpha=0.5)

            # ---- 軸ラベル・目盛り ----
            ax.set_xlabel('α  (Init → Goal)', color='#9999bb', fontsize=7, labelpad=6)
            ax.set_ylabel('β  (orthogonal)',   color='#9999bb', fontsize=7, labelpad=6)
            ax.set_zlabel('V(x)',              color='#9999bb', fontsize=7, labelpad=4)
            ax.tick_params(colors='#9999bb', labelsize=6, pad=1)
            ax.set_zlim(0.0, V_MAX)

            # ---- タイトル（フェーズカラーで色分け）----
            title_text = f'N = {N}'
            if phase_label:
                title_text += f'  [{phase_label}]'
            ax.set_title(
                title_text,
                color=phase_color, fontsize=9, fontweight='bold', pad=10,
            )

            # ---- フェーズラベルをパネル左下に固定テキストで追加 ----
            if phase_label:
                ax.text2D(
                    0.03, 0.04, phase_label,
                    transform=ax.transAxes,
                    color=phase_color, fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='#0a0a1a', alpha=0.7,
                              edgecolor=phase_color, boxstyle='round,pad=0.3'),
                )

            # ---- カラーバー ----
            cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.12, aspect=18)
            cbar.set_label('V(x)', color='#9999bb', fontsize=7)
            cbar.ax.tick_params(colors='#9999bb', labelsize=6)

        fig.subplots_adjust(
            left=0.02, right=0.98,
            top=0.92, bottom=0.05,
            wspace=0.30,
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"保存: {save_path}")
        if show:
            plt.show()
        return fig

    def _plot_landscape_contour(
        self,
        axes,
        ns_to_plot: list,
        resolution: int = 30,
        seed: int = 42,
        threshold_N: int = None,
    ):
        """
        axes リストの各軸に、ns_to_plot の各 N の 2D コンター図を描画する（3a + 3b）。

        座標系:
            α 軸（主軸）  : x_init → x_goal 方向。α=0 が初期状態、α=1 が目標状態。
            β 軸（直交軸）: η ∝ Gram-Schmidt(rand ⊥ δ)、スケールは ||δ|| に合わせる。

        各 (α, β) 点で
            x_cont = x_init + α·δ + β·η
        を計算し、_snap_to_state() で最近傍状態へ変換して V(x) を評価する。

        3b 拡張: threshold_N を受け取り、各パネルに以下を追加する。
            - フェーズラベル（Convex / Critical / Chaotic）
            - 境界色（青系=凸領域、橙=臨界、赤=カオス領域）
            - 荒れ指標 ∇V（平均勾配ノルム）

        Parameters
        ----------
        axes : list of matplotlib.axes.Axes
            描画先の軸リスト (len == len(ns_to_plot))。
        ns_to_plot : list[int]
            描画する N 値のリスト。
        resolution : int
            グリッド解像度 (resolution × resolution 点)。
        seed : int
            η ベクトル生成の乱数シード。
        threshold_N : int | None
            相転移の臨界点 N_c。None の場合は相ラベルを省略。
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # カラースケール上限: 1.0=初期状態、>1.0=障壁。2.0 でクリップすることで
        # 障壁の「棘」を表現しつつ、凸地形のなだらかさも見えるようにする。
        V_MAX = 2.0
        N_LEVELS = 25
        levels = np.linspace(0.0, V_MAX, N_LEVELS)

        for ax, N in zip(axes, ns_to_plot):
            alpha_arr, beta_arr, Z = self._build_landscape_grid(
                N,
                resolution=resolution,
                alpha_lim=(-0.3, 1.3),
                beta_lim=(-1.0, 1.0),
                seed=seed,
            )
            Z_clipped = np.clip(Z, 0.0, V_MAX)

            # ----------------------------------------------------------
            # 3b: フェーズ判定と視覚スタイルの決定
            # ----------------------------------------------------------
            if threshold_N is None:
                phase_label  = ''
                border_color = '#2a2a4a'
                border_lw    = 1.0
            elif N < threshold_N:
                phase_label  = 'Convex'
                border_color = '#44aaff'   # 青系: 凸地形
                border_lw    = 2.0
            elif N == threshold_N:
                phase_label  = 'Critical'
                border_color = '#ffaa00'   # 橙: 臨界点
                border_lw    = 2.5
            else:
                phase_label  = 'Chaotic'
                border_color = '#ff4444'   # 赤: カオス地形
                border_lw    = 2.0

            # ----------------------------------------------------------
            # 塗りつぶし等高線
            # ----------------------------------------------------------
            cf = ax.contourf(
                alpha_arr, beta_arr, Z_clipped,
                levels=levels, cmap='RdYlGn_r', extend='max',
            )
            # 白い輪郭線（間引いて重ねる）
            ax.contour(
                alpha_arr, beta_arr, Z_clipped,
                levels=levels[::4], colors='white', alpha=0.25, linewidths=0.4,
            )

            # ----------------------------------------------------------
            # 最適経路と特徴点のマーク
            # ----------------------------------------------------------
            ax.axhline(y=0.0, color='white', ls='--', lw=1.2, alpha=0.7,
                       label='Optimal path (β=0)')
            ax.scatter([0.0], [0.0], color='cyan', s=55, zorder=6,
                       edgecolors='white', linewidths=0.5, label='Init (α=0)')
            ax.scatter([1.0], [0.0], color='lime', s=55, zorder=6,
                       edgecolors='white', linewidths=0.5, label='Goal (α=1)')

            # ----------------------------------------------------------
            # カラーバー
            # ----------------------------------------------------------
            cbar = plt.colorbar(cf, ax=ax, shrink=0.92, pad=0.03)
            cbar.set_label('V(x)', color='#9999bb', fontsize=7)
            cbar.ax.tick_params(colors='#9999bb', labelsize=6)
            cbar.ax.yaxis.set_tick_params(color='#9999bb')

            # ----------------------------------------------------------
            # 軸スタイル
            # ----------------------------------------------------------
            ax.set_facecolor('#0a0a1a')
            for sp in ax.spines.values():
                sp.set_edgecolor(border_color)
                sp.set_linewidth(border_lw)

            ax.tick_params(colors='#9999bb', labelsize=7)
            ax.set_xlabel('α  (Init → Goal)', color='#9999bb', fontsize=8)
            ax.set_ylabel('β  (orthogonal)',   color='#9999bb', fontsize=8)

            # ----------------------------------------------------------
            # 3b: タイトルにフェーズラベルを組み込む
            # ----------------------------------------------------------
            title_text = f'N = {N}'
            if phase_label:
                title_text += f'  [{phase_label}]'
            ax.set_title(title_text, color=border_color if phase_label else '#ccccee',
                         fontsize=9, pad=5, fontweight='bold')

            # ----------------------------------------------------------
            # 3b: 荒れ指標 ∇V（平均勾配ノルム）をパネル左下に表示
            # ----------------------------------------------------------
            gy, gx = np.gradient(Z)
            roughness = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
            ax.text(
                0.03, 0.04, f'∇V = {roughness:.3f}',
                transform=ax.transAxes,
                color='#aaaacc', fontsize=6.5, ha='left', va='bottom',
                bbox=dict(facecolor='#0a0a1a', alpha=0.6, edgecolor='none',
                          boxstyle='round,pad=0.2'),
            )

            # ----------------------------------------------------------
            # 凡例は最初のパネルだけ表示
            # ----------------------------------------------------------
            if ax is axes[0]:
                ax.legend(fontsize=6, framealpha=0.35, labelcolor='white',
                          loc='upper right')

    def _build_landscape_grid(
        self,
        N: int,
        resolution: int = 30,
        alpha_lim: tuple = (-0.3, 1.3),
        beta_lim: tuple = (-1.0, 1.0),
        seed: int = 42,
    ) -> tuple:
        """
        N 枚ハノイの 2D 損失地形グリッドを計算して返す。

        直交ベクトル η は ||δ|| にスケールし、β ∈ [-1, 1] の範囲で
        主軸と同等の「振れ幅」をもつよう正規化する。

        Returns
        -------
        alpha_arr : np.ndarray  shape=(resolution,)
        beta_arr  : np.ndarray  shape=(resolution,)
        Z         : np.ndarray  shape=(resolution, resolution)
            Z[i, j] = V(snap(x_init + alpha_arr[j]·δ + beta_arr[i]·η))
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        env = TowerOfHanoiEnv(N)

        x_init = env._get_state_coord(env.initial_state)   # shape (3N,)
        x_goal = env._get_state_coord(env.goal_state)      # shape (3N,)
        delta  = x_goal - x_init                           # 主軸ベクトル δ
        delta_norm = np.linalg.norm(delta)

        # --- η の生成: ランダムベクトルから δ 成分を Gram-Schmidt で除去 ---
        eta = rng.standard_normal(3 * N)
        eta -= (np.dot(eta, delta) / (delta_norm ** 2 + 1e-12)) * delta
        eta_norm = np.linalg.norm(eta)
        # ||δ|| と同スケールにして β ∈ [-1,1] が主軸と対等な振れ幅をもつようにする
        eta = eta / (eta_norm + 1e-12) * delta_norm

        alpha_arr = np.linspace(*alpha_lim, resolution)
        beta_arr  = np.linspace(*beta_lim,  resolution)
        Z = np.zeros((resolution, resolution))

        for i, beta in enumerate(beta_arr):
            for j, alpha in enumerate(alpha_arr):
                x_cont = x_init + alpha * delta + beta * eta
                state  = self._snap_to_state(x_cont, N)
                Z[i, j] = env._compute_V(state)

        return alpha_arr, beta_arr, Z

    @staticmethod
    def _snap_to_state(x_cont: 'np.ndarray', N: int) -> dict:
        """
        連続ベクトル x_cont を最近傍の離散ハノイ状態にスナップする。

        各円盤 i (1..N) について、対応する 3 要素 [is_A, is_B, is_C] の
        最大値を持つペグを選択する (argmax)。
        ペグ内の積み順は大きい円盤が下になるよう整列する。

        Parameters
        ----------
        x_cont : np.ndarray  shape=(3*N,)
            連続空間上の座標ベクトル。
        N : int
            円盤数。

        Returns
        -------
        dict  {'A': [...], 'B': [...], 'C': [...]}
            各円盤が1つのペグに割り当てられた状態辞書。
        """
        import numpy as np
        pegs  = ['A', 'B', 'C']
        state = {'A': [], 'B': [], 'C': []}
        for disk in range(1, N + 1):
            idx    = (disk - 1) * 3
            chosen = pegs[int(np.argmax(x_cont[idx: idx + 3]))]
            state[chosen].append(disk)
        # 各ペグ内を大きい順(底が大)に並べる
        for peg in pegs:
            state[peg].sort(reverse=True)
        return state

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(records: list) -> list:
        """レコードリストから N ごとの集計行を生成する。"""
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in records:
            buckets[r['N']].append(r)
        rows = []
        for N in sorted(buckets):
            g = buckets[N]
            rows.append({
                'N'               : N,
                'mean_accuracy'   : sum(r['accuracy']    for r in g) / len(g),
                'mean_token_count': sum(r['token_count'] for r in g) / len(g),
                'mean_final_score': sum(r['final_score'] for r in g) / len(g),
            })
        return rows

    @staticmethod
    def _find_threshold(Ns: list, acc: list) -> int | None:
        """
        Accuracy が初めて 0.5 を下回る N を相転移の臨界点として返す。
        見つからなければ None を返す。
        """
        for N, a in zip(Ns, acc):
            if a < 0.5:
                return N
        return None

    @staticmethod
    def _auto_select_landscape_Ns(all_N: list, threshold_N: int | None) -> list:
        """
        Loss Landscape コンター図用に、相転移を跨ぐ代表的な N 値を最大 4 つ選ぶ。

        選択方針:
            1. all_N の最小値（凸地形の代表）
            2. threshold_N の直前（臨界直前の凸寄り）
            3. threshold_N そのもの（臨界点）
            4. all_N の最大値（カオス地形の代表）

        threshold_N が None の場合は等間隔で 4 点を選ぶ。
        """
        if len(all_N) <= 4:
            return list(all_N)

        if threshold_N is None or threshold_N not in all_N:
            step = max(1, len(all_N) // 4)
            return all_N[::step][:4]

        idx_nc = all_N.index(threshold_N)
        result = [all_N[0]]                             # 最小 N
        if idx_nc > 1:
            result.append(all_N[idx_nc - 1])            # 臨界直前
        if threshold_N not in result:
            result.append(threshold_N)                   # 臨界点
        if all_N[-1] not in result:
            result.append(all_N[-1])                     # 最大 N
        # 重複除去・昇順・最大 4 個
        return sorted(set(result))[:4]


# ---------------------------------------------------------------------------
# main 関数
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog='hanoi_phase_diagram',
        description=(
            'Tower of Hanoi - LLM Reasoning Phase Transition Experiment\n'
            'LLM の推論崩壊をポテンシャル地形・相図・トークン数グラフで可視化する。'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            使用例:
              # ダミーモードで全実験を実行してグラフを保存
              python hanoi_phase_diagram.py --provider dummy

              # Gemini API を使って N=2〜8、各 5 試行
              python hanoi_phase_diagram.py --provider gemini --n-max 8 --trials 5

              # 保存済み結果だけ読み込んでグラフを再描画
              python hanoi_phase_diagram.py --plot-only --output-dir experiment_results
        """),
    )

    # --- プロバイダ ---
    parser.add_argument(
        '--provider', '-p',
        choices=['dummy', 'anthropic', 'openai', 'gemini', 'ollama'],
        default='dummy',
        help='LLM プロバイダ (default: dummy)',
    )
    parser.add_argument(
        '--model', '-m',
        default=None,
        help=(
            'モデル名。省略時はプロバイダごとのデフォルト。\n'
            '  anthropic: claude-haiku-4-5-20251001\n'
            '  openai   : gpt-4o\n'
            '  gemini   : gemini-2.5-flash'
        ),
    )
    parser.add_argument(
        '--api-key', '-k',
        default=None,
        dest='api_key',
        help='API キー。省略時は環境変数から取得。',
    )

    # --- 実験パラメータ ---
    parser.add_argument(
        '--n-min', type=int, default=2,
        help='複雑度の最小値 (default: 2)',
    )
    parser.add_argument(
        '--n-max', type=int, default=12,
        help='複雑度の最大値 (default: 12)',
    )
    parser.add_argument(
        '--trials', '-t', type=int, default=3,
        help='各 N ごとの試行回数 (default: 3)',
    )

    # --- 入出力 ---
    parser.add_argument(
        '--output-dir', '-o',
        default='experiment_results',
        dest='output_dir',
        help='結果 JSON の保存ディレクトリ (default: experiment_results)',
    )
    parser.add_argument(
        '--save-fig',
        default='phase_diagram.png',
        dest='save_fig',
        help='グラフの保存先 PNG パス (default: phase_diagram.png)',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='グラフをウィンドウ表示する（非インタラクティブ環境では不要）',
    )

    # --- モード切替 ---
    parser.add_argument(
        '--plot-only',
        action='store_true',
        dest='plot_only',
        help=(
            '実験をスキップし、--output-dir の保存済み JSON からグラフを再描画する。'
        ),
    )

    # --- Loss Landscape オプション ---
    parser.add_argument(
        '--landscape',
        action='store_true',
        help=(
            'Loss Landscape コンター図を独立した Figure として追加保存する。'
            'メイン図（--save-fig）にも常にコンター図は含まれる。'
        ),
    )
    parser.add_argument(
        '--save-landscape',
        default='loss_landscape.png',
        dest='save_landscape',
        help='コンター図独立 Figure の保存先 PNG パス (default: loss_landscape.png)',
    )
    parser.add_argument(
        '--landscape-resolution',
        type=int, default=30,
        dest='landscape_resolution',
        help='コンター図グリッドの解像度 (default: 30)',
    )

    # --- 3D Loss Landscape オプション ---
    parser.add_argument(
        '--landscape-3d',
        action='store_true',
        dest='landscape_3d',
        help=(
            '3D サーフェスプロット（Loss Landscape）を独立した Figure として追加保存する。'
        ),
    )
    parser.add_argument(
        '--save-landscape-3d',
        default='loss_landscape_3d.png',
        dest='save_landscape_3d',
        help='3D サーフェスプロットの保存先 PNG パス (default: loss_landscape_3d.png)',
    )
    parser.add_argument(
        '--landscape-3d-elev',
        type=float, default=30.0,
        dest='landscape_3d_elev',
        help='3D 視点の仰角 elev（度）(default: 30)',
    )
    parser.add_argument(
        '--landscape-3d-azim',
        type=float, default=-60.0,
        dest='landscape_3d_azim',
        help='3D 視点の方位角 azim（度）(default: -60)',
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # ヘッダー表示
    # ------------------------------------------------------------------
    _print_header(args)

    # ------------------------------------------------------------------
    # 実験フェーズ
    # ------------------------------------------------------------------
    mapper   = PotentialMapper()
    reasoner = LLMReasoner(
        provider=args.provider,
        model=args.api_key,      # model は別引数
        api_key=args.api_key,
    )
    # model は api_key と別に設定
    reasoner.model   = args.model
    reasoner.api_key = args.api_key

    runner = ExperimentRunner(
        reasoner=reasoner,
        mapper=mapper,
        output_dir=args.output_dir,
        n_min=args.n_min,
        n_max=args.n_max,
        trials=args.trials,
    )

    if args.plot_only:
        print(f"[plot-only] {args.output_dir} から結果を読み込みます...")
        records = runner.load_results()
        if not records:
            print(f"ERROR: {args.output_dir} に結果ファイルが見つかりません。")
            print("先に実験を実行してください（--plot-only を外して再実行）。")
            raise SystemExit(1)
        print(f"  {len(records)} 件の試行レコードを読み込みました。")
    else:
        records = runner.run()
        _print_summary(args.output_dir)

    # ------------------------------------------------------------------
    # 可視化フェーズ
    # ------------------------------------------------------------------
    print()
    viz = Visualizer(records)

    # メイン Figure（Loss Landscape + 軌跡 + 相図 + トークン数）
    viz.plot(
        save_path=args.save_fig,
        show=args.show,
        landscape_resolution=args.landscape_resolution,
    )

    # --landscape フラグ: 2D コンター図を独立 Figure として追加保存
    if args.landscape:
        print(f"[landscape] 2D コンター図を独立 Figure として保存: {args.save_landscape}")
        viz.plot_landscape(
            resolution=args.landscape_resolution,
            save_path=args.save_landscape,
            show=args.show,
        )

    # --landscape-3d フラグ: 3D サーフェスプロットを独立 Figure として追加保存
    if args.landscape_3d:
        print(f"[landscape-3d] 3D サーフェスプロットを保存: {args.save_landscape_3d}")
        viz.plot_landscape_3d(
            resolution=args.landscape_resolution,
            elev=args.landscape_3d_elev,
            azim=args.landscape_3d_azim,
            save_path=args.save_landscape_3d,
            show=args.show,
        )

    print("完了。")


# ---------------------------------------------------------------------------
# main ヘルパー
# ---------------------------------------------------------------------------

def _print_header(args):
    import textwrap
    print("=" * 60)
    print("  Tower of Hanoi - LLM Reasoning Phase Transition")
    print("=" * 60)
    print(f"  provider   : {args.provider}")
    if args.model:
        print(f"  model      : {args.model}")
    if not args.plot_only:
        print(f"  N range    : {args.n_min} ~ {args.n_max}")
        print(f"  trials/N   : {args.trials}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  save_fig   : {args.save_fig}")
    if args.landscape:
        print(f"  landscape  : {args.save_landscape}  (resolution={args.landscape_resolution})")
    if args.landscape_3d:
        print(f"  landscape3d: {args.save_landscape_3d}"
              f"  (elev={args.landscape_3d_elev}, azim={args.landscape_3d_azim})")
    if args.plot_only:
        print("  mode       : plot-only (skip experiment)")
    print()


def _print_summary(output_dir: str):
    summary_path = os.path.join(output_dir, 'summary.json')
    if not os.path.exists(summary_path):
        return
    with open(summary_path, encoding='utf-8') as f:
        summary = json.load(f)
    print("\n--- summary ---")
    header = f"  {'N':>3}  {'acc':>5}  {'tokens':>8}  {'score':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in summary['per_n']:
        phase = 'OK  ' if row['mean_accuracy'] >= 0.5 else 'FAIL'
        print(
            f"  N={row['N']:2d}  [{phase}]"
            f"  acc={row['mean_accuracy']:.2f}"
            f"  tokens={row['mean_token_count']:7.1f}"
            f"  score={row['mean_final_score']:+.3f}"
        )


import textwrap


if __name__ == '__main__':
    main()
