"""
ハノイの塔 推論ポテンシャル・相図作成実験スクリプト
first_experiment.md の仕様に基づく実装
"""

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
    ハノイの塔の環境シミュレータおよび状態の絶対評価。

    円盤数 N で初期化し、初期状態と目標状態を管理する。
    evaluate_state は LLM の対数尤度や自信度に一切依存せず、
    物理的な盤面状態に基づく絶対評価でスコアを返す。
    """

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
        LLM が生成した手のリストを受け取り、目標状態への近さを返す。

        Returns
        -------
        float
            x ∈ [-1.0, 1.0]
            - ルール違反がある場合: 違反数に応じた負のスコア（-1.0 〜 0）
            - ルール違反がない場合: 目標達成度に応じた正のスコア（0 〜 +1.0）
              完全正解なら +1.0

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

        if illegal_count > 0:
            # 違反1回 → -0.2、5回以上 → -1.0
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

    def _compute_progress(self, state: dict) -> float:
        """
        目標状態（C ペグに全円盤を積む）への進捗を 0.0〜1.0 で返す。
        C ペグの底から連続して正しく積まれている枚数で評価する。
        """
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
    推論の軌跡データに Filter Normalization を適用し、
    ポテンシャル空間の軌跡へ変換する。

    Filter Normalization の定義（first_experiment.md §2-③）:
        横軸: 推論ステップ数 / min_moves  (min_moves = 2^N - 1)
        縦軸: evaluate_state で得られたスコアをポテンシャル高さ V として使用

    map_trajectory() が返す軌跡の各要素:
        {
            'step'           : int    生の手インデックス (1-indexed)
            'normalized_step': float  step / min_moves
            'score'          : float  evaluate_state(moves[:step]) ∈ [-1, 1]
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
            先頭要素は step=0（開始時点、score=0.0）を含む。
        """
        min_moves: int = env.min_moves  # 2^N - 1

        trajectory = [
            {'step': 0, 'normalized_step': 0.0, 'score': 0.0}
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
        accuracy    = 1 if final_score == 1.0 else 0

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

    def plot(self, save_path: str = None, show: bool = True):
        """
        3 サブプロットを描画する。

        Parameters
        ----------
        save_path : str | None
            指定すると PNG として保存する。例: 'phase_diagram.png'
        show : bool
            True なら plt.show() を呼ぶ。
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.cm as cm
        import numpy as np

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(14, 12))
        fig.patch.set_facecolor('#0a0a1a')
        fig.suptitle(
            'Tower of Hanoi  —  LLM Reasoning Phase Transition',
            color='#d0d0ff', fontsize=14, fontweight='bold', y=0.98,
        )

        gs = gridspec.GridSpec(
            2, 2, figure=fig,
            height_ratios=[1.4, 1.0],
            hspace=0.45, wspace=0.35,
            top=0.93, bottom=0.08, left=0.08, right=0.97,
        )

        ax_traj  = fig.add_subplot(gs[0, :])   # 上段全幅: ポテンシャル地形
        ax_phase = fig.add_subplot(gs[1, 0])   # 下段左 : 相図
        ax_token = fig.add_subplot(gs[1, 1])   # 下段右 : トークン数

        all_N = sorted({r['N'] for r in self.records})
        cmap  = plt.colormaps[self._CMAP].resampled(len(all_N))
        color_map = {N: cmap(i) for i, N in enumerate(all_N)}

        self._plot_trajectory(ax_traj,  color_map, all_N)
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
        縦軸: score  ∈ [-1, 1]  （= ポテンシャル高さ V の代理変数）
        各 N の全試行を半透明でプロットし、平均軌跡を太線で重ねる。
        """
        import numpy as np

        ax.set_facecolor('#0a0a1a')
        ax.axhline(y= 1.0, color='#44aa55', ls='--', lw=0.9, alpha=0.5,
                   label='Correct (score=+1)')
        ax.axhline(y=-1.0, color='#aa4444', ls='--', lw=0.9, alpha=0.5,
                   label='Fixation (score=−1)')
        ax.axhline(y= 0.0, color='#334455', ls=':',  lw=0.7)
        ax.axvline(x= 1.0, color='#556677', ls=':',  lw=0.8,
                   label='norm_step=1 (min_moves)')

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
                # 軌跡の範囲外は端の値で外挿（step_fill_value）
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
        ax.set_ylim(-1.25, 1.25)
        ax.set_xlabel('Normalized reasoning step  (step / min_moves)',
                      color='#9999bb', fontsize=9)
        ax.set_ylabel('Score  V(x)',  color='#9999bb', fontsize=9)
        ax.set_title('① Empirical Potential Landscape  (Filter Normalization)',
                     color='#ccccee', fontsize=10, pad=6)
        ax.tick_params(colors='#9999bb', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a4a')

        # 凡例: N 値だけ右外に出す
        handles, labels = ax.get_legend_handles_labels()
        n_labels  = [(h, l) for h, l in zip(handles, labels) if l.startswith('N=')]
        ref_labels = [(h, l) for h, l in zip(handles, labels) if not l.startswith('N=')]
        if n_labels:
            ax.legend(
                [h for h, _ in n_labels], [l for _, l in n_labels],
                loc='upper right', fontsize=7, ncol=2,
                framealpha=0.3, labelcolor='white',
                title='complexity N', title_fontsize=7,
            )

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
    viz.plot(save_path=args.save_fig, show=args.show)
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
