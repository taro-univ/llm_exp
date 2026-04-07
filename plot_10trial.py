"""
experiment_results から N=2~7(trial1-10), N=8(trial1-7) を読み込み
phase_diagram_10trial.png を生成する。
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

OUTPUT_DIR = 'experiment_results'
SAVE_PATH  = 'phase_diagram_10trial.png'
CMAP       = 'plasma'

# 読み込み対象の定義: N -> 使用するtrial番号リスト
TARGET = {n: list(range(1, 11)) for n in range(2, 8)}   # N=2~7: trial1-10
TARGET[8] = list(range(1, 8))                             # N=8  : trial1-7

# -----------------------------------------------------------------------
# データ読み込み
# -----------------------------------------------------------------------
records = []
for N, trials in sorted(TARGET.items()):
    for t in trials:
        path = os.path.join(OUTPUT_DIR, f'results_N{N:02d}_trial{t}.json')
        if not os.path.exists(path):
            print(f'[WARN] 見つかりません: {path}')
            continue
        with open(path, encoding='utf-8') as f:
            records.append(json.load(f))

print(f'読み込んだ試行数: {len(records)}')

# -----------------------------------------------------------------------
# サマリー集計
# -----------------------------------------------------------------------
buckets = defaultdict(list)
for r in records:
    buckets[r['N']].append(r)

summary = []
for N in sorted(buckets):
    g = buckets[N]
    summary.append({
        'N'               : N,
        'trials'          : len(g),
        'mean_accuracy'   : sum(r['accuracy']    for r in g) / len(g),
        'mean_token_count': sum(r['token_count'] for r in g) / len(g),
        'mean_final_score': sum(r['final_score'] for r in g) / len(g),
    })

print('\n--- summary ---')
print(f"  {'N':>3}  {'trials':>6}  {'acc':>5}  {'tokens':>8}  {'score':>7}")
for row in summary:
    phase = 'OK  ' if row['mean_accuracy'] >= 0.5 else 'FAIL'
    print(
        f"  N={row['N']:2d}  [{phase}]"
        f"  trials={row['trials']:2d}"
        f"  acc={row['mean_accuracy']:.2f}"
        f"  tokens={row['mean_token_count']:8.1f}"
        f"  score={row['mean_final_score']:+.3f}"
    )

# -----------------------------------------------------------------------
# 臨界点検出
# -----------------------------------------------------------------------
def find_threshold(Ns, acc):
    for N, a in zip(Ns, acc):
        if a < 0.5:
            return N
    return None

# -----------------------------------------------------------------------
# 描画
# -----------------------------------------------------------------------
all_N = sorted(buckets.keys())
cmap_obj = plt.colormaps[CMAP].resampled(len(all_N))
color_map = {N: cmap_obj(i) for i, N in enumerate(all_N)}

plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 12))
fig.patch.set_facecolor('#0a0a1a')
fig.suptitle(
    'Tower of Hanoi  —  LLM Reasoning Phase Transition\n'
    '(deepseek-r1:14b via Ollama,  N=2~7: 10 trials,  N=8: 7 trials)',
    color='#d0d0ff', fontsize=13, fontweight='bold', y=0.98,
)

gs = gridspec.GridSpec(
    2, 2, figure=fig,
    height_ratios=[1.4, 1.0],
    hspace=0.45, wspace=0.35,
    top=0.92, bottom=0.08, left=0.08, right=0.97,
)

ax_traj  = fig.add_subplot(gs[0, :])
ax_phase = fig.add_subplot(gs[1, 0])
ax_token = fig.add_subplot(gs[1, 1])

# ---- ① 経験的ポテンシャル地形 ----------------------------------------
ax = ax_traj
ax.set_facecolor('#0a0a1a')
ax.axhline(y= 1.0, color='#44aa55', ls='--', lw=0.9, alpha=0.5, label='Correct (score=+1)')
ax.axhline(y=-1.0, color='#aa4444', ls='--', lw=0.9, alpha=0.5, label='Fixation (score=−1)')
ax.axhline(y= 0.0, color='#334455', ls=':',  lw=0.7)
ax.axvline(x= 1.0, color='#556677', ls=':',  lw=0.8, label='norm_step=1 (min_moves)')

grid = np.linspace(0, 3.0, 300)

for N in all_N:
    color = color_map[N]
    trials_list = [r for r in records if r['N'] == N]
    interp_scores = []

    for rec in trials_list:
        traj = rec['trajectory']
        xs = np.array([p['normalized_step'] for p in traj])
        ys = np.array([p['score']           for p in traj])
        if len(xs) < 2:
            continue
        interp = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
        interp_scores.append(interp)
        ax.plot(xs, ys, color=color, lw=0.8, alpha=0.25)

    if interp_scores:
        mean_curve = np.mean(interp_scores, axis=0)
        ax.plot(grid, mean_curve, color=color, lw=2.0, label=f'N={N}', zorder=3)

ax.set_xlim(-0.05, 3.05)
ax.set_ylim(-1.25, 1.25)
ax.set_xlabel('Normalized reasoning step  (step / min_moves)', color='#9999bb', fontsize=9)
ax.set_ylabel('Score  V(x)', color='#9999bb', fontsize=9)
ax.set_title('① Empirical Potential Landscape  (Filter Normalization)',
             color='#ccccee', fontsize=10, pad=6)
ax.tick_params(colors='#9999bb', labelsize=8)
for sp in ax.spines.values():
    sp.set_edgecolor('#2a2a4a')

handles, labels = ax.get_legend_handles_labels()
n_labels   = [(h, l) for h, l in zip(handles, labels) if l.startswith('N=')]
if n_labels:
    ax.legend(
        [h for h, _ in n_labels], [l for _, l in n_labels],
        loc='upper right', fontsize=7, ncol=2,
        framealpha=0.3, labelcolor='white',
        title='complexity N', title_fontsize=7,
    )

# ---- ② 相図 ------------------------------------------------------------
ax = ax_phase
Ns  = [row['N']             for row in summary]
acc = [row['mean_accuracy'] for row in summary]

threshold_N = find_threshold(Ns, acc)
if threshold_N is not None:
    ax.axvspan(threshold_N - 0.5, max(Ns) + 0.5,
               alpha=0.08, color='#ff4444',
               label=f'Collapsed (N≥{threshold_N})')
    ax.axvline(x=threshold_N - 0.5, color='#ff6644', ls='--', lw=1.2, alpha=0.7)

ax.plot(Ns, acc, color='#88ddff', lw=1.8, zorder=2)
for N, a in zip(Ns, acc):
    ax.scatter(N, a, color=color_map[N], s=55, zorder=3,
               edgecolors='white', linewidths=0.5)

ax.set_xlim(min(Ns) - 0.5, max(Ns) + 0.5)
ax.set_ylim(-0.08, 1.12)
ax.set_xticks(Ns)
ax.set_xlabel('Complexity  N',  color='#9999bb', fontsize=9)
ax.set_ylabel('Mean Accuracy',  color='#9999bb', fontsize=9)
ax.set_title('② Phase Diagram', color='#ccccee', fontsize=10, pad=6)
ax.set_facecolor('#0a0a1a')
ax.tick_params(colors='#9999bb', labelsize=8)
for sp in ax.spines.values():
    sp.set_edgecolor('#2a2a4a')
if threshold_N is not None:
    ax.legend(fontsize=7, framealpha=0.3, labelcolor='white')

# ---- ③ トークン数推移 --------------------------------------------------
ax = ax_token
tokens = [row['mean_token_count'] for row in summary]

if threshold_N is not None:
    ax.axvspan(threshold_N - 0.5, max(Ns) + 0.5,
               alpha=0.08, color='#ff4444')
    ax.axvline(x=threshold_N - 0.5, color='#ff6644', ls='--', lw=1.2, alpha=0.7,
               label=f'Collapse point N={threshold_N}')

ax.fill_between(Ns, tokens, alpha=0.15, color='#ffaa44')
ax.plot(Ns, tokens, color='#ffcc55', lw=1.8, zorder=2)
for N, t in zip(Ns, tokens):
    ax.scatter(N, t, color=color_map[N], s=55, zorder=3,
               edgecolors='white', linewidths=0.5)

ax.set_xlim(min(Ns) - 0.5, max(Ns) + 0.5)
ax.set_xticks(Ns)
ax.set_xlabel('Complexity  N',           color='#9999bb', fontsize=9)
ax.set_ylabel('Mean Token Count',        color='#9999bb', fontsize=9)
ax.set_title('③ Reasoning Token Scaling', color='#ccccee', fontsize=10, pad=6)
ax.set_facecolor('#0a0a1a')
ax.tick_params(colors='#9999bb', labelsize=8)
for sp in ax.spines.values():
    sp.set_edgecolor('#2a2a4a')
if threshold_N is not None:
    ax.legend(fontsize=7, framealpha=0.3, labelcolor='white')

# -----------------------------------------------------------------------
fig.savefig(SAVE_PATH, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'\n保存: {SAVE_PATH}')
