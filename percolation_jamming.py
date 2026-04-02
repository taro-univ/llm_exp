"""
LLM Capability Emergence Simulation
Percolation Theory x Jamming Transition Analogy

Experiment 1: Percolation transition -- Giant Component formation
Experiment 2: Jamming transition     -- k-edge-disjoint reasoning paths
"""

import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import brentq

warnings.filterwarnings("ignore")


# ─── Theoretical giant component fraction ────────────────────────────────────

def giant_component_theory(k_bar_arr):
    """
    Solve S = 1 - exp(-<k> * S)  (mean-field, N -> inf limit).
    Returns S=0 for k_bar <= 1.
    """
    out = []
    for kb in k_bar_arr:
        if kb <= 1.0 + 1e-10:
            out.append(0.0)
        else:
            try:
                S = brentq(lambda S: S - (1 - np.exp(-kb * S)), 1e-9, 1 - 1e-9)
            except Exception:
                S = 0.0
            out.append(S)
    return np.array(out)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Percolation transition & giant component
# ─────────────────────────────────────────────────────────────────────────────

def experiment1():
    N        = 1000
    k_values = np.linspace(0.0, 5.0, 80)
    n_trials = 20
    rng      = np.random.default_rng(42)

    print("[Exp1] Percolation simulation  (N=1000, 80 mean-degree values x 20 trials)...")

    sim_fracs = []
    for i, kbar in enumerate(k_values):
        p = kbar / (N - 1)
        fracs = []
        for _ in range(n_trials):
            G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(1_000_000_000)))
            largest = max(nx.connected_components(G), key=len)
            fracs.append(len(largest) / N)
        sim_fracs.append(float(np.mean(fracs)))
        if (i + 1) % 20 == 0:
            print(f"  {i+1:3d}/{len(k_values)}  <k>={kbar:.2f}  S/N={sim_fracs[-1]:.3f}")

    theory = giant_component_theory(k_values)

    # ── Figure: two panels ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: full range ────────────────────────────────────────────────────
    ax1.plot(k_values, sim_fracs, "o", color="#2196F3", markersize=4,
             alpha=0.75, label=f"Simulation  (N={N}, {n_trials} trials avg)")
    ax1.plot(k_values, theory, "-",  color="#FF5722", linewidth=2.5,
             label=r"Theory: $S = 1 - e^{-\langle k\rangle S}$")
    ax1.axvline(1.0, color="#4CAF50", linestyle="--", linewidth=2.2,
                label=r"Critical point $\langle k\rangle_c = 1$  $(p_c = 1/N)$")
    ax1.fill_between(k_values, sim_fracs, 0, alpha=0.10, color="#2196F3")

    ax1.annotate("Sub-critical:\nfragmented knowledge",
                 xy=(0.45, 0.01), xytext=(0.15, 0.28), fontsize=9, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))
    ax1.annotate("Super-critical:\nGiant Component\n(emergent reasoning)",
                 xy=(2.8, float(giant_component_theory([2.8])[0])),
                 xytext=(3.0, 0.35), fontsize=9, color="#2196F3",
                 arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.0))

    ax1.set_xlabel(r"Mean degree $\langle k \rangle$  (model scale proxy)", fontsize=11)
    ax1.set_ylabel("Giant Component fraction  $S/N$", fontsize=11)
    ax1.set_title("Percolation: Knowledge Network Formation\n"
                  "(full range)", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(0, 5); ax1.set_ylim(-0.02, 1.02)

    # ── Right: zoom around critical point ───────────────────────────────────
    mask = (k_values >= 0.5) & (k_values <= 2.5)
    ax2.plot(k_values[mask], np.array(sim_fracs)[mask], "o-",
             color="#2196F3", markersize=5, linewidth=2.0, label="Simulation")
    ax2.plot(k_values[mask], theory[mask], "--",
             color="#FF5722", linewidth=2.5, label="Theory")
    ax2.axvline(1.0, color="#4CAF50", linestyle="--", linewidth=2.2,
                label=r"$\langle k\rangle_c = 1$")

    # Mark the steep rise region
    ax2.fill_between(k_values[mask], np.array(sim_fracs)[mask], 0,
                     alpha=0.12, color="#2196F3")

    ax2.set_xlabel(r"Mean degree $\langle k \rangle$", fontsize=11)
    ax2.set_ylabel("Giant Component fraction  $S/N$", fontsize=11)
    ax2.set_title(r"Zoom: Second-Order Phase Transition near $\langle k\rangle = 1$"
                  "\n(critical slowing-down analogy)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    fig.suptitle(
        "Experiment 1: Percolation Transition & Giant Component Emergence\n"
        r"Erdős–Rényi $G(N{=}1000,\, p)$ — fragmented knowledge → coherent reasoning network",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig("percolation_transition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Exp1] percolation_transition.png saved.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Jamming transition & k-edge-disjoint reasoning paths
# ─────────────────────────────────────────────────────────────────────────────

def _edge_conn_prob(N, kbar, k_target, n_pairs, n_trials, rng):
    """
    Estimate P(random node pair has edge-connectivity >= k_target).
    k=1: fast connected-component check.
    k>=2: max-flow (Menger's theorem).
    """
    p = min(kbar / max(N - 1, 1), 1.0)
    hits = 0
    total = 0

    for _ in range(n_trials):
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(1_000_000_000)))
        nodes = list(G.nodes())
        n = len(nodes)
        if n < 2:
            total += n_pairs
            continue

        # For k=1 build component map once per graph (O(N+E))
        if k_target == 1:
            comp_id = {}
            for cid, comp in enumerate(nx.connected_components(G)):
                for v in comp:
                    comp_id[v] = cid

        for _ in range(n_pairs):
            idx = rng.choice(n, size=2, replace=False)
            s, t = nodes[int(idx[0])], nodes[int(idx[1])]
            if k_target == 1:
                success = comp_id[s] == comp_id[t]
            else:
                try:
                    ec = nx.edge_connectivity(G, s, t)
                    success = ec >= k_target
                except Exception:
                    success = False
            hits  += int(success)
            total += 1

    return hits / max(total, 1)


def experiment2():
    N        = 300
    k_bars   = np.linspace(0.0, 12.0, 50)
    k_vals   = [1, 2, 3]
    n_pairs  = 20
    n_trials = 6
    rng      = np.random.default_rng(42)

    colors = ["#4CAF50", "#FF9800", "#F44336"]
    labels = [
        "k=1 — Simple reasoning  (single connected path)",
        "k=2 — Robust reasoning  (2 edge-disjoint paths)",
        "k=3 — Complex reasoning (3 edge-disjoint paths)",
    ]

    print(f"[Exp2] Jamming simulation  (N={N}, 50 mean-degree values, {n_pairs}x{n_trials} samples per point)...")

    results = {}
    for k in k_vals:
        print(f"  k={k} ...", end="", flush=True)
        probs = []
        for j, kbar in enumerate(k_bars):
            prob = _edge_conn_prob(N, kbar, k, n_pairs, n_trials, rng)
            probs.append(prob)
            if (j + 1) % 25 == 0:
                print(f" <k>={kbar:.1f}->P={prob:.2f}", end="", flush=True)
        results[k] = np.array(probs)
        print()

    # ── Theoretical onset thresholds (asymptotic) ───────────────────────────
    lnN   = math.log(N)
    lnlnN = math.log(lnN)
    thresholds = {
        1: lnN,
        2: lnN + lnlnN,
        3: lnN + 2.0 * lnlnN,
    }
    print(f"  Theory thresholds: " +
          ", ".join(f"k={k}: <k>~{thresholds[k]:.2f}" for k in k_vals))

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, (ax_main, ax_deriv) = plt.subplots(2, 1, figsize=(12, 10),
                                             gridspec_kw={"height_ratios": [3, 1]},
                                             sharex=True)

    # Main panel: probability curves
    for k, color, label in zip(k_vals, colors, labels):
        ax_main.plot(k_bars, results[k], "-o", color=color, markersize=4,
                     linewidth=2.5, label=label, alpha=0.92)
        ax_main.fill_between(k_bars, results[k], 0, alpha=0.07, color=color)

    # Threshold vertical lines
    for k, color in zip(k_vals, colors):
        th = thresholds[k]
        ax_main.axvline(th, color=color, linestyle=":", linewidth=1.8, alpha=0.65)
        ypos = 0.03 + (k - 1) * 0.065
        ax_main.text(th + 0.15, ypos, f"k={k}\nth~{th:.1f}",
                     fontsize=8, color=color, va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec=color, alpha=0.75))

    # Jamming zone shading
    lo, hi = thresholds[1] - 0.3, thresholds[3] + 0.3
    ax_main.axvspan(lo, hi, alpha=0.07, color="purple",
                    label="Jamming zone (k-transition cascade)")
    ax_main.text((lo + hi) / 2, 1.02, "Jamming Zone\n(emergence cascade)",
                 fontsize=9, ha="center", color="purple",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec="purple", alpha=0.85))

    # Phase region labels
    ax_main.text(1.2, 0.50, "Sub-critical\n(disconnected)", fontsize=9,
                 ha="center", color="gray", style="italic")
    ax_main.text(11.0, 0.55, "All paths\nrobust", fontsize=9,
                 ha="center", color="#333", style="italic")

    ax_main.set_ylabel(r"$P(\geq k$ edge-disjoint paths$)$  [reasoning success rate]",
                       fontsize=11)
    ax_main.set_title(
        f"Experiment 2: Jamming Transition & Emergence of Complex Reasoning\n"
        r"Erdős–Rényi $G(N{=}300,\, p)$ — k edge-disjoint paths between random node pairs"
        "\nAnalogy: jamming = multi-path rigidity emerges at staggered scale thresholds",
        fontsize=11,
    )
    ax_main.legend(fontsize=10, loc="upper left")
    ax_main.grid(True, alpha=0.22)
    ax_main.set_xlim(0, 12); ax_main.set_ylim(-0.03, 1.12)

    # Lower panel: d(P)/d(kbar) — "emergence rate" (approximate)
    dk = k_bars[1] - k_bars[0]
    for k, color, label in zip(k_vals, colors, labels):
        deriv = np.gradient(results[k], dk)
        ax_deriv.plot(k_bars, deriv, "-", color=color, linewidth=2.0,
                      alpha=0.85, label=f"k={k}")
        ax_deriv.fill_between(k_bars, deriv, 0, alpha=0.10, color=color)

    ax_deriv.axvspan(lo, hi, alpha=0.07, color="purple")
    ax_deriv.set_xlabel(r"Mean degree $\langle k \rangle$  (model scale proxy)",
                        fontsize=11)
    ax_deriv.set_ylabel(r"$dP/d\langle k\rangle$" + "\n(emergence rate)", fontsize=10)
    ax_deriv.set_title("Emergence rate per unit model scale  "
                       "(higher k peaks later & sharper → abrupt capability emergence)",
                       fontsize=10)
    ax_deriv.legend(fontsize=9, loc="upper right")
    ax_deriv.grid(True, alpha=0.22)
    ax_deriv.set_xlim(0, 12)

    plt.tight_layout()
    plt.savefig("jamming_reasoning_emergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Exp2] jamming_reasoning_emergence.png saved.\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("LLM Capability Emergence: Percolation & Jamming Analogy")
    print("=" * 65)
    experiment1()
    experiment2()
    print("=" * 65)
    print("Done.")
