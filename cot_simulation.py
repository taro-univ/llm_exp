"""
LLM Chain-of-Thought Reasoning Collapse Simulation
Non-equilibrium Statistical Mechanics x Strogatz Saddle-Node Bifurcation

Stochastic difference equation:
  x_{n+1} = x_n - dt * (dV/dx) + sqrt(2*T*dt) * xi_n
Potential function:
  V(x, r) = r*x - (1/3)*x^3
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Potential and its derivative ──────────────────────────────────────────────

def V(x, r):
    """V(x, r) = r*x - (1/3)*x^3"""
    return r * x - (1.0 / 3.0) * x ** 3


def dV_dx(x, r):
    """dV/dx = r - x^2"""
    return r - x ** 2


# ── One stochastic step ───────────────────────────────────────────────────────

def stochastic_step(x, r, T, dt, rng):
    """x_{n+1} = x_n - dt*(dV/dx) + sqrt(2*T*dt)*xi"""
    xi = rng.standard_normal(x.shape)
    return x - dt * dV_dx(x, r) + np.sqrt(2.0 * T * dt) * xi


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Potential landscape vs task difficulty
# ─────────────────────────────────────────────────────────────────────────────

def experiment1():
    fig, ax = plt.subplots(figsize=(10, 6))
    x_range = np.linspace(-2.5, 2.5, 2000)

    configs = [
        (1.0,  "r=1.0  (Easy)",    "#2196F3"),
        (0.2,  "r=0.2  (Hard)",    "#FF9800"),
        (-0.2, "r=-0.2 (Collapse)","#F44336"),
    ]

    for r, label, color in configs:
        V_vals = np.clip(V(x_range, r), -6, 6)
        ax.plot(x_range, V_vals, color=color, linewidth=2.5, label=label)

        if r > 0:
            sqrt_r = np.sqrt(r)

            # Stable fixed point: x = -sqrt(r)  (local minimum of V)
            xs = -sqrt_r
            ax.scatter([xs], [V(xs, r)], color=color, marker="o",
                       s=130, zorder=6, edgecolors="black", linewidths=1.2)
            ax.annotate(
                f"Stable FP\nx=-sqr(r)={xs:.2f}",
                xy=(xs, V(xs, r)),
                xytext=(xs - 0.7, V(xs, r) - 1.2),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            )

            # Unstable fixed point: x = +sqrt(r)  (local maximum = barrier top)
            xu = sqrt_r
            ax.scatter([xu], [V(xu, r)], color=color, marker="^",
                       s=130, zorder=6, edgecolors="black", linewidths=1.2)
            ax.annotate(
                f"Unstable FP\nx=+sqr(r)={xu:.2f}",
                xy=(xu, V(xu, r)),
                xytext=(xu + 0.15, V(xu, r) + 0.7),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("x  (reasoning error from correct context)", fontsize=12)
    ax.set_ylabel("V(x, r)  (potential energy)", fontsize=12)
    ax.set_title(
        "Experiment 1: Potential Landscape vs Task Difficulty\n"
        r"$V(x,r)=rx-\frac{1}{3}x^3$"
        "  --  Saddle-Node Bifurcation (Strogatz)",
        fontsize=12,
    )

    extra = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=9, label="● Stable FP: x=−sqrt(r)  [correct answer valley]"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None",
               markersize=9, label="▲ Unstable FP: x=+sqrt(r)  [collapse barrier]"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra, loc="upper left",
              fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    ax.text(
        1.3, -5.3,
        "r < 0: no fixed points\n→ reasoning always collapses",
        fontsize=8, color="#F44336",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#F44336", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig("potential_bifurcation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Exp1] potential_bifurcation.png saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Multi-agent trajectories & Kramers transition
# ─────────────────────────────────────────────────────────────────────────────

def experiment2():
    r   = 0.1
    T   = 0.05
    dt  = 0.1
    N   = 100          # steps
    M   = 100          # agents
    x0  = np.sqrt(r)   # ~0.3162  (unstable FP -- barrier top)
    rng = np.random.default_rng(seed=42)

    # Integrate trajectories
    traj = np.zeros((M, N + 1))
    traj[:, 0] = x0
    for n in range(N):
        traj[:, n + 1] = stochastic_step(traj[:, n], r, T, dt, rng)

    steps = np.arange(N + 1)

    # Classification: escaped = final x well above starting barrier top
    escaped_mask = traj[:, -1] > 1.5 * x0
    n_escaped = int(escaped_mask.sum())
    n_trapped  = M - n_escaped

    # Kramers theory: barrier height DeltaV = V(+sqrt(r)) - V(-sqrt(r)) = (4/3)*r^(3/2)
    delta_V = (4.0 / 3.0) * r ** 1.5
    kramers_theory = np.exp(-delta_V / T)

    print(f"[Exp2] r={r}, T={T}, deltaV={delta_V:.4f}")
    print(f"  Kramers escape rate (theory): exp(-dV/T)=exp(-{delta_V/T:.3f}) ~ {kramers_theory:.3f}")
    print(f"  Observed: escaped={n_escaped}/{M}, trapped={n_trapped}/{M}")
    print(f"  Observed escape fraction: {n_escaped/M:.3f}")

    # ── Figure layout ──────────────────────────────────────────────────────
    y_clip = 3.0
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.06)
    ax_traj = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharey=ax_traj)

    # Plot trajectories
    for i in range(M):
        y = np.clip(traj[i], -(y_clip + 0.3), y_clip + 0.3)
        color = "#FF3333" if escaped_mask[i] else "#2255CC"
        ax_traj.plot(steps, y, color=color, alpha=0.35, linewidth=0.7)

    # Reference lines for fixed points
    ax_traj.axhline(
        x0, color="#E65100", linestyle="--", linewidth=2.0,
        label=f"Unstable FP: +sqrt(r) ~ {x0:.3f}  [initial position / barrier top]",
    )
    ax_traj.axhline(
        -x0, color="#1A237E", linestyle="--", linewidth=2.0,
        label=f"Stable FP:  -sqrt(r) ~ {-x0:.3f}  [correct answer valley]",
    )

    legend_elements = [
        Line2D([0], [0], color="#FF3333", alpha=0.9, linewidth=2.0,
               label=f"Escaped (Kramers transition): {n_escaped}/{M}"),
        Line2D([0], [0], color="#2255CC", alpha=0.9, linewidth=2.0,
               label=f"Trapped (stable valley):      {n_trapped}/{M}"),
        Line2D([0], [0], color="#E65100", linestyle="--", linewidth=2.0,
               label=f"Unstable FP  x=+sqrt(r) ~ {x0:.3f}"),
        Line2D([0], [0], color="#1A237E", linestyle="--", linewidth=2.0,
               label=f"Stable FP    x=-sqrt(r) ~ {-x0:.3f}"),
    ]
    ax_traj.legend(handles=legend_elements, loc="upper left",
                   fontsize=9, framealpha=0.93)

    ax_traj.set_xlim(0, N)
    ax_traj.set_ylim(-y_clip, y_clip)
    ax_traj.set_xlabel("Step n", fontsize=12)
    ax_traj.set_ylabel("$x_n$  (reasoning error)", fontsize=12)
    ax_traj.set_title(
        f"Experiment 2: Multi-Agent Reasoning Trajectories & Kramers Transition\n"
        f"r={r}, T={T}, N={N} steps, M={M} agents"
        f"   (start: unstable FP  x0=sqrt(r)={x0:.4f})",
        fontsize=11,
    )
    ax_traj.grid(True, alpha=0.2)

    ax_traj.text(
        0.98, 0.03,
        f"deltaV = (4/3)*r^(3/2) = {delta_V:.4f}\n"
        f"T = {T}\n"
        f"Theory escape rate ~ exp(-dV/T) = {kramers_theory:.3f}\n"
        f"Observed escape fraction = {n_escaped/M:.3f}",
        transform=ax_traj.transAxes,
        fontsize=9, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.92),
    )

    # Right panel: final distribution histogram
    final_c = np.clip(traj[:, -1], -y_clip, y_clip)
    bins = np.linspace(-y_clip, y_clip, 40)
    ax_hist.hist(final_c[escaped_mask],  bins=bins, orientation="horizontal",
                 color="#FF3333", alpha=0.75, label="Escaped")
    ax_hist.hist(final_c[~escaped_mask], bins=bins, orientation="horizontal",
                 color="#2255CC", alpha=0.75, label="Trapped")
    ax_hist.axhline( x0, color="#E65100", linestyle="--", linewidth=1.5)
    ax_hist.axhline(-x0, color="#1A237E", linestyle="--", linewidth=1.5)
    ax_hist.set_xlabel("# agents", fontsize=10)
    ax_hist.set_title(f"Final dist.\n(n={N})", fontsize=10)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.2)
    plt.setp(ax_hist.get_yticklabels(), visible=False)

    plt.savefig("reasoning_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Exp2] reasoning_trajectories.png saved.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LLM CoT Reasoning Collapse Simulation")
    print("Non-equilibrium Statistical Mechanics Model")
    print("=" * 60)
    experiment1()
    experiment2()
    print("=" * 60)
    print("Done.")
