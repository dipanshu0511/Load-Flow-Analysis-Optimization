# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:00:18 2026

@author: Dipanshu Tripathi
"""

"""
dg_placement.py

Optimal Placement and Sizing of 3 Type-2 DG Units on the 33-bus radial
distribution system using Particle Swarm Optimization (PSO).

Design decisions:
  - 3 DG units        : covers both weak feeder ends + mid-feeder
  - Type-2 (P + Q)    : pf = 0.85 lagging for maximum voltage support
  - Objective         : 0.6 * (P_loss / P_loss_base)
                      + 0.4 * (V_dev  / V_dev_base)

PSO search space (6 variables — 2 per DG):
    [bus_1, P_1,  bus_2, P_2,  bus_3, P_3]
    bus  ∈ {2 … 33}   (bus 1 = slack, excluded)
    P    ∈ [0.2, 2.5]  MW

Q per DG is derived from: Q = P * tan(arccos(pf))
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from load_data  import load_branch_bus_csv
from lf_solver  import run_dlf_direct
from losses     import calculate_losses


# ======================================================================= #
#  CONSTANTS
# ======================================================================= #

N_DG        = 3
PF          = 0.85                         # DG power factor (lagging)
TAN_PHI     = np.tan(np.arccos(PF))        # Q/P ratio
P_MIN_MW    = 0.2                          # min DG size
P_MAX_MW    = 2.5                          # max DG size
SBASE_MVA   = 100.0
W1          = 0.6                          # loss weight in objective
W2          = 0.4                          # voltage deviation weight

# PSO hyperparameters
N_PARTICLES = 40
MAX_ITER    = 150
W_START     = 0.9
W_END       = 0.4
C1          = 2.0
C2          = 2.0


# ======================================================================= #
#  INJECT DG INTO S_bus_pu
# ======================================================================= #

def apply_dg(S_bus_pu: np.ndarray,
             dg_buses: list,
             dg_P_mw:  list) -> np.ndarray:
    """
    Return a new S_bus_pu with DG injections subtracted from bus loads.

    Generation convention: DG reduces net load at the bus.
        S_net[bus] = S_load[bus] - (P_dg + j*Q_dg)

    Parameters
    ----------
    S_bus_pu : np.ndarray  original bus loads (complex, p.u.)
    dg_buses : list[int]   bus numbers (1-indexed)
    dg_P_mw  : list[float] DG real power (MW)

    Returns
    -------
    S_new : np.ndarray  modified bus power array
    """
    S_new = S_bus_pu.copy()
    for bus, P_mw in zip(dg_buses, dg_P_mw):
        idx    = bus - 1
        P_pu   = P_mw / SBASE_MVA
        Q_pu   = P_pu * TAN_PHI
        S_new[idx] -= complex(P_pu, Q_pu)
    return S_new


# ======================================================================= #
#  DECODE PARTICLE → DG BUSES AND SIZES
# ======================================================================= #

def decode(particle: np.ndarray):
    """
    Decode a 6-element PSO particle into DG bus numbers and P sizes.
    Buses are rounded and clamped to [2, 33].
    P values are clamped to [P_MIN_MW, P_MAX_MW].
    Duplicate bus assignment is handled by shifting the second duplicate.
    """
    buses = []
    sizes = []
    for k in range(N_DG):
        bus = int(np.round(np.clip(particle[2*k],     2,        33)))
        P   = float(np.clip(particle[2*k + 1], P_MIN_MW, P_MAX_MW))
        buses.append(bus)
        sizes.append(P)

    # Resolve duplicate buses by nudging duplicates to nearest free bus
    for i in range(len(buses)):
        for j in range(i + 1, len(buses)):
            if buses[i] == buses[j]:
                candidate = buses[j] + 1
                while candidate in buses and candidate <= 33:
                    candidate += 1
                if candidate > 33:
                    candidate = buses[j] - 1
                    while candidate in buses and candidate >= 2:
                        candidate -= 1
                buses[j] = max(2, min(33, candidate))

    return buses, sizes


# ======================================================================= #
#  OBJECTIVE FUNCTION
# ======================================================================= #

def fitness(particle:     np.ndarray,
            branch_df:    pd.DataFrame,
            z_branch_pu:  np.ndarray,
            S_bus_base:   np.ndarray,
            P_loss_base:  float,
            V_dev_base:   float) -> float:
    """
    Evaluate objective for one particle.

    f = W1 * (P_loss / P_loss_base)  +  W2 * (V_dev / V_dev_base)

    Returns large penalty (1e6) for invalid solutions.
    """
    buses, sizes = decode(particle)

    # Build modified load vector
    S_mod = apply_dg(S_bus_base, buses, sizes)

    # Run load flow directly — no file I/O, no monkey patching
    try:
        v = run_dlf_direct(branch_df, z_branch_pu, S_mod)
    except Exception:
        return 1e6

    # Reject if any voltage is wildly out of range
    Vmag = np.abs(v)
    if Vmag.min() < 0.5 or Vmag.max() > 1.1:
        return 1e6

    # Total real power loss
    _, summary = calculate_losses(v, branch_df, z_branch_pu, S_mod,
                                  sbase_mva=SBASE_MVA)
    P_loss = summary["Total_P_loss_kW"]

    # Voltage deviation: sum of (1 - |Vi|)^2
    V_dev = float(np.sum((1.0 - Vmag) ** 2))

    # Normalised combined objective
    f_loss = P_loss / P_loss_base if P_loss_base > 0 else P_loss
    f_vdev = V_dev  / V_dev_base  if V_dev_base  > 0 else V_dev

    return W1 * f_loss + W2 * f_vdev


# ======================================================================= #
#  PSO
# ======================================================================= #

def run_pso(branch_df, z_branch_pu, S_bus_base,
            P_loss_base, V_dev_base):
    """
    Run PSO optimisation.

    Returns
    -------
    best_particle : np.ndarray   optimal PSO solution (6 values)
    best_fitness  : float
    history       : list[float]  best fitness per iteration
    """
    np.random.seed(0)
    dim = 2 * N_DG     # 6

    # Bounds: [bus_min, P_min, bus_min, P_min, ...]
    lb = np.array([2,       P_MIN_MW] * N_DG, dtype=float)
    ub = np.array([33,      P_MAX_MW] * N_DG, dtype=float)

    # Initialise swarm
    pos = lb + np.random.rand(N_PARTICLES, dim) * (ub - lb)
    vel = np.zeros_like(pos)

    # Evaluate initial fitness
    fit = np.array([
        fitness(pos[i], branch_df, z_branch_pu,
                S_bus_base, P_loss_base, V_dev_base)
        for i in range(N_PARTICLES)
    ])

    pbest_pos = pos.copy()
    pbest_fit = fit.copy()

    g_idx     = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    history = [gbest_fit]

    print(f"\n  PSO: {N_PARTICLES} particles × {MAX_ITER} iterations")
    print(f"  DGs: {N_DG}  |  PF: {PF}  |  P range: "
          f"[{P_MIN_MW}, {P_MAX_MW}] MW")
    print(f"  Objective weights: loss={W1}, Vdev={W2}")
    print(f"\n  {'Iter':>5}  {'Best Fit':>10}  "
          + "  ".join([f"Bus{k+1:>1} P{k+1}(MW)" for k in range(N_DG)]))
    print("  " + "-" * 70)

    for it in range(MAX_ITER):
        # Linearly decay inertia
        w = W_START - (W_START - W_END) * it / MAX_ITER

        r1 = np.random.rand(N_PARTICLES, dim)
        r2 = np.random.rand(N_PARTICLES, dim)

        vel = (w  * vel
               + C1 * r1 * (pbest_pos - pos)
               + C2 * r2 * (gbest_pos  - pos))
        pos = np.clip(pos + vel, lb, ub)

        for i in range(N_PARTICLES):
            f = fitness(pos[i], branch_df, z_branch_pu,
                        S_bus_base, P_loss_base, V_dev_base)
            if f < pbest_fit[i]:
                pbest_fit[i] = f
                pbest_pos[i] = pos[i].copy()
            if f < gbest_fit:
                gbest_fit = f
                gbest_pos = pos[i].copy()

        history.append(gbest_fit)

        if (it + 1) % 25 == 0 or it == 0:
            buses, sizes = decode(gbest_pos)
            vals = "  ".join([f"{buses[k]:>4d}  {sizes[k]:>6.3f}"
                               for k in range(N_DG)])
            print(f"  {it+1:>5}  {gbest_fit:>10.5f}  {vals}")

    print("  " + "-" * 70)
    print(f"  Converged — best fitness: {gbest_fit:.6f}\n")

    return gbest_pos, gbest_fit, history


# ======================================================================= #
#  PRINT SUMMARY
# ======================================================================= #

def print_summary(dg_buses, dg_P_mw, dg_Q_mvar,
                  summary_base, summary_dg,
                  v_base, v_dg):

    print("\n" + "="*65)
    print("  OPTIMAL DG PLACEMENT — RESULTS SUMMARY")
    print("="*65)
    print(f"  Type-2 DGs  |  PF = {PF}  |  tan(φ) = {TAN_PHI:.4f}\n")

    for k in range(N_DG):
        print(f"  DG-{k+1}  Bus {dg_buses[k]:>2d}  "
              f"P = {dg_P_mw[k]:.4f} MW  "
              f"Q = {dg_Q_mvar[k]:.4f} MVAr")

    print()
    hdr = f"  {'Metric':<38} {'Base':>10} {'With DG':>10} {'Reduc%':>8}"
    print(hdr)
    print("  " + "-"*65)

    def row(label, bv, dv):
        red = (bv - dv) / bv * 100 if bv != 0 else 0
        print(f"  {label:<38} {bv:>10.3f} {dv:>10.3f} {red:>7.2f}%")

    row("Total P Loss (kW)",
        summary_base["Total_P_loss_kW"],  summary_dg["Total_P_loss_kW"])
    row("Total Q Loss (kVAR)",
        summary_base["Total_Q_loss_kVAR"], summary_dg["Total_Q_loss_kVAR"])
    row("% Real Power Loss",
        summary_base["Pct_P_loss"],        summary_dg["Pct_P_loss"])

    Vmag_base = np.abs(v_base)
    Vmag_dg   = np.abs(v_dg)
    vmin_b = Vmag_base.min()
    vmin_d = Vmag_dg.min()
    vbus_b = Vmag_base.argmin() + 1
    vbus_d = Vmag_dg.argmin()   + 1
    below_b = (Vmag_base < 0.95).sum()
    below_d = (Vmag_dg   < 0.95).sum()

    print(f"\n  {'Min Voltage (p.u.)':<38} "
          f"{vmin_b:>10.5f} {vmin_d:>10.5f} "
          f"{(vmin_d-vmin_b)/vmin_b*100:>7.2f}%")
    print(f"  {'Bus of Min Voltage':<38} "
          f"{vbus_b:>10d} {vbus_d:>10d}")
    print(f"  {'Buses below 0.95 p.u.':<38} "
          f"{below_b:>10d} {below_d:>10d}")
    print("="*65)


# ======================================================================= #
#  PLOTS
# ======================================================================= #

def plot_convergence(history, results_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history, '-', color='steelblue', linewidth=2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title("PSO Convergence — Optimal DG Placement (33-Bus System)",
                 fontsize=13)
    ax.grid(True, alpha=0.6)
    plt.tight_layout()
    path = os.path.join(results_dir, "pso_convergence.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  PSO convergence plot          : {path}")


def plot_voltage_comparison(v_base, v_dg, dg_buses, results_dir):
    buses = np.arange(1, len(v_base) + 1)
    Vb    = np.abs(v_base)
    Vd    = np.abs(v_dg)

    ymin = min(Vb.min(), Vd.min()) - 0.02
    ymax = max(Vb.max(), Vd.max()) + 0.02

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(buses, Vb, 'o-', color='firebrick',  linewidth=2,
            markersize=5, label='Base Case (No DG)')
    ax.plot(buses, Vd, 's-', color='seagreen',   linewidth=2,
            markersize=5, label=f'With {N_DG} Optimal DGs (PF={PF})')
    ax.axhline(0.95, color='gray', linestyle=':', linewidth=1.3,
               label='0.95 p.u. limit')

    colors_dg = ['steelblue', 'darkorange', 'purple']
    for k, bus in enumerate(dg_buses):
        ax.axvline(bus, color=colors_dg[k], linestyle='--',
                   linewidth=1.3, alpha=0.8,
                   label=f'DG-{k+1} @ Bus {bus}')

    ax.set_xlabel("Bus Number", fontsize=12)
    ax.set_ylabel("Voltage Magnitude (p.u.)", fontsize=12)
    ax.set_title("Voltage Profile — Base Case vs Optimal DG Placement",
                 fontsize=13)
    ax.set_xlim([1, len(buses)])
    ax.set_ylim([ymin, ymax])
    ax.set_xticks(buses)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.6)
    plt.tight_layout()
    path = os.path.join(results_dir, "voltage_profile_dg_vs_base.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Voltage comparison plot       : {path}")


def plot_loss_comparison(loss_base, loss_dg,
                         summary_base, summary_dg, results_dir):
    branches = loss_base["Branch"].to_numpy()
    x        = np.arange(len(branches))
    w        = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Real power losses
    ax = axes[0]
    ax.bar(x - w/2, loss_base["P_loss(kW)"], w, color='firebrick',
           alpha=0.85, edgecolor='black', linewidth=0.4,
           label=f'Base  ({summary_base["Total_P_loss_kW"]:.1f} kW)')
    ax.bar(x + w/2, loss_dg["P_loss(kW)"],   w, color='seagreen',
           alpha=0.85, edgecolor='black', linewidth=0.4,
           label=f'DG    ({summary_dg["Total_P_loss_kW"]:.1f} kW)')
    ax.set_xlabel("Branch Number", fontsize=11)
    ax.set_ylabel("P Loss (kW)", fontsize=11)
    ax.set_title("Real Power Losses — Base vs DG", fontsize=12,
                 fontweight='bold')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(branches[::2], fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.5)

    # Reactive power losses
    ax = axes[1]
    ax.bar(x - w/2, loss_base["Q_loss(kVAR)"], w, color='firebrick',
           alpha=0.85, edgecolor='black', linewidth=0.4,
           label=f'Base  ({summary_base["Total_Q_loss_kVAR"]:.1f} kVAR)')
    ax.bar(x + w/2, loss_dg["Q_loss(kVAR)"],   w, color='seagreen',
           alpha=0.85, edgecolor='black', linewidth=0.4,
           label=f'DG    ({summary_dg["Total_Q_loss_kVAR"]:.1f} kVAR)')
    ax.set_xlabel("Branch Number", fontsize=11)
    ax.set_ylabel("Q Loss (kVAR)", fontsize=11)
    ax.set_title("Reactive Power Losses — Base vs DG", fontsize=12,
                 fontweight='bold')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(branches[::2], fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.5)

    plt.suptitle("Branch Power Losses — Base Case vs Optimal DG Placement",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(results_dir, "branch_losses_dg_vs_base.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Loss comparison plot          : {path}")


def plot_dg_dashboard(v_base, v_dg, loss_base, loss_dg,
                      summary_base, summary_dg,
                      history, dg_buses, dg_P_mw, dg_Q_mvar,
                      results_dir):
    buses    = np.arange(1, len(v_base) + 1)
    branches = loss_base["Branch"].to_numpy()
    x        = np.arange(len(branches))
    w        = 0.35

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

    # ── Convergence ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history, '-', color='steelblue', linewidth=2)
    ax1.set_xlabel("Iteration", fontsize=10)
    ax1.set_ylabel("Fitness", fontsize=10)
    ax1.set_title("PSO Convergence", fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.5)

    # ── Voltage profile ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(buses, np.abs(v_base), 'o-', color='firebrick',
             linewidth=2, markersize=4, label='Base Case')
    ax2.plot(buses, np.abs(v_dg),   's-', color='seagreen',
             linewidth=2, markersize=4, label='With DG')
    ax2.axhline(0.95, color='gray', linestyle=':', linewidth=1.2,
                label='0.95 p.u.')
    clrs = ['steelblue', 'darkorange', 'purple']
    for k, bus in enumerate(dg_buses):
        ax2.axvline(bus, color=clrs[k], linestyle='--',
                    linewidth=1.2, alpha=0.8,
                    label=f'DG-{k+1}@{bus}')
    ax2.set_xlabel("Bus Number", fontsize=10)
    ax2.set_ylabel("Voltage (p.u.)", fontsize=10)
    ax2.set_title("Voltage Profile", fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower left', ncol=2)
    ax2.grid(True, alpha=0.5)

    # ── P Loss bars ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.bar(x - w/2, loss_base["P_loss(kW)"], w, color='firebrick',
            alpha=0.85, edgecolor='black', linewidth=0.3,
            label=f'Base ({summary_base["Total_P_loss_kW"]:.1f} kW)')
    ax3.bar(x + w/2, loss_dg["P_loss(kW)"],   w, color='seagreen',
            alpha=0.85, edgecolor='black', linewidth=0.3,
            label=f'DG   ({summary_dg["Total_P_loss_kW"]:.1f} kW)')
    ax3.set_xlabel("Branch", fontsize=10)
    ax3.set_ylabel("P Loss (kW)", fontsize=10)
    ax3.set_title("Real Power Losses", fontsize=11, fontweight='bold')
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels(branches[::2], fontsize=8)
    ax3.legend(fontsize=9)
    ax3.grid(True, axis='y', alpha=0.5)

    # ── Summary text ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')

    Vmag_b  = np.abs(v_base)
    Vmag_d  = np.abs(v_dg)
    pl_red  = (1 - summary_dg["Total_P_loss_kW"]   /
                   summary_base["Total_P_loss_kW"])   * 100
    ql_red  = (1 - summary_dg["Total_Q_loss_kVAR"]  /
                   summary_base["Total_Q_loss_kVAR"]) * 100
    vm_imp  = (Vmag_d.min() - Vmag_b.min()) * 100

    txt = "─── RESULTS SUMMARY ───\n\n"
    txt += f"  PF = {PF}  |  DGs = {N_DG}\n\n"
    for k in range(N_DG):
        txt += (f"  DG-{k+1} @ Bus {dg_buses[k]:>2d}\n"
                f"    P = {dg_P_mw[k]:.3f} MW\n"
                f"    Q = {dg_Q_mvar[k]:.3f} MVAr\n\n")
    txt += (f"  P Loss:\n"
            f"    {summary_base['Total_P_loss_kW']:.2f} → "
            f"{summary_dg['Total_P_loss_kW']:.2f} kW\n"
            f"    ↓ {pl_red:.2f}% reduction\n\n"
            f"  Q Loss:\n"
            f"    {summary_base['Total_Q_loss_kVAR']:.2f} → "
            f"{summary_dg['Total_Q_loss_kVAR']:.2f} kVAR\n"
            f"    ↓ {ql_red:.2f}% reduction\n\n"
            f"  Vmin:\n"
            f"    {Vmag_b.min():.4f} → {Vmag_d.min():.4f} p.u.\n"
            f"    ↑ {vm_imp:.3f}% improvement\n\n"
            f"  Buses < 0.95 p.u.:\n"
            f"    {(Vmag_b < 0.95).sum()} → {(Vmag_d < 0.95).sum()}")

    ax4.text(0.05, 0.97, txt,
             transform=ax4.transAxes,
             fontsize=9, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#fffde7', alpha=0.9))

    fig.suptitle(
        f"33-Bus System — Optimal DG Placement Dashboard\n"
        f"{N_DG}× Type-2 DG (PF={PF}) via PSO  |  "
        f"Obj = {W1}×P_loss + {W2}×V_dev",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(results_dir, "dg_optimization_dashboard.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  DG dashboard                  : {path}")


# ======================================================================= #
#  SAVE CSVs
# ======================================================================= #

def save_csv(v_base, v_dg, loss_base, loss_dg,
             dg_buses, dg_P_mw, dg_Q_mvar, results_dir):

    nb = len(v_base)
    dg_flag = np.zeros(nb, dtype=int)
    for b in dg_buses:
        dg_flag[b - 1] = 1

    # Voltage
    pd.DataFrame({
        "Bus"                  : np.arange(1, nb + 1),
        "DG_Installed"         : dg_flag,
        "Base_Vmag(pu)"        : np.abs(v_base),
        "DG_Vmag(pu)"          : np.abs(v_dg),
        "Vmag_Improvement(pu)" : np.abs(v_dg) - np.abs(v_base),
        "Base_Vang(deg)"       : np.angle(v_base, deg=True),
        "DG_Vang(deg)"         : np.angle(v_dg,   deg=True),
    }).to_csv(os.path.join(results_dir, "dg_voltage_comparison.csv"),
              index=False)
    print(f"  Voltage CSV                   : "
          f"{results_dir}/dg_voltage_comparison.csv")

    # Losses
    pd.DataFrame({
        "Branch"                : loss_base["Branch"],
        "From_Bus"              : loss_base["From_Bus"],
        "To_Bus"                : loss_base["To_Bus"],
        "Base_P_loss(kW)"       : loss_base["P_loss(kW)"],
        "DG_P_loss(kW)"         : loss_dg["P_loss(kW)"],
        "P_Reduction(kW)"       : loss_base["P_loss(kW)"] -
                                   loss_dg["P_loss(kW)"],
        "Base_Q_loss(kVAR)"     : loss_base["Q_loss(kVAR)"],
        "DG_Q_loss(kVAR)"       : loss_dg["Q_loss(kVAR)"],
        "Q_Reduction(kVAR)"     : loss_base["Q_loss(kVAR)"] -
                                   loss_dg["Q_loss(kVAR)"],
    }).to_csv(os.path.join(results_dir, "dg_loss_comparison.csv"),
              index=False)
    print(f"  Loss CSV                      : "
          f"{results_dir}/dg_loss_comparison.csv")

    # DG summary
    pd.DataFrame({
        "DG"     : [f"DG-{k+1}" for k in range(N_DG)],
        "Bus"    : dg_buses,
        "P(MW)"  : dg_P_mw,
        "Q(MVAr)": dg_Q_mvar,
        "PF"     : [PF] * N_DG,
    }).to_csv(os.path.join(results_dir, "dg_placement_summary.csv"),
              index=False)
    print(f"  DG placement CSV              : "
          f"{results_dir}/dg_placement_summary.csv")


# ======================================================================= #
#  MAIN (standalone run)
# ======================================================================= #

def run_dg_optimization(branch_csv, bus_csv, results_dir,
                         branch_df, z_branch_pu, S_bus_pu,
                         v_base, loss_base, summary_base):
    """
    Entry point called from main.py.
    All pre-loaded data is passed in — no re-reading of CSVs.

    Returns
    -------
    v_dg       : np.ndarray  voltages after DG placement
    summary_dg : dict        loss summary after DG placement
    """
    P_loss_base = summary_base["Total_P_loss_kW"]
    V_dev_base  = float(np.sum((1.0 - np.abs(v_base)) ** 2))

    print(f"  Base P loss      : {P_loss_base:.3f} kW")
    print(f"  Base Vmin        : {np.abs(v_base).min():.5f} p.u."
          f"  @ Bus {np.abs(v_base).argmin() + 1}")
    print(f"  Buses < 0.95 p.u.: {(np.abs(v_base) < 0.95).sum()}")

    # ── Run PSO ────────────────────────────────────────────────────────
    best_pos, best_fit, history = run_pso(
        branch_df, z_branch_pu, S_bus_pu,
        P_loss_base, V_dev_base
    )

    # ── Decode solution ────────────────────────────────────────────────
    dg_buses, dg_P_mw = decode(best_pos)
    dg_Q_mvar = [round(P * TAN_PHI, 4) for P in dg_P_mw]
    dg_P_mw   = [round(P, 4) for P in dg_P_mw]

    # ── Load flow with optimal DG ──────────────────────────────────────
    S_dg  = apply_dg(S_bus_pu, dg_buses, dg_P_mw)
    v_dg  = run_dlf_direct(branch_df, z_branch_pu, S_dg)
    loss_dg, summary_dg = calculate_losses(
        v_dg, branch_df, z_branch_pu, S_dg, sbase_mva=SBASE_MVA
    )

    # ── Print and save ─────────────────────────────────────────────────
    print_summary(dg_buses, dg_P_mw, dg_Q_mvar,
                  summary_base, summary_dg, v_base, v_dg)
    save_csv(v_base, v_dg, loss_base, loss_dg,
             dg_buses, dg_P_mw, dg_Q_mvar, results_dir)
    plot_convergence(history, results_dir)
    plot_voltage_comparison(v_base, v_dg, dg_buses, results_dir)
    plot_loss_comparison(loss_base, loss_dg,
                         summary_base, summary_dg, results_dir)
    plot_dg_dashboard(v_base, v_dg, loss_base, loss_dg,
                      summary_base, summary_dg,
                      history, dg_buses, dg_P_mw, dg_Q_mvar,
                      results_dir)

    return v_dg, summary_dg


if __name__ == "__main__":
    import os
    base_path   = os.path.dirname(os.path.abspath(__file__))
    data_path   = os.path.join(base_path, "data")
    results_dir = os.path.join(base_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    branch_csv = os.path.join(data_path, "branch_data.csv")
    bus_csv    = os.path.join(data_path, "bus_data.csv")

    branch_df, bus_df, z_branch_pu, S_bus_pu, z_base = \
        load_branch_bus_csv(branch_csv, bus_csv)

    v_base = run_dlf_direct(branch_df, z_branch_pu, S_bus_pu)
    loss_base, summary_base = calculate_losses(
        v_base, branch_df, z_branch_pu, S_bus_pu)

    run_dg_optimization(branch_csv, bus_csv, results_dir,
                        branch_df, z_branch_pu, S_bus_pu,
                        v_base, loss_base, summary_base)