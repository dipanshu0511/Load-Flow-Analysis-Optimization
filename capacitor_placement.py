# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:59:11 2026

@author: Dipanshu Tripathi
"""

"""
capacitor_placement.py

Optimal Placement and Sizing of Shunt Capacitor Banks on the 33-bus
radial distribution system using Particle Swarm Optimization (PSO).

Design decisions:
  - 3 capacitor banks     : one per weak zone (mid-feeder, end-feeder 1,
                            end-feeder 2)
  - Discrete sizing       : capacitors come in standard kVAR steps
                            [50, 100, 150, 200, 300, 400, 500, 600, 900]
  - Objective (combined)  : 0.6 * (P_loss / P_loss_base)
                          + 0.4 * (V_dev  / V_dev_base)

Capacitors inject ONLY reactive power (Q) at their connected bus.
This is modelled as negative reactive load:
    S_new[bus] = S_load[bus] - j * Q_cap_pu

PSO search space (6 variables — 2 per capacitor):
    [bus_1, size_idx_1,  bus_2, size_idx_2,  bus_3, size_idx_3]
    bus      ∈ [2, 33]   (bus 1 = slack, excluded)
    size_idx ∈ [0, 8]    index into QCAP_STEPS_KVAR

All pre-loaded data is passed in from main.py — no CSV re-reading,
no monkey-patching, no module-level side effects.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from load_data import load_branch_bus_csv
from lf_solver import run_dlf_direct
from losses    import calculate_losses


# ======================================================================= #
#  CONSTANTS
# ======================================================================= #

N_CAP          = 3                              # number of capacitor banks
SBASE_MVA      = 100.0
SBASE_KVA      = SBASE_MVA * 1000.0

# Standard discrete capacitor sizes in kVAR
QCAP_STEPS_KVAR = np.array([50, 100, 150, 200, 300, 400, 500, 600, 900],
                             dtype=float)
N_STEPS         = len(QCAP_STEPS_KVAR)         # 9 discrete sizes

# Objective weights
W1 = 0.6        # loss weight
W2 = 0.4        # voltage deviation weight

# PSO hyperparameters
N_PARTICLES = 40
MAX_ITER    = 150
W_START     = 0.9
W_END       = 0.4
C1          = 2.0
C2          = 2.0


# ======================================================================= #
#  APPLY CAPACITOR INJECTION INTO S_bus_pu
# ======================================================================= #

def apply_capacitors(S_bus_pu: np.ndarray,
                     cap_buses: list,
                     cap_Q_kvar: list) -> np.ndarray:
    """
    Return a new S_bus_pu with capacitor Q injections applied.

    Capacitors inject reactive power (Q) only — no real power.
    Convention: capacitor reduces net reactive load at the bus.
        S_new[bus] = S_load[bus] - j * Q_cap_pu

    Parameters
    ----------
    S_bus_pu   : np.ndarray   original complex bus loads (p.u.)
    cap_buses  : list[int]    bus numbers (1-indexed)
    cap_Q_kvar : list[float]  capacitor sizes in kVAR

    Returns
    -------
    S_new : np.ndarray   modified bus power array (complex, p.u.)
    """
    S_new = S_bus_pu.copy()
    for bus, Q_kvar in zip(cap_buses, cap_Q_kvar):
        idx   = bus - 1
        Q_pu  = Q_kvar / SBASE_KVA
        S_new[idx] -= complex(0.0, Q_pu)       # pure Q injection
    return S_new


# ======================================================================= #
#  DECODE PARTICLE → CAPACITOR BUSES AND SIZES
# ======================================================================= #

def decode(particle: np.ndarray):
    """
    Decode a (2*N_CAP)-element PSO particle into capacitor buses and kVAR sizes.

    particle layout:  [bus_0, idx_0,  bus_1, idx_1,  bus_2, idx_2]
    bus      : rounded and clamped to [2, 33]
    size_idx : rounded and clamped to [0, N_STEPS-1], then looked up
               in QCAP_STEPS_KVAR

    Duplicate buses are resolved by shifting to nearest free bus.

    Returns
    -------
    buses    : list[int]    bus numbers
    Q_kvar   : list[float]  capacitor sizes in kVAR
    """
    buses  = []
    Q_kvar = []

    for k in range(N_CAP):
        bus = int(np.round(np.clip(particle[2 * k],     2,          33)))
        idx = int(np.round(np.clip(particle[2 * k + 1], 0, N_STEPS - 1)))
        buses.append(bus)
        Q_kvar.append(float(QCAP_STEPS_KVAR[idx]))

    # Resolve duplicate bus assignments
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
                buses[j] = int(np.clip(candidate, 2, 33))

    return buses, Q_kvar


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
    Evaluate combined objective for one PSO particle.

    f = W1 * (P_loss / P_loss_base)  +  W2 * (V_dev / V_dev_base)

    Returns 1e6 (large penalty) for numerically invalid solutions.
    """
    buses, Q_kvar = decode(particle)

    # Build modified load vector with capacitor injection
    S_mod = apply_capacitors(S_bus_base, buses, Q_kvar)

    # Run load flow — direct call, no CSV I/O
    try:
        v = run_dlf_direct(branch_df, z_branch_pu, S_mod)
    except Exception:
        return 1e6

    # Reject physically unreasonable voltages
    Vmag = np.abs(v)
    if Vmag.min() < 0.5 or Vmag.max() > 1.1:
        return 1e6

    # Total real power loss via calculate_losses
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
#  PSO ALGORITHM
# ======================================================================= #

def run_pso(branch_df:    pd.DataFrame,
            z_branch_pu:  np.ndarray,
            S_bus_base:   np.ndarray,
            P_loss_base:  float,
            V_dev_base:   float):
    """
    Run PSO to find optimal capacitor bus locations and kVAR sizes.

    Search space (2*N_CAP = 6 dimensions):
        dim 2k   : capacitor k bus     ∈ [2, 33]
        dim 2k+1 : capacitor k size idx ∈ [0, N_STEPS-1]

    Returns
    -------
    best_particle : np.ndarray   optimal PSO solution
    best_fitness  : float
    history       : list[float]  best fitness per iteration
    """
    np.random.seed(1)
    dim = 2 * N_CAP     # 6

    # Bounds
    lb = np.array([2,   0          ] * N_CAP, dtype=float)
    ub = np.array([33,  N_STEPS - 1] * N_CAP, dtype=float)

    # Initialise swarm uniformly within bounds
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
    print(f"  Caps: {N_CAP}  |  Sizes (kVAR): {QCAP_STEPS_KVAR.tolist()}")
    print(f"  Objective weights: loss={W1}, Vdev={W2}")
    print(f"\n  {'Iter':>5}  {'Best Fit':>10}  "
          + "  ".join([f"Bus{k+1}  Q{k+1}(kVAR)" for k in range(N_CAP)]))
    print("  " + "-" * 68)

    for it in range(MAX_ITER):
        # Linearly decay inertia weight
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

        # Print progress every 25 iterations and on iteration 0
        if (it + 1) % 25 == 0 or it == 0:
            buses, Q_kvar = decode(gbest_pos)
            vals = "  ".join([f"{buses[k]:>4d}  {Q_kvar[k]:>8.0f}"
                               for k in range(N_CAP)])
            print(f"  {it+1:>5}  {gbest_fit:>10.5f}  {vals}")

    print("  " + "-" * 68)
    print(f"  Converged — best fitness: {gbest_fit:.6f}\n")

    return gbest_pos, gbest_fit, history


# ======================================================================= #
#  PRINT SUMMARY
# ======================================================================= #

def print_cap_summary(cap_buses:    list,
                      cap_Q_kvar:   list,
                      summary_base: dict,
                      summary_cap:  dict,
                      v_base:       np.ndarray,
                      v_cap:        np.ndarray):
    """Print formatted terminal summary of capacitor optimization results."""

    print("\n" + "="*65)
    print("  OPTIMAL CAPACITOR PLACEMENT — RESULTS SUMMARY")
    print("="*65)
    print(f"  Capacitor banks : {N_CAP}")
    print(f"  Sizing steps    : {QCAP_STEPS_KVAR.tolist()} kVAR\n")

    total_Q = sum(cap_Q_kvar)
    for k in range(N_CAP):
        print(f"  Cap-{k+1}  Bus {cap_buses[k]:>2d}  "
              f"Q = {cap_Q_kvar[k]:.0f} kVAR")
    print(f"  Total Q installed : {total_Q:.0f} kVAR\n")

    print(f"  {'Metric':<38} {'Base':>10} {'With Cap':>10} {'Reduc%':>8}")
    print("  " + "-"*65)

    def row(label, bv, dv):
        red = (bv - dv) / bv * 100 if bv != 0 else 0.0
        print(f"  {label:<38} {bv:>10.3f} {dv:>10.3f} {red:>7.2f}%")

    row("Total P Loss (kW)",
        summary_base["Total_P_loss_kW"],   summary_cap["Total_P_loss_kW"])
    row("Total Q Loss (kVAR)",
        summary_base["Total_Q_loss_kVAR"], summary_cap["Total_Q_loss_kVAR"])
    row("% Real Power Loss",
        summary_base["Pct_P_loss"],        summary_cap["Pct_P_loss"])

    Vmag_b  = np.abs(v_base)
    Vmag_c  = np.abs(v_cap)
    vmin_b  = Vmag_b.min()
    vmin_c  = Vmag_c.min()

    print(f"\n  {'Min Voltage (p.u.)':<38} "
          f"{vmin_b:>10.5f} {vmin_c:>10.5f} "
          f"{(vmin_c - vmin_b) / vmin_b * 100:>7.2f}%")
    print(f"  {'Bus of Min Voltage':<38} "
          f"{Vmag_b.argmin()+1:>10d} {Vmag_c.argmin()+1:>10d}")
    print(f"  {'Buses below 0.95 p.u.':<38} "
          f"{(Vmag_b < 0.95).sum():>10d} {(Vmag_c < 0.95).sum():>10d}")
    print("="*65)


# ======================================================================= #
#  PLOTS
# ======================================================================= #

def plot_convergence(history: list, results_dir: str):
    """PSO convergence curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history, '-', color='teal', linewidth=2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title(
        "PSO Convergence — Optimal Capacitor Placement (33-Bus System)",
        fontsize=13)
    ax.grid(True, alpha=0.6)
    plt.tight_layout()
    path = os.path.join(results_dir, "cap_pso_convergence.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  PSO convergence plot          : {path}")


def plot_voltage_comparison(v_base:    np.ndarray,
                            v_cap:     np.ndarray,
                            cap_buses: list,
                            results_dir: str):
    """Voltage profile before and after capacitor placement."""
    buses   = np.arange(1, len(v_base) + 1)
    Vmag_b  = np.abs(v_base)
    Vmag_c  = np.abs(v_cap)
    ymin    = min(Vmag_b.min(), Vmag_c.min()) - 0.02
    ymax    = max(Vmag_b.max(), Vmag_c.max()) + 0.02

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(buses, Vmag_b, 'o-', color='firebrick', linewidth=2,
            markersize=5, label='Base Case (No Capacitors)')
    ax.plot(buses, Vmag_c, 's-', color='teal',      linewidth=2,
            markersize=5,
            label=f'With {N_CAP} Optimal Capacitor Banks')
    ax.axhline(0.95, color='gray', linestyle=':', linewidth=1.3,
               label='0.95 p.u. limit')

    colors_cap = ['steelblue', 'darkorange', 'purple']
    for k, bus in enumerate(cap_buses):
        ax.axvline(bus, color=colors_cap[k], linestyle='--',
                   linewidth=1.3, alpha=0.8,
                   label=f'Cap-{k+1} @ Bus {bus}')

    ax.set_xlabel("Bus Number", fontsize=12)
    ax.set_ylabel("Voltage Magnitude (p.u.)", fontsize=12)
    ax.set_title(
        "Voltage Profile — Base Case vs Optimal Capacitor Placement",
        fontsize=13)
    ax.set_xlim([1, len(buses)])
    ax.set_ylim([ymin, ymax])
    ax.set_xticks(buses)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.6)
    plt.tight_layout()
    path = os.path.join(results_dir, "cap_voltage_profile_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Voltage comparison plot       : {path}")


def plot_loss_comparison(loss_base:    pd.DataFrame,
                         loss_cap:     pd.DataFrame,
                         summary_base: dict,
                         summary_cap:  dict,
                         results_dir:  str):
    """Branch P and Q losses before and after capacitor placement."""
    branches = loss_base["Branch"].to_numpy()
    x        = np.arange(len(branches))
    w        = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Real power losses
    ax = axes[0]
    ax.bar(x - w/2, loss_base["P_loss(kW)"], w, color='firebrick',
           alpha=0.85, edgecolor='black', linewidth=0.4,
           label=f'Base  ({summary_base["Total_P_loss_kW"]:.1f} kW)')
    ax.bar(x + w/2, loss_cap["P_loss(kW)"],  w, color='teal',
           alpha=0.85, edgecolor='black', linewidth=0.4,
           label=f'Cap   ({summary_cap["Total_P_loss_kW"]:.1f} kW)')
    ax.set_xlabel("Branch Number", fontsize=11)
    ax.set_ylabel("P Loss (kW)", fontsize=11)
    ax.set_title("Real Power Losses — Base vs Capacitors",
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(branches[::2], fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.5)

    # Reactive power losses
    ax = axes[1]
    ax.bar(x - w/2, loss_base["Q_loss(kVAR)"], w, color='firebrick',
           alpha=0.85, edgecolor='black', linewidth=0.4,
           label=f'Base  ({summary_base["Total_Q_loss_kVAR"]:.1f} kVAR)')
    ax.bar(x + w/2, loss_cap["Q_loss(kVAR)"],  w, color='teal',
           alpha=0.85, edgecolor='black', linewidth=0.4,
           label=f'Cap   ({summary_cap["Total_Q_loss_kVAR"]:.1f} kVAR)')
    ax.set_xlabel("Branch Number", fontsize=11)
    ax.set_ylabel("Q Loss (kVAR)", fontsize=11)
    ax.set_title("Reactive Power Losses — Base vs Capacitors",
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(branches[::2], fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.5)

    plt.suptitle(
        "Branch Power Losses — Base Case vs Optimal Capacitor Placement",
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(results_dir, "cap_branch_losses_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Loss comparison plot          : {path}")


def plot_cap_dashboard(v_base:      np.ndarray,
                       v_cap:       np.ndarray,
                       loss_base:   pd.DataFrame,
                       loss_cap:    pd.DataFrame,
                       summary_base: dict,
                       summary_cap:  dict,
                       history:     list,
                       cap_buses:   list,
                       cap_Q_kvar:  list,
                       results_dir: str):
    """Combined 4-panel dashboard for capacitor optimization."""
    buses    = np.arange(1, len(v_base) + 1)
    branches = loss_base["Branch"].to_numpy()
    x        = np.arange(len(branches))
    w        = 0.35

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

    # ── Panel 1: PSO convergence ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history, '-', color='teal', linewidth=2)
    ax1.set_xlabel("Iteration", fontsize=10)
    ax1.set_ylabel("Fitness", fontsize=10)
    ax1.set_title("PSO Convergence", fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.5)

    # ── Panel 2: Voltage profile ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(buses, np.abs(v_base), 'o-', color='firebrick',
             linewidth=2, markersize=4, label='Base Case')
    ax2.plot(buses, np.abs(v_cap),  's-', color='teal',
             linewidth=2, markersize=4, label='With Capacitors')
    ax2.axhline(0.95, color='gray', linestyle=':', linewidth=1.2,
                label='0.95 p.u.')
    clrs = ['steelblue', 'darkorange', 'purple']
    for k, bus in enumerate(cap_buses):
        ax2.axvline(bus, color=clrs[k], linestyle='--',
                    linewidth=1.2, alpha=0.8,
                    label=f'Cap-{k+1}@{bus}')
    ax2.set_xlabel("Bus Number", fontsize=10)
    ax2.set_ylabel("Voltage (p.u.)", fontsize=10)
    ax2.set_title("Voltage Profile", fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower left', ncol=2)
    ax2.grid(True, alpha=0.5)

    # ── Panel 3: Real power losses bar chart ──────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.bar(x - w/2, loss_base["P_loss(kW)"], w, color='firebrick',
            alpha=0.85, edgecolor='black', linewidth=0.3,
            label=f'Base ({summary_base["Total_P_loss_kW"]:.1f} kW)')
    ax3.bar(x + w/2, loss_cap["P_loss(kW)"],  w, color='teal',
            alpha=0.85, edgecolor='black', linewidth=0.3,
            label=f'Cap  ({summary_cap["Total_P_loss_kW"]:.1f} kW)')
    ax3.set_xlabel("Branch", fontsize=10)
    ax3.set_ylabel("P Loss (kW)", fontsize=10)
    ax3.set_title("Real Power Losses", fontsize=11, fontweight='bold')
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels(branches[::2], fontsize=8)
    ax3.legend(fontsize=9)
    ax3.grid(True, axis='y', alpha=0.5)

    # ── Panel 4: Results summary text ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')

    Vmag_b  = np.abs(v_base)
    Vmag_c  = np.abs(v_cap)
    pl_red  = (1 - summary_cap["Total_P_loss_kW"]   /
                   summary_base["Total_P_loss_kW"])   * 100
    ql_red  = (1 - summary_cap["Total_Q_loss_kVAR"]  /
                   summary_base["Total_Q_loss_kVAR"]) * 100
    vm_imp  = (Vmag_c.min() - Vmag_b.min()) * 100

    txt  = "─── RESULTS SUMMARY ───\n\n"
    txt += f"  Capacitor Banks : {N_CAP}\n\n"
    for k in range(N_CAP):
        txt += (f"  Cap-{k+1} @ Bus {cap_buses[k]:>2d}\n"
                f"    Q = {cap_Q_kvar[k]:.0f} kVAR\n\n")
    txt += (f"  Total Q installed:\n"
            f"    {sum(cap_Q_kvar):.0f} kVAR\n\n"
            f"  P Loss:\n"
            f"    {summary_base['Total_P_loss_kW']:.2f} → "
            f"{summary_cap['Total_P_loss_kW']:.2f} kW\n"
            f"    ↓ {pl_red:.2f}% reduction\n\n"
            f"  Q Loss:\n"
            f"    {summary_base['Total_Q_loss_kVAR']:.2f} → "
            f"{summary_cap['Total_Q_loss_kVAR']:.2f} kVAR\n"
            f"    ↓ {ql_red:.2f}% reduction\n\n"
            f"  Vmin:\n"
            f"    {Vmag_b.min():.4f} → {Vmag_c.min():.4f} p.u.\n"
            f"    ↑ {vm_imp:.3f}% improvement\n\n"
            f"  Buses < 0.95 p.u.:\n"
            f"    {(Vmag_b < 0.95).sum()} → {(Vmag_c < 0.95).sum()}")

    ax4.text(0.05, 0.97, txt,
             transform=ax4.transAxes,
             fontsize=9, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.9))

    fig.suptitle(
        f"33-Bus System — Optimal Capacitor Placement Dashboard (PSO)\n"
        f"{N_CAP}× Shunt Capacitor Banks  |  "
        f"Obj = {W1}×P_loss + {W2}×V_dev",
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(results_dir, "cap_optimization_dashboard.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Capacitor dashboard           : {path}")


# ======================================================================= #
#  SAVE CSVs
# ======================================================================= #

def save_cap_csv(v_base:      np.ndarray,
                 v_cap:       np.ndarray,
                 loss_base:   pd.DataFrame,
                 loss_cap:    pd.DataFrame,
                 cap_buses:   list,
                 cap_Q_kvar:  list,
                 results_dir: str):
    """Save before/after voltage, loss comparison, and cap summary to CSV."""

    nb      = len(v_base)
    cap_flag = np.zeros(nb, dtype=int)
    for b in cap_buses:
        cap_flag[b - 1] = 1

    # Voltage comparison
    pd.DataFrame({
        "Bus"                  : np.arange(1, nb + 1),
        "Cap_Installed"        : cap_flag,
        "Base_Vmag(pu)"        : np.abs(v_base),
        "Cap_Vmag(pu)"         : np.abs(v_cap),
        "Vmag_Improvement(pu)" : np.abs(v_cap) - np.abs(v_base),
        "Base_Vang(deg)"       : np.angle(v_base, deg=True),
        "Cap_Vang(deg)"        : np.angle(v_cap,  deg=True),
    }).to_csv(
        os.path.join(results_dir, "cap_voltage_comparison.csv"),
        index=False)
    print(f"  Voltage CSV                   : "
          f"{results_dir}/cap_voltage_comparison.csv")

    # Loss comparison
    pd.DataFrame({
        "Branch"                : loss_base["Branch"],
        "From_Bus"              : loss_base["From_Bus"],
        "To_Bus"                : loss_base["To_Bus"],
        "Base_P_loss(kW)"       : loss_base["P_loss(kW)"],
        "Cap_P_loss(kW)"        : loss_cap["P_loss(kW)"],
        "P_Reduction(kW)"       : loss_base["P_loss(kW)"] -
                                   loss_cap["P_loss(kW)"],
        "Base_Q_loss(kVAR)"     : loss_base["Q_loss(kVAR)"],
        "Cap_Q_loss(kVAR)"      : loss_cap["Q_loss(kVAR)"],
        "Q_Reduction(kVAR)"     : loss_base["Q_loss(kVAR)"] -
                                   loss_cap["Q_loss(kVAR)"],
    }).to_csv(
        os.path.join(results_dir, "cap_loss_comparison.csv"),
        index=False)
    print(f"  Loss CSV                      : "
          f"{results_dir}/cap_loss_comparison.csv")

    # Capacitor placement summary
    pd.DataFrame({
        "Capacitor"   : [f"Cap-{k+1}" for k in range(N_CAP)],
        "Bus"         : cap_buses,
        "Q_kVAR"      : cap_Q_kvar,
    }).to_csv(
        os.path.join(results_dir, "cap_placement_summary.csv"),
        index=False)
    print(f"  Cap placement CSV             : "
          f"{results_dir}/cap_placement_summary.csv")


# ======================================================================= #
#  MAIN ENTRY POINT  (called from main.py)
# ======================================================================= #

def run_cap_optimization(results_dir:  str,
                          branch_df:    pd.DataFrame,
                          z_branch_pu:  np.ndarray,
                          S_bus_pu:     np.ndarray,
                          v_base:       np.ndarray,
                          loss_base:    pd.DataFrame,
                          summary_base: dict):
    """
    Entry point called from main.py.
    All pre-loaded data is passed in — no CSV re-reading.

    Parameters
    ----------
    results_dir  : str            output directory
    branch_df    : pd.DataFrame   branch data
    z_branch_pu  : np.ndarray     complex branch impedances (p.u.)
    S_bus_pu     : np.ndarray     original complex bus loads (p.u.)
    v_base       : np.ndarray     base case bus voltages
    loss_base    : pd.DataFrame   base case branch losses
    summary_base : dict           base case loss summary

    Returns
    -------
    v_cap       : np.ndarray   bus voltages after capacitor placement
    summary_cap : dict         loss summary after capacitor placement
    """
    P_loss_base = summary_base["Total_P_loss_kW"]
    V_dev_base  = float(np.sum((1.0 - np.abs(v_base)) ** 2))

    print(f"  Base P loss       : {P_loss_base:.3f} kW")
    print(f"  Base Vmin         : {np.abs(v_base).min():.5f} p.u."
          f"  @ Bus {np.abs(v_base).argmin() + 1}")
    print(f"  Buses < 0.95 p.u. : {(np.abs(v_base) < 0.95).sum()}")

    # ── Run PSO ────────────────────────────────────────────────────────
    best_pos, best_fit, history = run_pso(
        branch_df, z_branch_pu, S_bus_pu,
        P_loss_base, V_dev_base
    )

    # ── Decode best solution ───────────────────────────────────────────
    cap_buses, cap_Q_kvar = decode(best_pos)

    # ── Load flow with optimal capacitors ─────────────────────────────
    S_cap = apply_capacitors(S_bus_pu, cap_buses, cap_Q_kvar)
    v_cap = run_dlf_direct(branch_df, z_branch_pu, S_cap)
    loss_cap, summary_cap = calculate_losses(
        v_cap, branch_df, z_branch_pu, S_cap, sbase_mva=SBASE_MVA
    )

    # ── Print summary ──────────────────────────────────────────────────
    print_cap_summary(cap_buses, cap_Q_kvar,
                      summary_base, summary_cap,
                      v_base, v_cap)

    # ── Save all outputs ───────────────────────────────────────────────
    save_cap_csv(v_base, v_cap, loss_base, loss_cap,
                 cap_buses, cap_Q_kvar, results_dir)
    plot_convergence(history, results_dir)
    plot_voltage_comparison(v_base, v_cap, cap_buses, results_dir)
    plot_loss_comparison(loss_base, loss_cap,
                         summary_base, summary_cap, results_dir)
    plot_cap_dashboard(v_base, v_cap, loss_base, loss_cap,
                       summary_base, summary_cap,
                       history, cap_buses, cap_Q_kvar,
                       results_dir)

    return v_cap, summary_cap


# ======================================================================= #
#  STANDALONE RUN  (python capacitor_placement.py)
# ======================================================================= #

if __name__ == "__main__":
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

    run_cap_optimization(results_dir,
                          branch_df, z_branch_pu, S_bus_pu,
                          v_base, loss_base, summary_base)