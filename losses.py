# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:00:18 2026

@author: Dipanshu Tripathi
"""

"""
losses.py
Calculate and visualize real and reactive power losses for the 33-bus
radial distribution system using results from DLF and BFS load flow solvers.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from load_data import load_branch_bus_csv
from load_flow import run_load_flow
from bfs_flow  import run_bfs_load_flow


# ======================================================================= #
#  LOSS CALCULATION
# ======================================================================= #

def calculate_losses(voltages: np.ndarray, branch_df: pd.DataFrame,
                     z_branch_pu: np.ndarray, S_bus_pu: np.ndarray,
                     sbase_mva: float = 100.0):
    """
    Calculate branch-wise and total real/reactive power losses.

    For each branch k connecting bus i (from) to bus j (to):
        I_branch_k = (V_i - V_j) / Z_k
        S_loss_k   = |I_branch_k|^2 * Z_k
        P_loss_k   = Re(S_loss_k)
        Q_loss_k   = Im(S_loss_k)

    Parameters
    ----------
    voltages    : np.ndarray, complex, shape (nb,)   bus voltages in p.u.
    branch_df   : pd.DataFrame                        branch data
    z_branch_pu : np.ndarray, complex                 branch impedances p.u.
    S_bus_pu    : np.ndarray, complex                 bus power injections p.u.
    sbase_mva   : float                               system base MVA

    Returns
    -------
    loss_df : pd.DataFrame
        Branch-wise losses with columns:
        Branch, From_Bus, To_Bus, R(pu), X(pu),
        |I_branch|(pu), P_loss(kW), Q_loss(kVAR),
        P_loss(pu), Q_loss(pu)

    summary : dict
        Total P loss (kW/pu), Total Q loss (kVAR/pu),
        % P loss, % Q loss, worst branch by P and Q
    """
    from_buses = branch_df['from_bus'].to_numpy(dtype=int)
    to_buses   = branch_df['to_bus'].to_numpy(dtype=int)
    bnn        = len(branch_df)

    I_branch_mag = np.zeros(bnn)
    P_loss_pu    = np.zeros(bnn)
    Q_loss_pu    = np.zeros(bnn)

    for k in range(bnn):
        i = from_buses[k] - 1     # 0-based from_bus index
        j = to_buses[k]   - 1     # 0-based to_bus index

        # Branch current from voltage difference
        I_k          = (voltages[i] - voltages[j]) / z_branch_pu[k]
        I_branch_mag[k] = np.abs(I_k)

        # Complex power loss = |I|^2 * Z
        S_loss_k     = (np.abs(I_k) ** 2) * z_branch_pu[k]
        P_loss_pu[k] = S_loss_k.real
        Q_loss_pu[k] = S_loss_k.imag

    # Convert to physical units
    sbase_kva    = sbase_mva * 1000.0
    P_loss_kw    = P_loss_pu    * sbase_kva
    Q_loss_kvar  = Q_loss_pu    * sbase_kva

    # Total load (for % loss calculation)
    P_load_total = S_bus_pu.real.sum() * sbase_kva   # kW
    Q_load_total = S_bus_pu.imag.sum() * sbase_kva   # kVAR

    pct_P = (P_loss_kw.sum()   / P_load_total  * 100) if P_load_total  != 0 else 0
    pct_Q = (Q_loss_kvar.sum() / Q_load_total  * 100) if Q_load_total  != 0 else 0

    loss_df = pd.DataFrame({
        "Branch"         : branch_df['branch_num'].to_numpy(),
        "From_Bus"       : from_buses,
        "To_Bus"         : to_buses,
        "R_pu"           : z_branch_pu.real,
        "X_pu"           : z_branch_pu.imag,
        "|I_branch|(pu)" : I_branch_mag,
        "P_loss(kW)"     : P_loss_kw,
        "Q_loss(kVAR)"   : Q_loss_kvar,
        "P_loss(pu)"     : P_loss_pu,
        "Q_loss(pu)"     : Q_loss_pu,
    })

    summary = {
        "Total_P_loss_kW"    : P_loss_kw.sum(),
        "Total_Q_loss_kVAR"  : Q_loss_kvar.sum(),
        "Total_P_loss_pu"    : P_loss_pu.sum(),
        "Total_Q_loss_pu"    : Q_loss_pu.sum(),
        "Pct_P_loss"         : pct_P,
        "Pct_Q_loss"         : pct_Q,
        "Worst_Branch_P"     : int(loss_df.loc[P_loss_kw.argmax(),   "Branch"]),
        "Worst_Branch_Q"     : int(loss_df.loc[Q_loss_kvar.argmax(), "Branch"]),
        "Worst_P_loss_kW"    : P_loss_kw.max(),
        "Worst_Q_loss_kVAR"  : Q_loss_kvar.max(),
    }

    return loss_df, summary


# ======================================================================= #
#  TERMINAL SUMMARY
# ======================================================================= #

def print_loss_summary(summary_dlf: dict, summary_bfs: dict):
    """Print a formatted terminal comparison of DLF vs BFS losses."""

    print("\n" + "="*65)
    print("   POWER LOSS SUMMARY — DLF vs BFS")
    print("="*65)
    print(f"  {'Metric':<38} {'DLF':>12} {'BFS':>12}")
    print("-"*65)

    metrics = [
        ("Total Real Power Loss (kW)",    "Total_P_loss_kW",   ".3f"),
        ("Total Reactive Power Loss (kVAR)", "Total_Q_loss_kVAR", ".3f"),
        ("Total Real Power Loss (p.u.)",  "Total_P_loss_pu",   ".6f"),
        ("Total Q Power Loss (p.u.)",     "Total_Q_loss_pu",   ".6f"),
        ("% Real Power Loss",             "Pct_P_loss",        ".3f"),
        ("% Reactive Power Loss",         "Pct_Q_loss",        ".3f"),
        ("Worst Branch (P loss)",         "Worst_Branch_P",    "d"),
        ("Worst Branch (Q loss)",         "Worst_Branch_Q",    "d"),
        ("Worst Branch P loss (kW)",      "Worst_P_loss_kW",   ".3f"),
        ("Worst Branch Q loss (kVAR)",    "Worst_Q_loss_kVAR", ".3f"),
    ]

    for label, key, fmt in metrics:
        d_val = summary_dlf[key]
        b_val = summary_bfs[key]
        if fmt == "d":
            print(f"  {label:<38} {d_val:>12d} {b_val:>12d}")
        else:
            print(f"  {label:<38} {d_val:>12{fmt}} {b_val:>12{fmt}}")

    print("="*65)


# ======================================================================= #
#  SAVE LOSS CSV
# ======================================================================= #

def save_loss_csv(loss_dlf: pd.DataFrame, loss_bfs: pd.DataFrame,
                  results_dir: str):
    """Save branch-wise losses from both solvers to CSV."""

    # Individual CSVs
    path_dlf = os.path.join(results_dir, "branch_losses_dlf.csv")
    path_bfs = os.path.join(results_dir, "branch_losses_bfs.csv")
    loss_dlf.to_csv(path_dlf, index=False)
    loss_bfs.to_csv(path_bfs, index=False)
    print(f"\nDLF branch losses saved : {path_dlf}")
    print(f"BFS branch losses saved : {path_bfs}")

    # Combined comparison CSV
    combined = pd.DataFrame({
        "Branch"              : loss_dlf["Branch"],
        "From_Bus"            : loss_dlf["From_Bus"],
        "To_Bus"              : loss_dlf["To_Bus"],
        "DLF_P_loss(kW)"      : loss_dlf["P_loss(kW)"],
        "BFS_P_loss(kW)"      : loss_bfs["P_loss(kW)"],
        "DLF_Q_loss(kVAR)"    : loss_dlf["Q_loss(kVAR)"],
        "BFS_Q_loss(kVAR)"    : loss_bfs["Q_loss(kVAR)"],
        "DLF_|I|(pu)"         : loss_dlf["|I_branch|(pu)"],
        "BFS_|I|(pu)"         : loss_bfs["|I_branch|(pu)"],
    })
    path_combined = os.path.join(results_dir, "branch_losses_comparison.csv")
    combined.to_csv(path_combined, index=False)
    print(f"Combined loss comparison saved : {path_combined}")


# ======================================================================= #
#  PLOT 1 — BRANCH REAL POWER LOSSES
# ======================================================================= #

def plot_real_power_losses(loss_dlf: pd.DataFrame, loss_bfs: pd.DataFrame,
                           results_dir: str):
    """Bar chart of real power loss per branch — DLF vs BFS."""

    branches = loss_dlf["Branch"].to_numpy()
    x        = np.arange(len(branches))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, loss_dlf["P_loss(kW)"], width,
                   label='DLF (BIBC/BCBV)', color='steelblue',
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, loss_bfs["P_loss(kW)"], width,
                   label='BFS', color='darkorange',
                   edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Branch Number", fontsize=12)
    ax.set_ylabel("Real Power Loss (kW)", fontsize=12)
    ax.set_title("Branch-wise Real Power Losses — DLF vs BFS (33-Bus System)",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(branches, fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.6)

    # Annotate worst branch
    worst_idx = loss_dlf["P_loss(kW)"].argmax()
    ax.annotate(f'Worst\nBranch {branches[worst_idx]}',
                xy=(worst_idx - width/2, loss_dlf["P_loss(kW)"].iloc[worst_idx]),
                xytext=(worst_idx + 1.5, loss_dlf["P_loss(kW)"].max() * 0.85),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    plt.tight_layout()
    path = os.path.join(results_dir, "branch_real_power_losses.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Real power loss plot saved       : {path}")


# ======================================================================= #
#  PLOT 2 — BRANCH REACTIVE POWER LOSSES
# ======================================================================= #

def plot_reactive_power_losses(loss_dlf: pd.DataFrame, loss_bfs: pd.DataFrame,
                               results_dir: str):
    """Bar chart of reactive power loss per branch — DLF vs BFS."""

    branches = loss_dlf["Branch"].to_numpy()
    x        = np.arange(len(branches))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, loss_dlf["Q_loss(kVAR)"], width,
           label='DLF (BIBC/BCBV)', color='steelblue',
           edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, loss_bfs["Q_loss(kVAR)"], width,
           label='BFS', color='darkorange',
           edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Branch Number", fontsize=12)
    ax.set_ylabel("Reactive Power Loss (kVAR)", fontsize=12)
    ax.set_title("Branch-wise Reactive Power Losses — DLF vs BFS (33-Bus System)",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(branches, fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.6)

    # Annotate worst branch
    worst_idx = loss_dlf["Q_loss(kVAR)"].argmax()
    ax.annotate(f'Worst\nBranch {branches[worst_idx]}',
                xy=(worst_idx - width/2, loss_dlf["Q_loss(kVAR)"].iloc[worst_idx]),
                xytext=(worst_idx + 1.5, loss_dlf["Q_loss(kVAR)"].max() * 0.85),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    plt.tight_layout()
    path = os.path.join(results_dir, "branch_reactive_power_losses.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Reactive power loss plot saved   : {path}")


# ======================================================================= #
#  PLOT 3 — BRANCH CURRENT MAGNITUDE
# ======================================================================= #

def plot_branch_currents(loss_dlf: pd.DataFrame, loss_bfs: pd.DataFrame,
                         results_dir: str):
    """Line plot of branch current magnitudes — DLF vs BFS."""

    branches = loss_dlf["Branch"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(branches, loss_dlf["|I_branch|(pu)"], 'o-',
            color='steelblue', linewidth=2, markersize=5,
            label='DLF (BIBC/BCBV)')
    ax.plot(branches, loss_bfs["|I_branch|(pu)"], 's--',
            color='darkorange', linewidth=2, markersize=5,
            label='BFS')
    ax.set_xlabel("Branch Number", fontsize=12)
    ax.set_ylabel("Branch Current Magnitude (p.u.)", fontsize=12)
    ax.set_title("Branch Current Magnitudes — DLF vs BFS (33-Bus System)",
                 fontsize=13)
    ax.set_xticks(branches)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.6)
    plt.tight_layout()
    path = os.path.join(results_dir, "branch_currents.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Branch current plot saved        : {path}")


# ======================================================================= #
#  PLOT 4 — CUMULATIVE LOSS ALONG FEEDER
# ======================================================================= #

def plot_cumulative_losses(loss_dlf: pd.DataFrame, loss_bfs: pd.DataFrame,
                           results_dir: str):
    """
    Cumulative real and reactive power loss along the main feeder.
    Shows how losses build up from the substation toward the end of the feeder.
    """
    branches      = loss_dlf["Branch"].to_numpy()
    cum_P_dlf     = np.cumsum(loss_dlf["P_loss(kW)"].to_numpy())
    cum_P_bfs     = np.cumsum(loss_bfs["P_loss(kW)"].to_numpy())
    cum_Q_dlf     = np.cumsum(loss_dlf["Q_loss(kVAR)"].to_numpy())
    cum_Q_bfs     = np.cumsum(loss_bfs["Q_loss(kVAR)"].to_numpy())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Cumulative P loss
    ax1.plot(branches, cum_P_dlf, 'o-', color='steelblue',
             linewidth=2, markersize=4, label='DLF (BIBC/BCBV)')
    ax1.plot(branches, cum_P_bfs, 's--', color='darkorange',
             linewidth=2, markersize=4, label='BFS')
    ax1.set_ylabel("Cumulative P Loss (kW)", fontsize=12)
    ax1.set_title("Cumulative Power Losses Along Feeder — DLF vs BFS",
                  fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.6)

    # Cumulative Q loss
    ax2.plot(branches, cum_Q_dlf, 'o-', color='steelblue',
             linewidth=2, markersize=4, label='DLF (BIBC/BCBV)')
    ax2.plot(branches, cum_Q_bfs, 's--', color='darkorange',
             linewidth=2, markersize=4, label='BFS')
    ax2.set_xlabel("Branch Number", fontsize=12)
    ax2.set_ylabel("Cumulative Q Loss (kVAR)", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.6)
    ax2.set_xticks(branches)

    plt.tight_layout()
    path = os.path.join(results_dir, "cumulative_losses.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Cumulative loss plot saved       : {path}")


# ======================================================================= #
#  PLOT 5 — LOSS SUMMARY PIE CHART
# ======================================================================= #

def plot_loss_pie(loss_dlf: pd.DataFrame, summary_dlf: dict,
                  results_dir: str):
    """
    Pie chart showing top branches contributing most to total real power loss.
    Based on DLF results (BFS gives same values).
    """
    P_losses = loss_dlf["P_loss(kW)"].to_numpy()
    branches = loss_dlf["Branch"].to_numpy()

    # Top 8 branches by loss, rest grouped as "Others"
    sorted_idx  = np.argsort(P_losses)[::-1]
    top_n       = 8
    top_idx     = sorted_idx[:top_n]
    other_loss  = P_losses[sorted_idx[top_n:]].sum()

    labels = [f"Branch {branches[i]}" for i in top_idx] + ["Others"]
    sizes  = list(P_losses[top_idx]) + [other_loss]

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.tab10.colors[:len(labels)]
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title(
        f"Real Power Loss Distribution by Branch (DLF)\n"
        f"Total Loss = {summary_dlf['Total_P_loss_kW']:.2f} kW  "
        f"({summary_dlf['Pct_P_loss']:.2f}% of total load)",
        fontsize=12
    )
    plt.tight_layout()
    path = os.path.join(results_dir, "loss_distribution_pie.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Loss distribution pie saved      : {path}")


# ======================================================================= #
#  PLOT 6 — COMBINED DASHBOARD
# ======================================================================= #

def plot_loss_dashboard(loss_dlf: pd.DataFrame, loss_bfs: pd.DataFrame,
                        v_dlf: np.ndarray, v_bfs: np.ndarray,
                        summary_dlf: dict, results_dir: str):
    """
    Single-figure dashboard combining voltage profile + branch losses.
    Gives the complete picture in one image.
    """
    branches    = loss_dlf["Branch"].to_numpy()
    bus_numbers = np.arange(1, len(v_dlf) + 1)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Top-left: Voltage magnitude profile ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(bus_numbers, np.abs(v_dlf), 'o-', color='steelblue',
             linewidth=1.8, markersize=4, label='DLF')
    ax1.plot(bus_numbers, np.abs(v_bfs), 's--', color='darkorange',
             linewidth=1.8, markersize=4, label='BFS')
    ax1.axhline(y=0.95, color='red', linestyle=':', linewidth=1.2,
                label='0.95 p.u. limit')
    ax1.set_xlabel("Bus Number", fontsize=10)
    ax1.set_ylabel("Voltage Magnitude (p.u.)", fontsize=10)
    ax1.set_title("Voltage Profile", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.5)

    # ── Top-right: Branch current magnitudes ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(branches, loss_dlf["|I_branch|(pu)"], 'o-',
             color='steelblue', linewidth=1.8, markersize=4, label='DLF')
    ax2.plot(branches, loss_bfs["|I_branch|(pu)"], 's--',
             color='darkorange', linewidth=1.8, markersize=4, label='BFS')
    ax2.set_xlabel("Branch Number", fontsize=10)
    ax2.set_ylabel("|I_branch| (p.u.)", fontsize=10)
    ax2.set_title("Branch Current Magnitudes", fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.5)

    # ── Bottom-left: Real power losses ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    x     = np.arange(len(branches))
    width = 0.35
    ax3.bar(x - width/2, loss_dlf["P_loss(kW)"], width,
            label='DLF', color='steelblue', edgecolor='black', linewidth=0.4)
    ax3.bar(x + width/2, loss_bfs["P_loss(kW)"], width,
            label='BFS', color='darkorange', edgecolor='black', linewidth=0.4)
    ax3.set_xlabel("Branch Number", fontsize=10)
    ax3.set_ylabel("P Loss (kW)", fontsize=10)
    ax3.set_title(
        f"Real Power Losses  |  Total (DLF) = "
        f"{summary_dlf['Total_P_loss_kW']:.2f} kW",
        fontsize=11, fontweight='bold'
    )
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels(branches[::2], fontsize=8)
    ax3.legend(fontsize=9)
    ax3.grid(True, axis='y', alpha=0.5)

    # ── Bottom-right: Reactive power losses ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(x - width/2, loss_dlf["Q_loss(kVAR)"], width,
            label='DLF', color='steelblue', edgecolor='black', linewidth=0.4)
    ax4.bar(x + width/2, loss_bfs["Q_loss(kVAR)"], width,
            label='BFS', color='darkorange', edgecolor='black', linewidth=0.4)
    ax4.set_xlabel("Branch Number", fontsize=10)
    ax4.set_ylabel("Q Loss (kVAR)", fontsize=10)
    ax4.set_title(
        f"Reactive Power Losses  |  Total (DLF) = "
        f"{summary_dlf['Total_Q_loss_kVAR']:.2f} kVAR",
        fontsize=11, fontweight='bold'
    )
    ax4.set_xticks(x[::2])
    ax4.set_xticklabels(branches[::2], fontsize=8)
    ax4.legend(fontsize=9)
    ax4.grid(True, axis='y', alpha=0.5)

    fig.suptitle(
        "33-Bus Radial Distribution System — Load Flow & Loss Dashboard\n"
        "DLF (BIBC/BCBV) vs BFS",
        fontsize=14, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    path = os.path.join(results_dir, "loss_dashboard.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss dashboard saved             : {path}")


# ======================================================================= #
#  MAIN
# ======================================================================= #

def main():
    print("\n--- 33-BUS SYSTEM — POWER LOSS ANALYSIS ---")
    print("--- DLF (BIBC/BCBV)  vs  BFS             ---\n")

    base_path   = os.path.dirname(os.path.abspath(__file__))
    data_path   = os.path.join(base_path, "data")
    results_dir = os.path.join(base_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    branch_csv = os.path.join(data_path, "branch_data.csv")
    bus_csv    = os.path.join(data_path, "bus_data.csv")

    # ── Load raw data (for impedances and loads) ───────────────────────
    branch_df, bus_df, z_branch_pu, S_bus_pu, z_base = load_branch_bus_csv(
        branch_csv, bus_csv
    )

    # ── Run both solvers ───────────────────────────────────────────────
    print("Running DLF solver...")
    v_dlf = run_load_flow(branch_csv, bus_csv)
    print("DLF complete.\n")

    print("Running BFS solver...")
    v_bfs = run_bfs_load_flow(branch_csv, bus_csv)
    print("BFS complete.\n")

    # ── Calculate losses ───────────────────────────────────────────────
    print("Calculating losses...")
    loss_dlf, summary_dlf = calculate_losses(
        v_dlf, branch_df, z_branch_pu, S_bus_pu
    )
    loss_bfs, summary_bfs = calculate_losses(
        v_bfs, branch_df, z_branch_pu, S_bus_pu
    )

    # ── Print summary ──────────────────────────────────────────────────
    print_loss_summary(summary_dlf, summary_bfs)

    # ── Save CSVs ──────────────────────────────────────────────────────
    save_loss_csv(loss_dlf, loss_bfs, results_dir)

    # ── Generate all plots ─────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_real_power_losses(loss_dlf, loss_bfs, results_dir)
    plot_reactive_power_losses(loss_dlf, loss_bfs, results_dir)
    plot_branch_currents(loss_dlf, loss_bfs, results_dir)
    plot_cumulative_losses(loss_dlf, loss_bfs, results_dir)
    plot_loss_pie(loss_dlf, summary_dlf, results_dir)
    plot_loss_dashboard(loss_dlf, loss_bfs, v_dlf, v_bfs,
                        summary_dlf, results_dir)

    print("\n" + "="*55)
    print("  All loss results and plots saved in 'results/' folder.")
    print("="*55)


if __name__ == "__main__":
    main()