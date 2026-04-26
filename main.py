# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:59:11 2026

@author: Dipanshu Tripathi
"""

"""
main.py
Complete pipeline — 33-bus radial distribution system:

    STEP 1 — Load Flow              : DLF (BIBC/BCBV) + BFS
    STEP 2 — Voltage Results        : profiles, angles, DLF vs BFS diff
    STEP 3 — Loss Analysis          : branch losses, cumulative, dashboard
    STEP 4 — Optimal DG Placement   : PSO, 3× Type-2 DG (PF=0.85)
    STEP 5 — Optimal Cap Placement  : PSO, 3× shunt capacitor banks
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_data           import load_branch_bus_csv
from load_flow           import run_load_flow
from bfs_flow            import run_bfs_load_flow
from losses              import (calculate_losses, print_loss_summary,
                                  save_loss_csv, plot_real_power_losses,
                                  plot_reactive_power_losses,
                                  plot_branch_currents,
                                  plot_cumulative_losses,
                                  plot_loss_pie, plot_loss_dashboard)
from dg_placement        import run_dg_optimization
from capacitor_placement import run_cap_optimization


# ======================================================================= #
#  VOLTAGE HELPERS
# ======================================================================= #

def save_results_csv(v_dlf, v_bfs, results_dir):
    nb = len(v_dlf)
    pd.DataFrame({
        "Bus"                        : np.arange(1, nb + 1),
        "DLF Voltage Magnitude (pu)" : np.abs(v_dlf),
        "DLF Voltage Angle (deg)"    : np.angle(v_dlf, deg=True),
        "BFS Voltage Magnitude (pu)" : np.abs(v_bfs),
        "BFS Voltage Angle (deg)"    : np.angle(v_bfs, deg=True),
        "Magnitude Difference (pu)"  : np.abs(np.abs(v_dlf) - np.abs(v_bfs)),
        "Angle Difference (deg)"     : np.abs(np.angle(v_dlf, deg=True) -
                                               np.angle(v_bfs, deg=True)),
    }).to_csv(os.path.join(results_dir, "bus_voltages_comparison.csv"),
              index=False)
    print(f"  Voltage CSV                   : "
          f"{results_dir}/bus_voltages_comparison.csv")


def plot_voltage_magnitude(v_dlf, v_bfs, results_dir):
    buses  = np.arange(1, len(v_dlf) + 1)
    Vd, Vb = np.abs(v_dlf), np.abs(v_bfs)
    ymin   = min(Vd.min(), Vb.min()) - 0.01
    ymax   = max(Vd.max(), Vb.max()) + 0.01

    # Standard
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(buses, Vd, 'o-',  color='steelblue',  lw=2, ms=5,
            label='DLF (BIBC/BCBV)')
    ax.plot(buses, Vb, 's--', color='darkorange',  lw=2, ms=5,
            label='BFS')
    ax.axhline(0.95, color='red', ls=':', lw=1.2, label='0.95 p.u. limit')
    ax.set_xlabel("Bus Number", fontsize=12)
    ax.set_ylabel("Voltage Magnitude (p.u.)", fontsize=12)
    ax.set_title("Voltage Magnitude Profile — DLF vs BFS (33-Bus System)",
                 fontsize=13)
    ax.set_xlim([1, len(buses)])
    ax.set_ylim([ymin, ymax])
    ax.set_xticks(buses)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,
                "voltage_magnitude_comparison.png"), dpi=300)
    plt.close()

    # Magnified
    ym = min(Vd.min(), Vb.min()) - 0.05
    yM = max(Vd.max(), Vb.max()) + 0.01
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(buses, Vd, 'o-',  color='steelblue',  lw=2, ms=5,
            label='DLF (BIBC/BCBV)')
    ax.plot(buses, Vb, 's--', color='darkorange',  lw=2, ms=5,
            label='BFS')
    ax.axhline(0.95, color='red', ls=':', lw=1.2, label='0.95 p.u. limit')
    ax.set_xlabel("Bus Number", fontsize=12)
    ax.set_ylabel("Voltage Magnitude (p.u.)", fontsize=12)
    ax.set_title("Voltage Magnitude Profile (Magnified) — DLF vs BFS",
                 fontsize=13)
    ax.set_xlim([1, len(buses)])
    ax.set_ylim([ym, yM])
    ax.set_xticks(buses)
    ax.set_yticks(np.linspace(ym, yM, 15))
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,
                "voltage_magnitude_comparison_magnified.png"), dpi=300)
    plt.close()
    print(f"  Voltage magnitude plots       : "
          f"{results_dir}/voltage_magnitude_*")


def plot_voltage_angle(v_dlf, v_bfs, results_dir):
    buses = np.arange(1, len(v_dlf) + 1)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(buses, np.angle(v_dlf, deg=True), 'o-',
            color='steelblue', lw=2, ms=5, label='DLF (BIBC/BCBV)')
    ax.plot(buses, np.angle(v_bfs, deg=True), 's--',
            color='darkorange', lw=2, ms=5, label='BFS')
    ax.set_xlabel("Bus Number", fontsize=12)
    ax.set_ylabel("Voltage Angle (deg)", fontsize=12)
    ax.set_title("Voltage Angle Profile — DLF vs BFS (33-Bus System)",
                 fontsize=13)
    ax.set_xlim([1, len(buses)])
    ax.set_xticks(buses)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,
                "voltage_angle_comparison.png"), dpi=300)
    plt.close()
    print(f"  Voltage angle plot            : "
          f"{results_dir}/voltage_angle_comparison.png")


def plot_difference(v_dlf, v_bfs, results_dir):
    buses = np.arange(1, len(v_dlf) + 1)
    diff  = np.abs(np.abs(v_dlf) - np.abs(v_bfs))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(buses, diff, color='mediumseagreen', edgecolor='black', lw=0.5)
    ax.set_xlabel("Bus Number", fontsize=12)
    ax.set_ylabel("|V_DLF| − |V_BFS| (p.u.)", fontsize=12)
    ax.set_title(
        "Voltage Magnitude Difference: DLF vs BFS (should be ≈ 0)",
        fontsize=13)
    ax.set_xticks(buses)
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,
                "voltage_difference_dlf_vs_bfs.png"), dpi=300)
    plt.close()
    print(f"  Voltage difference plot       : "
          f"{results_dir}/voltage_difference_dlf_vs_bfs.png")


def print_voltage_summary(v_dlf, v_bfs):
    Vd, Vb = np.abs(v_dlf), np.abs(v_bfs)
    print("\n" + "="*60)
    print("  VOLTAGE RESULTS SUMMARY — DLF vs BFS")
    print("="*60)
    print(f"  {'Metric':<35} {'DLF':>10} {'BFS':>10}")
    print("-"*60)
    print(f"  {'Min Voltage (p.u.)':<35} "
          f"{Vd.min():>10.5f} {Vb.min():>10.5f}")
    print(f"  {'Bus of Min Voltage':<35} "
          f"{Vd.argmin()+1:>10d} {Vb.argmin()+1:>10d}")
    print(f"  {'Max Voltage (p.u.)':<35} "
          f"{Vd.max():>10.5f} {Vb.max():>10.5f}")
    print(f"  {'Mean Voltage (p.u.)':<35} "
          f"{Vd.mean():>10.5f} {Vb.mean():>10.5f}")
    print(f"  {'Buses below 0.95 p.u.':<35} "
          f"{(Vd < 0.95).sum():>10d} {(Vb < 0.95).sum():>10d}")
    print(f"  {'Max |DLF-BFS| diff':<35} "
          f"{np.max(np.abs(Vd - Vb)):>21.2e}")
    print("="*60)


# ======================================================================= #
#  MAIN
# ======================================================================= #

def main():
    print("\n" + "="*65)
    print("  33-BUS RADIAL DISTRIBUTION SYSTEM — FULL ANALYSIS")
    print("="*65)

    base_path   = os.path.dirname(os.path.abspath(__file__))
    data_path   = os.path.join(base_path, "data")
    results_dir = os.path.join(base_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    branch_csv = os.path.join(data_path, "branch_data.csv")
    bus_csv    = os.path.join(data_path, "bus_data.csv")

    # Load data once — shared by all steps
    branch_df, bus_df, z_branch_pu, S_bus_pu, z_base = \
        load_branch_bus_csv(branch_csv, bus_csv)

    # ================================================================== #
    #  STEP 1 — LOAD FLOW
    # ================================================================== #
    print("\n[STEP 1] Load Flow Solvers")
    print("  Running DLF (BIBC/BCBV)...")
    v_dlf = run_load_flow(branch_csv, bus_csv)
    print("  DLF complete.")
    print("  Running BFS...")
    v_bfs = run_bfs_load_flow(branch_csv, bus_csv)

    # ================================================================== #
    #  STEP 2 — VOLTAGE RESULTS
    # ================================================================== #
    print("\n[STEP 2] Voltage Results")
    print_voltage_summary(v_dlf, v_bfs)
    print("\n  Saving voltage outputs...")
    save_results_csv(v_dlf, v_bfs, results_dir)
    plot_voltage_magnitude(v_dlf, v_bfs, results_dir)
    plot_voltage_angle(v_dlf, v_bfs, results_dir)
    plot_difference(v_dlf, v_bfs, results_dir)

    # ================================================================== #
    #  STEP 3 — LOSS ANALYSIS
    # ================================================================== #
    print("\n[STEP 3] Power Loss Analysis")
    loss_dlf, summary_dlf = calculate_losses(
        v_dlf, branch_df, z_branch_pu, S_bus_pu)
    loss_bfs, summary_bfs = calculate_losses(
        v_bfs, branch_df, z_branch_pu, S_bus_pu)
    print_loss_summary(summary_dlf, summary_bfs)

    print("\n  Saving loss outputs...")
    save_loss_csv(loss_dlf, loss_bfs, results_dir)
    plot_real_power_losses(loss_dlf, loss_bfs, results_dir)
    plot_reactive_power_losses(loss_dlf, loss_bfs, results_dir)
    plot_branch_currents(loss_dlf, loss_bfs, results_dir)
    plot_cumulative_losses(loss_dlf, loss_bfs, results_dir)
    plot_loss_pie(loss_dlf, summary_dlf, results_dir)
    plot_loss_dashboard(loss_dlf, loss_bfs, v_dlf, v_bfs,
                        summary_dlf, results_dir)

    # ================================================================== #
    #  STEP 4 — OPTIMAL DG PLACEMENT (PSO)
    # ================================================================== #
    print("\n[STEP 4] Optimal DG Placement — PSO")
    print("  3× Type-2 DG  |  PF=0.85  |  "
          "Obj = 0.6×P_loss + 0.4×V_dev")

    v_dg, summary_dg = run_dg_optimization(
        branch_csv, bus_csv, results_dir,
        branch_df, z_branch_pu, S_bus_pu,
        v_dlf, loss_dlf, summary_dlf
    )

    # ================================================================== #
    #  STEP 5 — OPTIMAL CAPACITOR PLACEMENT (PSO)
    # ================================================================== #
    print("\n[STEP 5] Optimal Capacitor Placement — PSO")
    print("  3× Shunt Capacitor Banks  |  "
          "Obj = 0.6×P_loss + 0.4×V_dev")

    v_cap, summary_cap = run_cap_optimization(
        results_dir,
        branch_df, z_branch_pu, S_bus_pu,
        v_dlf, loss_dlf, summary_dlf
    )

    # ================================================================== #
    #  DONE
    # ================================================================== #
    print("\n" + "="*65)
    print("  COMPLETE — All outputs saved in 'results/'")
    print("  ─────────────────────────────────────────────")
    print("  STEP 1  Load Flow     :  DLF + BFS")
    print("  STEP 2  Voltage       :  4 plots  +  1 CSV")
    print("  STEP 3  Losses        :  6 plots  +  3 CSVs")
    print("  STEP 4  DG (PSO)      :  4 plots  +  3 CSVs")
    print("  STEP 5  Capacitors    :  4 plots  +  3 CSVs")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()