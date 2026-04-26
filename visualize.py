"""
visualize.py
Interactive visualization of DLF vs BFS load flow results
for the 33-bus radial distribution system.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from load_flow import run_load_flow
from bfs_flow  import run_bfs_load_flow


def plot_voltage_profile(v_dlf: np.ndarray, v_bfs: np.ndarray):
    """Overlay voltage magnitude profiles of DLF and BFS."""
    buses    = np.arange(1, len(v_dlf) + 1)
    mag_dlf  = np.abs(v_dlf)
    mag_bfs  = np.abs(v_bfs)

    plt.figure(figsize=(11, 6))
    plt.plot(buses, mag_dlf, 'o-',  color='steelblue',
             linewidth=2, markersize=5, label='DLF (BIBC/BCBV)')
    plt.plot(buses, mag_bfs, 's--', color='darkorange',
             linewidth=2, markersize=5, label='BFS')
    plt.axhline(y=0.95, color='red', linestyle=':', linewidth=1.2,
                label='0.95 p.u. limit')
    plt.xlabel("Bus Number", fontsize=12)
    plt.ylabel("Voltage Magnitude (p.u.)", fontsize=12)
    plt.title("Voltage Magnitude Profile — DLF vs BFS (33-Bus System)",
              fontsize=13)
    plt.xticks(buses)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_voltage_angle(v_dlf: np.ndarray, v_bfs: np.ndarray):
    """Overlay voltage angle profiles of DLF and BFS."""
    buses    = np.arange(1, len(v_dlf) + 1)
    ang_dlf  = np.angle(v_dlf, deg=True)
    ang_bfs  = np.angle(v_bfs, deg=True)

    plt.figure(figsize=(11, 6))
    plt.plot(buses, ang_dlf, 'o-',  color='steelblue',
             linewidth=2, markersize=5, label='DLF (BIBC/BCBV)')
    plt.plot(buses, ang_bfs, 's--', color='darkorange',
             linewidth=2, markersize=5, label='BFS')
    plt.xlabel("Bus Number", fontsize=12)
    plt.ylabel("Voltage Angle (deg)", fontsize=12)
    plt.title("Voltage Angle Profile — DLF vs BFS (33-Bus System)",
              fontsize=13)
    plt.xticks(buses)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_difference(v_dlf: np.ndarray, v_bfs: np.ndarray):
    """Bar chart of per-bus voltage magnitude difference between DLF and BFS."""
    buses = np.arange(1, len(v_dlf) + 1)
    diff  = np.abs(np.abs(v_dlf) - np.abs(v_bfs))

    plt.figure(figsize=(11, 5))
    plt.bar(buses, diff, color='mediumseagreen', edgecolor='black',
            linewidth=0.5)
    plt.xlabel("Bus Number", fontsize=12)
    plt.ylabel("|V_DLF| − |V_BFS|  (p.u.)", fontsize=12)
    plt.title("Voltage Magnitude Difference: DLF vs BFS (should be ≈ 0)",
              fontsize=13)
    plt.xticks(buses)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def main():
    print("\n--- Visualization: DLF vs BFS — 33-Bus System ---\n")

    base_path  = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(base_path, "data")
    branch_csv = os.path.join(data_path, "branch_data.csv")
    bus_csv    = os.path.join(data_path, "bus_data.csv")

    print("Running DLF solver...")
    v_dlf = run_load_flow(branch_csv, bus_csv)
    print("DLF done.\n")

    print("Running BFS solver...")
    v_bfs = run_bfs_load_flow(branch_csv, bus_csv)
    print("BFS done.\n")

    # Show all three plots sequentially
    plot_voltage_profile(v_dlf, v_bfs)
    plot_voltage_angle(v_dlf, v_bfs)
    plot_difference(v_dlf, v_bfs)


if __name__ == "__main__":
    main()