# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:38:03 2026

@author: Dipanshu Tripathi
"""

"""
bfs_flow.py
Perform load flow for 33-bus radial system using Backward-Forward Sweep (BFS) method.
Aligned with existing DLF-based load flow codebase.
"""

import numpy as np
import pandas as pd
from load_data import load_branch_bus_csv


def build_network_tree(branch_df: pd.DataFrame):
    """
    Build parent-child relationships and ordered traversal lists from branch data.

    Parameters
    ----------
    branch_df : pd.DataFrame
        Expected columns: branch_num, from_bus, to_bus, R, X

    Returns
    -------
    children   : dict  {bus: [child_buses]}
    parent     : dict  {bus: parent_bus}
    bfs_order  : list  buses in breadth-first order (root first)
    """
    from_buses = branch_df['from_bus'].to_numpy(dtype=int)
    to_buses   = branch_df['to_bus'].to_numpy(dtype=int)

    children = {}
    parent   = {}

    for fb, tb in zip(from_buses, to_buses):
        children.setdefault(fb, []).append(tb)
        children.setdefault(tb, [])   # ensure every bus is in dict
        parent[tb] = fb

    # BFS traversal order starting from slack bus (bus 1)
    root = int(from_buses[0])
    bfs_order = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        bfs_order.append(node)
        for child in children.get(node, []):
            queue.append(child)

    return children, parent, bfs_order


def run_bfs_load_flow(branch_csv: str, bus_csv: str):
    """
    Run Backward-Forward Sweep load flow.

    Parameters
    ----------
    branch_csv : str   Path to branch data CSV
    bus_csv    : str   Path to bus data CSV

    Returns
    -------
    v : np.ndarray, shape (nb,), complex
        Bus voltages in per-unit (index 0 = slack bus)
    """
    # ------------------------------------------------------------------ #
    # 1. Load data (same loader as DLF method)
    # ------------------------------------------------------------------ #
    branch_df, bus_df, z_branch_pu, S_bus_pu, z_base = load_branch_bus_csv(
        branch_csv, bus_csv
    )

    nb  = len(bus_df)       # 33 buses
    bnn = len(branch_df)    # 32 branches

    from_buses = branch_df['from_bus'].to_numpy(dtype=int)
    to_buses   = branch_df['to_bus'].to_numpy(dtype=int)

    # Map (from_bus, to_bus) → branch index for quick lookup
    branch_index = {
        (int(from_buses[k]), int(to_buses[k])): k for k in range(bnn)
    }

    # ------------------------------------------------------------------ #
    # 2. Build network tree structure
    # ------------------------------------------------------------------ #
    children, parent, bfs_order = build_network_tree(branch_df)

    # ------------------------------------------------------------------ #
    # 3. Initialise voltages (flat start: all 1.0 + 0j p.u.)
    # ------------------------------------------------------------------ #
    v = np.ones(nb, dtype=complex)   # index i → bus (i+1)

    tol      = 1e-5
    max_iter = 100

    # ------------------------------------------------------------------ #
    # 4. Iterative BFS loop
    # ------------------------------------------------------------------ #
    for iteration in range(max_iter):

        v_prev = v.copy()

        # -------------------------------------------------------------- #
        # BACKWARD SWEEP — leaves → root
        # Compute branch currents from load-end toward slack bus
        # -------------------------------------------------------------- #
        I_branch = np.zeros(bnn, dtype=complex)   # current in each branch

        # Traverse in reverse BFS order (leaves first, root last)
        for bus in reversed(bfs_order):
            if bus == bfs_order[0]:          # skip slack (root) bus
                continue

            bus_idx = bus - 1                # 0-based index

            # Load current injected at this bus
            I_load = np.conj(S_bus_pu[bus_idx] / v[bus_idx])

            # Branch feeding this bus
            pb  = parent[bus]
            k   = branch_index[(pb, bus)]

            # Branch current = load current + sum of currents in child branches
            I_branch[k] = I_load
            for child in children.get(bus, []):
                k_child = branch_index[(bus, child)]
                I_branch[k] += I_branch[k_child]

        # -------------------------------------------------------------- #
        # FORWARD SWEEP — root → leaves
        # Update bus voltages using branch currents
        # -------------------------------------------------------------- #
        for bus in bfs_order:
            if bus == bfs_order[0]:          # slack bus voltage fixed
                continue

            pb      = parent[bus]
            k       = branch_index[(pb, bus)]
            pb_idx  = pb  - 1               # 0-based parent index
            bus_idx = bus - 1               # 0-based current bus index

            # V_child = V_parent - Z_branch * I_branch
            v[bus_idx] = v[pb_idx] - z_branch_pu[k] * I_branch[k]

        # -------------------------------------------------------------- #
        # Convergence check
        # -------------------------------------------------------------- #
        if np.max(np.abs(v - v_prev)) < tol:
            print(f"BFS converged in {iteration + 1} iterations.")
            break

    else:
        print(f"BFS did NOT converge within {max_iter} iterations.")

    return v