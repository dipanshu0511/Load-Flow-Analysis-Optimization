"""
load_flow.py
Perform load flow for 33-bus radial system using BIBC/BCBV matrices.
"""

import numpy as np
import pandas as pd
from build_matrices import build_bibc_bcbv_ordered, build_dlf
from load_data import load_branch_bus_csv


def run_load_flow(branch_csv: str, bus_csv: str) -> np.ndarray:
    """
    Run DLF (BIBC/BCBV) load flow from CSV files.

    Parameters
    ----------
    branch_csv : str   path to branch data CSV
    bus_csv    : str   path to bus data CSV

    Returns
    -------
    v : np.ndarray, complex, shape (nb,)
        Bus voltages in per-unit (index 0 = slack bus)
    """
    # Load data and convert to per-unit
    branch_df, bus_df, z_branch_pu, S_bus_pu, z_base = load_branch_bus_csv(
        branch_csv, bus_csv
    )

    # Build BIBC, BCBV, DLF matrices
    BIBC, BCBV = build_bibc_bcbv_ordered(branch_df, z_branch_pu)
    DLF = build_dlf(BCBV, BIBC)

    nb = len(bus_df)

    # Initialise bus voltages — flat start
    v  = np.ones(nb, dtype=complex)

    # Branch currents (exclude slack bus index 0)
    I1 = np.conj(S_bus_pu[1:] / v[1:])

    tol      = 1e-6
    max_iter = 100

    for _ in range(max_iter):
        v[1:] = v[0] - DLF @ I1
        I2    = np.conj(S_bus_pu[1:] / v[1:])
        if np.max(np.abs(I1 - I2)) < tol:
            break
        I1 = I2

    return v