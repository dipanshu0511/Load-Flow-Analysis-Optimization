"""
load_flow.py

Perform load flow for 33-bus radial system using BIBC/BCBV matrices.
"""

import numpy as np
import pandas as pd
from build_matrices import build_bibc_bcbv_ordered, build_dlf
from load_data import load_branch_bus_csv

def run_load_flow(branch_csv: str, bus_csv: str):
    # Load data and convert to per-unit
    branch_df, bus_df, z_branch_pu, S_bus_pu, z_base = load_branch_bus_csv(branch_csv, bus_csv)

    # Build BIBC, BCBV, DLF matrices
    BIBC, BCBV = build_bibc_bcbv_ordered(branch_df, z_branch_pu)
    DLF = build_dlf(BCBV, BIBC)

    nb = len(bus_df)
    bnn = len(branch_df)

    # Initialize bus voltages
    v = np.ones(nb, dtype=complex)
    v0 = v.copy()

    # Branch currents (exclude slack bus, i.e., bus 1)
    I1 = np.conj(S_bus_pu[1:] / v[1:])

    tol = 1e-5
    max_iter = 100

    # Iterative load flow
    for _ in range(max_iter):
        v[1:] = v0[1:] - DLF @ I1
        I2 = np.conj(S_bus_pu[1:] / v[1:])
        if np.max(np.abs(I1 - I2)) < tol:
            break
        I1 = I2

    return v
