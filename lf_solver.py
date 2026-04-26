# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:46:47 2026

@author: Dipanshu Tripathi
"""

"""
lf_solver.py
Standalone load flow solver that accepts pre-built matrices and S_bus_pu
directly — no CSV re-reading. Used internally by dg_placement PSO loop
to avoid file I/O overhead and allow clean S_bus_pu modification.
"""

import numpy as np
from build_matrices import build_bibc_bcbv_ordered, build_dlf


def run_dlf_direct(branch_df, z_branch_pu, S_bus_pu):
    """
    Run DLF (BIBC/BCBV) load flow given pre-loaded data.

    Parameters
    ----------
    branch_df   : pd.DataFrame   branch data
    z_branch_pu : np.ndarray     complex branch impedances p.u.
    S_bus_pu    : np.ndarray     complex bus power injections p.u.
                                 (loads are positive, DG injection is negative)

    Returns
    -------
    v : np.ndarray, complex, shape (nb,)
        Bus voltages in per-unit
    """
    BIBC, BCBV = build_bibc_bcbv_ordered(branch_df, z_branch_pu)
    DLF        = build_dlf(BCBV, BIBC)

    nb  = S_bus_pu.shape[0]
    v   = np.ones(nb, dtype=complex)

    tol      = 1e-6
    max_iter = 100

    I1 = np.conj(S_bus_pu[1:] / v[1:])

    for _ in range(max_iter):
        v[1:] = v[0] - DLF @ I1
        I2    = np.conj(S_bus_pu[1:] / v[1:])
        if np.max(np.abs(I1 - I2)) < tol:
            break
        I1 = I2

    return v