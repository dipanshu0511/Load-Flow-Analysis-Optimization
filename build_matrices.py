"""
build_matrices.py
Build BIBC, BCBV and DLF matrices for a radial distribution system.
Corrected for 0-based Python indexing and 33-bus, 32-branch system.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def build_bibc_bcbv_ordered(
    branch_df: pd.DataFrame,
    z_branch_pu: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build BIBC and BCBV matrices using MATLAB-style branch ordering.

    Parameters
    ----------
    branch_df   : pd.DataFrame   columns: branch_num, from_bus, to_bus, R, X
    z_branch_pu : np.ndarray     complex branch impedances in per-unit

    Returns
    -------
    BIBC : np.ndarray, shape (bnn, bn-1)
    BCBV : np.ndarray, shape (bn-1, bnn)
    """
    bnn = len(branch_df)              # number of branches = 32
    bn  = branch_df['to_bus'].max()   # number of buses    = 33

    # BIBC (32×32),  BCBV (32×32)
    BIBC = np.zeros((bnn, bn - 1))
    BCBV = np.zeros((bn - 1, bnn), dtype=complex)

    b = branch_df['from_bus'].to_numpy(dtype=int)
    c = branch_df['to_bus'].to_numpy(dtype=int)

    for k in range(bnn):
        if k == 0:
            BIBC[0, c[0] - 2]    = 1
            BCBV[c[0] - 2, 0]    = z_branch_pu[0]
        else:
            i = b[k] - 2          # parent bus 0-based index
            j = c[k] - 2          # child  bus 0-based index
            BIBC[:, j]  = BIBC[:, i]
            BIBC[k, j]  = 1
            BCBV[j, :]  = BCBV[i, :]
            BCBV[j, k]  = z_branch_pu[k]

    return BIBC, BCBV


def build_dlf(bcbv: np.ndarray, bibc: np.ndarray) -> np.ndarray:
    """
    Compute Distribution Load Flow matrix:  DLF = BCBV × BIBC
    """
    return bcbv.dot(bibc)