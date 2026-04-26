"""
load_data.py
Load branch and bus CSV files for the 33-bus radial system
and convert values to per-unit.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import os


def load_branch_bus_csv(
    branch_csv: str,
    bus_csv: str,
    vbase_kv: float = 12.66,
    sbase_mva: float = 100.0
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, float]:
    """
    Load branch and bus CSVs and convert to per-unit.

    Parameters
    ----------
    branch_csv : str     path to branch data CSV
    bus_csv    : str     path to bus data CSV
    vbase_kv   : float   base voltage in kV  (default 12.66 kV)
    sbase_mva  : float   base apparent power in MVA (default 100 MVA)

    Returns
    -------
    branch_df   : pd.DataFrame   raw branch data
    bus_df      : pd.DataFrame   raw bus data
    z_branch_pu : np.ndarray     complex branch impedances in p.u.
    S_bus_pu    : np.ndarray     complex bus power injections in p.u.
    z_base      : float          base impedance in ohms
    """
    if not os.path.exists(branch_csv):
        raise FileNotFoundError(f"Branch CSV not found: {branch_csv}")
    if not os.path.exists(bus_csv):
        raise FileNotFoundError(f"Bus CSV not found: {bus_csv}")

    branch_df = pd.read_csv(branch_csv)
    bus_df    = pd.read_csv(bus_csv)

    # Base impedance
    z_base = (vbase_kv ** 2) / sbase_mva

    # Branch impedances → per-unit
    R_orig = branch_df["R"].to_numpy(dtype=float)
    X_orig = branch_df["X"].to_numpy(dtype=float)
    z_branch_pu = (R_orig / z_base) + 1j * (X_orig / z_base)

    # Bus power → per-unit complex power
    P_orig   = bus_df["P"].to_numpy(dtype=float)
    Q_orig   = bus_df["Q"].to_numpy(dtype=float)
    S_bus_pu = (P_orig / sbase_mva) + 1j * (Q_orig / sbase_mva)

    return branch_df, bus_df, z_branch_pu, S_bus_pu, z_base