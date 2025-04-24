import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
# from ticc.ticc import TICC
import stumpy

def ticc_pattern_discovery(df: pd.DataFrame, window_size: int, number_of_clusters: int, lambda_param: float, beta: float, max_iters: int = 100, threshold: float = 1e-5, write_out_file: bool = False) -> Tuple[List[int], List[np.ndarray]]:
    data = df.to_numpy()
    model = TICC(window_size=window_size, number_of_clusters=number_of_clusters, lambda_parameter=lambda_param, beta=beta, maxIters=max_iters, threshold=threshold, write_out_file=write_out_file)
    cluster_assignment, cluster_mrfs = model.fit(data)
    return cluster_assignment, cluster_mrfs

# This is for timeseries analysis
def matrix_profile_pattern_discovery(df: pd.DataFrame, window_size: int, top_k: int = 3) -> List[Tuple[Tuple[int, int], float]]:
    data = df.astype(np.float64).to_numpy().T
    P, I = stumpy.mstump(data, m=window_size)
    P_sum = np.sum(P, axis=0)
    idxs = np.argsort(P_sum)[:top_k]
    motifs = []
    for idx in idxs:
        neighbor = I[0, idx]
        distance = P_sum[idx]
        motifs.append(((int(idx), int(neighbor)), float(distance)))
    return motifs

# Also:
# Granger causality
# Vector Autoregression
# Dynamic Time Warping
# Dynamic Mode Decomposition
# Singular Spectrum Analysis
# Empirical Mode Decomposition


# Have a day_data dataframe
# A key looks like this events:luminette:lastSSM	
# Split by : and group by the first part


