import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_partregress
import warnings

from notebooks.Util.Data import require_no_missing_values
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_partregress_grid

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
# from sklearn.cluster import fuzzy_cmeans
from scipy.special import softmax
from sklearn.cluster import KMeans
import fastcluster
from scipy.cluster.hierarchy import fcluster
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union

@dataclass
class BinAnovaResult:
    """Statistical results for a single bin in a binning analysis."""
    bin_idx: int
    # y-mean
    mean: float
    # y-std
    std: float
    # number of values in bin
    size: int
    excluded: bool = False
    exclusion_reason: Optional[str] = None

@dataclass
class BinningAnovaResult:
    """Statistical results from an ANOVA analysis on binned data."""
    method: str
    f_value: float
    p_value: float
    bin_results: List[BinAnovaResult]
    
    @property
    def bin_means(self) -> Dict[int, float]:
        """Legacy accessor for bin means (for backwards compatibility)"""
        return {b.bin_idx: b.mean for b in self.bin_results}
    
    @property
    def bin_stds(self) -> Dict[int, float]:
        """Legacy accessor for bin standard deviations (for backwards compatibility)"""
        return {b.bin_idx: b.std for b in self.bin_results}
    
    @property
    def excluded_bins(self) -> Dict[int, str]:
        """Legacy accessor for excluded bins (for backwards compatibility)"""
        return {b.bin_idx: b.exclusion_reason for b in self.bin_results 
                if b.excluded and b.exclusion_reason is not None}

def compute_bin_anova(y: np.ndarray, bins: np.ndarray) -> BinningAnovaResult:
    """
    Compute ANOVA statistics for grouped data.
    
    Parameters:
    -----------
    y : np.ndarray
        The target values to compare across groups
    bins : np.ndarray
        The bin assignments for each data point
        
    Returns:
    --------
    BinningAnovaResult: Result of the ANOVA test with per-bin statistics
    """
    # Get unique bin indices, exclude NaN values
    unique_bins = np.unique(bins[~np.isnan(bins)])
    
    # Group data by bin
    groups = [y[bins == i] for i in unique_bins]
    
    # Create BinAnovaResult for each bin
    bin_results = []
    filtered_groups = []
    filtered_bin_indices = []
    
    for bin_idx, group in zip(unique_bins, groups):
        size = len(group)
        
        # Calculate statistics for this bin
        mean = float(np.mean(group)) if size > 0 else np.nan
        std = float(np.std(group, ddof=1)) if size > 1 else (float(np.std(group, ddof=0)) if size > 0 else np.nan)
        
        # Check if bin should be excluded from ANOVA
        excluded = size < 5
        exclusion_reason = "too few values" if excluded else None
        
        # Create bin result
        bin_result = BinAnovaResult(
            bin_idx=int(bin_idx),
            mean=mean,
            std=std,
            size=size,
            excluded=excluded,
            exclusion_reason=exclusion_reason
        )
        bin_results.append(bin_result)
        
        # Only include bins with sufficient data in ANOVA calculation
        if not excluded:
            filtered_groups.append(group)
            filtered_bin_indices.append(bin_idx)
    
    # Compute ANOVA if we have at least two valid groups
    if len(filtered_groups) >= 2:
        f_value, p_value = scipy_stats.f_oneway(*filtered_groups)
    else:
        f_value, p_value = np.nan, np.nan
        
    return BinningAnovaResult(
        method='ANOVA', 
        f_value=f_value, 
        p_value=p_value, 
        bin_results=bin_results
    )

def compute_t_test(y: np.ndarray, bins: np.ndarray) -> BinningAnovaResult:
    """
    Compute t-test statistics for two groups of data.
    
    Parameters:
    -----------
    y : np.ndarray
        The target values to compare across groups
    bins : np.ndarray
        The bin assignments for each data point (should have exactly 2 unique values)
        
    Returns:
    --------
    BinningAnovaResult: Result of the t-test with per-bin statistics
    """
    # Get unique bin indices, exclude NaN values
    unique_bins = np.unique(bins[~np.isnan(bins)])
    
    if len(unique_bins) != 2:
        raise ValueError("compute_t_test requires exactly two unique bins")
    
    # Group data by bin
    groups = [y[bins == i] for i in unique_bins]
    
    # Create BinAnovaResult for each bin
    bin_results = []
    filtered_groups = []
    filtered_bin_indices = []
    
    for bin_idx, group in zip(unique_bins, groups):
        size = len(group)
        
        # Calculate statistics for this bin
        mean = float(np.mean(group)) if size > 0 else np.nan
        std = float(np.std(group, ddof=1)) if size > 1 else (float(np.std(group, ddof=0)) if size > 0 else np.nan)
        
        # Check if bin should be excluded from t-test
        excluded = size < 5
        exclusion_reason = "too few values" if excluded else None
        
        # Create bin result
        bin_result = BinAnovaResult(
            bin_idx=int(bin_idx),
            mean=mean,
            std=std,
            size=size,
            excluded=excluded,
            exclusion_reason=exclusion_reason
        )
        bin_results.append(bin_result)
        
        # Only include bins with sufficient data in t-test calculation
        if not excluded:
            filtered_groups.append(group)
            filtered_bin_indices.append(bin_idx)
    
    # Compute t-test if we have exactly two valid groups
    if len(filtered_groups) == 2:
        t_value, p_value = scipy_stats.ttest_ind(*filtered_groups, equal_var=False)
    else:
        t_value, p_value = np.nan, np.nan
        
    return BinningAnovaResult(
        method='t_test', 
        f_value=t_value, 
        p_value=p_value, 
        bin_results=bin_results
    )

def compute_binning_anova(binning_result, y: np.ndarray) -> BinningAnovaResult:
    """
    Compute ANOVA statistics directly from a BinningResult.
    
    Parameters:
    -----------
    binning_result : BinningResult
        The binning result containing bins
    y : np.ndarray
        The target values to compare across bins
        
    Returns:
    --------
    BinningAnovaResult: Result of the ANOVA test with per-bin statistics
    """
    # Create bin_results list to store per-bin statistics
    bin_results = []
    
    # Group data for ANOVA calculation
    filtered_groups = []
    filtered_bin_indices = []
    
    # Process each bin
    for bin_obj in binning_result.bins:
        # Get the values in this bin
        if hasattr(bin_obj, 'assignments') and hasattr(bin_obj, 'values'):
            # New style Bin objects with assignments and values
            bin_idx = bin_obj.bin_idx
            size = len(bin_obj.values)
            
            # Use the assignments to get the corresponding y values
            if isinstance(bin_obj.assignments, np.ndarray) and len(bin_obj.assignments) == len(y):
                bin_y = y[bin_obj.assignments]
            else:
                # Fallback if assignments don't match
                bin_y = np.array(bin_obj.values)
        else:
            # Fallback for older structures
            bin_idx = getattr(bin_obj, 'bin_idx', 0)
            bin_mask = binning_result.bin_assignments == bin_idx
            bin_y = y[bin_mask]
            size = len(bin_y)
        
        # Calculate statistics for this bin
        mean = float(np.mean(bin_y)) if size > 0 else np.nan
        std = float(np.std(bin_y, ddof=1)) if size > 1 else (float(np.std(bin_y, ddof=0)) if size > 0 else np.nan)
        
        # Check if bin should be excluded from ANOVA
        excluded = size < 5
        exclusion_reason = "too few values" if excluded else None
        
        # Create bin result
        bin_result = BinAnovaResult(
            bin_idx=int(bin_idx),
            mean=mean,
            std=std,
            size=size,
            excluded=excluded,
            exclusion_reason=exclusion_reason
        )
        bin_results.append(bin_result)
        
        # Only include bins with sufficient data in ANOVA calculation
        if not excluded:
            filtered_groups.append(bin_y)
            filtered_bin_indices.append(bin_idx)
    
    # Compute ANOVA if we have at least two valid groups
    if len(filtered_groups) >= 2:
        if len(filtered_groups) == 2:
            # Use t-test for exactly two groups
            f_value, p_value = scipy_stats.ttest_ind(*filtered_groups, equal_var=False)
            method = 't_test'
        else:
            # Use ANOVA for more than two groups
            f_value, p_value = scipy_stats.f_oneway(*filtered_groups)
            method = 'ANOVA'
    else:
        f_value, p_value = np.nan, np.nan
        method = 'ANOVA' if len(bin_results) > 2 else 't_test'
    
    return BinningAnovaResult(
        method=method,
        f_value=f_value,
        p_value=p_value,
        bin_results=bin_results
    )
