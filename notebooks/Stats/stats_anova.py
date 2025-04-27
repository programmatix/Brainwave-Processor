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
from typing import Dict, List, Any

@dataclass
class BinningAnovaResult:
    method: str
    f_value: float
    p_value: float
    bin_means: Dict[int, float]
    bin_stds: Dict[int, float]
    excluded_bins: Dict[int, str] = field(default_factory=dict)

def compute_bin_anova(y: np.ndarray, bins: np.ndarray) -> BinningAnovaResult:
    unique_bins = np.unique(bins[~np.isnan(bins)])
    groups = [y[bins == i] for i in unique_bins]
    excluded_bins = {i: "too few values" for i, g in zip(unique_bins, groups) if len(g) < 5}
    filtered_groups = [g for g in groups if len(g) >= 5]

    f_value, p_value = scipy_stats.f_oneway(*filtered_groups)

    bin_means = {i: float(np.mean(g)) if len(g) > 0 else np.nan for i, g in zip(unique_bins, groups)}
    bin_stds = {i: float(np.std(g, ddof=1)) if len(g) > 1 else (float(np.std(g, ddof=0)) if len(g) > 0 else np.nan) for i, g in zip(unique_bins, groups)}
    return BinningAnovaResult(method='ANOVA', f_value=f_value, p_value=p_value, bin_means=bin_means, bin_stds=bin_stds, excluded_bins=excluded_bins)

def compute_t_test(y: np.ndarray, bins: np.ndarray) -> BinningAnovaResult:
    unique_bins = np.unique(bins[~np.isnan(bins)])
    if len(unique_bins) != 2:
        raise ValueError("compute_t_test requires exactly two unique bins")
    groups = [y[bins == i] for i in unique_bins]
    excluded_bins = {i: "too few values" for i, g in zip(unique_bins, groups) if len(g) < 5}
    filtered_groups = [g for g in groups if len(g) >= 5]

    if len(filtered_groups) != 2:
        t_value = np.nan
        p_value = np.nan
    else:
        t_value, p_value = scipy_stats.ttest_ind(*filtered_groups, equal_var=False)

    bin_means = {i: float(np.mean(g)) if len(g) > 0 else np.nan for i, g in zip(unique_bins, groups)}
    bin_stds = {i: float(np.std(g, ddof=1)) if len(g) > 1 else (float(np.std(g, ddof=0)) if len(g) > 0 else np.nan) for i, g in zip(unique_bins, groups)}
    return BinningAnovaResult(method='t_test', f_value=t_value, p_value=p_value, bin_means=bin_means, bin_stds=bin_stds, excluded_bins=excluded_bins)
