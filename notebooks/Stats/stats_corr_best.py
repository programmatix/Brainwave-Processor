import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, DotProduct, WhiteKernel
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mutual_info_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from scipy.stats import f_oneway, ttest_ind
import time
import warnings
from collections import defaultdict
from notebooks.DayData.day_data_processing import isMlKeyUselessOrDeprecated
import statsmodels.api as sm
from joblib import Parallel, delayed
import fastcluster
from scipy.cluster.hierarchy import fcluster, cut_tree
from scipy.spatial.distance import cdist
from statsmodels.nonparametric.smoothers_lowess import lowess
from notebooks.Stats.stats_binning import bin_fastcluster, determine_optimal_bin_count, BinningResult
from notebooks.Stats.stats_clustering import ClusterAnalysis, cluster_fastcluster_wrapper, ClusteringResult
from tqdm.auto import tqdm
from notebooks.Stats.stats_anova import BinningAnovaResult, compute_bin_anova, compute_t_test

@dataclass
class PairAnalysisResult:
    feat1: str
    feat2: str

    X: np.ndarray
    y: np.ndarray
    is_discrete: bool

    # 1D clusters on the x-axis that will be used for anova results
    clusters_x: BinningResult
    anova: BinningAnovaResult

    # 2D clusters
    clusters_2d: ClusteringResult
    clusters_2d_analysis: List[ClusterAnalysis]

    # smoothed values of the y-axis
    lowess_y: np.ndarray
    
    # Across full data
    overall_spearman_rho: float
    overall_spearman_p: float

    # Best p-value across all results
    best_p: float

def check_discrete_x_values(df, feat1, min_unique=5):
    """
    Check if the number of unique values in feat1 is <= min_unique.
    If so, return True (indicating no clustering should be performed).
    """
    unique_values = df[feat1].nunique()
    print("Unique values for ", feat1, ": ", unique_values)
    if unique_values <= min_unique:
        print(f"Skipping clustering: Only {unique_values} unique values in {feat1} (<= {min_unique})")
        return True
    return False


import statsmodels.api as sm
import time
from statsmodels.nonparametric.smoothers_lowess import lowess

def analyze_pair_best(df, x_feat, y_feat, n_clusters=3, random_state=42, use_merging=False, effect_size_threshold=0.3, visualize=False, profile=False):    
    profiling_info = {}
    start_time = time.time()

    data_process_start = time.time()

    df_with_values = df[df[x_feat].notna() & df[y_feat].notna()]

    X = df_with_values[x_feat].values.reshape(-1,1)
    y = df_with_values[y_feat].values

    if profile:
        profiling_info['data_process_ms'] = (time.time() - data_process_start) * 1000

    check_discrete_start = time.time()
    is_discrete = check_discrete_x_values(df_with_values, x_feat)
    if profile:
        profiling_info['check_discrete_ms'] = (time.time() - check_discrete_start) * 1000


    bin_start = time.time()
    bin_count = determine_optimal_bin_count(df_with_values[x_feat], method='fastcluster')
    bin_res = bin_fastcluster(df_with_values[x_feat], n_bins=bin_count, profile=profile)

    if profile:
        profiling_info['bin_time_ms'] = (time.time() - bin_start) * 1000
    
    spearman_start = time.time()
    overall_spearman_rho, overall_spearman_p = stats.spearmanr(X.flatten(), y)
    if profile:
        profiling_info['overall_spearman_time_ms'] = (time.time() - spearman_start) * 1000

    spearman_start = time.time()
    if bin_res.n_bins == 2:
        anova_result = compute_t_test(y, bin_res.bin_assignments)
    else:
        anova_result = compute_bin_anova(y, bin_res.bin_assignments)
    if profile:
        profiling_info['anova_time_ms'] = (time.time() - spearman_start) * 1000
    best_p = min(overall_spearman_p, anova_result.p_value)

    cluster_2d_result = None
    cluster_analyses = None
    smoothed = None

    if not is_discrete:
        if use_merging:
            cluster_start = time.time()
            cluster_2d_result = cluster_with_merging(df_with_values, x_feat, y_feat, cluster_fastcluster_wrapper, param_name='n_clusters', effect_size_threshold=effect_size_threshold, random_state=random_state, n_clusters=n_clusters, profile=profile)
            if profile:
                profiling_info['clustering_time_ms'] = (time.time() - cluster_start) * 1000
        else:
            cluster_start = time.time()
            cluster_2d_result = cluster_fastcluster_wrapper(df_with_values, x_feat, y_feat, n_clusters=n_clusters, random_state=random_state, profile=profile)
            if profile:
                profiling_info['clustering_time_ms'] = (time.time() - cluster_start) * 1000
                if hasattr(cluster_2d_result, 'additional_info') and cluster_2d_result.additional_info:
                    profiling_info['clustering_details'] = cluster_2d_result.additional_info
        
        smoothed = lowess(y, X.flatten(), frac=0.66)

        cluster_analysis_start = time.time()
        cluster_analyses = []
        for label in np.unique(cluster_2d_result.labels):
            mask = cluster_2d_result.labels==label
            Xc = X[mask]
            yc = y[mask]
            size = int(len(yc))
            x_min, x_max = (float(Xc.min()), float(Xc.max())) if size > 0 else (float('nan'), float('nan'))
            y_min, y_max = (float(np.min(yc)), float(np.max(yc))) if size > 0 else (float('nan'), float('nan'))
            y_mean = float(np.mean(yc)) if size > 0 else float('nan')
            y_std = float(np.std(yc)) if size > 0 else float('nan')
            if size > 1:
                spearman_rho_c, spearman_p_c = stats.spearmanr(Xc.flatten(), yc)
            else:
                spearman_rho_c, spearman_p_c = float('nan'), float('nan')
            cluster_analysis = ClusterAnalysis(
                label=int(label),
                size=size,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
                y_mean=y_mean,
                y_std=y_std,
                # regression=None,
                spearman_rho=float(spearman_rho_c),
                spearman_p=float(spearman_p_c)
            )
            cluster_analyses.append(cluster_analysis)
        if profile:
            profiling_info['cluster_analysis_time_ms'] = (time.time() - cluster_analysis_start) * 1000
        
        for c in cluster_analyses:
            if c.size >= 5:
                best_p = min(best_p, c.spearman_p)
        
    pair_result = PairAnalysisResult(
        feat1=x_feat,
        feat2=y_feat,
        X=X,
        y=y,
        is_discrete=is_discrete,
        clusters_x=bin_res,
        anova=anova_result,
        clusters_2d=cluster_2d_result,
        clusters_2d_analysis=cluster_analyses,
        lowess_y=smoothed,
        overall_spearman_rho=overall_spearman_rho,
        overall_spearman_p=overall_spearman_p,
        best_p=best_p
    )
    
    total_time = time.time() - start_time
    if profile:
        profiling_info['total_time_ms'] = total_time * 1000
        profiling_info['feature_pair'] = f"{x_feat}_{y_feat}"
        profiling_info['n_clusters'] = n_clusters
        pair_result.additional_info = profiling_info
        print(f"Profiling for {x_feat} vs {y_feat}:")
        for key, value in profiling_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}ms")
            elif isinstance(value, dict):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
    
    return pair_result

def _plot_anova(ax, result, x_feat, y_feat, bin_colors, bins_to_show):
    df_with_values = pd.DataFrame({x_feat: result.X.flatten(), y_feat: result.y, 'bin': result.clusters_x.bin_assignments})
    sns.boxplot(x='bin', y=y_feat, data=df_with_values, ax=ax, palette=bin_colors, order=bins_to_show, whis=0)
    ax.set_xlabel(f'{x_feat} bins')
    ax.set_ylabel(y_feat)
    ax.set_title(f'{result.anova.method} F={result.anova.f_value:.2f}, p={result.anova.p_value:.4f} (excluded={len(result.anova.excluded_bins)})')
    
    # Create legend entries for each bin
    legend_handles = []
    for i, b in enumerate(bins_to_show):
        mean = result.anova.bin_means[b]
        n = len(result.clusters_x.bin_contents[b])
        legend_handles.append(Patch(color=bin_colors[i], label=f'Bin {b}: n={n}, mean={mean:.2f}'))
    
    ax.legend(handles=legend_handles, title='Bin Details')

def visualise_pair_best(df, x_feat, y_feat, result: PairAnalysisResult):
    if result.is_discrete:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes.scatter(result.X, result.y, alpha=0.7, edgecolors='k')
        axes.set_xlabel(x_feat)
        axes.set_ylabel(y_feat)
        axes.set_title('Scatter Plot')
        bins_to_show = [b for b in result.clusters_x.bin_assignments.unique() if b not in result.anova.excluded_bins]
        bin_colors = sns.color_palette('husl', len(bins_to_show))
        if bins_to_show:
            _plot_anova(axes, result, x_feat, y_feat, bin_colors, bins_to_show)
        else:
            axes.text(0.5, 0.5, 'No bins to display', ha='center', va='center')
            axes.set_title('No bins available for ANOVA')


    if not result.is_discrete:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # Panel 1: 2D clusters with ellipses
        n_clusters_plot = len(result.clusters_2d_analysis)
        cluster_colors = plt.cm.viridis(np.linspace(0,1,n_clusters_plot))
        for i, c in enumerate(result.clusters_2d_analysis):
            mask = result.clusters_2d.labels == c.label
            axes[0].scatter(result.X[mask], result.y[mask],
                            c=[cluster_colors[i]], label=f'n={c.size:3d} s={c.spearman_rho:.2f} p={c.spearman_p:.2f}', alpha=0.7, edgecolors='k')
        # draw ellipses if available
        if result.clusters_2d.means is not None and result.clusters_2d.covariances is not None:
            for i, (mean, cov) in enumerate(zip(result.clusters_2d.means, result.clusters_2d.covariances)):
                if i < n_clusters_plot:
                    try:
                        v, w = np.linalg.eigh(cov)
                        angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
                        width, height = 2*np.sqrt(v)
                        ell = Ellipse(xy=(mean[0], mean[1]), width=width, height=height,
                                        angle=angle, color=cluster_colors[i], alpha=0.3)
                        axes[0].add_patch(ell)
                    except Exception:
                        pass
        axes[0].set_xlabel(x_feat)
        axes[0].set_ylabel(y_feat)
        axes[0].set_title('2D Clusters')
        axes[0].legend()
        # Panel 2: 2D scatter colored by 1D bins with LOESS
        bins_series = result.clusters_x.bin_assignments
        bin_labels = sorted(bins_series.unique())
        bin_colors = sns.color_palette('husl', len(bin_labels))
        for idx, b in enumerate(bin_labels):
            mask_b = bins_series == b
            axes[1].scatter(result.X[mask_b], result.y[mask_b],
                            c=[bin_colors[idx]], label=f'Bin {b} n={len(result.clusters_x.bin_contents[b])}', alpha=0.7, edgecolors='k')
        smoothed = lowess(result.y, result.X.flatten(), frac=0.66)
        axes[1].plot(smoothed[:,0], smoothed[:,1], color='black', lw=2)
        axes[1].set_xlabel(x_feat)
        axes[1].set_ylabel(y_feat)
        axes[1].set_title('1D Bins')
        axes[1].legend()
        # Panel 3: ANOVA boxplot of bin values
        bins_to_show = [b for b in bin_labels if b not in result.anova.excluded_bins]
        if bins_to_show:
            _plot_anova(axes[2], result, x_feat, y_feat, bin_colors, bins_to_show)
        else:
            axes[2].text(0.5, 0.5, 'No bins to display', ha='center', va='center')
            axes[2].set_title('No bins available for ANOVA')
        
        plt.tight_layout()
        plt.show()
        

def print_pair_analysis(pair_result: PairAnalysisResult):
    print(f"Analysis for {pair_result.feat1} vs {pair_result.feat2}")
    print(f"Best p-value: {pair_result.best_p:.3f}")
    #orr = pair_result.overall_regression
    #print(f"Overall Regression: slope={orr.slope:.3f} (CI={orr.slope_ci[0]:.3f}-{orr.slope_ci[1]:.3f}), intercept={orr.intercept:.3f} (CI={orr.intercept_ci[0]:.3f}-{orr.intercept_ci[1]:.3f}), R2={orr.r2:.3f}, p_slope={orr.slope_p_value:.3g}, p_intercept={orr.intercept_p_value:.3g}")
    print(f"Overall Spearman: rho={pair_result.overall_spearman_rho:.3f}, p={pair_result.overall_spearman_p:.3f}")
    a = pair_result.anova
    print(f"Binning:")
    print(f" Method: {pair_result.clusters_x.method}")
    print(f" Count={pair_result.clusters_x.n_bins} excluded from ANOVA={len(a.excluded_bins)}")

    print(f" {a.method}: F={a.f_value:.3f}, p={a.p_value:.3f}")

    for lbl, mean in a.bin_means.items():
        std = a.bin_stds[lbl]
        print(f"  Bin {lbl}: mean={mean:.3f}, std={std:.3f} size={len(pair_result.clusters_x.bin_contents[lbl])} {a.excluded_bins.get(lbl, '')}")
        # print(f"  Bin {lbl}: mean={mean:.3f}, std={std:.3f} size={len(pair_result.clusters_x.bin_contents[lbl])}")
    if pair_result.clusters_2d_analysis is not None:
        for c in pair_result.clusters_2d_analysis:
            print(f"2D Cluster {c.label}: size={c.size}, x_range={c.x_range[0]:.3f}-{c.x_range[1]:.3f}, y_range={c.y_range[0]:.3f}-{c.y_range[1]:.3f}, y_mean={c.y_mean:.3f}, y_std={c.y_std:.3f}")
            #reg = c.regression
            #print(f" Regression: slope={reg.slope:.3f} (CI={reg.slope_ci[0]:.3f}-{reg.slope_ci[1]:.3f}), intercept={reg.intercept:.3f} (CI={reg.intercept_ci[0]:.3f}-{reg.intercept_ci[1]:.3f}), R2={reg.r2:.3f}, p_slope={reg.slope_p_value:.3g}, p_intercept={reg.intercept_p_value:.3g}")
            print(f" Spearman: rho={c.spearman_rho:.3f}, p={c.spearman_p:.3f}")
    else:
        print("No 2D clusters available (probably as discrete data)")

def find_pairwise_best(df, profile=False, min_values=10): 
    import time
    import numpy as np
    
    profiling_info = {}
    start_time = time.time()
    
    out = pd.DataFrame(columns=['feat1', 'feat2', 'best_p', 'error', 'values'])
    if profile:
        out['profiling_info'] = None
    
    n_pairs = len(df.columns) * (len(df.columns) - 1)
    progress = tqdm(total=n_pairs)
    
    pair_times = []
    for c1 in df.columns:
        for c2 in df.columns:
            progress.update(1)
            if c1 == c2:
                continue
                
            pair_start = time.time()
            try:
                df_with_values = df[df[c1].notna() & df[c2].notna()]
                if len(df_with_values) < min_values:
                    row_data = [c1, c2, None, "Not enough values", len(df_with_values)]
                    out = pd.concat([out, pd.DataFrame([row_data], columns=out.columns)], ignore_index=True)
                    continue

                pair_result = analyze_pair_best(df, c1, c2, profile=profile)
                
                row_data = [c1, c2, pair_result.best_p, None, len(df_with_values)]
                if profile:
                    row_data.append(pair_result.additional_info if hasattr(pair_result, 'additional_info') else None)
                    pair_time = (time.time() - pair_start) * 1000  # Convert to ms
                    pair_times.append(pair_time)
                    progress.set_description(f"Avg pair time: {np.mean(pair_times):.2f}ms")
                
                # Use concat instead of loc assignment to avoid the warning
                out = pd.concat([out, pd.DataFrame([row_data], columns=out.columns)], ignore_index=True)
            except Exception as e:
                row_data = [c1, c2, None, str(e), len(df_with_values)]
                if profile:
                    row_data.append(None)
                # Use concat instead of loc assignment to avoid the warning
                out = pd.concat([out, pd.DataFrame([row_data], columns=out.columns)], ignore_index=True)
    
    progress.close()
    
    total_time = time.time() - start_time
    if profile:
        profiling_info['total_time_ms'] = total_time * 1000
        profiling_info['average_pair_time_ms'] = np.mean(pair_times) if pair_times else 0
        profiling_info['median_pair_time_ms'] = np.median(pair_times) if pair_times else 0
        profiling_info['min_pair_time_ms'] = min(pair_times) if pair_times else 0
        profiling_info['max_pair_time_ms'] = max(pair_times) if pair_times else 0
        profiling_info['total_pairs'] = n_pairs
        profiling_info['processed_pairs'] = len(pair_times)
        profiling_info['error_pairs'] = n_pairs - len(pair_times)
        
        print("Overall profiling information:")
        for key, value in profiling_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}ms")
            else:
                print(f"  {key}: {value}")
                
        out.attrs['profiling_info'] = profiling_info
    
    return out




def find_corrs_best(df, y_feat, profile=False, min_values=10): 
    import time
    import numpy as np
    
    profiling_info = {}
    start_time = time.time()
    
    out = pd.DataFrame(columns=['feat1', 'best_p', 'error', 'values', 'spearman_rho'])
    if profile:
        out['profiling_info'] = None
    
    n_pairs = len(df.columns)
    progress = tqdm(total=n_pairs)
    
    pair_times = []
    for x_feat in df.columns:
        progress.update(1)
        if x_feat == y_feat:
            continue

        if isMlKeyUselessOrDeprecated(x_feat):
            continue
            
        pair_start = time.time()
        try:
            df_with_values = df[df[x_feat].notna() & df[y_feat].notna()]
            if len(df_with_values) < min_values:
                row_data = [x_feat, None, "Not enough values", len(df_with_values), None]
                out = pd.concat([out, pd.DataFrame([row_data], columns=out.columns)], ignore_index=True)
                continue

            pair_result = analyze_pair_best(df, x_feat, y_feat, profile=profile)
            
            row_data = [x_feat, pair_result.best_p, None, len(df_with_values), pair_result.overall_spearman_rho]
            if profile:
                row_data.append(pair_result.additional_info if hasattr(pair_result, 'additional_info') else None)
                pair_time = (time.time() - pair_start) * 1000  # Convert to ms
                pair_times.append(pair_time)
                progress.set_description(f"Avg pair time: {np.mean(pair_times):.2f}ms")
            out = pd.concat([out, pd.DataFrame([row_data], columns=out.columns)], ignore_index=True)
        except Exception as e:
            row_data = [x_feat, None, str(e), len(df_with_values), None]
            if profile:
                row_data.append(None)
            out = pd.concat([out, pd.DataFrame([row_data], columns=out.columns)], ignore_index=True)
    
    progress.close()
    
    total_time = time.time() - start_time
    if profile:
        profiling_info['total_time_ms'] = total_time * 1000
        profiling_info['average_pair_time_ms'] = np.mean(pair_times) if pair_times else 0
        profiling_info['median_pair_time_ms'] = np.median(pair_times) if pair_times else 0
        profiling_info['min_pair_time_ms'] = min(pair_times) if pair_times else 0
        profiling_info['max_pair_time_ms'] = max(pair_times) if pair_times else 0
        profiling_info['total_pairs'] = n_pairs
        profiling_info['processed_pairs'] = len(pair_times)
        profiling_info['error_pairs'] = n_pairs - len(pair_times)
        
        print("Overall profiling information:")
        for key, value in profiling_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}ms")
            else:
                print(f"  {key}: {value}")
                
        out.attrs['profiling_info'] = profiling_info
    
    return out

def analyze_top_pairs(df, best_df, n_top=10):
    top_pairs = best_df.sort_values(by='best_p', ascending=True).head(n_top)
    for i, row in top_pairs.iterrows():
        c1, c2 = row['feat1'], row['feat2']
        print(f'\n{i}: {c1} vs {c2}')
        best = analyze_pair_best(df, c1, c2, profile=True, visualize=True)
        print_pair_analysis(best)





