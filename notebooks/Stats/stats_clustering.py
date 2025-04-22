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

# Set a seed for reproducibility
np.random.seed(42)

@dataclass
class ClusteringResult:
    """Standard container for clustering results to ensure consistency across methods."""
    name: str
    labels: np.ndarray
    probs: np.ndarray
    means: np.ndarray
    covariances: Optional[np.ndarray] = None
    model: Any = None
    computation_time: float = 0.0
    weights: Optional[np.ndarray] = None
    active_components: Optional[int] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

def find_optimal_data_subsets(df, feat1, feat2, model_factory, model_name, n_clusters=3, use_2d_clustering=True):
    """
    Find data subsets where a model performs particularly well using GMM clustering and local validation.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for shower time
    feat2 : str
        Column name for LEP time
    model_factory : function
        A function that takes X and y and returns a fitted model
    model_name : str
        Name of the model for display purposes
    n_clusters : int
        Number of clusters to try
    use_2d_clustering : bool
        Whether to cluster based on both X and Y (True) or just X (False)
        
    Returns:
    --------
    None (prints results and plots charts)
    dict or None:
        A dictionary containing clustering results and metrics if successful,
        otherwise None. Keys include:
        - 'gmm_labels': Cluster assignment for each point in df.
        - 'gmm_probs': Cluster membership probabilities for each point.
        - 'gmm_means': Mean of each GMM component.
        - 'gmm_covariances': Covariance of each GMM component.
        - 'cluster_metrics': List of dicts, one per cluster, with performance metrics.
        - 'n_clusters': The actual number of clusters found.
        - 'use_2d_clustering': Boolean flag indicating clustering dimension.
        - 'model_name': Name of the model analyzed.
        - 'feat1': Name of the independent variable.
        - 'feat2': Name of the dependent variable.
    """
    if len(df) < n_clusters * 2: # Ensure enough data for clustering and CV
        print(f"Skipping subset analysis for {model_name}: Insufficient data points ({len(df)}) for {n_clusters} clusters.")
        return None
        
    X_1d = df[feat1].values.reshape(-1, 1)
    y = df[feat2].values
    
    # For 2D clustering, include both variables
    if use_2d_clustering:
        X_2d = np.column_stack((df[feat1].values, df[feat2].values))
        print(f"Using 2D GMM clustering on both {feat1} and {feat2}")
    else:
        X_2d = X_1d
        if len(np.unique(X_1d)) < n_clusters:
            print(f"Skipping subset analysis for {model_name}: Not enough unique X values for {n_clusters} clusters in 1D.")
            return None
        print(f"Using 1D GMM clustering on {feat1} only")
    
    print(f"Finding GMM subsets for {model_name}...")
    
    # Use only Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(X_2d)
    gmm_labels = gmm.predict(X_2d)
    gmm_probs = gmm.predict_proba(X_2d)
    
    # Check if GMM converged and produced the expected number of clusters
    actual_n_clusters = len(np.unique(gmm_labels))
    if actual_n_clusters < 2: # Need at least 2 clusters to be meaningful
        print(f"Warning: GMM clustering resulted in only {actual_n_clusters} cluster(s). Skipping detailed subset analysis.")
        return None
    elif actual_n_clusters < n_clusters:
        print(f"Warning: GMM clustering resulted in {actual_n_clusters} clusters, fewer than the requested {n_clusters}.")
        # Proceeding with the clusters found
        
    # Get component details (handle potential differences in attribute names across sklearn versions)
    gmm_means = gmm.means_
    gmm_covariances = gmm.covariances_
        
    # Create a discrete colormap for the clusters
    # distinct_colors = plt.cm.tab10(np.linspace(0, 1, actual_n_clusters))
    
    # -- No plotting directly in this function anymore --
    # fig = plt.figure(figsize=(16, 10))
    # gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    # ax_scatter = plt.subplot(gs[0, :])
    # ax_bars = plt.subplot(gs[1, :])
    
    cluster_results = [] # Store results for each cluster
    debug_info = {"GMM": {}}
    
    unique_labels = np.unique(gmm_labels)
    cluster_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Evaluate each cluster
    print(f"\n--- {model_name} Performance within GMM Clusters ---")
    for label in unique_labels:
        mask = gmm_labels == label
        X_cluster = X_1d[mask]
        y_cluster = y[mask]
        cluster_idx = cluster_map[label] # Consistent index 0 to k-1
        
        # Skip if cluster is too small for meaningful evaluation
        if len(X_cluster) < 5: 
            print(f"Skipping GMM cluster {label} due to insufficient data points ({len(X_cluster)} < 5)")
            continue
            
        cluster_info = {
            'label': int(label), # Original GMM label
            'size': int(np.sum(mask)),
            'X_range': (float(X_cluster.min()), float(X_cluster.max())),
            'y_range': (float(y_cluster.min()), float(y_cluster.max())),
            'X_std': float(X_cluster.std()) if len(X_cluster) > 1 else 0.0,
            'y_std': float(y_cluster.std()) if len(y_cluster) > 1 else 0.0,
        }
        
        # Evaluate model performance within this cluster using Cross-Validation
        kf = KFold(n_splits=min(3, len(X_cluster)), shuffle=True, random_state=42)
        scores = []
        
        for train_idx, test_idx in kf.split(X_cluster):
            # Ensure enough data points for the model factory
            if len(train_idx) < 2 or len(test_idx) < 1: 
                continue
            
            # Need to handle potential errors in model fitting (e.g., convergence)
            try:
                model = model_factory(X_cluster[train_idx], y_cluster[train_idx])
                y_pred_cv = model.predict(X_cluster[test_idx])
            except Exception as e:
                print(f"  Warning: Model fitting/prediction failed for CV fold in cluster {label}: {e}")
                # Assign NaN score if prediction fails for a fold
                scores.append(float('nan')) 
                continue

            # Check if y_true_cv has variance before calculating r2
            if len(np.unique(y_cluster[test_idx])) > 1:
                scores.append(r2_score(y_cluster[test_idx], y_pred_cv))
            else:
                # Handle cases with no variance in test set
                # If prediction matches the constant value, R2 is 1, else it can be considered 0 or negative infinity
                # We'll use 0 as a more stable value than -inf
                score = 1.0 if np.allclose(y_cluster[test_idx], y_pred_cv) else 0.0
                scores.append(score)
        
        # Calculate average score, handling potential NaNs from failed folds
        valid_scores = [s for s in scores if not np.isnan(s)]
        avg_score = np.mean(valid_scores) if valid_scores else float('nan')
        
        # Calculate Spearman correlation and p-value within the cluster
        if len(X_cluster) > 1:
            try:
                spearman_rho, spearman_p = stats.spearmanr(X_cluster.flatten(), y_cluster)
            except ValueError:
                # Handle cases where spearmanr might fail (e.g., constant data)
                spearman_rho, spearman_p = float('nan'), float('nan')
        else:
             spearman_rho, spearman_p = float('nan'), float('nan')

        # Calculate Mutual Information within the cluster
        if len(X_cluster) > 1:
            # Ensure y_cluster is not constant for MI calculation
            if len(np.unique(y_cluster)) > 1:
                 mi = mutual_info_regression(X_cluster, y_cluster, random_state=42)[0]
            else:
                 mi = 0.0 # MI is 0 if one variable is constant
        else:
             mi = float('nan')
        
        # Add results to cluster info
        cluster_info['cv_r2'] = float(avg_score)
        cluster_info['spearman_rho'] = float(spearman_rho)
        cluster_info['spearman_p'] = float(spearman_p)
        cluster_info['mi'] = float(mi)
        
        print(f"  GMM - Cluster {label} (Index {cluster_idx}):")
        print(f"    Size: {cluster_info['size']} points")
        print(f"    {feat1} range: {cluster_info['X_range'][0]:.2f} to {cluster_info['X_range'][1]:.2f} (std={cluster_info['X_std']:.2f})")
        print(f"    {feat2} range: {cluster_info['y_range'][0]:.2f} to {cluster_info['y_range'][1]:.2f} (std={cluster_info['y_std']:.2f})")
        print(f"    CV R² Score: {cluster_info['cv_r2']:.4f}" + (" (Note: Negative R² indicates model performs worse than predicting the mean)" if cluster_info['cv_r2'] < 0 else ""))
        print(f"    Spearman Correlation (ρ): {cluster_info['spearman_rho']:.4f}")
        print(f"    Spearman p-value: {cluster_info['spearman_p']:.4f}")
        print(f"    Mutual Information: {cluster_info['mi']:.4f}")
        
        # Add results to the list
        cluster_results.append(cluster_info)
        debug_info["GMM"][f"cluster_{label}"] = cluster_info
    
    # --- Remove plotting from this function --- 
    # # Plot bar chart of cluster performance with matching colors
    # ... (bar chart plotting code removed)
    
    # fig.tight_layout(rect=[0, 0.03, 1, 0.97]) 
    # plt.show()
    
    # Return the collected data
    return {
        'gmm_labels': gmm_labels,
        'gmm_probs': gmm_probs,
        'gmm_means': gmm_means,
        'gmm_covariances': gmm_covariances,
        'cluster_metrics': cluster_results, # List of dicts for each valid cluster
        'n_clusters': actual_n_clusters, # Actual number of clusters analyzed
        'use_2d_clustering': use_2d_clustering,
        'model_name': model_name,
        'feat1': feat1,
        'feat2': feat2,
        # 'debug_info': debug_info # Optionally return debug info
    }

def analyze_clusters_with_anova(df, feat1, feat2, model_factory=None, model_name="Default Model", 
                                n_clusters=3, use_2d_clustering=True, remove_outliers=True, 
                                outlier_method='influence', max_remove_percent=10):
    """
    Analyze data by removing outliers, identifying clusters, visualizing them, and 
    performing ANOVA analysis across the clusters.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for independent variable
    feat2 : str
        Column name for dependent variable
    model_factory : function or None
        A function that takes X and y and returns a fitted model (for outlier detection)
    model_name : str
        Name of the model for display purposes
    n_clusters : int
        Number of clusters to try with GMM
    use_2d_clustering : bool
        Whether to cluster based on both X and Y (True) or just X (False)
    remove_outliers : bool
        Whether to remove outliers before clustering
    outlier_method : str
        Method for outlier detection ('residual', 'distance', 'influence')
    max_remove_percent : float
        Maximum percentage of points to remove as outliers (0-100)
        
    Returns:
    --------
    dict or None:
        A dictionary containing all results if successful, including:
        - 'cluster_results': Results from find_optimal_data_subsets
        - 'anova_results': Results from the ANOVA analysis
        - 'clean_df': DataFrame with outliers removed and cluster labels added
        - Other metrics and analysis results
    """
    # Import detect_outliers function dynamically
    from notebooks.Stats.stats_outliers import detect_outliers
    
    print(f"\n--- Analyzing {feat1} vs {feat2} with {model_name} ---")
    
    # Set up initial variables
    X = df[feat1].values.reshape(-1, 1)
    y = df[feat2].values
    clean_df = df.copy()
    
    # Default model factory if none provided
    if model_factory is None:
        def model_factory(X_train, y_train):
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model
    
    # Step 1: Remove outliers if requested
    if remove_outliers:
        print(f"Removing outliers using '{outlier_method}' method...")
        X_clean, y_clean, outlier_mask, outlier_scores, _ = detect_outliers(
            X, y, method=outlier_method, 
            max_remove_percent=max_remove_percent,
            model_factory=model_factory
        )
        
        # Create clean dataframe with outliers removed
        clean_df = df.iloc[outlier_mask].copy()
        print(f"After outlier removal: {len(clean_df)} data points remaining ({len(df) - len(clean_df)} removed)")
    else:
        X_clean = X
        y_clean = y
        outlier_mask = np.ones(len(df), dtype=bool)
        outlier_scores = np.zeros(len(df))
        print(f"Skipping outlier removal. Using all {len(df)} data points.")
    
    # Step 2: Find clusters using GMM
    print(f"\nFinding clusters in the data...")
    cluster_results = find_optimal_data_subsets(
        clean_df, feat1, feat2, 
        model_factory=model_factory,
        model_name=model_name,
        n_clusters=n_clusters,
        use_2d_clustering=use_2d_clustering
    )
    
    if cluster_results is None:
        print(f"Clustering failed. Cannot perform ANOVA analysis.")
        return None
    
    # Add cluster labels to the clean dataframe
    clean_df['cluster'] = cluster_results['gmm_labels']
    
    # Reorder clusters from left to right (along feat1 axis)
    # Get the mean position of each cluster along feat1 axis
    unique_clusters = np.unique(clean_df['cluster'])
    cluster_means_x = {}
    for label in unique_clusters:
        mask = clean_df['cluster'] == label
        cluster_means_x[label] = clean_df.loc[mask, feat1].mean()
    
    # Sort clusters by their mean x position
    sorted_clusters = sorted(cluster_means_x.items(), key=lambda x: x[1])
    
    # Create mapping from original labels to sorted labels (0, 1, 2, ...)
    cluster_mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(sorted_clusters)}
    
    # Remap the cluster labels
    clean_df['cluster'] = clean_df['cluster'].map(cluster_mapping)
    
    # Update the cluster labels in the cluster_results
    remapped_labels = np.array([cluster_mapping[label] for label in cluster_results['gmm_labels']])
    cluster_results['gmm_labels'] = remapped_labels
    
    # Also remap the cluster metrics if they exist
    if 'cluster_metrics' in cluster_results:
        for metric in cluster_results['cluster_metrics']:
            if 'label' in metric:
                metric['label'] = cluster_mapping[metric['label']]
    
    # Step 3: Visualize the clusters
    # Use viridis color scheme instead of tab10
    distinct_colors = plt.cm.viridis(np.linspace(0, 1, cluster_results['n_clusters']))
    
    # Create visualization
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2)
    
    # Plot 1: Scatter plot with clusters
    ax_scatter = plt.subplot(gs[0, 0])
    for label in np.unique(clean_df['cluster']):
        mask = clean_df['cluster'] == label
        ax_scatter.scatter(
            clean_df.loc[mask, feat1], 
            clean_df.loc[mask, feat2],
            c=[distinct_colors[label]],
            label=f'Cluster {label} (n={np.sum(mask)})',
            alpha=0.7, s=50, edgecolors='k'
        )
    
    # Plot GMM ellipses if 2D clustering was used
    if use_2d_clustering:
        means = cluster_results['gmm_means']
        covariances = cluster_results['gmm_covariances']
        for j, (mean, covar) in enumerate(zip(means, covariances)):
            if np.all(np.linalg.eigvalsh(covar) > 1e-6):
                try:
                    v, w = np.linalg.eigh(covar)
                    angle = np.arctan2(w[1, 0], w[0, 0])
                    angle = 180 * angle / np.pi
                    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                    ell = Ellipse(
                        xy=(mean[0], mean[1]), 
                        width=v[0], height=v[1], 
                        angle=angle,
                        color=distinct_colors[j], 
                        alpha=0.25, zorder=1
                    )
                    ax_scatter.add_artist(ell)
                    ax_scatter.scatter(
                        mean[0], mean[1], 
                        marker='X', s=100, 
                        c=[distinct_colors[j]], 
                        edgecolors='black', zorder=11
                    )
                except np.linalg.LinAlgError:
                    print(f"Warning: Could not draw ellipse for cluster {j}.")
            else:
                print(f"Warning: Skipping ellipse for cluster {j} due to non-positive definite covariance.")
    
    ax_scatter.set_xlabel(feat1)
    ax_scatter.set_ylabel(feat2)
    ax_scatter.set_title(f"GMM Clusters for {feat1} vs {feat2}")
    ax_scatter.legend(loc='best')
    ax_scatter.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Box plot for feat2 across clusters
    ax_box = plt.subplot(gs[0, 1])
    sns.boxplot(x='cluster', y=feat2, data=clean_df, ax=ax_box, palette=distinct_colors)
    
    # Step 4: Perform ANOVA
    print("\n--- ANOVA Analysis across Clusters ---")
    
    # ANOVA for feat2 across clusters
    unique_clusters = np.unique(clean_df['cluster'])
    groups = [clean_df.loc[clean_df['cluster'] == c, feat2].values for c in unique_clusters]
    
    # Add cluster means and std to the box plot
    for i, cluster in enumerate(unique_clusters):
        group = groups[i]
        cluster_mask = clean_df['cluster'] == cluster
        mean_val = np.mean(group)
        std_val = np.std(group)
        
        # Get X range for this cluster
        x_min = clean_df.loc[cluster_mask, feat1].min()
        x_max = clean_df.loc[cluster_mask, feat1].max()
        
        ax_box.text(
            i, ax_box.get_ylim()[0], 
            f'μ={mean_val:.2f}\nσ={std_val:.2f}\nn={len(group)}\nX:\n{x_min:.2f}-{x_max:.2f}',
            ha='center', va='bottom',
            fontsize=9
        )
    
    # Only perform ANOVA if we have at least 2 clusters
    if len(groups) >= 2:
        f_val, p_val = stats.f_oneway(*groups)
        print(f"ANOVA F-value: {f_val:.4f}")
        print(f"ANOVA p-value: {p_val:.2f}")
        print(f"Significant difference between clusters: {p_val < 0.05}")
        
        # Add F and P values to box plot title
        sig_text = "significant" if p_val < 0.05 else "not significant"
        ax_box.set_title(f"Distribution of {feat2} across clusters\nANOVA: F={f_val:.2f}, p={p_val:.2f} ({sig_text})")
        
        # Post-hoc Tukey HSD test for pairwise comparisons if ANOVA is significant
        tukey_results = None
        if p_val < 0.05:
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                tukey_results = pairwise_tukeyhsd(
                    clean_df[feat2].values, 
                    clean_df['cluster'].values,
                    alpha=0.05
                )
                print("\nTukey HSD Post-hoc Test:")
                print(tukey_results)
            except ImportError:
                print("statsmodels not available for Tukey HSD test")
                # Fallback to basic t-tests if statsmodels not available
                print("\nPairwise t-tests (not adjusted for multiple comparisons):")
                for i, c1 in enumerate(unique_clusters):
                    for c2 in unique_clusters[i+1:]:
                        t_val, t_p = stats.ttest_ind(
                            clean_df.loc[clean_df['cluster'] == c1, feat2].values,
                            clean_df.loc[clean_df['cluster'] == c2, feat2].values
                        )
                        print(f"Cluster {c1} vs Cluster {c2}: t={t_val:.4f}, p={t_p:.4f}")
        
        anova_results = {
            'f_value': f_val,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'cluster_means': dict(zip([str(c) for c in unique_clusters], [np.mean(g) for g in groups])),
            'cluster_stds': dict(zip([str(c) for c in unique_clusters], [np.std(g) for g in groups])),
            'tukey_results': str(tukey_results) if tukey_results is not None else None
        }
    else:
        print("Not enough clusters for ANOVA analysis")
        anova_results = None
        
    fig.tight_layout()
    plt.show()
    
    # Prepare and return results
    results = {
        'cluster_results': cluster_results,
        'anova_results': anova_results,
        'clean_df': clean_df,
        'outlier_mask': outlier_mask,
        'outlier_scores': outlier_scores,
        'n_outliers_removed': len(df) - len(clean_df)
    }
    
    return results

def sort_clusters_by_position(result, df, feat1):
    """
    Sort cluster labels from left to right along the x-axis.
    
    Parameters:
    -----------
    result : ClusteringResult
        The clustering result object
    df : DataFrame
        The dataframe with the data
    feat1 : str
        Column name for the x-axis feature
        
    Returns:
    --------
    ClusteringResult
        Updated result with remapped cluster labels
    """
    unique_labels = np.unique(result.labels)
    
    # Calculate mean position of each cluster along feat1 axis
    cluster_means_x = {}
    for label in unique_labels:
        mask = result.labels == label
        cluster_means_x[label] = df.loc[mask, feat1].mean()
    
    # Sort clusters by their mean x position
    sorted_clusters = sorted(cluster_means_x.items(), key=lambda x: x[1])
    
    # Create mapping from original labels to sorted labels (0, 1, 2, ...)
    cluster_mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(sorted_clusters)}
    
    # Remap the cluster labels
    new_labels = np.array([cluster_mapping[label] for label in result.labels])
    
    # Remap the probabilities matrix - handle case where dims might not match
    n_samples = len(df)
    n_clusters = len(unique_labels)
    new_probs = np.zeros((n_samples, n_clusters))
    
    # Map old probabilities to new positions based on cluster mapping
    if hasattr(result, 'probs') and result.probs is not None:
        old_probs = result.probs
        
        # Case 1: Simple remapping when dimensions match
        if old_probs.shape[1] == n_clusters:
            for old_label, new_label in cluster_mapping.items():
                old_idx = np.where(unique_labels == old_label)[0][0]
                if old_idx < old_probs.shape[1]:
                    new_probs[:, new_label] = old_probs[:, old_idx]
        # Case 2: If probs matrix has different dimensions, fall back to one-hot
        else:
            for old_label, new_label in cluster_mapping.items():
                mask = result.labels == old_label
                new_probs[mask, new_label] = 1.0
                
        # Normalize rows to sum to 1
        row_sums = new_probs.sum(axis=1)
        for i in range(n_samples):
            if row_sums[i] > 0:
                new_probs[i, :] = new_probs[i, :] / row_sums[i]
    else:
        # If no probs, create one-hot encoding
        for i, label in enumerate(new_labels):
            new_probs[i, label] = 1.0
    
    # Remap means and covariances (if present)
    new_means = np.zeros_like(result.means) if result.means is not None else np.zeros((n_clusters, 2))
    new_covariances = None
    
    if result.means is not None:
        for old_label, new_label in cluster_mapping.items():
            old_idx = np.where(unique_labels == old_label)[0][0]
            if old_idx < len(result.means) and new_label < len(new_means):
                new_means[new_label] = result.means[old_idx]
    
    if result.covariances is not None:
        new_covariances = np.zeros_like(result.covariances)
        for old_label, new_label in cluster_mapping.items():
            old_idx = np.where(unique_labels == old_label)[0][0]
            if old_idx < len(result.covariances) and new_label < len(new_covariances):
                new_covariances[new_label] = result.covariances[old_idx]
    
    # Create updated result
    updated_result = ClusteringResult(
        name=result.name,
        labels=new_labels,
        probs=new_probs,
        means=new_means,
        covariances=new_covariances,
        model=result.model,
        computation_time=result.computation_time,
        weights=result.weights,
        active_components=result.active_components,
        additional_info=result.additional_info
    )
    
    return updated_result

def cluster_mixture_t(df, feat1, feat2, n_clusters=3, random_state=42):
    """
    Perform clustering using Mixture of Student's t-distributions.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    ClusteringResult
        Standardized clustering result object
    """
    from sklearn.mixture import GaussianMixture
    from scipy import stats
    import time
    
    start_time = time.time()
    
    # Extract features
    X = np.column_stack((df[feat1].values, df[feat2].values))
    
    # Since sklearn doesn't have a Student's t-distribution mixture model,
    # we'll use a GaussianMixture as a base and adjust for heavier tails
    gmm = GaussianMixture(
        n_components=n_clusters, 
        covariance_type='full', 
        random_state=random_state
    )
    gmm.fit(X)
    
    # Get cluster probabilities (soft assignments)
    probs = gmm.predict_proba(X)
    
    # Get hard cluster assignments
    labels = gmm.predict(X)
    
    # Student's t adjustment - we simulate heavier tails by adjusting probabilities
    # This is an approximation, not a true t-mixture model
    df_freedom = 5  # Degrees of freedom for t-distribution (lower = heavier tails)
    adjusted_probs = np.zeros_like(probs)
    
    for i in range(n_clusters):
        # Get cluster center and covariance
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]
        
        # Calculate Mahalanobis distances
        inv_cov = np.linalg.inv(cov)
        for j in range(len(X)):
            diff = X[j] - mean
            m_dist = np.sqrt(diff.dot(inv_cov).dot(diff.T))
            
            # Adjust probability using t-distribution cdf vs normal cdf
            # This gives more weight to points far from the center
            norm_prob = stats.norm.cdf(m_dist)
            t_prob = stats.t.cdf(m_dist, df=df_freedom)
            
            # Adjust original probability
            adjustment = (t_prob / norm_prob) if norm_prob > 0 else 1.0
            adjusted_probs[j, i] = probs[j, i] * adjustment
    
    # Normalize adjusted probabilities
    row_sums = adjusted_probs.sum(axis=1)
    adjusted_probs = adjusted_probs / row_sums[:, np.newaxis]
    
    # Get the most likely cluster for each point
    adjusted_labels = np.argmax(adjusted_probs, axis=1)
    
    computation_time = time.time() - start_time
    
    result = ClusteringResult(
        name='Mixture of Student\'s t-distributions',
        labels=adjusted_labels,
        probs=adjusted_probs,
        means=gmm.means_,
        covariances=gmm.covariances_,
        model=gmm,
        computation_time=computation_time
    )
    
    # Sort clusters from left to right
    result = sort_clusters_by_position(result, df, feat1)
    
    return result

def cluster_dpmm(df, feat1, feat2, n_clusters=3, random_state=42):
    """
    Perform clustering using Dirichlet Process Mixture Model.
    Automatically determines the optimal number of clusters.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    n_clusters : int
        Maximum number of clusters to consider
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    ClusteringResult
        Standardized clustering result object
    """
    from sklearn.mixture import BayesianGaussianMixture
    import time
    
    start_time = time.time()
    
    # Extract features
    X = np.column_stack((df[feat1].values, df[feat2].values))
    
    # Use Bayesian Gaussian Mixture as an approximation to DPMM
    # Start with more components than needed, model will determine actual number
    max_components = max(n_clusters * 2, 10)  # Use more potential components
    
    model = BayesianGaussianMixture(
        n_components=max_components,
        weight_concentration_prior=0.1,  # Lower value encourages sparsity
        covariance_type='full',
        random_state=random_state,
        max_iter=200,
        n_init=3
    )
    model.fit(X)
    
    # Get cluster probabilities
    probs = model.predict_proba(X)
    
    # Get hard cluster assignments
    labels = model.predict(X)
    
    # Count actual components with non-zero weights
    active_components = np.sum(model.weights_ > 0.01)
    
    # Trim components that are essentially unused
    # This step removes empty clusters
    significant_indices = np.where(model.weights_ > 0.01)[0]
    
    # Remap labels to only include significant components
    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(significant_indices)}
    new_labels = np.array([label_map.get(label, 0) for label in labels])
    
    # Extract only the significant means and covariances
    significant_means = model.means_[significant_indices]
    significant_covs = model.covariances_[significant_indices]
    significant_weights = model.weights_[significant_indices]
    
    # Create trimmed probability matrix
    new_probs = np.zeros((len(X), len(significant_indices)))
    for i, idx in enumerate(significant_indices):
        new_probs[:, i] = probs[:, idx]
    
    computation_time = time.time() - start_time
    
    result = ClusteringResult(
        name='Dirichlet Process Mixture Model',
        labels=new_labels,
        probs=new_probs,
        means=significant_means,
        covariances=significant_covs,
        model=model,
        computation_time=computation_time,
        weights=significant_weights,
        active_components=active_components,
        additional_info={'max_components': max_components, 'significant_components': len(significant_indices)}
    )
    
    # Sort clusters from left to right
    result = sort_clusters_by_position(result, df, feat1)
    
    return result

def cluster_kde_based(df, feat1, feat2, n_clusters=3, bandwidth=None, random_state=42):
    """
    Perform clustering using Kernel Density-Based approach.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    n_clusters : int
        Number of clusters
    bandwidth : float or None
        Kernel bandwidth for KDE, if None it's estimated
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    ClusteringResult
        Standardized clustering result object
    """
    from sklearn.neighbors import KernelDensity
    from sklearn.cluster import KMeans
    import time
    
    start_time = time.time()
    
    # Extract features
    X = np.column_stack((df[feat1].values, df[feat2].values))
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Estimate bandwidth if not provided
    if bandwidth is None:
        # Scott's rule of thumb
        bandwidth = X_scaled.shape[0] ** (-1 / (X_scaled.shape[1] + 4))
    
    # Initialize KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(X_scaled)
    
    # Get density estimates for each point
    log_density = kde.score_samples(X_scaled)
    
    # Use KMeans to find cluster centers in the density space
    X_with_density = np.column_stack((X_scaled, log_density))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X_with_density)
    
    # Get cluster centers
    centers = kmeans.cluster_centers_[:, :2]  # Remove density dimension
    
    # Calculate distances to each center
    distances = np.zeros((len(X_scaled), n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.linalg.norm(X_scaled - centers[i], axis=1)
    
    # Convert distances to probabilities using softmax
    def softmax(x):
        e_x = np.exp(-x)  # Negative for inverting distance
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    probs = softmax(distances)
    
    # Get hard cluster assignments
    labels = np.argmax(probs, axis=1)
    
    # Transform centers back to original scale
    centers_original = scaler.inverse_transform(centers)
    
    # Create fake covariances (identity matrices) for visualization consistency
    fake_covariances = np.array([np.eye(2) for _ in range(n_clusters)])
    
    computation_time = time.time() - start_time
    
    result = ClusteringResult(
        name='Kernel Density-Based Clustering',
        labels=labels,
        probs=probs,
        means=centers_original,
        covariances=fake_covariances,
        model={'kde': kde, 'kmeans': kmeans},
        computation_time=computation_time,
        additional_info={'bandwidth': bandwidth, 'scaler': scaler}
    )
    
    # Sort clusters from left to right
    result = sort_clusters_by_position(result, df, feat1)
    
    return result

def cluster_fuzzy_cmeans(df, feat1, feat2, n_clusters=3, m=2, random_state=42):
    """
    Perform clustering using Fuzzy C-Means.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    n_clusters : int
        Number of clusters
    m : float
        Fuzziness parameter (m > 1, higher values = fuzzier clusters)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    ClusteringResult
        Standardized clustering result object
    """
    import time
    import numpy.random as rnd
    
    start_time = time.time()
    
    # Extract features
    X = np.column_stack((df[feat1].values, df[feat2].values))
    
    # Number of data points and dimensions
    n_samples, n_features = X.shape
    
    # Set random seed
    rnd.seed(random_state)
    
    # Initialize cluster centers using random points from the dataset
    idx = rnd.choice(n_samples, n_clusters, replace=False)
    centers = X[idx].copy()
    
    # Initialize membership matrix U
    U = np.zeros((n_samples, n_clusters))
    
    # Maximum number of iterations
    max_iter = 100
    
    # Convergence threshold
    epsilon = 1e-4
    
    # Optimization loop
    for iteration in range(max_iter):
        # Calculate distances to cluster centers
        distances = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            distances[:, i] = np.linalg.norm(X - centers[i], axis=1)**2
        
        # Update membership matrix
        U_old = U.copy()
        
        # Handle points exactly at cluster centers
        zero_distances = (distances == 0)
        if np.any(zero_distances):
            for i in range(n_samples):
                if np.any(zero_distances[i]):
                    U[i] = zero_distances[i] / np.sum(zero_distances[i])
                else:
                    for j in range(n_clusters):
                        U[i, j] = 1 / np.sum((distances[i, j] / distances[i]) ** (1 / (m - 1)))
        else:
            for i in range(n_samples):
                for j in range(n_clusters):
                    U[i, j] = 1 / np.sum((distances[i, j] / distances[i]) ** (1 / (m - 1)))
        
        # Update cluster centers
        for j in range(n_clusters):
            U_j_m = U[:, j] ** m
            if np.sum(U_j_m) > 0:
                centers[j] = np.sum(U_j_m.reshape(-1, 1) * X, axis=0) / np.sum(U_j_m)
        
        # Check for convergence
        if np.linalg.norm(U - U_old) < epsilon:
            break
    
    # Get hard cluster assignments
    labels = np.argmax(U, axis=1)
    
    # Calculate pseudo-covariances for visualization
    covariances = np.zeros((n_clusters, 2, 2))
    for i in range(n_clusters):
        cluster_mask = labels == i
        if np.sum(cluster_mask) > 1:
            cluster_points = X[cluster_mask]
            covariances[i] = np.cov(cluster_points.T)
        else:
            covariances[i] = np.eye(2)
    
    computation_time = time.time() - start_time
    
    result = ClusteringResult(
        name='Fuzzy C-Means',
        labels=labels,
        probs=U,
        means=centers,
        covariances=covariances,
        model={'centers': centers, 'memberships': U},
        computation_time=computation_time,
        additional_info={'fuzziness': m, 'iterations': iteration + 1}
    )
    
    # Sort clusters from left to right
    result = sort_clusters_by_position(result, df, feat1)
    
    return result

def cluster_spectral_prob(df, feat1, feat2, n_clusters=3, sigma=1.0, random_state=42):
    """
    Spectral clustering with probabilistic assignments.
    
    This method applies spectral clustering to the data and then uses the distance to cluster 
    centers to compute soft assignments.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature (x-axis)
    feat2 : str
        Column name for second feature (y-axis)
    n_clusters : int
        Number of clusters to form
    sigma : float
        Width of the Gaussian kernel
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    ClusteringResult
        Object containing cluster labels, probabilities, etc.
    """
    start_time = time.time()
    
    # Extract data
    X = np.column_stack((df[feat1].values, df[feat2].values))
    
    try:
        # Compute similarity matrix (RBF kernel)
        similarity = np.exp(-pdist(X, 'sqeuclidean') / (2 * sigma**2))
        similarity_matrix = squareform(similarity)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Compute normalized Laplacian
        D = np.diag(np.sum(similarity_matrix, axis=1))
        L = D - similarity_matrix
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.sum(similarity_matrix, axis=1)))
        L_norm = D_sqrt_inv @ L @ D_sqrt_inv
        
        # Compute eigenvectors of normalized Laplacian
        eigenvalues, eigenvectors = eigsh(L_norm, k=min(n_clusters + 1, X.shape[0] - 1), which='SM')
        
        # Use n_clusters eigenvectors corresponding to the smallest eigenvalues
        # (excluding the first one)
        embedding = eigenvectors[:, 1:n_clusters+1]
        
        # Check dimensionality for PCA
        n_components = min(2, embedding.shape[1])
        
        # Only apply PCA if we have multiple dimensions
        if n_components > 1 and embedding.shape[1] > 1:
            # Use PCA to reduce to 2D if needed
            pca = PCA(n_components=n_components).fit(embedding)
            embedding_2d = pca.transform(embedding)
        else:
            # If only one dimension, duplicate it to create a 2D representation
            embedding_2d = np.column_stack((embedding, np.zeros_like(embedding)))
        
        # Apply k-means to the embedding
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embedding)
        
        # Compute distances to centroids
        centers = kmeans.cluster_centers_
        distances = np.zeros((X.shape[0], n_clusters))
        
        for i in range(n_clusters):
            # Compute Euclidean distance in the embedding space
            distances[:, i] = np.linalg.norm(embedding - centers[i], axis=1)
        
        # Convert distances to probabilities using softmax
        def softmax(x):
            # Invert distances (smaller distance = higher probability)
            x_inv = -x
            # Shift for numerical stability
            x_inv = x_inv - np.max(x_inv, axis=1, keepdims=True)
            # Calculate softmax
            exp_x = np.exp(x_inv)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        # Compute probabilities
        probs = softmax(distances)
        
        # Compute cluster means and covariances
        means = np.zeros((n_clusters, 2))
        covariances = np.zeros((n_clusters, 2, 2))
        
        for i in range(n_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                means[i] = np.mean(X[mask], axis=0)
                if np.sum(mask) > 1:  # Need at least 2 points to compute covariance
                    covariances[i] = np.cov(X[mask].T)
                else:
                    covariances[i] = np.eye(2)  # Identity matrix as fallback
        
        # Create result object
        result = ClusteringResult(
            name="Spectral Prob",
            labels=labels,
            probs=probs,
            means=means,
            covariances=covariances,
            model=kmeans,
            computation_time=time.time() - start_time
        )
        
        # Sort clusters from left to right
        result = sort_clusters_by_position(result, df, feat1)
        
        return result
        
    except Exception as e:
        print(f"  Spectral Prob failed: {str(e)}")
        # Create a simple single-cluster fallback
        n = len(df)
        labels = np.zeros(n, dtype=int)
        probs = np.ones((n, 1))
        means = np.array([np.mean(X, axis=0)])
        covs = np.array([np.cov(X.T)]) if n > 1 else np.array([np.eye(2)])
        
        return ClusteringResult(
            name="Spectral Prob (Fallback)",
            labels=labels,
            probs=probs,
            means=means,
            covariances=covs,
            computation_time=time.time() - start_time
        )

def cluster_gmm_wrapper(df, feat1, feat2, n_clusters=3, random_state=42):
    """
    Wrapper for GMM clustering to provide the standard ClusteringResult format.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    ClusteringResult
        Standardized clustering result object
    """
    import time
    from sklearn.mixture import GaussianMixture
    
    start_time = time.time()
    
    # Extract features
    X = np.column_stack((df[feat1].values, df[feat2].values))
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gmm.fit(X)
    
    # Get probabilities and labels
    probs = gmm.predict_proba(X)
    labels = gmm.predict(X)
    
    computation_time = time.time() - start_time
    
    result = ClusteringResult(
        name='Gaussian Mixture Model',
        labels=labels,
        probs=probs,
        means=gmm.means_,
        covariances=gmm.covariances_,
        model=gmm,
        computation_time=computation_time,
        weights=gmm.weights_
    )
    
    # Sort clusters from left to right
    result = sort_clusters_by_position(result, df, feat1)
    
    return result

def cluster_xmeans(df, feat1, feat2, max_clusters=10, min_clusters=2, random_state=42):
    """
    Perform clustering using X-means, which automatically determines the optimal 
    number of clusters using BIC (Bayesian Information Criterion).
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    max_clusters : int
        Maximum number of clusters to consider
    min_clusters : int
        Minimum number of clusters to consider
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    ClusteringResult
        Standardized clustering result object
    """
    from sklearn.cluster import KMeans
    import time
    import numpy as np
    from scipy.spatial.distance import cdist
    
    start_time = time.time()
    
    # Extract features
    X = np.column_stack((df[feat1].values, df[feat2].values))
    
    # Scale the data for better clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_samples, n_features = X_scaled.shape
    
    # Function to calculate BIC score
    def compute_bic(kmeans, X):
        """
        Computes the BIC score for a given k-means model.
        Lower BIC values indicate better models.
        """
        # Get cluster centers and labels
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        n_clusters = kmeans.n_clusters
        n_samples, n_features = X.shape
        
        # Calculate log-likelihood
        dist = np.min(cdist(X, centers, 'euclidean'), axis=1)
        log_likelihood = -0.5 * np.sum(dist**2)
        
        # Calculate BIC: -2 * log-likelihood + k * log(n)
        # where k is the number of parameters: n_clusters * (n_features + 1)
        # n_features for each center coordinate + 1 for the variance
        k = n_clusters * (n_features + 1)
        bic = -2 * log_likelihood + k * np.log(n_samples)
        
        return bic
    
    # BIC scores for different k values
    bic_scores = []
    models = []
    
    # Try different k values
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X_scaled)
        bic = compute_bic(kmeans, X_scaled)
        bic_scores.append(bic)
        models.append(kmeans)
    
    # Find the best model (lowest BIC)
    best_idx = np.argmin(bic_scores)
    best_k = best_idx + min_clusters
    best_model = models[best_idx]
    
    # Get hard labels
    labels = best_model.labels_
    centers = best_model.cluster_centers_
    
    # Convert centers back to original scale
    centers_original = scaler.inverse_transform(centers)
    
    # Calculate distances to each center for soft assignments
    distances = cdist(X_scaled, centers, 'euclidean')
    
    # Convert distances to probabilities using softmax
    def softmax(x):
        e_x = np.exp(-x)  # Negative for inverting distance
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    probs = softmax(distances)
    
    # Create fake covariances based on within-cluster variances
    covariances = []
    for i in range(best_k):
        cluster_points = X_scaled[labels == i]
        if len(cluster_points) > 1:
            # Calculate empirical covariance matrix
            cov = np.cov(cluster_points.T)
            if cov.shape == ():  # Handle 1D case
                cov = np.array([[cov]])
            # Transform back to original scale
            scale_factors = np.diag(scaler.scale_)
            cov_original = np.dot(np.dot(scale_factors, cov), scale_factors)
            covariances.append(cov_original)
        else:
            # Use identity matrix if only one point in cluster
            covariances.append(np.eye(n_features))
    
    covariances = np.array(covariances)
    
    computation_time = time.time() - start_time
    
    result = ClusteringResult(
        name='X-means (BIC-optimized)',
        labels=labels,
        probs=probs,
        means=centers_original,
        covariances=covariances,
        model=best_model,
        computation_time=computation_time,
        additional_info={
            'bic_scores': bic_scores,
            'optimal_clusters': best_k,
            'scaler': scaler
        }
    )
    
    # Sort clusters from left to right
    result = sort_clusters_by_position(result, df, feat1)
    
    return result

def cluster_hdbscan(df, feat1, feat2, min_cluster_size=5, min_samples=None, random_state=42):
    """
    Perform clustering using HDBSCAN (Hierarchical Density-Based Spatial Clustering 
    of Applications with Noise), which automatically determines the number of clusters.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    min_cluster_size : int
        Minimum size of clusters
    min_samples : int or None
        Number of samples in a neighborhood for a point to be a core point
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    ClusteringResult
        Standardized clustering result object
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("HDBSCAN package is required. Install it using: pip install hdbscan")
    
    import time
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    start_time = time.time()
    
    # Extract features
    X = np.column_stack((df[feat1].values, df[feat2].values))
    
    # Scale the data for better clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set default min_samples if not provided
    if min_samples is None:
        min_samples = min_cluster_size
    
    # Create and fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,  # Enable soft clustering
        gen_min_span_tree=True,  # Required for membership vectors
        core_dist_n_jobs=-1  # Use all processors
    )
    clusterer.fit(X_scaled)
    
    # Get hard labels (noise points get label -1)
    labels = clusterer.labels_.copy()
    
    # Get unique non-noise cluster labels
    unique_clusters = np.unique(labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    
    # If no clusters found, make all points one cluster
    if len(unique_clusters) == 0:
        labels = np.zeros(len(X), dtype=int)
        unique_clusters = np.array([0])
    
    # Remap labels to be consecutive integers starting from 0
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
    # Remap -1 (noise) to the highest cluster number or to 0 if no clusters
    if -1 in labels:
        if len(unique_clusters) > 0:
            label_map[-1] = len(unique_clusters)
        else:
            label_map[-1] = 0
            
    # Apply the mapping
    new_labels = np.array([label_map.get(label, 0) for label in labels])
    
    # Get probabilities using membership_vector_ if available, otherwise approximate
    n_clusters = len(np.unique(new_labels))
    
    # Try to get soft assignments from HDBSCAN
    try:
        # Get soft cluster memberships
        all_probabilities = hdbscan.all_points_membership_vectors(clusterer)
        
        # Adjust probability matrix to match our remapped labels
        probs = np.zeros((len(X), n_clusters))
        for old_label, new_label in label_map.items():
            if old_label >= 0:  # Skip noise label for now
                probs[:, new_label] = all_probabilities[:, old_label]
                
        # Handle noise points (distribute their probability across clusters)
        if -1 in labels:
            noise_idx = labels == -1
            if np.any(noise_idx):
                # For noise points, assign uniform probability across all clusters
                noise_probs = np.ones((np.sum(noise_idx), n_clusters)) / n_clusters
                probs[noise_idx] = noise_probs
    except:
        # Fallback: use one-hot encoding for hard assignments
        probs = np.zeros((len(X), n_clusters))
        for i, label in enumerate(new_labels):
            probs[i, label] = 1.0
    
    # Calculate cluster centers and covariances
    means = np.zeros((n_clusters, 2))
    covariances = np.zeros((n_clusters, 2, 2))
    
    for i in range(n_clusters):
        mask = new_labels == i
        if np.sum(mask) > 1:
            means[i] = np.mean(X[mask], axis=0)
            covariances[i] = np.cov(X[mask].T)
        elif np.sum(mask) == 1:
            means[i] = X[mask][0]
            covariances[i] = np.eye(2)  # Use identity for single point clusters
        else:
            # Fallback for empty clusters (shouldn't happen due to the remapping)
            means[i] = np.mean(X, axis=0)
            covariances[i] = np.eye(2)
    
    computation_time = time.time() - start_time
    
    result = ClusteringResult(
        name='HDBSCAN',
        labels=new_labels,
        probs=probs,
        means=means,
        covariances=covariances,
        model=clusterer,
        computation_time=computation_time,
        additional_info={
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'original_labels': labels,  # Keep original labels for reference
            'noise_count': np.sum(labels == -1) if -1 in labels else 0
        }
    )
    
    # Sort clusters from left to right
    result = sort_clusters_by_position(result, df, feat1)
    
    return result

def cluster_affinity_propagation(df, feat1, feat2, damping=0.9, preference=None, random_state=42):
    """
    Perform clustering using Affinity Propagation, which automatically determines 
    the number of clusters based on message passing between data points.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    damping : float
        Damping factor for message passing (0.5 < damping <= 1.0)
    preference : float or None
        Controls the number of clusters. If None, use the median of input similarities.
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    ClusteringResult
        Standardized clustering result object
    """
    from sklearn.cluster import AffinityPropagation
    from sklearn.metrics import pairwise_distances
    import time
    import numpy as np
    
    start_time = time.time()
    
    # Extract features
    X = np.column_stack((df[feat1].values, df[feat2].values))
    
    # Scale the data for better clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # If preference is not provided, try different values to get sensible number of clusters
    if preference is None:
        # Calculate pairwise negative euclidean distances
        S = -pairwise_distances(X_scaled, metric='euclidean')
        preference = np.median(S)  # Default in sklearn
    
    # Create and fit Affinity Propagation
    ap = AffinityPropagation(
        damping=damping,
        preference=preference,
        random_state=random_state,
        max_iter=1000,
        convergence_iter=50
    )
    
    try:
        ap.fit(X_scaled)
        labels = ap.labels_
        cluster_centers_indices = ap.cluster_centers_indices_
        n_clusters = len(cluster_centers_indices)
        
        # If too many clusters (>20), try to reduce by lowering preference
        if n_clusters > 20:
            preference_lower = preference - abs(preference) * 0.5
            ap = AffinityPropagation(
                damping=damping,
                preference=preference_lower,
                random_state=random_state,
                max_iter=1000,
                convergence_iter=50
            )
            ap.fit(X_scaled)
            labels = ap.labels_
            cluster_centers_indices = ap.cluster_centers_indices_
            n_clusters = len(cluster_centers_indices)
            
            # If still too many, try one more time
            if n_clusters > 10:
                preference_much_lower = preference - abs(preference) * 0.9
                ap = AffinityPropagation(
                    damping=damping,
                    preference=preference_much_lower,
                    random_state=random_state,
                    max_iter=1000,
                    convergence_iter=50
                )
                ap.fit(X_scaled)
                labels = ap.labels_
                cluster_centers_indices = ap.cluster_centers_indices_
                n_clusters = len(cluster_centers_indices)
    except Exception as e:
        # If convergence fails, fall back to a simpler KMeans
        print(f"Affinity Propagation failed: {str(e)}")
        print("Falling back to KMeans with 3 clusters")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=random_state)
        kmeans.fit(X_scaled)
        labels = kmeans.labels_
        n_clusters = 3
        cluster_centers_indices = None
    
    # Extract cluster centers
    if cluster_centers_indices is not None:
        centers_scaled = X_scaled[cluster_centers_indices]
    else:
        # Compute centers manually if we used the fallback
        centers_scaled = np.array([X_scaled[labels == i].mean(axis=0) for i in range(n_clusters)])
    
    # Transform back to original scale
    centers_original = scaler.inverse_transform(centers_scaled)
    
    # Compute soft assignments based on negative distance to each center
    distances = pairwise_distances(X_scaled, centers_scaled)
    
    # Convert distances to probabilities using softmax
    def softmax(x):
        e_x = np.exp(-x)  # Negative for inverting distance
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    probs = softmax(distances)
    
    # Calculate covariances for each cluster
    covariances = np.zeros((n_clusters, 2, 2))
    for i in range(n_clusters):
        mask = labels == i
        if np.sum(mask) > 1:
            covariances[i] = np.cov(X[mask].T)
        else:
            covariances[i] = np.eye(2)  # Identity matrix for singleton clusters
    
    computation_time = time.time() - start_time
    
    result = ClusteringResult(
        name='Affinity Propagation',
        labels=labels,
        probs=probs,
        means=centers_original,
        covariances=covariances,
        model=ap,
        computation_time=computation_time,
        additional_info={
            'damping': damping,
            'preference': preference,
            'n_clusters': n_clusters,
            'cluster_centers_indices': cluster_centers_indices
        }
    )
    
    # Sort clusters from left to right
    result = sort_clusters_by_position(result, df, feat1)
    
    return result

def cluster_variational_dpmm(df, feat1, feat2, n_components=10, alpha=1.0, max_iter=100, random_state=42):
    """
    Perform clustering using Variational Inference DPMM, a faster approximation of DPMM.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    n_components : int
        Maximum number of components (truncation level)
    alpha : float
        Concentration parameter for the Dirichlet Process
    max_iter : int
        Maximum number of iterations
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    ClusteringResult
        Standardized clustering result object
    """
    import time
    from sklearn.mixture import BayesianGaussianMixture
    
    start_time = time.time()
    
    # Extract features
    X = np.column_stack((df[feat1].values, df[feat2].values))
    
    # Use sklearn's BayesianGaussianMixture with variational inference
    model = BayesianGaussianMixture(
        n_components=n_components,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=alpha,
        covariance_type='full',
        max_iter=max_iter,
        n_init=3,
        random_state=random_state
    )
    
    model.fit(X)
    
    # Get cluster probabilities
    probs = model.predict_proba(X)
    
    # Get hard cluster assignments
    labels = model.predict(X)
    
    # Count actual components with non-zero weights
    active_components = np.sum(model.weights_ > 0.01)
    
    # Trim components that are essentially unused
    significant_indices = np.where(model.weights_ > 0.01)[0]
    
    # If no significant components found, keep at least one
    if len(significant_indices) == 0:
        significant_indices = np.array([0])
    
    # Create a mapping from original indices to new indices
    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(significant_indices)}
    
    # Remap labels to only include significant components
    new_labels = np.array([label_map.get(label, 0) for label in labels])
    
    # Extract only the significant means and covariances
    significant_means = model.means_[significant_indices]
    significant_covs = model.covariances_[significant_indices]
    significant_weights = model.weights_[significant_indices]
    
    # Create trimmed probability matrix
    new_probs = np.zeros((len(X), len(significant_indices)))
    for i, idx in enumerate(significant_indices):
        new_probs[:, i] = probs[:, idx]
    
    # Normalize probabilities to sum to 1
    row_sums = new_probs.sum(axis=1, keepdims=True)
    new_probs = new_probs / row_sums
    
    computation_time = time.time() - start_time
    
    result = ClusteringResult(
        name='Variational DPMM',
        labels=new_labels,
        probs=new_probs,
        means=significant_means,
        covariances=significant_covs,
        model=model,
        computation_time=computation_time,
        weights=significant_weights,
        active_components=active_components,
        additional_info={
            'alpha': alpha,
            'n_components': n_components,
            'significant_components': len(significant_indices)
        }
    )
    
    # Sort clusters from left to right
    result = sort_clusters_by_position(result, df, feat1)
    
    return result

def evaluate_cluster_quality(df, feat2, labels, min_effect_size=0.3):
    """
    Evaluate the quality of clustering based on the separation of clusters 
    along the y-axis (feat2).
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat2 : str
        Column name for y-axis feature
    labels : array-like
        Cluster assignments
    min_effect_size : float
        Minimum Cohen's d effect size to consider clusters as meaningfully different
        
    Returns:
    --------
    dict
        Dictionary containing quality metrics
    """
    # Get unique clusters
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    # If only one cluster, return basic metrics
    if n_clusters <= 1:
        return {
            'n_clusters': n_clusters,
            'meaningful_clusters': False,
            'effect_sizes': [],
            'cluster_means': [],
            'cluster_stds': [],
            'cluster_sizes': []
        }
    
    # Collect statistics for each cluster
    cluster_stats = []
    for label in unique_clusters:
        values = df.loc[labels == label, feat2].values
        if len(values) > 0:
            cluster_stats.append({
                'label': label,
                'mean': np.mean(values),
                'std': np.std(values) if len(values) > 1 else 0,
                'size': len(values)
            })
    
    # Calculate pairwise effect sizes (Cohen's d)
    effect_sizes = []
    for i, stats1 in enumerate(cluster_stats):
        for j, stats2 in enumerate(cluster_stats):
            if i < j:  # Only compute for unique pairs
                # Get stats
                mean1, std1, n1 = stats1['mean'], stats1['std'], stats1['size']
                mean2, std2, n2 = stats2['mean'], stats2['std'], stats2['size']
                
                # Calculate pooled standard deviation
                pooled_std = np.sqrt(
                    ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / 
                    (n1 + n2 - 2)
                ) if n1 > 1 and n2 > 1 else max(std1, std2)
                
                # Calculate Cohen's d (handle division by zero)
                if pooled_std > 0:
                    d = abs(mean1 - mean2) / pooled_std
                else:
                    # If std is 0, check if means are different
                    d = float('inf') if mean1 != mean2 else 0
                
                effect_sizes.append({
                    'cluster1': stats1['label'],
                    'cluster2': stats2['label'],
                    'effect_size': d
                })
    
    # Check if all pairs have a meaningful effect size
    all_meaningful = all(e['effect_size'] >= min_effect_size for e in effect_sizes)
    
    # Return results
    return {
        'n_clusters': n_clusters,
        'meaningful_clusters': all_meaningful,
        'effect_sizes': effect_sizes,
        'cluster_means': [stats['mean'] for stats in cluster_stats],
        'cluster_stds': [stats['std'] for stats in cluster_stats],
        'cluster_sizes': [stats['size'] for stats in cluster_stats]
    }

def find_optimal_k(df, feat1, feat2, clustering_func, param_name='n_clusters', min_k=1, max_k=3, 
                  min_effect_size=0.3, random_state=42, **kwargs):
    """
    Find the optimal number of clusters by trying different values of k
    and evaluating cluster quality.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    clustering_func : function
        Clustering function to call
    param_name : str
        Name of the parameter that controls the number of clusters
    min_k : int
        Minimum number of clusters to try
    max_k : int
        Maximum number of clusters to try
    min_effect_size : float
        Minimum effect size to consider clusters as meaningfully different
    random_state : int
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters to pass to the clustering function
        
    Returns:
    --------
    ClusteringResult
        Best clustering result
    """
    results = []
    evaluations = []
    
    print(f"\n  === Testing {param_name} values from {min_k} to {max_k} ===")
    print(f"  Using minimum effect size threshold: {min_effect_size}")
    
    # Try different values of k
    for k in range(min_k, max_k + 1):
        # Set clustering parameters
        params = kwargs.copy()
        params[param_name] = k
        params['random_state'] = random_state
        
        # Run clustering
        try:
            result = clustering_func(df, feat1, feat2, **params)
            
            # Evaluate cluster quality
            eval_result = evaluate_cluster_quality(df, feat2, result.labels, min_effect_size)
            
            # Store results
            results.append(result)
            evaluations.append(eval_result)
            
            print(f"\n  *** Tried {param_name}={k}, found {eval_result['n_clusters']} clusters ***")
            
            # Display detailed cluster statistics
            unique_labels = np.unique(result.labels)
            print(f"  Cluster statistics for {feat2}:")
            for i, label in enumerate(unique_labels):
                mask = result.labels == label
                values = df.loc[mask, feat2].values
                size = len(values)
                
                if size > 0:
                    mean = np.mean(values)
                    std = np.std(values) if size > 1 else 0
                    min_val = np.min(values)
                    max_val = np.max(values)
                    x_min = df.loc[mask, feat1].min()
                    x_max = df.loc[mask, feat1].max()
                    
                    print(f"  Cluster {label}: size={size}, mean={mean:.2f}, std={std:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
                    print(f"           {feat1} range=[{x_min:.2f}, {x_max:.2f}]")
            
            # Display effect sizes between clusters
            if eval_result['effect_sizes']:
                print(f"\n  Pairwise effect sizes (Cohen's d):")
                for e in eval_result['effect_sizes']:
                    c1, c2, es = e['cluster1'], e['cluster2'], e['effect_size']
                    judgment = ""
                    if es < 0.2:
                        judgment = "negligible difference"
                    elif es < 0.5:
                        judgment = "small difference"
                    elif es < 0.8:
                        judgment = "medium difference"
                    else:
                        judgment = "large difference"
                        
                    significance = "SIGNIFICANT" if es >= min_effect_size else "NOT SIGNIFICANT"
                    print(f"  Clusters {c1}-{c2}: d={es:.2f} ({judgment}) - {significance}")
            
            # Overall assessment
            if eval_result['meaningful_clusters']:
                print(f"\n  Assessment: All clusters show meaningful differences (effect size ≥ {min_effect_size})")
            else:
                print(f"\n  Assessment: Not all clusters show meaningful differences (effect size ≥ {min_effect_size})")
                # Find which pairs don't meet the threshold
                if eval_result['effect_sizes']:
                    problematic_pairs = [(e['cluster1'], e['cluster2']) for e in eval_result['effect_sizes'] if e['effect_size'] < min_effect_size]
                    if problematic_pairs:
                        pairs_str = ", ".join([f"{c1}-{c2}" for c1, c2 in problematic_pairs])
                        print(f"  Problem: Clusters {pairs_str} are too similar and should be merged")
            
        except Exception as e:
            print(f"  Error with {param_name}={k}: {str(e)}")
    
    # Find the best result (highest k with meaningful clusters)
    best_idx = None
    best_k = 0
    
    print("\n  === Decision Process ===")
    
    for i, eval_result in enumerate(evaluations):
        k = i + min_k
        if eval_result['meaningful_clusters']:
            print(f"  k={k}: All clusters are meaningfully different")
            if k > best_k:
                best_idx = i
                best_k = k
                print(f"  --> This is better than previous best k={best_k if best_k > 0 else 'None'}")
        else:
            print(f"  k={k}: Not all clusters are meaningfully different")
    
    # If no meaningful clusters were found, return the result with k=1
    if best_idx is None:
        print("\n  No meaningful multi-cluster solution found, using k=1")
        print("  Reason: None of the tested values of k produced clusters with sufficient separation")
        if results:
            return results[0]  # Return k=1 result if available
        else:
            # If all clustering attempts failed, try once more with k=1
            print("  All clustering attempts failed, trying once more with k=1")
            params = kwargs.copy()
            params[param_name] = 1
            params['random_state'] = random_state
            return clustering_func(df, feat1, feat2, **params)
    
    print(f"\n  Selected optimal {param_name}={best_k} with meaningful cluster separation")
    return results[best_idx]

def cluster_with_merging(df, feat1, feat2, clustering_func, param_name='n_clusters', effect_size_threshold=0.3, random_state=42, **kwargs):
    """
    Generate 3 clusters and then merge those that aren't significantly different.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    clustering_func : function
        Clustering function to call
    param_name : str
        Name of the parameter that controls the number of clusters
    effect_size_threshold : float
        Minimum Cohen's d to consider clusters as significantly different
    random_state : int
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters to pass to the clustering function
        
    Returns:
    --------
    ClusteringResult
        Clustering result with merged clusters
    """
    print(f"\n  === Generate 3 clusters and merge similar ones ===")
    print(f"  Using effect size threshold: {effect_size_threshold}")
    
    # Set clustering parameters
    params = kwargs.copy()
    params[param_name] = 3  # Start with 3 clusters
    params['random_state'] = random_state
    
    # Run initial clustering
    try:
        result = clustering_func(df, feat1, feat2, **params)
        
        print(f"\n  Initial clustering with {param_name}=3")
        
        # Get unique labels
        unique_labels = np.unique(result.labels)
        n_clusters = len(unique_labels)
        
        # If we already have fewer than 3 clusters, just return the result
        if n_clusters < 3:
            print(f"  Only {n_clusters} clusters found initially, no merging needed")
            return result
        
        # Display detailed cluster statistics
        print(f"  Initial cluster statistics for {feat2}:")
        for i, label in enumerate(unique_labels):
            mask = result.labels == label
            values = df.loc[mask, feat2].values
            size = len(values)
            
            if size > 0:
                mean = np.mean(values)
                std = np.std(values) if size > 1 else 0
                min_val = np.min(values)
                max_val = np.max(values)
                x_min = df.loc[mask, feat1].min()
                x_max = df.loc[mask, feat1].max()
                
                print(f"  Cluster {label}: size={size}, mean={mean:.2f}, std={std:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
                print(f"           {feat1} range=[{x_min:.2f}, {x_max:.2f}]")
        
        # Compute effect sizes between all pairs of clusters
        print(f"\n  Calculating pairwise effect sizes:")
        effect_sizes = []
        merge_candidates = []
        
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                label_i, label_j = unique_labels[i], unique_labels[j]
                mask_i = result.labels == label_i
                mask_j = result.labels == label_j
                
                values_i = df.loc[mask_i, feat2].values
                values_j = df.loc[mask_j, feat2].values
                
                # Calculate Cohen's d
                mean_i, std_i, n_i = np.mean(values_i), np.std(values_i), len(values_i)
                mean_j, std_j, n_j = np.mean(values_j), np.std(values_j), len(values_j)
                
                # Pooled standard deviation
                pooled_std = np.sqrt(
                    ((n_i - 1) * std_i**2 + (n_j - 1) * std_j**2) / 
                    (n_i + n_j - 2)
                ) if n_i > 1 and n_j > 1 else max(std_i, std_j)
                
                # Handle division by zero
                if pooled_std > 0:
                    d = abs(mean_i - mean_j) / pooled_std
                else:
                    d = float('inf') if mean_i != mean_j else 0
                
                judgment = ""
                if d < 0.2:
                    judgment = "negligible difference"
                elif d < 0.5:
                    judgment = "small difference"
                elif d < 0.8:
                    judgment = "medium difference"
                else:
                    judgment = "large difference"
                    
                significance = "SIGNIFICANT" if d >= effect_size_threshold else "NOT SIGNIFICANT"
                print(f"  Clusters {label_i}-{label_j}: d={d:.2f} ({judgment}) - {significance}")
                
                effect_sizes.append((label_i, label_j, d))
                
                # If effect size is below threshold, mark for merging
                if d < effect_size_threshold:
                    merge_candidates.append((label_i, label_j))
        
        # If no clusters need to be merged, return original result
        if not merge_candidates:
            print("\n  All clusters are significantly different (no merging needed)")
            return result
        
        # Identify clusters to merge
        print("\n  Determining which clusters to merge:")
        
        # Create a mapping for merged clusters
        # Start with identity mapping
        new_labels_map = {label: label for label in unique_labels}
        
        # Process merge candidates
        for c1, c2 in merge_candidates:
            print(f"  Merging clusters {c1} and {c2} (effect size < {effect_size_threshold})")
            # Always merge into the lower-numbered cluster
            target = min(c1, c2)
            # Update all clusters that map to c1 or c2 to map to target
            for label in unique_labels:
                if new_labels_map[label] == c1 or new_labels_map[label] == c2:
                    new_labels_map[label] = target
        
        # Create new labels array
        new_labels = np.array([new_labels_map[label] for label in result.labels])
        
        # Get the new unique labels after merging
        new_unique_labels = np.unique(new_labels)
        remaining_clusters = len(new_unique_labels)
        print(f"\n  Result: {n_clusters} clusters merged into {remaining_clusters} clusters")
        
        # Create new one-hot encoded probabilities for the merged clusters
        n_samples = len(df)
        new_probs = np.zeros((n_samples, remaining_clusters))
        
        # For each sample, set probability 1.0 for its cluster
        for i, label in enumerate(new_labels):
            # Find the index of this label in the unique labels list
            idx = np.where(new_unique_labels == label)[0][0]
            new_probs[i, idx] = 1.0
        
        # Calculate new statistics for each merged cluster
        new_means = np.zeros((remaining_clusters, 2))
        new_covariances = np.zeros((remaining_clusters, 2, 2))
        
        for i, label in enumerate(new_unique_labels):
            mask = new_labels == label
            if np.sum(mask) > 0:
                # Get points in this cluster
                X_cluster = np.column_stack((df.loc[mask, feat1].values, df.loc[mask, feat2].values))
                # Update mean
                new_means[i] = np.mean(X_cluster, axis=0)
                # Update covariance
                if len(X_cluster) > 1:
                    new_covariances[i] = np.cov(X_cluster.T)
                else:
                    new_covariances[i] = np.eye(2)  # Default for single point
        
        # Create new result with merged clusters
        new_result = ClusteringResult(
            name=f"{result.name} (Merged)",
            labels=new_labels,
            probs=new_probs,
            means=new_means,
            covariances=new_covariances,
            model=result.model,
            computation_time=result.computation_time,
            weights=None,  # Can't easily update weights
            active_components=None,
            additional_info={
                'original_labels': result.labels,
                'merging_threshold': effect_size_threshold,
                'merged_pairs': merge_candidates
            }
        )
        
        # Display final cluster statistics
        print(f"\n  Final cluster statistics after merging:")
        final_unique_labels = np.unique(new_result.labels)
        for i, label in enumerate(final_unique_labels):
            mask = new_result.labels == label
            values = df.loc[mask, feat2].values
            size = len(values)
            
            if size > 0:
                mean = np.mean(values)
                std = np.std(values) if size > 1 else 0
                min_val = np.min(values)
                max_val = np.max(values)
                x_min = df.loc[mask, feat1].min()
                x_max = df.loc[mask, feat1].max()
                
                print(f"  Cluster {label}: size={size}, mean={mean:.2f}, std={std:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
                print(f"           {feat1} range=[{x_min:.2f}, {x_max:.2f}]")
        
        # Return the merged result - no need to sort again
        return new_result
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fall back to default with k=1
        print("\n  Error occurred, falling back to k=1")
        params = kwargs.copy()
        params[param_name] = 1
        params['random_state'] = random_state
        return clustering_func(df, feat1, feat2, **params)

def compare_all_clustering_methods(df, feat1, feat2, n_clusters=3, random_state=42):
    """
    Compare all clustering methods on the given data with visualizations and ANOVA analysis.
    For methods that don't automatically find optimal clusters, start with 3 clusters and
    merge those that aren't significantly different.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for first feature
    feat2 : str
        Column name for second feature
    n_clusters : int
        Default number of clusters (some methods may determine their own optimal number)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing results from all clustering methods
    """
    print(f"\n--- Comparing clustering methods on {feat1} vs {feat2} ---")
    
    # Define which methods automatically determine optimal clusters
    auto_optimal_methods = ['DPMM', 'Variational DPMM', 'X-means', 'HDBSCAN', 'Affinity Prop.']
    
    # List of clustering methods with appropriate parameter mappings
    clustering_methods = [
        {
            'name': 'GMM',
            'function': cluster_gmm_wrapper,
            'params': {'n_clusters': n_clusters},
            'auto_optimal': False
        },
        {
            'name': 'Mixture of t',
            'function': cluster_mixture_t,
            'params': {'n_clusters': n_clusters},
            'auto_optimal': False
        },
        {
            'name': 'DPMM',
            'function': cluster_dpmm,
            'params': {'n_clusters': n_clusters},
            'auto_optimal': True
        },
        {
            'name': 'Variational DPMM',
            'function': cluster_variational_dpmm,
            'params': {'n_components': n_clusters * 2},
            'auto_optimal': True
        },
        {
            'name': 'X-means',
            'function': cluster_xmeans,
            'params': {'max_clusters': n_clusters * 2, 'min_clusters': 2},
            'auto_optimal': True
        },
        {
            'name': 'HDBSCAN',
            'function': cluster_hdbscan,
            'params': {'min_cluster_size': 5},
            'auto_optimal': True
        },
        {
            'name': 'Affinity Prop.',
            'function': cluster_affinity_propagation,
            'params': {'damping': 0.9},
            'auto_optimal': True
        },
        {
            'name': 'KDE-Based',
            'function': cluster_kde_based,
            'params': {'n_clusters': n_clusters},
            'auto_optimal': False
        },
        {
            'name': 'Fuzzy C-Means',
            'function': cluster_fuzzy_cmeans,
            'params': {'n_clusters': n_clusters},
            'auto_optimal': False
        },
        {
            'name': 'Spectral Prob',
            'function': cluster_spectral_prob,
            'params': {'n_clusters': n_clusters},
            'auto_optimal': False
        }
    ]
    
    # Run all methods and collect results
    results = {}
    computation_times = []
    anova_results = {}
    
    for method_info in clustering_methods:
        name = method_info['name']
        func = method_info['function']
        params = method_info['params'].copy()  # Copy to avoid modifying the original
        auto_optimal = method_info['auto_optimal']
        
        print(f"\nRunning {name}...")
        try:
            if auto_optimal:
                # For methods that already determine optimal clusters
                # Always add random_state
                params['random_state'] = random_state
                result = func(df, feat1, feat2, **params)
                print(f"  Method automatically determines optimal clusters")
            else:
                # Use the new approach: Start with 3 clusters and merge if needed
                param_name = next((k for k in params.keys() if 'cluster' in k), 'n_clusters')
                print(f"  Using {param_name}=3 and merging similar clusters...")
                result = cluster_with_merging(
                    df, feat1, feat2, func, 
                    param_name=param_name, 
                    effect_size_threshold=0.3,  # Use 0.3 as requested by the user
                    random_state=random_state, 
                    **{k: v for k, v in params.items() if k != param_name}
                )
            
            # Display number of clusters found
            actual_clusters = len(np.unique(result.labels))
            print(f"  Found {actual_clusters} final clusters")
            
            results[name] = result
            
            # Extract computation time
            computation_times.append((name, result.computation_time))
            print(f"  {name} completed in {result.computation_time:.4f} seconds")
            
            # Visualize results with ANOVA
            print(f"  Visualizing clustering results with ANOVA...")
            fig, anova = visualize_clustering_with_anova(df, feat1, feat2, result)
            plt.show()
            
            # Store ANOVA results
            anova_results[name] = anova
            
            # Print summary of ANOVA results
            if 'f_value' in anova and anova['f_value'] is not None:
                print(f"  ANOVA results for {name}:")
                print(f"    F-value: {anova['f_value']:.4f}")
                print(f"    p-value: {anova['p_value']:.4f}")
                print(f"    Significant difference: {anova['significant']}")
                
                # If significant, show which clusters differ
                if anova['significant'] and 'tukey_results' in anova:
                    print(f"    Tukey HSD post-hoc test:")
                    tukey_lines = anova['tukey_results'].split('\n')
                    # Print only the significant differences
                    for line in tukey_lines:
                        if 'reject' in line and 'True' in line:
                            print(f"      {line}")
            else:
                print(f"  ANOVA analysis not possible for {name}")
                
        except Exception as e:
            print(f"  {name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results[name] = None
    
    # Create a bar chart of computation times at the end
    plt.figure(figsize=(12, 6))
    names = [name for name, _ in computation_times]
    times = [time for _, time in computation_times]
    
    bars = plt.bar(names, times, color='skyblue')
    
    # Add time labels on top of the bars
    for bar, time in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{time:.4f}s",
            ha='center', va='bottom',
            fontsize=10
        )
    
    plt.xlabel('Clustering Method')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Clustering Methods Computation Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Return comprehensive results
    return {
        'clustering_results': results,
        'anova_results': anova_results,
        'computation_times': computation_times
    }

