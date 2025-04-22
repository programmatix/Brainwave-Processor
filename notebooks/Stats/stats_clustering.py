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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
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
            f'μ={mean_val:.2f}\nσ={std_val:.2f}\nn={len(group)}\n{feat1}:\n{x_min:.2f}-{x_max:.2f}',
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
    
    # Remap the probabilities matrix
    new_probs = np.zeros_like(result.probs)
    for old_label, new_label in cluster_mapping.items():
        new_probs[:, new_label] = result.probs[:, old_label]
    
    # Remap means and covariances (if present)
    new_means = np.zeros_like(result.means)
    for old_label, new_label in cluster_mapping.items():
        if old_label < len(result.means) and new_label < len(result.means):
            new_means[new_label] = result.means[old_label]
    
    new_covariances = None
    if result.covariances is not None:
        new_covariances = np.zeros_like(result.covariances)
        for old_label, new_label in cluster_mapping.items():
            if old_label < len(result.covariances) and new_label < len(result.covariances):
                new_covariances[new_label] = result.covariances[old_label]
    
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
    Perform clustering using Spectral Clustering with probabilistic assignments.
    
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
    sigma : float
        Parameter for Gaussian kernel
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    ClusteringResult
        Standardized clustering result object
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import kneighbors_graph
    from scipy.spatial.distance import pdist, squareform
    import time
    
    start_time = time.time()
    
    # Extract features
    X = np.column_stack((df[feat1].values, df[feat2].values))
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute affinity matrix (RBF kernel)
    distances = squareform(pdist(X_scaled, 'euclidean'))
    affinity = np.exp(-distances**2 / (2 * sigma**2))
    
    # Perform spectral clustering
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=random_state
    )
    labels = model.fit_predict(affinity)
    
    # Compute normalized Laplacian
    D = np.diag(np.sum(affinity, axis=1))
    L = D - affinity
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(affinity, axis=1)))
    L_norm = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)
    
    # Get eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    # Sort by eigenvalues
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Get embedding (skip 0th eigenvector)
    embedding = eigenvectors[:, 1:n_clusters+1]
    
    # Calculate probabilistic assignments
    from sklearn.decomposition import PCA
    
    # Use PCA to find cluster centers in the embedded space
    pca = PCA(n_components=2).fit(embedding)
    X_pca = pca.transform(embedding)
    
    # Calculate cluster centers
    centers = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        cluster_points = X_pca[labels == i]
        centers[i] = np.mean(cluster_points, axis=0)
    
    # Calculate distances to each center in embedded space
    distances = np.zeros((len(X_pca), n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.linalg.norm(X_pca - centers[i], axis=1)
    
    # Convert distances to probabilities using softmax
    def softmax(x):
        e_x = np.exp(-x)  # Negative for inverting distance
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    probs = softmax(distances)
    
    # Map centers back to original space (approximate)
    # We can't directly map back, but we can estimate the centers
    original_centers = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        mask = labels == i
        if np.sum(mask) > 0:
            original_centers[i] = np.mean(X[mask], axis=0)
    
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
        name='Spectral Clustering with Probabilistic Assignments',
        labels=labels,
        probs=probs,
        means=original_centers,
        covariances=covariances,
        model={'spectral': model, 'embedding': embedding},
        computation_time=computation_time,
        additional_info={'sigma': sigma, 'eigenvalues': eigenvalues[:n_clusters+1]}
    )
    
    # Sort clusters from left to right
    result = sort_clusters_by_position(result, df, feat1)
    
    return result

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

def compare_all_clustering_methods(df, feat1, feat2, n_clusters=3, random_state=42):
    """
    Compare all clustering methods on the given data with visualizations and ANOVA analysis.
    
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
    
    # List of clustering methods to compare
    clustering_methods = [
        ('GMM', cluster_gmm_wrapper),
        ('Mixture of t', cluster_mixture_t),
        ('DPMM', cluster_dpmm),
        ('KDE-Based', cluster_kde_based),
        ('Fuzzy C-Means', cluster_fuzzy_cmeans),
        ('Spectral Prob', cluster_spectral_prob)
    ]
    
    # Run all methods and collect results
    results = {}
    computation_times = []
    anova_results = {}
    
    for name, method in clustering_methods:
        print(f"\nRunning {name}...")
        try:
            # Special case for DPMM which can determine its own number of clusters
            result = method(df, feat1, feat2, n_clusters=n_clusters, random_state=random_state)
            
            # Display number of clusters found
            actual_clusters = len(np.unique(result.labels))
            print(f"  Found {actual_clusters} clusters")
            
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

