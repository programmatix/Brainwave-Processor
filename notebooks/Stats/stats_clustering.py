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

# Set a seed for reproducibility
np.random.seed(42)

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

