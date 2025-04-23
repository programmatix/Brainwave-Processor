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
from stats_outliers import detect_outliers
from stats_clustering import find_optimal_data_subsets
from stats_shared import create_prediction_grid, plot_regression_relationship

# Set a seed for reproducibility
np.random.seed(42)

# ML Models for relationship discovery between two variables


def analyze_regional_performance(X, y, y_pred, n_segments=5):
    sorted_indices = np.argsort(X.ravel())
    X_sorted = X.ravel()[sorted_indices]
    y_sorted = y[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    segment_size = len(X_sorted) // n_segments
    segment_metrics = []
    
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(X_sorted)
        
        if start_idx == end_idx:
            continue
            
        X_segment = X_sorted[start_idx:end_idx]
        y_segment = y_sorted[start_idx:end_idx]
        y_pred_segment = y_pred_sorted[start_idx:end_idx]
        
        segment_rmse = np.sqrt(mean_squared_error(y_segment, y_pred_segment))
        segment_r2 = r2_score(y_segment, y_pred_segment) if len(np.unique(y_segment)) > 1 else float('nan')
        
        segment_metrics.append({
            'segment': i + 1,
            'x_range': (X_segment.min(), X_segment.max()),
            'count': len(X_segment),
            'rmse': segment_rmse,
            'r2': segment_r2
        })
    
    return segment_metrics

def plot_regional_performance(segment_metrics, model_name):
    segments = [f"{m['segment']}\n({m['x_range'][0]:.1f}-{m['x_range'][1]:.1f})" for m in segment_metrics]
    rmse_values = [m['rmse'] for m in segment_metrics]
    r2_values = [m['r2'] for m in segment_metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.bar(segments, rmse_values)
    ax1.set_title(f'{model_name}: RMSE by Region (lower is better)')
    ax1.set_xlabel('Region (Shower Time Range)')
    ax1.set_ylabel('RMSE')
    
    ax2.bar(segments, r2_values)
    ax2.set_title(f'{model_name}: R² by Region (higher is better)')
    ax2.set_xlabel('Region (Shower Time Range)')
    ax2.set_ylabel('R²')
    ax2.set_ylim(-0.5, 1.0)
    
    plt.tight_layout()
    plt.show()

def find_optimal_threshold(X, y, threshold_range=None, step=0.1):
    if threshold_range is None:
        threshold_range = (X.min(), X.max())
    
    thresholds = np.arange(threshold_range[0], threshold_range[1], step)
    results = []
    
    for threshold in thresholds:
        mask = X.ravel() >= threshold
        
        if sum(mask) < 5 or sum(~mask) < 5:  # Skip if too few points
            continue
            
        X_after = X[mask]
        y_after = y[mask]
        
        # Try a simple linear model on the segment after threshold
        if len(X_after) > 0:
            # Fit simple linear regression for after threshold
            corr_after, _ = stats.pearsonr(X_after.ravel(), y_after) if len(X_after) > 1 else (0, 1)
            r2_after = corr_after**2
            
            results.append({
                'threshold': threshold,
                'points_after': sum(mask),
                'points_before': sum(~mask),
                'r2_after': r2_after,
                'mean_after': y_after.mean() if len(y_after) > 0 else None,
                'std_after': y_after.std() if len(y_after) > 0 else None
            })
    
    if not results:
        return None
        
    results_df = pd.DataFrame(results)
    
    # Find threshold with best R² on the right side
    best_idx = results_df['r2_after'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(results_df['threshold'], results_df['r2_after'])
    ax1.axvline(x=best_threshold, color='red', linestyle='--')
    ax1.set_title(f'R² After Threshold (Best: {best_threshold:.2f})')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('R² After Threshold')
    
    ax2.scatter(X, y, alpha=0.6)
    ax2.axvline(x=best_threshold, color='red', linestyle='--', 
               label=f'Threshold: {best_threshold:.2f}')
    ax2.set_title('Data with Best Threshold')
    ax2.set_xlabel('Shower Time')
    ax2.set_ylabel('LEP Time')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return best_threshold, results_df

def fit_decision_tree_regressor(df, feat1, feat2, max_depth=3, find_subsets=True, remove_outliers=True, outlier_method='influence'):
    X = df[[feat1]].values
    y = df[feat2].values
    
    # Detect and remove outliers if requested
    outlier_mask = np.ones(len(X), dtype=bool)
    if remove_outliers:
        X_filtered, y_filtered, outlier_mask, outlier_scores, _ = detect_outliers(
            X, y, method=outlier_method,
            model_factory=lambda X, y: DecisionTreeRegressor(max_depth=max_depth).fit(X, y)
        )
    else:
        X_filtered = X
        y_filtered = y
    
    # Define model factory function for consistent model creation
    def dt_factory(X_train, y_train):
        return DecisionTreeRegressor(max_depth=max_depth, random_state=42).fit(X_train, y_train)
    
    # Fit model on filtered data
    X_filtered = X_filtered
    y_filtered = y_filtered
    
    model = dt_factory(X_filtered, y_filtered)
    
    # Predictions on filtered data
    y_pred_filtered = model.predict(X_filtered)
    
    # Calculate errors and metrics on filtered data
    errors_filtered = np.abs(y_filtered - y_pred_filtered)
    
    metrics = {
        'r2': r2_score(y_filtered, y_pred_filtered),
        'rmse': np.sqrt(mean_squared_error(y_filtered, y_pred_filtered))
    }
    
    # Extract the decision thresholds from the tree
    thresholds = []
    tree = model.tree_
    for i in range(tree.node_count):
        if tree.children_left[i] != tree.children_right[i]:  # it's a split node
            threshold = tree.threshold[i]
            if tree.feature[i] == 0:  # Only if it's our feature
                thresholds.append(threshold)
    
    # Create prediction grid for smooth curve
    x_new = create_prediction_grid(X.ravel())
    y_new = model.predict(x_new)
    
    # Print decision tree rules
    tree_text = export_text(model, feature_names=[feat1])
    print("Decision Tree Rules:")
    print(tree_text)
    
    # Find optimal data subsets if requested
    best_subset = None
    subset_results = None
    if find_subsets:
        # Use only the filtered data for subset discovery
        df_filtered = df.iloc[np.where(outlier_mask)[0]]
        subset_results = find_optimal_data_subsets(df_filtered, feat1, feat2, dt_factory, 
                                                   f"Decision Tree (max_depth={max_depth})", use_2d_clustering=True)
    
    # --- Combined Plotting with Tree Visualization ---
    # Create a 1x3 grid: [Tree | Prediction+Clusters | Performance]
    fig = plt.figure(figsize=(22, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.5, 1])  # Tree, Prediction, Performance
    
    # Panel 1 (Left): Tree Visualization
    ax_tree = fig.add_subplot(gs[0])
    plot_tree(model, filled=True, feature_names=[feat1], rounded=True, 
              ax=ax_tree, fontsize=10, proportion=True)
    ax_tree.set_title(f"Decision Tree (max_depth={max_depth})", fontsize=14)
    
    # Panel 2 (Middle): Prediction Plot with Clusters
    ax_pred = fig.add_subplot(gs[1])
    
    # Plot common elements - prediction line, outliers, decision boundaries
    if not np.all(outlier_mask):
        ax_pred.scatter(X[~outlier_mask], y[~outlier_mask], color='grey', alpha=0.4, s=30,
                      label=f'Outliers ({np.sum(~outlier_mask)} points)')
    
    # Plot the model prediction curve
    ax_pred.plot(x_new, y_new, color='red', linewidth=2.5, label='Decision Tree Prediction', zorder=10)
    
    # Add vertical lines for decision boundaries to Prediction Panel
    y_range = ax_pred.get_ylim()
    for threshold in sorted(thresholds):
        ax_pred.axvline(x=threshold, color='green', linestyle='--', alpha=0.7, zorder=8)
        ax_pred.text(threshold, y_range[0] + 0.05 * (y_range[1] - y_range[0]), 
                   f"{threshold:.2f}", rotation=90, color='green', fontweight='bold',
                   verticalalignment='bottom')
    
    # Plot specific content based on subset_results
    if subset_results:
        # --- Plotting for Combined View ---
        gmm_labels = subset_results['gmm_labels']
        gmm_probs = subset_results['gmm_probs']
        actual_n_clusters = subset_results['n_clusters']
        distinct_colors = plt.cm.tab10(np.linspace(0, 1, actual_n_clusters))
        
        # Blend colors for scatter plot
        blended_colors = np.zeros((len(X_filtered), 3))
        if gmm_probs.shape[1] == actual_n_clusters:
            rgb_colors = distinct_colors[:, :3]
            blended_colors = gmm_probs @ rgb_colors
        else:
            print(f"Warning: Mismatch between GMM probability columns ({gmm_probs.shape[1]}) and reported clusters ({actual_n_clusters}). Using discrete assignments.")
            gmm_all_labels = subset_results['gmm_labels']
            unique_labels = sorted(np.unique(gmm_all_labels))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            mapped_labels = np.array([label_map[l] for l in gmm_all_labels])
            safe_labels = np.clip(mapped_labels, 0, actual_n_clusters - 1)
            blended_colors = distinct_colors[safe_labels][:, :3]
        
        # Plot filtered points colored by cluster
        ax_pred.scatter(X_filtered, y_filtered, c=blended_colors, 
                      s=60, edgecolors='k', alpha=0.8, zorder=5,
                      label='Filtered Data (colored by GMM Cluster)')
        
        # Draw GMM ellipses/means
        if subset_results['use_2d_clustering']:
            means = subset_results['gmm_means']
            covariances = subset_results['gmm_covariances']
            gmm_components_used_for_cov = subset_results.get('gmm_covariances_', covariances)
            for j in range(actual_n_clusters):
                mean = means[j]
                if j < len(gmm_components_used_for_cov):
                    covar = gmm_components_used_for_cov[j]
                    if np.all(np.linalg.eigvalsh(covar) > 1e-6):
                        try:
                            v, w = np.linalg.eigh(covar)
                            angle = np.arctan2(w[1, 0], w[0, 0])
                            angle = 180 * angle / np.pi
                            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                            ell = Ellipse(xy=(mean[0], mean[1]), width=v[0], height=v[1], angle=angle,
                                        color=distinct_colors[j], alpha=0.25, zorder=1)
                            ax_pred.add_artist(ell)
                            ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
                        except np.linalg.LinAlgError:
                            print(f"Warning: Could not draw ellipse for cluster {j}.")
                            ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
                    else:
                        print(f"Warning: Skipping ellipse for cluster {j} due to non-positive definite covariance.")
                        ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
                else:
                    print(f"Warning: Covariance data not found for cluster index {j}.")
                    ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
        else: # 1D clustering
            means = subset_results['gmm_means']
            for j in range(actual_n_clusters):
                mean_x = means[j][0]
                ax_pred.axvline(x=mean_x, color=distinct_colors[j], linestyle='--', alpha=0.7, linewidth=2, zorder=1)
                ax_pred.text(mean_x, ax_pred.get_ylim()[1] * 0.95, f' C{j}', color=distinct_colors[j], ha='center', va='top', fontweight='bold')
        
        # Add combined legend for ax_pred
        handles, labels = ax_pred.get_legend_handles_labels()
        cluster_metrics_list = subset_results.get('cluster_metrics', [])
        legend_elements = []
        if cluster_metrics_list:
            legend_elements = [Patch(facecolor=distinct_colors[i], edgecolor='black', alpha=0.6,
                                   label=f"Cluster {cluster_metrics_list[i]['label']} (n={cluster_metrics_list[i]['size']})") 
                             for i in range(min(actual_n_clusters, len(cluster_metrics_list)))]
        
        ax_pred.legend(handles=handles + legend_elements, loc='best', fontsize='small')
        
        # Panel 3 (Right): Cluster Performance Bar Chart
        ax_bars = fig.add_subplot(gs[2])
        cluster_metrics = subset_results['cluster_metrics']
        if cluster_metrics:
            cluster_labels = [f"C{m['label']}\n(n={m['size']})" for m in cluster_metrics]
            cv_r2 = [m['cv_r2'] for m in cluster_metrics]
            spearman_rho = [m['spearman_rho'] for m in cluster_metrics]
            spearman_p = [m['spearman_p'] for m in cluster_metrics]
            x = np.arange(len(cluster_labels))
            width = 0.25
            bar_colors = distinct_colors[:len(cluster_labels)]
            
            rects1 = ax_bars.bar(x - width, cv_r2, width, label='CV R²', color=bar_colors)
            rects2 = ax_bars.bar(x, spearman_rho, width, label='Spearman ρ', color=[(c[0], c[1], c[2], 0.7) for c in bar_colors])
            rects3 = ax_bars.bar(x + width, spearman_p, width, label='Spearman p', color=[(c[0], c[1], c[2], 0.4) for c in bar_colors])
            
            ax_bars.set_ylabel('Score / p-value')
            ax_bars.set_title(f"Cluster Performance Metrics", fontsize=14)
            ax_bars.set_xticks(x)
            ax_bars.set_xticklabels(cluster_labels)
            ax_bars.legend(fontsize='small')
            ax_bars.axhline(0, color='grey', linewidth=0.8)
            
            # Dynamic Y limits
            min_val = min([min(cv_r2, default=0), min(spearman_rho, default=0)])
            max_val = max([max(cv_r2, default=0), max(spearman_rho, default=0), max(spearman_p, default=0)])
            ax_bars.set_ylim(min(min_val * 1.1 if min_val < 0 else -0.1, -0.1),
                           max(1.05, max_val * 1.1 if max_val > 0 else 0.1))
            
            # Add bar labels
            def add_bar_labels(bars, format_str="{:.2f}"):
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax_bars.annotate(format_str.format(height),
                                      xy=(bar.get_x() + bar.get_width() / 2, height),
                                      xytext=(0, 3 if height >= 0 else -12),
                                      textcoords="offset points",
                                      ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            
            add_bar_labels(rects1)
            add_bar_labels(rects2)
            add_bar_labels(rects3, format_str="{:.3f}")
            ax_bars.grid(True, axis='y', linestyle='--', alpha=0.6)
        else:
            # If no cluster metrics, use this space to show error distribution
            ax_bars.text(0.5, 0.5, "No cluster metrics available", 
                       ha='center', va='center', fontsize=12,
                       transform=ax_bars.transAxes)
            ax_bars.set_title("Error Distribution", fontsize=14)
    
    else:
        # --- Plotting for Simple View (No Subset Analysis) ---
        ax_pred.scatter(X_filtered, y_filtered, color='blue', alpha=0.6, s=50, edgecolors='k',
                      label='Filtered Data', zorder=5)
        
        # Use the third panel for something relevant even without clustering
        ax_bars = fig.add_subplot(gs[2])
        ax_bars.hist(errors_filtered, bins=15, color='skyblue', edgecolor='black')
        ax_bars.set_title('Prediction Error Distribution', fontsize=14)
        ax_bars.set_xlabel('Absolute Error')
        ax_bars.set_ylabel('Frequency')
        ax_bars.grid(True, linestyle='--', alpha=0.4)
        
        # Add legend to prediction plot
        ax_pred.legend(loc='best', fontsize='small')
    
    # Common settings for prediction plot
    ax_pred.set_title(f"Decision Tree (R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f})", fontsize=14)
    ax_pred.set_xlabel(feat1, fontsize=12)
    ax_pred.set_ylabel(feat2, fontsize=12)
    ax_pred.grid(True, linestyle='--', alpha=0.6)
    
    # Final figure adjustments
    fig.suptitle(f"Decision Tree Analysis (max_depth={max_depth})", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    print(f"Decision Tree Regression Results:")
    print(f"R² score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    if not np.all(outlier_mask):
        print(f"Outliers removed: {np.sum(~outlier_mask)} points ({np.sum(~outlier_mask)/len(X)*100:.1f}%)")
    
    return model, metrics, subset_results, {'mask': outlier_mask, 'scores': outlier_scores}

def fit_gaussian_process_regressor(df, feat1, feat2, find_subsets=True, remove_outliers=True, outlier_method='influence', alpha=0.1):
    X = df[[feat1]].values
    y = df[feat2].values

    # Define a factory function to use for outlier detection
    def gp_factory(X_train, y_train):
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Use a simple RBF kernel for outlier detection to avoid overfitting
        kernel = 1.0 * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, random_state=42)
        gp.fit(X_train_scaled, y_train)
        
        # Wrapper to handle scaling
        class GPWrapper:
            def __init__(self, gp, scaler):
                self.gp = gp
                self.scaler = scaler
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                return self.gp.predict(X_scaled)
                
        return GPWrapper(gp, scaler)

    # Detect and remove outliers
    outlier_mask = np.ones(len(X), dtype=bool)
    if remove_outliers:
        X_filtered, y_filtered, outlier_mask, outlier_scores, _ = detect_outliers(
            X, y, method=outlier_method, 
            model_factory=gp_factory
        )
    else:
        X_filtered = X
        y_filtered = y
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_filtered)
    
    # Try different kernels for better non-linear relationships
    kernels = [
        ("RBF", 1.0 * RBF(length_scale=1.0)),
        ("Matern", 1.0 * Matern(length_scale=1.0, nu=1.5)),
        ("RBF + Matern", 1.0 * RBF(length_scale=1.0) + 1.0 * Matern(length_scale=1.0, nu=1.5)),
        ("Dot Product", 1.0 * DotProduct(sigma_0=1.0)),
        ("Complex", 1.0 * RBF(length_scale=1.0) + 1.0 * DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=0.1))
    ]
    
    # Test different length scales for RBF kernels
    length_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    for length_scale in length_scales:
        # Add kernels with this length scale
        kernels.append((f"RBF (ls={length_scale})", 1.0 * RBF(length_scale=length_scale)))
    
    
    # Use filtered data for model fitting
    X_filtered = X_filtered
    y_filtered = y_filtered
    X_filtered_scaled = scaler_X.transform(X_filtered)
    
    # Find best kernel on filtered data
    best_model = None
    best_score = -float('inf')
    best_kernel_name = ""
    best_length_scale = 1.0  # Default
    
    # Try all kernels on filtered data
    for kernel_name, kernel in kernels:
        model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, random_state=42)
        model.fit(X_filtered_scaled, y_filtered)
        
        y_pred = model.predict(X_filtered_scaled)
        score = r2_score(y_filtered, y_pred)
        
        if score > best_score:
            best_score = score
            best_model = model
            best_kernel_name = kernel_name
            
            # Get length scale if it's an RBF kernel
            if hasattr(model.kernel_, 'k1') and hasattr(model.kernel_.k1, 'length_scale'):
                best_length_scale = model.kernel_.k1.length_scale
            elif hasattr(model.kernel_, 'length_scale'):
                best_length_scale = model.kernel_.length_scale
    
    model = best_model
    y_pred_filtered = model.predict(X_filtered_scaled)
    errors_filtered = np.abs(y_filtered - y_pred_filtered)
    
    metrics = {
        'r2': r2_score(y_filtered, y_pred_filtered),
        'rmse': np.sqrt(mean_squared_error(y_filtered, y_pred_filtered))
    }
    
    # Create prediction grid for smooth curve
    x_new = create_prediction_grid(X.ravel(), mesh_step_size=0.01)
    x_new_scaled = scaler_X.transform(x_new)
    y_new, y_std = model.predict(x_new_scaled, return_std=True)
    
    # Print kernel information
    print(f"Best Gaussian Process Kernel: {best_kernel_name}")
    print(f"Optimized kernel parameters: {model.kernel_}")
    
    # Find optimal data subsets if requested
    best_subset = None
    subset_results = None
    if find_subsets:
        # Use only the filtered data for subset discovery
        df_filtered = df.iloc[np.where(outlier_mask)[0]]
        
        # Create a model factory for GP with the best kernel
        def gp_subset_factory(X_train, y_train):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Recreate the best kernel (simpler than trying to extract the exact same one)
            if "RBF" in best_kernel_name and "Matern" in best_kernel_name:
                kernel = 1.0 * RBF(length_scale=best_length_scale) + 1.0 * Matern(length_scale=best_length_scale, nu=1.5)
            elif "RBF" in best_kernel_name:
                kernel = 1.0 * RBF(length_scale=best_length_scale)
            elif "Matern" in best_kernel_name:
                kernel = 1.0 * Matern(length_scale=best_length_scale, nu=1.5)
            elif "Dot" in best_kernel_name:
                kernel = 1.0 * DotProduct(sigma_0=1.0)
            elif "Complex" in best_kernel_name:
                kernel = 1.0 * RBF(length_scale=best_length_scale) + 1.0 * DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=0.1)
            else:
                kernel = 1.0 * RBF(length_scale=best_length_scale)
                
            gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, random_state=42)
            gp.fit(X_train_scaled, y_train)
            
            # Create a wrapper that handles scaling internally
            class GPWrapper:
                def __init__(self, gp, scaler):
                    self.gp = gp
                    self.scaler = scaler
                
                def predict(self, X):
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    X_scaled = self.scaler.transform(X)
                    return self.gp.predict(X_scaled)
            
            return GPWrapper(gp, scaler)
            
        subset_results = find_optimal_data_subsets(df_filtered, feat1, feat2, gp_subset_factory, 
                                                  f"Gaussian Process ({best_kernel_name})", use_2d_clustering=True)
    
    # --- Combined Plotting ---
    # Always create figure and axes using gridspec initially
    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax_pred = fig.add_subplot(gs[0])
    ax_bars = fig.add_subplot(gs[1])
    
    # Plot 1 (ax_pred): Common elements - prediction line, uncertainty, outliers
    if not np.all(outlier_mask):
        ax_pred.scatter(X[~outlier_mask], y[~outlier_mask], color='grey', alpha=0.4, s=30,
                      label=f'Outliers ({np.sum(~outlier_mask)} points)')
    
    # Plot prediction line with uncertainty bands
    ax_pred.plot(x_new, y_new, color='red', linewidth=2.5, label='GP Prediction', zorder=10)
    ax_pred.fill_between(x_new.ravel(), y_new - 2*y_std, y_new + 2*y_std, 
                       color='red', alpha=0.2, label='Uncertainty (±2σ)', zorder=9)
    
    # Plot specific content based on subset_results
    if subset_results:
        # --- Plotting for Combined View ---
        gmm_labels = subset_results['gmm_labels']
        gmm_probs = subset_results['gmm_probs']
        actual_n_clusters = subset_results['n_clusters']
        distinct_colors = plt.cm.tab10(np.linspace(0, 1, actual_n_clusters))
        
        # Blend colors for scatter plot
        blended_colors = np.zeros((len(X_filtered), 3))
        if gmm_probs.shape[1] == actual_n_clusters:
            rgb_colors = distinct_colors[:, :3]
            blended_colors = gmm_probs @ rgb_colors
        else:
            print(f"Warning: Mismatch between GMM probability columns ({gmm_probs.shape[1]}) and reported clusters ({actual_n_clusters}). Using discrete assignments.")
            gmm_all_labels = subset_results['gmm_labels']
            unique_labels = sorted(np.unique(gmm_all_labels))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            mapped_labels = np.array([label_map[l] for l in gmm_all_labels])
            safe_labels = np.clip(mapped_labels, 0, actual_n_clusters - 1)
            blended_colors = distinct_colors[safe_labels][:, :3]
        
        # Plot filtered points colored by cluster
        ax_pred.scatter(X_filtered, y_filtered, c=blended_colors, 
                      s=60, edgecolors='k', alpha=0.8, zorder=5,
                      label='Filtered Data (colored by GMM Cluster)')
        
        # Draw GMM ellipses/means
        if subset_results['use_2d_clustering']:
            means = subset_results['gmm_means']
            covariances = subset_results['gmm_covariances']
            gmm_components_used_for_cov = subset_results.get('gmm_covariances_', covariances)
            for j in range(actual_n_clusters):
                mean = means[j]
                if j < len(gmm_components_used_for_cov):
                    covar = gmm_components_used_for_cov[j]
                    if np.all(np.linalg.eigvalsh(covar) > 1e-6):
                        try:
                            v, w = np.linalg.eigh(covar)
                            angle = np.arctan2(w[1, 0], w[0, 0])
                            angle = 180 * angle / np.pi
                            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                            ell = Ellipse(xy=(mean[0], mean[1]), width=v[0], height=v[1], angle=angle,
                                        color=distinct_colors[j], alpha=0.25, zorder=1)
                            ax_pred.add_artist(ell)
                            ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
                        except np.linalg.LinAlgError:
                            print(f"Warning: Could not draw ellipse for cluster {j}.")
                            ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
                    else:
                        print(f"Warning: Skipping ellipse for cluster {j} due to non-positive definite covariance.")
                        ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
                else:
                    print(f"Warning: Covariance data not found for cluster index {j}.")
                    ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
        else: # 1D clustering
            means = subset_results['gmm_means']
            for j in range(actual_n_clusters):
                mean_x = means[j][0]
                ax_pred.axvline(x=mean_x, color=distinct_colors[j], linestyle='--', alpha=0.7, linewidth=2, zorder=1)
                ax_pred.text(mean_x, ax_pred.get_ylim()[1] * 0.95, f' C{j}', color=distinct_colors[j], ha='center', va='top', fontweight='bold')
        
        # Add combined legend for ax_pred
        handles, labels = ax_pred.get_legend_handles_labels()
        cluster_metrics_list = subset_results.get('cluster_metrics', [])
        legend_elements = []
        if cluster_metrics_list:
            legend_elements = [Patch(facecolor=distinct_colors[i], edgecolor='black', alpha=0.6,
                                   label=f"Cluster {cluster_metrics_list[i]['label']} (n={cluster_metrics_list[i]['size']})") 
                             for i in range(min(actual_n_clusters, len(cluster_metrics_list)))]
        
        ax_pred.legend(handles=handles + legend_elements, loc='best', fontsize='small')
        ax_pred.set_title(f"Overall Fit (R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}) & GMM Clusters", fontsize=14)
        
        # Plot 2 (ax_bars): Cluster Performance Bar Chart
        cluster_metrics = subset_results['cluster_metrics']
        if cluster_metrics:
            cluster_labels = [f"C{m['label']}\n(n={m['size']})" for m in cluster_metrics]
            cv_r2 = [m['cv_r2'] for m in cluster_metrics]
            spearman_rho = [m['spearman_rho'] for m in cluster_metrics]
            spearman_p = [m['spearman_p'] for m in cluster_metrics]
            x = np.arange(len(cluster_labels))
            width = 0.25
            bar_colors = distinct_colors[:len(cluster_labels)]
            
            rects1 = ax_bars.bar(x - width, cv_r2, width, label='CV R²', color=bar_colors)
            rects2 = ax_bars.bar(x, spearman_rho, width, label='Spearman ρ', color=[(c[0], c[1], c[2], 0.7) for c in bar_colors])
            rects3 = ax_bars.bar(x + width, spearman_p, width, label='Spearman p', color=[(c[0], c[1], c[2], 0.4) for c in bar_colors])
            
            ax_bars.set_ylabel('Score / p-value')
            ax_bars.set_title(f"{subset_results['model_name']} Performance within GMM Clusters", fontsize=14)
            ax_bars.set_xticks(x)
            ax_bars.set_xticklabels(cluster_labels)
            ax_bars.legend(fontsize='small')
            ax_bars.axhline(0, color='grey', linewidth=0.8)
            
            # Dynamic Y limits
            min_val = min([min(cv_r2, default=0), min(spearman_rho, default=0)])
            max_val = max([max(cv_r2, default=0), max(spearman_rho, default=0), max(spearman_p, default=0)])
            ax_bars.set_ylim(min(min_val * 1.1 if min_val < 0 else -0.1, -0.1),
                           max(1.05, max_val * 1.1 if max_val > 0 else 0.1))
            
            # Add bar labels
            def add_bar_labels(bars, format_str="{:.2f}"):
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax_bars.annotate(format_str.format(height),
                                      xy=(bar.get_x() + bar.get_width() / 2, height),
                                      xytext=(0, 3 if height >= 0 else -12),
                                      textcoords="offset points",
                                      ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            
            add_bar_labels(rects1)
            add_bar_labels(rects2)
            add_bar_labels(rects3, format_str="{:.3f}")
            ax_bars.grid(True, axis='y', linestyle='--', alpha=0.6)
        else:
            ax_bars.set_visible(False)
            print("No cluster metrics available to plot performance bars.")
        
        # Set overall figure title
        fig.suptitle(f"Gaussian Process Regression ({best_kernel_name}) Analysis", fontsize=16, y=0.98)
    
    else:
        # --- Plotting for Simple View (No Subset Analysis) ---
        ax_pred.scatter(X_filtered, y_filtered, color='blue', alpha=0.6, s=50, edgecolors='k',
                      label='Filtered Data', zorder=5)
        ax_pred.legend(loc='best', fontsize='small')
        ax_pred.set_title(f"Gaussian Process Regression ({best_kernel_name})\nOverall Fit (R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f})", fontsize=14)
        # Hide the second subplot
        ax_bars.set_visible(False)
    
    # Configure ax_pred general settings
    ax_pred.set_xlabel(feat1, fontsize=12)
    ax_pred.set_ylabel(feat2, fontsize=12)
    ax_pred.grid(True, linestyle='--', alpha=0.6)
    
    # Final adjustments and display
    if not ax_bars.get_visible():
        # Resize figure for single plot view
        fig.set_size_inches(10, 7)
        fig.tight_layout()
    else:
        # Adjust layout for combined view with suptitle
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()
    
    print(f"\nGaussian Process Regression Results (Kernel: {best_kernel_name}):")
    print(f"R² score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    if not np.all(outlier_mask):
        print(f"Outliers removed: {np.sum(~outlier_mask)} points ({np.sum(~outlier_mask)/len(X)*100:.1f}%)")
    
    return model, scaler_X, metrics, subset_results, {'mask': outlier_mask, 'scores': outlier_scores}

def fit_svr(df, feat1, feat2, find_subsets=True, remove_outliers=True, outlier_method='influence'):
    X = df[[feat1]].values
    y = df[feat2].values
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Try different C values for flexibility
    c_values = [0.1, 1.0, 10.0, 100.0]
    kernels = ['rbf', 'poly', 'sigmoid']
    
    # Define factory function for SVR
    def svr_factory(X_train, y_train):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Use default parameters for outlier detection
        svr_model = SVR(kernel='rbf', C=1.0)
        svr_model.fit(X_train_scaled, y_train)
        
        # Create a wrapper that handles scaling internally
        class SVRWrapper:
            def __init__(self, svr, scaler):
                self.svr = svr
                self.scaler = scaler
            
            def predict(self, X):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                X_scaled = self.scaler.transform(X)
                return self.svr.predict(X_scaled)
        
        return SVRWrapper(svr_model, scaler)
    
    # Detect and remove outliers if requested
    outlier_mask = np.ones(len(X), dtype=bool)  # Default: keep all points
    outlier_scores = np.zeros(len(X))
    
    if remove_outliers:
        X_filtered, y_filtered, outlier_mask, outlier_scores, _ = detect_outliers(
            X, y, method=outlier_method, max_remove_percent=10, 
            model_factory=svr_factory
        )
    else:
        X_filtered = X
        y_filtered = y
    
    X_filtered_scaled = scaler_X.transform(X_filtered)
    
    # Find best parameters on filtered data
    best_model = None
    best_score = -float('inf')
    best_params = {}
    
    for k in kernels:
        for c in c_values:
            model = SVR(kernel=k, C=c)
            model.fit(X_filtered_scaled, y_filtered)
            
            y_pred = model.predict(X_filtered_scaled)
            score = r2_score(y_filtered, y_pred)
            
            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'kernel': k, 'C': c}
    
    model = best_model
    kernel = best_params['kernel']
    C = best_params['C']
    
    y_pred_filtered = model.predict(X_filtered_scaled)
    errors_filtered = np.abs(y_filtered - y_pred_filtered)
    
    metrics = {
        'r2': r2_score(y_filtered, y_pred_filtered),
        'rmse': np.sqrt(mean_squared_error(y_filtered, y_pred_filtered)),
        'mae': mean_absolute_error(y_filtered, y_pred_filtered),
        'mse': mean_squared_error(y_filtered, y_pred_filtered)
    }
    
    # Create prediction grid for smooth curve
    x_new = create_prediction_grid(X.ravel())
    x_new_scaled = scaler_X.transform(x_new)
    y_new = model.predict(x_new_scaled)
    
    best_subset = None
    subset_results = None
    cluster_models = {}
    
    if find_subsets:
        # Use only the filtered data for subset discovery
        df_filtered = df.iloc[np.where(outlier_mask)[0]]
        
        # Create a model factory for SVR with best parameters
        def svr_subset_factory(X_train, y_train):
            # Need to handle scaling within the factory if the main model used scaling
            scaler_subset = StandardScaler()
            X_train_scaled = scaler_subset.fit_transform(X_train)
            
            svr_model = SVR(kernel=kernel, C=C)
            svr_model.fit(X_train_scaled, y_train)
            
            # Create a wrapper that handles scaling internally for prediction
            class SVRWrapper:
                def __init__(self, svr, scaler):
                    self.svr = svr
                    self.scaler = scaler
                
                def predict(self, X):
                    # Ensure X is 2D before scaling
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    X_scaled = self.scaler.transform(X)
                    return self.svr.predict(X_scaled)
            
            return SVRWrapper(svr_model, scaler_subset)
            
        subset_results = find_optimal_data_subsets(
            df_filtered, feat1, feat2, svr_subset_factory, 
            f"SVR ({kernel}, C={C})", use_2d_clustering=True
        )
        
        if subset_results:
            # Create a best subset dataframe with cluster assignments
            best_subset = df_filtered.copy()
            best_subset['cluster'] = subset_results['gmm_labels']
            
            # Create models for each cluster
            if 'cluster_metrics' in subset_results and subset_results['cluster_metrics']:
                for cluster_info in subset_results['cluster_metrics']:
                    cluster_id = cluster_info['label']
                    cluster_mask = best_subset['cluster'] == cluster_id
                    
                    if np.sum(cluster_mask) >= 5:  # Only create models for clusters with sufficient data
                        cluster_X = best_subset.loc[cluster_mask, feat1].values.reshape(-1, 1)
                        cluster_y = best_subset.loc[cluster_mask, feat2].values
                        cluster_models[cluster_id] = svr_subset_factory(cluster_X, cluster_y)

    # Create main visualization figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot the main model on the first axis
    plot_regression_relationship(
        model_name=f"SVR ({kernel}, C={C})", 
        X=X_filtered, 
        y=y_filtered, 
        y_pred=y_pred_filtered,
        x_new=x_new,
        y_new=y_new,
        ax=axes[0],
        metrics=metrics
    )
    
    # Plot per-cluster models on the second axis if available
    if find_subsets and best_subset is not None:
        subset_X = best_subset[feat1].values.reshape(-1, 1)
        subset_y = best_subset[feat2].values
        
        if cluster_models:
            axes[1].set_title(f"SVR (Per-Cluster Models)")
            
            # First plot the global model as a red line
            axes[1].plot(x_new, y_new, color='red', linewidth=2.5, label='Global Model', zorder=10)
            
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for cluster_id, cluster_data in best_subset.groupby('cluster'):
                cluster_X = cluster_data[feat1].values.reshape(-1, 1)
                cluster_y = cluster_data[feat2].values
                cluster_model = cluster_models.get(cluster_id)
                
                if cluster_model:
                    cluster_y_pred = cluster_model.predict(cluster_X)
                    
                    axes[1].scatter(cluster_X, cluster_y, alpha=0.6, 
                                   color=colors[cluster_id % len(colors)], 
                                   label=f"Cluster {cluster_id}")
                    
                    x_sorted = np.sort(cluster_X, axis=0)
                    y_pred_sorted = cluster_model.predict(x_sorted)
                    
                    axes[1].plot(x_sorted, y_pred_sorted, '--', 
                                color=colors[cluster_id % len(colors)], 
                                linewidth=2)
                    
                    cluster_r2 = r2_score(cluster_y, cluster_y_pred)
                    cluster_rmse = np.sqrt(mean_squared_error(cluster_y, cluster_y_pred))
                    
                    handles, labels = axes[1].get_legend_handles_labels()
                    labels = [f"{label} (R²={cluster_r2:.2f}, RMSE={cluster_rmse:.2f})" 
                             if label == f"Cluster {cluster_id}" else label 
                             for label in labels]
                    axes[1].legend(handles, labels)
            
            # Create separate figure for cluster performance metrics
            if 'cluster_metrics' in subset_results and subset_results['cluster_metrics']:
                fig_metrics, ax_bars = plt.subplots(figsize=(12, 6))
                
                cluster_metrics = subset_results['cluster_metrics']
                cluster_labels = [f"C{m['label']}\n(n={m['size']})" for m in cluster_metrics]
                cv_r2 = [m['cv_r2'] for m in cluster_metrics]
                spearman_rho = [m['spearman_rho'] for m in cluster_metrics]
                spearman_p = [m['spearman_p'] for m in cluster_metrics]
                mi_values = [m.get('mi', 0) for m in cluster_metrics]  # Add MI values
                
                x = np.arange(len(cluster_labels))
                width = 0.2  # Make bars narrower to fit 4 bars
                distinct_colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_labels)))
                bar_colors = distinct_colors[:len(cluster_labels)]
                
                rects1 = ax_bars.bar(x - 1.5*width, cv_r2, width, label='CV R²', color=bar_colors)
                rects2 = ax_bars.bar(x - 0.5*width, spearman_rho, width, label='Spearman ρ', 
                                    color=[(c[0], c[1], c[2], 0.7) for c in bar_colors])
                rects3 = ax_bars.bar(x + 0.5*width, spearman_p, width, label='Spearman p', 
                                    color=[(c[0], c[1], c[2], 0.4) for c in bar_colors])
                rects4 = ax_bars.bar(x + 1.5*width, mi_values, width, label='Mutual Info', 
                                    color=[(c[0], c[1], c[2], 0.9) for c in bar_colors])
                
                ax_bars.set_ylabel('Score / p-value / MI')
                ax_bars.set_title(f"SVR Performance within Clusters", fontsize=14)
                ax_bars.set_xticks(x)
                ax_bars.set_xticklabels(cluster_labels)
                ax_bars.legend(fontsize='small')
                ax_bars.axhline(0, color='grey', linewidth=0.8)
                
                # Dynamic Y limits
                all_values = cv_r2 + spearman_rho + mi_values
                min_val = min([min(all_values, default=0), min(spearman_p, default=0)])
                max_val = max([max(all_values, default=0), max(spearman_p, default=0)])
                ax_bars.set_ylim(min(min_val * 1.1 if min_val < 0 else -0.1, -0.1),
                               max(1.05, max_val * 1.1 if max_val > 0 else 0.1))
                
                # Add bar labels
                def add_bar_labels(bars, format_str="{:.2f}"):
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax_bars.annotate(format_str.format(height),
                                          xy=(bar.get_x() + bar.get_width() / 2, height),
                                          xytext=(0, 3 if height >= 0 else -12),
                                          textcoords="offset points",
                                          ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            
                add_bar_labels(rects1)
                add_bar_labels(rects2)
                add_bar_labels(rects3, format_str="{:.3f}")
                add_bar_labels(rects4, format_str="{:.3f}")
                ax_bars.grid(True, axis='y', linestyle='--', alpha=0.6)
        else:
            subset_y_pred = model.predict(scaler_X.transform(subset_X))
            plot_regression_relationship(
                model_name=f"SVR (Best Subset, {len(subset_X)} points)",
                X=subset_X,
                y=subset_y,
                y_pred=subset_y_pred,
                x_new=np.linspace(subset_X.min(), subset_X.max(), 100).reshape(-1, 1),
                y_new=model.predict(scaler_X.transform(np.linspace(subset_X.min(), subset_X.max(), 100).reshape(-1, 1))),
                ax=axes[1],
                metrics={
                    'r2': r2_score(subset_y, subset_y_pred),
                    'mae': mean_absolute_error(subset_y, subset_y_pred),
                    'mse': mean_squared_error(subset_y, subset_y_pred),
                    'rmse': np.sqrt(mean_squared_error(subset_y, subset_y_pred))
                }
            )
    else:
        axes[1].scatter(X_filtered, y_filtered, alpha=0.6)
        axes[1].plot(x_new, y_new, 'r-', linewidth=2)
        axes[1].set_title(f"SVR ({kernel}, C={C}) Original Data")
        axes[1].set_xlabel(feat1)
        axes[1].set_ylabel(feat2)
        axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Return comprehensive results
    results = {
        'model': model,
        'scaler_X': scaler_X,
        'metrics': metrics,
        'X': X_filtered,
        'y': y_filtered,
        'y_pred': y_pred_filtered,
        'best_subset': best_subset,
        'subset_results': subset_results,
        'cluster_models': cluster_models,
        'outlier_mask': outlier_mask, 
        'outlier_scores': outlier_scores,
        'kernel': kernel,
        'C': C,
    }
    
    print(f"SVR ({kernel}, C={C}) Performance:")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    # Print per-cluster performance
    if find_subsets and subset_results:
        best_cluster_metrics = subset_results.get('cluster_metrics', [])
        if best_cluster_metrics:
            print("\nPer-Cluster Model Performance:")
            for cluster_info in best_cluster_metrics:
                cluster_id = cluster_info['label']
                size = cluster_info['size']
                r2 = cluster_info.get('cv_r2', 0)
                rmse = cluster_info.get('y_std', 0) * np.sqrt(1 - max(0, r2))
                mi = cluster_info.get('mi', 0)
                print(f"  Cluster {cluster_id} (size: {size}): R²={r2:.4f}, RMSE={rmse:.4f}, MI={mi:.4f}")
    
    return results

def fit_knn_regressor(df, feat1, feat2, find_subsets=True):
    X = df[[feat1]].values
    y = df[feat2].values
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Try different k values
    k_values = [3, 5, 7, 9, 11]
    
    best_model = None
    best_score = -float('inf')
    best_k = 0
    
    for k in k_values:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_scaled, y)
        
        y_pred = model.predict(X_scaled)
        score = r2_score(y, y_pred)
        
        if score > best_score:
            best_score = score
            best_model = model
            best_k = k
    
    model = best_model
    n_neighbors = best_k
    
    y_pred = model.predict(X_scaled)
    errors = np.abs(y - y_pred)
    
    metrics = {
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'mse': mean_squared_error(y, y_pred)
    }
    
    # Create prediction grid for smooth curve
    x_new = create_prediction_grid(X.ravel())
    x_new_scaled = scaler_X.transform(x_new)
    y_new = model.predict(x_new_scaled)
    
    # Find optimal data subsets if requested
    best_subset = None
    subset_results = None
    cluster_models = {}
    
    if find_subsets:
        # Create a model factory for KNN
        def knn_factory(X_train, y_train):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            knn_model = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(X_train)-1))
            knn_model.fit(X_train_scaled, y_train)
            
            # Create a wrapper that handles scaling internally
            class KNNWrapper:
                def __init__(self, knn, scaler):
                    self.knn = knn
                    self.scaler = scaler
                
                def predict(self, X):
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    X_scaled = self.scaler.transform(X)
                    return self.knn.predict(X_scaled)
            
            return KNNWrapper(knn_model, scaler)
            
        subset_results = find_optimal_data_subsets(df, feat1, feat2, knn_factory, 
                                                 f"k-NN Regressor (k={n_neighbors})", use_2d_clustering=True)
        
        if subset_results:
            # Create a best subset dataframe with cluster assignments
            best_subset = df.copy()
            best_subset['cluster'] = subset_results['gmm_labels']
            
            # Create models for each cluster
            if 'cluster_metrics' in subset_results and subset_results['cluster_metrics']:
                for cluster_info in subset_results['cluster_metrics']:
                    cluster_id = cluster_info['label']
                    cluster_mask = best_subset['cluster'] == cluster_id
                    
                    if np.sum(cluster_mask) >= 5:  # Only create models for clusters with sufficient data
                        cluster_X = best_subset.loc[cluster_mask, feat1].values.reshape(-1, 1)
                        cluster_y = best_subset.loc[cluster_mask, feat2].values
                        cluster_models[cluster_id] = knn_factory(cluster_X, cluster_y)
    
    # Create main visualization figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot the main model on the first axis
    plot_regression_relationship(
        model_name=f"k-NN Regressor (k={n_neighbors})", 
        X=X, 
        y=y, 
        y_pred=y_pred,
        x_new=x_new,
        y_new=y_new,
        ax=axes[0],
        metrics=metrics
    )
    
    # Show regional performance in a separate figure
    regional_metrics = analyze_regional_performance(X, y, y_pred)
    plot_regional_performance(regional_metrics, f"k-NN Regressor (k={n_neighbors})")
    
    # Plot per-cluster models on the second axis if available
    if find_subsets and best_subset is not None:
        subset_X = best_subset[feat1].values.reshape(-1, 1)
        subset_y = best_subset[feat2].values
        
        if cluster_models:
            axes[1].set_title(f"k-NN Regressor (Per-Cluster Models)")
            
            # First plot the global model as a red line
            axes[1].plot(x_new, y_new, color='red', linewidth=2.5, label='Global Model', zorder=10)
            
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for cluster_id, cluster_data in best_subset.groupby('cluster'):
                cluster_X = cluster_data[feat1].values.reshape(-1, 1)
                cluster_y = cluster_data[feat2].values
                cluster_model = cluster_models.get(cluster_id)
                
                if cluster_model:
                    cluster_y_pred = cluster_model.predict(cluster_X)
                    
                    axes[1].scatter(cluster_X, cluster_y, alpha=0.6, 
                                   color=colors[cluster_id % len(colors)], 
                                   label=f"Cluster {cluster_id}")
                    
                    x_sorted = np.sort(cluster_X, axis=0)
                    y_pred_sorted = cluster_model.predict(x_sorted)
                    
                    axes[1].plot(x_sorted, y_pred_sorted, '--', 
                                color=colors[cluster_id % len(colors)], 
                                linewidth=2)
                    
                    cluster_r2 = r2_score(cluster_y, cluster_y_pred)
                    cluster_rmse = np.sqrt(mean_squared_error(cluster_y, cluster_y_pred))
                    
                    handles, labels = axes[1].get_legend_handles_labels()
                    labels = [f"{label} (R²={cluster_r2:.2f}, RMSE={cluster_rmse:.2f})" 
                             if label == f"Cluster {cluster_id}" else label 
                             for label in labels]
                    axes[1].legend(handles, labels)
            
            # Create separate figure for cluster performance metrics with MI bar added
            if 'cluster_metrics' in subset_results and subset_results['cluster_metrics']:
                fig_metrics, ax_bars = plt.subplots(figsize=(12, 6))
                
                cluster_metrics = subset_results['cluster_metrics']
                cluster_labels = [f"C{m['label']}\n(n={m['size']})" for m in cluster_metrics]
                cv_r2 = [m['cv_r2'] for m in cluster_metrics]
                spearman_rho = [m['spearman_rho'] for m in cluster_metrics]
                spearman_p = [m['spearman_p'] for m in cluster_metrics]
                mi_values = [m.get('mi', 0) for m in cluster_metrics]  # Add MI values
                
                x = np.arange(len(cluster_labels))
                width = 0.2  # Make bars narrower to fit 4 bars
                bar_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(cluster_labels)]
                
                rects1 = ax_bars.bar(x - 1.5*width, cv_r2, width, label='CV R²', color=bar_colors)
                rects2 = ax_bars.bar(x - 0.5*width, spearman_rho, width, label='Spearman ρ', 
                                    color=[(c[0], c[1], c[2], 0.7) for c in bar_colors])
                rects3 = ax_bars.bar(x + 0.5*width, spearman_p, width, label='Spearman p', 
                                    color=[(c[0], c[1], c[2], 0.4) for c in bar_colors])
                rects4 = ax_bars.bar(x + 1.5*width, mi_values, width, label='Mutual Info', 
                                    color=[(c[0], c[1], c[2], 0.9) for c in bar_colors])
                
                ax_bars.set_ylabel('Score / p-value / MI')
                ax_bars.set_title(f"k-NN Regressor Performance within Clusters", fontsize=14)
                ax_bars.set_xticks(x)
                ax_bars.set_xticklabels(cluster_labels)
                ax_bars.legend(fontsize='small')
                ax_bars.axhline(0, color='grey', linewidth=0.8)
                
                # Dynamic Y limits
                all_values = cv_r2 + spearman_rho + mi_values
                min_val = min([min(all_values, default=0), min(spearman_p, default=0)])
                max_val = max([max(all_values, default=0), max(spearman_p, default=0)])
                ax_bars.set_ylim(min(min_val * 1.1 if min_val < 0 else -0.1, -0.1),
                               max(1.05, max_val * 1.1 if max_val > 0 else 0.1))
                
                # Add bar labels
                def add_bar_labels(bars, format_str="{:.2f}"):
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax_bars.annotate(format_str.format(height),
                                          xy=(bar.get_x() + bar.get_width() / 2, height),
                                          xytext=(0, 3 if height >= 0 else -12),
                                          textcoords="offset points",
                                          ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
                
                add_bar_labels(rects1)
                add_bar_labels(rects2)
                add_bar_labels(rects3, format_str="{:.3f}")
                add_bar_labels(rects4, format_str="{:.3f}")
                ax_bars.grid(True, axis='y', linestyle='--', alpha=0.6)
                
                plt.tight_layout()
                plt.show()
        else:
            # If no cluster models, just show the data with the global model
            axes[1].scatter(X, y, color='blue', alpha=0.6, s=50, edgecolors='k')
            axes[1].plot(x_new, y_new, color='red', linewidth=2.5, label='Global Model')
            axes[1].set_title(f"k-NN Regressor (k={n_neighbors})")
            axes[1].legend()
    
    # Configure axes general settings
    for ax in axes:
        ax.set_xlabel(feat1, fontsize=12)
        ax.set_ylabel(feat2, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"k-NN Regression Results (k={n_neighbors}):")
    print(f"R² score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    # Print per-cluster performance
    if find_subsets and subset_results:
        best_cluster_metrics = subset_results.get('cluster_metrics', [])
        if best_cluster_metrics:
            print("\nPer-Cluster Model Performance:")
            for cluster_info in best_cluster_metrics:
                cluster_id = cluster_info['label']
                size = cluster_info['size']
                r2 = cluster_info.get('cv_r2', 0)
                rmse = cluster_info.get('y_std', 0) * np.sqrt(1 - max(0, r2))
                mi = cluster_info.get('mi', 0)
                print(f"  Cluster {cluster_id} (size: {size}): R²={r2:.4f}, RMSE={rmse:.4f}, MI={mi:.4f}")
    
    # Return comprehensive results
    results = {
        'model': model,
        'scaler_X': scaler_X,
        'metrics': metrics,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'n_neighbors': n_neighbors,
        'best_subset': best_subset,
        'subset_results': subset_results,
        'cluster_models': cluster_models
    }
    
    return results

def fit_random_forest_regressor(df, feat1, feat2, find_subsets=True):
    X = df[feat1].values.reshape(-1, 1)
    y = df[feat2].values
    
    # Try different parameter combinations
    depths = [2, 3, 4, 5]
    estimators = [10, 20, 50]
    
    best_model = None
    best_score = -float('inf')
    best_params = {}
    
    for depth in depths:
        for n_est in estimators:
            model = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42)
            model.fit(X, y)
            
            y_pred = model.predict(X)
            score = r2_score(y, y_pred)
            
            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'n_estimators': n_est, 'max_depth': depth}
    
    model = best_model
    n_estimators = best_params['n_estimators']
    max_depth = best_params['max_depth']
    
    y_pred = model.predict(X)
    errors = np.abs(y - y_pred)
    
    metrics = {
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred))
    }
    
    # Print feature importance
    print(f"Random Forest Feature Importance: {model.feature_importances_[0]:.4f}")
    
    # Create prediction grid for smooth curve
    x_new = create_prediction_grid(X.ravel())
    y_new = model.predict(x_new)
    
    # Find optimal data subsets if requested
    best_subset = None
    subset_results = None
    if find_subsets:
        # Create a model factory for Random Forest
        def rf_factory(X_train, y_train):
            rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf_model.fit(X_train, y_train)
            return rf_model
            
        subset_results = find_optimal_data_subsets(df, feat1, feat2, rf_factory, 
                                                 f"Random Forest (n={n_estimators}, depth={max_depth})", use_2d_clustering=True)
    
    # --- Combined Plotting ---
    # Always create figure and axes using gridspec initially
    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax_pred = fig.add_subplot(gs[0])
    ax_bars = fig.add_subplot(gs[1])
    
    # Plot 1 (ax_pred): Common elements - prediction line
    # Plot the prediction curve
    ax_pred.plot(x_new, y_new, color='red', linewidth=2.5, label='Random Forest Prediction', zorder=10)
    
    # Plot specific content based on subset_results
    if subset_results:
        # --- Plotting for Combined View ---
        gmm_labels = subset_results['gmm_labels']
        gmm_probs = subset_results['gmm_probs']
        actual_n_clusters = subset_results['n_clusters']
        distinct_colors = plt.cm.tab10(np.linspace(0, 1, actual_n_clusters))
        
        # Blend colors for scatter plot
        blended_colors = np.zeros((len(X), 3))  # Using full X since no outliers in RF
        if gmm_probs.shape[1] == actual_n_clusters:
            rgb_colors = distinct_colors[:, :3]
            blended_colors = gmm_probs @ rgb_colors
        else:
            print(f"Warning: Mismatch between GMM probability columns ({gmm_probs.shape[1]}) and reported clusters ({actual_n_clusters}). Using discrete assignments.")
            gmm_all_labels = subset_results['gmm_labels']
            unique_labels = sorted(np.unique(gmm_all_labels))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            mapped_labels = np.array([label_map[l] for l in gmm_all_labels])
            safe_labels = np.clip(mapped_labels, 0, actual_n_clusters - 1)
            blended_colors = distinct_colors[safe_labels][:, :3]
        
        # Plot points colored by cluster
        ax_pred.scatter(X, y, c=blended_colors, s=60, edgecolors='k', alpha=0.8, zorder=5,
                       label='Data (colored by GMM Cluster)')
        
        # Draw GMM ellipses/means
        if subset_results['use_2d_clustering']:
            means = subset_results['gmm_means']
            covariances = subset_results['gmm_covariances']
            gmm_components_used_for_cov = subset_results.get('gmm_covariances_', covariances)
            for j in range(actual_n_clusters):
                mean = means[j]
                if j < len(gmm_components_used_for_cov):
                    covar = gmm_components_used_for_cov[j]
                    if np.all(np.linalg.eigvalsh(covar) > 1e-6):
                        try:
                            v, w = np.linalg.eigh(covar)
                            angle = np.arctan2(w[1, 0], w[0, 0])
                            angle = 180 * angle / np.pi
                            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                            ell = Ellipse(xy=(mean[0], mean[1]), width=v[0], height=v[1], angle=angle,
                                        color=distinct_colors[j], alpha=0.25, zorder=1)
                            ax_pred.add_artist(ell)
                            ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
                        except np.linalg.LinAlgError:
                            print(f"Warning: Could not draw ellipse for cluster {j}.")
                            ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
                    else:
                        print(f"Warning: Skipping ellipse for cluster {j} due to non-positive definite covariance.")
                        ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
                else:
                    print(f"Warning: Covariance data not found for cluster index {j}.")
                    ax_pred.scatter(mean[0], mean[1], marker='X', s=100, c=[distinct_colors[j]], edgecolors='black', zorder=11)
        else: # 1D clustering
            means = subset_results['gmm_means']
            for j in range(actual_n_clusters):
                mean_x = means[j][0]
                ax_pred.axvline(x=mean_x, color=distinct_colors[j], linestyle='--', alpha=0.7, linewidth=2, zorder=1)
                ax_pred.text(mean_x, ax_pred.get_ylim()[1] * 0.95, f' C{j}', color=distinct_colors[j], ha='center', va='top', fontweight='bold')
        
        # Add combined legend for ax_pred
        handles, labels = ax_pred.get_legend_handles_labels()
        cluster_metrics_list = subset_results.get('cluster_metrics', [])
        legend_elements = []
        if cluster_metrics_list:
            legend_elements = [Patch(facecolor=distinct_colors[i], edgecolor='black', alpha=0.6,
                                   label=f"Cluster {cluster_metrics_list[i]['label']} (n={cluster_metrics_list[i]['size']})") 
                             for i in range(min(actual_n_clusters, len(cluster_metrics_list)))]
        
        ax_pred.legend(handles=handles + legend_elements, loc='best', fontsize='small')
        ax_pred.set_title(f"Overall Fit (R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}) & GMM Clusters", fontsize=14)
        
        # Plot 2 (ax_bars): Cluster Performance Bar Chart
        cluster_metrics = subset_results['cluster_metrics']
        if cluster_metrics:
            cluster_labels = [f"C{m['label']}\n(n={m['size']})" for m in cluster_metrics]
            cv_r2 = [m['cv_r2'] for m in cluster_metrics]
            spearman_rho = [m['spearman_rho'] for m in cluster_metrics]
            spearman_p = [m['spearman_p'] for m in cluster_metrics]
            x = np.arange(len(cluster_labels))
            width = 0.25
            bar_colors = distinct_colors[:len(cluster_labels)]
            
            rects1 = ax_bars.bar(x - width, cv_r2, width, label='CV R²', color=bar_colors)
            rects2 = ax_bars.bar(x, spearman_rho, width, label='Spearman ρ', color=[(c[0], c[1], c[2], 0.7) for c in bar_colors])
            rects3 = ax_bars.bar(x + width, spearman_p, width, label='Spearman p', color=[(c[0], c[1], c[2], 0.4) for c in bar_colors])
            
            ax_bars.set_ylabel('Score / p-value')
            ax_bars.set_title(f"{subset_results['model_name']} Performance within GMM Clusters", fontsize=14)
            ax_bars.set_xticks(x)
            ax_bars.set_xticklabels(cluster_labels)
            ax_bars.legend(fontsize='small')
            ax_bars.axhline(0, color='grey', linewidth=0.8)
            
            # Dynamic Y limits
            min_val = min([min(cv_r2, default=0), min(spearman_rho, default=0)])
            max_val = max([max(cv_r2, default=0), max(spearman_rho, default=0), max(spearman_p, default=0)])
            ax_bars.set_ylim(min(min_val * 1.1 if min_val < 0 else -0.1, -0.1),
                           max(1.05, max_val * 1.1 if max_val > 0 else 0.1))
            
            # Add bar labels
            def add_bar_labels(bars, format_str="{:.2f}"):
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax_bars.annotate(format_str.format(height),
                                      xy=(bar.get_x() + bar.get_width() / 2, height),
                                      xytext=(0, 3 if height >= 0 else -12),
                                      textcoords="offset points",
                                      ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            
            add_bar_labels(rects1)
            add_bar_labels(rects2)
            add_bar_labels(rects3, format_str="{:.3f}")
            ax_bars.grid(True, axis='y', linestyle='--', alpha=0.6)
        else:
            ax_bars.set_visible(False)
            print("No cluster metrics available to plot performance bars.")
        
        # Set overall figure title
        fig.suptitle(f"Random Forest Regression (trees={n_estimators}, depth={max_depth}) Analysis", fontsize=16, y=0.98)
    
    else:
        # --- Plotting for Simple View (No Subset Analysis) ---
        # Color points by error magnitude
        error_norm = plt.Normalize(vmin=0, vmax=errors.max())
        scatter = ax_pred.scatter(X, y, c=errors, cmap='YlOrRd', norm=error_norm, s=50, edgecolors='k', zorder=3)
        plt.colorbar(scatter, ax=ax_pred, label='Prediction Error')
        
        ax_pred.set_title(f"Random Forest (trees={n_estimators}, depth={max_depth})\nR² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
        # Hide the second subplot
        ax_bars.set_visible(False)
    
    # Configure ax_pred general settings
    ax_pred.set_xlabel(feat1, fontsize=12)
    ax_pred.set_ylabel(feat2, fontsize=12)
    ax_pred.grid(True, linestyle='--', alpha=0.6)
    
    # Final adjustments and display
    if not ax_bars.get_visible():
        # Resize figure for single plot view
        fig.set_size_inches(10, 7)
        fig.tight_layout()
    else:
        # Adjust layout for combined view with suptitle
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()
    
    # --- Remove redundant plotting section ---
    # # Plot predictions with errors highlighted
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.scatter(X, y, color='blue', alpha=0.6, label='Actual data')
    # ax.plot(x_new, y_new, color='red', linewidth=2, label='Predicted relationship')
    # 
    # # Color points by error magnitude
    # error_norm = plt.Normalize(vmin=0, vmax=errors.max())
    # scatter = ax.scatter(X, y, c=errors, cmap='YlOrRd', norm=error_norm, s=50, edgecolors='k', zorder=3)
    # plt.colorbar(scatter, ax=ax, label='Prediction Error')
    # 
    # ax.set_title(f"Random Forest (trees={n_estimators}, depth={max_depth})\nR² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    # ax.set_xlabel('Shower Time')
    # ax.set_ylabel('LEP Time')
    # ax.legend()
    # plt.tight_layout()
    # plt.show()
    
    print(f"Random Forest Regression Results (trees={n_estimators}, depth={max_depth}):")
    print(f"R² score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    return model, metrics, subset_results

def fit_linear_regression(df, feat1, feat2, find_subsets=True, remove_outliers=True, outlier_method='influence'):
    X = df[[feat1]].values
    y = df[feat2].values
    
    outlier_mask = np.ones(len(df), dtype=bool)
    if remove_outliers:
        X_filtered, y_filtered, outlier_mask, outlier_scores, _ = detect_outliers(X, y, method=outlier_method, model_factory=lambda X, y: LinearRegression().fit(X, y))
    else:
        X_filtered = X
        y_filtered = y
    
    def lr_factory(X_train, y_train):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        
        class LRWrapper:
            def __init__(self, lr, scaler):
                self.lr = lr
                self.scaler = scaler
            
            def predict(self, X):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                X_scaled = self.scaler.transform(X)
                return self.lr.predict(X_scaled)
        
        return LRWrapper(lr, scaler)
    
    model = lr_factory(X_filtered, y_filtered)
    
    y_pred = model.predict(X_filtered)
    
    metrics = {
        'r2': r2_score(y_filtered, y_pred),
        'mae': mean_absolute_error(y_filtered, y_pred),
        'mse': mean_squared_error(y_filtered, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_filtered, y_pred))
    }
    
    best_subset = None
    subset_results = None
    cluster_models = {}
    
    if find_subsets:
        df_filtered = df.iloc[np.where(outlier_mask)[0]]
        
        def lr_subset_factory(X_train, y_train):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            
            class LRWrapper:
                def __init__(self, lr, scaler):
                    self.lr = lr
                    self.scaler = scaler
                
                def predict(self, X):
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    X_scaled = self.scaler.transform(X)
                    return self.lr.predict(X_scaled)
            
            return LRWrapper(lr, scaler)
        
        subset_results = find_optimal_data_subsets(
            df_filtered, feat1, feat2, 
            model_factory=lr_subset_factory,
            model_name="Linear Regression"
        )
        
        if subset_results:
            # Create a best subset dataframe with cluster assignments
            best_subset = df_filtered.copy()
            best_subset['cluster'] = subset_results['gmm_labels']
            
            # Create models for each cluster
            if 'cluster_metrics' in subset_results and subset_results['cluster_metrics']:
                for cluster_info in subset_results['cluster_metrics']:
                    cluster_id = cluster_info['label']
                    cluster_mask = best_subset['cluster'] == cluster_id
                    
                    if np.sum(cluster_mask) >= 5:  # Only create models for clusters with sufficient data
                        cluster_X = best_subset.loc[cluster_mask, feat1].values.reshape(-1, 1)
                        cluster_y = best_subset.loc[cluster_mask, feat2].values
                        cluster_models[cluster_id] = lr_subset_factory(cluster_X, cluster_y)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    plot_regression_relationship(
        model_name="Linear Regression", 
        X=X_filtered, 
        y=y_filtered, 
        y_pred=y_pred,
        x_new=np.linspace(X_filtered.min(), X_filtered.max(), 100).reshape(-1, 1),
        y_new=model.predict(np.linspace(X_filtered.min(), X_filtered.max(), 100).reshape(-1, 1)),
        ax=axes[0],
        metrics=metrics
    )
    
    if find_subsets and best_subset is not None:
        subset_X = best_subset[feat1].values.reshape(-1, 1)
        subset_y = best_subset[feat2].values
        
        if cluster_models:
            axes[1].set_title(f"Linear Regression (Per-Cluster Models)")
            
            # First plot the global model as a red line
            x_new = np.linspace(X_filtered.min(), X_filtered.max(), 100).reshape(-1, 1)
            y_new = model.predict(np.linspace(X_filtered.min(), X_filtered.max(), 100).reshape(-1, 1))
            axes[1].plot(x_new, y_new, color='red', linewidth=2.5, label='Global Model', zorder=10)
            
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for cluster_id, cluster_data in best_subset.groupby('cluster'):
                cluster_X = cluster_data[feat1].values.reshape(-1, 1)
                cluster_y = cluster_data[feat2].values
                cluster_model = cluster_models.get(cluster_id)
                
                if cluster_model:
                    cluster_y_pred = cluster_model.predict(cluster_X)
                    
                    axes[1].scatter(cluster_X, cluster_y, alpha=0.6, 
                                   color=colors[cluster_id % len(colors)], 
                                   label=f"Cluster {cluster_id}")
                    
                    x_sorted = np.sort(cluster_X, axis=0)
                    y_pred_sorted = cluster_model.predict(x_sorted)
                    
                    axes[1].plot(x_sorted, y_pred_sorted, '--', 
                                color=colors[cluster_id % len(colors)], 
                                linewidth=2)
                    
                    cluster_r2 = r2_score(cluster_y, cluster_y_pred)
                    cluster_rmse = np.sqrt(mean_squared_error(cluster_y, cluster_y_pred))
                    
                    handles, labels = axes[1].get_legend_handles_labels()
                    labels = [f"{label} (R²={cluster_r2:.2f}, RMSE={cluster_rmse:.2f})" 
                             if label == f"Cluster {cluster_id}" else label 
                             for label in labels]
                    axes[1].legend(handles, labels)
            
            # Create separate figure for cluster performance metrics
            if 'cluster_metrics' in subset_results and subset_results['cluster_metrics']:
                fig_metrics, ax_bars = plt.subplots(figsize=(12, 6))
                
                cluster_metrics = subset_results['cluster_metrics']
                cluster_labels = [f"C{m['label']}\n(n={m['size']})" for m in cluster_metrics]
                cv_r2 = [m['cv_r2'] for m in cluster_metrics]
                spearman_rho = [m['spearman_rho'] for m in cluster_metrics]
                spearman_p = [m['spearman_p'] for m in cluster_metrics]
                mi_values = [m.get('mi', 0) for m in cluster_metrics]  # Add MI values
                
                x = np.arange(len(cluster_labels))
                width = 0.2  # Make bars narrower to fit 4 bars
                bar_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(cluster_labels)]
                
                rects1 = ax_bars.bar(x - 1.5*width, cv_r2, width, label='CV R²', color=bar_colors)
                rects2 = ax_bars.bar(x - 0.5*width, spearman_rho, width, label='Spearman ρ', 
                                    color=[(c[0], c[1], c[2], 0.7) for c in bar_colors])
                rects3 = ax_bars.bar(x + 0.5*width, spearman_p, width, label='Spearman p', 
                                    color=[(c[0], c[1], c[2], 0.4) for c in bar_colors])
                rects4 = ax_bars.bar(x + 1.5*width, mi_values, width, label='Mutual Info', 
                                    color=[(c[0], c[1], c[2], 0.9) for c in bar_colors])
                
                ax_bars.set_ylabel('Score / p-value / MI')
                ax_bars.set_title(f"Linear Regression Performance within Clusters", fontsize=14)
                ax_bars.set_xticks(x)
                ax_bars.set_xticklabels(cluster_labels)
                ax_bars.legend(fontsize='small')
                ax_bars.axhline(0, color='grey', linewidth=0.8)
                
                # Dynamic Y limits
                all_values = cv_r2 + spearman_rho + mi_values
                min_val = min([min(all_values, default=0), min(spearman_p, default=0)])
                max_val = max([max(all_values, default=0), max(spearman_p, default=0)])
                ax_bars.set_ylim(min(min_val * 1.1 if min_val < 0 else -0.1, -0.1),
                               max(1.05, max_val * 1.1 if max_val > 0 else 0.1))
                
                # Add bar labels
                def add_bar_labels(bars, format_str="{:.2f}"):
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax_bars.annotate(format_str.format(height),
                                          xy=(bar.get_x() + bar.get_width() / 2, height),
                                          xytext=(0, 3 if height >= 0 else -12),
                                          textcoords="offset points",
                                          ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
                
                add_bar_labels(rects1)
                add_bar_labels(rects2)
                add_bar_labels(rects3, format_str="{:.3f}")
                add_bar_labels(rects4, format_str="{:.3f}")
                ax_bars.grid(True, axis='y', linestyle='--', alpha=0.6)
                
                plt.tight_layout()
                plt.show()
        else:
            subset_y_pred = model.predict(subset_X)
            plot_regression_relationship(
                model_name=f"Linear Regression (Best Subset, {len(subset_X)} points)",
                X=subset_X,
                y=subset_y,
                y_pred=subset_y_pred,
                x_new=np.linspace(subset_X.min(), subset_X.max(), 100).reshape(-1, 1),
                y_new=model.predict(np.linspace(subset_X.min(), subset_X.max(), 100).reshape(-1, 1)),
                ax=axes[1],
                metrics={
                    'r2': r2_score(subset_y, subset_y_pred),
                    'mae': mean_absolute_error(subset_y, subset_y_pred),
                    'mse': mean_squared_error(subset_y, subset_y_pred),
                    'rmse': np.sqrt(mean_squared_error(subset_y, subset_y_pred))
                }
            )
    else:
        x_new = np.linspace(X_filtered.min(), X_filtered.max(), 100).reshape(-1, 1)
        y_new = model.predict(x_new)
        axes[1].scatter(X_filtered, y_filtered, alpha=0.6)
        axes[1].plot(x_new, y_new, 'r-', linewidth=2)
        axes[1].set_title("Linear Regression Original Data")
        axes[1].set_xlabel(feat1)
        axes[1].set_ylabel(feat2)
        axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    results = {
        'model': model,
        'metrics': metrics,
        'X': X_filtered,
        'y': y_filtered,
        'y_pred': y_pred,
        'best_subset': best_subset,
        'subset_results': subset_results,
        'cluster_models': cluster_models,
        'outlier_mask': outlier_mask,
        'outlier_scores': outlier_scores
    }
    
    print(f"Linear Regression Performance:")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    if find_subsets and subset_results:
        best_cluster_metrics = subset_results.get('cluster_metrics', [])
        if best_cluster_metrics:
            print("\nPer-Cluster Model Performance:")
            for cluster_info in best_cluster_metrics:
                cluster_id = cluster_info['label']
                size = cluster_info['size']
                r2 = cluster_info.get('cv_r2', 0)
                rmse = cluster_info.get('y_std', 0) * np.sqrt(1 - max(0, r2))
                mi = cluster_info.get('mi', 0)
                print(f"  Cluster {cluster_id} (size: {size}): R²={r2:.4f}, RMSE={rmse:.4f}, MI={mi:.4f}")
    
    return results
