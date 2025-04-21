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
from sklearn.metrics import mean_squared_error, r2_score, mutual_info_score
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

# Set a seed for reproducibility
np.random.seed(42)

def calculate_correlations(df, var1, var2):
    pearson_corr, pearson_p = stats.pearsonr(df[var1], df[var2])
    spearman_corr, spearman_p = stats.spearmanr(df[var1], df[var2])
    kendall_corr, kendall_p = stats.kendalltau(df[var1], df[var2])
    
    # Calculate mutual information
    # Reshape for mutual_info_regression
    X = df[var1].values.reshape(-1, 1)
    y = df[var2].values
    mi = mutual_info_regression(X, y, random_state=42)[0]
    
    results = {
        'Pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
        'Spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
        'Kendall': {'correlation': kendall_corr, 'p_value': kendall_p},
        'Mutual Information': {'score': mi}
    }
    
    return results

def print_correlation_results(results):
    for method, values in results.items():
        if method == 'Mutual Information':
            print(f"{method}: {values['score']:.4f}")
        else:
            corr = values['correlation']
            p_val = values['p_value']
            print(f"{method} correlation: {corr:.4f}, p-value: {p_val:.4f}")

def visualize_scatter(df, var1, var2, correlation_results=None):
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x=df[var1], y=df[var2])
    plt.xlabel(var1)
    plt.ylabel(var2)
    
    if correlation_results:
        title = f"Scatter plot of {var1} vs {var2}\n"
        for method, values in correlation_results.items():
            if method == 'Mutual Information':
                title += f"{method}: {values['score']:.4f}  "
            else:
                corr = values['correlation']
                title += f"{method}: {corr:.4f}  "
        plt.title(title)
    else:
        plt.title(f"Scatter plot of {var1} vs {var2}")
    
    plt.tight_layout()
    plt.show()
    
    # Visualize mutual information with conditional entropy
    if correlation_results and 'Mutual Information' in correlation_results:
        plt.figure(figsize=(10, 6))
        
        # Use a colormap to visualize mutual information contribution
        # Bin the x-axis and calculate MI for each bin
        n_bins = min(10, len(df) // 5)  # Ensure we have enough data in each bin
        bins = pd.cut(df[var1], bins=n_bins)
        
        # Fix the categories attribute error by getting the bin intervals
        if hasattr(bins, 'categories'):
            bin_intervals = bins.categories
        else:
            bin_intervals = bins.array.categories
            
        bin_centers = [(x.left + x.right)/2 for x in bin_intervals]
        
        mi_values = []
        for bin_name in bin_intervals:
            bin_data = df[bins == bin_name]
            if len(bin_data) > 2:  # Need at least 3 points for meaningful MI
                X_bin = bin_data[var1].values.reshape(-1, 1)
                y_bin = bin_data[var2].values
                try:
                    mi_bin = mutual_info_regression(X_bin, y_bin, random_state=42)[0]
                    mi_values.append(mi_bin)
                except:
                    mi_values.append(0)
            else:
                mi_values.append(0)
        
        # Plot bin-wise mutual information
        plt.bar(bin_centers, mi_values, width=(bin_centers[1]-bin_centers[0]) if len(bin_centers) > 1 else 1)
        plt.title(f"Mutual Information by {var1} Ranges")
        plt.xlabel(var1)
        plt.ylabel("Mutual Information")
        plt.tight_layout()
        plt.show()
        
        # Visualize data colored by contribution to mutual information
        if len(df) > 10:
            plt.figure(figsize=(10, 6))
            
            # Estimate point-wise contributions (this is an approximation)
            # For each point, how much does it contribute to the overall MI?
            X = df[var1].values.reshape(-1, 1)
            y = df[var2].values
            
            # Use KNN to estimate local density and contribution
            n_neighbors = min(5, len(df) - 1)
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            knn.fit(X, y)
            y_pred = knn.predict(X)
            
            # Points with high error contribute more to MI
            point_contribution = np.abs(y - y_pred)
            
            # Plot points colored by contribution
            plt.scatter(X, y, c=point_contribution, cmap='viridis', s=50, edgecolor='k')
            plt.colorbar(label='Estimated MI Contribution')
            plt.title(f"Data Points Colored by Mutual Information Contribution")
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.tight_layout()
            plt.show()

def visualize_joint_plot(df, var1, var2, kind='reg'):
    g = sns.jointplot(x=df[var1], y=df[var2], kind=kind)
    g.fig.suptitle(f"Joint plot of {var1} vs {var2}", y=1.02)
    plt.tight_layout()
    plt.show()

def visualize_correlation_heatmap(df, variables=None):
    if variables is None:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[variables].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def analyze_two_variables(df, var1, var2):
    results = calculate_correlations(df, var1, var2)
    print_correlation_results(results)
    
    visualize_scatter(df, var1, var2, results)
    visualize_joint_plot(df, var1, var2)
    
    return results

# ML Models for relationship discovery between two variables
def create_prediction_grid(X, padding=0.5, mesh_step_size=0.02):
    x_min, x_max = X.min() - padding, X.max() + padding
    xx = np.arange(x_min, x_max, mesh_step_size)
    return xx.reshape(-1, 1)

def plot_regression_relationship(model_name, X, y, y_pred, x_new, y_new, ax, metrics, errors=None, threshold=None):
    ax.scatter(X, y, color='blue', alpha=0.6, label='Actual data')
    ax.plot(x_new, y_new, color='red', linewidth=2, label='Predicted relationship')
    
    if threshold is not None:
        ax.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold: {threshold:.2f}')
    
    # Color points by error magnitude if errors are provided
    if errors is not None:
        error_norm = plt.Normalize(vmin=0, vmax=errors.max())
        scatter = ax.scatter(X, y, c=errors, cmap='YlOrRd', norm=error_norm, s=50, edgecolors='k', zorder=3)
        plt.colorbar(scatter, ax=ax, label='Prediction Error')
    
    ax.set_title(f"{model_name}\nR² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    ax.set_xlabel('Shower Time')
    ax.set_ylabel('LEP Time')
    ax.legend()
    
    return ax

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

def fit_decision_tree_regressor(df, shower_var, lep_var, max_depth=3, find_subsets=True, remove_outliers=True, outlier_method='influence'):
    X = df[shower_var].values.reshape(-1, 1)
    y = df[lep_var].values
    
    # Define model factory function for consistent model creation
    def dt_factory(X_train, y_train):
        return DecisionTreeRegressor(max_depth=max_depth, random_state=42).fit(X_train, y_train)
    
    # Detect and remove outliers if requested
    outlier_mask = np.ones(len(X), dtype=bool)  # Default: keep all points
    outlier_scores = np.zeros(len(X))
    
    if remove_outliers:
        outlier_mask, outlier_scores = detect_outliers(
            X, y, method=outlier_method, max_remove_percent=10, 
            model_factory=dt_factory
        )
    
    # Fit model on filtered data
    X_filtered = X[outlier_mask]
    y_filtered = y[outlier_mask]
    
    model = dt_factory(X_filtered, y_filtered)
    
    # Predictions on filtered data
    y_pred_filtered = model.predict(X_filtered)
    
    # Calculate errors and metrics on filtered data
    errors_filtered = np.abs(y_filtered - y_pred_filtered)
    
    metrics = {
        'r2': r2_score(y_filtered, y_pred_filtered),
        'rmse': np.sqrt(mean_squared_error(y_filtered, y_pred_filtered))
    }
    
    # Get decision rules
    tree_text = export_text(model, feature_names=[shower_var])
    print("Decision Tree Rules:")
    print(tree_text)
    
    # Visualize the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=[shower_var], rounded=True, fontsize=12)
    plt.title(f"Decision Tree (max_depth={max_depth})")
    plt.tight_layout()
    plt.show()
    
    # Create prediction grid for smooth curve
    x_new = create_prediction_grid(X.ravel())
    y_new = model.predict(x_new)
    
    # Plot predictions with errors highlighted and outliers marked
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # First plot removed outliers in grey
    if not np.all(outlier_mask):
        ax.scatter(X[~outlier_mask], y[~outlier_mask], color='grey', alpha=0.5, 
                  label=f'Outliers ({np.sum(~outlier_mask)} points)')
    
    # Plot kept points with color based on error
    scatter = ax.scatter(X_filtered, y_filtered, c=errors_filtered, cmap='YlOrRd', 
                       norm=plt.Normalize(vmin=0, vmax=errors_filtered.max()),
                       s=50, edgecolors='k', zorder=3)
    
    # Plot the model prediction curve
    ax.plot(x_new, y_new, color='red', linewidth=2, label='Predicted relationship')
    
    plt.colorbar(scatter, ax=ax, label='Prediction Error')
    ax.set_title(f"Decision Tree Regression\nR² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    ax.set_xlabel('Shower Time')
    ax.set_ylabel('LEP Time')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Find optimal data subsets if requested
    best_subset = None
    if find_subsets:
        # Use only the filtered data for subset discovery
        df_filtered = df.iloc[np.where(outlier_mask)[0]]
        best_subset = find_optimal_data_subsets(df_filtered, shower_var, lep_var, dt_factory, "Decision Tree")
    
    print(f"Decision Tree Regression Results:")
    print(f"R² score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    if not np.all(outlier_mask):
        print(f"Outliers removed: {np.sum(~outlier_mask)} points ({np.sum(~outlier_mask)/len(X)*100:.1f}%)")
    
    return model, metrics, best_subset, {'mask': outlier_mask, 'scores': outlier_scores}

def fit_gaussian_process_regressor(df, shower_var, lep_var, find_subsets=True, remove_outliers=True, outlier_method='influence'):
    X = df[shower_var].values.reshape(-1, 1)
    y = df[lep_var].values
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
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
    
    # Define a factory function to use for outlier detection
    def gp_factory(X_train, y_train):
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Use a simple RBF kernel for outlier detection to avoid overfitting
        kernel = 1.0 * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True, random_state=42)
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
    
    # Detect and remove outliers if requested
    outlier_mask = np.ones(len(X), dtype=bool)  # Default: keep all points
    outlier_scores = np.zeros(len(X))
    
    if remove_outliers:
        outlier_mask, outlier_scores = detect_outliers(
            X, y, method=outlier_method, max_remove_percent=10, 
            model_factory=gp_factory
        )
    
    # Use filtered data for model fitting
    X_filtered = X[outlier_mask]
    y_filtered = y[outlier_mask]
    X_filtered_scaled = scaler_X.transform(X_filtered)
    
    # Find best kernel on filtered data
    best_model = None
    best_score = -float('inf')
    best_kernel_name = ""
    best_length_scale = 1.0  # Default
    
    # Try all kernels on filtered data
    for kernel_name, kernel in kernels:
        model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True, random_state=42)
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
    
    # Plot predictions with uncertainties
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # First plot removed outliers in grey
    if not np.all(outlier_mask):
        ax.scatter(X[~outlier_mask], y[~outlier_mask], color='grey', alpha=0.5, 
                  label=f'Outliers ({np.sum(~outlier_mask)} points)')
    
    # Plot the prediction curve with uncertainty
    ax.plot(x_new, y_new, color='red', linewidth=2, label='Predicted relationship')
    ax.fill_between(x_new.ravel(), y_new - 2*y_std, y_new + 2*y_std, 
                   color='red', alpha=0.2, label='Uncertainty (±2σ)')
    
    # Plot kept points with color based on error
    scatter = ax.scatter(X_filtered, y_filtered, c=errors_filtered, cmap='YlOrRd', 
                       norm=plt.Normalize(vmin=0, vmax=errors_filtered.max()),
                       s=50, edgecolors='k', zorder=3)
    
    plt.colorbar(scatter, ax=ax, label='Prediction Error')
    ax.set_title(f"Gaussian Process Regression ({best_kernel_name})\nR² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    ax.set_xlabel('Shower Time')
    ax.set_ylabel('LEP Time')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Find optimal data subsets if requested
    best_subset = None
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
                
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True, random_state=42)
            gp.fit(X_train_scaled, y_train)
            
            # Create a wrapper that handles scaling internally
            class GPWrapper:
                def __init__(self, gp, scaler):
                    self.gp = gp
                    self.scaler = scaler
                
                def predict(self, X):
                    X_scaled = self.scaler.transform(X)
                    return self.gp.predict(X_scaled)
            
            return GPWrapper(gp, scaler)
            
        best_subset = find_optimal_data_subsets(df_filtered, shower_var, lep_var, gp_subset_factory, "Gaussian Process")
    
    print(f"Gaussian Process Regression Results (Kernel: {best_kernel_name}):")
    print(f"R² score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    if not np.all(outlier_mask):
        print(f"Outliers removed: {np.sum(~outlier_mask)} points ({np.sum(~outlier_mask)/len(X)*100:.1f}%)")
    
    return model, scaler_X, metrics, best_subset, {'mask': outlier_mask, 'scores': outlier_scores}

def fit_svr(df, shower_var, lep_var, find_subsets=True, remove_outliers=True, outlier_method='influence'):
    X = df[shower_var].values.reshape(-1, 1)
    y = df[lep_var].values
    
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
                X_scaled = self.scaler.transform(X)
                return self.svr.predict(X_scaled)
        
        return SVRWrapper(svr_model, scaler)
    
    # Detect and remove outliers if requested
    outlier_mask = np.ones(len(X), dtype=bool)  # Default: keep all points
    outlier_scores = np.zeros(len(X))
    
    if remove_outliers:
        outlier_mask, outlier_scores = detect_outliers(
            X, y, method=outlier_method, max_remove_percent=10, 
            model_factory=svr_factory
        )
    
    # Use filtered data for model fitting
    X_filtered = X[outlier_mask]
    y_filtered = y[outlier_mask]
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
        'rmse': np.sqrt(mean_squared_error(y_filtered, y_pred_filtered))
    }
    
    # Create prediction grid for smooth curve
    x_new = create_prediction_grid(X.ravel())
    x_new_scaled = scaler_X.transform(x_new)
    y_new = model.predict(x_new_scaled)
    
    # Plot predictions with errors highlighted
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # First plot removed outliers in grey
    if not np.all(outlier_mask):
        ax.scatter(X[~outlier_mask], y[~outlier_mask], color='grey', alpha=0.5, 
                  label=f'Outliers ({np.sum(~outlier_mask)} points)')
    
    # Plot the prediction curve
    ax.plot(x_new, y_new, color='red', linewidth=2, label='Predicted relationship')
    
    # Plot kept points with color based on error
    scatter = ax.scatter(X_filtered, y_filtered, c=errors_filtered, cmap='YlOrRd', 
                       norm=plt.Normalize(vmin=0, vmax=errors_filtered.max()),
                       s=50, edgecolors='k', zorder=3)
    
    plt.colorbar(scatter, ax=ax, label='Prediction Error')
    ax.set_title(f"Support Vector Regression ({kernel}, C={C})\nR² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    ax.set_xlabel('Shower Time')
    ax.set_ylabel('LEP Time')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Find optimal data subsets if requested
    best_subset = None
    if find_subsets:
        # Use only the filtered data for subset discovery
        df_filtered = df.iloc[np.where(outlier_mask)[0]]
        
        # Create a model factory for SVR with best parameters
        def svr_subset_factory(X_train, y_train):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            svr_model = SVR(kernel=kernel, C=C)
            svr_model.fit(X_train_scaled, y_train)
            
            # Create a wrapper that handles scaling internally
            class SVRWrapper:
                def __init__(self, svr, scaler):
                    self.svr = svr
                    self.scaler = scaler
                
                def predict(self, X):
                    X_scaled = self.scaler.transform(X)
                    return self.svr.predict(X_scaled)
            
            return SVRWrapper(svr_model, scaler)
            
        best_subset = find_optimal_data_subsets(df_filtered, shower_var, lep_var, svr_subset_factory, "SVR")
    
    print(f"SVR Results (kernel={kernel}, C={C}):")
    print(f"R² score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    if not np.all(outlier_mask):
        print(f"Outliers removed: {np.sum(~outlier_mask)} points ({np.sum(~outlier_mask)/len(X)*100:.1f}%)")
    
    return model, scaler_X, metrics, best_subset, {'mask': outlier_mask, 'scores': outlier_scores}

def fit_knn_regressor(df, shower_var, lep_var, find_subsets=True):
    X = df[shower_var].values.reshape(-1, 1)
    y = df[lep_var].values
    
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
        'rmse': np.sqrt(mean_squared_error(y, y_pred))
    }
    
    # Create prediction grid for smooth curve
    x_new = create_prediction_grid(X.ravel())
    x_new_scaled = scaler_X.transform(x_new)
    y_new = model.predict(x_new_scaled)
    
    # Plot predictions with errors highlighted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.6, label='Actual data')
    ax.plot(x_new, y_new, color='red', linewidth=2, label='Predicted relationship')
    
    # Color points by error magnitude
    error_norm = plt.Normalize(vmin=0, vmax=errors.max())
    scatter = ax.scatter(X, y, c=errors, cmap='YlOrRd', norm=error_norm, s=50, edgecolors='k', zorder=3)
    plt.colorbar(scatter, ax=ax, label='Prediction Error')
    
    ax.set_title(f"k-Nearest Neighbors (k={n_neighbors})\nR² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    ax.set_xlabel('Shower Time')
    ax.set_ylabel('LEP Time')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Find optimal data subsets if requested
    best_subset = None
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
                    X_scaled = self.scaler.transform(X)
                    return self.knn.predict(X_scaled)
            
            return KNNWrapper(knn_model, scaler)
            
        best_subset = find_optimal_data_subsets(df, shower_var, lep_var, knn_factory, "k-NN")
    
    print(f"k-NN Regression Results (k={n_neighbors}):")
    print(f"R² score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    return model, scaler_X, metrics, best_subset

def fit_random_forest_regressor(df, shower_var, lep_var, find_subsets=True):
    X = df[shower_var].values.reshape(-1, 1)
    y = df[lep_var].values
    
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
    
    # Plot predictions with errors highlighted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.6, label='Actual data')
    ax.plot(x_new, y_new, color='red', linewidth=2, label='Predicted relationship')
    
    # Color points by error magnitude
    error_norm = plt.Normalize(vmin=0, vmax=errors.max())
    scatter = ax.scatter(X, y, c=errors, cmap='YlOrRd', norm=error_norm, s=50, edgecolors='k', zorder=3)
    plt.colorbar(scatter, ax=ax, label='Prediction Error')
    
    ax.set_title(f"Random Forest (trees={n_estimators}, depth={max_depth})\nR² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    ax.set_xlabel('Shower Time')
    ax.set_ylabel('LEP Time')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Find optimal data subsets if requested
    best_subset = None
    if find_subsets:
        # Create a model factory for Random Forest
        def rf_factory(X_train, y_train):
            rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf_model.fit(X_train, y_train)
            return rf_model
            
        best_subset = find_optimal_data_subsets(df, shower_var, lep_var, rf_factory, "Random Forest")
    
    print(f"Random Forest Regression Results (trees={n_estimators}, depth={max_depth}):")
    print(f"R² score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    return model, metrics, best_subset

def compare_model_performance(models_results):
    model_names = []
    r2_scores = []
    rmse_scores = []
    thresholds = []
    
    for name, results in models_results.items():
        model_names.append(name)
        metrics = results['metrics']
        r2_scores.append(metrics['r2'])
        rmse_scores.append(metrics['rmse'])
        thresholds.append(results['threshold'] if 'threshold' in results else None)
    
    performance_df = pd.DataFrame({
        'Model': model_names,
        'R² Score': r2_scores,
        'RMSE': rmse_scores,
        'Threshold': thresholds
    })
    
    performance_df = performance_df.sort_values('R² Score', ascending=False)
    print("\nModel Performance Comparison:")
    print(performance_df)
    
    # Plot performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(x='Model', y='R² Score', data=performance_df, ax=ax1)
    ax1.set_title('R² Score Comparison (higher is better)')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    sns.barplot(x='Model', y='RMSE', data=performance_df, ax=ax2)
    ax2.set_title('RMSE Comparison (lower is better)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return performance_df

def find_optimal_data_subsets(df, shower_var, lep_var, model_factory, model_name, n_clusters=3, min_subset_size=10, use_2d_clustering=True):
    """
    Find data subsets where a model performs particularly well using clustering and local validation.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    shower_var : str
        Column name for shower time
    lep_var : str
        Column name for LEP time
    model_factory : function
        A function that takes X and y and returns a fitted model
    model_name : str
        Name of the model for display purposes
    n_clusters : int
        Number of clusters to try
    min_subset_size : int
        Minimum number of points required in a subset
    use_2d_clustering : bool
        Whether to cluster based on both X and Y (True) or just X (False)
        
    Returns:
    --------
    dict with performance metrics for the best subset
    """
    X_1d = df[shower_var].values.reshape(-1, 1)
    y = df[lep_var].values
    
    # For 2D clustering, include both variables
    if use_2d_clustering:
        X_2d = np.column_stack((df[shower_var].values, df[lep_var].values))
        print(f"Using 2D clustering on both {shower_var} and {lep_var}")
    else:
        X_2d = X_1d
        print(f"Using 1D clustering on {shower_var} only")
    
    print(f"Finding optimal subsets for {model_name}...")
    
    # Try different clustering approaches
    clustering_methods = []
    
    # 1. K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_2d)
    clustering_methods.append(("K-Means", cluster_labels))
    
    # 2. DBSCAN for density-based clustering (finds outliers too)
    # Estimate eps parameter based on nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(5, len(X_2d)-1)).fit(X_2d)
    distances, _ = nn.kneighbors(X_2d)
    distances = np.sort(distances[:, min(3, len(X_2d)-2)])  # Distances to 3rd nearest neighbor
    eps = np.percentile(distances, 80)  # Try using 80th percentile as eps
    
    dbscan = DBSCAN(eps=eps, min_samples=min(5, len(X_2d)-1))
    dbscan_labels = dbscan.fit_predict(X_2d)
    clustering_methods.append(("DBSCAN", dbscan_labels))
    
    # 3. Gaussian Mixture Model (GMM) with fuzzy assignment
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(X_2d)
    
    # Get both hard labels and probabilities
    gmm_labels = gmm.predict(X_2d)
    gmm_probs = gmm.predict_proba(X_2d)
    
    clustering_methods.append(("GMM", gmm_labels))
    
    # 4. Simple quantile-based clustering
    n_bins = min(5, len(X_1d) // min_subset_size)
    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    quantiles = kb.fit_transform(X_1d).astype(int).flatten()
    clustering_methods.append(("Quantiles", quantiles))
    
    # Evaluate model performance in each cluster
    best_score = -float('inf')
    best_subset = None
    best_performance = {}
    best_method = None
    
    # Create a discrete colormap for the clusters
    n_max_clusters = max([len(np.unique(labels)) for _, labels in clustering_methods])
    distinct_colors = plt.cm.tab20(np.linspace(0, 1, max(n_max_clusters, n_clusters)))
    
    # Create a more compact multi-panel figure
    n_methods = len(clustering_methods)
    n_cols = min(2, n_methods)  # 2 methods per row for better visualization
    n_rows = (n_methods + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=(14, 4*n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols*2)  # Double the columns for scatter+bar pairs
    
    debug_info = {}  # Store debugging information
    
    for i, (method_name, labels) in enumerate(clustering_methods):
        # Calculate row and column for this method
        row = i // n_cols
        col = i % n_cols
        
        # Skip if only one cluster or all points are noise (-1)
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
            print(f"WARNING: {method_name} produced only {len(unique_labels)} unique clusters. Skipping.")
            continue
        
        # Create a colormap that matches the cluster labels
        if method_name == "DBSCAN":
            # Special case for DBSCAN that might have -1 for noise
            cmap = ListedColormap(['gray'] + [distinct_colors[i] for i in range(len(unique_labels)-1)]) if -1 in unique_labels else ListedColormap([distinct_colors[i] for i in range(len(unique_labels))])
            norm = BoundaryNorm(np.arange(-1.5, len(unique_labels)-0.5, 1), cmap.N)
        else:
            cmap = ListedColormap([distinct_colors[i] for i in range(len(unique_labels))])
            norm = BoundaryNorm(np.arange(-0.5, len(unique_labels)+0.5, 1), cmap.N)
        
        # Create axes for this method's scatter plot
        ax1 = plt.subplot(gs[row, col*2])
        ax2 = plt.subplot(gs[row, col*2+1])
        
        # Use special coloring for GMM to show fuzzy assignments
        if method_name == "GMM":
            # Blend colors based on membership probabilities
            blended_colors = np.zeros((len(X_1d), 3))
            for point_idx in range(len(X_1d)):
                for cluster_idx in range(n_clusters):
                    if cluster_idx < len(distinct_colors):
                        weight = gmm_probs[point_idx, cluster_idx]
                        color = distinct_colors[cluster_idx][:3]  # Get RGB without alpha
                        blended_colors[point_idx] += weight * np.array(color)
            
            # Plot the clusters with blended colors
            scatter = ax1.scatter(X_1d, y, c=blended_colors, s=50, edgecolor='k')
            
            # Draw ellipses to show GMM components
            for j in range(n_clusters):
                if use_2d_clustering:
                    # For 2D, draw the actual covariance ellipses
                    mean = gmm.means_[j]
                    covar = gmm.covariances_[j]
                    v, w = np.linalg.eigh(covar)
                    angle = np.arctan2(w[1, 0], w[0, 0])
                    angle = 180 * angle / np.pi  # Convert to degrees
                    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                    
                    ell = Ellipse(xy=(mean[0], mean[1]),
                                  width=v[0], height=v[1],
                                  angle=angle, 
                                  color=distinct_colors[j])
                    ell.set_clip_box(ax1.bbox)
                    ell.set_alpha(0.3)
                    ax1.add_artist(ell)
                else:
                    # For 1D, just draw vertical lines at means
                    ax1.axvline(x=gmm.means_[j][0], color=distinct_colors[j], linestyle='--', alpha=0.7)
        else:
            # Plot with discrete colors for other methods
            scatter = ax1.scatter(X_1d, y, c=labels, cmap=cmap, norm=norm, s=50, edgecolor='k')
        
        ax1.set_title(f"Clusters using {method_name}")
        ax1.set_xlabel(shower_var)
        ax1.set_ylabel(lep_var)
        
        # Add legend for clusters
        if method_name == "GMM":
            legend_elements = [Patch(facecolor=distinct_colors[i], edgecolor='black', alpha=0.7, 
                                     label=f'Cluster {i} (n={np.sum(labels==i)})')
                              for i in range(n_clusters)]
            ax1.legend(handles=legend_elements, loc="best", fontsize='small')
        else:
            legend1 = ax1.legend(*scatter.legend_elements(), title="Clusters", loc="best", fontsize='small')
            ax1.add_artist(legend1)
        
        # Set up for bar chart
        cluster_r2 = []
        cluster_spearman = []  # Add Spearman's correlation
        cluster_mi = []        # Add Mutual Information
        cluster_sizes = []
        cluster_names = []
        cluster_colors = []
        
        debug_info[method_name] = {}
        
        # Evaluate each cluster
        for idx, label in enumerate(unique_labels):
            if label == -1:  # DBSCAN noise points
                continue
                
            mask = labels == label
            if np.sum(mask) < min_subset_size:  # Skip clusters with too few points
                continue
                
            X_cluster = X_1d[mask]
            y_cluster = y[mask]
            
            # Save diagnostics
            cluster_info = {
                'size': int(np.sum(mask)),
                'X_range': (float(X_cluster.min()), float(X_cluster.max())),
                'y_range': (float(y_cluster.min()), float(y_cluster.max())),
                'X_std': float(X_cluster.std()),
                'y_std': float(y_cluster.std())
            }
            
            # Debug: Print cluster stats
            print(f"\n{method_name} - Cluster {label}:")
            print(f"  Size: {cluster_info['size']} points")
            print(f"  {shower_var} range: {cluster_info['X_range'][0]:.2f} to {cluster_info['X_range'][1]:.2f} (std={cluster_info['X_std']:.2f})")
            print(f"  {lep_var} range: {cluster_info['y_range'][0]:.2f} to {cluster_info['y_range'][1]:.2f} (std={cluster_info['y_std']:.2f})")
            
            # Use cross-validation if enough data
            if len(X_cluster) >= min_subset_size:
                cv = min(5, len(X_cluster))
                kf = KFold(n_splits=cv, shuffle=True, random_state=42)
                cv_scores = []
                
                for train_idx, test_idx in kf.split(X_cluster):
                    X_train, X_test = X_cluster[train_idx], X_cluster[test_idx]
                    y_train, y_test = y_cluster[train_idx], y_cluster[test_idx]
                    
                    # Fit model on this fold
                    try:
                        model = model_factory(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Calculate score
                        score = r2_score(y_test, y_pred)
                        cv_scores.append(score)
                        cluster_info[f'fold_scores'] = [float(s) for s in cv_scores]
                    except Exception as e:
                        print(f"  ERROR in CV fold: {str(e)}")
                        cluster_info['error'] = str(e)
                
                if cv_scores:
                    avg_score = np.mean(cv_scores)
                    cluster_info['avg_r2'] = float(avg_score)
                    print(f"  CV R² Score: {avg_score:.4f}")
                else:
                    avg_score = -float('inf')
                    cluster_info['avg_r2'] = float(avg_score)
                    print("  Failed to calculate CV R² score")
            else:
                # For small clusters, use the full dataset but don't consider for best subset
                try:
                    model = model_factory(X_cluster, y_cluster)
                    y_pred = model.predict(X_cluster)
                    avg_score = r2_score(y_cluster, y_pred)
                    cluster_info['avg_r2'] = float(avg_score)
                    print(f"  R² Score: {avg_score:.4f} (not cross-validated)")
                except Exception as e:
                    avg_score = -float('inf')
                    cluster_info['avg_r2'] = float(avg_score)
                    cluster_info['error'] = str(e)
                    print(f"  ERROR: {str(e)}")
            
            # Calculate Spearman correlation
            try:
                spearman = stats.spearmanr(X_cluster.flatten(), y_cluster).correlation
                cluster_info['spearman'] = float(spearman)
                print(f"  Spearman's correlation: {spearman:.4f}")
            except Exception as e:
                spearman = 0
                cluster_info['spearman_error'] = str(e)
                print(f"  ERROR calculating Spearman: {str(e)}")
            
            # Calculate Mutual Information
            try:
                # Bin the data to estimate MI (continuous MI is challenging)
                X_binned = KBinsDiscretizer(n_bins=min(10, len(X_cluster)//5), encode='ordinal', strategy='uniform').fit_transform(X_cluster)
                y_binned = KBinsDiscretizer(n_bins=min(10, len(y_cluster)//5), encode='ordinal', strategy='uniform').fit_transform(y_cluster.reshape(-1, 1))
                mi = mutual_info_score(X_binned.flatten(), y_binned.flatten())
                cluster_info['mi'] = float(mi)
                print(f"  Mutual Information: {mi:.4f}")
            except Exception as e:
                mi = 0
                cluster_info['mi_error'] = str(e)
                print(f"  ERROR calculating MI: {str(e)}")
            
            # Get color for this cluster (same as scatter plot)
            if method_name == "DBSCAN" and label == -1:
                color = "gray"
            else:
                color = distinct_colors[label]
            
            # Save for bar chart
            cluster_r2.append(avg_score)
            cluster_spearman.append(spearman)
            cluster_mi.append(mi)
            cluster_sizes.append(np.sum(mask))
            cluster_names.append(f"{label}\n(n={np.sum(mask)})")
            cluster_colors.append(color)
            
            # Store in debug info
            debug_info[method_name][f'cluster_{label}'] = cluster_info
            
            # Track best performing cluster
            if avg_score > best_score and np.sum(mask) >= min_subset_size:
                best_score = avg_score
                best_subset = mask
                best_method = method_name
                
                # Calculate additional metrics for this subset
                model = model_factory(X_cluster, y_cluster)
                y_pred = model.predict(X_cluster)
                rmse = np.sqrt(mean_squared_error(y_cluster, y_pred))
                
                # Calculate boundary of this subset
                x_min, x_max = X_cluster.min(), X_cluster.max()
                
                best_performance = {
                    'method': method_name,
                    'cluster': label,
                    'r2': avg_score,
                    'rmse': rmse,
                    'spearman': spearman,
                    'mi': mi,
                    'size': np.sum(mask),
                    'x_range': (x_min, x_max),
                    'subset_mask': mask
                }
        
        # Plot bar chart of cluster performance with matching colors
        x = np.arange(len(cluster_names))
        width = 0.25  # width of the bars
        
        # Plot R², Spearman, and MI side by side
        bars1 = ax2.bar(x - width, cluster_r2, width, color=cluster_colors, alpha=0.7, label='R²')
        bars2 = ax2.bar(x, cluster_spearman, width, color=cluster_colors, alpha=0.5, hatch='///', label='Spearman')
        bars3 = ax2.bar(x + width, cluster_mi, width, color=cluster_colors, alpha=0.5, hatch='...', label='MI')
        
        ax2.set_title(f"{method_name}: Performance by Cluster")
        ax2.set_xticks(x)
        ax2.set_xticklabels(cluster_names)
        ax2.set_ylim(-0.1, 1.0)
        ax2.axhline(y=0, color='r', linestyle='-')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        ax2.legend()
        
        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height) and height != 0:
                    value_text = f"{height:.2f}" if abs(height) < 10 else f"{int(height)}"
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            value_text, ha='center', va='bottom', rotation=90, fontsize=7)
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
    
    plt.tight_layout()
    plt.show()
    
    # Save debug info to file
    import json
    with open('cluster_debug_info.json', 'w') as f:
        json.dump(debug_info, f, indent=2)
    print("Debugging information saved to 'cluster_debug_info.json'")
    
    # If we found a good subset, visualize it with decision boundaries
    if best_subset is not None:
        X_best = X_1d[best_subset]
        y_best = y[best_subset]
        
        model = model_factory(X_best, y_best)
        
        # Create a more detailed visualization for decision boundaries
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Original data with subset highlighted
        plt.subplot(2, 1, 1)
        plt.scatter(X_1d, y, color='lightgray', label='All data')
        plt.scatter(X_best, y_best, color='blue', label=f'Best subset (n={len(X_best)})')
        
        # Sort points for line plotting
        sort_idx = np.argsort(X_best.flatten())
        X_sorted = X_best[sort_idx]
        
        # Generate a finer grid for smoother prediction line
        X_fine = np.linspace(X_sorted.min(), X_sorted.max(), 100).reshape(-1, 1)
        try:
            y_pred_fine = model.predict(X_fine)
            plt.plot(X_fine, y_pred_fine, 'r-', linewidth=2, label='Model prediction')
        except Exception as e:
            print(f"Error generating prediction line: {str(e)}")
            y_pred_sorted = model.predict(X_sorted)
            plt.plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='Model prediction')
        
        plt.title(f"Best Subset for {model_name} (using {best_method})")
        plt.xlabel(shower_var)
        plt.ylabel(lep_var)
        plt.legend()
        
        # Plot 2: Model visualization with uncertainty (if possible)
        plt.subplot(2, 1, 2)
        
        # Set range with padding
        x_range = X_best.max() - X_best.min()
        x_min, x_max = X_best.min() - 0.1 * x_range, X_best.max() + 0.1 * x_range
        
        # Create meshgrid for detailed visualization
        xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        
        # Try to get prediction with std deviation if model supports it
        try:
            if hasattr(model, 'predict_with_std') or (hasattr(model, 'predict') and model.__class__.__name__ == 'GaussianProcessRegressor'):
                if hasattr(model, 'predict_with_std'):
                    y_pred, y_std = model.predict_with_std(xx)
                else:
                    y_pred, y_std = model.predict(xx, return_std=True)
                
                # Plot mean prediction
                plt.plot(xx, y_pred, 'r-', lw=2, label='Prediction')
                
                # Plot uncertainty bands
                plt.fill_between(xx.ravel(), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std,
                                alpha=0.2, color='r', label='95% confidence interval')
            else:
                # For tree-based models, show decision boundaries if possible
                if hasattr(model, 'tree_') or (hasattr(model, 'estimators_') and len(model.estimators_) > 0):
                    y_pred = model.predict(xx)
                    plt.plot(xx, y_pred, 'r-', lw=2, label='Prediction')
                    
                    # For decision trees, show the actual decision boundaries
                    if hasattr(model, 'tree_'):
                        # Find unique decision thresholds
                        thresholds = []
                        tree = model.tree_
                        for i in range(tree.node_count):
                            if tree.children_left[i] != tree.children_right[i]:  # it's a split node
                                threshold = tree.threshold[i]
                                if tree.feature[i] == 0:  # Only if it's our feature
                                    thresholds.append(threshold)
                        
                        # Plot vertical lines at split points
                        for threshold in np.unique(thresholds):
                            plt.axvline(x=threshold, color='g', linestyle='--', alpha=0.5)
                            plt.text(threshold, plt.ylim()[0], f"{threshold:.2f}", 
                                   rotation=90, verticalalignment='bottom', color='g')
                else:
                    # Generic approach for any model
                    y_pred = model.predict(xx)
                    plt.plot(xx, y_pred, 'r-', lw=2, label='Prediction')
        except Exception as e:
            print(f"Error visualizing model: {str(e)}")
            # Fallback to simple visualization
            y_pred = model.predict(xx)
            plt.plot(xx, y_pred, 'r-', lw=2, label='Prediction')
        
        plt.scatter(X_best, y_best, c='blue', alpha=0.6, label='Data points')
        plt.title(f"Model Details - R² = {best_performance['r2']:.4f}, RMSE = {best_performance['rmse']:.4f}")
        plt.xlabel(shower_var)
        plt.ylabel(lep_var)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nBest subset found using {best_method}:")
        print(f"R² score: {best_performance['r2']:.4f}")
        print(f"RMSE: {best_performance['rmse']:.4f}")
        print(f"Spearman's correlation: {best_performance['spearman']:.4f}")
        print(f"Mutual Information: {best_performance['mi']:.4f}")
        print(f"Subset size: {best_performance['size']} points ({best_performance['size']/len(X_1d)*100:.1f}% of data)")
        print(f"{shower_var} range: {best_performance['x_range'][0]:.2f} to {best_performance['x_range'][1]:.2f}")
    else:
        print("No good subsets found with the minimum required size.")
    
    return best_performance if best_subset is not None else None

def detect_outliers(X, y, method='residual', max_remove_percent=10, model_factory=None):
    """
    Detect outliers in the data that if removed would improve model performance.
    
    Parameters:
    -----------
    X : array-like
        Feature values
    y : array-like
        Target values
    method : str
        Method to detect outliers: 'residual', 'distance', 'influence'
    max_remove_percent : float
        Maximum percentage of points to remove (0-100)
    model_factory : function
        Function that takes X and y and returns a fitted model
        
    Returns:
    --------
    mask : boolean array
        Mask for non-outlier points (True = keep, False = outlier)
    outlier_scores : array
        Scores for each point indicating how much of an outlier it is
    """
    n = len(X)
    max_remove = int(n * max_remove_percent / 100)
    
    if max_remove < 1:
        # Nothing to remove
        return np.ones(n, dtype=bool), np.zeros(n)
    
    # Default model if none provided
    if model_factory is None:
        def model_factory(X_train, y_train):
            model = KNeighborsRegressor(n_neighbors=min(5, len(X_train)-1))
            model.fit(X_train, y_train)
            return model
    
    # Initial model on all data
    model = model_factory(X, y)
    y_pred = model.predict(X)
    initial_r2 = r2_score(y, y_pred)
    
    # Calculate outlier scores based on selected method
    if method == 'residual':
        # Simple method: largest residuals
        outlier_scores = np.abs(y - y_pred)
    
    elif method == 'distance':
        # Distance-based method for X-space outliers
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(5, n-1)).fit(X)
        distances, _ = nbrs.kneighbors(X)
        outlier_scores = distances.mean(axis=1)
        
    elif method == 'influence':
        # Influence-based: how much does removing each point improve the model
        outlier_scores = np.zeros(n)
        
        for i in range(n):
            # Create a mask excluding this point
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            
            # Fit model without this point
            leave_one_out_model = model_factory(X[mask], y[mask])
            y_loo_pred = leave_one_out_model.predict(X[mask])
            loo_r2 = r2_score(y[mask], y_loo_pred)
            
            # Improvement in R² is the outlier score
            outlier_scores[i] = max(0, loo_r2 - initial_r2)
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    # Sort points by outlier score
    sorted_idx = np.argsort(outlier_scores)[::-1]  # Descending order
    
    # Try removing points one by one until we reach max_remove
    best_r2 = initial_r2
    best_mask = np.ones(n, dtype=bool)
    best_n_removed = 0
    
    for i in range(1, min(max_remove + 1, n)):
        # Create mask with top i outliers removed
        mask = np.ones(n, dtype=bool)
        mask[sorted_idx[:i]] = False
        
        # Skip if removing too many points
        if np.sum(mask) < 3:  # Need at least 3 points for meaningful model
            break
            
        # Fit model on remaining points
        model_i = model_factory(X[mask], y[mask])
        y_pred_i = model_i.predict(X[mask])
        r2_i = r2_score(y[mask], y_pred_i)
        
        # Keep track of best model
        if r2_i > best_r2:
            best_r2 = r2_i
            best_mask = mask.copy()
            best_n_removed = i
    
    # If no improvement was found, keep all points
    if best_n_removed == 0:
        return np.ones(n, dtype=bool), outlier_scores
    
    print(f"Outlier removal: removed {best_n_removed} points ({best_n_removed/n*100:.1f}%), R² improved from {initial_r2:.4f} to {best_r2:.4f}")
    return best_mask, outlier_scores

# Example usage:
# df = pd.read_csv('your_data.csv')
# 
# # Analyzing correlations between shower_time and lep_time including Mutual Information
# results = calculate_correlations(df, 'shower_time', 'lep_time')
# print_correlation_results(results)
# visualize_scatter(df, 'shower_time', 'lep_time', results)
# visualize_joint_plot(df, 'shower_time', 'lep_time')
# 
# # Running models with automatic outlier removal and subset discovery
# dt_results = fit_decision_tree_regressor(df, 'shower_time', 'lep_time', remove_outliers=True)
# gp_results = fit_gaussian_process_regressor(df, 'shower_time', 'lep_time', remove_outliers=True)
# svr_results = fit_svr(df, 'shower_time', 'lep_time', remove_outliers=True)
# 
# # Compare results table
# models_results = {
#     'Decision Tree': {'metrics': dt_results[1], 'subset': dt_results[2], 'outliers': dt_results[3]},
#     'Gaussian Process': {'metrics': gp_results[2], 'subset': gp_results[3], 'outliers': gp_results[4]},
#     'SVR': {'metrics': svr_results[2], 'subset': svr_results[3], 'outliers': svr_results[4]},
# }
# 
# # Create a table of metrics for comparison
# metrics_df = pd.DataFrame({
#     'Model': list(models_results.keys()),
#     'R²': [m['metrics']['r2'] for m in models_results.values()],
#     'RMSE': [m['metrics']['rmse'] for m in models_results.values()],
#     'Outliers Removed': [np.sum(~m['outliers']['mask']) if 'outliers' in m else 0 for m in models_results.values()],
#     'Best Subset R²': [m['subset']['r2'] if m['subset'] is not None else None for m in models_results.values()],
#     'Best Subset Size': [m['subset']['size'] if m['subset'] is not None else None for m in models_results.values()]
# })
# 
# print("\nModel Performance Comparison:")
# print(metrics_df.sort_values('R²', ascending=False))
