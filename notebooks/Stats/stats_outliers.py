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
from sklearn.decomposition import PCA

# Set a seed for reproducibility
np.random.seed(42)


def windorize_outliers(X, y, lower_percentile=0.005, upper_percentile=0.995):
    """
    Remove outliers in the data based on percentile thresholds for each feature and target.
    
    Parameters:
    -----------
    X : array-like
        Feature values
    y : array-like
        Target values
    lower_percentile : float
        Lower percentile threshold for windsorizing (0.0-1.0)
    upper_percentile : float
        Upper percentile threshold for windsorizing (0.0-1.0)
        
    Returns:
    --------
    X_clean : array
        Filtered X data with outliers removed
    y_clean : array
        Filtered y data with outliers removed
    mask : boolean array
        Mask for non-outlier points (True = keep, False = outlier)
    removal_reasons : array
        Array indicating which points were removed (0=kept, 1=windsorized)
    """
    n = len(X)
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Initialize mask (True = keep point)
    mask = np.ones(n, dtype=bool)
    
    # Initialize removal reasons array (0=kept, 1=windsorized)
    removal_reasons = np.zeros(n, dtype=int)
    
    # For each feature, apply percentile-based removal
    for j in range(X.shape[1]):
        # Calculate percentile thresholds
        lower_bound = np.percentile(X[:, j], lower_percentile * 100)
        upper_bound = np.percentile(X[:, j], upper_percentile * 100)
        
        # Find points outside the thresholds
        feature_outliers = (X[:, j] <= lower_bound) | (X[:, j] >= upper_bound)
        
        # Update mask and removal reasons
        removal_reasons[feature_outliers & mask] = 1  # Mark as windsorized
        mask[feature_outliers] = False
    
    # Apply windsorizing to target variable y as well
    lower_bound_y = np.percentile(y, lower_percentile * 100)
    upper_bound_y = np.percentile(y, upper_percentile * 100)
    y_outliers = (y <= lower_bound_y) | (y >= upper_bound_y)
    removal_reasons[y_outliers & mask] = 1  # Mark as windsorized
    mask[y_outliers] = False
    
    windsorized_count = np.sum(~mask)
    if windsorized_count > 0:
        print(f"Windsorizing: removed {windsorized_count} points ({windsorized_count/n*100:.1f}%) outside of percentile range {lower_percentile:.4f}-{upper_percentile:.4f}")
    
    return X[mask], y[mask], mask, removal_reasons


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
        Function that takes X and y and returns a fitted model. If None, no outliers are removed.
        
    Returns:
    --------
    X_clean : array
        Filtered X data with outliers removed
    y_clean : array
        Filtered y data with outliers removed
    mask : boolean array
        Mask for non-outlier points (True = keep, False = outlier)
    outlier_scores : array
        Scores for each point indicating how much of an outlier it is
    removal_reasons : array
        Array indicating why each point was removed (0=kept, 2=model improvement)
    """
    n = len(X)
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Initialize mask (True = keep point)
    mask = np.ones(n, dtype=bool)
    
    # Initialize removal reasons array (0=kept, 2=model improvement)
    removal_reasons = np.zeros(n, dtype=int)
    
    # Initialize outlier scores
    outlier_scores = np.zeros(n)
    
    # If no model_factory is provided, don't remove any outliers
    if model_factory is None:
        print("No model_factory provided. No outliers will be removed based on model improvement.")
        return X, y, mask, outlier_scores, removal_reasons
    
    # Calculate max points to remove in model-based detection
    max_remove = int(n * max_remove_percent / 100)
    
    if max_remove < 1 or len(X) < 3:
        # Nothing to remove or too few points
        return X, y, mask, np.zeros(n), removal_reasons
    
    # Initial model on data
    model = model_factory(X, y)
    y_pred_full = model.predict(X)
    initial_r2 = r2_score(y, y_pred_full)
    
    # Calculate outlier scores based on selected method
    if method == 'residual':
        # Simple method: largest residuals
        outlier_scores = np.abs(y - y_pred_full)
    
    elif method == 'distance':
        # Distance-based method for X-space outliers
        from sklearn.neighbors import NearestNeighbors
        if len(X) >= 2:  # Need at least 2 points for neighbors
            nbrs = NearestNeighbors(n_neighbors=min(5, len(X)-1)).fit(X)
            distances, _ = nbrs.kneighbors(X)
            outlier_scores = distances.mean(axis=1)
        
    elif method == 'influence':
        # Influence-based: how much does removing each point improve the model
        for i in range(n):
            # Create a submask excluding this point
            submask = np.ones(n, dtype=bool)
            submask[i] = False
            
            # Need at least 3 points for a meaningful model
            if np.sum(submask) < 3:
                continue
            
            # Fit model without this point
            leave_one_out_model = model_factory(X[submask], y[submask])
            y_loo_pred = leave_one_out_model.predict(X[submask])
            loo_r2 = r2_score(y[submask], y_loo_pred)
            
            # Improvement in R² is the outlier score
            outlier_scores[i] = max(0, loo_r2 - initial_r2)
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    # Sort points by outlier score
    sorted_indices = np.argsort(outlier_scores)[::-1]
    
    # Try removing points one by one until we reach max_remove
    best_r2 = initial_r2
    best_mask = mask.copy()
    best_n_removed = 0
    
    for i in range(1, min(max_remove + 1, n + 1)):
        # Create mask with top i outliers removed
        test_mask = np.ones(n, dtype=bool)
        test_mask[sorted_indices[:i]] = False
        
        # Skip if removing too many points
        if np.sum(test_mask) < 3:  # Need at least 3 points for meaningful model
            break
            
        # Fit model on remaining points
        model_i = model_factory(X[test_mask], y[test_mask])
        y_pred_i = model_i.predict(X[test_mask])
        r2_i = r2_score(y[test_mask], y_pred_i)
        
        # Keep track of best model
        if r2_i > best_r2:
            best_r2 = r2_i
            best_mask = test_mask.copy()
            best_n_removed = i
    
    # If improvement was found, update mask and removal reasons
    if best_n_removed > 0:
        model_removed_idx = sorted_indices[:best_n_removed]
        removal_reasons[model_removed_idx] = 2  # Mark as removed for model improvement
        mask = best_mask
        print(f"Model-based removal: removed {best_n_removed} points ({best_n_removed/n*100:.1f}%), R² improved from {initial_r2:.4f} to {best_r2:.4f}")
    
    total_removed = np.sum(~mask)    
    print(f"Total removed for model improvement: {total_removed} points ({total_removed/n*100:.1f}%)")
    
    return X[mask], y[mask], mask, outlier_scores, removal_reasons


def plot_outlier_detection(X, y, mask, outlier_scores, method='residual', removal_reasons=None, fig_size=(12, 10)):
    """
    Visualize the outlier detection results with multiple plots.
    
    Parameters:
    -----------
    X : array-like
        Original feature values
    y : array-like
        Original target values
    mask : boolean array
        Mask for non-outlier points (True = keep, False = outlier)
    outlier_scores : array
        Scores for each point indicating how much of an outlier it is
    method : str
        Method used for outlier detection
    removal_reasons : array or None
        Array indicating why each point was removed (0=kept, 1=windsorized, 2=model improvement)
    fig_size : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    fig : matplotlib figure
        The figure containing the plots
    """
    # Convert inputs to numpy arrays if they aren't already
    X = np.asarray(X)
    y = np.asarray(y)
    mask = np.asarray(mask)
    outlier_scores = np.asarray(outlier_scores)
    
    # Handle removal_reasons if not provided (backward compatibility)
    if removal_reasons is None:
        removal_reasons = np.zeros(len(mask), dtype=int)
        removal_reasons[~mask] = 2  # Default: all removed points are for model improvement
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])
    
    # Determine point sizes based on outlier scores (normalized)
    if outlier_scores.max() > 0:
        point_sizes = 20 + 100 * (outlier_scores / outlier_scores.max())
    else:
        point_sizes = np.ones_like(outlier_scores) * 30
    
    # Determine colors for points: kept, windsorized, or model-improved
    color_map = {
        0: '#1f77b4',  # blue for kept points
        1: '#ff7f0e',  # orange for windsorized points
        2: '#d62728'   # red for model improvement points
    }
    colors = np.array([color_map[reason] for reason in removal_reasons])
    
    # For high-dimensional data, use PCA to visualize in 2D
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_
        
        # Plot in feature space (PCA)
        ax1 = fig.add_subplot(gs[0, 0])
        sc1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], s=point_sizes, c=colors, alpha=0.7)
        ax1.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        ax1.set_title('Feature Space (PCA projection)')
    else:
        # For 1D or 2D data, plot directly
        ax1 = fig.add_subplot(gs[0, 0])
        if X.shape[1] == 1:
            sc1 = ax1.scatter(X[:, 0], y, s=point_sizes, c=colors, alpha=0.7)
            ax1.set_xlabel('Feature')
            ax1.set_ylabel('Target')
        else:  # 2D
            sc1 = ax1.scatter(X[:, 0], X[:, 1], s=point_sizes, c=colors, alpha=0.7)
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
        ax1.set_title('Feature Space')
    
    # Plot target vs predictions (if method is residual)
    if method == 'residual':
        # Create a model to get predictions (similar to detect_outliers function)
        model = KNeighborsRegressor(n_neighbors=min(5, len(X[mask])-1))
        model.fit(X[mask], y[mask])
        y_pred = model.predict(X)
        
        ax2 = fig.add_subplot(gs[0, 1])
        sc2 = ax2.scatter(y, y_pred, s=point_sizes, c=colors, alpha=0.7)
        
        # Add diagonal line for perfect predictions
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Actual vs Predicted')

    # Plot histogram of outlier scores with threshold
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Get threshold value (minimum score of removed points for model improvement)
    model_improved_idx = (removal_reasons == 2)
    if np.any(model_improved_idx):
        threshold = min(outlier_scores[model_improved_idx])
        # Create histogram
        hist_bins = min(50, len(outlier_scores) // 5 + 1)
        ax3.hist(outlier_scores, bins=hist_bins, alpha=0.7, color='#1f77b4')
        ax3.axvline(x=threshold, color='#d62728', linestyle='--')
        ax3.text(threshold * 1.05, ax3.get_ylim()[1] * 0.9, 'Model Threshold', 
                 color='#d62728', va='top', ha='left')
    else:
        # If no model improvement points, just show histogram
        hist_bins = min(50, len(outlier_scores) // 5 + 1)
        ax3.hist(outlier_scores, bins=hist_bins, alpha=0.7, color='#1f77b4')
    
    # Set labels
    ax3.set_xlabel('Outlier Score')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Distribution of Outlier Scores (method: {method})')
    
    # Plot legend and statistics in the bottom right
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_axis_off()
    
    # Create dummy scatter points for size legend
    sizes = [30, 50, 70, 90, 110]
    y_pos = np.linspace(0.8, 0.2, len(sizes))
    
    for size, y in zip(sizes, y_pos):
        ax4.scatter(0.5, y, s=size, c='#1f77b4', alpha=0.7)
        
        # Calculate corresponding outlier score
        if outlier_scores.max() > 0:
            score = (size - 20) / 100 * outlier_scores.max()
            ax4.text(0.6, y, f'{score:.2f}', va='center')
    
    # Create legend for colors
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Kept points'),
        Patch(facecolor='#ff7f0e', label='Removed (windsorized)'),
        Patch(facecolor='#d62728', label='Removed (model improvement)')
    ]
    ax4.legend(handles=legend_elements, loc='upper center')
    
    ax4.text(0.5, 0.95, 'Outlier Score Legend', ha='center', va='top', fontweight='bold')
    
    # Add counts for each category
    kept = np.sum(mask)
    windsorized = np.sum(removal_reasons == 1)
    model_improved = np.sum(removal_reasons == 2)
    total = len(mask)
    
    stats_text = (
        f"Total points: {total}\n"
        f"Kept: {kept} ({kept/total*100:.1f}%)\n"
        f"Windsorized: {windsorized} ({windsorized/total*100:.1f}%)\n"
        f"Model improved: {model_improved} ({model_improved/total*100:.1f}%)"
    )
    
    ax4.text(0.5, 0.4, stats_text, ha='center', va='top', fontsize=9)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    # Add overall title
    fig.suptitle(f'Outlier Detection Results: {kept} points kept, {total-kept} points removed', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def remove_std_outliers(X, y, std_threshold=3):
    """
    Remove outliers in the data based on standard deviation thresholds for each feature and target.
    
    Parameters:
    -----------
    X : array-like
        Feature values
    y : array-like
        Target values
    std_threshold : float
        Number of standard deviations beyond which points are considered outliers
        
    Returns:
    --------
    X_clean : array
        Filtered X data with outliers removed
    y_clean : array
        Filtered y data with outliers removed
    mask : boolean array
        Mask for non-outlier points (True = keep, False = outlier)
    removal_reasons : array
        Array indicating which points were removed (0=kept, 1=std_outlier)
    """
    n = len(X)
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Initialize mask (True = keep point)
    mask = np.ones(n, dtype=bool)
    
    # Initialize removal reasons array (0=kept, 1=std_outlier)
    removal_reasons = np.zeros(n, dtype=int)
    
    # For each feature, apply std-based removal
    for j in range(X.shape[1]):
        # Calculate mean and std
        mean = np.mean(X[:, j])
        std = np.std(X[:, j])
        
        # Skip if std is zero or close to zero
        if std < 1e-10:
            continue
            
        # Find points outside the thresholds
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        
        feature_outliers = (X[:, j] < lower_bound) | (X[:, j] > upper_bound)
        
        # Update mask and removal reasons
        removal_reasons[feature_outliers & mask] = 1  # Mark as std_outlier
        mask[feature_outliers] = False
    
    # Apply std-based filtering to target variable y as well
    mean_y = np.mean(y)
    std_y = np.std(y)
    
    # Skip if std is zero or close to zero
    if std_y >= 1e-10:
        lower_bound_y = mean_y - std_threshold * std_y
        upper_bound_y = mean_y + std_threshold * std_y
        
        y_outliers = (y < lower_bound_y) | (y > upper_bound_y)
        
        # Update mask and removal reasons
        removal_reasons[y_outliers & mask] = 1  # Mark as std_outlier
        mask[y_outliers] = False
    
    std_outlier_count = np.sum(~mask)
    if std_outlier_count > 0:
        print(f"Std-based removal: removed {std_outlier_count} points ({std_outlier_count/n*100:.1f}%) outside of {std_threshold} standard deviations")
    
    return X[mask], y[mask], mask, removal_reasons

