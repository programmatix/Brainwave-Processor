import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_partregress

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

def advanced_visualizations(X, y, model, title, top_n=10):
    """Generate advanced visualizations for regression models"""
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 20))
    
    # 1. Effect Size Plot (with standardized coefficients)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_scaled = LinearRegression().fit(X_scaled, y)
    
    # Create DataFrame for coefficients
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model_scaled.coef_,
        'abs_coefficient': abs(model_scaled.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    ax1 = fig.add_subplot(321)
    sns.barplot(data=coef_df.head(top_n), x='coefficient', y='feature', palette='coolwarm', ax=ax1)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title(f'Standardized Coefficients (bit like shaply beeswarm)')
    ax1.set_xlabel('Coefficient Value (Standardized)')
    
    # 2. Predicted vs Actual Plot
    y_pred = model.predict(X)
    
    ax2 = fig.add_subplot(322)
    ax2.scatter(y_pred, y, alpha=0.5)
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax2.set_title(f'Predicted vs Actual - {title}')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Actual Values')
    
    # 3. Residual Plot
    residuals = y - y_pred
    
    ax3 = fig.add_subplot(323)
    ax3.scatter(y_pred, residuals, alpha=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title(f'Residual/Error Plot (how far the prediction was from the actual - looking for random scatter)')
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals')
    
    # 4. Variable Importance using Permutation Importance
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean,
    }).sort_values('importance', ascending=False)
    
    ax4 = fig.add_subplot(324)
    sns.barplot(data=importance_df.head(top_n), x='importance', y='feature', palette='viridis', ax=ax4)
    ax4.set_title(f'Permutation Importance - {title}')
    ax4.set_xlabel('Mean Decrease in R² Score')
    
    # 5. Histogram of Residuals
    ax5 = fig.add_subplot(325)
    sns.histplot(residuals, kde=True, ax=ax5)
    ax5.set_title(f'Distribution of Residuals/Error - {title}')
    ax5.set_xlabel('Residual Value')
    
    # 6. Top Feature Partial Regression Plot (using a different approach to avoid the error)
    ax6 = fig.add_subplot(326)
    
    # Create a model with statsmodels
    X_with_const = sm.add_constant(X)
    sm_model = sm.OLS(y, X_with_const).fit()
    
    if len(X.columns) > 0:
        top_feature = coef_df.iloc[0]['feature']
        top_idx = list(X.columns).index(top_feature)
        
        # Manual partial regression plot
        # Get residuals from regressing the target variable on all other predictors
        X_others = X_with_const.drop(columns=[top_feature])
        y_resid = y - sm.OLS(y, X_others).fit().predict(X_others)
        
        # Get residuals from regressing the selected predictor on all other predictors
        x_resid = X[top_feature] - sm.OLS(X[top_feature], X_others).fit().predict(X_others)
        
        # Plot the relationship between these residuals
        ax6.scatter(x_resid, y_resid, alpha=0.5)
        
        # Add regression line
        slope, intercept = np.polyfit(x_resid, y_resid, 1)
        x_range = np.linspace(min(x_resid), max(x_resid), 100)
        ax6.plot(x_range, intercept + slope * x_range, 'r-')
        
        ax6.set_title(f'Partial Regression Plot for Top Feature: {top_feature}')
        ax6.set_xlabel(f'Residuals of {top_feature}')
        ax6.set_ylabel(f'Residuals of {title}')
    
    plt.tight_layout()
    plt.suptitle(f'Advanced Regression Analysis - {title}', fontsize=16, y=1.02)
    plt.show()
    
    return importance_df, coef_df
def visualize_feature_importance(feature_importance_df, title):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature', palette='viridis')
    plt.title(f'Top 15 Most Important Features - {title}')
    plt.xlabel('Feature Importance (Absolute Coefficient Value)')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.show()


def multiple_regression(df, target1: str):
    require_no_missing_values(df, df.columns)

    # Verify that dayAndNightOf is the index
    assert df.index.name == 'dayAndNightOf', "DataFrame index must be 'dayAndNightOf'"
    
    df1 = df.copy()
    # df1 = df1.dropna()

    total_days = len(df1.index.unique())
    print(f"\nAnalyzing {target1}")
    print(f"Total days in dataset: {total_days}")

    X = df1.copy().drop(columns=[target1])
    y1 = df1[target1]
    
    def calculate_r2_contributions(X, y, model):
        predictions = model.predict(X)
        mean_y = y.mean()
        total_ss = ((y - mean_y) ** 2).sum()
        residuals = y - predictions
        r2_contributions = 1 - (residuals ** 2) / total_ss
        return r2_contributions, predictions, residuals

    model1 = LinearRegression().fit(X, y1)
    r2_contributions, predictions, residuals = calculate_r2_contributions(X, y1, model1)
    
    # Add predictions and errors to the original dataframe
    df1[f'{target1}_predicted'] = predictions
    df1[f'{target1}_error'] = residuals
    df1[f'{target1}_abs_error'] = abs(residuals)
    df1['r2_contribution'] = r2_contributions
    
    threshold = r2_contributions.mean() - 2 * r2_contributions.std()
    bad_days = df1[r2_contributions < threshold].index.unique()
    
    df_kept = df1[~df1.index.isin(bad_days)]
    df_removed = df1[df1.index.isin(bad_days)]
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(model1.coef_)
    }).sort_values('importance', ascending=False)

    results_df = None
    
    if len(bad_days) > 0:
        removed_percentage = (len(bad_days) / total_days) * 100
        print(f"\nRemoving {len(bad_days)} days ({removed_percentage:.1f}%) that negatively impact R²:")
        for day in bad_days:
            print(f"  - {day}")
        
        X_filtered = df_kept.drop(columns=[target1, 
                                          f'{target1}_predicted', f'{target1}_error', 
                                          f'{target1}_abs_error', 'r2_contribution'])
        y1_filtered = df_kept[target1]
        
        model1_filtered = LinearRegression().fit(X_filtered, y1_filtered)
        r2_filtered = model1_filtered.score(X_filtered, y1_filtered)
        r2_original = model1.score(X, y1)
        r2_improvement = r2_filtered - r2_original
        
        # Add filtered model predictions to kept dataframe
        filtered_predictions = model1_filtered.predict(X_filtered)
        df_kept[f'{target1}_filtered_predicted'] = filtered_predictions
        df_kept[f'{target1}_filtered_error'] = y1_filtered - filtered_predictions
        df_kept[f'{target1}_filtered_abs_error'] = abs(y1_filtered - filtered_predictions)
        
        print(f"\nFinal Statistics:")
        print(f"Total days: {total_days}")
        print(f"Days removed: {len(bad_days)} ({removed_percentage:.1f}%)")
        print(f"Original R²: {r2_original:.4f}")
        print(f"Filtered R²: {r2_filtered:.4f}")
        print(f"R² improvement: {r2_improvement:.4f}")
        
        # Add average error metrics
        print(f"Original mean absolute error: {df1[f'{target1}_abs_error'].mean():.4f}")
        print(f"Filtered mean absolute error: {df_kept[f'{target1}_filtered_abs_error'].mean():.4f}")
        error_improvement = df1[f'{target1}_abs_error'].mean() - df_kept[f'{target1}_filtered_abs_error'].mean()
        print(f"Error improvement: {error_improvement:.4f}")

        results_df = pd.DataFrame({'r2_original': [r2_original], 'r2_filtered': [r2_filtered], 'r2_improvement': [r2_improvement], 'error_improvement': [error_improvement], 'target': [target1], 'total_days': [total_days], 'days_removed': [len(bad_days)], 'removed_percentage': [removed_percentage]})
            #results_df = pd.concat([results_df, pd.DataFrame({'r2_original': r2_original, 'r2_filtered': r2_filtered, 'r2_improvement': r2_improvement, 'error_improvement': error_improvement, 'target': target1, 'total_days': total_days, 'days_removed': len(bad_days), 'removed_percentage': removed_percentage})])
    else:
        print(f"\nNo days found that significantly reduce R² for {target1}")
        print(f"Final Statistics:")
        print(f"Total days: {total_days}")
        print(f"Days removed: 0 (0.0%)")
        print(f"R²: {model1.score(X, y1):.4f}")
        print(f"Mean absolute error: {df1[f'{target1}_abs_error'].mean():.4f}")
    
    # Assert that dayAndNightOf is still the index in returned dataframes
    assert df_kept.index.name == 'dayAndNightOf', "Kept DataFrame should maintain 'dayAndNightOf' as index"
    assert df_removed.index.name == 'dayAndNightOf', "Removed DataFrame should maintain 'dayAndNightOf' as index"
    
    return df_kept, df_removed, feature_importance, results_df

def bin_and_viz(series, method='quantile', n_bins=5, **kwargs):
    binned_series, result_info = bin_continuous_data(series, method, n_bins, **kwargs)
    fig = visualize_binning(series, binned_series, result_info, method)
    return binned_series, result_info, fig

def fuzzy_bin_and_viz(series, method='gmm_prob', n_bins=5, **kwargs):
    membership_df, result_info = fuzzy_bin_data(series, method, n_bins, **kwargs)
    fig = visualize_fuzzy_binning(series, membership_df, result_info, method)
    return membership_df, result_info, fig

def bin_continuous_data(series, method='quantile', n_bins=5, **kwargs):
    """
    Bin continuous data using various methods.
    
    Parameters:
    -----------
    series : pd.Series
        The continuous data to bin
    method : str, default='quantile'
        Binning method: 'kmeans', 'gmm', 'tree', 'quantile', or 'kbins'
    n_bins : int or None, default=5
        Number of bins to create. If None, will automatically determine optimal bin count.
    **kwargs : 
        Additional parameters for specific methods
        
    Returns:
    --------
    pd.Series: Series with bin labels
    dict: Additional information about the binning (models, bin edges, etc.)
    """
    data = series.values.reshape(-1, 1)
    result_info = {}
    
    # Handle automatic bin count determination if n_bins is None
    if n_bins is None:
        auto_method = kwargs.get('auto_method', 'kmeans')
        n_bins = determine_optimal_bin_count(series, method=auto_method)
        result_info['auto_determined_bins'] = n_bins
        print(f"Automatically determined optimal number of bins: {n_bins}")
    
    # Handle 'auto' method for automatic determination of bin count
    if method == 'auto':
        method = kwargs.get('base_method', 'kmeans')
        n_bins = determine_optimal_bin_count(series, method=method)
        result_info['auto_determined_bins'] = n_bins
        print(f"Automatically determined optimal number of bins: {n_bins}")
    
    if method == 'kmeans':
        model = KMeans(n_clusters=n_bins, random_state=42)
        bins = model.fit_predict(data)
        result_info['model'] = model
        result_info['centers'] = model.cluster_centers_.flatten()
        
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_bins, random_state=42)
        model.fit(data)
        bins = model.predict(data)
        result_info['model'] = model
        result_info['means'] = model.means_.flatten()
        result_info['variances'] = model.covariances_.flatten()
        
    elif method == 'tree':
        # Fix for the tree binning method to ensure it creates the requested number of bins
        # We need to use the value as the feature, not the position
        X = data  # Use the actual values, not array indices
        # For very small datasets or with many duplicates, max_leaf_nodes might not result
        # in the exact number of bins we want, so we'll add some noise to help
        if len(np.unique(data)) < n_bins:
            # Add small noise only to duplicates to make them unique
            X = data + np.random.normal(0, 1e-6, size=data.shape)
        
        model = DecisionTreeRegressor(max_leaf_nodes=n_bins, random_state=42)
        model.fit(X, data)  # Predict the value itself
        bins = model.apply(X)  # Get leaf indices
        
        # Extract the threshold values used for splitting
        tree = model.tree_
        thresholds = tree.threshold
        # Filter out leaf nodes which have -2 as threshold
        valid_thresholds = thresholds[thresholds != -2]
        result_info['model'] = model
        result_info['thresholds'] = np.sort(valid_thresholds)
        
    elif method == 'quantile':
        bins = pd.qcut(series, q=n_bins, labels=False, duplicates='drop')
        bin_edges = pd.qcut(series, q=n_bins, duplicates='drop').categories
        result_info['bin_edges'] = bin_edges
        
    elif method == 'kbins':
        strategy = kwargs.get('strategy', 'quantile')
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        bins = discretizer.fit_transform(data).flatten()
        result_info['model'] = discretizer
        result_info['bin_edges'] = discretizer.bin_edges_[0]
    
    else:
        raise ValueError(f"Unsupported binning method: {method}")
    
    binned_series = pd.Series(bins, index=series.index)
    
    # Assign meaningful bin names if requested
    if kwargs.get('assign_bin_names', False):
        bin_names = generate_bin_names(series, binned_series, method, result_info)
        result_info['bin_names'] = bin_names
        
        # Create a mapping dictionary for bin names
        bin_map = {i: name for i, name in enumerate(bin_names)}
        binned_series = binned_series.map(bin_map)
    
    return binned_series, result_info

def determine_optimal_bin_count(series, method='kmeans', max_bins=15):
    """
    Determine the optimal number of bins/clusters for a given series.
    
    Parameters:
    -----------
    series : pd.Series
        The data to bin
    method : str, default='kmeans'
        Method to use for determining optimal bins: 'kmeans', 'gmm', 'silhouette', 'gap', 'elbow'
    max_bins : int, default=15
        Maximum number of bins to consider
        
    Returns:
    --------
    int: Optimal number of bins
    """
    data = series.values.reshape(-1, 1)
    
    # For very small datasets
    if len(series) < 2 * max_bins:
        return max(2, len(series) // 4)  # Use a reasonable default
    
    # Ensure we have enough unique values
    unique_count = series.nunique()
    if unique_count < max_bins:
        return max(2, unique_count // 2)
        
    if method == 'elbow' or method == 'kmeans':
        # Elbow method for KMeans
        distortions = []
        K_range = range(2, min(max_bins + 1, len(series) // 5 + 1))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
        
        # Find elbow point through acceleration
        if len(K_range) < 3:
            return 2  # Default to 2 if we don't have enough points
            
        deltas = np.diff(distortions)
        accelerations = np.diff(deltas)
        
        if len(accelerations) == 0:
            return 2
            
        k_opt = K_range[:-2][np.argmax(accelerations)] 
        return k_opt
        
    elif method == 'gmm':
        # BIC for Gaussian Mixture Models
        bic_values = []
        K_range = range(2, min(max_bins + 1, len(series) // 5 + 1))
        
        for k in K_range:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(data)
            bic_values.append(gmm.bic(data))
        
        if len(bic_values) == 0:
            return 2
            
        k_opt = K_range[np.argmin(bic_values)]
        return k_opt
        
    elif method == 'silhouette':
        # Silhouette score
        from sklearn.metrics import silhouette_score
        
        silhouette_scores = []
        K_range = range(2, min(max_bins + 1, len(series) // 5 + 1))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            
            try:
                score = silhouette_score(data, labels)
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(-1)  # Invalid score
        
        if len(silhouette_scores) == 0 or max(silhouette_scores) <= 0:
            return 2
            
        k_opt = K_range[np.argmax(silhouette_scores)]
        return k_opt
        
    elif method == 'gap':
        # Gap statistic
        
        K_range = range(2, min(max_bins + 1, len(series) // 5 + 1))
        gaps = []
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            
            # Calculate inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
            
            # For reference distribution, use uniform random data in same range
            reference_inertias = []
            
            # Generate reference datasets
            for _ in range(5):  # Simplification: use 5 references instead of typical 10+
                min_val, max_val = np.min(data), np.max(data)
                random_data = np.random.uniform(min_val, max_val, data.shape)
                
                # Fit KMeans to reference data
                kmeans_ref = KMeans(n_clusters=k, random_state=42)
                kmeans_ref.fit(random_data)
                reference_inertias.append(kmeans_ref.inertia_)
            
            # Compute gap statistic
            gap = np.log(np.mean(reference_inertias)) - np.log(inertia)
            gaps.append(gap)
        
        if len(gaps) == 0:
            return 2
            
        # Find first local maximum or significant increase
        for i in range(1, len(gaps)-1):
            if gaps[i] > gaps[i-1] and gaps[i] > gaps[i+1]:
                return K_range[i]
        
        # If no clear maximum, return the k with highest gap
        k_opt = K_range[np.argmax(gaps)]
        return k_opt
    
    else:
        # Default to simple heuristic based on data size
        return min(int(np.sqrt(len(series)/2)), max_bins)

def generate_bin_names(series, binned_series, method, result_info):
    """
    Generate meaningful names for bins based on their characteristics.
    
    Parameters:
    -----------
    series : pd.Series
        Original data
    binned_series : pd.Series
        Binned data
    method : str
        Binning method used
    result_info : dict
        Additional information about the binning
        
    Returns:
    --------
    list: List of bin names
    """
    unique_bins = sorted(binned_series.unique())
    bin_names = []
    
    # Get series name if available, or use a generic name
    series_name = getattr(series, 'name', 'Value')
    
    for bin_idx in unique_bins:
        bin_mask = binned_series == bin_idx
        bin_data = series[bin_mask]
        
        if len(bin_data) == 0:
            bin_names.append(f"Empty Bin {bin_idx}")
            continue
        
        # Calculate bin statistics
        bin_min = bin_data.min()
        bin_max = bin_data.max()
        bin_mean = bin_data.mean()
        bin_count = len(bin_data)
        bin_percent = (bin_count / len(series)) * 100
        
        # Determine quartile or location description
        all_values = sorted(series.values)
        min_percentile = scipy_stats.percentileofscore(all_values, bin_min)
        max_percentile = scipy_stats.percentileofscore(all_values, bin_max)
        
        # Descriptive location
        if min_percentile < 15:
            location = "Very Low"
        elif min_percentile < 35:
            location = "Low"
        elif min_percentile < 65:
            location = "Medium"
        elif min_percentile < 85:
            location = "High"
        else:
            location = "Very High"
            
        # Different naming strategies based on method and data characteristics
        if method in ['kmeans', 'gmm']:
            # For clustering methods, use center-based names
            if 'centers' in result_info:
                center = result_info['centers'][bin_idx]
                name = f"{location} {series_name}: ~{center:.2f}"
            elif 'means' in result_info:
                mean = result_info['means'][bin_idx]
                name = f"{location} {series_name}: ~{mean:.2f}"
            else:
                name = f"{location} {series_name}: {bin_min:.2f}-{bin_max:.2f}"
        
        elif method in ['quantile', 'kbins']:
            # For quantile-based methods, use range with percentile info
            name = f"{location} {series_name}: {bin_min:.2f}-{bin_max:.2f}"
            
        elif method == 'tree':
            # For tree-based methods, use decision rules if available
            name = f"{location} {series_name}: {bin_min:.2f}-{bin_max:.2f}"
            
        else:
            # Generic naming
            name = f"Bin {bin_idx}: {bin_min:.2f}-{bin_max:.2f}"
            
        bin_names.append(name)
    
    return bin_names

def visualize_binning(original_series, binned_series, result_info, method, figsize=(12, 4)):
    """
    Visualize the results of binning continuous data.
    
    Parameters:
    -----------
    original_series : pd.Series
        The original continuous data
    binned_series : pd.Series
        The binned data (output from bin_continuous_data)
    result_info : dict
        Information about the binning (output from bin_continuous_data)
    method : str
        The binning method used
    figsize : tuple, default=(12, 4)
        Figure size
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    
    # Plot binned data
    ax2 = axes
    
    # Scatter plot of original data, colored by bin
    unique_bins = pd.Series(binned_series).nunique()
    palette = sns.color_palette("hsv", unique_bins)
    
    # If bin names were provided, use them for labels
    if 'bin_names' in result_info:
        bin_labels = result_info['bin_names']
        unique_bin_values = binned_series.unique()
        
        for i, bin_value in enumerate(unique_bin_values):
            bin_mask = binned_series == bin_value
            try:
                label = bin_labels[i] if i < len(bin_labels) else f"Bin {bin_value}"
                ax2.scatter(
                    original_series[bin_mask].index,
                    original_series[bin_mask],
                    label=label,
                    color=palette[i % len(palette)],
                    alpha=0.7
                )
            except Exception as e:
                print(f"Error plotting bin {bin_value}: {e}")
    else:
        # Numerical bin indices
        for bin_idx in range(unique_bins):
            # For string/categorical bins, we need to handle differently
            if isinstance(binned_series.iloc[0], (str, np.str_)):
                unique_values = binned_series.unique()
                if bin_idx >= len(unique_values):
                    continue
                bin_value = unique_values[bin_idx]
                bin_mask = binned_series == bin_value
            else:
                bin_mask = binned_series == bin_idx
                
            ax2.scatter(
                original_series[bin_mask].index, 
                original_series[bin_mask], 
                label=f'Bin {bin_idx}',
                color=palette[bin_idx % len(palette)],
                alpha=0.7
            )
    
    # Plot centroids or other method-specific information
    if method == 'kmeans' and 'centers' in result_info:
        centers = result_info['centers']
        for i, center in enumerate(centers):
            ax2.axhline(y=center, color=palette[i % len(palette)], linestyle='--', alpha=0.5)
            
    elif method == 'gmm' and 'means' in result_info:
        means = result_info['means']
        variances = result_info['variances']
        for i, (mean, var) in enumerate(zip(means, variances)):
            ax2.axhline(y=mean, color=palette[i % len(palette)], linestyle='--', alpha=0.5)
            
    elif method == 'tree' and 'thresholds' in result_info:
        thresholds = result_info['thresholds']
        for threshold in thresholds:
            ax2.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
            
    elif (method == 'quantile' or method == 'kbins') and 'bin_edges' in result_info:
        bin_edges = result_info['bin_edges']
        for edge in bin_edges:
            if hasattr(edge, 'left') and hasattr(edge, 'right'):
                # For quantile binning with IntervalIndex
                ax2.axhline(y=edge.left, color='gray', linestyle='--', alpha=0.5)
                ax2.axhline(y=edge.right, color='gray', linestyle='--', alpha=0.5)
            else:
                # For kbins with array of edges
                ax2.axhline(y=edge, color='gray', linestyle='--', alpha=0.5)
    
    auto_text = " (Auto)" if 'auto_determined_bins' in result_info else ""
    ax2.set_title(f'Data Binned with {method.upper()}{auto_text} ({unique_bins} bins)')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    
    # Adjust legend based on number of bins
    if unique_bins > 10:
        ax2.legend(loc='best', ncol=3, fontsize='small')
    else:
        ax2.legend(loc='best')
    
    plt.tight_layout()
    return fig

def analyze_binned_data(original_series, binned_series):
    """
    Analyze binned data to provide statistics for each bin.
    
    Parameters:
    -----------
    original_series : pd.Series
        The original continuous data
    binned_series : pd.Series
        The binned data (output from bin_continuous_data)
        
    Returns:
    --------
    pd.DataFrame: DataFrame with statistics for each bin
    """
    bin_stats = []
    
    for bin_idx in range(binned_series.nunique()):
        bin_mask = binned_series == bin_idx
        bin_data = original_series[bin_mask]
        
        stats = {
            'bin': bin_idx,
            'count': len(bin_data),
            'percentage': len(bin_data) / len(original_series) * 100,
            'min': bin_data.min(),
            'max': bin_data.max(),
            'mean': bin_data.mean(),
            'median': bin_data.median(),
            'std': bin_data.std()
        }
        bin_stats.append(stats)
    
    return pd.DataFrame(bin_stats)

def fuzzy_bin_data(series, method='fuzzy_cmeans', n_bins=5, **kwargs):
    """
    Bin continuous data with fuzzy methods allowing values to belong to multiple bins.
    
    Parameters:
    -----------
    series : pd.Series
        The continuous data to bin
    method : str, default='fuzzy_cmeans'
        Binning method: 'fuzzy_cmeans', 'gmm_prob', 'soft_quantile'
    n_bins : int or None, default=5
        Number of bins to create. If None, will automatically determine optimal bin count.
    **kwargs : 
        Additional parameters for specific methods
        
    Returns:
    --------
    pd.DataFrame: DataFrame with membership values for each bin
    dict: Additional information about the binning (models, bin edges, etc.)
    """
    data = series.values.reshape(-1, 1)
    result_info = {}
    
    # Handle automatic bin count determination if n_bins is None
    if n_bins is None:
        auto_method = kwargs.get('auto_method', 'kmeans')
        n_bins = determine_optimal_bin_count(series, method=auto_method)
        result_info['auto_determined_bins'] = n_bins
        print(f"Automatically determined optimal number of bins: {n_bins}")
    
    if method == 'fuzzy_cmeans':
        # Using scikit-fuzzy if available
        try:
            from skfuzzy.cluster import cmeans
            
            # Normalize data for better FCM results
            data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            
            # Apply Fuzzy C-means
            center, u, _, _, _, _, _ = cmeans(
                data_normalized.T, n_bins, 
                m=kwargs.get('m', 2),  # fuzziness parameter
                error=kwargs.get('error', 0.005),
                maxiter=kwargs.get('maxiter', 1000)
            )
            
            # u is the membership matrix (each row is a bin, each column is a data point)
            memberships = u.T  # transpose to have rows as data points, columns as bins
            result_info['centers'] = center.flatten()
            
        except ImportError:
            # Fallback to simpler approach if scikit-fuzzy not available
            # Use KMeans to get centers, then compute distances as probabilities
            model = KMeans(n_clusters=n_bins, random_state=42)
            model.fit(data)
            centers = model.cluster_centers_
            
            # Calculate distances to each center
            distances = np.zeros((len(data), n_bins))
            for i, center in enumerate(centers):
                distances[:, i] = np.abs(data.flatten() - center.flatten())
            
            # Convert distances to memberships (closer = higher membership)
            # Invert and normalize distances
            distances_inv = 1.0 / (1.0 + distances)
            memberships = distances_inv / np.sum(distances_inv, axis=1, keepdims=True)
            
            result_info['centers'] = centers.flatten()
    
    elif method == 'gmm_prob':
        # Gaussian Mixture Model with probabilities instead of hard assignments
        model = GaussianMixture(n_components=n_bins, random_state=42)
        model.fit(data)
        
        # Get probabilities of belonging to each component
        memberships = model.predict_proba(data)
        
        result_info['model'] = model
        result_info['means'] = model.means_.flatten()
        result_info['variances'] = model.covariances_.flatten()
    
    elif method == 'soft_quantile':
        # Create quantile-based bins but with soft memberships
        
        # Get quantile bin edges
        quantiles = np.linspace(0, 1, n_bins+1)
        bin_edges = np.quantile(data, quantiles)
        result_info['bin_edges'] = bin_edges
        
        # Create matrix to store memberships
        memberships = np.zeros((len(data), n_bins))
        
        # Compute memberships using Gaussian kernel around each value
        bandwidth = kwargs.get('bandwidth', 0.1) * (np.max(data) - np.min(data))
        
        for i, val in enumerate(data.flatten()):
            # For each bin, compute membership based on distance to bin center
            for j in range(n_bins):
                bin_center = (bin_edges[j] + bin_edges[j+1]) / 2
                # Use Gaussian function for smooth transition
                dist = abs(val - bin_center)
                memberships[i, j] = np.exp(-(dist**2) / (2 * bandwidth**2))
            
            # Normalize memberships to sum to 1
            if np.sum(memberships[i]) > 0:
                memberships[i] = memberships[i] / np.sum(memberships[i])
    
    else:
        raise ValueError(f"Unsupported fuzzy binning method: {method}")
    
    # Create DataFrame with memberships
    columns = [f"bin_{i}" for i in range(n_bins)]
    membership_df = pd.DataFrame(memberships, index=series.index, columns=columns)
    
    # Assign bin names if requested
    if kwargs.get('assign_bin_names', False):
        # Generate hard assignments to use with bin naming function
        hard_assignments = membership_df.idxmax(axis=1).str.replace('bin_', '').astype(int)
        hard_assignments = pd.Series(hard_assignments, index=series.index)
        
        # Get naming mode
        naming_mode = kwargs.get('naming_mode', True)
        
        if naming_mode == 'hours':
            bin_names = generate_hour_bin_names(series, result_info, method, n_bins)
        else:
            # Use the standard bin naming function with hard assignments
            bin_names = generate_bin_names(series, hard_assignments, method, result_info)
        
        result_info['bin_names'] = bin_names
        
        # Rename columns in membership_df
        new_columns = {}
        for i, name in enumerate(bin_names):
            new_columns[f"bin_{i}"] = name
        
        membership_df = membership_df.rename(columns=new_columns)
    
    return membership_df, result_info

def generate_hour_bin_names(series, result_info, method, n_bins):
    """
    Generate time-based bin names for hour data.
    
    Parameters:
    -----------
    series : pd.Series
        Original hour data (e.g., 21.5 for 21:30)
    result_info : dict
        Information about binning method
    method : str
        Binning method used
    n_bins : int
        Number of bins
        
    Returns:
    --------
    list: List of bin names in time format
    """
    bin_names = []
    
    def format_time(hours):
        """Format hour value as HH:MM, handling values over 24 hours."""
        day_suffix = ""
        original_hours = hours
        
        # Handle multi-day hours
        if hours >= 24:
            days = int(hours // 24)
            hours = hours % 24
            day_suffix = f" (+{days}d)"
        
        # Format as HH:MM
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h:02d}:{m:02d}{day_suffix}"
    
    if method in ['kmeans', 'fuzzy_cmeans']:
        centers = sorted(result_info.get('centers', []))
        for i, center in enumerate(centers):
            time_str = format_time(center)
            bin_names.append(f"{time_str}")
            
    elif method == 'gmm_prob':
        means = sorted(result_info.get('means', []))
        for i, mean in enumerate(means):
            time_str = format_time(mean)
            bin_names.append(f"{time_str}")
            
    elif method == 'soft_quantile':
        bin_edges = result_info.get('bin_edges', [])
        
        if len(bin_edges) >= 2:
            for i in range(len(bin_edges) - 1):
                start_time = format_time(bin_edges[i])
                end_time = format_time(bin_edges[i+1])
                bin_names.append(f"{start_time}_to_{end_time}")
        else:
            # Fallback if edges not available
            min_val = series.min()
            max_val = series.max()
            step = (max_val - min_val) / n_bins
            
            for i in range(n_bins):
                start_val = min_val + i * step
                end_val = min_val + (i + 1) * step
                start_time = format_time(start_val)
                end_time = format_time(end_val)
                bin_names.append(f"{start_time}_to_{end_time}")
    
    else:
        # Generic time bins
        min_val = series.min()
        max_val = series.max()
        step = (max_val - min_val) / n_bins
        
        for i in range(n_bins):
            start_val = min_val + i * step
            end_val = min_val + (i + 1) * step
            start_time = format_time(start_val)
            end_time = format_time(end_val)
            bin_names.append(f"{start_time}_to_{end_time}")
    
    return bin_names

def visualize_fuzzy_binning(original_series, membership_df, result_info, method, figsize=(15, 10)):
    """
    Visualize the results of fuzzy binning where values can belong to multiple bins.
    
    Parameters:
    -----------
    original_series : pd.Series
        The original continuous data
    membership_df : pd.DataFrame
        DataFrame with membership values for each bin (output from fuzzy_bin_data)
    result_info : dict
        Information about the binning (output from fuzzy_bin_data)
    method : str
        The fuzzy binning method used
    figsize : tuple, default=(15, 10)
        Figure size
    """
    n_bins = membership_df.shape[1]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 2, 1]})
    
    # Generate a color palette for the bins
    palette = sns.color_palette("hsv", n_bins)
    
    # Plot 1: Original data with color blending based on membership
    ax1 = axes[0]
    
    # Normalize memberships if needed to sum to 1
    normalized_memberships = membership_df.copy()
    row_sums = normalized_memberships.sum(axis=1)
    for i, row_sum in enumerate(row_sums):
        if row_sum > 0:
            normalized_memberships.iloc[i] = normalized_memberships.iloc[i] / row_sum
    
    # For each point, calculate blended color based on membership
    colors = []
    for _, row in normalized_memberships.iterrows():
        # Initialize RGB as zeros
        blended_color = np.zeros(3)
        # Add weighted contribution from each bin color
        for bin_idx in range(n_bins):
            weight = row[f"bin_{bin_idx}"]
            if weight > 0:
                bin_color = np.array(palette[bin_idx])
                blended_color += weight * bin_color
        colors.append(blended_color)
    
    # Plot scatter points with blended colors
    sc = ax1.scatter(
        original_series.index,
        original_series.values,
        c=colors,
        alpha=0.8,
        s=50
    )
    
    # Add method-specific elements
    if method in ['fuzzy_cmeans', 'gmm_prob'] and 'centers' in result_info:
        centers = result_info['centers']
        for i, center in enumerate(centers):
            ax1.axhline(y=center, color=palette[i], linestyle='--', alpha=0.7)
            ax1.text(original_series.index.min(), center + 0.02 * (original_series.max() - original_series.min()), 
                     f"Bin {i} center", color=palette[i], fontweight='bold')
    
    elif method == 'soft_quantile' and 'bin_edges' in result_info:
        bin_edges = result_info['bin_edges']
        for i, edge in enumerate(bin_edges):
            ax1.axhline(y=edge, color='black', linestyle='--', alpha=0.5)
    
    ax1.set_title('Data Points Colored by Fuzzy Bin Membership')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    
    # Add a custom legend for bin colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=palette[i], edgecolor='black', alpha=0.7, label=f'Bin {i}')
                      for i in range(n_bins)]
    ax1.legend(handles=legend_elements, loc='upper right', title="Pure Bin Colors")
    
    # Plot 2: Membership visualization as stacked areas
    ax2 = axes[1]
    
    # Sort data for better visualization
    sort_idx = np.argsort(original_series.values)
    sorted_values = original_series.values[sort_idx]
    sorted_memberships = membership_df.values[sort_idx]
    
    # Create x-axis for plotting (positions)
    x_positions = np.arange(len(sorted_values))
    
    # Plot stacked area chart of memberships
    bottoms = np.zeros(len(sorted_values))
    
    for i in range(n_bins):
        ax2.fill_between(
            x_positions, 
            bottoms, 
            bottoms + sorted_memberships[:, i],
            color=palette[i], 
            alpha=0.7,
            label=f'Bin {i}'
        )
        bottoms += sorted_memberships[:, i]
    
    # Add the sorted original values as a line
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_positions, sorted_values, 'k-', alpha=0.5, label='Original Values')
    
    ax2.set_title(f'Fuzzy Memberships ({method.upper()}) - Stacked by Bin')
    ax2.set_xlabel('Sorted Data Points')
    ax2.set_ylabel('Membership Degree')
    ax2_twin.set_ylabel('Original Value')
    ax2.legend(loc='upper left')
    
    # Plot 3: Heatmap of memberships
    ax3 = axes[2]
    
    # Get a subset of data for better visualization if too many points
    max_display = min(300, len(sorted_memberships))
    if len(sorted_memberships) > max_display:
        step = len(sorted_memberships) // max_display
        display_memberships = sorted_memberships[::step]
        display_positions = x_positions[::step]
        display_values = sorted_values[::step]
    else:
        display_memberships = sorted_memberships
        display_positions = x_positions
        display_values = sorted_values
    
    # Create heatmap
    heatmap_data = display_memberships.T  # transpose for proper orientation
    
    # Plot colormap
    im = ax3.imshow(
        heatmap_data, 
        aspect='auto', 
        cmap='viridis', 
        extent=[0, len(display_positions), -0.5, n_bins-0.5]
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax3)
    cbar.set_label('Membership Degree')
    
    # Set labels
    ax3.set_title(f'Membership Heatmap ({method.upper()})')
    ax3.set_xlabel('Sorted Data Points')
    ax3.set_ylabel('Bin')
    ax3.set_yticks(np.arange(n_bins))
    ax3.set_yticklabels([f'Bin {i}' for i in range(n_bins)])
    
    plt.tight_layout()
    return fig

def analyze_fuzzy_binned_data(original_series, membership_df):
    """
    Analyze fuzzy binned data to provide statistics for each bin.
    
    Parameters:
    -----------
    original_series : pd.Series
        The original continuous data
    membership_df : pd.DataFrame
        DataFrame with membership values for each bin (output from fuzzy_bin_data)
        
    Returns:
    --------
    pd.DataFrame: DataFrame with statistics for each bin
    """
    n_bins = membership_df.shape[1]
    bin_stats = []
    
    for bin_idx in range(n_bins):
        # Get memberships for this bin
        bin_memberships = membership_df.iloc[:, bin_idx]
        
        # Compute weighted statistics
        total_membership = bin_memberships.sum()
        weighted_values = original_series * bin_memberships
        
        # Compute primary membership
        primary_membership_mask = membership_df.idxmax(axis=1) == f"bin_{bin_idx}"
        primary_count = primary_membership_mask.sum()
        
        # Find values with significant membership (e.g., > 0.3)
        significant_membership_mask = bin_memberships > 0.3
        significant_count = significant_membership_mask.sum()
        
        # Calculate statistics
        stats = {
            'bin': bin_idx,
            'total_membership': total_membership,
            'primary_membership_count': primary_count,
            'significant_membership_count': significant_count,
            'weighted_mean': weighted_values.sum() / total_membership if total_membership > 0 else np.nan,
            'min_membership': bin_memberships.min(),
            'max_membership': bin_memberships.max(),
            'overlap_with_other_bins': (bin_memberships > 0).sum() - primary_count
        }
        
        bin_stats.append(stats)
    
    return pd.DataFrame(bin_stats)

def compare_hard_vs_fuzzy_binning(original_series, n_bins=5, figsize=(15, 12)):
    """
    Compare hard binning vs fuzzy binning for the same data.
    
    Parameters:
    -----------
    original_series : pd.Series
        The continuous data to bin
    n_bins : int, default=5
        Number of bins to create
    figsize : tuple, default=(15, 12)
        Figure size
        
    Returns:
    --------
    fig : matplotlib Figure object
    """
    # Apply different binning methods
    kmeans_bins, kmeans_info = bin_continuous_data(original_series, method='kmeans', n_bins=n_bins)
    gmm_bins, gmm_info = bin_continuous_data(original_series, method='gmm', n_bins=n_bins)
    quantile_bins, quantile_info = bin_continuous_data(original_series, method='quantile', n_bins=n_bins)
    
    # Apply fuzzy binning
    fuzzy_cmeans, fuzzy_cmeans_info = fuzzy_bin_data(original_series, method='fuzzy_cmeans', n_bins=n_bins)
    gmm_probs, gmm_probs_info = fuzzy_bin_data(original_series, method='gmm_prob', n_bins=n_bins)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # Plot hard binning methods
    for bin_idx in range(kmeans_bins.nunique()):
        bin_mask = kmeans_bins == bin_idx
        axes[0, 0].scatter(
            original_series[bin_mask].index, 
            original_series[bin_mask], 
            label=f'Bin {bin_idx}', 
            alpha=0.7
        )
    axes[0, 0].set_title('K-Means Binning (Hard)')
    axes[0, 0].set_ylabel('Value')
    
    for bin_idx in range(gmm_bins.nunique()):
        bin_mask = gmm_bins == bin_idx
        axes[1, 0].scatter(
            original_series[bin_mask].index, 
            original_series[bin_mask], 
            label=f'Bin {bin_idx}', 
            alpha=0.7
        )
    axes[1, 0].set_title('GMM Binning (Hard)')
    axes[1, 0].set_ylabel('Value')
    
    for bin_idx in range(quantile_bins.nunique()):
        bin_mask = quantile_bins == bin_idx
        axes[2, 0].scatter(
            original_series[bin_mask].index, 
            original_series[bin_mask], 
            label=f'Bin {bin_idx}', 
            alpha=0.7
        )
    axes[2, 0].set_title('Quantile Binning (Hard)')
    axes[2, 0].set_xlabel('Index')
    axes[2, 0].set_ylabel('Value')
    
    # Plot fuzzy binning - use primary memberships for coloring
    primary_cmeans_bins = fuzzy_cmeans.idxmax(axis=1).str.replace('bin_', '').astype(int)
    for bin_idx in range(n_bins):
        bin_mask = primary_cmeans_bins == bin_idx
        # Get alpha based on membership strength
        alpha_values = fuzzy_cmeans.iloc[:, bin_idx][bin_mask].values
        
        # Plot with variable alpha based on membership
        axes[0, 1].scatter(
            original_series[bin_mask].index, 
            original_series[bin_mask], 
            label=f'Bin {bin_idx}',
            alpha=0.7
        )
    axes[0, 1].set_title('Fuzzy C-Means (Primary Membership)')
    
    primary_gmm_bins = gmm_probs.idxmax(axis=1).str.replace('bin_', '').astype(int)
    for bin_idx in range(n_bins):
        bin_mask = primary_gmm_bins == bin_idx
        axes[1, 1].scatter(
            original_series[bin_mask].index, 
            original_series[bin_mask], 
            label=f'Bin {bin_idx}',
            alpha=0.7
        )
    axes[1, 1].set_title('GMM Probabilities (Primary Membership)')
    
    # Bottom right: Show membership strengths for a sample of points
    # Get a subset of points for visualization
    sample_size = min(20, len(original_series))
    sample_indices = np.linspace(0, len(original_series)-1, sample_size, dtype=int)
    sample_data = original_series.iloc[sample_indices]
    
    # Show membership values for these points
    sample_memberships_cmeans = fuzzy_cmeans.iloc[sample_indices]
    
    # Create a heatmap of memberships
    im = axes[2, 1].imshow(
        sample_memberships_cmeans.values.T, 
        aspect='auto',
        cmap='viridis',
        extent=[0, sample_size, -0.5, n_bins-0.5]
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[2, 1])
    cbar.set_label('Membership Degree')
    
    axes[2, 1].set_title('Fuzzy Memberships for Sample Points')
    axes[2, 1].set_xlabel('Sample Points')
    axes[2, 1].set_ylabel('Bin')
    axes[2, 1].set_yticks(np.arange(n_bins))
    axes[2, 1].set_yticklabels([f'Bin {i}' for i in range(n_bins)])
    
    # Add legend to one subplot and it will be used for all
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=n_bins)
    
    plt.tight_layout()
    return fig

