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


def calculate_correlations(df, var1, var2):
    pearson_corr, pearson_p = stats.pearsonr(df[var1], df[var2])
    spearman_corr, spearman_p = stats.spearmanr(df[var1], df[var2])
    kendall_corr, kendall_p = stats.kendalltau(df[var1], df[var2])
    distance_corr = distance_correlation(df[var1], df[var2])
    
    # Calculate mutual information
    # Reshape for mutual_info_regression
    X = df[var1].values.reshape(-1, 1)
    y = df[var2].values
    mi = mutual_info_regression(X, y, random_state=42)[0]
    
    results = {
        'Pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
        'Spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
        'Kendall': {'correlation': kendall_corr, 'p_value': kendall_p},
        'Mutual Information': {'score': mi},
        'Distance Correlation': {'score': distance_corr}
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



def distance_correlation(x, y):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError('Number of samples must match')
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dCov2_xy = (A * B).sum() / (n * n)
    dCov2_xx = (A * A).sum() / (n * n)
    dCov2_yy = (B * B).sum() / (n * n)
    if dCov2_xx * dCov2_yy == 0:
        return 0.0
    return np.sqrt(dCov2_xy) / np.sqrt(np.sqrt(dCov2_xx * dCov2_yy))

