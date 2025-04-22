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