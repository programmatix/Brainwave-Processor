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
from stats_shared import plot_regression_relationship, create_prediction_grid
from stats_clustering import find_optimal_data_subsets

def fit_pytorch_neural_net(df, feat1, feat2, hidden_layers=[10, 5], learning_rate=0.01, epochs=100, find_subsets=True, remove_outliers=True, outlier_method='influence'):
    """
    Fit a PyTorch neural network to model the relationship between two variables.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    feat1 : str
        Column name for feature 1
    feat2 : str
        Column name for feature 2
    hidden_layers : list
        List of integers specifying the number of neurons in each hidden layer
    learning_rate : float
        Learning rate for optimizer
    epochs : int
        Number of training epochs
    find_subsets : bool
        Whether to find optimal data subsets
    remove_outliers : bool
        Whether to remove outliers before fitting
    outlier_method : str
        Method for outlier detection ('residual', 'distance', 'influence')
        
    Returns:
    --------
    dict
        Dictionary containing model, metrics, and other results
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    X = df[[feat1]].values
    y = df[feat2].values.reshape(-1, 1)
    
    # Define neural network model
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size=1):
            super(NeuralNet, self).__init__()
            self.layers = nn.ModuleList()
            
            # Input layer to first hidden layer
            self.layers.append(nn.Linear(input_size, hidden_layers[0]))
            
            # Hidden layers
            for i in range(len(hidden_layers) - 1):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            
            # Last hidden layer to output layer
            self.layers.append(nn.Linear(hidden_layers[-1], output_size))
            
            # Activation function
            self.relu = nn.ReLU()
            
        def forward(self, x):
            for i in range(len(self.layers) - 1):
                x = self.relu(self.layers[i](x))
            
            # No activation for output layer (regression)
            x = self.layers[-1](x)
            return x
    
    # Wrapper class for integration with scikit-learn style API
    class TorchNNWrapper:
        def __init__(self, input_size=1, hidden_layers=[10, 5], output_size=1, lr=0.01, epochs=1000):
            self.input_size = input_size
            self.hidden_layers = hidden_layers
            self.output_size = output_size
            self.lr = lr
            self.epochs = epochs
            self.model = NeuralNet(input_size, hidden_layers, output_size)
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
        def fit(self, X, y):
            # Scale the data
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.FloatTensor(y_scaled).to(self.device)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                # Forward pass
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Print loss every 100 epochs
                if (epoch+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.6f}')
                    
            return self
            
        def predict(self, X):
            # Ensure X is 2D
            if X.ndim == 1:
                X = X.reshape(-1, 1)
                
            # Scale the input
            X_scaled = self.scaler_X.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Switch to evaluation mode
            self.model.eval()
            
            # Make predictions
            with torch.no_grad():
                y_scaled_pred = self.model(X_tensor).cpu().numpy()
                
            # Inverse transform to get predictions in original scale
            y_pred = self.scaler_y.inverse_transform(y_scaled_pred)
            return y_pred.flatten()
    
    # Detect and remove outliers if requested
    outlier_mask = np.ones(len(X), dtype=bool)
    outlier_scores = np.zeros(len(X))
    
    if remove_outliers:
        def simple_model_factory(X_train, y_train):
            nn_simple = TorchNNWrapper(input_size=1, hidden_layers=[5], epochs=500)
            nn_simple.fit(X_train, y_train)
            return nn_simple
            
        _, _, outlier_mask, outlier_scores, _ = detect_outliers(
            X, y, method=outlier_method, max_remove_percent=10, 
            model_factory=simple_model_factory
        )
    
    # Use filtered data for model fitting
    X_filtered = X[outlier_mask]
    y_filtered = y[outlier_mask]
    
    # Create and fit the PyTorch model
    model = TorchNNWrapper(input_size=1, hidden_layers=hidden_layers, 
                          lr=learning_rate, epochs=epochs)
    model.fit(X_filtered, y_filtered)
    
    # Make predictions
    y_pred = model.predict(X_filtered)
    
    # Calculate metrics
    metrics = {
        'r2': r2_score(y_filtered, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_filtered, y_pred)),
        'mae': mean_absolute_error(y_filtered, y_pred),
        'mse': mean_squared_error(y_filtered, y_pred)
    }
    
    # Create prediction grid for smooth curve
    x_new = create_prediction_grid(X.ravel())
    y_new = model.predict(x_new)
    
    # Find optimal data subsets if requested
    best_subset = None
    subset_results = None
    cluster_models = {}
    
    if find_subsets:
        # Use only the filtered data for subset discovery
        df_filtered = df.iloc[np.where(outlier_mask)[0]]
        
        # Create a model factory for NN
        def nn_subset_factory(X_train, y_train):
            # Create a smaller NN for each subset to avoid overfitting
            nn_subset = TorchNNWrapper(input_size=1, hidden_layers=[max(3, hidden_layers[0]//2)], 
                                      lr=learning_rate, epochs=min(500, epochs))
            nn_subset.fit(X_train, y_train)
            return nn_subset
            
        subset_results = find_optimal_data_subsets(
            df_filtered, feat1, feat2, nn_subset_factory, 
            f"Neural Network ({hidden_layers})", use_2d_clustering=True
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
                        cluster_models[cluster_id] = nn_subset_factory(cluster_X, cluster_y)
    
    # Create main visualization figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot the main model on the first axis
    uncertainties = np.full_like(y_new, metrics['rmse'])
    plot_regression_relationship(
        model_name=f"Neural Network ({hidden_layers})", 
        X=X_filtered, 
        y=y_filtered, 
        y_pred=y_pred,
        x_new=x_new,
        y_new=y_new,
        ax=axes[0],
        metrics=metrics,
        uncertainties=uncertainties
    )
    
    # Show regional performance in a separate figure
    regional_metrics = analyze_regional_performance(X_filtered, y_filtered, y_pred)
    plot_regional_performance(regional_metrics, f"Neural Network ({hidden_layers})")
    
    # Plot per-cluster models on the second axis if available
    if find_subsets and best_subset is not None:
        subset_X = best_subset[feat1].values.reshape(-1, 1)
        subset_y = best_subset[feat2].values
        
        if cluster_models:
            axes[1].set_title(f"Neural Network (Per-Cluster Models)")
            
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
                ax_bars.set_title(f"Neural Network Performance within Clusters", fontsize=14)
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
            axes[1].scatter(X_filtered, y_filtered, color='blue', alpha=0.6, s=50, edgecolors='k')
            axes[1].plot(x_new, y_new, color='red', linewidth=2.5, label='Global Model')
            axes[1].set_title(f"Neural Network ({hidden_layers})")
            axes[1].legend()
    
    # Configure axes general settings
    for ax in axes:
        ax.set_xlabel(feat1, fontsize=12)
        ax.set_ylabel(feat2, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"Neural Network ({hidden_layers}) Results:")
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
    
    return results
