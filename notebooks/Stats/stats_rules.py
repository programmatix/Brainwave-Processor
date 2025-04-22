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

def fit_rulefit(df, feat1, feat2, find_subsets=True, remove_outliers=True, outlier_method='influence'):
    try:
        from rulefit import RuleFit
    except ImportError:
        print("RuleFit not installed. Please install with: pip install rulefit")
        return None
        
    X = df[[feat1]].values
    y = df[feat2].values
    
    outlier_mask = np.ones(len(X), dtype=bool)
    if remove_outliers:
        X_filtered, y_filtered, outlier_mask, outlier_scores, _ = detect_outliers(
            X, y, method=outlier_method,
            model_factory=lambda X, y: LinearRegression().fit(X, y)
        )
    else:
        X_filtered = X
        y_filtered = y
    
    def rulefit_factory(X_train, y_train):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        rf = RuleFit(tree_size=3, rfmode='regress')
        rf.fit(X_train_scaled, y_train, feature_names=[feat1])
        
        class RuleFitWrapper:
            def __init__(self, rf, scaler):
                self.rf = rf
                self.scaler = scaler
            
            def predict(self, X):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                X_scaled = self.scaler.transform(X)
                return self.rf.predict(X_scaled)
        
        return RuleFitWrapper(rf, scaler)
    
    scaler_X = StandardScaler()
    X_filtered_scaled = scaler_X.fit_transform(X_filtered)
    
    try:
        model = rulefit_factory(X_filtered, y_filtered)
        
        # Print the rules from RuleFit
        print("RuleFit Rules:")
        try:
            rules = model.rf.get_rules()
            rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
            print(rules.head(10)[["rule", "coef", "importance"]])
        except Exception as rule_error:
            print(f"Could not extract rules: {rule_error}")
        
        y_pred_filtered = model.predict(X_filtered)
        errors_filtered = np.abs(y_filtered - y_pred_filtered)
        
        metrics = {
            'r2': r2_score(y_filtered, y_pred_filtered),
            'rmse': np.sqrt(mean_squared_error(y_filtered, y_pred_filtered)),
            'mae': mean_absolute_error(y_filtered, y_pred_filtered),
            'mse': mean_squared_error(y_filtered, y_pred_filtered)
        }
        
        x_new = create_prediction_grid(X.ravel())
        y_new = model.predict(x_new)
        
        best_subset = None
        subset_results = None
        cluster_models = {}
        
        if find_subsets:
            df_filtered = df.iloc[np.where(outlier_mask)[0]]
            
            def rulefit_subset_factory(X_train, y_train):
                scaler_subset = StandardScaler()
                X_train_scaled = scaler_subset.fit_transform(X_train)
                
                rf_model = RuleFit(tree_size=2, rfmode='regress')
                rf_model.fit(X_train_scaled, y_train, feature_names=[feat1])
                
                class RuleFitWrapper:
                    def __init__(self, rf, scaler):
                        self.rf = rf
                        self.scaler = scaler
                    
                    def predict(self, X):
                        if X.ndim == 1:
                            X = X.reshape(-1, 1)
                        X_scaled = self.scaler.transform(X)
                        return self.rf.predict(X_scaled)
                
                return RuleFitWrapper(rf_model, scaler_subset)
                
            subset_results = find_optimal_data_subsets(
                df_filtered, feat1, feat2, rulefit_subset_factory, 
                "RuleFit", use_2d_clustering=True
            )
            
            if subset_results:
                best_subset = df_filtered.copy()
                best_subset['cluster'] = subset_results['gmm_labels']
                
                if 'cluster_metrics' in subset_results and subset_results['cluster_metrics']:
                    for cluster_info in subset_results['cluster_metrics']:
                        cluster_id = cluster_info['label']
                        cluster_mask = best_subset['cluster'] == cluster_id
                        
                        if np.sum(cluster_mask) >= 5:
                            cluster_X = best_subset.loc[cluster_mask, feat1].values.reshape(-1, 1)
                            cluster_y = best_subset.loc[cluster_mask, feat2].values
                            cluster_models[cluster_id] = rulefit_subset_factory(cluster_X, cluster_y)
    
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        plot_regression_relationship(
            model_name="RuleFit", 
            X=X_filtered, 
            y=y_filtered, 
            y_pred=y_pred_filtered,
            x_new=x_new,
            y_new=y_new,
            ax=axes[0],
            metrics=metrics
        )
        
        if find_subsets and best_subset is not None:
            subset_X = best_subset[feat1].values.reshape(-1, 1)
            subset_y = best_subset[feat2].values
            
            if cluster_models:
                axes[1].set_title("RuleFit (Per-Cluster Models)")
                
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
                
                if 'cluster_metrics' in subset_results and subset_results['cluster_metrics']:
                    fig_metrics, ax_bars = plt.subplots(figsize=(12, 6))
                    
                    cluster_metrics = subset_results['cluster_metrics']
                    cluster_labels = [f"C{m['label']}\n(n={m['size']})" for m in cluster_metrics]
                    
                    # Safely get metrics with .get() method
                    cv_r2 = [m.get('cv_r2', 0) for m in cluster_metrics]
                    train_r2 = [m.get('train_r2', 0) for m in cluster_metrics]
                    cv_rmse = [m.get('cv_rmse', 0) for m in cluster_metrics]
                    train_rmse = [m.get('train_rmse', 0) for m in cluster_metrics]
                    
                    x = np.arange(len(cluster_labels))
                    width = 0.2
                    bar_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(cluster_labels)]
                    
                    rects1 = ax_bars.bar(x - 1.5*width, cv_r2, width, label='CV R²', color=bar_colors)
                    rects2 = ax_bars.bar(x - 0.5*width, train_r2, width, label='Train R²', color=bar_colors, alpha=0.7)
                    rects3 = ax_bars.bar(x + 0.5*width, cv_rmse, width, label='CV RMSE', color=bar_colors, hatch='///', alpha=0.9)
                    rects4 = ax_bars.bar(x + 1.5*width, train_rmse, width, label='Train RMSE', color=bar_colors, hatch='///', alpha=0.6)
                    
                    ax_bars.set_xlabel('Clusters')
                    ax_bars.set_title('RuleFit Performance Across Clusters')
                    ax_bars.set_xticks(x)
                    ax_bars.set_xticklabels(cluster_labels)
                    ax_bars.legend()
                    
                    def add_bar_labels(bars, format_str="{:.2f}"):
                        for bar in bars:
                            height = bar.get_height()
                            ax_bars.annotate(format_str.format(height),
                                            xy=(bar.get_x() + bar.get_width() / 2, height),
                                            xytext=(0, 3),
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
                axes[1].scatter(subset_X, subset_y, alpha=0.6)
                axes[1].plot(x_new, y_new, 'r-', linewidth=2)
                axes[1].set_title(f"RuleFit (Single Model)")
                axes[1].set_xlabel(feat1)
                axes[1].set_ylabel(feat2)
                axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        results = {
            'model': model,
            'metrics': metrics,
            'filtered_mask': outlier_mask,
            'subset_results': subset_results,
            'cluster_models': cluster_models,
            'best_subset': best_subset
        }
        
        return results
        
    except Exception as e:
        print(f"Error in RuleFit fitting: {e}")
        return None

def fit_ripper(df, feat1, feat2, find_subsets=True, remove_outliers=True, outlier_method='influence'):
    try:
        from wittgenstein import RIPPER
    except ImportError:
        print("Wittgenstein (RIPPER) not installed. Please install with: pip install wittgenstein")
        return None
        
    X = df[[feat1]].values
    y = df[feat2].values
    
    from sklearn.preprocessing import KBinsDiscretizer
    
    outlier_mask = np.ones(len(X), dtype=bool)
    if remove_outliers:
        X_filtered, y_filtered, outlier_mask, outlier_scores, _ = detect_outliers(
            X, y, method=outlier_method,
            model_factory=lambda X, y: LinearRegression().fit(X, y)
        )
    else:
        X_filtered = X
        y_filtered = y
    
    def ripper_factory(X_train, y_train):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        y_train_binned = disc.fit_transform(y_train.reshape(-1, 1)).ravel().astype(int)
        
        df_train = pd.DataFrame(X_train_scaled, columns=[feat1])
        df_train['target'] = y_train_binned
        
        # Find max bin value to use as positive class
        max_bin = np.max(y_train_binned)
        
        rip = RIPPER()
        rip.fit(df_train, class_feat='target', pos_class=max_bin)
        
        class RIPPERWrapper:
            def __init__(self, ripper, scaler, disc, input_feat, orig_y):
                self.ripper = ripper
                self.scaler = scaler
                self.disc = disc
                self.input_feat = input_feat
                
                self.bin_to_value = {}
                binned = disc.transform(orig_y.reshape(-1, 1)).ravel().astype(int)
                for bin_val in np.unique(binned):
                    mask = binned == bin_val
                    self.bin_to_value[bin_val] = np.mean(orig_y[mask])
            
            def predict(self, X):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                X_scaled = self.scaler.transform(X)
                
                df_pred = pd.DataFrame(X_scaled, columns=[self.input_feat])
                
                y_pred_binned = self.ripper.predict(df_pred)
                
                y_pred = np.array([self.bin_to_value.get(int(p), np.mean(list(self.bin_to_value.values()))) 
                                  for p in y_pred_binned])
                return y_pred
        
        return RIPPERWrapper(rip, scaler, disc, feat1, y_train)
    
    try:
        model = ripper_factory(X_filtered, y_filtered)
        
        # Print the rules from RIPPER
        print("RIPPER Rules:")
        try:
            print(model.ripper.ruleset_)
        except Exception as rule_error:
            print(f"Could not access RIPPER ruleset: {rule_error}")
        
        y_pred_filtered = model.predict(X_filtered)
        errors_filtered = np.abs(y_filtered - y_pred_filtered)
        
        metrics = {
            'r2': r2_score(y_filtered, y_pred_filtered),
            'rmse': np.sqrt(mean_squared_error(y_filtered, y_pred_filtered)),
            'mae': mean_absolute_error(y_filtered, y_pred_filtered),
            'mse': mean_squared_error(y_filtered, y_pred_filtered)
        }
        
        x_new = create_prediction_grid(X.ravel())
        y_new = model.predict(x_new)
        
        best_subset = None
        subset_results = None
        cluster_models = {}
        
        if find_subsets:
            df_filtered = df.iloc[np.where(outlier_mask)[0]]
            
            def ripper_subset_factory(X_train, y_train):
                return ripper_factory(X_train, y_train)
                
            subset_results = find_optimal_data_subsets(
                df_filtered, feat1, feat2, ripper_subset_factory, 
                "RIPPER", use_2d_clustering=True
            )
            
            if subset_results:
                best_subset = df_filtered.copy()
                best_subset['cluster'] = subset_results['gmm_labels']
                
                if 'cluster_metrics' in subset_results and subset_results['cluster_metrics']:
                    for cluster_info in subset_results['cluster_metrics']:
                        cluster_id = cluster_info['label']
                        cluster_mask = best_subset['cluster'] == cluster_id
                        
                        if np.sum(cluster_mask) >= 5:
                            cluster_X = best_subset.loc[cluster_mask, feat1].values.reshape(-1, 1)
                            cluster_y = best_subset.loc[cluster_mask, feat2].values
                            cluster_models[cluster_id] = ripper_subset_factory(cluster_X, cluster_y)
    
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        plot_regression_relationship(
            model_name="RIPPER", 
            X=X_filtered, 
            y=y_filtered, 
            y_pred=y_pred_filtered,
            x_new=x_new,
            y_new=y_new,
            ax=axes[0],
            metrics=metrics
        )
        
        if find_subsets and best_subset is not None:
            subset_X = best_subset[feat1].values.reshape(-1, 1)
            subset_y = best_subset[feat2].values
            
            if cluster_models:
                axes[1].set_title("RIPPER (Per-Cluster Models)")
                
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
                
                if 'cluster_metrics' in subset_results and subset_results['cluster_metrics']:
                    fig_metrics, ax_bars = plt.subplots(figsize=(12, 6))
                    
                    cluster_metrics = subset_results['cluster_metrics']
                    cluster_labels = [f"C{m['label']}\n(n={m['size']})" for m in cluster_metrics]
                    
                    # Safely get metrics with .get() method
                    cv_r2 = [m.get('cv_r2', 0) for m in cluster_metrics]
                    train_r2 = [m.get('train_r2', 0) for m in cluster_metrics]
                    cv_rmse = [m.get('cv_rmse', 0) for m in cluster_metrics]
                    train_rmse = [m.get('train_rmse', 0) for m in cluster_metrics]
                    
                    x = np.arange(len(cluster_labels))
                    width = 0.2
                    bar_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(cluster_labels)]
                    
                    rects1 = ax_bars.bar(x - 1.5*width, cv_r2, width, label='CV R²', color=bar_colors)
                    rects2 = ax_bars.bar(x - 0.5*width, train_r2, width, label='Train R²', color=bar_colors, alpha=0.7)
                    rects3 = ax_bars.bar(x + 0.5*width, cv_rmse, width, label='CV RMSE', color=bar_colors, hatch='///', alpha=0.9)
                    rects4 = ax_bars.bar(x + 1.5*width, train_rmse, width, label='Train RMSE', color=bar_colors, hatch='///', alpha=0.6)
                    
                    ax_bars.set_xlabel('Clusters')
                    ax_bars.set_title('RIPPER Performance Across Clusters')
                    ax_bars.set_xticks(x)
                    ax_bars.set_xticklabels(cluster_labels)
                    ax_bars.legend()
                    
                    def add_bar_labels(bars, format_str="{:.2f}"):
                        for bar in bars:
                            height = bar.get_height()
                            ax_bars.annotate(format_str.format(height),
                                            xy=(bar.get_x() + bar.get_width() / 2, height),
                                            xytext=(0, 3),
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
                axes[1].scatter(subset_X, subset_y, alpha=0.6)
                axes[1].plot(x_new, y_new, 'r-', linewidth=2)
                axes[1].set_title(f"RIPPER (Single Model)")
                axes[1].set_xlabel(feat1)
                axes[1].set_ylabel(feat2)
                axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        results = {
            'model': model,
            'metrics': metrics,
            'filtered_mask': outlier_mask,
            'subset_results': subset_results,
            'cluster_models': cluster_models,
            'best_subset': best_subset
        }
        
        return results
        
    except Exception as e:
        print(f"Error in RIPPER fitting: {e}")
        return None
