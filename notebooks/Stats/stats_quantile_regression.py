import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

def plot_quantile_regression(df, feature_col, target_col, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """
    Perform quantile regression on the provided dataframe and plot the results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the feature and target columns
    feature_col : str
        Name of the feature column
    target_col : str
        Name of the target column
    quantiles : list, default=[0.1, 0.25, 0.5, 0.75, 0.9]
        List of quantiles to fit
    """
    X = df[feature_col].values.reshape(-1, 1)
    y = df[target_col].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Data points')
    
    X_range = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    X_range_scaled = scaler.transform(X_range)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles)))
    
    for q, color in zip(quantiles, colors):
        qr = QuantileRegressor(quantile=q, alpha=0.5, solver='highs')
        qr.fit(X_scaled, y)
        
        y_pred = qr.predict(X_range_scaled)
        
        plt.plot(X_range, y_pred, color=color, 
                 label=f'Quantile: {q}', linewidth=2)
    
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.title(f'Quantile Regression: {feature_col} vs {target_col}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
