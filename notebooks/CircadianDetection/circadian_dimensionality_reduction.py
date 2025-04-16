import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import wittgenstein as wt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime, timedelta

def apply_tsne_to_circadian(df_lep, n_components=2, perplexity=30, learning_rate='auto', target_col=None):
    """
    Apply t-SNE dimensionality reduction to circadian data.
    
    Parameters:
    -----------
    df_lep : pandas DataFrame
        DataFrame with circadian data
    n_components : int, default=2
        Dimension of the embedded space
    perplexity : float, default=30
        The perplexity is related to the number of nearest neighbors
    learning_rate : float or 'auto', default='auto'
        The learning rate for t-SNE optimization
    target_col : str, default=None
        Optional target column for visualization
        
    Returns:
    --------
    tuple:
        - df_tsne: DataFrame with t-SNE components
        - tsne_model: Fitted t-SNE model
    """
    from sklearn.manifold import TSNE
    
    # Prepare data - select numeric columns only
    df = df_lep.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle datetime columns - extract hour of day as a feature
    time_cols = [col for col in df.columns if ':datetime' in col or ':time' in col]
    for col in time_cols:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            # Extract hour of day as a new feature
            df[f"{col}_hour"] = df[col].dt.hour + df[col].dt.minute/60
            numeric_cols.append(f"{col}_hour")
    
    # Remove target column from features if specified
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Check if we have enough data
    if len(df) < 10:
        print("Not enough data for t-SNE (minimum 10 samples recommended)")
        return None, None
    
    # Select only complete rows for the selected features
    X = df[numeric_cols].dropna()
    print(f"Applying t-SNE to {X.shape[0]} samples with {X.shape[1]} features")
    
    # Apply t-SNE
    tsne_model = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(X)-1),  # Perplexity must be < n_samples
        learning_rate=learning_rate,
        random_state=42
    )
    
    tsne_result = tsne_model.fit_transform(X)
    
    # Create a DataFrame with t-SNE results
    df_tsne = pd.DataFrame(
        data=tsne_result,
        columns=[f'tsne_{i+1}' for i in range(n_components)],
        index=X.index
    )
    
    # Add the target column if specified
    if target_col and target_col in df.columns:
        df_tsne[target_col] = df.loc[X.index, target_col]
    
    # Visualize the results if 2D or 3D
    if n_components in (2, 3):
        visualize_embedding(df_tsne, target_col, method='t-SNE')
    
    return df_tsne, tsne_model

def apply_pca_to_circadian(df_lep, n_components=None, target_col=None):
    """
    Apply PCA dimensionality reduction to circadian data.
    
    Parameters:
    -----------
    df_lep : pandas DataFrame
        DataFrame with circadian data
    n_components : int or None, default=None
        Number of components to keep. If None, keep all components.
    target_col : str, default=None
        Optional target column for visualization
        
    Returns:
    --------
    tuple:
        - df_pca: DataFrame with PCA components
        - pca_model: Fitted PCA model
        - explained_variance_ratio: Explained variance ratio per component
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data - select numeric columns only
    df = df_lep.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle datetime columns - extract hour of day as a feature
    time_cols = [col for col in df.columns if ':datetime' in col or ':time' in col]
    for col in time_cols:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_hour"] = df[col].dt.hour + df[col].dt.minute/60
            numeric_cols.append(f"{col}_hour")
    
    # Remove target column from features if specified
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Select only complete rows for the selected features
    X = df[numeric_cols].dropna()
    print(f"Applying PCA to {X.shape[0]} samples with {X.shape[1]} features")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # If n_components is None, use all features
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])
    
    # Apply PCA
    pca_model = PCA(n_components=n_components, random_state=42)
    pca_result = pca_model.fit_transform(X_scaled)
    
    # Create a DataFrame with PCA results
    df_pca = pd.DataFrame(
        data=pca_result,
        columns=[f'pc_{i+1}' for i in range(pca_model.n_components_)],
        index=X.index
    )
    
    # Add the target column if specified
    if target_col and target_col in df.columns:
        df_pca[target_col] = df.loc[X.index, target_col]
    
    # Calculate explained variance
    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Print explained variance
    print("\nExplained variance by principal components:")
    for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
        print(f"PC{i+1}: {var:.3f} ({cum_var:.3f} cumulative)")
    
    # Plot explained variance
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'ro-')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.title('Scree Plot - PCA Explained Variance')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Visualize the first two components if available
    if pca_model.n_components_ >= 2:
        visualize_embedding(df_pca.iloc[:, :2], target_col, method='PCA')
    
    # Feature importance
    if hasattr(pca_model, 'components_'):
        feature_importance = pd.DataFrame(
            data=np.abs(pca_model.components_),
            columns=numeric_cols,
            index=[f'PC{i+1}' for i in range(pca_model.n_components_)]
        )
        
        # Display top contributing features for each PC
        print("\nTop contributing features by principal component:")
        for i in range(min(3, pca_model.n_components_)):
            pc = f'PC{i+1}'
            top_features = feature_importance.loc[pc].nlargest(5)
            print(f"\n{pc} top features:")
            for feature, importance in top_features.items():
                print(f"  {feature}: {importance:.3f}")
    
    return df_pca, pca_model, explained_variance

def apply_factor_analysis_to_circadian(df_lep, n_factors=5, rotation='varimax', target_col=None):
    """
    Apply Factor Analysis to circadian data.
    
    Parameters:
    -----------
    df_lep : pandas DataFrame
        DataFrame with circadian data
    n_factors : int, default=5
        Number of factors to extract
    rotation : str, default='varimax'
        Rotation method ('varimax', 'quartimax', 'promax', etc.)
    target_col : str, default=None
        Optional target column for visualization
        
    Returns:
    --------
    tuple:
        - df_fa: DataFrame with factor scores
        - fa_model: Fitted Factor Analysis model
        - loadings: DataFrame with factor loadings
    """
    from sklearn.decomposition import FactorAnalysis
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data - select numeric columns only
    df = df_lep.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle datetime columns - extract hour of day as a feature
    time_cols = [col for col in df.columns if ':datetime' in col or ':time' in col]
    for col in time_cols:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_hour"] = df[col].dt.hour + df[col].dt.minute/60
            numeric_cols.append(f"{col}_hour")
    
    # Remove target column from features if specified
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Select only complete rows for the selected features
    X = df[numeric_cols].dropna()
    print(f"Applying Factor Analysis to {X.shape[0]} samples with {X.shape[1]} features")
    
    # Check if sample size is sufficient
    if len(X) < n_factors * 5:
        print(f"Warning: Sample size ({len(X)}) is less than recommended (5 × {n_factors} = {n_factors * 5})")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Factor Analysis
    fa_model = FactorAnalysis(n_components=n_factors, random_state=42, rotation=rotation)
    fa_result = fa_model.fit_transform(X_scaled)
    
    # Create a DataFrame with factor scores
    df_fa = pd.DataFrame(
        data=fa_result,
        columns=[f'factor_{i+1}' for i in range(n_factors)],
        index=X.index
    )
    
    # Add the target column if specified
    if target_col and target_col in df.columns:
        df_fa[target_col] = df.loc[X.index, target_col]
    
    # Create a DataFrame with factor loadings
    loadings = pd.DataFrame(
        data=fa_model.components_.T,
        columns=[f'factor_{i+1}' for i in range(n_factors)],
        index=numeric_cols
    )
    
    # Display factor loadings
    print("\nFactor loadings:")
    pd.set_option('display.max_rows', 20)
    print(loadings)
    
    # Plot factor loadings heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(loadings, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Loading strength')
    plt.xticks(range(n_factors), [f'Factor {i+1}' for i in range(n_factors)])
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title('Factor Analysis Loadings')
    plt.tight_layout()
    plt.show()
    
    # Visualize the first two factors if available
    if n_factors >= 2:
        visualize_embedding(df_fa.iloc[:, :2], target_col, method='Factor Analysis')
    
    return df_fa, fa_model, loadings

def apply_ica_to_circadian(df_lep, n_components=5, target_col=None):
    """
    Apply Independent Component Analysis to circadian data.
    
    Parameters:
    -----------
    df_lep : pandas DataFrame
        DataFrame with circadian data
    n_components : int, default=5
        Number of independent components
    target_col : str, default=None
        Optional target column for visualization
        
    Returns:
    --------
    tuple:
        - df_ica: DataFrame with independent components
        - ica_model: Fitted ICA model
        - mixing_matrix: Mixing matrix from ICA
    """
    from sklearn.decomposition import FastICA
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data - select numeric columns only
    df = df_lep.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle datetime columns - extract hour of day as a feature
    time_cols = [col for col in df.columns if ':datetime' in col or ':time' in col]
    for col in time_cols:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_hour"] = df[col].dt.hour + df[col].dt.minute/60
            numeric_cols.append(f"{col}_hour")
    
    # Remove target column from features if specified
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Select only complete rows for the selected features
    X = df[numeric_cols].dropna()
    print(f"Applying ICA to {X.shape[0]} samples with {X.shape[1]} features")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply ICA
    ica_model = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    ica_result = ica_model.fit_transform(X_scaled)
    
    # Create a DataFrame with ICA results
    df_ica = pd.DataFrame(
        data=ica_result,
        columns=[f'ic_{i+1}' for i in range(n_components)],
        index=X.index
    )
    
    # Add the target column if specified
    if target_col and target_col in df.columns:
        df_ica[target_col] = df.loc[X.index, target_col]
    
    # Calculate mixing matrix
    mixing_matrix = pd.DataFrame(
        data=ica_model.mixing_,
        columns=[f'ic_{i+1}' for i in range(n_components)],
        index=numeric_cols
    )
    
    # Display top contributing features for each IC
    print("\nFeature contributions to independent components:")
    for ic in mixing_matrix.columns:
        abs_contributions = np.abs(mixing_matrix[ic])
        top_features = abs_contributions.nlargest(5)
        print(f"\n{ic} top contributing features:")
        for feature, contribution in top_features.items():
            print(f"  {feature}: {contribution:.3f}")
    
    # Visualize the first two components if available
    if n_components >= 2:
        visualize_embedding(df_ica.iloc[:, :2], target_col, method='ICA')
    
    return df_ica, ica_model, mixing_matrix

def apply_nmf_to_circadian(df_lep, n_components=5, target_col=None, min_value=0):
    """
    Apply Non-negative Matrix Factorization to circadian data.
    
    Parameters:
    -----------
    df_lep : pandas DataFrame
        DataFrame with circadian data
    n_components : int, default=5
        Number of components
    target_col : str, default=None
        Optional target column for visualization
    min_value : float, default=0
        Minimum value to use for features (ensuring non-negativity)
        
    Returns:
    --------
    tuple:
        - df_nmf: DataFrame with NMF components
        - nmf_model: Fitted NMF model
        - components_df: DataFrame with NMF components
    """
    from sklearn.decomposition import NMF
    from sklearn.preprocessing import MinMaxScaler
    
    # Prepare data - select numeric columns only
    df = df_lep.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle datetime columns - extract hour of day as a feature
    time_cols = [col for col in df.columns if ':datetime' in col or ':time' in col]
    for col in time_cols:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_hour"] = df[col].dt.hour + df[col].dt.minute/60
            numeric_cols.append(f"{col}_hour")
    
    # Remove target column from features if specified
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Select only complete rows for the selected features
    X = df[numeric_cols].dropna()
    print(f"Applying NMF to {X.shape[0]} samples with {X.shape[1]} features")
    
    # Scale data to be non-negative
    scaler = MinMaxScaler(feature_range=(min_value, 1))
    X_scaled = scaler.fit_transform(X)
    
    # Apply NMF
    nmf_model = NMF(n_components=n_components, random_state=42, max_iter=1000)
    nmf_result = nmf_model.fit_transform(X_scaled)
    
    # Create a DataFrame with NMF results
    df_nmf = pd.DataFrame(
        data=nmf_result,
        columns=[f'nmf_{i+1}' for i in range(n_components)],
        index=X.index
    )
    
    # Add the target column if specified
    if target_col and target_col in df.columns:
        df_nmf[target_col] = df.loc[X.index, target_col]
    
    # Get components
    components_df = pd.DataFrame(
        data=nmf_model.components_,
        columns=numeric_cols,
        index=[f'nmf_{i+1}' for i in range(n_components)]
    )
    
    # Display components
    print("\nNMF components (feature weights):")
    for comp in components_df.index:
        top_features = components_df.loc[comp].nlargest(5)
        print(f"\n{comp} top features:")
        for feature, weight in top_features.items():
            print(f"  {feature}: {weight:.3f}")
    
    # Visualize the first two components if available
    if n_components >= 2:
        visualize_embedding(df_nmf.iloc[:, :2], target_col, method='NMF')
    
    return df_nmf, nmf_model, components_df

def visualize_embedding(embedding_df, target_col=None, method='t-SNE'):
    """
    Visualize embeddings from dimensionality reduction techniques.
    
    Parameters:
    -----------
    embedding_df : pandas DataFrame
        DataFrame with the embedding coordinates
    target_col : str, default=None
        Target column name for coloring the points
    method : str, default='t-SNE'
        The dimensionality reduction method name for the plot title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get the first two embedding dimensions
    x_col = embedding_df.columns[0]
    y_col = embedding_df.columns[1]
    
    # Set up the plot
    plt.figure(figsize=(10, 8))
    
    # Color by target if provided
    if target_col and target_col in embedding_df.columns:
        # Check if target is numeric or categorical
        if pd.api.types.is_numeric_dtype(embedding_df[target_col]):
            # Numeric target - use a continuous colormap
            scatter = plt.scatter(
                embedding_df[x_col], 
                embedding_df[y_col],
                c=embedding_df[target_col],
                cmap='viridis',
                alpha=0.7,
                s=50
            )
            plt.colorbar(scatter, label=target_col)
        else:
            # Categorical target - use discrete colors
            sns.scatterplot(
                x=x_col, 
                y=y_col,
                hue=target_col,
                data=embedding_df,
                palette='tab10',
                alpha=0.7,
                s=50
            )
            plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # No target - simple scatter plot
        plt.scatter(
            embedding_df[x_col], 
            embedding_df[y_col],
            alpha=0.7,
            s=50
        )
    
    # Add labels and title
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{method} - 2D Embedding of Circadian Data')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # If we have 3 dimensions, create a 3D plot
    if len(embedding_df.columns) > 2 and embedding_df.columns[2].startswith(('tsne_', 'pc_', 'factor_', 'ic_', 'nmf_')):
        from mpl_toolkits.mplot3d import Axes3D
        
        z_col = embedding_df.columns[2]
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by target if provided
        if target_col and target_col in embedding_df.columns:
            if pd.api.types.is_numeric_dtype(embedding_df[target_col]):
                scatter = ax.scatter(
                    embedding_df[x_col],
                    embedding_df[y_col],
                    embedding_df[z_col],
                    c=embedding_df[target_col],
                    cmap='viridis',
                    alpha=0.7,
                    s=50
                )
                fig.colorbar(scatter, ax=ax, label=target_col)
            else:
                # Get unique categories
                categories = embedding_df[target_col].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                
                for i, category in enumerate(categories):
                    mask = embedding_df[target_col] == category
                    ax.scatter(
                        embedding_df.loc[mask, x_col],
                        embedding_df.loc[mask, y_col],
                        embedding_df.loc[mask, z_col],
                        color=colors[i],
                        label=category,
                        alpha=0.7,
                        s=50
                    )
                ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(
                embedding_df[x_col],
                embedding_df[y_col],
                embedding_df[z_col],
                alpha=0.7,
                s=50
            )
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f'{method} - 3D Embedding of Circadian Data')  
        plt.tight_layout()
        plt.show()
