import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import wittgenstein as wt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime, timedelta
from IPython.display import display, HTML
import os
import openai
import dotenv
import concurrent.futures
import time

dotenv.load_dotenv()

def name_components_with_openai(feature_importance_df, feature_thresholds=None, model="gpt-3.5-turbo", method_prefix=""):
    """
    Generate interpretable names for components based on their top features using OpenAI API.
    Uses parallel processing for faster results.
    
    Parameters:
    -----------
    feature_importance_df : pandas DataFrame
        DataFrame with features as columns and components as rows, containing loadings/importance values
    feature_thresholds : dict or None, default=None
        Dictionary with component names as keys and threshold values for feature importance
        If None, the top 5 features by absolute importance will be used for each component
    model : str, default="gpt-3.5-turbo"
        The OpenAI model to use (default is the most cost-effective)
    method_prefix : str, default=""
        Prefix to add to component names (e.g., "PCA", "FA", "ICA")
        
    Returns:
    --------
    dict: Component names with AI-generated descriptions
    """
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OpenAI API key not found in environment variables. Component naming skipped.")
        return {}
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Prepare the component naming tasks
    component_tasks = []
    
    # For each component, get the top features and their importance
    for component in feature_importance_df.index:
        # Get the feature loadings for this component
        loadings = feature_importance_df.loc[component]
        
        # Sort by absolute importance and take top features
        if feature_thresholds and component in feature_thresholds:
            threshold = feature_thresholds[component]
            top_features = loadings[abs(loadings) >= threshold]
        else:
            top_features = loadings.abs().sort_values(ascending=False).head(5)
            top_features = loadings[top_features.index]
        
        # Skip if no significant features
        if len(top_features) == 0:
            continue
        
        # Create a more informative prompt for the API
        feature_details = []
        for feature, importance in top_features.items():
            direction = "positive" if importance > 0 else "negative"
            feature_details.append(f"- {feature}: {importance:.3f} ({direction} correlation)")
        
        features_text = "\n".join(feature_details)
        
        prompt = f"""Based on the following features and their loadings, provide a short, meaningful name for this component:

{features_text}

The name should be concise (2-4 words maximum) and capture the essence of what these features collectively represent. 
Think about what these features might measure together in a circadian rhythm or health context.
Only return the suggested name with no additional text or explanation."""
        
        # Add task to the list
        component_tasks.append((component, prompt))
    
    component_descriptions = {}
    
    # Function to call OpenAI for a single component
    def generate_component_name(task):
        component, prompt = task
        max_retries = 3
        retry_delay = 1  # seconds
        
        for retry in range(max_retries):
            try:
                # Call the OpenAI API
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful expert in feature interpretation and dimensionality reduction."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=30,
                    temperature=0.3
                )
                
                # Get the generated name and clean it
                component_name = response.choices[0].message.content.strip()
                if component_name.startswith('"') and component_name.endswith('"'):
                    component_name = component_name[1:-1]
                
                # Add method prefix if provided
                if method_prefix:
                    component_name = f"{method_prefix} {component_name}"
                
                return component, component_name
            
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Error generating name for {component}: {str(e)}")
                    return component, f"{component} (unnamed)"
    
    # Process components in parallel
    print(f"Generating names for {len(component_tasks)} components in parallel...")
    start_time = time.time()
    
    # Use ThreadPoolExecutor for I/O-bound tasks like API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(component_tasks))) as executor:
        results = list(executor.map(generate_component_name, component_tasks))
    
    # Process results
    for component, name in results:
        if name:
            component_descriptions[component] = name
    
    elapsed_time = time.time() - start_time
    print(f"Component naming completed in {elapsed_time:.2f} seconds")
    
    return component_descriptions

def create_toggle_summary(summary_text, title="Summary"):
    """
    Creates a toggleable HTML summary section
    
    Parameters:
    -----------
    summary_text : str
        Text content for the summary (can include HTML)
    title : str, default="Summary"
        Title for the toggle button
        
    Returns:
    --------
    None (displays HTML directly)
    """
    toggle_html = f"""
    <details>
        <summary style="cursor: pointer; font-weight: bold;">{title}</summary>
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-top: 5px;">
            {summary_text}
        </div>
    </details>
    """
    display(HTML(toggle_html))

# Updated t-SNE implementation with z-scaling option
def apply_tsne_to_circadian(df_lep, n_components=2, perplexity=30, learning_rate='auto', target_col=None, use_scaling=True, verbose=True):
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
    use_scaling : bool, default=True
        Whether to apply z-scaling to normalize features before t-SNE
    verbose : bool, default=True
        Whether to display plots and print information during execution
        
    Returns:
    --------
    tuple:
        - df_tsne: DataFrame with t-SNE components
        - tsne_model: Fitted t-SNE model
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    
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
        if verbose:
            print("Not enough data for t-SNE (minimum 10 samples recommended)")
        return None, None
    
    # Select only complete rows for the selected features
    X = df[numeric_cols].dropna()
    if verbose:
        print(f"Applying t-SNE to {X.shape[0]} samples with {X.shape[1]} features")
    
    # Apply z-scaling if requested
    if use_scaling:
        if verbose:
            print("Applying z-scaling to normalize features")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Apply t-SNE
    tsne_model = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(X)-1),  # Perplexity must be < n_samples
        learning_rate=learning_rate,
        random_state=42
    )
    
    tsne_result = tsne_model.fit_transform(X_scaled)
    
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
    if n_components in (2, 3) and verbose:
        visualize_embedding(df_tsne, target_col, method='t-SNE', verbose=verbose)
    
    return df_tsne, tsne_model

def visualize_embedding(embedding_df, target_col=None, method='t-SNE', verbose=True):
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
    verbose : bool, default=True
        Whether to display the visualization (if False, no plot is created)
    """
    if not verbose:
        return
        
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
    if len(embedding_df.columns) > 2 and embedding_df.columns[2].startswith(('tsne_', 'pc_', 'factor_', 'ic_', 'nmf_', 'umap_', 'svd_')):
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

# Updated PCA implementation with factor loadings visualization
def apply_pca_to_circadian(df_lep, n_components=None, target_col=None, use_openai_naming=True, verbose=True):
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
    use_openai_naming : bool, default=True
        Whether to use OpenAI to generate interpretable names for components
    verbose : bool, default=True
        Whether to display plots and print information during execution
        
    Returns:
    --------
    tuple:
        - df_pca: DataFrame with PCA components
        - pca_model: Fitted PCA model
        - feature_importance: DataFrame with feature loadings/importance
        - component_names: Dictionary with OpenAI-generated component names (if use_openai_naming=True)
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
    if verbose:
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
    
    # Feature importance/loadings
    feature_importance = None
    component_names = {}
    
    if hasattr(pca_model, 'components_'):
        feature_importance = pd.DataFrame(
            data=pca_model.components_,
            columns=numeric_cols,
            index=[f'PC{i+1}' for i in range(pca_model.n_components_)]
        )
        
        # Use OpenAI to name components if requested
        if use_openai_naming and pca_model.n_components_ > 0:
            if verbose:
                print("Generating interpretable component names with OpenAI...")
            component_names = name_components_with_openai(feature_importance, method_prefix="PCA")
            
            # If we got component names, add them to the summary and rename dataframe columns
            if component_names and verbose:
                print("\nGenerated component names:")
                for comp, name in component_names.items():
                    print(f"  {comp}: {name}")
                    
            # Update dataframe column names with generated names
            col_mapping = {}
            for i, pc in enumerate([f'pc_{i+1}' for i in range(pca_model.n_components_)]):
                pc_key = f'PC{i+1}'
                if pc_key in component_names:
                    col_mapping[pc] = component_names[pc_key]
            
            # Rename the columns in df_pca
            df_pca.rename(columns=col_mapping, inplace=True)
    
    # Add the target column if specified
    if target_col and target_col in df.columns:
        df_pca[target_col] = df.loc[X.index, target_col]
    
    # Calculate explained variance
    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    if verbose:
        # Create summary for toggle
        pca_summary = f"""
        <h3>PCA Summary</h3>
        <p>Number of components: {pca_model.n_components_}</p>
        <p>Number of observations: {len(X)}</p>
        <p>Number of variables: {len(numeric_cols)}</p>
        
        <h4>Explained Variance:</h4>
        <table style="width:100%; border-collapse: collapse;">
        <tr>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Explained Variance</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Cumulative</th>
        </tr>
        """
        
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
            comp_name = f"PC{i+1}"
            ai_name = component_names.get(comp_name, "")
            ai_name_display = f" - {ai_name}" if ai_name else ""
            
            pca_summary += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">{comp_name}{ai_name_display}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{var:.3f} ({var*100:.1f}%)</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{cum_var:.3f} ({cum_var*100:.1f}%)</td>
            </tr>
            """
        
        pca_summary += "</table>"
        
        # Add top contributing features if available
        if hasattr(pca_model, 'components_'):
            pca_summary += "<h4>Top Contributing Features:</h4><ul>"
            for i in range(min(3, pca_model.n_components_)):
                pc = f'PC{i+1}'
                ai_name = component_names.get(pc, "")
                ai_name_display = f" - {ai_name}" if ai_name else ""
                
                top_features = abs(feature_importance.loc[pc]).nlargest(5)
                pca_summary += f"<li><strong>{pc}{ai_name_display}:</strong><ul>"
                for feature in top_features.index:
                    importance = feature_importance.loc[pc, feature]
                    direction = "+" if importance > 0 else "-"
                    pca_summary += f"<li>{feature}: {abs(importance):.3f} ({direction})</li>"
                pca_summary += "</ul></li>"
            pca_summary += "</ul>"
        
        # Display the toggle summary
        create_toggle_summary(pca_summary, title="PCA Summary")
        
        # Print explained variance
        print("\nExplained variance by principal components:")
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
            comp_name = f"PC{i+1}"
            ai_name = component_names.get(comp_name, "")
            ai_name_display = f" - {ai_name}" if ai_name else ""
            print(f"{comp_name}{ai_name_display}: {var:.3f} ({cum_var:.3f} cumulative)")
        
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
        
        # Feature importance/loadings
        if hasattr(pca_model, 'components_'):
            # Display top contributing features for each PC
            print("\nTop contributing features by principal component:")
            for i in range(min(3, pca_model.n_components_)):
                pc = f'PC{i+1}'
                ai_name = component_names.get(pc, "")
                ai_name_display = f" - {ai_name}" if ai_name else ""
                
                top_features = abs(feature_importance.loc[pc]).nlargest(5)
                print(f"\n{pc}{ai_name_display} top features:")
                for feature in top_features.index:
                    importance = feature_importance.loc[pc, feature]
                    direction = "+" if importance > 0 else "-"
                    print(f"  {feature}: {abs(importance):.3f} ({direction})")
            
            # Plot factor loadings heatmap
            plt.figure(figsize=(12, 8))
            plt.imshow(feature_importance, cmap='coolwarm', aspect='auto')
            plt.colorbar(label='Loading strength')
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
            plt.yticks(range(pca_model.n_components_), [f'PC{i+1}' for i in range(pca_model.n_components_)])
            plt.title('PCA Factor Loadings')
            plt.tight_layout()
            plt.show()
            
            # For a more detailed visualization of the top components
            if pca_model.n_components_ > 1:
                top_n = min(pca_model.n_components_, 3)
                plt.figure(figsize=(15, top_n*4))
                
                for i in range(top_n):
                    pc = f'PC{i+1}'
                    ai_name = component_names.get(pc, "")
                    title_suffix = f" - {ai_name}" if ai_name else ""
                    
                    pc_loadings = feature_importance.loc[pc]
                    # Sort loadings by absolute value
                    pc_loadings = pc_loadings.reindex(pc_loadings.abs().sort_values(ascending=False).index)
                    
                    plt.subplot(top_n, 1, i+1)
                    colors = ['red' if x < 0 else 'blue' for x in pc_loadings]
                    plt.barh(range(len(pc_loadings)), pc_loadings, color=colors)
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    plt.yticks(range(len(pc_loadings)), pc_loadings.index)
                    plt.title(f'{pc}{title_suffix} Loadings')
                    plt.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        # Visualize the first two components if available
        if pca_model.n_components_ >= 2:
            visualize_embedding(df_pca.iloc[:, :2], target_col, method='PCA', verbose=verbose)
    
    return df_pca, pca_model, feature_importance, component_names

def apply_factor_analysis_to_circadian(df_lep, n_factors=5, rotation='varimax', target_col=None, use_openai_naming=True, verbose=True):
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
    use_openai_naming : bool, default=True
        Whether to use OpenAI to generate interpretable names for factors
    verbose : bool, default=True
        Whether to display plots and print information during execution
        
    Returns:
    --------
    tuple:
        - df_fa: DataFrame with factor scores
        - fa_model: Fitted Factor Analysis model
        - loadings: DataFrame with factor loadings
        - results_dict: Dictionary containing communalities and uniqueness values
        - factor_names: Dictionary with OpenAI-generated factor names (if use_openai_naming=True)
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
    if verbose:
        print(f"Applying Factor Analysis to {X.shape[0]} samples with {X.shape[1]} features")
    
    # Check if sample size is sufficient
    if len(X) < n_factors * 5 and verbose:
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
    
    # Create a DataFrame with factor loadings
    loadings = pd.DataFrame(
        data=fa_model.components_.T,
        columns=[f'factor_{i+1}' for i in range(n_factors)],
        index=numeric_cols
    )
    
    # Calculate communalities and uniqueness
    communalities = pd.DataFrame(index=numeric_cols)
    communalities['communality'] = np.sum(loadings.values ** 2, axis=1)
    communalities['uniqueness'] = 1 - communalities['communality']
    
    # Use OpenAI to name factors if requested
    factor_names = {}
    if use_openai_naming and n_factors > 0:
        # We need to transpose loadings to match the expected format (factors as rows)
        loadings_for_naming = loadings.T
        if verbose:
            print("Generating interpretable factor names with OpenAI...")
        factor_names = name_components_with_openai(loadings_for_naming, method_prefix="FA")
        
        # If we got factor names, print them and rename dataframe columns
        if factor_names and verbose:
            print("\nGenerated factor names:")
            for factor, name in factor_names.items():
                print(f"  {factor}: {name}")
                
        # Update dataframe column names with generated names
        col_mapping = {}
        for i in range(n_factors):
            factor_col = f'factor_{i+1}'
            factor_key = f'factor_{i+1}'
            if factor_key in factor_names:
                col_mapping[factor_col] = factor_names[factor_key]
        
        # Rename the columns in df_fa
        df_fa.rename(columns=col_mapping, inplace=True)
    
    # Add the target column if specified
    if target_col and target_col in df.columns:
        df_fa[target_col] = df.loc[X.index, target_col]
    
    # Store results in a dictionary
    results_dict = {
        'communalities': communalities,
        'factor_scores': df_fa,
        'loadings': loadings,
        'factor_names': factor_names
    }
    
    if verbose:
        # Display factor loadings
        print("\nFactor loadings:")
        pd.set_option('display.max_rows', 20)
        print(loadings)
        
        # Display communalities and uniqueness
        print("\nCommunalities and Uniqueness:")
        print(communalities)
        
        # Create a summary for the toggle display
        factor_summary = f"""
        <h3>Factor Analysis Summary</h3>
        <p>Number of factors: {n_factors}</p>
        <p>Number of observations: {len(X)}</p>
        <p>Number of variables: {len(numeric_cols)}</p>
        <p>Rotation method: {rotation}</p>
        
        <h4>Factors:</h4>
        <ul>
        """
        for i in range(n_factors):
            factor = f'factor_{i+1}'
            ai_name = factor_names.get(factor, "")
            ai_name_display = f" - {ai_name}" if ai_name else ""
            
            top_features = abs(loadings[factor]).nlargest(5)
            factor_summary += f"<li><strong>Factor {i+1}{ai_name_display}:</strong><ul>"
            for feature in top_features.index:
                importance = loadings.loc[feature, factor]
                direction = "+" if importance > 0 else "-"
                factor_summary += f"<li>{feature}: {abs(importance):.3f} ({direction})</li>"
            factor_summary += "</ul></li>"
        factor_summary += "</ul>"
        
        # Show the toggle summary
        create_toggle_summary(factor_summary, title="Factor Analysis Summary")
        
        # Plot factor loadings heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(loadings, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Loading strength')
        
        # Add factor names to the x-axis labels if available
        if factor_names:
            x_labels = []
            for i in range(n_factors):
                factor = f'factor_{i+1}'
                ai_name = factor_names.get(factor, "")
                if ai_name:
                    x_labels.append(f"Factor {i+1}\n{ai_name}")
                else:
                    x_labels.append(f"Factor {i+1}")
            plt.xticks(range(n_factors), x_labels)
        else:
            plt.xticks(range(n_factors), [f'Factor {i+1}' for i in range(n_factors)])
            
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.title('Factor Analysis Loadings')
        plt.tight_layout()
        plt.show()
        
        # Visualize the first two factors if available
        if n_factors >= 2:
            visualize_embedding(df_fa.iloc[:, :2], target_col, method='Factor Analysis', verbose=verbose)
    
    return df_fa, fa_model, loadings, results_dict, factor_names

# Updated ICA implementation with factor loadings visualization
def apply_ica_to_circadian(df_lep, n_components=5, target_col=None, use_openai_naming=True, verbose=True):
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
    use_openai_naming : bool, default=True
        Whether to use OpenAI to generate interpretable names for components
    verbose : bool, default=True
        Whether to display plots and print information during execution
        
    Returns:
    --------
    tuple:
        - df_ica: DataFrame with independent components
        - ica_model: Fitted ICA model
        - mixing_matrix: DataFrame with mixing matrix and feature loadings
        - component_names: Dictionary with OpenAI-generated component names (if use_openai_naming=True)
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
    if verbose:
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
    
    # Calculate mixing matrix - this shows how features contribute to each IC
    mixing_matrix = pd.DataFrame(
        data=ica_model.mixing_,
        columns=[f'ic_{i+1}' for i in range(n_components)],
        index=numeric_cols
    )
    
    # Use OpenAI to name components if requested
    component_names = {}
    if use_openai_naming and n_components > 0:
        # We need to transpose mixing matrix to match the expected format (components as rows)
        mixing_matrix_for_naming = mixing_matrix.T
        if verbose:
            print("Generating interpretable component names with OpenAI...")
        component_names = name_components_with_openai(mixing_matrix_for_naming, method_prefix="ICA")
        
        # If we got component names, print them and rename dataframe columns
        if component_names and verbose:
            print("\nGenerated component names:")
            for comp, name in component_names.items():
                print(f"  {comp}: {name}")
                
        # Update dataframe column names with generated names
        col_mapping = {}
        for i in range(n_components):
            ic_col = f'ic_{i+1}'
            ic_key = f'ic_{i+1}'
            if ic_key in component_names:
                col_mapping[ic_col] = component_names[ic_key]
        
        # Rename the columns in df_ica
        df_ica.rename(columns=col_mapping, inplace=True)
    
    # Add the target column if specified
    if target_col and target_col in df.columns:
        df_ica[target_col] = df.loc[X.index, target_col]
    
    if verbose:
        # Create summary for toggle
        ica_summary = f"""
        <h3>ICA Summary</h3>
        <p>Number of components: {n_components}</p>
        <p>Number of observations: {len(X)}</p>
        <p>Number of variables: {len(numeric_cols)}</p>
        
        <h4>Independent Component Analysis</h4>
        <p>ICA maximizes statistical independence between components and is useful for separating mixed signals.</p>
        
        <h4>Independent Components:</h4>
        <ul>
        """
        
        for i in range(n_components):
            ic = f'ic_{i+1}'
            ai_name = component_names.get(ic, "")
            ai_name_display = f" - {ai_name}" if ai_name else ""
            
            top_features = abs(mixing_matrix[ic]).nlargest(5)
            ica_summary += f"<li><strong>IC{i+1}{ai_name_display}:</strong><ul>"
            for feature in top_features.index:
                contribution = mixing_matrix.loc[feature, ic]
                direction = "+" if contribution > 0 else "-"
                ica_summary += f"<li>{feature}: {abs(contribution):.3f} ({direction})</li>"
            ica_summary += "</ul></li>"
        
        ica_summary += "</ul>"
        
        # Display the toggle summary
        create_toggle_summary(ica_summary, title="ICA Summary")
        
        # Display top contributing features for each IC
        print("\nFeature contributions to independent components:")
        for ic in mixing_matrix.columns:
            ai_name = component_names.get(ic, "")
            ai_name_display = f" - {ai_name}" if ai_name else ""
            
            top_features = abs(mixing_matrix[ic]).nlargest(5)
            print(f"\n{ic}{ai_name_display} top features:")
            for feature in top_features.index:
                contribution = mixing_matrix.loc[feature, ic]
                direction = "+" if contribution > 0 else "-"
                print(f"  {feature}: {abs(contribution):.3f} ({direction})")
        
        # Plot factor loadings heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(mixing_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Loading strength')
        
        # Add component names to the x-axis labels if available
        if component_names:
            x_labels = []
            for i in range(n_components):
                ic = f'ic_{i+1}'
                ai_name = component_names.get(ic, "")
                if ai_name:
                    x_labels.append(f"IC{i+1}\n{ai_name}")
                else:
                    x_labels.append(f"IC{i+1}")
            plt.xticks(range(n_components), x_labels)
        else:
            plt.xticks(range(n_components), [f'IC{i+1}' for i in range(n_components)])
            
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.title('ICA Mixing Matrix (Feature Loadings)')
        plt.tight_layout()
        plt.show()
        
        # For a more detailed visualization of each component
        plt.figure(figsize=(15, n_components*3))
        
        for i in range(n_components):
            ic = f'ic_{i+1}'
            ai_name = component_names.get(ic, "")
            title_suffix = f" - {ai_name}" if ai_name else ""
            
            ic_loadings = mixing_matrix[ic]
            # Sort loadings by absolute value
            ic_loadings = ic_loadings.reindex(ic_loadings.abs().sort_values(ascending=False).index)
            
            # Only show top 15 features for readability
            if len(ic_loadings) > 15:
                ic_loadings = ic_loadings.iloc[:15]
            
            plt.subplot(n_components, 1, i+1)
            colors = ['red' if x < 0 else 'blue' for x in ic_loadings]
            plt.barh(range(len(ic_loadings)), ic_loadings, color=colors)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.yticks(range(len(ic_loadings)), ic_loadings.index)
            plt.title(f'IC{i+1}{title_suffix} Feature Loadings')
            plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Visualize the first two components if available
        if n_components >= 2:
            visualize_embedding(df_ica.iloc[:, :2], target_col, method='ICA', verbose=verbose)
    
    return df_ica, ica_model, mixing_matrix, component_names

# Updated NMF implementation with factor loadings visualization
def apply_nmf_to_circadian(df_lep, n_components=5, target_col=None, min_value=0, use_openai_naming=True, verbose=True):
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
    use_openai_naming : bool, default=True
        Whether to use OpenAI to generate interpretable names for components
    verbose : bool, default=True
        Whether to display plots and print information during execution
        
    Returns:
    --------
    tuple:
        - df_nmf: DataFrame with NMF components
        - nmf_model: Fitted NMF model
        - components_df: DataFrame with NMF components/factor loadings
        - component_names: Dictionary with OpenAI-generated component names (if use_openai_naming=True)
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
    if verbose:
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
    
    # Get components (factor loadings)
    components_df = pd.DataFrame(
        data=nmf_model.components_,
        columns=numeric_cols,
        index=[f'nmf_{i+1}' for i in range(n_components)]
    )
    
    # Use OpenAI to name components if requested
    component_names = {}
    if use_openai_naming and n_components > 0:
        if verbose:
            print("Generating interpretable component names with OpenAI...")
        component_names = name_components_with_openai(components_df, method_prefix="NMF")
        
        # If we got component names, print them and rename dataframe columns
        if component_names and verbose:
            print("\nGenerated component names:")
            for comp, name in component_names.items():
                print(f"  {comp}: {name}")
                
        # Update dataframe column names with generated names
        col_mapping = {}
        for i in range(n_components):
            nmf_col = f'nmf_{i+1}'
            nmf_key = f'nmf_{i+1}'
            if nmf_key in component_names:
                col_mapping[nmf_col] = component_names[nmf_key]
        
        # Rename the columns in df_nmf
        df_nmf.rename(columns=col_mapping, inplace=True)
    
    # Add the target column if specified
    if target_col and target_col in df.columns:
        df_nmf[target_col] = df.loc[X.index, target_col]
    
    if verbose:
        # Create summary for toggle
        nmf_summary = f"""
        <h3>NMF Summary</h3>
        <p>Number of components: {n_components}</p>
        <p>Number of observations: {len(X)}</p>
        <p>Number of variables: {len(numeric_cols)}</p>
        
        <h4>Non-negative Matrix Factorization</h4>
        <p>NMF finds patterns in data by learning parts-based representations with non-negative constraints.</p>
        
        <h4>Components:</h4>
        <ul>
        """
        
        for i in range(n_components):
            comp = f'nmf_{i+1}'
            ai_name = component_names.get(comp, "")
            ai_name_display = f" - {ai_name}" if ai_name else ""
            
            top_features = components_df.loc[comp].nlargest(5)
            nmf_summary += f"<li><strong>Component {i+1}{ai_name_display}:</strong><ul>"
            for feature, weight in top_features.items():
                nmf_summary += f"<li>{feature}: {weight:.3f}</li>"
            nmf_summary += "</ul></li>"
        
        nmf_summary += "</ul>"
        
        # Display the toggle summary
        create_toggle_summary(nmf_summary, title="NMF Summary")
        
        # Display top contributing features for each component
        print("\nNMF components (feature weights):")
        for comp in components_df.index:
            ai_name = component_names.get(comp, "")
            ai_name_display = f" - {ai_name}" if ai_name else ""
            
            top_features = components_df.loc[comp].nlargest(5)
            print(f"\n{comp}{ai_name_display} top features:")
            for feature, weight in top_features.items():
                print(f"  {feature}: {weight:.3f}")
        
        # Plot factor loadings heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(components_df, cmap='YlOrRd', aspect='auto')  # Using YlOrRd for non-negative data
        plt.colorbar(label='Loading strength')
        
        # Add component names to the x-axis labels if available
        if component_names:
            x_labels = []
            for i in range(n_components):
                comp = f'nmf_{i+1}'
                ai_name = component_names.get(comp, "")
                if ai_name:
                    x_labels.append(f"Component {i+1}\n{ai_name}")
                else:
                    x_labels.append(f"Component {i+1}")
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
            plt.yticks(range(n_components), x_labels)
        else:
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
            plt.yticks(range(n_components), [f'Component {i+1}' for i in range(n_components)])
            
        plt.title('NMF Components (Factor Loadings)')
        plt.tight_layout()
        plt.show()
        
        # For a more detailed visualization of each component
        plt.figure(figsize=(15, n_components*3))
        
        for i in range(n_components):
            comp = components_df.index[i]
            ai_name = component_names.get(comp, "")
            title_suffix = f" - {ai_name}" if ai_name else ""
            
            comp_loadings = components_df.loc[comp]
            # Sort loadings by value (descending)
            comp_loadings = comp_loadings.sort_values(ascending=False)
            
            # Only show top 15 features for readability
            if len(comp_loadings) > 15:
                comp_loadings = comp_loadings.iloc[:15]
            
            plt.subplot(n_components, 1, i+1)
            plt.barh(range(len(comp_loadings)), comp_loadings, color='orange')
            plt.yticks(range(len(comp_loadings)), comp_loadings.index)
            plt.title(f'Component {i+1}{title_suffix} Feature Loadings')
            plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Visualize the first two components if available
        if n_components >= 2:
            visualize_embedding(df_nmf.iloc[:, :2], target_col, method='NMF', verbose=verbose)
    
    return df_nmf, nmf_model, components_df, component_names

def apply_umap_to_circadian(df_lep, n_components=2, n_neighbors=15, min_dist=0.1, target_col=None, use_scaling=True, use_openai_naming=True, verbose=True):
    """
    Apply UMAP dimensionality reduction to circadian data.
    
    Parameters:
    -----------
    df_lep : pandas DataFrame
        DataFrame with circadian data
    n_components : int, default=2
        Dimension of the embedded space
    n_neighbors : int, default=15
        Number of neighbors to consider for manifold approximation
    min_dist : float, default=0.1
        Controls how tightly points are packed together
    target_col : str, default=None
        Optional target column for visualization
    use_scaling : bool, default=True
        Whether to apply z-scaling to normalize features before UMAP
    use_openai_naming : bool, default=True
        Whether to use OpenAI to generate interpretable names for components
    verbose : bool, default=True
        Whether to display plots and print information during execution
        
    Returns:
    --------
    tuple:
        - df_umap: DataFrame with UMAP components
        - umap_model: Fitted UMAP model
        - component_names: Dictionary with OpenAI-generated component names (if use_openai_naming=True)
    """
    try:
        import umap
    except ImportError:
        if verbose:
            print("UMAP is not installed. Install it with: pip install umap-learn")
        return None, None, {}
    
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
    
    # Check if we have enough data
    if len(df) < n_neighbors:
        if verbose:
            print(f"Not enough data for UMAP (minimum {n_neighbors} samples recommended)")
        return None, None, {}
    
    # Select only complete rows for the selected features
    X = df[numeric_cols].dropna()
    if verbose:
        print(f"Applying UMAP to {X.shape[0]} samples with {X.shape[1]} features")
    
    # Apply z-scaling if requested
    if use_scaling:
        if verbose:
            print("Applying z-scaling to normalize features")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Apply UMAP
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    
    umap_result = umap_model.fit_transform(X_scaled)
    
    # Create a DataFrame with UMAP results
    df_umap = pd.DataFrame(
        data=umap_result,
        columns=[f'umap_{i+1}' for i in range(n_components)],
        index=X.index
    )
    
    # For UMAP, we need to create a simple representation of feature importances
    # We'll use the correlation between original features and the UMAP components
    component_names = {}
    if use_openai_naming and n_components > 0:
        if verbose:
            print("Generating UMAP component interpretations with OpenAI...")
        # Create a DataFrame to hold the correlations between features and UMAP components
        umap_feature_correlations = pd.DataFrame(index=[f'umap_{i+1}' for i in range(n_components)])
        
        # Calculate correlations between original features and UMAP dimensions
        for i, col in enumerate(df_umap.columns[:n_components]):
            for feature in numeric_cols:
                # Join the original feature with the UMAP component
                df_corr = pd.DataFrame({
                    'feature': X[feature],
                    'component': df_umap[col]
                }).dropna()
                
                if len(df_corr) > 5:  # Need enough points for correlation
                    corr = df_corr['feature'].corr(df_corr['component'])
                    umap_feature_correlations.loc[col, feature] = corr
        
        # Use this correlation matrix as "loadings" for OpenAI to interpret
        if not umap_feature_correlations.empty:
            component_names = name_components_with_openai(umap_feature_correlations, method_prefix="UMAP")
            
            # If we got component names, print them and rename dataframe columns
            if component_names and verbose:
                print("\nGenerated UMAP component interpretations:")
                for comp, name in component_names.items():
                    print(f"  {comp}: {name}")
                    
            # Update dataframe column names with generated names
            col_mapping = {}
            for i in range(n_components):
                umap_col = f'umap_{i+1}'
                umap_key = f'umap_{i+1}'
                if umap_key in component_names:
                    col_mapping[umap_col] = component_names[umap_key]
            
            # Rename the columns in df_umap
            df_umap.rename(columns=col_mapping, inplace=True)
    
    # Add the target column if specified
    if target_col and target_col in df.columns:
        df_umap[target_col] = df.loc[X.index, target_col]
    
    if verbose:
        # Create summary for toggle
        umap_summary = f"""
        <h3>UMAP Summary</h3>
        <p>Number of components: {n_components}</p>
        <p>Number of observations: {len(X)}</p>
        <p>Number of variables: {len(numeric_cols)}</p>
        <p>Number of neighbors: {n_neighbors}</p>
        <p>Minimum distance: {min_dist}</p>
        
        <h4>Uniform Manifold Approximation and Projection</h4>
        <p>UMAP is a dimension reduction technique that preserves both local and global structure in the data.
        It's particularly good at revealing clusters and maintaining the separation between groups.</p>
        
        <h4>Key UMAP Parameters:</h4>
        <ul>
            <li><strong>n_neighbors</strong>: Controls how UMAP balances local versus global structure. Lower values (e.g., 5-15) focus on local structure, higher values (e.g., 30-100) preserve more global structure.</li>
            <li><strong>min_dist</strong>: Controls how tightly UMAP packs points together. Smaller values (0.0-0.2) result in tighter clusters, while larger values (0.5-0.8) allow for more even spacing of points.</li>
        </ul>
        """
        
        # Add component names if available
        if component_names:
            umap_summary += "<h4>UMAP Components:</h4><ul>"
            for i in range(n_components):
                comp = f'umap_{i+1}'
                ai_name = component_names.get(comp, "")
                if ai_name:
                    umap_summary += f"<li><strong>{comp}</strong>: {ai_name}</li>"
            umap_summary += "</ul>"
        
        # Display the toggle summary
        create_toggle_summary(umap_summary, title="UMAP Summary")
        
        # Visualize the results
        if n_components in (2, 3):
            # Just use the already renamed dataframe for visualization
            visualize_embedding(df_umap.iloc[:, :2], target_col, method='UMAP', verbose=verbose)
    
    return df_umap, umap_model, component_names

def apply_svd_to_circadian(df_lep, n_components=5, target_col=None, use_openai_naming=True, verbose=True):
    """
    Apply Singular Value Decomposition to circadian data.
    
    Parameters:
    -----------
    df_lep : pandas DataFrame
        DataFrame with circadian data
    n_components : int, default=5
        Number of singular vectors to compute
    target_col : str, default=None
        Optional target column for visualization
    use_openai_naming : bool, default=True
        Whether to use OpenAI to generate interpretable names for components
    verbose : bool, default=True
        Whether to display plots and print information during execution
        
    Returns:
    --------
    tuple:
        - df_svd: DataFrame with SVD components
        - svd_model: Fitted TruncatedSVD model
        - explained_variance: DataFrame with singular values and explained variance
        - components_df: DataFrame with right singular vectors (V.T)
        - component_names: Dictionary with OpenAI-generated component names (if use_openai_naming=True)
    """
    from sklearn.decomposition import TruncatedSVD
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
    if verbose:
        print(f"Applying SVD to {X.shape[0]} samples with {X.shape[1]} features")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply SVD
    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    svd_result = svd_model.fit_transform(X_scaled)
    
    # Create a DataFrame with SVD results
    df_svd = pd.DataFrame(
        data=svd_result,
        columns=[f'svd_{i+1}' for i in range(n_components)],
        index=X.index
    )
    
    # Create DataFrame with singular values and explained variance
    explained_variance = pd.DataFrame({
        'singular_value': svd_model.singular_values_,
        'explained_variance': svd_model.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(svd_model.explained_variance_ratio_)
    }, index=[f'component_{i+1}' for i in range(n_components)])
    
    # Create DataFrame with right singular vectors (V.T)
    components_df = pd.DataFrame(
        data=svd_model.components_,
        columns=numeric_cols,
        index=[f'component_{i+1}' for i in range(n_components)]
    )
    
    # Use OpenAI to name components if requested
    component_names = {}
    if use_openai_naming and n_components > 0:
        if verbose:
            print("Generating interpretable component names with OpenAI...")
        component_names = name_components_with_openai(components_df, method_prefix="SVD")
        
        # If we got component names, print them and rename dataframe columns
        if component_names and verbose:
            print("\nGenerated component names:")
            for comp, name in component_names.items():
                print(f"  {comp}: {name}")
                
        # Update dataframe column names with generated names
        col_mapping = {}
        for i in range(n_components):
            svd_col = f'svd_{i+1}'
            component_key = f'component_{i+1}'
            if component_key in component_names:
                col_mapping[svd_col] = component_names[component_key]
        
        # Rename the columns in df_svd
        df_svd.rename(columns=col_mapping, inplace=True)
    
    # Add the target column if specified
    if target_col and target_col in df.columns:
        df_svd[target_col] = df.loc[X.index, target_col]
    
    if verbose:
        # Create summary for toggle
        svd_summary = f"""
        <h3>SVD Summary</h3>
        <p>Number of components: {n_components}</p>
        <p>Number of observations: {len(X)}</p>
        <p>Number of variables: {len(numeric_cols)}</p>
        
        <h4>Singular Value Decomposition</h4>
        <p>SVD decomposes a matrix into three matrices: U, Σ, and V<sup>T</sup>, where Σ contains singular values.
        It's the mathematical foundation for many dimensionality reduction techniques including PCA.</p>
        
        <h4>Explained Variance by Component:</h4>
        <table style="width:100%; border-collapse: collapse;">
        <tr>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Singular Value</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Explained Variance</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Cumulative</th>
        </tr>
        """
        
        for i in range(n_components):
            comp_idx = f'component_{i+1}'
            ai_name = component_names.get(comp_idx, "")
            ai_name_display = f" - {ai_name}" if ai_name else ""
            
            sing_val = explained_variance.loc[comp_idx, 'singular_value']
            var = explained_variance.loc[comp_idx, 'explained_variance']
            cum_var = explained_variance.loc[comp_idx, 'cumulative_variance']
            
            svd_summary += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Component {i+1}{ai_name_display}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{sing_val:.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{var:.3f} ({var*100:.1f}%)</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{cum_var:.3f} ({cum_var*100:.1f}%)</td>
            </tr>
            """
        
        svd_summary += "</table>"
        
        # Add top contributing features
        svd_summary += "<h4>Top Contributing Features:</h4><ul>"
        for i in range(min(3, n_components)):
            comp = f'component_{i+1}'
            ai_name = component_names.get(comp, "")
            ai_name_display = f" - {ai_name}" if ai_name else ""
            
            top_features = abs(components_df.loc[comp]).nlargest(5)
            svd_summary += f"<li><strong>Component {i+1}{ai_name_display}:</strong><ul>"
            for feature in top_features.index:
                importance = components_df.loc[comp, feature]
                direction = "+" if importance > 0 else "-"
                svd_summary += f"<li>{feature}: {abs(importance):.3f} ({direction})</li>"
            svd_summary += "</ul></li>"
        svd_summary += "</ul>"
        
        # Display the toggle summary
        create_toggle_summary(svd_summary, title="SVD Summary")
        
        # Plot singular values and explained variance
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Bar plot for singular values
        ax1.bar(range(1, n_components + 1), explained_variance['singular_value'], color='lightblue')
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Singular Value', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Line plot for cumulative explained variance
        ax2 = ax1.twinx()
        ax2.plot(range(1, n_components + 1), explained_variance['cumulative_variance'], 
                 'ro-', linewidth=2)
        ax2.set_ylabel('Cumulative Explained Variance', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Set x-axis ticks
        plt.xticks(range(1, n_components + 1))
        plt.title('SVD - Singular Values and Explained Variance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Plot component loadings heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(components_df, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Loading strength')
        
        # Add component names to the x-axis labels if available
        if component_names:
            x_labels = []
            for i in range(n_components):
                comp = f'component_{i+1}'
                ai_name = component_names.get(comp, "")
                if ai_name:
                    y_labels = [f"Component {i+1}\n{ai_name}" for i in range(n_components)]
                else:
                    y_labels = [f"Component {i+1}" for i in range(n_components)]
            plt.yticks(range(n_components), y_labels)
        else:
            plt.yticks(range(n_components), [f'Component {i+1}' for i in range(n_components)])
            
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
        plt.title('SVD Components (Feature Loadings)')
        plt.tight_layout()
        plt.show()
        
        # For a more detailed visualization of the top components
        plt.figure(figsize=(15, n_components*3))
        
        for i in range(n_components):
            comp = components_df.index[i]
            ai_name = component_names.get(comp, "")
            title_suffix = f" - {ai_name}" if ai_name else ""
            
            comp_loadings = components_df.loc[comp]
            # Sort loadings by absolute value
            comp_loadings = comp_loadings.reindex(comp_loadings.abs().sort_values(ascending=False).index)
            
            # Only show top 15 features for readability
            if len(comp_loadings) > 15:
                comp_loadings = comp_loadings.iloc[:15]
            
            plt.subplot(n_components, 1, i+1)
            colors = ['red' if x < 0 else 'blue' for x in comp_loadings]
            plt.barh(range(len(comp_loadings)), comp_loadings, color=colors)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.yticks(range(len(comp_loadings)), comp_loadings.index)
            plt.title(f'Component {i+1}{title_suffix} Feature Loadings')
            plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Visualize the first two components if available
        if n_components >= 2:
            visualize_embedding(df_svd.iloc[:, :2], target_col, method='SVD', verbose=verbose)
    
    return df_svd, svd_model, explained_variance, components_df, component_names
