import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.search.FCMBased import lingam

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased.lingam import DirectLiNGAM
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss

from causallearn.utils.cit import fisherz, mv_fisherz
CAUSAL_LEARN_AVAILABLE = True

# Additional imports for other methods - wrapped in try-except
try:
    from lingam import ICALiNGAM, DirectLiNGAM as DLiNGAM, VARLiNGAM
    from lingam.utils import make_dot
    LINGAM_AVAILABLE = True
except ImportError:
    print("lingam package not available. LiNGAM-based methods will be skipped.")
    LINGAM_AVAILABLE = False

try:
    import cdt
    from cdt.causality.graph import CAM, GES as CDTGES, GIES
    # from cdt.causality.pairwise.model import ANM, PNL
    CDT_AVAILABLE = True
except ImportError:
    print("cdt (CausalDiscoveryToolbox) not available. Some algorithms will be skipped.")
    CDT_AVAILABLE = False

def generate_synthetic_circadian_data(n_samples=500, random_state=42):
    """
    Generate synthetic circadian rhythm data with known causal relationships
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Dataframe containing synthetic circadian data with causal relationships
    """
    np.random.seed(random_state)
    
    # Create empty dataframe
    df = pd.DataFrame()
    
    # Create light exposure variable (exogenous)
    df['light_exposure'] = np.random.normal(loc=600, scale=300, size=n_samples)
    df['light_exposure'] = np.maximum(0, df['light_exposure'])  # No negative light exposure
    
    # Caffeine intake (partially influenced by light exposure - people drink coffee when it's light)
    df['caffeine_intake'] = 50 + 0.1 * df['light_exposure'] + np.random.normal(loc=0, scale=50, size=n_samples)
    df['caffeine_intake'] = np.maximum(0, df['caffeine_intake'])  # No negative caffeine intake
    
    # Melatonin production (negatively affected by light and caffeine)
    df['melatonin'] = 200 - 0.15 * df['light_exposure'] - 0.3 * df['caffeine_intake'] + np.random.normal(loc=0, scale=20, size=n_samples)
    df['melatonin'] = np.maximum(0, df['melatonin'])  # No negative melatonin
    
    # Core body temperature (affected by light exposure and activity)
    df['physical_activity'] = np.random.normal(loc=5000, scale=2000, size=n_samples)
    df['physical_activity'] = np.maximum(0, df['physical_activity'])  # No negative activity
    
    # Core body temperature (affected by light, activity and time of day)
    df['body_temperature'] = 36.5 + 0.0005 * df['light_exposure'] + 0.0002 * df['physical_activity'] - 0.002 * df['melatonin'] + np.random.normal(loc=0, scale=0.2, size=n_samples)
    
    # Sleep propensity (affected by melatonin, body temperature, and caffeine)
    df['sleep_propensity'] = 30 + 0.25 * df['melatonin'] - 2 * (df['body_temperature'] - 36.5) - 0.1 * df['caffeine_intake'] + np.random.normal(loc=0, scale=5, size=n_samples)
    
    # DLMO (Dim Light Melatonin Onset) time - influenced by light exposure and genetics
    # For simplicity, measured in minutes from a reference point
    df['genetic_factor'] = np.random.normal(loc=0, scale=1, size=n_samples)
    df['dlmo_time'] = 1080 - 0.1 * df['light_exposure'] + 30 * df['genetic_factor'] + np.random.normal(loc=0, scale=20, size=n_samples)
    
    # LEP (Light Exposure Point) - mainly determined by DLMO but with noise
    df['lep_time'] = df['dlmo_time'] + 120 + np.random.normal(loc=0, scale=30, size=n_samples)
    
    return df

def preprocess_for_causal_discovery(data, variance_threshold=1e-10, correlation_threshold=0.99):
    """
    Preprocess data to avoid singular correlation matrices in causal discovery algorithms
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to preprocess
    variance_threshold : float
        Threshold below which columns are considered zero-variance
    correlation_threshold : float
        Threshold above which columns are considered too highly correlated
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with problematic columns removed
    list
        Names of removed columns
    """
    # Start with a copy of the data
    processed_data = data.copy()
    
    # Check for and remove columns with near-zero variance
    variances = processed_data.var()
    low_var_cols = variances[variances < variance_threshold].index.tolist()
    if low_var_cols:
        print(f"Removing {len(low_var_cols)} columns with near-zero variance: {low_var_cols}")
        processed_data = processed_data.drop(columns=low_var_cols)
    
    # Check for and remove highly correlated features
    removed_high_corr = []
    corr_matrix = processed_data.corr().abs()
    
    # Create a mask to ignore self-correlations
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Find pairs of features with correlation above threshold
    high_corr_pairs = []
    for i, col_i in enumerate(corr_matrix.columns):
        for j, col_j in enumerate(corr_matrix.columns):
            if i < j and corr_matrix.iloc[i, j] > correlation_threshold:
                high_corr_pairs.append((col_i, col_j, corr_matrix.iloc[i, j]))
    
    # Sort by correlation to remove highest correlations first
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Remove one feature from each highly correlated pair
    for col_i, col_j, corr_val in high_corr_pairs:
        if col_i in processed_data.columns and col_j in processed_data.columns:
            # Remove the column with higher mean absolute correlation with other features
            if col_i not in removed_high_corr and col_j not in removed_high_corr:
                col_i_mean_corr = corr_matrix[col_i].mean()
                col_j_mean_corr = corr_matrix[col_j].mean()
                
                if col_i_mean_corr > col_j_mean_corr:
                    processed_data = processed_data.drop(columns=[col_i])
                    removed_high_corr.append(col_i)
                else:
                    processed_data = processed_data.drop(columns=[col_j])
                    removed_high_corr.append(col_j)
                
                print(f"Removed highly correlated feature: {removed_high_corr[-1]} (corr: {corr_val:.4f})")
    
    # Verify the correlation matrix is not singular
    try:
        corr_matrix = processed_data.corr()
        np.linalg.inv(corr_matrix.values)
        print("Correlation matrix is now invertible")
    except np.linalg.LinAlgError:
        print("Warning: Correlation matrix is still singular after preprocessing")
    
    # Combine all removed columns
    removed_cols = low_var_cols + removed_high_corr
    
    return processed_data, removed_cols

def run_pc_algorithm(data, alpha=0.05, skip_preprocessing=False, verbose=False):
    """
    Run the PC algorithm for causal discovery
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
    alpha : float
        Significance level for conditional independence tests
        
    Returns:
    --------
    object
        Causal graph from PC algorithm
    """
    if not CAUSAL_LEARN_AVAILABLE:
        print("PC algorithm not available. Skipping.")
        return None
    
    # Preprocess data to avoid singular correlation matrix
    if not skip_preprocessing:
        processed_data, removed_cols = preprocess_for_causal_discovery(data)
        if len(removed_cols) > 0:
            print(f"Removed {len(removed_cols)} problematic columns for PC algorithm")
    else:
        processed_data = data
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed_data)
    
    # try:
    # Run PC algorithm
    cg = pc(scaled_data, alpha=alpha, indep_test=mv_fisherz, mvpc=True, verbose=verbose, node_names=processed_data.columns.tolist())
    return cg, processed_data.columns.tolist()
    # except ValueError as e:
    #     print(f"Error running PC algorithm: {e}")
    #     return None, processed_data.columns.tolist()

def run_ges_algorithm(data):
    """
    Run the GES (Greedy Equivalence Search) algorithm for causal discovery
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
        
    Returns:
    --------
    object
        Causal graph from GES algorithm
    """
    if not CAUSAL_LEARN_AVAILABLE:
        print("GES algorithm not available. Skipping.")
        return None
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for GES algorithm")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed_data)
    
    try:
        # Run GES algorithm
        cg = ges(scaled_data)
        return cg, processed_data.columns.tolist()
    except ValueError as e:
        print(f"Error running GES algorithm: {e}")
        return None, processed_data.columns.tolist()

def run_fci_algorithm(data, alpha=0.05, verbose=False):
    """
    Run the FCI (Fast Causal Inference) algorithm for causal discovery
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
    alpha : float
        Significance level for conditional independence tests
        
    Returns:
    --------
    tuple:
        - Causal graph from FCI algorithm
        - Processed feature names
    """
    # if not CAUSAL_LEARN_AVAILABLE:
    #     print("FCI algorithm not available. Skipping.")
    #     return None, []
    
    # Preprocess data to avoid singular correlation matrix
    # processed_data, removed_cols = preprocess_for_causal_discovery(data)
    # if len(removed_cols) > 0:
    #     print(f"Removed {len(removed_cols)} problematic columns for FCI algorithm")
    
    # Standardize the data
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(data)
    
    try:
        # Run FCI algorithm
        print("Running FCI algorithm...")
        cg, edges = fci(data, alpha=alpha, mvpc=True, indep_test=mv_fisherz)
        return cg, edges, data.columns.tolist()
    except ValueError as e:
        print(f"Error running FCI algorithm: {e}")
        return None, data.columns.tolist()

def run_cd_nod_algorithm(data, c_indx):
    """
    Run the CD-NOD (Causal Discovery from Non-stationary/heterogeneous Data) algorithm
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
    alpha : float
        Significance level for conditional independence tests
        
    Returns:
    --------
    tuple:
        - Causal graph from CD-NOD algorithm
        - Processed feature names
    """
    if not CDT_AVAILABLE:
        print("CD-NOD algorithm not available. Skipping.")
        return None, []
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for CD-NOD algorithm")
    
    try:
        # Run CD-NOD algorithm from CDT
        print("Running CD-NOD algorithm...")
        model = cdnod(processed_data, c_indx)
        skeleton = model.predict(processed_data)
        return skeleton, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running CD-NOD algorithm: {e}")
        return None, processed_data.columns.tolist()

def run_ica_lingam_algorithm(data):
    """
    Run the ICA-based LiNGAM algorithm for causal discovery
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
        
    Returns:
    --------
    tuple:
        - NetworkX DiGraph from ICA-LiNGAM algorithm
        - Processed feature names
    """
    if not LINGAM_AVAILABLE:
        print("ICA-LiNGAM algorithm not available. Skipping.")
        return None, []
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for ICA-LiNGAM algorithm")
    
    try:
        # Run ICA-LiNGAM algorithm
        print("Running ICA-LiNGAM algorithm...")
        model = ICALiNGAM()
        model.fit(processed_data)
        
        # Convert model to NetworkX graph
        nx_graph = nx.DiGraph()
        
        # Add nodes with original feature names
        feature_names = processed_data.columns.tolist()
        for name in feature_names:
            nx_graph.add_node(name)
        
        # Add edges based on adjacency matrix
        B = model.adjacency_matrix_
        
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if B[i, j] != 0:
                    nx_graph.add_edge(feature_names[i], feature_names[j], weight=B[i, j])
        
        print(f"Created graph with {len(nx_graph.nodes())} nodes and {len(nx_graph.edges())} edges")
        return nx_graph, feature_names
    except Exception as e:
        print(f"Error running ICA-LiNGAM algorithm: {e}")
        return None, processed_data.columns.tolist()

def run_direct_lingam_algorithm(data):
    """
    Run the DirectLiNGAM algorithm for causal discovery
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
        
    Returns:
    --------
    tuple:
        - Model from DirectLiNGAM algorithm
        - Processed feature names
    """
    if not LINGAM_AVAILABLE:
        print("DirectLiNGAM algorithm not available. Skipping.")
        return None, []
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for DirectLiNGAM algorithm")
    
    try:
        # Run DirectLiNGAM algorithm
        print("Running DirectLiNGAM algorithm...")
        model = DLiNGAM()
        model.fit(processed_data)
        return model, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running DirectLiNGAM algorithm: {e}")
        return None, processed_data.columns.tolist()

def run_var_lingam_algorithm(data, lags=1):
    """
    Run the VAR-LiNGAM algorithm for causal discovery on time series data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data to analyze
    lags : int
        Number of lags to consider
        
    Returns:
    --------
    tuple:
        - Model from VAR-LiNGAM algorithm
        - Processed feature names
    """
    if not LINGAM_AVAILABLE:
        print("VAR-LiNGAM algorithm not available. Skipping.")
        return None, []
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for VAR-LiNGAM algorithm")
    
    try:
        # Run VAR-LiNGAM algorithm
        print("Running VAR-LiNGAM algorithm...")
        model = VARLiNGAM(lags=lags)
        model.fit(processed_data)
        return model, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running VAR-LiNGAM algorithm: {e}")
        return None, processed_data.columns.tolist()

def run_rcd_algorithm(data, **kwargs):
    """
    Run the RCD (Recursive Causal Discovery) algorithm
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
        
    Returns:
    --------
    tuple:
        - Causal graph from RCD algorithm
        - Processed feature names
    """
    if not CDT_AVAILABLE:
        print("RCD algorithm not available. Skipping.")
        return None, []
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for RCD algorithm")
    
    try:
        # Run RCD algorithm from CDT
        print("Running RCD algorithm...")
        model =  lingam.RCD(**kwargs)
        skeleton = model.fit(processed_data)
        return skeleton, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running RCD algorithm: {e}")
        return None, processed_data.columns.tolist()

# Requires R.
def run_cam_uv_algorithm(data):
    """
    Run the CAM-UV (Causal Additive Models with Unobserved Variables) algorithm
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
        
    Returns:
    --------
    tuple:
        - Causal graph from CAM-UV algorithm
        - Processed feature names
    """
    if not CDT_AVAILABLE:
        print("CAM-UV algorithm not available. Skipping.")
        return None, []
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for CAM-UV algorithm")
    
    try:
        # Run CAM algorithm from CDT
        print("Running CAM-UV algorithm...")
        model = CAM(pruning=True)  # With pruning for better results
        skeleton = model.predict(processed_data)
        return skeleton, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running CAM-UV algorithm: {e}")
        return None, processed_data.columns.tolist()

# Needs R...
# And giant PITA even then due to  package 'RCIT' is not available for this version of R
# Note that it's just doing PC anyway and then using ANM to identify the direction of the edges
def run_anm_algorithm(data):
    """
    Run the ANM (Additive Noise Model) algorithm for pairwise causal discovery
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
        
    Returns:
    --------
    tuple:
        - Causal graph from ANM algorithm
        - Processed feature names
    """
    if not CDT_AVAILABLE:
        print("ANM algorithm not available. Skipping.")
        return None, []
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for ANM algorithm")
    
    try:
        # First compute undirected graph using a constraint-based method
        print("Running ANM algorithm...")
        und_model = cdt.causality.graph.PC()
        skeleton = und_model.predict(processed_data, return_type='skeleton')
        
        # Then orient edges using ANM
        model = ANM()
        oriented_graph = cdt.causality.graph.orient_graph(skeleton, model, processed_data)
        return oriented_graph, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running ANM algorithm: {e}")
        return None, processed_data.columns.tolist()

# Needs R...
def run_pnl_algorithm(data):
    """
    Run the PNL (Post-Nonlinear causal model) algorithm for pairwise causal discovery
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
        
    Returns:
    --------
    tuple:
        - Causal graph from PNL algorithm
        - Processed feature names
    """
    if not CDT_AVAILABLE:
        print("PNL algorithm not available. Skipping.")
        return None, []
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for PNL algorithm")
    
    try:
        # First compute undirected graph using a constraint-based method
        print("Running PNL algorithm...")
        und_model = cdt.causality.graph.PC()
        skeleton = und_model.predict(processed_data, return_type='skeleton')
        
        # Then orient edges using PNL
        model = PNL()
        oriented_graph = cdt.causality.graph.orient_graph(skeleton, model, processed_data)
        return oriented_graph, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running PNL algorithm: {e}")
        return None, processed_data.columns.tolist()

def run_grasp_algorithm(data, **kwargs):
    """
    Run the GRaSP (Greedy relaxation of Sparsest Permutation) algorithm
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
        
    Returns:
    --------
    tuple:
        - Causal graph from GRaSP algorithm
        - Processed feature names
    """
    if not CAUSAL_LEARN_AVAILABLE:
        print("GRaSP algorithm not available. Skipping.")
        return None, []
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for GRaSP algorithm")
    
    try:
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(processed_data)
        
        # Run GRaSP algorithm
        print("Running GRaSP algorithm...")
        cg = grasp(scaled_data, **kwargs)
        return cg, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running GRaSP algorithm: {e}")
        return None, processed_data.columns.tolist()

def run_boss_algorithm(data, **kwargs):
    """
    Run the BOSS (Best Order Score Search) algorithm
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
        
    Returns:
    --------
    tuple:
        - Causal graph from BOSS algorithm
        - Processed feature names
    """
    if not CDT_AVAILABLE:
        print("BOSS algorithm not available. Skipping.")
        return None, []
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for BOSS algorithm")
    
    try:
        # Run BOSS algorithm from CDT
        print("Running BOSS algorithm...")
        model = boss(processed_data, **kwargs)
        return model, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running BOSS algorithm: {e}")
        return None, processed_data.columns.tolist()

def run_exactsearch_algorithm(data):
    """
    Run the ExactSearch algorithm for causal discovery (for small graphs)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
        
    Returns:
    --------
    tuple:
        - Causal graph from ExactSearch algorithm
        - Processed feature names
    """
    if not CDT_AVAILABLE:
        print("ExactSearch algorithm not available. Skipping.")
        return None, []
    
    # For exact search, we need a small number of variables
    if data.shape[1] > 8:
        print("Warning: ExactSearch algorithm is only feasible for small graphs (≤8 variables).")
        print(f"Current data has {data.shape[1]} variables, which may cause very long computation times.")
        # Limit to top 8 variables with highest variance
        variances = data.var().sort_values(ascending=False)
        top_vars = variances.index[:8].tolist()
        print(f"Limiting to top 8 variables with highest variance: {top_vars}")
        data = data[top_vars]
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for ExactSearch algorithm")
    
    try:
        # Run ExactSearch algorithm from CDT
        print("Running ExactSearch algorithm...")
        model = cdt.causality.graph.ExactSearch()
        skeleton = model.predict(processed_data)
        return skeleton, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running ExactSearch algorithm: {e}")
        return None, processed_data.columns.tolist()

def run_gin_algorithm(data, hidden_layers=2, hidden_units=64, epochs=1000):
    """
    Run the GIN (Graph Identification Network) algorithm for causal discovery
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
    hidden_layers : int
        Number of hidden layers in the neural network
    hidden_units : int
        Number of hidden units per layer
    epochs : int
        Number of training epochs
        
    Returns:
    --------
    tuple:
        - Causal graph from GIN algorithm
        - Processed feature names
    """
    from importlib.util import find_spec
    
    if find_spec("gcastle") is None:
        print("GIN algorithm not available (requires gcastle package). Skipping.")
        return None, []
    
    # Import here to avoid loading if not used
    try:
        from gcastle.castle.algorithms import GIN as GINModel
    except ImportError:
        print("GIN algorithm not available. Skipping.")
        return None, []
    
    # Preprocess data to avoid singular correlation matrix
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for GIN algorithm")
    
    try:
        # Run GIN algorithm
        print("Running GIN algorithm...")
        gin = GINModel(
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            epochs=epochs
        )
        gin.learn(processed_data.values)
        
        # Create a NetworkX graph from the adjacency matrix
        nx_graph = nx.DiGraph()
        for i, name in enumerate(processed_data.columns):
            nx_graph.add_node(name)
        
        adj_matrix = gin.causal_matrix
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] != 0:
                    nx_graph.add_edge(processed_data.columns[i], processed_data.columns[j])
        
        return nx_graph, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running GIN algorithm: {e}")
        return None, processed_data.columns.tolist()

def convert_lingam_to_networkx(model, feature_names):
    """
    Convert a LiNGAM model to a NetworkX DiGraph
    
    Parameters:
    -----------
    model : object
        LiNGAM model
    feature_names : list
        List of feature names
        
    Returns:
    --------
    nx.DiGraph
        NetworkX DiGraph representation of the causal model
    """
    G = nx.DiGraph()
    
    # Add nodes
    for name in feature_names:
        G.add_node(name)
    
    # Extract adjacency matrix
    B = model.adjacency_matrix_
    
    # Add edges
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if B[i, j] != 0:
                G.add_edge(feature_names[i], feature_names[j], weight=B[i, j])
    
    return G

def convert_cdt_to_networkx(graph, feature_names):
    """
    Convert a CDT graph to a NetworkX DiGraph with proper feature names
    
    Parameters:
    -----------
    graph : object
        CDT graph
    feature_names : list
        List of feature names
        
    Returns:
    --------
    nx.DiGraph
        NetworkX DiGraph representation of the causal model
    """
    if isinstance(graph, nx.Graph):
        # It's already a NetworkX graph, just need to relabel nodes if they're not strings
        if graph.nodes() and not isinstance(list(graph.nodes())[0], str):
            mapping = {i: name for i, name in enumerate(feature_names)}
            return nx.relabel_nodes(graph, mapping)
        return graph
    
    # Create a new NetworkX DiGraph
    G = nx.DiGraph()
    
    # Add nodes
    for name in feature_names:
        G.add_node(name)
    
    # If graph is a pandas DataFrame (adjacency matrix), convert to NetworkX
    if isinstance(graph, pd.DataFrame):
        for i, row in enumerate(graph.values):
            for j, value in enumerate(row):
                if value != 0:
                    G.add_edge(feature_names[i], feature_names[j], weight=value)
    
    return G

def create_neural_causal_model(data, target_col='lep_time', epochs=1000, lr=0.001):
    """
    Create a neural causal model to infer causal relationships
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
    target_col : str
        Target column to predict
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
        
    Returns:
    --------
    dict
        Dictionary containing model, feature importances, and predicted values
    """
    feature_cols = [col for col in data.columns if col != target_col]
    X = data[feature_cols].values
    y = data[target_col].values.reshape(-1, 1)
    
    # Standardize the data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define neural network with masked connections to learn causal weights
    class NeuralCausalModel(nn.Module):
        def __init__(self, input_dim):
            super(NeuralCausalModel, self).__init__()
            self.causal_layer = nn.Linear(input_dim, 1, bias=True)
            
            # Initialize causal mask and weights
            self.causal_mask = nn.Parameter(torch.ones(input_dim), requires_grad=True)
            
        def forward(self, x):
            # Apply causal mask to inputs
            masked_input = x * self.causal_mask
            return self.causal_layer(masked_input)
        
        def get_causal_strengths(self):
            # Get the causal strengths by multiplying weights by mask
            return self.causal_layer.weight.data[0] * self.causal_mask.data
    
    # Initialize model, loss, and optimizer
    model = NeuralCausalModel(X_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Add L1 regularization for sparsity in causal mask
    l1_lambda = 0.01
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Calculate MSE loss
            mse_loss = criterion(outputs, batch_y)
            
            # Add L1 regularization to causal mask
            l1_reg = l1_lambda * torch.norm(model.causal_mask, 1)
            
            # Total loss
            loss = mse_loss + l1_reg
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {running_loss / len(dataloader):.4f}')
    
    # Get causal strengths
    model.eval()
    causal_strengths = model.get_causal_strengths().detach().numpy()
    
    # Get feature importances
    feature_importances = dict(zip(feature_cols, causal_strengths))
    
    # Make predictions
    with torch.no_grad():
        y_pred_scaled = model(X_tensor)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.numpy())
    
    return {
        'model': model,
        'feature_importances': feature_importances,
        'y_pred': y_pred,
        'feature_names': feature_cols
    }
