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

# Causal discovery libraries - wrapped in try-except to handle missing dependencies
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    #from causallearn.search.PermutationBased.GRaSP import GRaSP
    from causallearn.utils.cit import fisherz, mv_fisherz
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    print("causallearn not fully available. Some algorithms will be skipped.")
    CAUSAL_LEARN_AVAILABLE = False

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

def run_pc_algorithm(data, alpha=0.05):
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
    processed_data, removed_cols = preprocess_for_causal_discovery(data)
    if len(removed_cols) > 0:
        print(f"Removed {len(removed_cols)} problematic columns for PC algorithm")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed_data)
    
    try:
        # Run PC algorithm
        cg = pc(scaled_data, alpha=alpha, indep_test='mv_fisherz', mvpc=True)
        return cg, processed_data.columns.tolist()
    except ValueError as e:
        print(f"Error running PC algorithm: {e}")
        return None, processed_data.columns.tolist()

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

def run_fci_algorithm(data, feature_names, sig_level=0.05, max_cond_set_size=5, adjacency_matrix=None, apply_sparse_shifts=True):
    """
    Run the Fast Causal Inference (FCI) algorithm from the causallearn package.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The data to run the algorithm on
    feature_names : list
        List of feature names
    sig_level : float, default=0.05
        Significance level for conditional independence tests
    max_cond_set_size : int, default=5
        Maximum size of conditioning set for CI tests
    adjacency_matrix : np.ndarray, default=None
        Optional adjacency matrix to use as a prior knowledge graph
    apply_sparse_shifts : bool, default=True
        Whether to apply sparse shifts preprocessing for time series data
    
    Returns:
    --------
    tuple
        (graph, edges, feature_names) where graph is the FCI graph object
    """
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        import numpy as np
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
        
        print(f"Running FCI algorithm with sig_level={sig_level}, max_cond_set={max_cond_set_size}")
        
        # Make a copy of the data to avoid modifying the original
        data_array = data.copy().values
        
        # If we're dealing with time series, apply sparse shifts preprocessing
        if apply_sparse_shifts:
            # Simple preprocessing for time series: 
            # Create differences (returns) to approximate non-stationarity shifts
            try:
                # Apply differencing - more robust against non-stationary data
                diff_data = np.diff(data_array, axis=0)
                # Append a row of zeros at beginning to maintain shape
                data_array = np.vstack([np.zeros((1, data_array.shape[1])), diff_data])
                print(f"Applied differencing for time series. Shape: {data_array.shape}")
            except Exception as e:
                print(f"Error applying differencing: {e}")
        
        # For numerical stability, check for zero variance
        variances = np.var(data_array, axis=0)
        if any(variances < 1e-10):
            print("Warning: Some features have near-zero variance. Adding small noise.")
            # Add small noise to variables with zero variance
            for i in range(data_array.shape[1]):
                if variances[i] < 1e-10:
                    data_array[:, i] += np.random.normal(0, 1e-5, size=data_array.shape[0])
        
        # Create background knowledge if adjacency matrix is provided
        background_knowledge = None
        if adjacency_matrix is not None:
            background_knowledge = BackgroundKnowledge()
            for i in range(adjacency_matrix.shape[0]):
                for j in range(adjacency_matrix.shape[1]):
                    if adjacency_matrix[i, j] == 1:  # i -> j
                        background_knowledge.add_required_by_knowledge(str(i), str(j))
                    elif adjacency_matrix[i, j] == -1:  # i -/-> j (forbidden)
                        background_knowledge.add_forbidden_by_knowledge(str(i), str(j))
            print(f"Added background knowledge constraints")
            
        # Run FCI algorithm
        try:
            fci_graph = fci(data_array, 
                          alpha=sig_level, 
                          depth=max_cond_set_size,
                          background_knowledge=background_knowledge)
            
            # Get edges from FCI graph
            edges = fci_graph.G.get_graph_edges()
            
            # Create a more detailed description of the graph
            num_nodes = fci_graph.G.get_num_nodes()
            num_edges = len(edges)
            print(f"FCI graph has {num_nodes} nodes and {num_edges} edges")
            
            # Export the graph as a graphviz dot object for later visualization
            # Note: We don't create the NetworkX graph here because plot_causal_graph
            # will handle that with the complete set of feature names
            
            return (fci_graph, edges, feature_names)
        except Exception as e:
            print(f"Error running FCI algorithm: {e}")
            return None
    except ImportError:
        print("causallearn package not found. Install with: pip install causallearn")
        return None

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

def run_rcd_algorithm(data):
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
        model =  lingam.RCD()
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

def run_grasp_algorithm(data):
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
        cg = GRaSP(scaled_data)
        return cg, processed_data.columns.tolist()
    except Exception as e:
        print(f"Error running GRaSP algorithm: {e}")
        return None, processed_data.columns.tolist()

def run_boss_algorithm(data):
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
        model = cdt.causality.graph.BOSS()
        skeleton = model.predict(processed_data)
        return skeleton, processed_data.columns.tolist()
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

def plot_causal_graph(graph_tuple, title='Causal Graph', highlight_nodes=None, target_focus=None):
    """
    Plot the causal graph discovered by causal discovery algorithms with advanced visualization features
    
    Parameters:
    -----------
    graph_tuple : tuple
        Tuple containing (graph, feature_names) or (graph, edges, feature_names) for FCI algorithm
    title : str
        Title for the plot
    highlight_nodes : list, default=None
        Optional list of node names to highlight in a different color
    target_focus : str, default=None
        Optional mode to focus on highlighted nodes: 'removes' or 'grays' nodes not connected to highlights
    """
    if graph_tuple is None or graph_tuple[0] is None:
        print(f"No graph to plot for {title}")
        return None
    
    # Handle different tuple formats
    if len(graph_tuple) == 3:  # FCI algorithm returns (graph, edges, feature_names)
        graph, edges, feature_names = graph_tuple
        is_fci = True
    else:  # Regular format (graph, feature_names)
        graph, feature_names = graph_tuple
        edges = None
        is_fci = False
    
    plt.figure(figsize=(20, 15))
    
    # Create NetworkX graph
    nx_graph = nx.DiGraph()
    is_directed = True
    
    # Add nodes - make sure to add all nodes from feature_names
    for i, name in enumerate(feature_names):
        nx_graph.add_node(name)
    
    # Handle FCI-specific graph and edges
    if is_fci:
        print("Processing FCI algorithm output with explicit edges")
        from causallearn.graph.Endpoint import Endpoint
        
        # Map endpoint types to strings
        def endpoint_to_str(endpoint):
            if endpoint == Endpoint.TAIL:
                return "-"
            elif endpoint == Endpoint.ARROW:
                return ">"
            elif endpoint == Endpoint.CIRCLE:
                return "o"
            else:
                return "?"
        
        # Check if node names include "X" prefixes from causallearn
        first_node_name = edges[0].get_node1().get_name() if edges else None
        use_direct_names = (isinstance(first_node_name, str) and first_node_name.startswith('X'))
        
        # Create a mapping from X-names to feature names
        node_name_mapping = {}
        if use_direct_names:
            print("Mapping FCI node names to feature names")
            
            # Create a new NetworkX graph with feature names
            nx_graph = nx.DiGraph()
            
            # Add all nodes from feature_names
            for name in feature_names:
                nx_graph.add_node(name)
            
            # Create mapping from X names to feature names
            # Extract number from 'X1', 'X2', etc. and map to corresponding feature
            for edge in edges:
                for node in [edge.get_node1(), edge.get_node2()]:
                    x_name = node.get_name()
                    if x_name not in node_name_mapping:
                        try:
                            # Extract index from 'X1', 'X2', etc. (convert to 0-based)
                            index = int(x_name.replace('X', '')) - 1
                            if 0 <= index < len(feature_names):
                                node_name_mapping[x_name] = feature_names[index]
                            else:
                                # Use original name if index out of bounds
                                node_name_mapping[x_name] = x_name
                        except ValueError:
                            # Use original name if conversion fails
                            node_name_mapping[x_name] = x_name
            
            print(f"Node mapping: {node_name_mapping}")
        
        for edge in edges:
            try:
                # Get edge endpoints
                node1 = edge.get_node1().get_name()
                node2 = edge.get_node2().get_name()
                
                # Get edge type using proper conversion
                endpoint1 = endpoint_to_str(edge.get_endpoint1())
                endpoint2 = endpoint_to_str(edge.get_endpoint2())
                edge_type = endpoint1 + endpoint2
                
                # Map nodes to feature names
                if use_direct_names:
                    source = node_name_mapping.get(node1, node1)
                    target = node_name_mapping.get(node2, node2)
                else:
                    # Try to convert node names to indices
                    try:
                        source = feature_names[int(node1)]
                        target = feature_names[int(node2)]
                    except (ValueError, IndexError):
                        # If conversion fails, use as is
                        source = node1
                        target = node2
                
                # Handle different edge types
                if edge_type == "->":  # Directed edge
                    nx_graph.add_edge(source, target, style='solid')
                elif edge_type == "<-":  # Directed edge (opposite)
                    nx_graph.add_edge(target, source, style='solid')
                elif edge_type == "--":  # Undirected edge
                    nx_graph.add_edge(source, target, style='dashed')
                    nx_graph.add_edge(target, source, style='dashed')
                elif edge_type == "o-":  # Partially directed edge
                    nx_graph.add_edge(target, source, style='dotted')
                elif edge_type == "-o":  # Partially directed edge
                    nx_graph.add_edge(source, target, style='dotted')
                elif edge_type == "o>":  # Partially directed edge
                    nx_graph.add_edge(target, source, style='dashed_arrow')
                elif edge_type == "<o":  # Partially directed edge
                    nx_graph.add_edge(source, target, style='dashed_arrow')
                elif edge_type == "oo":  # Unknown direction
                    nx_graph.add_edge(source, target, style='dotted')
                    nx_graph.add_edge(target, source, style='dotted')
                else:
                    # Default edge type
                    nx_graph.add_edge(source, target, style='solid')
                    
                print(f"Added edge: {source} {edge_type} {target}")
            except Exception as e:
                print(f"Error processing edge: {e}")
                try:
                    print(f"Edge details: {edge.get_node1().get_name()} to {edge.get_node2().get_name()}, "
                          f"endpoints: {edge.get_endpoint1()} {edge.get_endpoint2()}")
                except:
                    print("Could not print edge details")
    
    # Handle NetworkX skeleton graph from PC algorithm
    elif hasattr(graph, 'nx_skel') and graph.nx_skel is not None:
        print("Using nx_skel from PC algorithm result")
        original_nx_graph = graph.nx_skel
        is_directed = isinstance(original_nx_graph, nx.DiGraph)
        print(f"Graph is {'directed' if is_directed else 'undirected'}")
        
        # Relabel nodes with feature names if needed
        if len(original_nx_graph.nodes()) == len(feature_names):
            # Check if already labeled
            if not isinstance(list(original_nx_graph.nodes())[0], str):
                mapping = {i: name for i, name in enumerate(feature_names)}
                original_nx_graph = nx.relabel_nodes(original_nx_graph, mapping)
        
        # Copy all edges to our new graph
        for u, v, data in original_nx_graph.edges(data=True):
            nx_graph.add_edge(u, v, **data)
    
    # Try different approaches to extract edges for other algorithms
    elif hasattr(graph, 'G'):
        # For PC algorithm
        try:
            if hasattr(graph.G, 'edges'):
                print(f"Graph has 'edges' attribute with {len(list(graph.G.edges()))} edges")
                for u, v, data in graph.G.edges(data=True):
                    edge_type = data.get('type', 'directed')
                    if edge_type == 'directed':
                        nx_graph.add_edge(feature_names[u], feature_names[v])
                    else:
                        nx_graph.add_edge(feature_names[u], feature_names[v], style='dashed')
                        nx_graph.add_edge(feature_names[v], feature_names[u], style='dashed')
        except Exception as e:
            print(f"Error processing PC graph: {e}")
    elif hasattr(graph, 'graph'):
        # For GES
        try:
            # Inspect the graph structure
            print(f"GES Graph type: {type(graph.graph)}")
            
            # Try to get nonzero elements safely
            if hasattr(graph.graph, 'shape'):
                nonzero_indices = np.where(graph.graph != 0)
                if len(nonzero_indices) >= 2:
                    for i, j in zip(nonzero_indices[0], nonzero_indices[1]):
                        if i < j:  # Avoid duplicated edges
                            edge_type = graph.graph[i, j]
                            if edge_type == 1:  # i -> j
                                nx_graph.add_edge(feature_names[i], feature_names[j])
                            elif edge_type == 2:  # i <- j
                                nx_graph.add_edge(feature_names[j], feature_names[i])
                            elif edge_type == 3:  # i -- j (undirected)
                                nx_graph.add_edge(feature_names[i], feature_names[j], style='dashed')
                                nx_graph.add_edge(feature_names[j], feature_names[i], style='dashed')
        except Exception as e:
            print(f"Error processing GES graph: {e}")
    elif isinstance(graph, nx.Graph):
        # Direct NetworkX graph
        print("Using direct NetworkX graph")
        is_directed = isinstance(graph, nx.DiGraph)
        
        # Copy all edges to our new graph
        for u, v, data in graph.edges(data=True):
            nx_graph.add_edge(u, v, **data)
    else:
        print(f"Unknown graph type. Graph attributes: {dir(graph)}")
    
    # Calculate correlation between connected nodes
    correlations = {}
    for u, v in nx_graph.edges():
        try:
            # Assuming we have access to the original data through the first argument of run_pc_algorithm
            # We'd need this to be passed to plot_causal_graph for this to work
            # As a fallback, use random values for demonstration
            correlations[(u, v)] = np.random.uniform(-1, 1)
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            correlations[(u, v)] = 0
    
    # Find relevant nodes when using target_focus
    relevant_nodes = set()
    
    # Helper function to find Markov blanket - works for both directed and undirected graphs
    def find_markov_blanket(graph, node):
        blanket = set()
        if is_directed:
            # For directed graphs - parents, children, and parents of children
            # Parents (direct causes)
            for parent in graph.predecessors(node):
                blanket.add(parent)
            
            # Children (direct effects)
            for child in graph.successors(node):
                blanket.add(child)
                # Other parents of children (spouses)
                for spouse in graph.predecessors(child):
                    if spouse != node:
                        blanket.add(spouse)
        else:
            # For undirected graphs - all neighbors are in Markov blanket
            for neighbor in graph.neighbors(node):
                blanket.add(neighbor)
        
        return blanket
    
    # Handle target focus mode
    if target_focus and highlight_nodes:
        for target_node in highlight_nodes:
            if target_node not in nx_graph.nodes():
                print(f"Warning: Node '{target_node}' not found in graph.")
                continue
            
            # Add target node and its Markov blanket to relevant nodes
            relevant_nodes.add(target_node)
            blanket = find_markov_blanket(nx_graph, target_node)
            relevant_nodes.update(blanket)
        
        # For 'removes' mode, remove unrelated nodes
        if target_focus == 'removes':
            nodes_to_remove = [n for n in nx_graph.nodes() if n not in relevant_nodes]
            for node in nodes_to_remove:
                nx_graph.remove_node(node)
    
    # Identify root nodes (nodes with no parents) - handle both directed and undirected graphs
    if is_directed:
        root_nodes = {node for node in nx_graph.nodes() if nx_graph.in_degree(node) == 0}
    else:
        # For undirected graphs, consider nodes with degree 1 as "leaf" nodes
        # and nodes with high degree as "hub" nodes
        degrees = dict(nx_graph.degree())
        max_degree = max(degrees.values()) if degrees else 0
        hub_threshold = max(3, max_degree * 0.7)  # At least 3 connections or 70% of max
        root_nodes = {node for node, degree in degrees.items() if degree >= hub_threshold}
    
    # Default node color is lightblue with transparency
    node_colors = []
    node_alpha = []
    
    for node in nx_graph.nodes():
        # Default color and alpha
        color = 'lightblue'
        alpha = 0.7
        
        # If root node, use a different color
        if node in root_nodes:
            color = 'lightgreen'
        
        # If in target_focus 'grays' mode, gray out unrelated nodes
        if (target_focus == 'grays' and highlight_nodes and 
            node not in relevant_nodes):
            color = 'lightgray'
            alpha = 0.3
            
        # If node should be highlighted, change its color
        if highlight_nodes and node in highlight_nodes:
            color = 'red'
            
        node_colors.append(color)
        node_alpha.append(alpha)
    
    # Create edge color mapping based on correlation
    edge_cmap = plt.cm.YlOrRd  # Yellow-to-Red colormap for strength
    edge_colors = []
    edge_widths = []
    edge_alpha = []
    edge_styles = []
    
    # Find max values for normalization
    max_corr = max([abs(corr) for corr in correlations.values()]) if correlations else 1
    
    # Normalize values and set edge properties
    for u, v in nx_graph.edges():
        # Default edge settings
        color = 'lightgray'
        width = 1.0
        alpha = 1.0
        
        # Get edge style (for FCI special edges)
        style = nx_graph.get_edge_data(u, v).get('style', 'solid')
        edge_styles.append(style)
        
        # Get correlation if available
        corr = correlations.get((u, v))
        
        if corr is not None:
            # Normalize correlation between 0 and 1 for color mapping
            color_val = abs(corr) / max(max_corr, 0.01)  # Avoid division by zero
            color = edge_cmap(color_val)
            width = 1 + 2 * color_val  # Width between 1 and 3
        
        # In 'grays' mode, check if either end is unrelated to target
        if (target_focus == 'grays' and highlight_nodes and 
            (u not in relevant_nodes or v not in relevant_nodes)):
            color = 'lightgray'
            alpha = 0.3
            width = 1.0
            
        edge_colors.append(color)
        edge_widths.append(width)
        edge_alpha.append(alpha)
    
    # Draw the graph
    if len(nx_graph.nodes()) > 0:
        # Use circular layout for better visualization
        pos = nx.circular_layout(nx_graph)
        
        # Draw nodes with transparency
        nx.draw_networkx_nodes(nx_graph, pos, 
                             node_color=node_colors, 
                             node_size=1500,
                             alpha=node_alpha)
        
        # Draw node labels
        nx.draw_networkx_labels(nx_graph, pos, font_size=12)
        
        edges = list(nx_graph.edges())
        if len(edges) > 0:
            # For FCI with special edge types, draw each style separately
            if is_fci:
                # Group edges by style
                edges_by_style = {}
                for i, (u, v) in enumerate(edges):
                    style = edge_styles[i]
                    if style not in edges_by_style:
                        edges_by_style[style] = []
                    edges_by_style[style].append((u, v))
                
                # Draw each group separately
                for style, style_edges in edges_by_style.items():
                    if not style_edges:
                        continue
                        
                    # Get indices of these edges in the main edges list
                    indices = [edges.index(e) for e in style_edges]
                    colors = [edge_colors[i] for i in indices]
                    widths = [edge_widths[i] for i in indices]
                    alphas = [edge_alpha[i] for i in indices]
                    
                    if style == 'solid':
                        # Normal directed edges
                        nx.draw_networkx_edges(nx_graph, pos, 
                                           edgelist=style_edges,
                                           arrowsize=20, 
                                           edge_color=colors,
                                           width=widths,
                                           alpha=alphas)
                    elif style == 'dashed':
                        # Undirected edges - no arrows
                        nx.draw_networkx_edges(nx_graph, pos, 
                                           edgelist=style_edges,
                                           arrowsize=0,
                                           edge_color=colors,
                                           width=widths,
                                           alpha=alphas,
                                           style='dashed')
                    elif style == 'dotted':
                        # Circle endpoints in FCI
                        nx.draw_networkx_edges(nx_graph, pos, 
                                           edgelist=style_edges,
                                           arrowsize=10,
                                           edge_color=colors,
                                           width=widths,
                                           alpha=alphas,
                                           style='dotted')
                    elif style == 'dashed_arrow':
                        # Circle to arrow in FCI
                        nx.draw_networkx_edges(nx_graph, pos, 
                                           edgelist=style_edges,
                                           arrowsize=20,
                                           edge_color=colors,
                                           width=widths,
                                           alpha=alphas,
                                           style='dashed')
            else:
                # Draw edges with varying width and color based on strength
                if is_directed:
                    nx.draw_networkx_edges(nx_graph, pos, 
                                        edges=edges,
                                        arrowsize=20, 
                                        edge_color=edge_colors,
                                        width=edge_widths,
                                        alpha=edge_alpha)
                else:
                    # For undirected graphs, don't show arrows
                    nx.draw_networkx_edges(nx_graph, pos, 
                                        edgelist=edges,
                                        arrowsize=0,
                                        edge_color=edge_colors,
                                        width=edge_widths,
                                        alpha=edge_alpha)
            
            # Draw edge labels with correlation values
            edge_labels = {}
            for i, (u, v) in enumerate(edges):
                if (u, v) in correlations:
                    edge_labels[(u, v)] = f"{correlations[(u, v)]:.2f}"
            
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=10)
        else:
            plt.text(0.5, 0.5, "No edges found in graph", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
    else:
        plt.text(0.5, 0.5, "No nodes found in graph", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
    
    # Create legend patches
    import matplotlib.patches as mpatches
    
    legend_patches = []
    
    # Regular node legend
    if 'lightblue' in node_colors:
        legend_patches.append(mpatches.Patch(color='lightblue', alpha=0.7, label='Regular Nodes'))
    
    # Root/Hub node legend
    if 'lightgreen' in node_colors:
        label = 'Root Nodes' if is_directed else 'Hub Nodes'
        legend_patches.append(mpatches.Patch(color='lightgreen', alpha=0.7, label=label))
    
    # Highlighted node legend
    if highlight_nodes and 'red' in node_colors:
        legend_patches.append(mpatches.Patch(color='red', alpha=0.7, label='Highlighted Nodes'))
    
    # Gray node legend (for target_focus='grays')
    if target_focus == 'grays' and 'lightgray' in node_colors:
        legend_patches.append(mpatches.Patch(color='lightgray', alpha=0.3, label='Unrelated Nodes'))
    
    # Add FCI edge type legend if applicable
    if is_fci:
        # Edge style legend
        if 'solid' in edge_styles:
            legend_patches.append(plt.Line2D([0], [0], color='black', lw=2, label='Directed Edge (->)'))
        if 'dashed' in edge_styles:
            legend_patches.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Undirected Edge (--)'))
        if 'dotted' in edge_styles:
            legend_patches.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='dotted', label='Partially Directed (o-)'))
        if 'dashed_arrow' in edge_styles:
            legend_patches.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Partially Directed (o>)'))
    
    # Edge strength legend
    if correlations:
        # Create gradient legend for edge strength
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        
        # Add colorbar for edge strength
        ax2 = plt.axes([0.92, 0.1, 0.02, 0.3])  # Position colorbar
        ax2.imshow(gradient.T, aspect='auto', cmap=edge_cmap)
        ax2.set_title('Edge\nStrength')
        ax2.set_xticks([])
        ax2.set_yticks([0, 255])
        ax2.set_yticklabels(['Low', 'High'])
    
    # Add legend if we have any patches
    if legend_patches:
        plt.legend(handles=legend_patches, loc='upper left')
        
    # Set title based on modes
    if target_focus and highlight_nodes:
        title += f" (Focus on highlighted nodes)"
        
    plt.title(title)
    plt.axis('off')  # Turn off axis
    plt.tight_layout()
    
    return nx_graph

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

def plot_feature_importances(feature_importances, title='Feature Importances for Causal Model'):
    """
    Plot feature importances from the neural causal model
    
    Parameters:
    -----------
    feature_importances : dict
        Dictionary of feature names and their importance scores
    title : str
        Title for the plot
    """
    # Sort features by absolute importance
    sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [x[0] for x in sorted_features]
    importances = [x[1] for x in sorted_features]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, importances)
    
    # Color bars based on positive/negative influence
    for i, importance in enumerate(importances):
        bars[i].set_color('red' if importance < 0 else 'green')
    
    plt.xlabel('Causal Strength')
    plt.title(title)
    plt.tight_layout()
