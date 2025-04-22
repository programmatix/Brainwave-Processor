import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
import ruptures as rpt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
# from pyearth import Earth
import pymc as pm
import arviz as az
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

@dataclass
class ThresholdResult:
    """Standard container for threshold detection results to ensure consistency across methods."""
    name: str
    thresholds: np.ndarray
    scores: Optional[np.ndarray] = None
    model: Any = None
    computation_time: float = 0.0
    additional_info: Dict[str, Any] = field(default_factory=dict)

def detect_binary_segmentation(df, feat_x, feat_y, min_size=10, n_bkps=3, penalty=None, model="l2"):
    """
    Binary Segmentation for change point detection.
    
    Args:
        df: DataFrame with the data
        feat_x: Name of the x feature (horizontal axis)
        feat_y: Name of the y feature (vertical axis)
        min_size: Minimum segment size
        n_bkps: Number of breakpoints to find
        penalty: Penalty term for the cost function
        model: Cost model ("l1", "l2", "rbf", etc)
        
    Returns:
        ThresholdResult object
    """
    start_time = time.time()
    
    # Sort data by x values and extract features
    df_sorted = df.sort_values(by=feat_x).reset_index(drop=True)
    x = df_sorted[feat_x].values
    y = df_sorted[feat_y].values
    
    # Create the model
    algo = rpt.Binseg(model=model, min_size=min_size, jump=1).fit(y)
    
    # Find breakpoints
    if penalty is not None:
        bkps = algo.predict(pen=penalty)
    else:
        bkps = algo.predict(n_bkps=n_bkps)
    
    # Convert breakpoint indices to x-value thresholds
    thresholds = x[np.array(bkps[:-1])]  # exclude the last breakpoint which is the signal length
    
    # Calculate segment scores (here: variance reduction)
    segments = np.split(y, bkps[:-1])
    total_var = np.var(y) * len(y)
    segment_vars = sum(np.var(segment) * len(segment) for segment in segments)
    variance_reduction = 1 - (segment_vars / total_var)
    
    # Calculate threshold confidence scores based on change in mean
    confidence_scores = []
    for i in range(len(segments) - 1):
        mean_diff = abs(np.mean(segments[i]) - np.mean(segments[i+1]))
        std_pooled = np.sqrt((np.var(segments[i]) + np.var(segments[i+1])) / 2)
        effect_size = mean_diff / std_pooled if std_pooled > 0 else 0
        confidence_scores.append(min(1.0, effect_size))
    
    # Print detailed information
    print(f"\nBinary Segmentation Details:")
    print(f"  Model: {model}, Min Size: {min_size}, Requested Breakpoints: {n_bkps}")
    print(f"  Found {len(thresholds)} thresholds at: {thresholds}")
    for i, threshold in enumerate(thresholds):
        confidence = confidence_scores[i] if i < len(confidence_scores) else 0
        print(f"  - Threshold at {threshold:.4f}, confidence: {confidence:.4f}")
    print(f"  Overall variance reduction: {variance_reduction:.4f}")
    
    computation_time = time.time() - start_time
    
    return ThresholdResult(
        name="Binary Segmentation",
        thresholds=thresholds,
        scores=np.array(confidence_scores) if confidence_scores else np.ones(len(thresholds)),
        model=algo,
        computation_time=computation_time,
        additional_info={
            "bkps_indices": bkps[:-1],
            "cost_model": model,
            "n_segments": len(bkps),
            "variance_reduction": variance_reduction
        }
    )

def detect_pelt(df, feat_x, feat_y, min_size=10, penalty=None, model="l2"):
    """
    Pruned Exact Linear Time (PELT) change point detection.
    
    Args:
        df: DataFrame with the data
        feat_x: Name of the x feature (horizontal axis)
        feat_y: Name of the y feature (vertical axis)
        min_size: Minimum segment size
        penalty: Penalty term for the cost function
        model: Cost model ("l1", "l2", "rbf", etc)
        
    Returns:
        ThresholdResult object
    """
    start_time = time.time()
    
    # Sort data by x values and extract features
    df_sorted = df.sort_values(by=feat_x).reset_index(drop=True)
    x = df_sorted[feat_x].values
    y = df_sorted[feat_y].values
    
    # Set default penalty if none provided - make it MUCH less stringent
    if penalty is None:
        penalty = 'rbf' if model == 'rbf' else 0.5 * np.log(len(y))
    
    print(f"\nPELT Details:")
    print(f"  Model: {model}, Min Size: {min_size}, Penalty: {penalty}")
    
    # Create the model
    algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(y)
    
    # Try a range of decreasing penalties until we find some breakpoints
    penalties_to_try = [penalty, penalty/2, penalty/5, penalty/10, 1.0, 0.5, 0.1]
    
    bkps = []
    used_penalty = penalty
    
    for p in penalties_to_try:
        bkps = algo.predict(pen=p)
        if len(bkps) > 1:  # Found at least one breakpoint
            used_penalty = p
            print(f"  Found breakpoints with penalty={p}")
            break
    
    # Convert breakpoint indices to x-value thresholds - ensure integers
    bkps_indices = [int(idx) for idx in bkps[:-1]]  # exclude the last breakpoint
    
    thresholds = x[bkps_indices] if bkps_indices else np.array([])
    
    # Calculate segment scores and confidence
    confidence_scores = []
    if bkps_indices:
        segments = np.split(y, bkps_indices)
        total_var = np.var(y) * len(y)
        segment_vars = sum(np.var(segment) * len(segment) for segment in segments)
        variance_reduction = 1 - (segment_vars / total_var)
        
        # Calculate confidence based on change in mean between segments
        for i in range(len(segments) - 1):
            mean_diff = abs(np.mean(segments[i]) - np.mean(segments[i+1]))
            std_pooled = np.sqrt((np.var(segments[i]) + np.var(segments[i+1])) / 2)
            effect_size = mean_diff / std_pooled if std_pooled > 0 else 0
            confidence_scores.append(min(1.0, effect_size))
            
        # Print detailed information
        print(f"  Found {len(thresholds)} thresholds at: {thresholds}")
        for i, threshold in enumerate(thresholds):
            confidence = confidence_scores[i] if i < len(confidence_scores) else 0
            print(f"  - Threshold at {threshold:.4f}, confidence: {confidence:.4f}")
        print(f"  Overall variance reduction: {variance_reduction:.4f}")
    else:
        print("  No thresholds found even with multiple penalties.")
        variance_reduction = 0
    
    computation_time = time.time() - start_time
    
    return ThresholdResult(
        name="PELT",
        thresholds=thresholds,
        scores=np.array(confidence_scores) if confidence_scores else np.ones(len(thresholds)),
        model=algo,
        computation_time=computation_time,
        additional_info={
            "bkps_indices": bkps_indices,
            "cost_model": model,
            "n_segments": len(bkps),
            "used_penalty": used_penalty,
            "variance_reduction": variance_reduction if bkps_indices else 0
        }
    )

def detect_bayesian_changepoint(df, feat_x, feat_y, n_samples=1000, tune=1000, random_seed=42, threshold_prob=0.05):
    """
    Bayesian Change Point Detection.
    
    Args:
        df: DataFrame with the data
        feat_x: Name of the x feature (horizontal axis)
        feat_y: Name of the y feature (vertical axis)
        n_samples: Number of posterior samples
        tune: Number of tuning samples
        random_seed: Random seed for reproducibility
        threshold_prob: Probability threshold for detecting change points
        
    Returns:
        ThresholdResult object
    """
    start_time = time.time()
    
    print(f"\nBayesian Change Point Detection Details:")
    print(f"  Samples: {n_samples}, Threshold Probability: {threshold_prob}")
    
    # Sort data by x values and extract features
    df_sorted = df.sort_values(by=feat_x).reset_index(drop=True)
    x = df_sorted[feat_x].values
    y = df_sorted[feat_y].values
    n = len(y)
    
    # Standardize data
    scaler = StandardScaler()
    y_std = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Build a simpler model that will work with current PyMC
    with pm.Model() as model:
        # Prior on number of change points - keep it reasonable
        max_cp = min(10, n//10)
        print(f"  Maximum number of change points: {max_cp}")
        
        # Fixed number of possible change points
        # We'll use a Bernoulli variable for each potential change point
        # to decide if it's active or not
        potential_cp_idx = np.linspace(10, n-10, max_cp).astype(int)
        potential_cp_x = x[potential_cp_idx]
        
        # Probability of each change point being active
        p_active = pm.Beta("p_active", alpha=1, beta=4)  # Prior favoring fewer change points
        
        # Bernoulli variables for each potential change point
        active = pm.Bernoulli("active", p=p_active, shape=max_cp)
        
        # Get indices of segments
        segment_ends = []
        segment_ends.append(0)  # Start of data
        
        for i in range(max_cp):
            # If this potential change point is active, add its index
            idx = potential_cp_idx[i]
            segment_ends.append(pm.math.switch(active[i], idx, -1))
        
        segment_ends.append(n)  # End of data
        
        # Filter out inactive change points (those with idx = -1)
        segment_ends_filtered = segment_ends[0:1]  # Always include the start
        for i in range(1, max_cp+1):
            segment_ends_filtered = pm.math.concatenate([
                segment_ends_filtered,
                pm.math.switch(segment_ends[i] >= 0, 
                              segment_ends[i:i+1], 
                              pm.math.zeros(0, dtype='int64'))
            ])
        segment_ends_filtered = pm.math.concatenate([segment_ends_filtered, segment_ends[-1:]])  # Always include the end
        
        # Number of actual segments
        n_segments = pm.Deterministic("n_segments", pm.math.sum(active) + 1)
        
        # Priors for segment means and standard deviations
        means = pm.Normal('means', mu=0, sigma=2, shape=max_cp+1)
        sigmas = pm.HalfNormal('sigmas', sigma=2, shape=max_cp+1)
        
        # Likelihood
        for i in range(max_cp+1):
            # Check if this segment exists
            if i < max_cp:
                exists = active[i]
                start_idx = segment_ends[i]
                end_idx = pm.math.switch(
                    active[i],
                    segment_ends[i+1],
                    segment_ends[i+1]
                )
            else:
                exists = True
                start_idx = segment_ends[max_cp]
                end_idx = segment_ends[max_cp+1]
            
            # Add likelihood for this segment if it exists
            segment_length = end_idx - start_idx
            idx = pm.math.arange(start_idx, end_idx)
            
            # Add likelihood only if segment exists and has length > 0
            pm.Normal('y_segment_%d' % i,
                     mu=means[i],
                     sigma=sigmas[i],
                     observed=y_std[start_idx:end_idx],
                     observed_data=y_std[start_idx:end_idx])
        
        # Sample from posterior
        try:
            trace = pm.sample(n_samples, tune=tune, chains=2, random_seed=random_seed, 
                            progressbar=False, return_inferencedata=True)
            
            # Extract active change points
            active_samples = az.extract(trace, var_names=["active"])["active"].values
            
            # Calculate probability of each potential change point being active
            cp_probs = np.mean(active_samples, axis=0)
            
            # Find change points above threshold probability
            cp_indices = [potential_cp_idx[i] for i, prob in enumerate(cp_probs) if prob > threshold_prob]
            thresholds = x[cp_indices]
            
            print(f"  Found {len(thresholds)} thresholds at: {thresholds}")
            for i, idx in enumerate(cp_indices):
                print(f"  - Threshold at {x[idx]:.4f}, probability: {cp_probs[i]:.4f}")
                
            computation_time = time.time() - start_time
            
            return ThresholdResult(
                name="Bayesian Change Point",
                thresholds=thresholds,
                scores=cp_probs[cp_probs > threshold_prob] if len(cp_indices) > 0 else np.array([]),
                model=trace,
                computation_time=computation_time,
                additional_info={
                    "potential_cp_x": potential_cp_x,
                    "cp_probs": cp_probs,
                    "threshold_prob": threshold_prob
                }
            )
        except Exception as e:
            print(f"  Error in Bayesian sampling: {str(e)}")
            print("  Using fallback method...")
            
            # Fallback to a simpler change point detection method
            # Sliding window approach
            window_size = max(20, n // 10)
            step_size = max(5, window_size // 4)
            
            # We'll detect jumps in the mean
            means = np.array([np.mean(y_std[i:i+window_size]) 
                             for i in range(0, n-window_size, step_size)])
            x_positions = np.array([x[i+window_size//2] 
                                   for i in range(0, n-window_size, step_size)])
            
            # Calculate absolute differences between consecutive means
            mean_diffs = np.abs(np.diff(means))
            
            # Identify peaks in the differences
            threshold = np.mean(mean_diffs) + 1.5 * np.std(mean_diffs)
            peaks, _ = find_peaks(mean_diffs, height=threshold, distance=window_size/step_size)
            
            # Get the x positions of the peaks
            threshold_indices = peaks
            thresholds = x_positions[threshold_indices]
            
            # Calculate scores based on the height of the peaks
            scores = mean_diffs[peaks] / np.max(mean_diffs) if len(peaks) > 0 else np.array([])
            
            print(f"  Fallback method found {len(thresholds)} thresholds at: {thresholds}")
            for i, threshold in enumerate(thresholds):
                print(f"  - Threshold at {threshold:.4f}, score: {scores[i]:.4f}")
                
            computation_time = time.time() - start_time
            
            return ThresholdResult(
                name="Bayesian Change Point (Fallback)",
                thresholds=thresholds,
                scores=scores,
                model=None,
                computation_time=computation_time,
                additional_info={
                    "window_size": window_size,
                    "threshold": threshold,
                    "method": "sliding_window"
                }
            )

def detect_piecewise_linear(df, feat_x, feat_y, n_segments=4, max_iter=100, tol=1e-4):
    """
    Piecewise Linear Regression for change point detection.
    
    Args:
        df: DataFrame with the data
        feat_x: Name of the x feature (horizontal axis)
        feat_y: Name of the y feature (vertical axis)
        n_segments: Number of linear segments to fit
        max_iter: Maximum number of iterations for optimization
        tol: Tolerance for convergence
        
    Returns:
        ThresholdResult object
    """
    start_time = time.time()
    
    # Sort data by x values and extract features
    df_sorted = df.sort_values(by=feat_x).reset_index(drop=True)
    x = df_sorted[feat_x].values
    y = df_sorted[feat_y].values
    n = len(x)
    
    # Ensure n_segments is at least 2
    n_segments = max(2, n_segments)
    
    print(f"\nPiecewise Linear Details:")
    print(f"  Segments: {n_segments}, Max Iterations: {max_iter}")
    
    # Initialize breakpoints evenly spaced
    breakpoints = np.linspace(0, n-1, n_segments+1).astype(int)[1:-1]
    prev_breakpoints = np.copy(breakpoints)
    
    # Iterative optimization
    converged = False
    iter_count = 0
    
    while not converged and iter_count < max_iter:
        # Fit linear regression to each segment
        segments = np.split(np.arange(n), breakpoints)
        mse = 0
        
        for segment in segments:
            if len(segment) > 1:  # Ensure segment has enough points
                x_seg = x[segment]
                y_seg = y[segment]
                model = LinearRegression().fit(x_seg.reshape(-1, 1), y_seg)
                y_pred = model.predict(x_seg.reshape(-1, 1))
                mse += np.sum((y_seg - y_pred) ** 2)
        
        # Optimize breakpoint positions
        for i in range(len(breakpoints)):
            best_mse = float('inf')
            best_pos = breakpoints[i]
            
            # Set search limits for this breakpoint
            lower = 1 if i == 0 else breakpoints[i-1] + 1
            upper = n-1 if i == len(breakpoints)-1 else breakpoints[i+1] - 1
            
            for pos in range(lower, upper):
                # Try this position
                test_breakpoints = np.copy(breakpoints)
                test_breakpoints[i] = pos
                
                # Calculate MSE with this position
                test_segments = np.split(np.arange(n), test_breakpoints)
                test_mse = 0
                
                for segment in test_segments:
                    if len(segment) > 1:
                        x_seg = x[segment]
                        y_seg = y[segment]
                        model = LinearRegression().fit(x_seg.reshape(-1, 1), y_seg)
                        y_pred = model.predict(x_seg.reshape(-1, 1))
                        test_mse += np.sum((y_seg - y_pred) ** 2)
                
                if test_mse < best_mse:
                    best_mse = test_mse
                    best_pos = pos
            
            breakpoints[i] = best_pos
        
        # Check convergence
        if np.all(np.abs(breakpoints - prev_breakpoints) < tol):
            converged = True
        
        prev_breakpoints = np.copy(breakpoints)
        iter_count += 1
    
    # Convert breakpoint indices to x-value thresholds
    thresholds = x[breakpoints]
    
    # Fit final model with optimal breakpoints
    segments = np.split(np.arange(n), breakpoints)
    models = []
    mse = 0
    r2 = 0
    
    # Calculate confidence scores based on slope changes
    confidence_scores = []
    segment_models = []
    
    for i, segment in enumerate(segments):
        if len(segment) > 1:
            x_seg = x[segment]
            y_seg = y[segment]
            model = LinearRegression().fit(x_seg.reshape(-1, 1), y_seg)
            segment_models.append((model.coef_[0], model.intercept_))
            y_pred = model.predict(x_seg.reshape(-1, 1))
            mse += np.sum((y_seg - y_pred) ** 2) / n
            
            # Calculate segment R2
            y_mean = np.mean(y_seg)
            ss_tot = np.sum((y_seg - y_mean) ** 2)
            ss_res = np.sum((y_seg - y_pred) ** 2)
            if ss_tot > 0:
                segment_r2 = 1 - (ss_res / ss_tot)
                r2 += segment_r2 * len(segment) / n
            
            models.append(model)
    
    # Calculate confidence based on slope changes
    for i in range(len(segment_models) - 1):
        slope_diff = abs(segment_models[i][0] - segment_models[i+1][0])
        max_slope = max(abs(segment_models[i][0]), abs(segment_models[i+1][0]))
        confidence = min(1.0, slope_diff / max_slope) if max_slope > 0 else 0
        confidence_scores.append(confidence)
    
    # Print detailed information
    print(f"  Found {len(thresholds)} thresholds at: {thresholds}")
    print(f"  Iterations: {iter_count}, Converged: {converged}")
    for i, threshold in enumerate(thresholds):
        confidence = confidence_scores[i] if i < len(confidence_scores) else 0
        left_slope = segment_models[i][0] if i < len(segment_models) else 0
        right_slope = segment_models[i+1][0] if i+1 < len(segment_models) else 0
        print(f"  - Threshold at {threshold:.4f}, confidence: {confidence:.4f}")
        print(f"    Left slope: {left_slope:.4f}, right slope: {right_slope:.4f}")
    print(f"  Overall R²: {r2:.4f}")
    
    computation_time = time.time() - start_time
    
    return ThresholdResult(
        name="Piecewise Linear",
        thresholds=thresholds,
        scores=np.array(confidence_scores) if confidence_scores else np.ones(len(thresholds)),
        model=models,
        computation_time=computation_time,
        additional_info={
            "bkps_indices": breakpoints,
            "iterations": iter_count,
            "converged": converged,
            "mse": mse,
            "r2": r2,
            "segment_models": segment_models
        }
    )

def detect_regression_tree(df, feat_x, feat_y, max_depth=3, min_samples_split=10):
    """
    Regression Tree for threshold detection.
    
    Args:
        df: DataFrame with the data
        feat_x: Name of the x feature (horizontal axis)
        feat_y: Name of the y feature (vertical axis)
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        
    Returns:
        ThresholdResult object
    """
    start_time = time.time()
    
    # Extract features
    x = df[feat_x].values.reshape(-1, 1)
    y = df[feat_y].values
    
    print(f"\nRegression Tree Details:")
    print(f"  Max Depth: {max_depth}, Min Samples Split: {min_samples_split}")
    
    # Fit decision tree
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    tree.fit(x, y)
    
    # Extract thresholds and importance scores from the tree
    thresholds = []
    importances = []
    
    def extract_thresholds(tree, node_id=0):
        # Check if the node is a leaf
        if tree.children_left[node_id] == -1:
            return
        
        # Add the threshold from this decision node
        if tree.feature[node_id] == 0:  # If split on the x feature
            thresholds.append(tree.threshold[node_id])
            # Calculate importance as weighted impurity decrease
            n_node = tree.n_node_samples[node_id]
            n_left = tree.n_node_samples[tree.children_left[node_id]]
            n_right = tree.n_node_samples[tree.children_right[node_id]]
            
            impurity_decrease = (
                n_node * tree.impurity[node_id] - 
                n_left * tree.impurity[tree.children_left[node_id]] -
                n_right * tree.impurity[tree.children_right[node_id]]
            ) / n_node
            
            importances.append(impurity_decrease)
        
        # Recurse to children
        extract_thresholds(tree, tree.children_left[node_id])
        extract_thresholds(tree, tree.children_right[node_id])
    
    extract_thresholds(tree.tree_)
    
    # Normalize importances to [0, 1]
    if importances:
        max_importance = max(importances)
        if max_importance > 0:
            importances = [i / max_importance for i in importances]
    
    # Sort thresholds and corresponding importances
    sorted_idx = np.argsort(thresholds)
    thresholds = np.array(thresholds)[sorted_idx]
    importances = np.array(importances)[sorted_idx] if importances else np.ones(len(thresholds))
    
    # Calculate R-squared as score
    y_pred = tree.predict(x)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Print detailed information
    print(f"  Found {len(thresholds)} thresholds at: {thresholds}")
    for i, threshold in enumerate(thresholds):
        importance = importances[i]
        print(f"  - Threshold at {threshold:.4f}, importance: {importance:.4f}")
    print(f"  Overall R²: {r2:.4f}")
    
    computation_time = time.time() - start_time
    
    return ThresholdResult(
        name="Regression Tree",
        thresholds=thresholds,
        scores=importances,
        model=tree,
        computation_time=computation_time,
        additional_info={
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "r2": r2
        }
    )

def detect_mars(df, feat_x, feat_y, max_terms=10, penalty=2.0):
    """
    Multivariate Adaptive Regression Splines (MARS) for threshold detection.
    
    Args:
        df: DataFrame with the data
        feat_x: Name of the x feature (horizontal axis)
        feat_y: Name of the y feature (vertical axis)
        max_terms: Maximum number of terms in the model
        penalty: GCV penalty per knot
        
    Returns:
        ThresholdResult object
    """
    start_time = time.time()
    
    # Extract features
    x = df[feat_x].values.reshape(-1, 1)
    y = df[feat_y].values
    
    # Fit MARS model
    mars = Earth(max_terms=max_terms, penalty=penalty, allow_linear=True)
    mars.fit(x, y)
    
    # Extract knots (thresholds) from the model
    thresholds = []
    
    for bf in mars.basis_:
        if hasattr(bf, 'get_knots'):
            knots = bf.get_knots()
            if len(knots) > 0:
                thresholds.extend(knots[0])
    
    thresholds = np.sort(np.unique(thresholds))
    
    # Calculate R-squared as score
    y_pred = mars.predict(x)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    computation_time = time.time() - start_time
    
    return ThresholdResult(
        name="MARS",
        thresholds=thresholds,
        scores=np.array([r2]),
        model=mars,
        computation_time=computation_time,
        additional_info={
            "n_terms": len(mars.basis_),
            "r2": r2,
            "gcv": mars.gcv_
        }
    )

def compare_all_threshold_methods(df, feat_x, feat_y, **kwargs):
    """
    Compare all available threshold detection methods.
    
    Args:
        df: DataFrame with the data
        feat_x: Name of the x feature (horizontal axis)
        feat_y: Name of the y feature (vertical axis)
        **kwargs: Additional parameters for the specific methods
    
    Returns:
        List of ThresholdResult objects
    """
    # Extract default parameters from kwargs
    binary_seg_params = {
        'min_size': kwargs.get('binary_seg_min_size', 10),
        'n_bkps': kwargs.get('binary_seg_n_bkps', 3),
        'model': kwargs.get('binary_seg_model', 'l2')
    }
    
    pelt_params = {
        'min_size': kwargs.get('pelt_min_size', 10),
        'model': kwargs.get('pelt_model', 'l2')
    }
    
    bayesian_params = {
        'n_samples': kwargs.get('bayesian_n_samples', 1000),
        'tune': kwargs.get('bayesian_tune', 1000),
        'threshold_prob': kwargs.get('bayesian_threshold_prob', 0.3)
    }
    
    piecewise_params = {
        'n_segments': kwargs.get('piecewise_n_segments', 4),
        'max_iter': kwargs.get('piecewise_max_iter', 100)
    }
    
    tree_params = {
        'max_depth': kwargs.get('tree_max_depth', 3),
        'min_samples_split': kwargs.get('tree_min_samples_split', 10)
    }
    
    mars_params = {
        'max_terms': kwargs.get('mars_max_terms', 10),
        'penalty': kwargs.get('mars_penalty', 2.0)
    }
    
    # Run all methods
    methods = [
        (detect_binary_segmentation, binary_seg_params),
        (detect_pelt, pelt_params),
        (detect_bayesian_changepoint, bayesian_params),
        (detect_piecewise_linear, piecewise_params),
        (detect_regression_tree, tree_params),
        # (detect_mars, mars_params)
    ]
    
    results = []
    
    for method_func, params in methods:
        try:
            result = method_func(df, feat_x, feat_y, **params)
            results.append(result)
            print(f"✓ {result.name}: {len(result.thresholds)} thresholds found in {result.computation_time:.2f}s")
        except Exception as e:
            warnings.warn(f"Method {method_func.__name__} failed: {str(e)}")
    
    return results

def plot_thresholds(df, feat_x, feat_y, results, figsize=(15, 10), alpha=0.5, display_time=True, individual_plots=True, graphs_per_row=3):
    """
    Plot detected thresholds on the data.
    
    Args:
        df: DataFrame with the data
        feat_x: Name of the x feature (horizontal axis)
        feat_y: Name of the y feature (vertical axis)
        results: List of ThresholdResult objects
        figsize: Figure size
        alpha: Alpha value for the data points
        display_time: Whether to display computation times
        individual_plots: Whether to create individual plots for each method
        graphs_per_row: Number of graphs per row when using individual plots
        
    Returns:
        None (displays plots)
    """
    # Define colors here so they're available for all plot sections
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    if individual_plots:
        # Filter out results with no thresholds
        plot_results = [r for r in results if len(r.thresholds) > 0]
        
        if not plot_results:
            print("No thresholds found by any method to plot.")
            return
            
        # Calculate number of rows needed
        n_methods = len(plot_results)
        n_rows = (n_methods + graphs_per_row - 1) // graphs_per_row
        
        # Create subplot grid
        fig, axes = plt.subplots(n_rows, graphs_per_row, figsize=(figsize[0], figsize[1] * n_rows / 2))
        
        # Make axes accessible for both single row and multi-row cases
        if n_rows == 1 and graphs_per_row == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_methods <= graphs_per_row:
            axes = axes.reshape(1, -1)
        
        # Create individual plots
        for i, result in enumerate(plot_results):
            # Calculate position in grid
            row = i // graphs_per_row
            col = i % graphs_per_row
            
            # Get the axis to plot on
            ax = axes[row, col]
            
            # Plot the data and thresholds
            ax.scatter(df[feat_x], df[feat_y], alpha=alpha, s=10)
            
            # Plot thresholds with varying line width based on confidence
            for j, threshold in enumerate(result.thresholds):
                score = result.scores[j] if j < len(result.scores) else 1.0
                # Scale line width and alpha with confidence
                linewidth = 1 + 3 * score
                line_alpha = 0.3 + 0.7 * score
                ax.axvline(x=threshold, color='r', linestyle='--', 
                          alpha=line_alpha, linewidth=linewidth)
                # Add text annotation for confidence
                y_pos = ax.get_ylim()[0] + 0.95 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.text(threshold, y_pos, f"{score:.2f}", 
                       backgroundcolor='white', fontsize=8,
                       horizontalalignment='center')
            
            ax.set_title(f"{result.name}\n{len(result.thresholds)} thresholds", fontsize=12)
            ax.set_xlabel(feat_x, fontsize=10)
            ax.set_ylabel(feat_y, fontsize=10)
            
            if display_time:
                ax.text(0.02, 0.98, f"Time: {result.computation_time:.2f}s", 
                      transform=ax.transAxes, fontsize=8, va='top')
        
        # Hide unused subplots
        for i in range(n_methods, n_rows * graphs_per_row):
            row = i // graphs_per_row
            col = i % graphs_per_row
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        # Combined plot with all methods
        plt.figure(figsize=figsize)
        plt.scatter(df[feat_x], df[feat_y], alpha=alpha, s=10, label='Data')
        
        # Plot thresholds for all methods
        for i, result in enumerate(results):
            if len(result.thresholds) == 0:
                continue
                
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            for j, threshold in enumerate(result.thresholds):
                plt.axvline(x=threshold, color=color, linestyle=linestyle, alpha=0.7,
                          label=f"{result.name}" if j == 0 else "")
        
        plt.title(f"Comparison of Threshold Detection Methods", fontsize=14)
        plt.xlabel(feat_x, fontsize=12)
        plt.ylabel(feat_y, fontsize=12)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    
    # Plot computation times
    if display_time and len(results) > 1:
        plt.figure(figsize=(10, 5))
        names = [result.name for result in results]
        times = [result.computation_time for result in results]
        
        plt.barh(names, times, color=colors[:len(results)])
        plt.xlabel('Computation Time (seconds)', fontsize=12)
        plt.title('Method Performance Comparison', fontsize=14)
        
        # Add time values as text
        for i, time in enumerate(times):
            plt.text(time + max(times) * 0.02, i, f"{time:.2f}s", va='center')
        
        plt.tight_layout()
        plt.show()

def suggest_optimal_thresholds(results, max_thresholds=3):
    """
    Suggest optimal thresholds based on all methods.
    
    Args:
        results: List of ThresholdResult objects
        max_thresholds: Maximum number of thresholds to return
        
    Returns:
        Array of suggested optimal thresholds
    """
    all_thresholds = []
    
    # Collect all thresholds from all methods
    for result in results:
        all_thresholds.extend(result.thresholds)
    
    all_thresholds = np.array(all_thresholds)
    
    if len(all_thresholds) == 0:
        return np.array([])
    
    # Use KDE to find peaks in the distribution of thresholds
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    
    # Handle the case where there are too few thresholds for KDE
    if len(all_thresholds) < 3:
        return np.sort(all_thresholds)[:max_thresholds]
    
    # Compute KDE
    kde = gaussian_kde(all_thresholds)
    
    # Evaluate KDE on a grid
    x_grid = np.linspace(min(all_thresholds), max(all_thresholds), 1000)
    y_kde = kde(x_grid)
    
    # Find peaks in the KDE
    peaks, _ = find_peaks(y_kde, height=0)
    peak_heights = y_kde[peaks]
    
    # Sort peaks by height (density)
    sorted_indices = np.argsort(peak_heights)[::-1]
    sorted_peaks = peaks[sorted_indices]
    
    # Get the x values at the peaks (these are the suggested thresholds)
    suggested_thresholds = x_grid[sorted_peaks[:max_thresholds]]
    
    return suggested_thresholds

def find_optimal_thresholds(df, feat_x, feat_y, **kwargs):
    """
    Find optimal thresholds using multiple methods and suggest the best ones.
    
    Args:
        df: DataFrame with the data
        feat_x: Name of the x feature (horizontal axis) 
        feat_y: Name of the y feature (vertical axis)
        **kwargs: Additional parameters for the specific methods
        
    Returns:
        Tuple of (results, suggested_thresholds)
    """
    print("Running all threshold detection methods...")
    results = compare_all_threshold_methods(df, feat_x, feat_y, **kwargs)
    
    max_thresholds = kwargs.get('max_suggested_thresholds', 3)
    suggested = suggest_optimal_thresholds(results, max_thresholds=max_thresholds)
    
    print(f"\nSuggested optimal thresholds: {suggested}")
    
    return results, suggested

# Example usage:
# df = pd.DataFrame({
#     'x': np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100), np.random.normal(10, 1, 100)]),
#     'y': np.concatenate([np.random.normal(0, 1, 100), np.random.normal(3, 1, 100), np.random.normal(1, 1, 100)])
# })
# df = df.sort_values(by='x').reset_index(drop=True)
# 
# results, suggested = find_optimal_thresholds(df, 'x', 'y')
# plot_thresholds(df, 'x', 'y', results, display_time=True)
