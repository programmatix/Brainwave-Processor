import os, json, codecs, time, hashlib
import numpy as np
from math import log, sqrt
from collections.abc import Iterable
from scipy import stats
from scipy.stats import chi2, norm, spearmanr
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from notebooks.Stats.stats_binning import Bin, bin_fastcluster, determine_optimal_bin_count
from notebooks.Stats.stats_corr_best import PairAnalysisResult, analyze_pair_best

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import KBinsDiscretizer

# Copied from causal-learn
class CIT_Base(object):
    # Base class for CIT, contains basic operations for input check and caching, etc.
    def __init__(self, data, cache_path=None, **kwargs):
        '''
        Parameters
        ----------
        data: data matrix, np.ndarray, in shape (n_samples, n_features)
        cache_path: str, path to save cache .json file. default as None (no io to local file).
        kwargs: for future extension.
        '''
        assert isinstance(data, np.ndarray), "Input data must be a numpy array."
        self.data = data
        self.data_hash = hashlib.md5(str(data).encode('utf-8')).hexdigest()
        self.sample_size, self.num_features = data.shape
        self.cache_path = cache_path
        self.SAVE_CACHE_CYCLE_SECONDS = 30
        self.last_time_cache_saved = time.time()
        self.pvalue_cache = {'data_hash': self.data_hash}
        if cache_path is not None:
            assert cache_path.endswith('.json'), "Cache must be stored as .json file."  
            if os.path.exists(cache_path):
                with codecs.open(cache_path, 'r') as fin: self.pvalue_cache = json.load(fin)
                assert self.pvalue_cache['data_hash'] == self.data_hash, "Data hash mismatch."
            else: os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    def check_cache_method_consistent(self, method_name, parameters_hash):
        self.method = method_name
        if method_name not in self.pvalue_cache:
            self.pvalue_cache['method_name'] = method_name # a newly created cache
            self.pvalue_cache['parameters_hash'] = parameters_hash
        else:
            assert self.pvalue_cache['method_name'] == method_name, "CI test method name mismatch." # a loaded cache
            assert self.pvalue_cache['parameters_hash'] == parameters_hash, "CI test method parameters mismatch."

    def assert_input_data_is_valid(self, allow_nan=False, allow_inf=False):
        assert allow_nan or not np.isnan(self.data).any(), "Input data contains NaN. Please check."
        assert allow_inf or not np.isinf(self.data).any(), "Input data contains Inf. Please check."

    def save_to_local_cache(self):
        if not self.cache_path is None and time.time() - self.last_time_cache_saved > self.SAVE_CACHE_CYCLE_SECONDS:
            with codecs.open(self.cache_path, 'w') as fout: fout.write(json.dumps(self.pvalue_cache, indent=2))
            self.last_time_cache_saved = time.time()

    def get_formatted_XYZ_and_cachekey(self, X, Y, condition_set):
        '''
        reformat the input X, Y and condition_set to
            1. convert to built-in types for json serialization
            2. handle multi-dim unconditional variables (for kernel-based)
            3. basic check for valid input (X, Y no overlap with condition_set)
            4. generate unique and hashable cache key

        Parameters
        ----------
        X: int, or np.*int*, or Iterable<int | np.*int*>
        Y: int, or np.*int*, or Iterable<int | np.*int*>
        condition_set: Iterable<int | np.*int*>

        Returns
        -------
        Xs: List<int>, sorted. may swapped with Ys for cache key uniqueness.
        Ys: List<int>, sorted.
        condition_set: List<int>
        cache_key: string. Unique for <X,Y|S> in any input type or order.
        '''
        def _stringize(ulist1, ulist2, clist):
            # ulist1, ulist2, clist: list of ints, sorted.
            _strlst  = lambda lst: '.'.join(map(str, lst))
            return f'{_strlst(ulist1)};{_strlst(ulist2)}|{_strlst(clist)}' if len(clist) > 0 else \
                   f'{_strlst(ulist1)};{_strlst(ulist2)}'

        # every time when cit is called, auto save to local cache.
        self.save_to_local_cache()

        METHODS_SUPPORTING_MULTIDIM_DATA = ["kci"]
        if condition_set is None: condition_set = []
        # 'int' to convert np.*int* to built-in int; 'set' to remove duplicates; sorted for hashing
        condition_set = sorted(set(map(int, condition_set)))

        # usually, X and Y are 1-dimensional index (in constraint-based methods)
        if self.method not in METHODS_SUPPORTING_MULTIDIM_DATA:
            X, Y = (int(X), int(Y)) if (X < Y) else (int(Y), int(X))
            assert X not in condition_set and Y not in condition_set, "X, Y cannot be in condition_set."
            return [X], [Y], condition_set, _stringize([X], [Y], condition_set)

        # also to support multi-dimensional unconditional X, Y (usually in kernel-based tests)
        Xs = sorted(set(map(int, X))) if isinstance(X, Iterable) else [int(X)]  # sorted for comparison
        Ys = sorted(set(map(int, Y))) if isinstance(Y, Iterable) else [int(Y)]
        Xs, Ys = (Xs, Ys) if (Xs < Ys) else (Ys, Xs)
        assert len(set(Xs).intersection(condition_set)) == 0 and \
               len(set(Ys).intersection(condition_set)) == 0, "X, Y cannot be in condition_set."
        return Xs, Ys, condition_set, _stringize(Xs, Ys, condition_set)

    def __call__(self, X, Y, condition_set=None):
        """
        Perform an independence test.
        
        Parameters
        ----------
        X, Y: column indices of data
        condition_set: conditioning variables, default None
        
        Returns
        -------
        p: the p-value of the test
        """
        raise NotImplementedError("Subclasses must implement __call__ method")


NO_SPECIFIED_PARAMETERS_MSG = "NO SPECIFIED PARAMETERS"

def _profile_step(prev_time, label):
    now = time.perf_counter()
    print(f"MV_FisherZ profile: {label} took {now - prev_time:.6f}s")
    return now

class MV_FisherZ(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('mv_fisherz', NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid(allow_nan=True)

    # Get the indices of the rows that do not have missing values
    def _get_index_no_mv_rows(self, mvdata):
        nrow, ncol = np.shape(mvdata)
        bindxRows = np.ones((nrow,), dtype=bool)
        indxRows = np.array(list(range(nrow)))
        for i in range(ncol):
            bindxRows = np.logical_and(bindxRows, ~np.isnan(mvdata[:, i]))
        indxRows = indxRows[bindxRows]
        return indxRows

    def __call__(self, X, Y, condition_set=None, node_names = None, verbose = True):
        '''
        Perform an independence test using Fisher-Z's test for data with missing values.

        Parameters
        ----------
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        '''
        t0 = time.perf_counter()
        # Format inputs and check if we already computed this test
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        t1 = _profile_step(t0, "input formatting")
        # If we already ran this test, return the cached result
        if cache_key in self.pvalue_cache:
            return self.pvalue_cache[cache_key]

        try:
            # Combine all variables we need for this test, into a list.
            # Remember we're comparing X vs Y, and seeing if they still correlate when we know about other variables (condition_set)
            var = Xs + Ys + condition_set
            test_wise_deletion_XYcond_rows_index = self._get_index_no_mv_rows(self.data[:, var])
            # Make sure we have at least some complete rows
            if len(test_wise_deletion_XYcond_rows_index) == 0:
                print(f"MV_FisherZ X {node_names[X] if node_names else X} Y {node_names[Y] if node_names else Y} no rows in common")
                return 1.0
            # assert len(test_wise_deletion_XYcond_rows_index) != 0, \
            #     f"A test-wise deletion fisher-z test appears no overlapping data of involved variables. Please check the input data. X={node_names[X]} Y={node_names[Y]} condition_set={condition_set}"
        
            # Get the data subset with only the relevant variables and no missing values
            test_wise_deleted_data_var = self.data[test_wise_deletion_XYcond_rows_index][:, var]
            t2 = _profile_step(t1, "data selection")

            # Calculate the correlation matrix for this subset, using Pearson
            sub_corr_matrix = np.corrcoef(test_wise_deleted_data_var.T)
            t3 = _profile_step(t2, "corr matrix computation")

            # Try to calculate partial correlation using matrix inversion
            try:
                inv = np.linalg.inv(sub_corr_matrix)
            except np.linalg.LinAlgError:
                raise ValueError(f'Data correlation matrix is singular. Cannot run fisherz test. Please check your data. X={node_names[X]} Y={node_names[Y]} condition_set={condition_set}')
        
            # Calculate the partial correlation between X and Y given the conditioning set
            t4 = _profile_step(t3, "matrix inversion")
            r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))

            # Make sure r is within valid range
            if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(r) # may happen when samplesize is very small or relation is deterministic

            # Apply Fisher's Z transformation - transforms r into a normal distribution so we can calculate a p-value
            Z = 0.5 * log((1 + r) / (1 - r))

            # Calculate the test statistic
            X_stat = sqrt(len(test_wise_deletion_XYcond_rows_index) - len(condition_set) - 3) * abs(Z)
        
            # Calculate p-value
            t5 = _profile_step(t4, "statistic computation")
            p = 2 * (1 - norm.cdf(abs(X_stat)))

            # Cache and return the result
            self.pvalue_cache[cache_key] = p
            t6 = _profile_step(t5, "caching")

            if verbose:
                named_condition_set = [node_names[i] if node_names else i for i in condition_set]
                print(f"MV_FisherZ X {node_names[X] if node_names else X} Y {node_names[Y] if node_names else Y} condition_set {named_condition_set} r {r:.2f} Z {Z:.2f} X_stat {X_stat:.2f} p {p:.2f}")

            if np.isnan(p):
                return 1.0

            return p
        except Exception as e:
            print(f"MV_FisherZ X {node_names[X] if node_names else X} Y {node_names[Y] if node_names else Y} hit error {e}")
            return 1.0


@dataclass
class SpearmanResult:
    """Simplified result object for Spearman correlation."""
    x_feat: str
    y_feat: str 
    correlation: float
    p_value: float
    sample_size: int
    bin: Bin
    
    @property
    def best_p(self):
        """For compatibility with PairAnalysisResult interface."""
        return self.p_value
        
    @property
    def is_significant(self):
        """Check if p-value is significant (< 0.05)."""
        return self.p_value < 0.05

@dataclass
class ConditionalIndependenceResult:
    x_feat: str
    y_feat: str
    conditioning_feat: str
    direct_p: float
    conditional_p: float
    is_conditionally_independent: bool
    direct_relationship_exists: bool
    is_mediator: bool
    bin_results: List[SpearmanResult]
    direct_result: PairAnalysisResult
    
    def print_summary(self):
        """Print a concise, informative summary of the conditional independence test results."""
        print(f"\n{'=' * 60}")
        print(f"CONDITIONAL INDEPENDENCE TEST SUMMARY")
        print(f"{'=' * 60}")
        
        print(f"Variables: {self.x_feat} → {self.y_feat} | {self.conditioning_feat}")
        print(f"Direct p-value: {self.direct_p:.4f} {'(significant)' if self.direct_relationship_exists else '(not significant)'}")
        print(f"Conditional p-value: {self.conditional_p:.4f} {'(independent)' if self.is_conditionally_independent else '(not independent)'}")
        
        print("Bins:")
        for b in self.bin_results:
            print(f"  {b.bin.name}: {b.best_p:.4f} n={b.sample_size}")

        print(f"\n{'-' * 60}")
        print("INTERPRETATION:")
        
        if self.is_mediator:
            print(f"✓ MEDIATION DETECTED")
            print(f"  {self.x_feat} likely affects {self.y_feat} THROUGH {self.conditioning_feat}.")
            print(f"  The direct relationship disappears when controlling for {self.conditioning_feat}.")
        elif self.direct_relationship_exists and not self.is_conditionally_independent:
            print(f"✓ DIRECT RELATIONSHIP REMAINS")
            print(f"  {self.x_feat} likely affects {self.y_feat} both directly AND through other pathways.")
            print(f"  {self.conditioning_feat} does NOT fully explain the relationship.")
        elif not self.direct_relationship_exists:
            print(f"✓ NO SIGNIFICANT RELATIONSHIP")
            print(f"  {self.x_feat} and {self.y_feat} don't have a significant relationship.")
            print(f"  There is no relationship for {self.conditioning_feat} to mediate.")
        else:
            print(f"✓ COMPLEX RELATIONSHIP")
            print(f"  The relationship may be complex or involve other variables.")
        
        print(f"{'=' * 60}")


def test_conditional_independence(df, x_feat, y_feat, conditioning_feat, 
                                 n_clusters=3, random_state=42, 
                                 alpha=0.05, **kwargs):
    """
    Test whether x_feat and y_feat are conditionally independent given conditioning_feat.
    
    Parameters:
        df: DataFrame with the data
        x_feat: Name of the X variable
        y_feat: Name of the Y variable
        conditioning_feat: Name of the conditioning variable
        n_clusters: Number of clusters for binning
        random_state: Random state for reproducibility
        alpha: Significance level
        **kwargs: Additional arguments passed to analyze_pair_best for the direct relationship
        
    Returns:
        ConditionalIndependenceResult: Results containing conditional independence assessment
    """
    # Step 1: Test direct relationship between x and y
    direct_result = analyze_pair_best(df, x_feat, y_feat, n_clusters=n_clusters, 
                                     random_state=random_state, **kwargs)
    direct_p = direct_result.best_p
    
    # Step 2: Bin the conditioning variable
    df_valid = df[df[x_feat].notna() & df[y_feat].notna() & df[conditioning_feat].notna()]
    bin_count = determine_optimal_bin_count(df_valid[conditioning_feat], method='fastcluster')
    bin_res = bin_fastcluster(df_valid[conditioning_feat], n_bins=bin_count)
    
    # Step 3: Test relationship within each bin of the conditioning variable using Spearman correlation
    conditional_p_values = []
    bin_results = []
    
    for bin_idx in range(bin_res.n_bins):
        bin_mask = bin_res.bin_assignments == bin_idx
        if sum(bin_mask) > 5:  # Ensure sufficient data in the bin
            bin_df = df_valid[bin_mask]
            
            # Use Spearman's correlation for each bin
            # No point using analyze_pair_best as there won't be enough data in each bin to be worth clustering further
            corr, p_value = spearmanr(
                bin_df[x_feat].values, 
                bin_df[y_feat].values,
                nan_policy='omit'
            )
            bin_result = SpearmanResult(
                x_feat=x_feat,
                y_feat=y_feat,
                correlation=corr,
                p_value=p_value,
                sample_size=len(bin_df),
                bin=bin_res.bins[bin_idx]
            )
                
            conditional_p_values.append(bin_result.best_p)
            bin_results.append(bin_result)
    
    # Step 4: Fisher's method to combine p-values from conditional tests
    if conditional_p_values:
        chi_square = -2 * sum(np.log(p) for p in conditional_p_values)
        combined_p = 1 - stats.chi2.cdf(chi_square, df=2*len(conditional_p_values))
        #print(f"Combined p-value: {combined_p:.4f} from {len(conditional_p_values)} bins {conditional_p_values}")
    else:
        combined_p = 1.0
    
    # Step 5: Assess conditional independence
    is_conditionally_independent = combined_p > alpha
    direct_significant = direct_p < alpha
    
    return ConditionalIndependenceResult(
        x_feat=x_feat,
        y_feat=y_feat,
        conditioning_feat=conditioning_feat,
        direct_p=direct_p,
        conditional_p=combined_p,
        is_conditionally_independent=is_conditionally_independent,
        direct_relationship_exists=direct_significant,
        is_mediator=direct_significant and is_conditionally_independent,
        bin_results=bin_results,
        direct_result=direct_result
    )

def visualize_conditional_independence(result: ConditionalIndependenceResult, df, figsize=(18, 16)):
    """
    Visualize the results of a conditional independence test.
    
    Parameters:
    - result: The output from test_conditional_independence
    - df: The original dataframe with the data
    - figsize: Size of the figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import numpy as np
    from sklearn.preprocessing import KBinsDiscretizer
    
    x_feat = result.x_feat
    y_feat = result.y_feat
    z_feat = result.conditioning_feat
    
    # Create figure with properly sized grid
    fig = plt.figure(figsize=figsize)
    
    # Create custom grid with appropriate ratios for each section
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 2, 1.5, 1.5])
    
    # Create title with summary of findings
    if result.is_mediator:
        title = f"{z_feat} mediates the relationship between {x_feat} and {y_feat}"
        relationship_type = "MEDIATION"
    elif result.direct_relationship_exists and not result.is_conditionally_independent:
        title = f"{z_feat} does NOT explain away the relationship between {x_feat} and {y_feat}"
        relationship_type = "DIRECT EFFECT REMAINS"
    elif not result.direct_relationship_exists:
        title = f"No significant relationship between {x_feat} and {y_feat} to begin with"
        relationship_type = "NO RELATIONSHIP"
    else:
        title = f"Unclear relationship pattern between {x_feat}, {y_feat}, and {z_feat}"
        relationship_type = "COMPLEX RELATIONSHIP"
    
    fig.suptitle(f"{title}\n", fontsize=16, weight='bold')
    
    # 1. Original relationship scatterplot (X -> Y)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.regplot(x=x_feat, y=y_feat, data=df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax1)
    direct_p_str = f"p = {result.direct_p:.4f}"
    if result.direct_relationship_exists:
        direct_p_str += " (significant)"
    ax1.set_title(f"Direct relationship: {x_feat} → {y_feat}\n{direct_p_str}")
    
    # 2. Conditioning variable relationships
    ax2 = fig.add_subplot(gs[0, 1])
    sns.regplot(x=x_feat, y=z_feat, data=df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax2)
    
    # Calculate and show p-value for X -> Z relationship
    xz_corr, xz_p = stats.spearmanr(df[x_feat], df[z_feat], nan_policy='omit')
    xz_p_str = f"p = {xz_p:.4f} (rho = {xz_corr:.2f})"
    if xz_p < 0.05:
        xz_p_str += " (significant)"
    
    ax2.set_title(f"Relationship: {x_feat} → {z_feat}\n{xz_p_str}")
    
    # 3. Z-Y relationship with points colored by bins
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Prepare data for plotting
    valid_data = df[df[x_feat].notna() & df[y_feat].notna() & df[z_feat].notna()].copy()
    
    # Create a dictionary mapping bin ranges to bin indexes
    bin_mapping = {}
    bin_labels = []
    
    # Use the actual bins from the result instead of rebinning
    for i, bin_result in enumerate(result.bin_results):
        bin_range = (bin_result.bin.start, bin_result.bin.end)
        bin_mapping[bin_range] = i
        bin_labels.append(f"{z_feat}: {bin_range[0]:.2f} to {bin_range[1]:.2f}")
    
    # Assign bin index to each row in valid_data
    valid_data['bin'] = -1  # Default value for points that don't fall in any bin
    
    for i, bin_result in enumerate(result.bin_results):
        bin_range = (bin_result.bin.start, bin_result.bin.end)
        # Assign bin index to rows where z_feat is within this bin's range
        if i == len(result.bin_results) - 1:
            # For the highest bin, include the upper bound
            valid_data.loc[(valid_data[z_feat] >= bin_range[0]) & 
                          (valid_data[z_feat] <= bin_range[1]), 'bin'] = i
        else:
            # For other bins, exclude the upper bound
            valid_data.loc[(valid_data[z_feat] >= bin_range[0]) & 
                          (valid_data[z_feat] < bin_range[1]), 'bin'] = i
    
    # Plot points colored by bin
    palette = plt.cm.viridis(np.linspace(0, 1, len(result.bin_results)))
    
    # Plot the Z-Y relationship with points colored by bin
    for i, bin_result in enumerate(result.bin_results):
        bin_data = valid_data[valid_data['bin'] == i]
        ax3.scatter(bin_data[z_feat], bin_data[y_feat], 
                   color=palette[i], alpha=0.7, label=bin_labels[i])
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_data[z_feat], valid_data[y_feat])
    x_line = np.linspace(min(valid_data[z_feat]), max(valid_data[z_feat]), 100)
    y_line = intercept + slope * x_line
    ax3.plot(x_line, y_line, '-', color='red', linewidth=2)
    
    # Calculate and show p-value for Z -> Y relationship
    zy_corr, zy_p = stats.spearmanr(df[z_feat], df[y_feat], nan_policy='omit')
    zy_p_str = f"p = {zy_p:.4f} (rho = {zy_corr:.2f})"
    if zy_p < 0.05:
        zy_p_str += " (significant)"
            
    ax3.set_title(f"Relationship: {z_feat} → {y_feat}\n{zy_p_str}")
    
    # MAIN PLOT: Large scatter plot with conditional regression lines
    ax_main = fig.add_subplot(gs[1, :])
    
    # First, plot all points with light alpha
    ax_main.scatter(valid_data[x_feat], valid_data[y_feat], 
                   alpha=0.2, color='gray', s=20, label='All data points')
    
    # List to store legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # Add regression lines for each bin with their own colors
    bin_count = len(result.bin_results)
    for i, bin_result in enumerate(result.bin_results):
        bin_data = valid_data[valid_data['bin'] == i]
        if len(bin_data) > 5:  # Ensure sufficient data in the bin
            color = palette[i]
            
            x_vals = np.array(bin_data[x_feat])
            y_vals = np.array(bin_data[y_feat])
            
            # Plot the points for this bin
            scatter = ax_main.scatter(x_vals, y_vals, color=color, alpha=0.6, s=40)
            
            # Only calculate regression if we have multiple points
            if len(x_vals) > 2:
                # Calculate regression line
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                
                # Generate points for line across the range
                x_line = np.linspace(min(x_vals), max(x_vals), 100)
                y_line = intercept + slope * x_line
                
                # Plot regression line
                line, = ax_main.plot(x_line, y_line, '-', color=color, linewidth=2.5)
                
                # Add to legend with correlation coefficient and p-value
                p_value_str = f"p={bin_result.p_value:.4f}"
                rho_str = f"ρ={bin_result.correlation:.2f}"
                sample_size = f"n={bin_result.sample_size}"
                
                legend_handles.append(line)
                legend_labels.append(f"{bin_labels[i]}: {rho_str}, {p_value_str}, {sample_size}")
    
    # Add overall regression line
    x_vals_all = np.array(valid_data[x_feat])
    y_vals_all = np.array(valid_data[y_feat])
    if len(x_vals_all) > 2:
        slope_all, intercept_all, r_all, p_all, _ = stats.linregress(x_vals_all, y_vals_all)
        x_line_all = np.linspace(min(x_vals_all), max(x_vals_all), 100)
        y_line_all = intercept_all + slope_all * x_line_all
        overall_line, = ax_main.plot(x_line_all, y_line_all, '--', color='red', linewidth=3)
        
        # Add to legend
        legend_handles.append(overall_line)
        legend_labels.append(f"Overall relationship: ρ={r_all:.2f}, p={p_all:.4f}, n={len(x_vals_all)}")
    
    # Add legend with statistics
    ax_main.legend(legend_handles, legend_labels, title=f"Relationship by {z_feat} bins", 
                  loc='best', frameon=True, framealpha=0.9, fontsize=9)
    
    # Set title and labels
    ax_main.set_title(f"Relationship between {x_feat} and {y_feat} conditional on {z_feat}\np={result.conditional_p:.4f}", fontsize=14)
    ax_main.set_xlabel(x_feat, fontsize=12)
    ax_main.set_ylabel(y_feat, fontsize=12)
    
    # Add explanation annotation
    explanation_text = (
        f"How to read this plot: Each colored line shows the relationship between {x_feat} and {y_feat}\n"
        f"within a specific range of {z_feat}. If the lines differ in slope or significance,\n"
        f"then {z_feat} influences how {x_feat} and {y_feat} relate to each other."
    )
    
    # ax_main.annotate(explanation_text, xy=(0.5, -0.15), xycoords='axes fraction', 
    #                 ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", 
    #                                                   facecolor='wheat', alpha=0.5))
    
    # Explanation text
    ax_explanation = fig.add_subplot(gs[2, :])
    ax_explanation.axis('off')
    
    # Generate and display explanation text
    explanation = generate_explanation_text(result)
    explanation += "\n\nHOW CONDITIONAL P-VALUE IS CALCULATED:\n"
    explanation += f"1. The conditioning variable {z_feat} is binned into {bin_count} groups\n"
    explanation += f"2. Within each bin, we test if the {x_feat}-{y_feat} relationship holds using Spearman correlation\n"
    explanation += f"3. The p-values from each bin are combined using Fisher's method\n"
    explanation += f"4. This gives an overall p-value for the {x_feat}-{y_feat} relationship after controlling for {z_feat}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_explanation.text(0.5, 0.5, explanation, transform=ax_explanation.transAxes, 
                       fontsize=12, verticalalignment='center', horizontalalignment='center',
                       bbox=props)
    
    # Causal diagram
    ax_diagram = fig.add_subplot(gs[3, :])
    ax_diagram.axis('off')
    
    # Draw causal diagram based on findings
    if result.is_mediator:
        draw_mediation_path(ax_diagram, x_feat, y_feat, z_feat, 
                           direct_significant=False, 
                           x_to_z_significant=True, 
                           z_to_y_significant=True)
    elif result.direct_relationship_exists and not result.is_conditionally_independent:
        draw_mediation_path(ax_diagram, x_feat, y_feat, z_feat, 
                           direct_significant=True, 
                           x_to_z_significant=True, 
                           z_to_y_significant=True)
    else:
        draw_mediation_path(ax_diagram, x_feat, y_feat, z_feat, 
                           direct_significant=result.direct_relationship_exists, 
                           x_to_z_significant=False, 
                           z_to_y_significant=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.5, w_pad=1.0)
    
    return fig


def draw_mediation_path(ax, x_label, y_label, z_label, direct_significant=True, 
                       x_to_z_significant=True, z_to_y_significant=True):
    """Draw a causal path diagram showing mediation relationships"""
    import numpy as np
    
    # Define node positions
    x_pos = (0.2, 0.5)
    y_pos = (0.8, 0.5)
    z_pos = (0.5, 0.8)
    
    # Draw nodes
    node_size = 0.1
    circle_x = plt.Circle(x_pos, node_size, color='skyblue', zorder=10)
    circle_y = plt.Circle(y_pos, node_size, color='skyblue', zorder=10)
    circle_z = plt.Circle(z_pos, node_size, color='lightgreen', zorder=10)
    
    ax.add_patch(circle_x)
    ax.add_patch(circle_y)
    ax.add_patch(circle_z)
    
    # Add node labels
    ax.text(x_pos[0], x_pos[1]-0.15, x_label, ha='center', fontsize=12, fontweight='bold')
    ax.text(y_pos[0], y_pos[1]-0.15, y_label, ha='center', fontsize=12, fontweight='bold')
    ax.text(z_pos[0], z_pos[1]+0.15, z_label, ha='center', fontsize=12, fontweight='bold')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', linewidth=2, color='gray')
    
    # Direct path X -> Y
    if direct_significant:
        ax.annotate('', xy=y_pos, xytext=x_pos, arrowprops=dict(arrowstyle='->', linewidth=3, color='red'))
    else:
        # Draw dashed line if not significant
        ax.annotate('', xy=y_pos, xytext=x_pos, 
                   arrowprops=dict(arrowstyle='->', linewidth=1.5, color='gray', linestyle='--'))
    
    # Path X -> Z
    if x_to_z_significant:
        ax.annotate('', xy=z_pos, xytext=x_pos, arrowprops=dict(arrowstyle='->', linewidth=2, color='blue'))
    
    # Path Z -> Y
    if z_to_y_significant:
        ax.annotate('', xy=y_pos, xytext=z_pos, arrowprops=dict(arrowstyle='->', linewidth=2, color='blue'))
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)




def generate_explanation_text(result):
    """Generate a clear explanation of the conditional independence test results"""
    x_feat = result.x_feat
    y_feat = result.y_feat
    z_feat = result.conditioning_feat
    
    direct_p = result.direct_p
    conditional_p = result.conditional_p
    
    if result.is_mediator:
        explanation = (
            f"MEDIATION DETECTED:\n\n"
            f"1. {x_feat} and {y_feat} are related (p={direct_p:.4f}).\n"
            f"2. When controlling for {z_feat}, this relationship disappears (p={conditional_p:.4f}).\n"
            f"3. This suggests {z_feat} mediates the relationship between {x_feat} and {y_feat}.\n\n"
            f"Interpretation: {x_feat} likely affects {y_feat} THROUGH {z_feat}."
        )
    elif result.direct_relationship_exists and not result.is_conditionally_independent:
        explanation = (
            f"DIRECT RELATIONSHIP REMAINS:\n\n"
            f"1. {x_feat} and {y_feat} are related (p={direct_p:.4f}).\n"
            f"2. When controlling for {z_feat}, this relationship still exists (p={conditional_p:.4f}).\n"
            f"3. This suggests {z_feat} does NOT fully explain the relationship.\n\n"
            f"Interpretation: {x_feat} likely affects {y_feat} both directly AND through other pathways."
        )
    elif not result.direct_relationship_exists:
        explanation = (
            f"NO SIGNIFICANT RELATIONSHIP:\n\n"
            f"1. {x_feat} and {y_feat} don't have a significant relationship (p={direct_p:.4f}).\n"
            f"2. There is no relationship for {z_feat} to mediate.\n\n"
            f"Interpretation: No evidence of a relationship between {x_feat} and {y_feat}."
        )
    else:
        explanation = (
            f"COMPLEX RELATIONSHIP:\n\n"
            f"1. {x_feat} and {y_feat} are related (p={direct_p:.4f}).\n"
            f"2. The relationship after controlling for {z_feat} is unclear (p={conditional_p:.4f}).\n\n"
            f"Interpretation: The relationship may be complex or involve other variables."
        )
    
    return explanation

def process_conditional_independence_results(df, conditioning_feat, n_clusters=3, random_state=42, alpha=0.05, **kwargs):
    """
    Process each feature in the DataFrame and append conditional independence test results.
    
    Parameters:
        df: DataFrame with the data
        conditioning_feat: Name of the conditioning variable
        n_clusters: Number of clusters for binning
        random_state: Random state for reproducibility
        alpha: Significance level
        **kwargs: Additional arguments passed to analyze_pair_best for the direct relationship
        
    Returns:
        DataFrame: Copy of the input DataFrame with appended results
    """
    result_df = df.copy()
    
    for idx, row in df.iterrows():
        feat1 = row['feat1']
        
        # Assume feat1 is the x_feat and y_feat is another feature (adjust as needed)
        result = test_conditional_independence(
            df, 
            x_feat=feat1, 
            y_feat=conditioning_feat, 
            conditioning_feat=conditioning_feat,
            n_clusters=n_clusters, 
            random_state=random_state, 
            alpha=alpha, 
            **kwargs
        )
        
        # Append results to the DataFrame
        result_df.at[idx, 'conditional_p'] = result.conditional_p
        result_df.at[idx, 'is_conditionally_independent'] = result.is_conditionally_independent
        result_df.at[idx, 'is_mediator'] = result.is_mediator
        result_df.at[idx, 'direct_relationship_exists'] = result.direct_relationship_exists
    
    return result_df