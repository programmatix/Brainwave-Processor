import os, json, codecs, time, hashlib
import numpy as np
from math import log, sqrt
from collections.abc import Iterable
from scipy import stats
from scipy.stats import chi2, norm

from notebooks.Stats.stats_binning import bin_fastcluster, determine_optimal_bin_count
from notebooks.Stats.stats_corr_best import analyze_pair_best

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



def test_conditional_independence(df, x_feat, y_feat, conditioning_feat, 
                                 n_clusters=3, random_state=42, 
                                 alpha=0.05, **kwargs):
    """
    Test whether x_feat and y_feat are conditionally independent given conditioning_feat.
    
    Returns:
        dict: Results containing conditional independence assessment
    """
    # Step 1: Test direct relationship between x and y
    direct_result = analyze_pair_best(df, x_feat, y_feat, n_clusters=n_clusters, 
                                     random_state=random_state, **kwargs)
    direct_p = direct_result.best_p
    
    # Step 2: Bin the conditioning variable
    df_valid = df[df[x_feat].notna() & df[y_feat].notna() & df[conditioning_feat].notna()]
    bin_count = determine_optimal_bin_count(df_valid[conditioning_feat], method='fastcluster')
    bin_res = bin_fastcluster(df_valid[conditioning_feat], n_bins=bin_count)
    bin_res._generate_bins_info()
    
    # Step 3: Test relationship within each bin of the conditioning variable
    conditional_p_values = []
    bin_results = []
    
    for bin_idx in range(bin_res.n_bins):
        bin_mask = bin_res.bin_assignments == bin_idx
        if sum(bin_mask) > 10:  # Ensure sufficient data in the bin
            bin_df = df_valid[bin_mask]
            # Analyze relationship within this bin
            bin_result = analyze_pair_best(bin_df, x_feat, y_feat, n_clusters=n_clusters, 
                                          random_state=random_state, **kwargs)
            conditional_p_values.append(bin_result.best_p)
            bin_results.append(bin_result)
    
    # Step 4: Fisher's method to combine p-values from conditional tests
    if conditional_p_values:
        chi_square = -2 * sum(np.log(p) for p in conditional_p_values)
        combined_p = 1 - stats.chi2.cdf(chi_square, df=2*len(conditional_p_values))
    else:
        combined_p = 1.0
    
    # Step 5: Assess conditional independence
    is_conditionally_independent = combined_p > alpha
    direct_significant = direct_p < alpha
    
    return {
        'x_feat': x_feat,
        'y_feat': y_feat,
        'conditioning_feat': conditioning_feat,
        'direct_p': direct_p,
        'conditional_p': combined_p,
        'is_conditionally_independent': is_conditionally_independent,
        'direct_relationship_exists': direct_significant,
        'is_mediator': direct_significant and is_conditionally_independent,
        'bin_results': bin_results,
        'direct_result': direct_result
    }

def visualize_conditional_independence(result, df, figsize=(18, 12)):
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
    
    x_feat = result['x_feat']
    y_feat = result['y_feat']
    z_feat = result['conditioning_feat']
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig)
    
    # Create title with summary of findings
    if result['is_mediator']:
        title = f"{z_feat} mediates the relationship between {x_feat} and {y_feat}"
        relationship_type = "MEDIATION"
    elif result['direct_relationship_exists'] and not result['is_conditionally_independent']:
        title = f"{z_feat} does NOT explain away the relationship between {x_feat} and {y_feat}"
        relationship_type = "DIRECT EFFECT REMAINS"
    elif not result['direct_relationship_exists']:
        title = f"No significant relationship between {x_feat} and {y_feat} to begin with"
        relationship_type = "NO RELATIONSHIP"
    else:
        title = f"Unclear relationship pattern between {x_feat}, {y_feat}, and {z_feat}"
        relationship_type = "COMPLEX RELATIONSHIP"
    
    fig.suptitle(f"{title}\n", fontsize=16, weight='bold')
    
    # 1. Original relationship scatterplot (X -> Y)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(x=x_feat, y=y_feat, data=df, alpha=0.5, ax=ax1)
    direct_p_str = f"p = {result['direct_p']:.4f}"
    if result['direct_relationship_exists']:
        direct_p_str += " (significant)"
    ax1.set_title(f"Direct relationship: {x_feat} → {y_feat}\n{direct_p_str}")
    
    # 2. Conditioning variable relationships
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(x=x_feat, y=z_feat, data=df, alpha=0.5, ax=ax2)
    
    # Calculate and show p-value for X -> Z relationship
    xz_result = analyze_pair_best(df, x_feat, z_feat)
    xz_p_str = f"p = {xz_result.best_p:.4f}"
    if xz_result.best_p < 0.05:
        xz_p_str += " (significant)"
    ax2.set_title(f"Relationship: {x_feat} → {z_feat}\n{xz_p_str}")
    
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Color points in Z-Y scatterplot based on their bin
    valid_data = df[df[x_feat].notna() & df[y_feat].notna() & df[z_feat].notna()].copy()
    discretizer = KBinsDiscretizer(n_bins=min(5, len(valid_data[z_feat].unique())), encode='ordinal', strategy='quantile')
    valid_data['bin'] = discretizer.fit_transform(valid_data[z_feat].values.reshape(-1, 1))
    sns.scatterplot(x=z_feat, y=y_feat, data=valid_data, hue='bin', palette='viridis', alpha=0.5, ax=ax3)
    
    # Calculate and show p-value for Z -> Y relationship
    zy_result = analyze_pair_best(df, z_feat, y_feat)
    zy_p_str = f"p = {zy_result.best_p:.4f}"
    if zy_result.best_p < 0.05:
        zy_p_str += " (significant)"
    ax3.set_title(f"Relationship: {z_feat} → {y_feat}\n{zy_p_str}")
    
    # Move explanation box to avoid overlapping with bottom graphs
    ax_diagram = fig.add_subplot(gs[2, :])
    ax_diagram.axis('off')
    
    # Draw causal diagram based on findings
    if result['is_mediator']:
        draw_mediation_path(ax_diagram, x_feat, y_feat, z_feat, 
                           direct_significant=False, 
                           x_to_z_significant=True, 
                           z_to_y_significant=True)
    elif result['direct_relationship_exists'] and not result['is_conditionally_independent']:
        draw_mediation_path(ax_diagram, x_feat, y_feat, z_feat, 
                           direct_significant=True, 
                           x_to_z_significant=True, 
                           z_to_y_significant=True)
    else:
        draw_mediation_path(ax_diagram, x_feat, y_feat, z_feat, 
                           direct_significant=result['direct_relationship_exists'], 
                           x_to_z_significant=False, 
                           z_to_y_significant=False)
    
    # Position explanation box above the causal diagram
    ax_explanation = fig.add_subplot(gs[1, :])
    ax_explanation.axis('off')
    
    # Update explanation text with more details
    explanation = generate_explanation_text(result)
    
    # Add detailed explanation of how conditional p-value is calculated
    explanation += "\n\nHOW CONDITIONAL P-VALUE IS CALCULATED:\n"
    explanation += f"1. The conditioning variable {z_feat} is binned into {discretizer.n_bins} groups\n"
    explanation += "2. Within each bin, we test the X-Y relationship\n"
    explanation += "3. The p-values from each bin are combined using Fisher's method\n"
    explanation += "4. This gives an overall p-value for X-Y relationship after controlling for Z"
    
    # Make the text box larger and more prominent
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_explanation.text(0.5, 0.5, explanation, transform=ax_explanation.transAxes, 
                       fontsize=12, verticalalignment='center', horizontalalignment='center',
                       bbox=props)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    x_feat = result['x_feat']
    y_feat = result['y_feat']
    z_feat = result['conditioning_feat']
    
    direct_p = result['direct_p']
    conditional_p = result['conditional_p']
    
    if result['is_mediator']:
        explanation = (
            f"MEDIATION DETECTED:\n\n"
            f"1. {x_feat} and {y_feat} are related (p={direct_p:.4f}).\n"
            f"2. When controlling for {z_feat}, this relationship disappears (p={conditional_p:.4f}).\n"
            f"3. This suggests {z_feat} mediates the relationship between {x_feat} and {y_feat}.\n\n"
            f"Interpretation: {x_feat} likely affects {y_feat} THROUGH {z_feat}."
        )
    elif result['direct_relationship_exists'] and not result['is_conditionally_independent']:
        explanation = (
            f"DIRECT RELATIONSHIP REMAINS:\n\n"
            f"1. {x_feat} and {y_feat} are related (p={direct_p:.4f}).\n"
            f"2. When controlling for {z_feat}, this relationship still exists (p={conditional_p:.4f}).\n"
            f"3. This suggests {z_feat} does NOT fully explain the relationship.\n\n"
            f"Interpretation: {x_feat} likely affects {y_feat} both directly AND through other pathways."
        )
    elif not result['direct_relationship_exists']:
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