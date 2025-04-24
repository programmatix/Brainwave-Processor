import os, json, codecs, time, hashlib
import numpy as np
from math import log, sqrt
from collections.abc import Iterable
from scipy.stats import chi2, norm


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
