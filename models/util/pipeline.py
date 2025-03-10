import pandas as pd
import json

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler


class TransformOrDropUnusableTypes(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()

        object_columns = out.select_dtypes(include=['object']).columns

        if len(object_columns) > 0:
            print(f"TransformOrDropUnusableTypes, dropping cols {len(object_columns)} with type object")

            out = out.drop(columns=object_columns)

        datetime_columns = out.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, Europe/London]', 'datetime', 'timedelta']).columns

        if len(datetime_columns) > 0:
            print(f"TransformOrDropUnusableTypes, dropping cols {len(datetime_columns)} with type datetime64")

            out = out.drop(columns=datetime_columns)


        print(f"TransformOrDropUnusableTypes {X.shape} to {out.shape}, first index {out.index[0]}")
        return out

class DropBadRows(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Identify rows with NaN values
        bad_rows = X.isin([np.inf, -np.inf]).any(axis=1)

        # Log the indexes of dropped and kept rows
        dropped_indexes = X[bad_rows].index.tolist()
        kept_indexes = X[~bad_rows].index.tolist()

        # Log the first column that had NaN for each dropped row
        #reasons = X[bad_rows].apply(lambda row: row[row.isin([np.inf, -np.inf])].index[0], axis=1).tolist()
        #
        # print(f"Dropped row indexes: {dropped_indexes}")
        # print(f"Kept row indexes: {kept_indexes}")
        # print(f"Reasons for dropping: {reasons}")

        out = X[~bad_rows]

        print(f"DropBadRows {X.shape} to {out.shape}, first index {out.index[0]}")
        # print(f"DropBadRows: before {len(X)} rows after {len(out)} rows")
        return out

class RequireNonEmptyRows(BaseEstimator, TransformerMixin):
    def __init__(self, rows_must_be_non_empty: list[str] = None):
        self.rows_must_be_non_empty = rows_must_be_non_empty

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Remove rows with NaN, NaT, or other missing values in the specified columns
        out = X.dropna(subset=self.rows_must_be_non_empty)
        # print(f"RequireNonEmptyRows: before {len(X)} rows after {len(out)} rows")
        print(f"RequireNonEmptyRows {X.shape} to {out.shape}, first index {out.index[0]}")
        return out

class Condition(BaseEstimator, TransformerMixin):
    def __init__(self, condition: callable):
        self.condition = condition

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Remove rows with NaN, NaT, or other missing values in the specified columns
        out = self.condition(X)
        # print(f"Condition: before {len(X)} rows after {len(out)} rows")
        print(f"Condition {X.shape} to {out.shape}, first index {out.index[0]}")
        return out

# Removes outliers.  Currently just so we have visualise things better in dtreeviz etc.
class ClipOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, scale_target=False):
        self.target_col = target_col
        self.scale_target = scale_target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        exclude_cols = [] # col for col in X if "_norm" in col]

        # Calculate the quantiles for all columns at once
        lower_quantiles = X.quantile(0.01)
        upper_quantiles = X.quantile(0.99)

        # Clip the values for all columns except those in exclude_cols
        out = out.clip(lower=lower_quantiles, upper=upper_quantiles, axis=1)

        # HRV seems especially prone to outliers
        hrv = [col for col in X.columns if col.startswith("hrv:")]
        out = out.apply(lambda col: col.clip(lower=0, upper=100) if col.name in hrv else col)


        print(f"ClipOutliers {X.shape} to {out.shape}, first index {out.index[0]}")
        return out

class ClipAbsOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, scale_target=False):
        self.target_col = target_col
        self.scale_target = scale_target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        exclude_cols = [] # col for col in X if "_norm" in col]

        # Need to clip these aggressively
        # lower_quantiles = X.quantile(0.25)
        # upper_quantiles = X.quantile(0.75)

        # Clip the values for all columns except those in exclude_cols
        # out = out.clip(lower=lower_quantiles, upper=upper_quantiles, axis=1)

        # ... very aggressively.  This is still a very high value for e.g. betaabs.  Maybe need to remove these sections from the data completely.        
        out = out.clip(lower=-1, upper=1)

        print(f"ClipAbsOutliers {X.shape} to {out.shape}, first index {out.index[0]}")
        return out

class ScaleColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns: [str]):
        self.columns = columns
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        out = X.copy()
        cols = [col for col in self.columns if col in out.columns]
        out[cols] = self.scaler.transform(X[cols])
        print(f"ScaleColumns {X.shape} to {out.shape}, first index {out.index[0]}")
        return out


import numpy as np

class CleanTargetCol(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, max_value=1e6):
        self.target_col = target_col
        self.max_value = max_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        y = X[self.target_col]

        # Check for NaN values
        nan_indices = np.where(np.isnan(y))[0]
        #print(f"NaN indices: {nan_indices} ({len(nan_indices)} total of {len(y)})")

        # Check for infinity values
        infinity_indices = np.where(np.isinf(y))[0]
        #print(f"Infinity indices: {infinity_indices}")

        # Check for excessively large values
        too_large_indices = np.where(y > self.max_value)[0]
        #print(f"Too large indices: {too_large_indices}")

        # Remove rows with NaN, infinity, or excessively large values
        valid_indices = np.setdiff1d(np.arange(len(y)), np.concatenate((nan_indices, infinity_indices, too_large_indices)))
        X_cleaned = X.iloc[valid_indices]

        print(f"CleanTargetCol {X.shape} to {X_cleaned.shape}")

        return X_cleaned


class DropCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols: [str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.drop(columns=self.cols)

        print(f"DropCols {X.shape} to {out.shape}, first index {out.index[0]}")
        return out


