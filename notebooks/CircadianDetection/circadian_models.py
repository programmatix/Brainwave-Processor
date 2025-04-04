from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
import pandas as pd
import json
import xgboost as xgb


import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

from models.util.epoch_level_features import EpochLevelFeaturesHandler
from models.util.pipeline import CleanTargetCol


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
        # reasons = X[bad_rows].apply(lambda row: row[row.isin([np.inf, -np.inf])].index[0], axis=1).tolist()
        #
        # print(f"Dropped row indexes: {dropped_indexes}")
        # print(f"Kept row indexes: {kept_indexes}")
        # print(f"Reasons for dropping: {reasons}")

        out = X[~bad_rows].select_dtypes(exclude=['object', 'datetime64[ns]'])

        print(f"DropBadRows: before {len(X)} rows after {len(out)} rows")
        return out

class RequireNonEmptyRows(BaseEstimator, TransformerMixin):
    def __init__(self, rows_must_be_non_empty: list[str]):
        self.rows_must_be_non_empty = rows_must_be_non_empty

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Remove rows with NaN, NaT, or other missing values in the specified columns
        out = X.dropna(subset=self.rows_must_be_non_empty)
        print(f"RequireNonEmptyRows: before {len(X)} rows after {len(out)} rows")
        return out

@dataclass
class ModelAndData:
    name: str
    target_col: str
    is_classifier: bool
    prepared_df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    model: object = None
    X_train: pd.DataFrame = None
    y_train: pd.Series = None
    X_val: pd.DataFrame = None
    y_val: pd.Series = None


def create_and_add(predict_mode: bool,
                   models_and_data: [ModelAndData],
                   is_classifier: bool,
                   name: str,
                   target_col: str,
                   sources: list[str],
                   rows_must_be_non_empty: list[str],
                   input,
                   condition: callable,
                   ):
    p = []

    if not predict_mode:
        p.extend([
        ])

    p.extend([
        ('clean_target', CleanTargetCol(target_col)),
        ('drop_empty', RequireNonEmptyRows(rows_must_be_non_empty)),
        ('drop_bad', DropBadRows()),
    ])

    pipeline = Pipeline(p)

    prepared_df = pipeline.fit_transform(input)

    if predict_mode:
        X = prepared_df
        y = None
    else:
        X = prepared_df.drop(columns=[target_col])
        y = prepared_df[target_col]

    md = ModelAndData(name, target_col, is_classifier, prepared_df, X, y)
    models_and_data.append(md)

def create_and_add_all(merged, predict_mode: bool, target_col: str):
    models_and_data: list[ModelAndData] = []

    create_and_add(predict_mode, models_and_data,  False, target_col, target_col, ["all"], [], merged, lambda X:X)

    return models_and_data



