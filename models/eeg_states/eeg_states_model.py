from dataclasses import dataclass
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
import pandas as pd
import json

class RowsWithTargetCol(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X[~X[self.target_col].isna()]
        return X

class TiredVsWiredTargetColMapper(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        mapping = {
            'alert': 0.0,
            'tired': 1.0
        }
        X[self.target_col] = X[self.target_col].map(mapping)
        return X

class SettlingEventV4TargetColMapper(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        mapping = {
            'alert': 0.0,
            'tired': 1.0
        }
        X[self.target_col] = X[self.target_col].map(mapping)
        return X

day_energy_mapping = {
    'near worst': 0.0,
    'struggling': 0.1,
    'tired': 0.3,
    'standard tired': 0.4,
    'sub-optimal': 0.4,
    'okish': 0.7,
    'lockable': 0.9,
    'great': 1.0
}

class DayEnergyTargetColMapper(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.target_col] = X[self.target_col].map(day_energy_mapping)
        return X

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, realtime):
        self.target_col = target_col
        self.realtime = realtime

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        columns_to_keep = [col for col in X.columns if ("Main_" in col and col.endswith("_s") and "perment" not in col) or col == self.target_col]
        if self.realtime:
            columns_to_keep = [col for col in columns_to_keep if "_c7" not in col and "_p2" not in col]
        print(columns_to_keep)
        return X[columns_to_keep]



@dataclass
class ModelAndData:
    name: str
    target_col: str
    prepared_df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    model: object = None
    X_train: pd.DataFrame = None
    y_train: pd.Series = None
    X_val: pd.DataFrame = None
    y_val: pd.Series = None

def generic_state_model_pipeline(name: str, input, target_col: str, realtime: bool) -> ModelAndData:
    pipeline = Pipeline([
        ('col_selector', DataFrameSelector(target_col, realtime)),
        ('row_selector', RowsWithTargetCol(target_col))
    ])

    prepared_df = pipeline.fit_transform(input)

    X = prepared_df.drop(columns=[target_col])
    y = prepared_df[target_col]

    return ModelAndData(name, target_col, prepared_df, X, y)


def predict_only_generic_state_model_pipeline(name: str, input, realtime: bool) -> ModelAndData:
    pipeline = Pipeline([
        ('col_selector', DataFrameSelector(None, realtime)),
    ])

    prepared_df = pipeline.fit_transform(input)

    return ModelAndData(name, None, prepared_df, prepared_df, None)




def tired_vs_wired_model_pipeline(name: str, input, target_col: str, realtime: bool) -> ModelAndData:
    pipeline = Pipeline([
        ('col_selector', DataFrameSelector(target_col, realtime)),
        ('row_selector', RowsWithTargetCol(target_col))
        # ('target_col_mapper', TiredVsWiredTargetColMapper(target_col))
    ])

    prepared_df = pipeline.fit_transform(input)

    X = prepared_df.drop(columns=[target_col])
    y = prepared_df[target_col]

    return ModelAndData(name, target_col, prepared_df, X, y)


def predict_only_tired_vs_wired_model_pipeline(name: str, input, realtime: bool) -> ModelAndData:
    pipeline = Pipeline([
        ('col_selector', DataFrameSelector(None, realtime)),
    ])

    prepared_df = pipeline.fit_transform(input)

    return ModelAndData(name, None, prepared_df, prepared_df, None)


def day_energy_model_pipeline(name: str, input, target_col: str, realtime: bool) -> ModelAndData:
    pipeline = Pipeline([
        ('col_selector', DataFrameSelector(target_col, realtime)),
        ('row_selector', RowsWithTargetCol(target_col))
        # ('target_col_mapper', DayEnergyTargetColMapper(target_col))
    ])

    prepared_df = pipeline.fit_transform(input)

    X = prepared_df.drop(columns=[target_col])
    y = prepared_df[target_col]

    return ModelAndData(name, target_col, prepared_df, X, y)


def predict_only_day_energy_model_pipeline(name: str, input, realtime: bool) -> ModelAndData:
    pipeline = Pipeline([
        ('col_selector', DataFrameSelector(None, realtime)),
    ])

    prepared_df = pipeline.fit_transform(input)

    return ModelAndData(name, None, prepared_df, prepared_df, None)




