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

class TargetColMapper(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.target_col] = X[self.target_col].map({'tired': 1, 'wired': 0})
        return X

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col

    def fit(self, X, y=None):
        self.columns_to_keep = [col for col in X.columns if "Main_" in col or col == self.target_col]
        return self

    def transform(self, X):
        return X[self.columns_to_keep]



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

def model_pipeline(name: str, input, target_col: str) -> ModelAndData:
    pipeline = Pipeline([
        ('col_selector', DataFrameSelector(target_col)),
        ('row_selector', RowsWithTargetCol(target_col)),
        ('target_col_mapper', TargetColMapper(target_col))
    ])

    prepared_df = pipeline.fit_transform(input)

    X = prepared_df.drop(columns=[target_col])
    y = prepared_df[target_col]

    return ModelAndData(name, target_col, prepared_df, X, y)


def predict_only_model_pipeline(name: str, input) -> ModelAndData:
    pipeline = Pipeline([
        ('col_selector', DataFrameSelector(None)),
    ])

    prepared_df = pipeline.fit_transform(input)

    return ModelAndData(name, None, prepared_df, prepared_df, None)



