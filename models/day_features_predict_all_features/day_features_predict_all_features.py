from memory import garbage_collect
from dataclasses import dataclass
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
import pandas as pd
import json
import xgboost as xgb


import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

from models.util.pipeline import CleanTargetCol
from models.util.day_data_level_features import DayDataFeaturesHandler



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

        print(f"DropBadRows {X.shape} to {X.shape}")
        # print(f"DropBadRows: before {len(X)} rows after {len(out)} rows")
        return out

class RequireNonEmptyRows(BaseEstimator, TransformerMixin):
    def __init__(self, rows_must_be_non_empty: list[str]):
        self.rows_must_be_non_empty = rows_must_be_non_empty

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Remove rows with NaN, NaT, or other missing values in the specified columns
        out = X.dropna(subset=self.rows_must_be_non_empty)
        # print(f"RequireNonEmptyRows: before {len(X)} rows after {len(out)} rows")
        print(f"RequireNonEmptyRows {X.shape} to {X.shape}")
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
        print(f"Condition {X.shape} to {X.shape}")
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
                   # target_set: [int],
                   target_col: str,
                   sources: list[str],
                   rows_must_be_non_empty: list[str],
                   input,
                   condition: callable):
    #name = f"target:{target_col} allowed_sources:{allowed_sources} not_allowed_sources:{not_allowed_sources}"

    p = []

    if not predict_mode:
        p.extend([
            # ('all_target_features', AddAllTargetCols(target_set)),
            # ('rows', DropAllNearTargetCols(target_col))
            ('clean_target', CleanTargetCol(target_col)),
        ])

    p.extend([
        ('condition', Condition(condition)),
        ('features_generic', DayDataFeaturesHandler(target_col, sources, [])),
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

def run_all(merged):
    models_and_data = create_and_add_all(merged)
    all_models_filenames = [
        "models/PredictFinalWakeWithinNext10Mins_xgboost_model.cbm",
        "models/PredictFinalWakeWithinNext10MinsEEGOnly_xgboost_model.cbm"
    ]
    predictions_df = pd.DataFrame(index=merged.index)

    for md, model_filename in zip(models_and_data, all_models_filenames):
        model = load_model(model_filename)
        dmatrix = xgb.DMatrix(md.X)
        predictions = model.predict(dmatrix)
        predictions_df[md.name] = predictions

    return predictions_df


def create_and_add_all(day_data, target_feature: str, predict_mode: bool, remove_cols: list[str]):
    models_and_data: list[ModelAndData] = []

    create_and_add(predict_mode, models_and_data,  False, target_feature, target_feature, ["literally_all"], [], day_data, lambda X: X.drop(columns=remove_cols, errors='ignore'))

    return models_and_data


def load_model(filename):
    model = xgb.Booster()
    model.load_model(filename)
    return model


