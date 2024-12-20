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

from models.util.features import FeaturesHandler


class AddAllTargetCols(BaseEstimator, TransformerMixin):
    def __init__(self, target_set):
        self.target_set = target_set

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for y in self.target_set:
            X[f'WillWakeWithin{y}Mins'] = X['minsUntilWake'].apply(lambda x: True if 0 < x <= y else False)
        return X

class DropAllNearTargetCols(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        keep = [col for col in X.columns if (self.target_col == col or ('WillWakeWithin' not in col and 'minsUntilWake' not in col))]
        return X[keep]

class UsefulFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, sources: list[str]):
        self.target_col = target_col
        self.sources = sources

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Just not really ready for primetime
        # not_ready_for_primetime = ["presence:", "settling:"]
        # useless = ["generatedAt", "hasYasa", "morningQuestionnaire"]
        # do_not_want_in_model = ["date:", "-M1", "Fpz", "perment"]
        # # Will want to use adjusted instead
        # duplicates = ["night:fitbit:source:", "night:yasa:source:"]
        # # Need to debug
        # not_present_on_all_rows_for_some_reason = ["SleepHour", "Stability:Aggregated", "TiredVsWired", "BeforeSleep", "ReadyToSleep"]
        # remove_list = not_ready_for_primetime + useless + do_not_want_in_model + duplicates + not_present_on_all_rows_for_some_reason

        # remove_if_includes_list = ["energy"]

        remove_list = ["-M1_eeg"]

        useful_features = [col for col in X.columns \
                           if (self.target_col in col) \
                           # Part of target list
                           or not '-M1_eeg' in col \
                           # or not any(rem in col for rem in remove_list)
                           ]

        if "eeg" in self.sources:
            useful_features = [col for col in useful_features if ('yasa' in col.lower()) or self.target_col in col]

        return X[useful_features].select_dtypes(exclude=['object', 'datetime64[ns]', 'datetime64[ns, Europe/London]'])

class NotAllowedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, sources: list[str]):
        self.target_col = target_col
        self.sources = sources

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        useful_features = X.columns

        if "times" in self.sources:
            useful_features = [col for col in useful_features if 'minsSince' not in col and 'epoch' not in col.lower()]

        return X[useful_features]

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


def create_and_add(predict_mode: bool, models_and_data: [ModelAndData], is_classifier: bool, name: str, target_set: [int], target_col: str, sources: list[str], input):
    #name = f"target:{target_col} allowed_sources:{allowed_sources} not_allowed_sources:{not_allowed_sources}"

    p = []

    if not predict_mode:
        p.extend([
            ('all_target_features', AddAllTargetCols(target_set)),
            ('rows', DropAllNearTargetCols(target_col))
        ])

    p.extend([
        ('features_generic', FeaturesHandler(target_col, sources)),
        # ('features', UsefulFeatures(target_col, allowed_sources)),
        # ('not_allowed_features', NotAllowedFeatures(target_col, not_allowed_sources)),
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


DEFAULT_TARGET_SET = [10, 30, 60, 90, 120, 150, 240]
# DEFAULT_TARGET_SET = [10]

def create_and_add_all(merged, predict_mode: bool, target_set = DEFAULT_TARGET_SET):
    models_and_data: list[ModelAndData] = []
    create_and_add(predict_mode, models_and_data,  False, f"minsUntilWake", target_set, "minsUntilWake", ["best_eeg", "physical"], merged)

    # for y in target_set:
    #     #create_and_add(models_and_data, f"WillWakeWithin{y}Mins", ["all"], [], merged)
    #     # create_and_add(models_and_data,  f"PredictFinalWakeWithinNext{y}MinsLiterallyAll", target_set, f"WillWakeWithin{y}Mins", ["literally_all"], merged)
    #     # create_and_add(models_and_data,  f"PredictFinalWakeWithinNext{y}MinsAll", target_set, f"WillWakeWithin{y}Mins", ["eeg", "physical"], merged)
    #     create_and_add(models_and_data,  True, f"PredictFinalWakeWithinNext{y}Mins", target_set, f"WillWakeWithin{y}Mins", ["best_eeg", "physical"], merged)
    #     # create_and_add(models_and_data,  f"PredictFinalWakeWithinNext{y}MinsEEGOnly", target_set, f"WillWakeWithin{y}Mins", ["best_eeg"], merged)

    return models_and_data


def load_model(filename):
    model = xgb.Booster()
    model.load_model(filename)
    return model


