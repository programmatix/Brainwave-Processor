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

from models.util.epoch_level_features import EpochLevelFeaturesHandler
from models.util.pipeline import CleanTargetCol, Condition, RequireNonEmptyRows, DropBadRows


@dataclass
class ModelAndData:
    name: str
    target_col: str
    is_classifier: bool
    prepared_df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    removed_nan: bool
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
                   removed_nan: bool,
                    condition: callable):
    #name = f"target:{target_col} allowed_sources:{allowed_sources} not_allowed_sources:{not_allowed_sources}"

    p = []

    if not predict_mode:
        p.extend([
            # ('all_target_features', AddAllTargetCols(target_set)),
            # ('rows', DropAllNearTargetCols(target_col))
        ])

    p.extend([
        ('clean_target', CleanTargetCol(target_col)),
        ('condition', Condition(condition)),
        ('features_generic', EpochLevelFeaturesHandler(target_col, sources)),
        ('drop_empty', RequireNonEmptyRows(rows_must_be_non_empty)),
        ('drop_bad', DropBadRows()),
    ])

    if removed_nan:
        p.append(('drop_nan', RequireNonEmptyRows()))

    pipeline = Pipeline(p)

    prepared_df = pipeline.fit_transform(input)

    if predict_mode:
        X = prepared_df
        y = None
    else:
        X = prepared_df.drop(columns=[target_col])
        y = prepared_df[target_col]

    md = ModelAndData(name, target_col, is_classifier, prepared_df, X, y, removed_nan)
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


def create_and_add_all(merged, predict_mode: bool):
    models_and_data: list[ModelAndData] = []

    # create_and_add(predict_mode, models_and_data,  False, f"minsSinceSleepAllEeg1HrBeforeJustEeg", "minsSinceAsleep", ["eeg"], [], merged, True, lambda X: X[(X['minsSinceReadyToSleep'] <= 0) & (X['minsSinceReadyToSleep'] >= -60)])
    # create_and_add(predict_mode, models_and_data,  False, f"minsSinceReadyToSleepAllEeg1HrBeforeJustEeg", "minsSinceReadyToSleep", ["eeg"], [], merged, True, lambda X: X[(X['minsSinceReadyToSleep'] <= 0) & (X['minsSinceReadyToSleep'] >= -60)])
    # create_and_add(predict_mode, models_and_data,  False, f"minsSinceAsleepAllEeg1HrBefore", "minsSinceAsleep", ["eeg",  "physical"], [], merged, False, lambda X: X[(X['minsSinceAsleep'] <= 0) & (X['minsSinceAsleep'] >= -60)])
    create_and_add(predict_mode, models_and_data,  False, f"minsSinceReadyToSleep_AllEeg+Physical_1HrBefore", "minsSinceReadyToSleep", ["eeg",  "physical"], [], merged, False, lambda X: X[(X['minsSinceReadyToSleep'] <= 0) & (X['minsSinceReadyToSleep'] >= -60)])
    create_and_add(predict_mode, models_and_data,  False, f"minsSinceReadyToSleep_BestEeg+Physical_1HrBefore", "minsSinceReadyToSleep", ["best_eeg",  "physical"], [], merged, False, lambda X: X[(X['minsSinceReadyToSleep'] <= 0) & (X['minsSinceReadyToSleep'] >= -60)])
    create_and_add(predict_mode, models_and_data,  False, f"minsSinceReadyToSleep_BestEeg+Temp_1HrBefore", "minsSinceReadyToSleep", ["best_eeg",  "temp"], [], merged, False, lambda X: X[(X['minsSinceReadyToSleep'] <= 0) & (X['minsSinceReadyToSleep'] >= -60)])
    create_and_add(predict_mode, models_and_data,  False, f"minsSinceReadyToSleep_BestEeg_1HrBefore", "minsSinceReadyToSleep", ["best_eeg"], [], merged, False, lambda X: X[(X['minsSinceReadyToSleep'] <= 0) & (X['minsSinceReadyToSleep'] >= -60)])
    #create_and_add(predict_mode, models_and_data,  False, f"minsSinceReadyToSleep_Eeg+Physical", "minsSinceReadyToSleep", ["eeg",  "physical"], [], merged, False, lambda X: X[X['minsSinceReadyToSleep'] <= 0])
    #create_and_add(predict_mode, models_and_data,  False, f"minsSinceReadyToSleep", "minsSinceReadyToSleep", ["best_eeg", "physical"], [], merged, False, lambda X: X[X['minsSinceReadyToSleep'] <= 0])

    return models_and_data


def load_model(filename):
    model = xgb.Booster()
    model.load_model(filename)
    return model


