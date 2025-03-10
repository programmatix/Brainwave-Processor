from dataclasses import dataclass

from models.util import importances
from models.util.evaluation import evaluate_regression_model_quick, evaluate_classification_model_quick
import shap
import pandas as pd
import numpy as np

@dataclass
class ModelAndSettings:
    model: object
    name: str
    # For RFE, need to keep the data we trained on
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    settings: dict
    is_classifier: bool = False


def train_rfe(create_model: callable, tl, name: str, X_train, y_train, X_val, y_val, do_rfe: bool = True):
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_val = X_val.copy()

    n_features = X_train.shape[1]
    models = []

    while n_features > 2:
        type = tl['type']

        print(f"Training {type} model for {name} with {X_train.shape}", end='')

        model, extra, params = create_model(tl)
        if 'requiresNumpy' in params and params['requiresNumpy'] == True:
            model.fit(X_train.values, y_train.values, feature_names=X_train.columns)
        else:
            if tl.get('supportsFeatureWeights', False):
                # Bit hacky that these are hard-coded here
                feature_weights = pd.DataFrame(1.0, index=X_train.columns, columns=['weight'])
                # Prefer mean to percentile
                feature_weights.loc[feature_weights.index.str.contains("percentile"), 'weight'] = 0.5
                # Probably too harsh
                feature_weights.loc[feature_weights.index.str.contains("MiddayTo"), 'weight'] = 0.5
                feature_weights.loc[feature_weights.index.str.contains("MiddayOf"), 'weight'] = 0.5
                # Standardising on Main
                feature_weights.loc[feature_weights.index.str.contains("-M1:"), 'weight'] = 0.2
                # I just don't trust stuff that spans the chaos of Wake and insomnia
                feature_weights.loc[feature_weights.index.str.contains("AnyStage"), 'weight'] = 0.2
                feature_weights.loc[feature_weights.index.str.contains(":W:"), 'weight'] = 0.5
                # Short
                feature_weights.loc[feature_weights.index.str.contains(":N1:"), 'weight'] = 0.5
                model.fit(X_train, y_train, feature_weights=feature_weights)
            else:
                model.fit(X_train, y_train)

        try:
            is_classifier = 'isClassifier' in params and params['isClassifier'] == True
            if is_classifier:
                result = evaluate_classification_model_quick(model, X_train, y_train, X_val, y_val)

                print(f"... F1 train: {result['F1Train']} F1 val: {result['F1Val']}")
            else:
                result = evaluate_regression_model_quick(model, X_train, y_train, X_val, y_val)

                print(f"... RMSE train: {result['RMSE_Train']} RMSE val: {result['RMSE_Val']}")
        except Exception as e:
            print(f"Error evaluating model: {e}")

        # Store the model
        aux = {
            'n_features': n_features,
        }
        aux.update(extra)
        aux.update(params)
        models.append(ModelAndSettings(model, name, X_train.copy(), y_train.copy(), X_val, y_val, aux))

        if not do_rfe:
            break

        if n_features > 16:
            n_features = max(2, n_features // 2)
        else:
            n_features = n_features - 4

        # Update the training data with the selected features
        # SHAP gives us slower but better feature importances
        useSHAPForRFE = tl.get('useSHAPForRFE', True)
        if useSHAPForRFE:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_train)
            mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
            top_features_indices = np.argsort(mean_abs_shap_values)[-n_features:][::-1]
            top_features = [shap_values.feature_names[i] for i in top_features_indices]
        else:
            feature_importances = importances.get_importances(model, X_train.columns)
            top_features = feature_importances['Feature'].head(n_features)

        X_train = X_train[top_features]
        # X_train_selected = X_train[:, rfe.support_]
        X_val = X_val[top_features]

    return models


from models.util.evaluation import evaluate_regression_model_quick
from dataclasses import dataclass
import xgboost as xgb
from sklearn.feature_selection import RFE
import models.util.importances as importances
from importlib import reload

reload(importances)
import memory
from imodels import get_clean_dataset, HSTreeClassifierCV, HSTreeRegressor
from imodels import FIGSClassifier, FIGSRegressor
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from interpret import show
from sklearn.linear_model import LinearRegression


def get_model(tl):
    type = tl["type"]

    if type == "HSTreeClassifierCV":
        model = HSTreeClassifierCV()

        return (model, {
            "source": type,
        }, {
                    "requiresNumpy": True,
                    "isClassifier": True
                })

    elif type == "FIGSClassifier":
        model = FIGSClassifier()

        return (model, {
            "source": type,
        }, {
                    "requiresNumpy": False,
                    "isClassifier": True
                })

    elif type == "FIGSRegressor":
        model = FIGSRegressor()

        return (model, {
            "source": type,
        }, {
                    "requiresNumpy": False,
                    "isClassifier": False
                })

    elif type == "ExplainableBoostingClassifier":
        model = ExplainableBoostingClassifier()

        return (model, {
            "source": type,
        }, {
                    "requiresNumpy": False,
                    "isClassifier": True
                })

    elif type == "ExplainableBoostingRegressor":
        model = ExplainableBoostingRegressor()

        return (model, {
            "source": type,
        }, {
                    "requiresNumpy": False,
                    "isClassifier": False
                })

    elif type == "XGBoostClassifier":
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
        )
        return (model, {
            "source": type
        }, {
                    "isClassifier": True
                })

    elif type == "XGBoostRegressor":
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
        )
        return (model, {
            "source": type
        }, {
                    "isClassifier": False
                })

    elif type == "XGBoostClassifier+HSTreeClassifierCV":
        base_model = get_model({"type": "XGBoostClassifier"})[0]
        model = HSTreeClassifierCV(estimator_ = base_model)

        return (model, {
            "source": type,
        }, {
            "requiresNumpy": True,
            "isClassifier": True
        })

    elif type == "XGBoostRegressor+HSTreeRegressor":
        base_model = get_model({"type": "XGBoostRegressor"})[0]
        model = HSTreeRegressor(estimator_ = base_model)

        return (model, {
            "source": type,
        }, {
            "requiresNumpy": True,
                    "isClassifier": False
                })

    elif type == "LinearRegression":
        model = LinearRegression()

        return (model, {
            "source": type,
        }, {
                    "requiresNumpy": False,
                    "isClassifier": False
                })

    else:
        raise Exception(f"Unknown model type: {type}")
