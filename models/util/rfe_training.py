from dataclasses import dataclass

from models.util import importances
from models.util.evaluation import evaluate_regression_model_quick
import pandas as pd

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

def train_rfe(create_model: callable, name: str, X_train, y_train, X_val, y_val, do_rfe: bool = True):
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_val = X_val.copy()

    n_features = X_train.shape[1]
    models = []

    while n_features > 2:
        print(f"Training RFE model for {name} with {n_features} features", end='')

        model, extra = create_model()
        model.fit(X_train, y_train, verbose=True)

        result = evaluate_regression_model_quick(model, X_train, y_train, X_val, y_val)

        print(f"... RMSE train: {result['RMSE_Train']} RMSE val: {result['RMSE_Val']}")

        # Store the model
        aux = {
            'n_features': n_features,
        }
        aux.update(extra)
        models.append(ModelAndSettings(model, name, X_train.copy(), y_train.copy(), X_val, y_val, aux))

        if not do_rfe:
            break

        if n_features > 16:
            n_features = max(2, n_features // 2)
        else:
            n_features = n_features - 4

        # Update the training data with the selected features
        feature_importances = importances.get_importances(model, X_train.columns)
        # Select the top n_features features
        top_features = feature_importances['Feature'].head(n_features)
        X_train = X_train[top_features]
        # X_train_selected = X_train[:, rfe.support_]
        X_val = X_val[top_features]

    return models