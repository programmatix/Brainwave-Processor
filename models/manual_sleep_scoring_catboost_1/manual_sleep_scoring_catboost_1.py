from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
import pandas as pd
import json

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_to_keep = [col for col in X.columns if "_eeg_" in col or 'ManualStage' in col]
        return self

    def transform(self, X):
        return X[self.columns_to_keep]

class DropUnscoredRows(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if 'ManualStage_nan' in X.columns:
            filtered = X[X['ManualStage_nan'] != 1.0]
            return filtered.drop(columns=['ManualStage_nan'])
        return X

# Catboost doesn't need one-hot, but maybe this fits better as these our out output labels and we also want tags
class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = SklearnOneHotEncoder(sparse_output=False)

    def fit(self, X, y=None):
        self.encoder.fit(X[['ManualStage']])
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X[['ManualStage']])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(['ManualStage']), index=X.index)
        out = X.drop(columns=['ManualStage']).join(encoded_df)
        out['ManualStage_AnyDeep'] = ((out['ManualStage_Deep'] == 1) | (out['ManualStage_Ambiguous Deep'] == 1)).astype(int)
        return out


class AddTagCols(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['ManualStage_has_microwaking_start'] = X['tags'].apply(lambda x: isinstance(x, list) and ('Microwaking' in x or 'MicrowakingComplete' in x))
        X['ManualStage_has_microwaking_end'] = X['tags'].apply(lambda x: isinstance(x, list) and ('MicrowakingEnd' in x or 'MicrowakingComplete' in x))
        # X['ManualStage_possible_non_wake_disturbance'] = X['tags'].apply(lambda x: 'Possible non-wake disturbance' in x)
        # X['ManualStage_strong_deep'] = X['tags'].apply(lambda x: 'Strong deep' in x)
        # X['ManualStage_weak_deep'] = X['tags'].apply(lambda x: 'Weak deep' in x)
        # X['ManualStage_unusual'] = X['tags'].apply(lambda x: 'Unusual' in x)
        return X

class BooleanConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        boolean_columns = X.select_dtypes(include=['bool']).columns
        X[boolean_columns] = X[boolean_columns].astype(int)
        return X

class IndexPreserver(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.index = X.index
        return X

    def inverse_transform(self, X):
        X.index = self.index
        return X

def pipeline(input):

    pipeline = Pipeline([
        ('AddTagCols', AddTagCols()),
        ('selector', DataFrameSelector()),
        ('bool_converter', BooleanConverter()),
        ('one_hot_encoder', OneHotEncoder()),
        ('drop_unscored', DropUnscoredRows()),
        ('index_preserver', IndexPreserver())
    ])

    prepared_df = pipeline.fit_transform(input)
    prepared_df = pipeline.named_steps['index_preserver'].inverse_transform(prepared_df)

    target_cols = [col for col in prepared_df.columns if 'ManualStage' in col]
    X = prepared_df.drop(columns=target_cols)
    y = prepared_df[target_cols]

    return prepared_df, X, y


def pipeline_predictions(input):

    pipeline = Pipeline([
        ('selector', DataFrameSelector()),
        ('bool_converter', BooleanConverter()),
        ('index_preserver', IndexPreserver())
    ])

    prepared_df = pipeline.fit_transform(input)
    prepared_df = pipeline.named_steps['index_preserver'].inverse_transform(prepared_df)

    target_cols = [col for col in prepared_df.columns if 'ManualStage' in col]
    X = prepared_df.drop(columns=target_cols)

    return prepared_df, X


def load_scoring_file(scoring_file_path: str):
    with open(scoring_file_path, 'r') as file:
        data = json.load(file)
    df = pd.json_normalize(data, 'scorings', errors='ignore')
    df['tags'] = df['tags'].apply(lambda x: [tag['tag'] for tag in x] if isinstance(x, list) else [])
    df.rename(columns={'stage': 'ManualStage', 'epochIndex': 'epoch'}, inplace=True)
    df.set_index('epoch', inplace=True)
    return df


from catboost import CatBoostClassifier


def run_model(df):
    model = CatBoostClassifier()
    model.load_model("c:\\dev\\play\\BrainwaveModel\\sleep_scoring_catboost_model_20240903_184357.cbm")
    prepared_df, X = pipeline_predictions(df)

    to_save_df = df.copy()

    cols = ['ManualStage_has_microwaking_start',
            'ManualStage_has_microwaking_end',
            'ManualStage_Ambiguous Deep',
            'ManualStage_Deep',
            'ManualStage_Noise',
            'ManualStage_Non-Deep',
            'ManualStage_Unsure',
            'ManualStage_Wake',
            'ManualStage_AnyDeep']
    cols = [col.replace('ManualStage', 'Predictions') for col in cols]

    # May have predictions from previous runs that we want to overwrite
    for col in cols:
        if col in to_save_df.columns:
            to_save_df.drop(columns=col, inplace=True)

    predictions_raw = model.predict(X)
    predictions_df = pd.DataFrame(predictions_raw, columns=cols)
    joined = df.join(predictions_df)
    return joined