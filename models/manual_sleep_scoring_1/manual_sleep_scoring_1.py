from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
import pandas as pd

def pipeline(input):
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.columns_to_keep = [col for col in X.columns if "_eeg_" in col or col == 'ManualStage']
            return self

        def transform(self, X):
            return X[self.columns_to_keep]

    class OneHotEncoder(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.encoder = SklearnOneHotEncoder(sparse_output=False)

        def fit(self, X, y=None):
            self.encoder.fit(X[['ManualStage']])
            return self

        def transform(self, X):
            encoded = self.encoder.transform(X[['ManualStage']])
            encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(['ManualStage']), index=X.index)
            return X.drop(columns=['ManualStage']).join(encoded_df)

    class DropUnscoredRows(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            filtered = X[X['ManualStage_nan'] != 1.0]
            return filtered.drop(columns=['ManualStage_nan'])

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

    pipeline = Pipeline([
        ('AddTagCols', AddTagCols()),
        ('selector', DataFrameSelector()),
        ('bool_converter', BooleanConverter()),
        ('one_hot_encoder', OneHotEncoder()),
        ('drop_unscored', DropUnscoredRows()),
        ('index_preserver', IndexPreserver())
    ])

    # Example usage
    prepared_df = pipeline.fit_transform(input)
    prepared_df = pipeline.named_steps['index_preserver'].inverse_transform(prepared_df)

    target_cols = [col for col in prepared_df.columns if 'ManualStage' in col]
    X = prepared_df.drop(columns=target_cols)
    y = prepared_df[target_cols]

    return prepared_df, X, y