from sklearn.base import TransformerMixin, BaseEstimator


def allowed_eeg_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        return 'Main_eeg' in feature

    def blacklist(feature: str) -> bool:
        # perment has been removed and perm is anyway almost 1:1 with another feature
        return 'perm' in feature \
            or 'svdent' in feature

    return whitelist(feature) and not blacklist(feature)

def allowed_eeg_no_future_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        return True

    def blacklist(feature: str) -> bool:
        # Remove the central rolling average as it includes future knowledge
        return '_c7min_' in feature

    return allowed_eeg_feature(feature) and whitelist(feature) and not blacklist(feature)

# Assumes has already gone through all_allowed_eeg_features
def allowed_best_eeg_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        return True

    def blacklist(feature: str) -> bool:
        # Don't understand svdent!
        # Unconvinced by kurt and skew in context of EEG
        # Not using the derived ones as they're confusing to think about and hopefully the model can learn them
        return 'svdent' in feature \
            or 'kurt' in feature \
            or 'skew' in feature \
            or '_ds_' in feature \
            or '_dt_' in feature \
            or '_db_' in feature \
            or '_at_' in feature \

    return whitelist(feature) and not blacklist(feature)

def allowed_physical_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        # Not position just yet, needs more work
        return feature.startswith("HR") or feature.startswith("Hrv") \
            or feature == "O2" or feature.startswith("Movement") \
            or feature.startswith("Temp")

    def blacklist(feature: str) -> bool:
        return False

    return whitelist(feature) and not blacklist(feature)

# Anything that gives some clue as to current time
def allowed_time_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        # Not position just yet, needs more work
        return feature.startswith("minsSince") or feature.startswith("minsUntil") \
            or feature.lower().startswith("epoch")

    def blacklist(feature: str) -> bool:
        return False

    return whitelist(feature) and not blacklist(feature)

def allowed_sleep_stages(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        return feature.startswith("SS")

    def blacklist(feature: str) -> bool:
        return False

    return whitelist(feature) and not blacklist(feature)

def allowed_sleep_stages_no_times(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        return True

    def blacklist(feature: str) -> bool:
        return "MinsUntil" in feature or "MinsSince" in feature

    return allowed_sleep_stages(feature) and whitelist(feature) and not blacklist(feature)

def allowed_feature(sources: [str], feature: str) -> bool:
    if 'literally_all' in sources:
        return True
    if 'eeg' in sources:
        if allowed_eeg_feature(feature):
            return True
    if 'eeg_no_future' in sources:
        if allowed_eeg_no_future_feature(feature):
            return True
    if 'best_eeg' in sources:
        if allowed_eeg_feature(feature) and allowed_best_eeg_feature(feature):
            return True
    if 'physical' in sources:
        if allowed_physical_feature(feature):
            return True
    if 'times' in sources:
        if allowed_time_feature(feature):
            return True
    if 'sleep_stage' in sources:
        if allowed_sleep_stages(feature):
            return True
    if 'sleep_stage_no_times' in sources:
        if allowed_sleep_stages_no_times(feature):
            return True
    return False

def allowed_features(sources: [str], features: [str]) -> [str]:
    return [feature for feature in features if allowed_feature(sources, feature)]


class FeaturesHandler(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, sources: [str]):
        self.target_col = target_col
        self.sources = sources

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = X.columns
        allowed = allowed_features(self.sources, features)

        if self.target_col not in allowed and self.target_col in X.columns:
            allowed.append(self.target_col)

        # Sorting helps ensure model will line up with data later
        X = X.reindex(sorted(X.columns), axis=1)
        allowed = sorted(allowed)

        out = X[allowed]
        print(f"FeaturesHandler {X.shape} to {X.shape} Had {len(features)} features, after filtering have features", list(allowed))
        return out


import numpy as np

class CleanTargetCol(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, max_value=1e6):
        self.target_col = target_col
        self.max_value = max_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        y = X[self.target_col]

        # Check for NaN values
        nan_indices = np.where(np.isnan(y))[0]
        print(f"NaN indices: {nan_indices} ({len(nan_indices)} total of {len(y)})")

        # Check for infinity values
        infinity_indices = np.where(np.isinf(y))[0]
        print(f"Infinity indices: {infinity_indices}")

        # Check for excessively large values
        too_large_indices = np.where(y > self.max_value)[0]
        print(f"Too large indices: {too_large_indices}")

        # Remove rows with NaN, infinity, or excessively large values
        valid_indices = np.setdiff1d(np.arange(len(y)), np.concatenate((nan_indices, infinity_indices, too_large_indices)))
        X_cleaned = X.iloc[valid_indices]

        print(f"CleanTargetCol {X.shape} to {X_cleaned.shape}")

        return X_cleaned
