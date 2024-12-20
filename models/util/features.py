from sklearn.base import TransformerMixin, BaseEstimator


def allowed_eeg_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        return 'Main_eeg' in feature

    def blacklist(feature: str) -> bool:
        # perment has been removed and perm is anyway almost 1:1 with another feature
        return 'perm' in feature \
            or 'svdent' in feature

    return whitelist(feature) and not blacklist(feature)

# Assumes has already gone through all_allowed_eeg_features
def allowed_best_eeg_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        return True

    def blacklist(feature: str) -> bool:
        # Don't understand svdent!
        # Unconvinced by kurt and skew in context of EEG
        return 'svdent' in feature \
            or 'kurt' in feature \
            or 'skew' in feature

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

def allowed_feature(sources: [str], feature: str) -> bool:
    if 'literally_all' in sources:
        return True
    if 'eeg' in sources:
        if allowed_eeg_feature(feature):
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
        print(f"Had {len(features)} features, after filtering have features", list(allowed))

        return X[allowed]
