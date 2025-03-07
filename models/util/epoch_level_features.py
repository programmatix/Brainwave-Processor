from sklearn.base import TransformerMixin, BaseEstimator


def allowed_eeg_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        # Shouldn't matter much if use _s or not, but makes sense to only choose one set, and the epoch viewer app is
        # currently setup to show non
        return 'Main_eeg' in feature and not feature.endswith("_s")

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

# When I'm looing for a clean easy to understand model
def allowed_eeg_easiest(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        return True

    def blacklist(feature: str) -> bool:
        return ('_c7min_' in feature
            or '_p2min' in feature
            or 'hcomp' in feature
            or 'hmob' in feature
            or 'std' in feature
                )

    return allowed_eeg_feature(feature) and allowed_best_eeg_feature(feature) and whitelist(feature) and not blacklist(feature)

# Assumes has already gone through all_allowed_eeg_features
def allowed_best_eeg_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        return True

    def blacklist(feature: str) -> bool:
        return (
            # Don't understand svdent!
                'svdent' in feature
                # Unconvinced by kurt and skew in context of EEG
            or 'kurt' in feature
            or 'skew' in feature
            # Higuchi is almost identical, and clearer
            or 'petrosian' in feature
        # Not using the derived ones as they're confusing to think about and hopefully the model can learn them (not convinced on that, restoring)

        # or '_ds_' in feature
            # or '_dt_' in feature
            # or '_db_' in feature
            # or '_at_' in feature
        )

    return whitelist(feature) and not blacklist(feature)

def allowed_physical_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        # Not position just yet, needs more work
        return feature.startswith("HR") or feature.startswith("Hrv") \
            or feature == "O2" or feature.startswith("Movement") \
            or feature.startswith("Temp")
        # todo add glucose and BP

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

def allowed_microwakings(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        return "Microwakings" in feature

    def blacklist(feature: str) -> bool:
        return "MicrowakingsDbg" in feature

    return whitelist(feature) and not blacklist(feature)

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
    if 'eeg_easiest' in sources:
        if allowed_eeg_easiest(feature):
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
    if 'microwakings' in sources:
        if allowed_microwakings(feature):
            return True
    return False

def allowed_features(sources: [str], features: [str]) -> [str]:
    return [feature for feature in features if allowed_feature(sources, feature)]



class EpochLevelFeaturesHandler(BaseEstimator, TransformerMixin):
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

        out = out.replace({True: 1, False: 0})

        print(f"EpochLevelFeaturesHandler {X.shape} to {out.shape}, first index {out.index[0]}")
        return out


