from sklearn.base import TransformerMixin, BaseEstimator


# Anything that leaks info about what day/month this is.
# If the model can just learn that I took Concerta in August, it's not useful.
def allowed_days(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        # These effectively reveal what day/month it is
        return (feature.startswith("sunrise:")
            or feature.startswith("date:")
                # sns.scatterplot(x=df['weather:pressureAfternoon'], y=df['date:month'], alpha=0.2)
                or feature == "weather:pressureAfternoon"
        )

    def blacklist(feature: str) -> bool:
        return False

    return whitelist(feature) and not blacklist(feature)

def allowed_physical_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        # Not position just yet, needs more work
        return feature.startswith("hr:") or feature.startswith("hrv:") \
            or feature.startswith("o2:") or feature.startswith("movement:") \
            or feature.startswith("diastolic:") or feature.startswith("systolic:") \
            or feature.startswith("glucose:") \
            or feature.startswith("circadian:combined:") \
            or feature.startswith("coreBodyTemp:")

    def blacklist(feature: str) -> bool:
        return False

    return whitelist(feature) and not blacklist(feature)

def allowed_yasa_feature(feature: str) -> bool:
    def whitelist(feature: str) -> bool:
        return feature.startswith('night:') and 'yasa' in feature

    def blacklist(feature: str) -> bool:
        # Let's make this its own source. Not sure I trust it yet, plus if it does work it's just the same as time asleep?
        return ("finalWakeModel" in feature
            # Standardising on Main
            or "-M1:" in feature)

    return whitelist(feature) and not blacklist(feature)

def generic_blacklist(feature: str) -> bool:
    # Typo, replaced with night:yasaExtended:Stability:Aggregated:N3
    if feature == "night:yasaExtended:Stability:Aggregated:De ep":
        return True
    # Standardising on using non _s when training
    if feature.endswith("_s") or "_s:" in feature:
        return True
    # Just doesn't seem helpful to have the prediction error from any models?
    if "Prediction" in feature and "Error" in feature:
        return True
    # See https://dynalist.io/d/yGW81fUWLorPq2ydZX79NSG1
    if feature == "night:yasaExtended:Statistics:SE" or feature == "night:yasaExtended:Statistics:TIB" or feature.startswith("night:yasaExtended:Statistics:Lat_"):
        return True
    # Keeps getting picked up and I think it's basically a surrogate for time awake and insomniac
    if "discardedMicrowakingsCount" in feature:
        return True
    return False


def allowed_day_data_feature(sources: [str], feature: str) -> bool:
    if generic_blacklist(feature):
        return False

    if 'literally_all' in sources:
        return True
    if 'days' in sources:
        if allowed_days(feature):
            return True
    if 'physical' in sources:
        if allowed_physical_feature(feature):
            return True
    if 'yasa' in sources:
        if allowed_yasa_feature(feature):
            return True
    for s in sources:
        if s.endswith(':'):
            if feature.startswith(s):
                return True
    return False

def allowed_day_data_features(sources: [str], features: [str]) -> [str]:
    return [feature for feature in features if allowed_day_data_feature(sources, feature)]




def get_sleep_architecture_keys():
    return [
        {'key': 'night:yasa:adjusted:betweenAsleepAndWakeSecs', 'name': 'Asleep to Wake'},
        {'key': 'night:aggregated:wakeTimeSSM', 'name': 'Wake Time'}
    ]

def get_physical_keys():
    return [
        {'key': 'coreBodyTemp:AsleepToAwake:percentile10', 'name': 'Core Body Temp Low'},
        {'key': 'coreBodyTemp:AsleepToAwake:percentile90', 'name': 'Core Body Temp High'},
        {'key': 'coreBodyTemp:AsleepToAwake:mean', 'name': 'Core Body Temp Mean'},
        {'key': 'hr:AsleepToAwake:mean', 'name': 'HR'},
        {'key': 'hrv:AsleepToAwake:mean', 'name': 'HRV'},
        {'key': 'o2:AsleepToAwake:mean', 'name': 'O2'},
        {'key': 'o2:AsleepToAwake:percentile10', 'name': 'O2 Low'},
        {'key': 'o2:AsleepToAwake:percentile90', 'name': 'O2 High'},
        {'key': 'movement:AsleepToAwake:mean', 'name': 'Movement'},
        {'key': 'glucose:AsleepToAwake:mean', 'name': 'Glucose'},
        {'key': 'diastolic:AsleepToAwake:mean', 'name': 'Diastolic'},
        {'key': 'systolic:AsleepToAwake:mean', 'name': 'Systolic'}
    ]

def get_stage_keys(stage):
    alt_name1 = 'N1' if stage == 'N1' else 'N2' if stage == 'N2' else 'Deep' if stage == 'N3' else 'REM' if stage == 'R' else 'Sleep'
    alt_name2 = 'Rem' if stage == 'R' else alt_name1
    alt_name3 = 'Light' if stage == 'N2' else alt_name1
    alt_name4 = 'REM' if stage == 'R' else stage

    keys = [
        {'key': f'night:yasa:adjusted:{alt_name3.lower()}SleepSecs', 'name': 'Amount'} if stage != 'Sleep' else None,
        {'key': f'night:yasaExtended:Stability:Aggregated:{alt_name4}', 'name': 'Stability'},
        {'key': f'night:yasaExtended:{stage}:iqr:mean', 'name': 'Power'},
        {'key': f'night:yasaExtended:{stage}:auc:mean', 'name': 'AUC'},
        {'key': f'night:yasaExtended:{stage}:higuchi:mean', 'name': 'Complexity'},
        {'key': f'night:aggregated:microwakingsWithin{alt_name2}PerHour', 'name': 'Microwakings'} if stage in ['N3', 'R'] else None,
        {'key': f'night:yasaExtended:{stage}:sdelta:mean', 'name': 'Sdelta'},
        {'key': f'night:yasaExtended:{stage}:fdelta:mean', 'name': 'Fdelta'},
        {'key': f'night:yasaExtended:{stage}:theta:mean', 'name': 'Theta'},
        {'key': f'night:yasaExtended:{stage}:alpha:mean', 'name': 'Alpha'},
        {'key': f'night:yasaExtended:{stage}:sigma:mean', 'name': 'Sigma'},
        {'key': f'night:yasaExtended:{stage}:beta:mean', 'name': 'Beta'},
        {'key': f'night:yasaExtended:{stage}:sdeltaabs:mean', 'name': 'SdeltaAbs'},
        {'key': f'night:yasaExtended:{stage}:fdeltaabs:mean', 'name': 'FdeltaAbs'},
        {'key': f'night:yasaExtended:{stage}:thetaabs:mean', 'name': 'ThetaAbs'},
        {'key': f'night:yasaExtended:{stage}:alphaabs:mean', 'name': 'AlphaAbs'},
        {'key': f'night:yasaExtended:{stage}:sigmaabs:mean', 'name': 'SigmaAbs'},
        {'key': f'night:yasaExtended:{stage}:betaabs:mean', 'name': 'BetaAbs'},
        {'key': f'night:yasaExtended:{stage}:fdeltaaa:mean', 'name': 'Fdelta And Above'},
        {'key': f'night:yasaExtended:{stage}:thetaaa:mean', 'name': 'Theta And Above'},
        {'key': f'night:yasaExtended:{stage}:alphaaa:mean', 'name': 'Alpha And Above'},
        {'key': f'night:yasaExtended:{stage}:sigmaaa:mean', 'name': 'Sigma And Above'},
        {'key': f'night:yasaExtended:{stage}:fdeltaabsab:mean', 'name': 'FdeltaAbs And Below'},
        {'key': f'night:yasaExtended:{stage}:thetaabsab:mean', 'name': 'ThetaAbs And Below'},
        {'key': f'night:yasaExtended:{stage}:alphaabsab:mean', 'name': 'AlphaAbs And Below'},
        {'key': f'night:yasaExtended:{stage}:sigmaabsab:mean', 'name': 'SigmaAbs And Below'},
        {'key': f'night:yasaExtended:{stage}:fdeltaab:mean', 'name': 'Fdelta And Below'},
        {'key': f'night:yasaExtended:{stage}:thetaab:mean', 'name': 'Theta And Below'},
        {'key': f'night:yasaExtended:{stage}:alphaab:mean', 'name': 'Alpha And Below'},
        {'key': f'night:yasaExtended:{stage}:sigmaab:mean', 'name': 'Sigma And Below'},

    ]
    return [key for key in keys if key is not None]

def get_wakings_keys():
    return [
        {'key': 'night:aggregated:microwakingsPerHour', 'name': 'Microwakings'},
        {'key': 'night:yasa:adjusted:wakingsWithDurationCount', 'name': 'Wakings'},
        {'key': 'night:yasa:adjusted:timeAwakeAfterSleepSecs', 'name': 'Time Awake After Sleep'}
    ]

def get_next_day_energy_keys():
    return [
        {'key': 'energy:energyScore', 'name': 'Energy'}
    ]

def get_settling_keys():
    return [
        {'key': 'night:aggregated:asleepTimeSSM', 'name': 'Asleep Time'},
        {'key': 'night:aggregated:betweenReadyToSleepAndAsleepSecs', 'name': 'RTS to Asleep'},
        # Not working yet
        # {'key': 'night:yasaExtended:30MinsBeforeReadyToSleep:sdelta:mean', 'name': 'Sdelta before RTS'},
        # {'key': 'night:yasaExtended:30MinsBeforeReadyToSleep:fdelta:mean', 'name': 'Fdelta before RTS'},
        # {'key': 'night:yasaExtended:30MinsBeforeReadyToSleep:theta:mean', 'name': 'Theta before RTS'},
        # {'key': 'night:yasaExtended:30MinsBeforeReadyToSleep:alpha:mean', 'name': 'Alpha before RTS'},
        # {'key': 'night:yasaExtended:30MinsBeforeReadyToSleep:sigma:mean', 'name': 'Sigma before RTS'},
        # {'key': 'night:yasaExtended:30MinsBeforeReadyToSleep:beta:mean', 'name': 'Beta before RTS'}
    ]

def handle_best_night(X, target_col):
    # Get all the keys
    deep_keys = get_stage_keys('N3')
    rem_keys = get_stage_keys('R')
    n1_keys = get_stage_keys('N1')
    n2_keys = get_stage_keys('N2')
    all_sleep_keys = get_stage_keys('Sleep')
    # not_rem_keys = get_stage_keys('NotREM')
    # not_deep_keys = get_stage_keys('NotDeep')
    wakings_keys = get_wakings_keys()
    settling_keys = get_settling_keys()
    physical_keys = get_physical_keys()
    sleep_architecture_keys = get_sleep_architecture_keys()

    # Combine all keys
    # all_keys = deep_keys + rem_keys + n1_keys + n2_keys + all_sleep_keys + wakings_keys + settling_keys +  temp remove physical_keys + sleep_architecture_keys # will be added later + not_rem_keys + not_deep_keys
    all_keys = deep_keys + rem_keys + n1_keys + n2_keys + all_sleep_keys + wakings_keys + settling_keys + sleep_architecture_keys # will be added later + not_rem_keys + not_deep_keys

    # Extract the keys and names from the dictionaries
    keys = [item['key'] for item in all_keys]
    if target_col not in keys:
        keys.append(target_col)
    keys = [key for key in keys if key in X.columns]
    columns_mapping = {item['key']: item['name'] for item in all_keys}

    # Filter the DataFrame to include only the specified columns
    filtered_X = X[keys]

    # Rename the columns
    #renamed_X = filtered_X.rename(columns=columns_mapping)

    return filtered_X

def fully_under_my_control(ml_key: str) -> bool:
    lower = ml_key.lower()
    return (ml_key.startswith("drugsAndSupplements")
            or ml_key == "night:aggregated:gotIntoBedTimeSSM"
            or ml_key.startswith("exercise")
            or ml_key.startswith("sunExposure")
            or ml_key.startswith("bedroomEnvironment")
            or ml_key.startswith("phoneUsage")
            or ml_key.startswith("events")
            or ml_key.startswith("food")
            or ml_key.startswith("oxa"))

def somewhat_under_my_control(ml_key: str) -> bool:
    return (ml_key.find("asleepTimeSSM") != -1
            or ml_key.find("hasYasa") != -1
            or ml_key.endswith("readyToSleepTimeSSM")
            or ml_key.find("pees") != -1
            or ml_key.find("fitbit:heartrate") != -1
            or ml_key.startswith("hr"))

def handle_under_my_control(X, target_col):
    cols = [col for col in X.columns if col == target_col or fully_under_my_control(col) or somewhat_under_my_control(col)]
    return X[cols]

class DayDataFeaturesHandler(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, whitelist_sources: [str], blacklist_sources: [str]):
        self.target_col = target_col
        self.whitelist_sources = whitelist_sources
        self.blacklist_sources = blacklist_sources

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = X.columns

        out = X.copy()

        if "curated_night" in self.whitelist_sources:
            out = handle_best_night(out, self.target_col)
        elif "under_my_control" in self.whitelist_sources:
            out = handle_under_my_control(out, self.target_col)
        else:
            whitelisted = allowed_day_data_features(self.whitelist_sources, features)
            blacklisted = allowed_day_data_features(self.blacklist_sources, features)

            allowed = [feature for feature in whitelisted if feature not in blacklisted]

            if self.target_col not in allowed and self.target_col in X.columns:
                allowed.append(self.target_col)

            # Sorting helps ensure model will line up with data later
            X = X.reindex(sorted(X.columns), axis=1)
            allowed = sorted(allowed)
            out = X[allowed]

        print(f"DayDataFeaturesHandler {X.shape} to {out.shape}, first index {out.index[0]}")
        return out


