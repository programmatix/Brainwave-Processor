import pandas as pd
from tqdm import tqdm
from typing_extensions import Dict

from models.microwakings_1.microwakings1 import load_model
from sleep_events import connect_to_firebase
from sleep_events import convert_timestamps_to_uk

def load_and_prepare_all_events():
    events = load_generic_events(None, ['pee', 'wake', 'lights out, ready for sleep', 'test - can delete'])
    out = load_and_prepare_shared_eeg_state_events(events)
    return out

def load_and_prepare_settling_eeg_state_events():
    events = load_settling_events()

    out = load_and_prepare_shared_eeg_state_events(events)
    return map_settling_events(out)

def load_and_prepare_custom_eeg_state_events():
    event_types = load_custom_event_types()
    events = load_generic_events(event_types)
    return load_and_prepare_shared_eeg_state_events(events)

def load_and_prepare_day_data_energy_eeg_state_events():
    events = load_day_data_energy_events()
    return load_and_prepare_shared_eeg_state_events(events)

def load_and_prepare_shared_eeg_state_events(events):
    events['BaseEvent'] = events['event'].str.replace('_long', '')
    events['Preserve_SettlingEvent'] = events['BaseEvent']
    events['Preserve_SettlingEventVersion'] = events['version']
    events = debounce_events(events)
    events = find_durations(events)
    convert_timestamps(events)

    events['dayAndNightOf'] = pd.to_datetime(events['TimestampUK']).dt.date
    events.loc[events['TimestampUK'].dt.hour < 5, 'dayAndNightOf'] += pd.Timedelta(days=1)

    return events

def load_day_data_energy_events():
    return load_generic_events([
        "near worst", "near worst_long",
        "struggling", "struggling_long",
        "tired", "tired_long",
        "standard tired", "standard tired_long",
        "okish", "okish_long",
        "lockable", "lockable_long",
        "great", "great_long"
    ])


def load_custom_event_types(phase: str = None):
    db = connect_to_firebase()
    docs = db.collection('genericState').stream()
    custom_state_types = [doc.to_dict()['type'] for doc in docs]
    plus_long = [f"{state}_long" for state in custom_state_types]
    return custom_state_types + plus_long

def add_long_variants(input: [str]):
    return input + [f"{state}_long" for state in input]

def load_settling_events():
    event_types = add_long_variants(['tired', 'wired', 'mid', 'sleepy', 'other', 'awake', 'alert',
                                     'relaxed', 'too alert', 'drowsy or sleepy', 'wired or alert',
                                     'tired but alert', 'tired but wired'])
    events = load_generic_events(event_types)
    events = events[(events['phase'] == 'settling') | (events['phase'].isna())]
    return events

def load_generic_events(white_list: [str] = None, black_list: [str] = None, phase: str = None):
    db = connect_to_firebase()
    query = db.collection('homeAssistantExperimental')

    if phase is not None:
        query = query.where('phase', '==', phase)

    if white_list is not None:
        query = query.where('event', 'in', white_list)

    if black_list is not None:
        query = query.where('event', 'not-in', black_list)

    docs = query.stream()
    records = [doc.to_dict() for doc in docs]

    events = pd.DataFrame(records)
    return events


def debounce_events(events: pd.DataFrame):
    events = events.sort_values(by='timestamp').reset_index(drop=True)
    rows_to_drop = []

    for i, row in events.iterrows():
        if row['event'].endswith('_long'):
            current_time = row['timestamp']
            previous_time = current_time - pd.Timedelta(seconds=2)
            base_event = row['event'].replace('_long', '')

            for j in range(i-1, -1, -1):
                previous_row = events.iloc[j]
                if previous_row['timestamp'] >= previous_time:
                    if previous_row['event'] == base_event:
                        rows_to_drop.append(j)
                else:
                    break

    events = events.drop(rows_to_drop).reset_index(drop=True)
    return events


def find_durations(events: pd.DataFrame):
    events = events.sort_values(by='timestamp').reset_index(drop=True)
    events['since'] = pd.NaT
    
    for i, row in events.iterrows():
        if row['event'].endswith('_long'):
            current_time = row['timestamp']
            since_time = current_time - pd.Timedelta(minutes=5)
    
            for j in range(i-1, -1, -1):
                previous_row = events.iloc[j]
                if previous_row['timestamp'] >= since_time:
                    since_time = previous_row['timestamp']
                else:
                    break
    
            events.at[i, 'since'] = since_time
    
    events['duration'] = events['timestamp'] - events['since']
    
    return events


def convert_timestamps(events: pd.DataFrame):
    convert_timestamps_to_uk(events, 'timestamp', 'TimestampUK')
    convert_timestamps_to_uk(events, 'since', 'SinceUK')
    events.drop(columns=['timestamp', 'since'], inplace=True)

def process_row_for_pulearn(yasa_row: pd.Series, events: pd.DataFrame, just_types: list) -> Dict[str, int]:
    epoch_start = yasa_row['TimestampUK']
    epoch_end = epoch_start + pd.Timedelta(seconds=30)

    # In pulearn, 0 doesn't mean "not event_type", it means may or may not be
    matched_events = {event_type: 0 for event_type in just_types}

    for _, event_row in events.iterrows():
        for event_type in just_types:
            event = event_row['event']
            event_end = event_row['TimestampUK']
            event_start = event_row['SinceUK']

            base_event = event.replace('_long', '')

            if event_type == base_event:
                if event.endswith('_long'):
                    if not (event_start > epoch_end or event_start < epoch_start):
                        matched_events[event_type] = 1
                        break
                else:
                    if epoch_start <= event_end <= epoch_end:
                        matched_events[event_type] = 1
                        break

    return matched_events

def process_row(yasa_row, events, verbose=False):
    epoch_start = yasa_row['TimestampUK']
    epoch_end = epoch_start + pd.Timedelta(seconds=30)

    matched_event = None
    matched_event_idx = None

    for j, event_row in events.iterrows():
        event = event_row['event']
        event_end = event_row['TimestampUK']
        event_start = event_row['SinceUK']

        # base_event = event.replace('_long', '')

        if verbose:
            print(f"Epoch start {epoch_start.strftime('%Y-%m-%d %H:%M:%S')} end {epoch_end.strftime('%Y-%m-%d %H:%M:%S')} event {event} start {event_start.strftime('%Y-%m-%d %H:%M:%S')} end {event_end.strftime('%Y-%m-%d %H:%M:%S')} > {event_start > epoch_end} < {event_start < epoch_start}")

        if event.endswith('_long'):
            if epoch_end > event_start and epoch_start < event_end:
                matched_event_idx = j
                matched_event = event_row
                break
        else:
            if epoch_start <= event_end <= epoch_end:
                matched_event_idx = j
                matched_event = event_row
                break

    return matched_event, matched_event_idx


# def add_event_type(yasa_df, events):
#     yasa_df['epoch_type'] = None
#     yasa_df['matched_event'] = None
#
#     unique_days = yasa_df['dayAndNightOf'].astype(str).unique()
#     filtered_events = events[events['dayAndNightOf'].astype(str).isin(unique_days)]
#
#     for i, yasa_row in tqdm(yasa_df.iterrows(), total=yasa_df.shape[0]):
#         epoch_type, matched_night_event = process_row(yasa_row, filtered_events)
#         yasa_df.at[i, 'epoch_type'] = epoch_type
#         yasa_df.at[i, 'matched_event'] = matched_night_event
#



def add_event_type(yasa_df, events, verbose=False):
    current_day = None
    filtered_events = pd.DataFrame()

    for i, yasa_row in tqdm(yasa_df.iterrows(), total=yasa_df.shape[0]):
        day_and_night_of = yasa_row['dayAndNightOf']

        if day_and_night_of != current_day:
            current_day = day_and_night_of
            filtered_events = events[events['dayAndNightOf'].astype(str) == current_day]
            #print(f"Filtered events for {current_day} with {filtered_events.shape[0]} events")

        matched_event, matched_event_idx = process_row(yasa_row, filtered_events, verbose)
        if matched_event is not None:
            yasa_df.at[i, 'SettlingEventMatchedIdx'] = matched_event_idx
            preserve_cols = [col for col in matched_event.index if col.startswith("Preserve_")]
            for col in preserve_cols:
                new_col_name = col.replace("Preserve_", "")
                yasa_df.at[i, new_col_name] = matched_event[col]


def map_settling_events(df):
    df = df.copy()

    # Replace 'n/a' with 'v1'
    df['version'] = df['version'].replace('n/a', 'v1')

    # Drop rows where version is 'v3'
    df = df[df['version'] != 'v3']

    mapping_v1_and_v2 = {
        'wired': (0, 0),
        'too alert': (0, 0),
        'mid': (0, 50),
        'other': (0, 50),
        'tired': (100, 75),
        'sleepy': (100, 100)
    }

    mapping_v4 = {
        'wired or alert': (0, 0),
        'tired but wired': (0, 25),
        'tired but alert': (0, 25),
        'tired': (100, 75),
        'drowsy or sleepy': (100, 100)
    }

    def map_event(row):
        if row['version'] == 'v4':
            return mapping_v4.get(row['BaseEvent'], row['BaseEvent'])
        else:
            return mapping_v1_and_v2.get(row['BaseEvent'], row['BaseEvent'])

    def map_event1(row):
        return map_event(row)[0]

    def map_event2(row):
        return map_event(row)[1]

    df['Preserve_SettlingManualScore'] = df.apply(map_event2, axis=1)
    df['Preserve_SettlingV4ManualScore'] = df[df['version'] == 'v4'].apply(map_event2, axis=1)
    # A very broad signal abstracting over as many settling event versions as possible
    df['Preserve_TiredVsAlertManualScore'] = df.apply(map_event1, axis=1)

    return df
