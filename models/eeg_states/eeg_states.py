import pandas as pd
from tqdm import tqdm
from typing_extensions import Dict

from models.microwakings_1.microwakings1 import load_model
from sleep_events import connect_to_firebase
from sleep_events import convert_timestamps_to_uk

def load_and_prepare_tired_wired_eeg_state_events():
    events = load_tired_wired_events()

    return load_and_prepare_shared_eeg_state_events(events)


def load_and_prepare_generic_eeg_state_events():
    event_types = load_generic_event_types()
    events = load_generic_events(event_types)
    return load_and_prepare_shared_eeg_state_events(events)

def load_and_prepare_day_data_energy_eeg_state_events():
    events = load_day_data_energy_events()
    return load_and_prepare_shared_eeg_state_events(events)


def load_and_prepare_shared_eeg_state_events(events):
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
    ], "day")


def load_generic_event_types(phase: str = None):
    db = connect_to_firebase()
    docs = db.collection('genericState').stream()
    generic_state_types = [doc.to_dict()['type'] for doc in docs]
    built_in_types = ["wired", "mid", "tired", "sleepy"]
    generic_state_types = generic_state_types + built_in_types
    plus_long = [f"{state}_long" for state in generic_state_types]
    return generic_state_types + plus_long

def load_tired_wired_events():
    events = load_generic_events([
        "wired", "wired_long",
        "mid", "mid_long",
        "tired", "tired_long",
        "sleepy", "sleepy_long"
    ])
    events = events[(events['phase'].isna()) | (events['phase'] == 'settling')]
    return events

def load_generic_events(filter_list: [str], phase: str = None):
    db = connect_to_firebase()
    docs = db.collection('homeAssistantExperimental').stream()
    records = [doc.to_dict() for doc in docs]
    if phase is not None:
        records = [record for record in records if 'phase' in record and record['phase'] == phase]
    events = pd.DataFrame(records)
    events = events[events['event'].isin(filter_list)]
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

    epoch_type = None
    matched_event = None

    for j, event_row in events.iterrows():
        event = event_row['event']
        event_end = event_row['TimestampUK']
        event_start = event_row['SinceUK']

        base_event = event.replace('_long', '')

        if verbose:
            print(f"Epoch start {epoch_start} end {epoch_end} event {event} start {event_start} end {event_end} > {event_start > epoch_end} < {event_start < epoch_start}")
        
        if event.endswith('_long'):
            if epoch_end > event_start and epoch_start < event_end:
                epoch_type = base_event
                matched_event = j
                break
        else:
            if epoch_start <= event_end <= epoch_end:
                epoch_type = base_event
                matched_event = j
                break

    return epoch_type, matched_event


def add_event_type(yasa_df, events):
    yasa_df['epoch_type'] = None
    yasa_df['matched_event'] = None

    unique_days = yasa_df['dayAndNightOf'].astype(str).unique()
    filtered_events = events[events['dayAndNightOf'].astype(str).isin(unique_days)]

    for i, yasa_row in tqdm(yasa_df.iterrows(), total=yasa_df.shape[0]):
        epoch_type, matched_night_event = process_row(yasa_row, filtered_events)
        yasa_df.at[i, 'epoch_type'] = epoch_type
        yasa_df.at[i, 'matched_event'] = matched_night_event



