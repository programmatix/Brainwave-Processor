import pandas as pd
from tqdm import tqdm

from sleep_events import connect_to_firebase
from sleep_events import convert_timestamps_to_uk

def load_and_prepare_eeg_state_events():
    events = load_events()
    events = debounce_events(events)
    events = find_durations(events)
    convert_timestamps(events)

    events['dayAndNightOf'] = pd.to_datetime(events['TimestampUK']).dt.date
    events.loc[events['TimestampUK'].dt.hour < 9, 'dayAndNightOf'] += pd.Timedelta(days=1)

    return events


def load_events():
    db = connect_to_firebase()
    docs = db.collection('homeAssistantExperimental').stream()
    records = [doc.to_dict() for doc in docs]

    events = pd.DataFrame(records)
    events = events[events['event'].isin([
        "wired", "wired_long",
        "mid", "mid_long",
        "tired", "tired_long",
        "sleepy", "sleepy_long"
    ])]
    
    return events


def debounce_events(events: pd.DataFrame):
    events = events.sort_values(by='timestamp').reset_index(drop=True)

    # Initialize a list to keep track of rows to drop
    rows_to_drop = []

    for i, row in events.iterrows():
        if row['event'] in ['wired_long', 'tired_long']:
            current_time = row['timestamp']
            previous_time = current_time - pd.Timedelta(seconds=2)

            # Check for any 'wired' or 'tired' event within the last 2 seconds
            for j in range(i-1, -1, -1):
                previous_row = events.iloc[j]
                if previous_row['timestamp'] >= previous_time:
                    if (row['event'] == 'wired_long' and previous_row['event'] == 'wired') or \
                            (row['event'] == 'tired_long' and previous_row['event'] == 'tired'):
                        rows_to_drop.append(j)
                else:
                    break

    # Drop the identified rows
    events = events.drop(rows_to_drop).reset_index(drop=True)
    return events


def find_durations(events: pd.DataFrame):
    events = events.sort_values(by='timestamp').reset_index(drop=True)
    
    # Initialize the 'since' column with NaN values
    events['since'] = pd.NaT
    
    for i, row in events.iterrows():
        if row['event'] == 'tired_long' or row['event'] == 'wired_long':
            current_time = row['timestamp']
            since_time = current_time - pd.Timedelta(minutes=5)
    
            # Check for any event within the last 5 minutes
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


def process_row(yasa_row, events):
    epoch_start = yasa_row['TimestampUK']
    epoch_end = epoch_start + pd.Timedelta(seconds=30)

    epoch_type = None
    matched_event = None

    for j, event_row in events.iterrows():
        event = event_row['event']
        event_end = event_row['TimestampUK']
        event_start = event_row['SinceUK']

        base_event = event.replace('_long', '')
        
        if event.endswith('_long'):
            if not (event_start > epoch_end or event_start < epoch_start):
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



