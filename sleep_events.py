import firebase_admin
from firebase_admin import credentials, firestore
import os
import pandas as pd
import pytz

def connect_to_firebase():
    if not firebase_admin._apps:
        home_dir = os.path.expanduser("~")
        firebase_credentials_path = os.path.join(home_dir, "examined-life-dd234-firebase-adminsdk-f515f-124ed5962e.json")

        cred = credentials.Certificate(firebase_credentials_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    return db



def load_sleep_events(log, start_date, end_date, waking_start_time_tz, waking_end_time_tz):
    db = connect_to_firebase()

    docs = db.collection('homeAssistantExperimental').stream()

    # Convert to list of dictionaries
    records = [doc.to_dict() for doc in docs]

    ha_events = pd.DataFrame(records)

    ha_events_for_file = ha_events[ha_events['timestamp'] >= start_date][ha_events['timestamp'] <= end_date][ha_events['event'] == 'wake']


    uk_timezone = pytz.timezone("Europe/London")
    ha_events_for_file['timestamp_uk'] = ha_events_for_file['timestamp'].dt.tz_convert(uk_timezone)

    ha_events_for_file['source'] = 'night_event_buttons'

    ha_events_for_file['duration_secs'] = 1

    if waking_start_time_tz is not None and waking_end_time_tz is not None:
        additional_data = {
            'event': ['wake'],
            'timestamp': [None],
            'timestamp_uk': [waking_start_time_tz],
            'source': ['manual_waking'],
            'duration_secs': [(waking_end_time_tz - waking_start_time_tz).total_seconds()]
        }
        additional_df = pd.DataFrame(additional_data)
        combined_df = pd.concat([ha_events_for_file, additional_df]).sort_values(by='timestamp_uk').reset_index(drop=True)
    else:
        combined_df = ha_events_for_file

    combined_df.drop(columns=['timestamp'], inplace=True)

    combined_df['timestamp_uk_str'] = combined_df['timestamp_uk'].dt.strftime('%Y-%m-%d %H:%M:%S%z')

    return combined_df

