import firebase_admin
from firebase_admin import credentials, firestore
import os
import pandas as pd
import pytz
import pandas as pd
from pandas import json_normalize
import json
import mongo_client

def connect_to_firebase():
    if not firebase_admin._apps:
        home_dir = os.path.expanduser("~")
        # firebase_credentials_path = os.path.join(home_dir, "examined-life-dd234-firebase-adminsdk-f515f-f30d76e25d.json")
        firebase_credentials_path = os.path.join(home_dir, "examined-life-dd234-firebase-adminsdk-f515f-09f1331d09.json")

        cred = credentials.Certificate(firebase_credentials_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    return db



def load_sleep_events(log, start_date, end_date, waking_start_time_tz = None, waking_end_time_tz = None):
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


def load_eeg_events(log):
    db = connect_to_firebase()

    docs = db.collection('eegEvents').stream()

    records = [doc.to_dict() for doc in docs]

    events_df = pd.DataFrame(records)

    return events_df

def convert_timestamp_to_uk(timestamp):
    """
    Convert a single timestamp to UK time, handling mixed timezone formats.

    Parameters:
    -----------
    timestamp : str or Timestamp
        A single timestamp value

    Returns:
    --------
    Timestamp
        The timestamp converted to UK time
    """
    try:
        # Try parsing with timezone inference
        ts = pd.to_datetime(timestamp)

        # Check if timestamp is timezone-aware
        is_tz_aware = ts.tz is not None

        if is_tz_aware:
            # If already timezone-aware, convert directly to UK time
            return ts.tz_convert('Europe/London')
        else:
            # If timezone-naive, localize to UTC first
            return ts.tz_localize('UTC').tz_convert('Europe/London')
    except Exception:
        try:
            # Try parsing with explicit timezone offsets
            ts = pd.to_datetime(timestamp, utc=True)
            return ts.tz_convert('Europe/London')
        except Exception as e:
            print(f"Error converting timestamp: {timestamp}")
            print(e)
            return pd.NaT

def convert_timestamps_to_uk(df, src_timestamp_col='Timestamp', dest_timestamp_col='Timestamp'):
    """
    Convert timestamps in a DataFrame to UK time, handling mixed timezone formats.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the timestamp column
    timestamp_col : str
        Name of the column containing timestamps

    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional column for UK time
    """
    df[dest_timestamp_col] = df[src_timestamp_col].apply(convert_timestamp_to_uk)

# Converts anything to datetime64[ns, Europe/London]
# If there is anything suggesting a timezone, it'll use that, and convert into Europe/London.
# Otherwise it'll assume it's a localtime in Europe/London already.
def convert_timestamps_to_uk_optimised(df_series: pd.Series, debug = False) -> pd.Series:
    out = df_series.copy()

    if (len(out) == 0):
        if debug:
            print("Empty series, returning empty series")
        return out

    dtype = df_series.dtype
    if dtype == 'datetime64[ns]':
        if debug:
            print("Assuming datetime64[ns] is already a localtime in Europe/London zone, not timezone converting")
        #return pd.to_datetime(out, errors='coerce').dt.tz_localize('Europe/London')
        return pd.to_datetime(out).dt.tz_localize('Europe/London')
    elif dtype == 'str' or dtype == 'object':
        # Handle strings with Z info
        if debug:
            print("Trying to handle string as though it has timezone info")
        # Mixed as have seen microwakings file with mixed formats e.g. '2024-09-30 21:03:51.300000+00:00,2024-09-30 21:03:58+00:00' (2024-09-30)
        # Also have issues when crossing DST.  2025-03-29 has both +00:00 and +01:00.  Adding utc=True to deal with this:
        #  timezone-naive inputs are localized as UTC, while timezone-aware inputs are converted to UTC.
        return pd.to_datetime(out, format='mixed', utc=True).dt.tz_convert('Europe/London')
    else:
        raise ValueError(f"Unsupported dtype for convert_timestamps_to_uk_optimised: {dtype}")


        # Check if timestamp is timezone-aware
        # is_tz_aware = ts.tz is not None
        #
        # if is_tz_aware:
        #     # If already timezone-aware, convert directly to UK time
        #     return ts.tz_convert('Europe/London')
        # else:
        #     # If timezone-naive, localize to UTC first
        #     return ts.tz_localize('UTC').tz_convert('Europe/London')
    # except Exception as e:
    #     try:
    #         # Try parsing with explicit timezone offsets
    #         ts = pd.to_datetime(out, utc=True)
    #         return ts.tz_convert('Europe/London')
    #     except Exception as e:
    #         print(f"Error converting timestamp: {df_series}")
    #         print(e)
    #         raise e

# Converts a col like 'night:aggregated:asleepTimeSSM' which is in local time to a datetime64[ns, Europe/London]
def convert_to_datetime(row, time_column):
    if pd.notna(row[time_column]):
        #return pd.to_datetime(row[time_column], unit='s', origin=pd.Timestamp(row['dayAndNightOf']).normalize()).tz_localize('UTC').tz_convert('Europe/London')
        origin = pd.Timestamp(row['dayAndNightOf']).normalize()
        # Already in local time to don't pass through convert_timestamps_to_uk - just say it's a London timezone
        return pd.to_datetime(row[time_column], unit='s', origin=origin).tz_localize('Europe/London')
    else:
        return pd.NaT

def load_days_data(pimp = True):
    # db = connect_to_firebase()

    # docs = db.collection('daysExperimental').stream()

    docs = mongo_client.find('daysExperimental', None, { "ui": 0 })

    days = pd.DataFrame(docs)

    rename_dict = {}
    for col in days.columns:
        if not (col == 'ml' or col.endswith("_CONVERT_TO_DAY_AND_NIGHT_OF")):
            days.drop(col, axis=1, inplace=True)
        if col.endswith("_CONVERT_TO_DAY_AND_NIGHT_OF"):
            new_key = col[:-len("_CONVERT_TO_DAY_AND_NIGHT_OF")]
            rename_dict[col] = new_key

    days = days.rename(columns=rename_dict)

    df = days
    json_column = 'ml'
    exploded_df = json_normalize(df[json_column])
    result_df = pd.concat([df['dayAndNightOf'], exploded_df], axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    if pimp:
        result_df = pimp_my_days_data(result_df)

    return result_df

def test_days_data(results_df):
    assert results_df[results_df['dayAndNightOf'] == '2024-09-02']['asleepTime'].iloc[0] == pd.Timestamp('2024-09-02 23:30:00+01:00'), "Assertion failed: asleepTime does not match"
    #assert results_df[results_df['dayAndNightOf'] == '2024-09-02']['wakeTime'].iloc[0] == pd.Timestamp('2024-09-02 23:30:00+01:00'), "Assertion failed: asleepTime does not match"

def pimp_my_days_data(result_df):
    day_data = result_df.copy()

    day_data['dayAndNightOf'] = pd.to_datetime(day_data['dayAndNightOf'])

    time_columns = ['night:aggregated:asleepTimeSSM', 'night:aggregated:wakeTimeSSM', 'night:aggregated:gotIntoBedTimeSSM', 'night:aggregated:readyToSleepTimeSSM', 'circadian:basic:entries:LEP:datetimeSSM']
    new_columns = ['asleepTime', 'wakeTime', 'gotIntoBedTime', 'readyToSleepTime', 'LEPTime']

    for time_col, new_col in zip(time_columns, new_columns):
        day_data[new_col] = day_data.apply(lambda row: convert_to_datetime(row, time_col), axis=1)

    duration_columns = ['night:aggregated:timeAwakeAfterSleepSecs']
    new_duration_columns = ['timeAwakeAfterSleep']

    for old_col, new_col in zip(duration_columns, new_duration_columns):
        day_data[new_col] = pd.to_timedelta(day_data[old_col], unit='s')

    day_data.index = day_data['dayAndNightOf']

    return day_data


def load_nights_data(pimp = True):
    db = connect_to_firebase()
    docs = db.collection('nightsExperimental').stream()
    records = [doc.to_dict() for doc in docs]
    days = pd.DataFrame(records)

    df = days
    json_column = 'morningQuestionnaire'
    exploded_df = json_normalize(df[json_column])
    result_df = pd.concat([df['dayAndNightOf'], exploded_df], axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    result_df['dayAndNightOf'] = pd.to_datetime(result_df['dayAndNightOf'])

    return result_df



