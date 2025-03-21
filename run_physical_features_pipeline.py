import pandas as pd
import os
import pytz
from datetime import timedelta, datetime

import pandas as pd
from yasa import sw_detect, spindles_detect

import convert
import run_yasa
import scaling
import sleep
import sleep_events
import wakings
import sleep
import yasa_features
from models.microwakings_1 import microwakings1
from models.microwakings_1.microwakings1 import PerFile
import tensorflow as tf
from memory import garbage_collect
import traceback
import warnings
import logging
import mne
from datetime import timezone
from influxdb import InfluxDBClient
import dotenv
import os
dotenv.load_dotenv()
import pandas as pd


force_if_older_than = datetime(2024, 12, 2, 0, 0, 0)



# yasa_df: results from the YASA pipeline - raw.yasa.csv
def cached_physical_features_pipeline(log, input_file: str, yasa_df: pd.DataFrame, force: bool = False, join: bool = False, pimp: bool = False):
    input_file_without_ext = os.path.splitext(input_file)[0]
    cached_30s = input_file_without_ext + ".physical_features.csv"
    cached_1s = input_file_without_ext + ".physical_features.1s.csv"

    def combine(out: pd.DataFrame):
        if pimp:
            sleep_events.convert_timestamps_to_uk(out, 'Timestamp', 'TimestampUK')
            if 'Movement' not in out:
                out['Movement'] = pd.Series(dtype='float64')
            if 'HR' not in out:
                out['HR'] = pd.Series(dtype='float64')
            if 'HrvRMSSDSomewhatRecent' not in out:
                out['HrvRMSSDSomewhatRecent'] = pd.Series(dtype='float64')
            if 'Temp' not in out:
                out['Temp'] = pd.Series(dtype='float64')
            if 'O2' not in out:
                out['O2'] = pd.Series(dtype='float64')
            if 'PositionMinX' not in out:
                out['PositionMinX'] = pd.Series(dtype='float64')
            if 'PositionMaxX' not in out:
                out['PositionMaxX'] = pd.Series(dtype='float64')
            if 'PositionMedX' not in out:
                out['PositionMedX'] = pd.Series(dtype='float64')
            if 'PositionMinY' not in out:
                out['PositionMinY'] = pd.Series(dtype='float64')
            if 'PositionMaxY' not in out:
                out['PositionMaxY'] = pd.Series(dtype='float64')
            if 'PositionMedY' not in out:
                out['PositionMedY'] = pd.Series(dtype='float64')
            if 'PositionMinZ' not in out:
                out['PositionMinZ'] = pd.Series(dtype='float64')
            if 'PositionMaxZ' not in out:
                out['PositionMaxZ'] = pd.Series(dtype='float64')
            if 'PositionMedZ' not in out:
                out['PositionMedZ'] = pd.Series(dtype='float64')
        return out

    def needs_rebuild(cached_file):
        if not os.path.exists(cached_file):
            return True
        modification_time = os.path.getmtime(cached_file)
        modification_date = datetime.fromtimestamp(modification_time)
        return force or modification_date < force_if_older_than

    def load_or_generate_30s():
        if needs_rebuild(cached_30s):
            log(f"Generating 30s data and saving to: {cached_30s}")
            out_30s = physical_features_pipeline(log, yasa_df, 30)
            out_30s.to_csv(cached_30s, index=False)
            return combine(out_30s), False
        else:
            log(f"Loading cached 30s file: {cached_30s}")
            out_30s = pd.read_csv(cached_30s)
            return combine(out_30s), True

    def load_or_generate_1s():
        if needs_rebuild(cached_1s):
            log(f"Generating 1s data and saving to: {cached_1s}")
            out_1s = physical_features_pipeline(log, yasa_df, 1)
            out_1s.to_csv(cached_1s, index=False)
            return combine(out_1s), False
        else:
            log(f"Loading cached 1s file: {cached_1s}")
            out_1s = pd.read_csv(cached_1s)
            return combine(out_1s), True

    out_30s, cached_30s = load_or_generate_30s()
    out_1s, cached_1s = load_or_generate_1s()
    
    return out_30s, out_1s, (cached_30s and cached_1s)

def calculate_offset(start_time, interval_seconds):
    epoch_seconds = int(start_time.timestamp())
    misalignment = epoch_seconds % interval_seconds
    return (start_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z', misalignment)


def get_influx_data(start_time: str, end_time: str, group_by: str):
    host = os.getenv('INFLUXDB_HOST')
    port = os.getenv('INFLUXDB_PORT')
    username = os.getenv('INFLUXDB_USERNAME')
    password = os.getenv('INFLUXDB_PASSWORD')
    database = os.getenv('INFLUXDB_DATABASE')

    client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database, ssl=True, verify_ssl=False)

    # Queries for each measurement
    queries = {
        # HRV needs porting from web
        "android_hr": f'''
            SELECT median("hr") AS HR
            FROM "XL"."autogen"."android_hr"
            WHERE time >= '{start_time}' AND time <= '{end_time}'
            {group_by}
            ORDER BY time
        ''',
        "android_temp": f'''
            SELECT median("temp") AS Temp
            FROM "XL"."autogen"."android_temp"
            WHERE time >= '{start_time}' AND time <= '{end_time}'
            {group_by}
            ORDER BY time
        ''',
        "android_o2": f'''
            SELECT median("o2") AS O2
            FROM "XL"."autogen"."android_o2"
            WHERE time >= '{start_time}' AND time <= '{end_time}'
            {group_by}
            ORDER BY time
        ''',
        # See viatomMovementStatement for logic
        "android_movement": f'''
            SELECT ceil(max("movement") / max("movement")) as Movement
            FROM "XL"."autogen"."android_o2"
            WHERE movement != 1 and movement != 2 and time >= '2025-01-10T00:00:00Z'
            AND time >= '{start_time}' AND time <= '{end_time}'
            {group_by}
            ORDER BY time
        ''',
        # Just not super useful right now
        # "android_accel": f'''
        #     SELECT
        #         min("x") as PositionMinX, max("x") as PositionMaxX, median("x") as PositionMedX,
        #         min("y") as PositionMinY, max("y") as PositionMaxY, median("y") as PositionMedY,
        #         min("z") as PositionMinZ, max("z") as PositionMaxZ, median("z") as PositionMedZ
        #     FROM "XL"."autogen"."android_accel"
        #     WHERE time >= '{start_time}' AND time <= '{end_time}'
        #     {group_by}
        #     ORDER BY time
        # '''
    }

    dataframes = {}
    for key, query in queries.items():
        result = client.query(query)
        points = list(result.get_points())
        df = pd.DataFrame(points)
        if 'time' in df:
            df['time'] = pd.to_datetime(df['time'])  # Ensure time is in datetime format
            df.set_index('time', inplace=True)
        dataframes[key] = df

    merged_df = pd.concat(dataframes.values(), axis=1)
    merged_df.reset_index(inplace=True)  # Keep the time column
    merged_df.rename(columns={'time': 'DatabaseTimestamp'}, inplace=True)
    return merged_df


def physical_features_pipeline(log, yasa_df: pd.DataFrame, epoch_length_secs: int = 1):
    sleep_events.convert_timestamps_to_uk(yasa_df, 'Timestamp', 'TimestampUK')

    start_time, misalignment = calculate_offset(yasa_df['TimestampUK'].iloc[0], epoch_length_secs)
    end_time, misalignment = calculate_offset(yasa_df['TimestampUK'].iloc[-1], epoch_length_secs)
    # Influx will return buckets starting from nearest round 30s without misalignment adjustment
    group_by = f'GROUP BY time({epoch_length_secs}s, {misalignment}s)'

    log(f"start_time: {start_time}, end_time: {end_time}, misalignment: {misalignment}")

    merged_df = get_influx_data(start_time, end_time, group_by)

    if epoch_length_secs == 30:
        merged_df.reset_index(drop=True, inplace=True)
        
        merged_df['Timestamp'] = yasa_df['Timestamp'].copy()
    return merged_df