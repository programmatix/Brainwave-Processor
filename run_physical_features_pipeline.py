import pandas as pd
import os
import pytz
from datetime import timedelta, datetime

import pandas as pd
from yasa import sw_detect, spindles_detect

import convert
import models.manual_sleep_scoring_catboost_1.manual_sleep_scoring_catboost_1 as best_model
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
    cached = input_file_without_ext + ".physical_features.csv"

    def combine(out: pd.DataFrame):
        # yasa_df and out are already merged, but probably shouldn't be...

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

    def regenerate():
        out = physical_features_pipeline(log, yasa_df)
        log("Saving to: " + cached)
        out.to_csv(cached, index=False)
        return combine(out), False

    if os.path.exists(cached):
        log("Loading cached file " + cached)
        out = pd.read_csv(cached)

        modification_time = os.path.getmtime(cached)
        modification_date = datetime.fromtimestamp(modification_time)
        if force:
            log("Forced rebuild")
            return regenerate()
        if modification_date < force_if_older_than:
            log("Cached file " + cached + f" mod date {modification_date} is < {force_if_older_than}, rebuilding")
            return regenerate()

        out.set_index('Epoch', inplace=True)
        return combine(out), True
    else:
        log(f"No cached file {cached}, rebuilding")
        return regenerate()

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
        "android_hr": f'''
            SELECT median("hr") AS HR, median("hrvRMSSDSomewhatRecent") AS HrvRMSSDSomewhatRecent
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
            SELECT median("o2") AS O2,
            median("movement") as Movement
            FROM "XL"."autogen"."android_o2"
            WHERE time >= '{start_time}' AND time <= '{end_time}'
            {group_by}
            ORDER BY time
        ''',
        "android_accel": f'''
            SELECT
                min("x") as PositionMinX, max("x") as PositionMaxX, median("x") as PositionMedX,
                min("y") as PositionMinY, max("y") as PositionMaxY, median("y") as PositionMedY,
                min("z") as PositionMinZ, max("z") as PositionMaxZ, median("z") as PositionMedZ
            FROM "XL"."autogen"."android_accel"
            WHERE time >= '{start_time}' AND time <= '{end_time}'
            {group_by}
            ORDER BY time
        '''
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
    return merged_df


def physical_features_pipeline(log, yasa_df: pd.DataFrame):
    sleep_events.convert_timestamps_to_uk(yasa_df, 'Timestamp', 'TimestampUK')

    epoch_length_secs = 30
    start_time, misalignment = calculate_offset(yasa_df['TimestampUK'].iloc[0], epoch_length_secs)
    end_time, misalignment = calculate_offset(yasa_df['TimestampUK'].iloc[-1], epoch_length_secs)
    # Influx will return buckets starting from nearest round 30s without misalignment adjustment
    group_by = f'GROUP BY time({epoch_length_secs}s, {misalignment}s)'

    log(f"start_time, end_time, misalignment")

    merged_df = get_influx_data(start_time, end_time, group_by)

    # Probably shouldn't have merged these...
    yasa_df.reset_index(drop=True, inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return pd.DataFrame.merge(yasa_df, merged_df, left_index=True, right_index=True, suffixes=['', '_r'])