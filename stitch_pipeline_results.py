import numpy as np
from tqdm.auto import tqdm
import os

import papermill as pm
from jupyter_client.manager import KernelManager
import scrapbook as sb
import pandas as pd

# 2024-12-28 needed 7
# from_hour and to_hour are because I'm looking for insomnia really - either waking up way too early, or waking for
# a long period in middle of night.  Also want to avoid false starts like 2024-11-25.
def find_long_wakes(df, min_epochs=60, max_gap=7, from_hour_inc=0, to_hour_exc=6):
    """Find stretches of SSWakeDuringSleep with small allowed gaps"""

    current_stretch = []
    gap_count = 0
    long_wakes = []

    for idx, is_wake in enumerate(df['SSWakeDuringSleep']):
        if is_wake:
            hour = df['Timestamp'].iloc[idx].hour
            if hour >= from_hour_inc and hour < to_hour_exc:
                if gap_count > 0:
                    # Fill in the gap epochs if the gap was small enough
                    if gap_count <= max_gap:
                        current_stretch.extend(range(idx - gap_count, idx))
                current_stretch.append(idx)
                gap_count = 0
        else:
            if len(current_stretch) > 0:
                gap_count += 1
                if gap_count > max_gap:
                    # Gap too large, check if current stretch is long enough
                    if len(current_stretch) >= min_epochs:
                        long_wakes.append(current_stretch)
                    current_stretch = []
                    gap_count = 0

    # Check final stretch
    if len(current_stretch) >= min_epochs:
        long_wakes.append(current_stretch)

    return long_wakes

def add_mins_until_long_wake(df):
    df['SSMinsUntilLongWake'] = -1
    for day, group in df.groupby('dayAndNightOf'):
        long_wake_indices = group[group['SSDuringLongWake']].index
        if not long_wake_indices.empty:
            first_long_wake_idx = long_wake_indices[0]
            df.loc[group.index, 'SSMinsUntilLongWake'] = (df.loc[first_long_wake_idx, 'Timestamp'] - df.loc[group.index, 'Timestamp']).dt.total_seconds() / 60
    return df

def add_ss_long_wake_this_night_and_before_it(df):
    df['SSLongWakeThisNight'] = df.groupby('dayAndNightOf')['SSDuringLongWake'].transform('any')
    df['SSLongWakeThisNightAndIsBefore'] = df['SSLongWakeThisNight'] & (df['SSMinsUntilLongWake'] > 0)
    return df


# Generally when stitching the data together we want to work with solid nights.
def remove_days_per_questionnaire(df):
    nights = sleep_events.load_nights_data()
    # Could rescue a few more partial days if required - see UsuableEEGDataEDA.  Not enough to bother with for now.
    usable_nights = nights[nights['yasa.usable'] == 'Yes']
    df = df[df['dayAndNightOf'].isin(usable_nights['dayAndNightOf'])]
    return df


def add_sleep_stages(out):
    out['SSDeep'] = out['Stage'] == 'N3'
    out['SSWake'] = out['Stage'] == 'W'
    out['SSN1'] = out['Stage'] == 'N1'
    out['SSN2'] = out['Stage'] == 'N2'
    out['SSR'] = out['Stage'] == 'R'

    out['SSPreReadyToSleep'] = out['minsSinceReadyToSleep'] <= 0
    out['SSAfterSleep'] = out['minsSinceAsleep'] >= 0
    out['SSDuringReadyToSleep'] = out['SSPreReadyToSleep'] & ~out['SSAfterSleep']
    out['SSAfterFinalWake'] = out['minsUntilWake'] < 0
    out['SSDuringSleep'] = out['SSAfterSleep'] & ~out['SSAfterFinalWake']
    out['SSWakeDuringSleep'] = out['SSDuringSleep'] & out['SSWake']

    long_wake_periods = find_long_wakes(out)
    out['SSDuringLongWake'] = False
    for period in long_wake_periods:
        out.loc[period, 'SSDuringLongWake'] = True
    add_mins_until_long_wake(out)
    add_ss_long_wake_this_night_and_before_it(out)


def post_stitch(df,
                should_remove_days_per_questionnaire: bool = True) -> pd.DataFrame:
    out = df.copy()
    out['dayAndNightOf'] = pd.to_datetime(out['dayAndNightOf'])

    if should_remove_days_per_questionnaire:
        out = remove_days_per_questionnaire(out)

    out.reset_index(drop=True, inplace=True)

    add_sleep_stages(out)

    return out

def stitch_all_days(input_dir: str, force: bool = False):
    all_dfs = []
    km = KernelManager()

    print(os.getcwd())

    dirs = next(os.walk(input_dir))[1]
    for idx, dir_name in enumerate(tqdm(dirs)):
        #tqdm.write(f"Processing notebook in: {dir_name}")

        output_filename = f'papermill_out/Stitch{dir_name}.ipynb'
        pm.execute_notebook(
            '../../StitchPipelineResultsSingleDay.ipynb',
            output_filename,
            km,
            parameters=dict(input_dir=input_dir, dir_name=dir_name, verbose=False, force=force)
        )

        nb = sb.read_notebook(output_filename)
        output_filename = f"out_{dir_name}.csv"
        if not os.path.exists(output_filename):
            continue
        out_df = pd.read_csv(output_filename)
        # if 'out_df' not in nb.scraps:
        #     continue
        # out_df = nb.scraps['out_df'].data
        all_dfs.append(out_df)

    return post_stitch(pd.concat(all_dfs))

# The papermill workflow is about 6x slower than this sadly
def stitch_all_days_optimised(input_dir: str,
                              force: bool = False,
                              remove_non_main_eeg: bool = True,
                              should_remove_days_per_questionnaire: bool = True,
                              # False by default as it's a bit slow
                              include_microwakings: bool = False):
    all_dfs = []

    dirs = next(os.walk(input_dir))[1]
    for idx, dir_name in enumerate(tqdm(dirs)):
        #tqdm.write(f"Processing notebook in: {dir_name}")

        out_df = stitch_day_optimised(input_dir,
                                      dir_name,
                                      force,
                                      remove_non_main_eeg,
                                      include_microwakings)
        if out_df is None:
            continue
        all_dfs.append(out_df)

    return post_stitch(pd.concat(all_dfs), should_remove_days_per_questionnaire)

from sleep_events import convert_timestamps_to_uk
from models.util.papermill_util import exit_early
import os
import pandas as pd
import sleep_events
from importlib import reload
reload(sleep_events)
from sleep_events import convert_timestamps_to_uk_optimised

def stitch_day_optimised(input_dir: str,
                         dir_name: str,
                         force: bool = False,
                         remove_non_main_eeg: bool = True,
                         include_microwakings: bool = False):


    yasa_file = os.path.join(input_dir, dir_name, "raw.yasa.csv")
    if not os.path.exists(yasa_file):
        return None
    yasa_df = pd.read_csv(yasa_file)

    post_yasa_file = os.path.join(input_dir, dir_name, "raw.post_yasa.csv")
    post_yasa_df = None
    if os.path.exists(post_yasa_file):
        post_yasa_df = pd.read_csv(post_yasa_file)

    post_human_file = os.path.join(input_dir, dir_name, "raw.post_human.csv")
    post_human_df = None
    if os.path.exists(post_human_file):
        post_human_df = pd.read_csv(post_human_file)

    physical_features_file = os.path.join(input_dir, dir_name, "raw.physical_features.csv")
    physical_features_df = None
    if os.path.exists(physical_features_file):
        physical_features_df = pd.read_csv(physical_features_file)

    final_wake_file1 = os.path.join(input_dir, dir_name, "raw.final_wake_model.csv")
    final_wake_file2 = os.path.join(input_dir, dir_name, "raw.final_wake_model_post_human.csv")
    final_wake_df = None
    if os.path.exists(final_wake_file2):
        final_wake_df = pd.read_csv(final_wake_file2)
    elif os.path.exists(final_wake_file1):
        final_wake_df = pd.read_csv(final_wake_file1)
    if final_wake_df is not None:
        final_wake_df.drop(final_wake_df.columns[0], axis=1, inplace=True)

    out_df = yasa_df.copy()

    if post_yasa_df is not None:
        out_df = pd.merge(out_df, post_yasa_df, left_index=True, right_index=True, how='outer', suffixes=('', '_duplicate_from_post_yasa'))

    if post_human_df is not None:
        out_df = pd.merge(out_df, post_human_df, left_index=True, right_index=True, how='outer',
                          suffixes=('', '_duplicate_from_post_human'))

    if physical_features_df is not None:
        out_df = pd.merge(out_df, physical_features_df, left_index=True, right_index=True, how='outer',
                          suffixes=('', '_duplicate_from_physical_features'))

    if final_wake_df is not None:
        out_df = pd.merge(out_df, final_wake_df, left_index=True, right_index=True, how='outer',
                          suffixes=('', '_duplicate_from_final_wake'))

    duplicate_columns = [col for col in out_df.columns if 'duplicate' in col]
    out_df.drop(columns=duplicate_columns, inplace=True)
    out_df['Timestamp'] = convert_timestamps_to_uk_optimised(out_df['Timestamp'])
    out_df['EpochEnd'] = out_df['Timestamp'] + pd.Timedelta(seconds=30)

    microwakings_file = os.path.join(input_dir, dir_name, "raw.microwakings.csv")
    if include_microwakings and os.path.exists(microwakings_file):
        out_df = load_and_merge_microwakings(microwakings_file, out_df)

    if remove_non_main_eeg:
        # Getting rid of "T4-M1_Stage" here also
        cols_to_remove = [col for col in out_df if ("_eeg_" in col and "Main" not in col) or "-M1_" in col]
        out_df.drop(columns=cols_to_remove, inplace=True)

    return out_df

def load_and_merge_microwakings(microwakings_file, df):
    df = df.copy().reset_index()
    microwakings = pd.read_csv(microwakings_file)

    # Convert timestamps in microwakings to UK-optimized format
    microwakings['Start'] = convert_timestamps_to_uk_optimised(microwakings['Start'])
    microwakings['End'] = convert_timestamps_to_uk_optimised(microwakings['End'])

    # Add EpochEnd and initialize new columns
    df['EpochEnd'] = df['Timestamp'] + pd.Timedelta(seconds=30)
    df['MicrowakingDbgIndex'] = -1
    df['MicrowakingDbgStart'] = pd.NaT
    df['MicrowakingDbgEnd'] = pd.NaT

    # Use a vectorized approach to check overlaps
    microwakings_array = microwakings[['Start', 'End']].to_numpy()
    timestamp_array = df[['Timestamp', 'EpochEnd']].to_numpy()

    # Calculate overlaps
    starts = np.maximum(timestamp_array[:, 0][:, None], microwakings_array[:, 0])
    ends = np.minimum(timestamp_array[:, 1][:, None], microwakings_array[:, 1])
    overlap_durations = (ends - starts).astype('timedelta64[s]').astype(float)
    overlaps = overlap_durations > 1

    # Determine the most recent microwaking index for overlapping rows
    idxs, microwaking_idxs = np.where(overlaps)
    df.loc[idxs, 'MicrowakingDbgIndex'] = microwaking_idxs
    df.loc[idxs, 'MicrowakingDbgStart'] = microwakings['Start'].iloc[microwaking_idxs].values
    df.loc[idxs, 'MicrowakingDbgEnd'] = microwakings['End'].iloc[microwaking_idxs].values

    # Calculate nearby microwaking flags
    df['MicrowakingHere'] = df['MicrowakingDbgIndex'] != -1
    df['MicrowakingInNext'] = df['MicrowakingHere'].shift(-1, fill_value=False)
    df['MicrowakingInPrev'] = df['MicrowakingHere'].shift(1, fill_value=False)
    df['MicrowakingHereOrNearby'] = df['MicrowakingHere'] | df['MicrowakingInNext'] | df['MicrowakingInPrev']

    return df
