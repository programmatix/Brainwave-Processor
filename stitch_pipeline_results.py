
from tqdm.auto import tqdm
import os

import papermill as pm
from jupyter_client.manager import KernelManager
import scrapbook as sb
import pandas as pd

def post_stitch(df) -> pd.DataFrame:
    out = df.copy()
    out['dayAndNightOf'] = pd.to_datetime(out['dayAndNightOf'])
    out['SleepStageDeep'] = out['Stage'] == 'N3'
    out['SleepStageWake'] = out['Stage'] == 'W'
    out['SleepStageN1'] = out['Stage'] == 'N1'
    out['SleepStageN2'] = out['Stage'] == 'N2'
    out['SleepStageR'] = out['Stage'] == 'R'
    out['SleepStagePreReadyToSleep'] = out['minsSinceReadyToSleep'] <= 0
    out['SleepStageDuringReadyToSleep'] = out['DuringReadyToSleep']
    out['SleepStageAfterSleep'] = out['minsSinceAsleep'] >= 0
    out['SleepStageAfterWake'] = out['minsUntilWake'] < 0

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
def stitch_all_days_optimised(input_dir: str, force: bool = False):
    all_dfs = []

    dirs = next(os.walk(input_dir))[1]
    for idx, dir_name in enumerate(tqdm(dirs)):
        #tqdm.write(f"Processing notebook in: {dir_name}")

        out_df = stitch_day_optimised(input_dir, dir_name, force)
        if out_df is None:
            continue
        all_dfs.append(out_df)

    return post_stitch(pd.concat(all_dfs))

from sleep_events import convert_timestamps_to_uk
from models.util.papermill_util import exit_early
import os
import pandas as pd
import sleep_events
from importlib import reload
reload(sleep_events)
from sleep_events import convert_timestamps_to_uk_optimised

def stitch_day_optimised(input_dir: str, dir_name: str, force: bool = False):


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
    return out_df