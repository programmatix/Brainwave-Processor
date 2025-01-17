import os
import pytz
from datetime import timedelta, datetime

import pandas as pd

import convert
import run_yasa
import scaling
import sleep
import yasa_features
from models.microwakings_1 import microwakings1
import tensorflow as tf
from memory import garbage_collect
import traceback
import warnings
import logging
import mne
from datetime import timezone

logging.getLogger('yasa').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
warnings.filterwarnings("ignore", message="Channel locations not available. Disabling spatial colors.")
warnings.filterwarnings("ignore", message="WARNING - Hypnogram is SHORTER than data")
mne.set_log_level('ERROR')

force_if_older_than = datetime(2025, 1, 8, 0, 0, 0)

# This is the 2nd pipeline, the post-YASA one.  Post-human runs next.
# This gets YASA features, scales them, chooses main channel.


# yasa_df: results from the YASA pipeline - raw.yasa.csv
def cached_post_yasa_pipeline(log, input_file: str, yasa_df: pd.DataFrame, stats_df: pd.DataFrame, force: bool = False):
    input_file_without_ext = os.path.splitext(input_file)[0]
    cached = input_file_without_ext + ".post_yasa.csv"

    def regenerate():
        out = post_yasa_pipeline(log, input_file, yasa_df, stats_df)
        log("Saving to: " + cached)
        out.to_csv(cached, index=False)
        return out, False

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
        if "Main_eeg_sigmaabs" not in out.columns:
            log("Cached file " + cached + " is missing columns, rebuilding")
            return regenerate()

        out.set_index('Epoch', inplace=True)
        return out, True
    else:
        log(f"No cached file {cached}, rebuilding")
        return regenerate()


def post_yasa_pipeline(log, input_file: str, yasa_df: pd.DataFrame, stats_df: pd.DataFrame):
    # Load MNE
    mne.use_log_level("warning")
    log("Loading MNE file " + input_file)
    raw, input_file_without_ext, mne_filtered = convert.load_mne_file(log, input_file)

    channels = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    start_date = raw.info['meas_date']
    end_date = start_date + timedelta(seconds=float(raw.times[-1]))

    # Sleep events - v expensive, needs pushdown filter, not used for much
    # garbage_collect(log)
    # log("Loading sleep events")
    # ha_events = sleep_events.load_sleep_events(log, start_date, end_date)
    # output_csv_file = input_file_without_ext + ".night_events.csv"
    # ha_events.to_csv(output_csv_file, index=False)


    # YASA features
    garbage_collect(log)
    log("Extracting YASA features")
    yasa_feats, channel_feats_dict = yasa_features.extract_yasa_features2(log, channels, mne_filtered)

    # # Combine epochs and YASA features
    # garbage_collect(log)
    # df = yasa_feats.copy()
    # df['epoch'] = df['Epoch']
    # df.set_index('epoch', inplace=True)
    combined_df = yasa_df.join(yasa_feats)

    # Scaled
    scale_by_stats = scaling.scale_by_stats(combined_df, stats_df)

    yasa_feats = combined_df.join(scale_by_stats.add_suffix('_s'))


    # YASA slow waves
    # Disabling - I haven't got much signal from this and it's very expensive
    # garbage_collect(log)
    # log("Detecting slow waves")
    # sw = sw_detect(mne_filtered, sfreq)
    # if sw is not None:
    #     sw_summary = sw.summary()
    #     output_csv_file = input_file_without_ext + ".sw_summary.csv"
    #     sw_summary.to_csv(output_csv_file, index=False)

    # YASA spindles
    # Too intensive for Pi
    # garbage_collect(log)
    # log("Detecting spindles")
    # sp = spindles_detect(mne_filtered, sfreq)
    # if sp is not None:
    #     sp_summary = sp.summary()
    #     output_csv_file = input_file_without_ext + ".spindle_summary.csv"
    #     sp_summary.to_csv(output_csv_file, index=False)

    # Main channel (want to do after scaling)
    yasa_feats = scaling.add_main_channel(yasa_feats)

    # Automated waking scoring
    # We're training a model to more accurately predict waking than YASA.  So we have to be judicious in what YASA data we use - while being aware that manually scoring waking is challenging.  So only use data where YASA is supremely confident in wakefulness.
    # This stuff needs updating to support Main channel (should be easy)
    # garbage_collect(log)
    # log("Automated waking scoring")
    # df_probably_awake = wakings.get_yasa_probably_awake(log, combined_df)
    #
    # # Manual waking scoring
    # garbage_collect(log)
    # log("Manual waking scoring")
    # df_definitely_awake = wakings.get_definitely_awake(df_probably_awake, ha_events)
    #
    # # Combine probably and definitely awake
    # df_combined_awake = df_probably_awake.copy()
    # df_combined_awake['DefinitelyAwake'] = df_definitely_awake['DefinitelyAwake']
    # df_combined_awake['ProbablyAwake'] = (df_combined_awake['DefinitelyAwake'] == True) | (df_combined_awake['YASAProbablyAwake'] == True)
    #
    # # Epochs that are probably sleep
    # garbage_collect(log)
    # log("Epochs that are probably sleep")
    # df_asleep = sleep.probably_asleep(df_combined_awake)

    # Run current best YASAesque model
    # Skipping as seems to require Fpz
    # log("Running YASAesque model")
    # df_with_predictions = best_model.run_model(df_asleep)
    # output_csv_file = input_file_without_ext + ".with_features.csv"
    # df_with_predictions.to_csv(output_csv_file, index=False)

    log("All done! " + input_file)

    return yasa_feats


# def combine_all_file(log, input_dir: str):
#     errors = []
#     dataframes = []
#
#     for root, dirs, files in os.walk(input_dir):
#         for dir_name in dirs:
#             input_file = os.path.join(root, dir_name, "raw.with_features.csv")
#             try:
#                 log("Processing file: " + input_file)
#                 if os.path.exists(input_file):
#                     df = pd.read_csv(input_file)
#                     df['filename'] = input_file
#                     dataframes.append(df)
#             except Exception as e:
#                 log("Error processing file: " + input_dir)
#                 errors.append("Error processing file: " + input_file + " - " + str(e))
#                 log(e)
#
#     # Concatenate all dataframes into a single dataframe
#     if dataframes:
#         combined_df = pd.concat(dataframes, ignore_index=True)
#     else:
#         combined_df = pd.DataFrame()
#
#     for err in errors:
#         log(err)
#     df = combined_df
#     return df