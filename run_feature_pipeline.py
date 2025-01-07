import shutil

import os
import pytz
from datetime import datetime

import pandas as pd

import convert
import run_yasa
from memory import garbage_collect
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

force_if_older_than = datetime(2024, 12, 2, 0, 0, 0)

# This YASA pipeline is slow to run, so try to keep this immutable and solely related to YASA.

def cached_pipeline(log, input_file: str, force: bool = False):
    input_file_without_ext = os.path.splitext(input_file)[0]
    cached = input_file_without_ext + ".yasa.csv"

    def regenerate():
        out = pipeline(log, input_file)
        log("Saving to: " + cached)
        out.to_csv(cached, index=False)
        return out, False

    if os.path.exists(cached):
        log("Loading cached file " + cached)
        out = pd.read_csv(cached)

        # Most recent files
        json_exist = os.path.exists(input_file_without_ext + ".sleep.json")

        modification_time = os.path.getmtime(cached)
        modification_date = datetime.fromtimestamp(modification_time)

        if force:
            log("Forced rebuild")
            return regenerate()
        if modification_date < force_if_older_than:
            log("Cached file " + cached + f" mod date {modification_date} is < {force_if_older_than}, rebuilding")
            return regenerate()
        if not json_exist:
            log("No sleep.json, rebuilding")
            return regenerate()

        # Fixing old data bug without having to rebuild everything
        if any('abspow' in col for col in out.columns):

            if 'Fpz-M1_eeg_abspow' in out.columns:
                log("Has excess columns, truncating and saving")
                # Truncate the DataFrame to the specified columns
                out = out[['Stage', 'Confidence', 'Epoch', 'Timestamp', 'Source', 'Fpz-M1_Stage', 'Fpz-M1_Confidence', 'StageInt']]
                out.to_csv(cached, index=False)
            else:
                log("Has excess columns but cannot handle, rebuilding")
                return regenerate()

        out['epoch'] = out['Epoch']
        out.set_index('epoch', inplace=True)
        return out, True
    else:
        # Support the old way so we don't have to rebuild everything
        # if os.path.exists(input_file_without_ext + ".with_features.csv"):
        #     log("Cached file " + cached + " is missing, but with_features.csv exists, copying and using")
        #     # shutil.copyfile(input_file_without_ext + ".with_features.csv", cached)
        #     with_features = pd.read_csv(input_file_without_ext + ".with_features.csv")

        log(f"No cached file {cached}, rebuilding")
        return regenerate()


def pipeline(log, input_file: str):
    mne.use_log_level("warning")

    # Load MNE
    log("Loading MNE file " + input_file)
    raw, input_file_without_ext, mne_filtered = convert.load_mne_file(log, input_file)

    channels = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    start_date = raw.info['meas_date']


    log(f"Start date: {start_date} channels: {channels} sfreq: {sfreq}")

    # Hardcoded fixes for some borked files (may not be needed anymore)
    if (start_date.year < 2000):
        date_time_str = os.path.basename(os.path.dirname(input_file))

        # Parse the date and time string into a datetime object
        date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d-%H-%M-%S')

        # Set the timezone to UK time
        uk_timezone = pytz.timezone('Europe/London')
        date_time_uk = uk_timezone.localize(date_time_obj)
        date_time_utc = date_time_uk.astimezone(timezone.utc)

        log(f"Have tried to fix broken startdate in {input_file} from {start_date} to {date_time_utc}")
        start_date = date_time_utc
        mne_filtered.set_meas_date(start_date)
        raw.set_meas_date(start_date)

    # Save as EDF
    garbage_collect(log)
    log("Saving as EDF")
    convert.save_mne_as_downsample_edf(log, mne_filtered, input_file_without_ext)


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

    # YASA proper
    garbage_collect(log)
    log("Running YASA")
    yasa_copy, json_out = run_yasa.run_yasa_report(log, input_file_without_ext, raw, False)

    return yasa_copy


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