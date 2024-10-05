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

logging.getLogger('yasa').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
warnings.filterwarnings("ignore", message="Channel locations not available. Disabling spatial colors.")
warnings.filterwarnings("ignore", message="WARNING - Hypnogram is SHORTER than data")
mne.set_log_level('ERROR')

force_if_older_than = datetime(2024, 9, 21, 15, 0, 0)

def cached_pipeline(log, input_file: str, stats_df: pd.DataFrame):
    input_file_without_ext = os.path.splitext(input_file)[0]
    cached = input_file_without_ext + ".with_features.csv"

    if os.path.exists(cached):
        log("Loading cached file " + cached)
        out = pd.read_csv(cached)

        # Most recent files
        microwakings_exist = os.path.exists(input_file_without_ext + ".microwakings.csv")
        json_exist = os.path.exists(input_file_without_ext + ".sleep.json")

        modification_time = os.path.getmtime(cached)
        modification_date = datetime.fromtimestamp(modification_time)

        # Check for most recently added column - skipping as we are not using that model currently
        # if 'Predictions_Noise' not in out.columns:
        #     log("Cached file " + cached + " is missing recent columns, rebuilding")
        #     return pipeline(log, input_file)
        # el

        if modification_date < force_if_older_than:
            log("Cached file " + cached + f" mod date {modification_date} is < {force_if_older_than}, rebuilding")
            return pipeline(log, input_file, stats_df)
        if not any(col.endswith("_s") for col in out.columns):
            log("Cached file " + cached + " is missing scaled features, rebuilding")
            return pipeline(log, input_file, stats_df)
        if not any('svdent' in col for col in out.columns):
            log("Cached file " + cached + " is missing svdent, rebuilding")
            return pipeline(log, input_file, stats_df)
        if not any('eeg_auc' in col for col in out.columns):
            log("Cached file " + cached + " is missing eeg_auc, rebuilding")
            return pipeline(log, input_file, stats_df)
        if not microwakings_exist:
            log("No microwakings, rebuilding")
            return pipeline(log, input_file, stats_df)
        if not json_exist:
            log("No sleep.json, rebuilding")
            return pipeline(log, input_file, stats_df)

        out['epoch'] = out['Epoch']
        out.set_index('epoch', inplace=True)
        return out
    else:
        log(f"No cached file {cached}, rebuilding")
        return pipeline(log, input_file, stats_df)


def pipeline(log, input_file: str, stats_df: pd.DataFrame):
    # Load MNE
    log("Loading MNE file " + input_file)
    raw, input_file_without_ext, mne_filtered = convert.load_mne_file(log, input_file)

    channels = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    start_date = raw.info['meas_date']

    log(f"Start date: {start_date} channels: {channels} sfreq: {sfreq}")

    # Hardcoded fixes for some borked files
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
    end_date = start_date + timedelta(seconds=float(raw.times[-1]))

    # Save as EDF
    garbage_collect(log)
    log("Saving as EDF")
    convert.save_mne_as_downsample_edf(log, mne_filtered, input_file_without_ext)

    # Sleep events
    garbage_collect(log)
    log("Loading sleep events")
    ha_events = sleep_events.load_sleep_events(log, start_date, end_date)
    output_csv_file = input_file_without_ext + ".night_events.csv"
    ha_events.to_csv(output_csv_file, index=False)

    # YASA features
    garbage_collect(log)
    log("Extracting YASA features")
    yasa_feats, channel_feats_dict = yasa_features.extract_yasa_features2(log, channels, mne_filtered)

    # Scaled
    scale_by_stats = scaling.scale_by_stats(yasa_feats, stats_df)
    yasa_feats = yasa_feats.join(scale_by_stats.add_suffix('_s'))

    # YASA slow waves
    garbage_collect(log)
    log("Detecting slow waves")
    sw = sw_detect(mne_filtered, sfreq)
    if sw is not None:
        sw_summary = sw.summary()
        output_csv_file = input_file_without_ext + ".sw_summary.csv"
        sw_summary.to_csv(output_csv_file, index=False)

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

    # Combine epochs and YASA features
    garbage_collect(log)
    df = yasa_copy.copy()
    df['epoch'] = df['Epoch']
    df.set_index('epoch', inplace=True)
    combined_df = df.join(yasa_feats)

    # Main channel
    scaling.add_main_channel(combined_df)

    # Automated waking scoring
    # We're training a model to more accurately predict waking than YASA.  So we have to be judicious in what YASA data we use - while being aware that manually scoring waking is challenging.  So only use data where YASA is supremely confident in wakefulness.
    garbage_collect(log)
    log("Automated waking scoring")
    df_probably_awake = wakings.get_yasa_probably_awake(log, combined_df)

    # Manual waking scoring
    garbage_collect(log)
    log("Manual waking scoring")
    df_definitely_awake = wakings.get_definitely_awake(df_probably_awake, ha_events)

    # Combine probably and definitely awake
    df_combined_awake = df_probably_awake.copy()
    df_combined_awake['DefinitelyAwake'] = df_definitely_awake['DefinitelyAwake']
    df_combined_awake['ProbablyAwake'] = (df_combined_awake['DefinitelyAwake'] == True) | (df_combined_awake['YASAProbablyAwake'] == True)

    # Epochs that are probably sleep
    garbage_collect(log)
    log("Epochs that are probably sleep")
    df_asleep = sleep.probably_asleep(df_combined_awake)

    # Remvoe when YASA-esque model restored
    output_csv_file = input_file_without_ext + ".with_features.csv"
    df_asleep.to_csv(output_csv_file, index=False)

    # Run current best YASAesque model
    # Skipping as seems to require Fpz
    # log("Running YASAesque model")
    # df_with_predictions = best_model.run_model(df_asleep)
    # output_csv_file = input_file_without_ext + ".with_features.csv"
    # df_with_predictions.to_csv(output_csv_file, index=False)

    # Run current best microwakings model
    garbage_collect(log)
    log("Running microwakings model")
    try:
        microwakings_model = microwakings1.load_model()
        pf = PerFile(None, mne_filtered, yasa_copy, input_file_without_ext)
        pf.prepare_model_data(microwakings1.RESAMPLING_RATE, False)
        microwakings1.predict_file(log, microwakings_model, pf)
    except Exception as e:
        log("Error running microwakings model: " + str(e))
        log(traceback.format_exc())
        raise e

    log("All done! " + input_file)

    return df_asleep




def combine_all_file(log, input_dir: str):
    errors = []
    dataframes = []

    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            input_file = os.path.join(root, dir_name, "raw.with_features.csv")
            try:
                log("Processing file: " + input_file)
                if os.path.exists(input_file):
                    df = pd.read_csv(input_file)
                    df['filename'] = input_file
                    dataframes.append(df)
            except Exception as e:
                log("Error processing file: " + input_dir)
                errors.append("Error processing file: " + input_file + " - " + str(e))
                log(e)

    # Concatenate all dataframes into a single dataframe
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    for err in errors:
        log(err)
    df = combined_df
    return df