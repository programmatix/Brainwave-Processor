import pandas as pd
import os
from datetime import timedelta

import pandas as pd
from yasa import sw_detect, spindles_detect

import convert
import models.manual_sleep_scoring_catboost_1.manual_sleep_scoring_catboost_1 as best_model
import run_yasa
import sleep_events
import wakings
import sleep
import yasa_features
from models.microwakings_1 import microwakings1
from models.microwakings_1.microwakings1 import PerFile
import tensorflow as tf
import mne
from memory import garbage_collect
import traceback


def cached_pipeline(log, input_file: str):
    input_file_without_ext = os.path.splitext(input_file)[0]
    cached = input_file_without_ext + ".with_features.csv"

    if os.path.exists(cached):
        log("Loading cached file " + cached)
        out = pd.read_csv(cached)

        # Most recent file
        microwakings_exist = os.path.exists(input_file_without_ext + ".microwakings.csv")

        # Check for most recently added column
        if 'Predictions_Noise' not in out.columns:
            log("Cached file " + cached + " is missing recent columns, rebuilding")
            return pipeline(log, input_file)
        elif not microwakings_exist:
            log("No microwakings, rebuilding")
            return pipeline(log, input_file)
        else:
            out['epoch'] = out['Epoch']
            out.set_index('epoch', inplace=True)
            return out
    else:
        return pipeline(log, input_file)


def pipeline(log, input_file: str, waking_start_time_tz = None, waking_end_time_tz = None):
    # Load MNE
    log("Loading MNE file " + input_file)
    raw, input_file_without_ext, mne_filtered = convert.load_mne_file(log, input_file)

    channels = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    start_date = raw.info['meas_date']
    end_date = start_date + timedelta(seconds=float(raw.times[-1]))

    # Save as EDF
    garbage_collect(log)
    log("Saving as EDF")
    convert.save_mne_as_downsample_edf(log, mne_filtered, input_file_without_ext)

    # Sleep events
    garbage_collect(log)
    log("Loading sleep events")
    ha_events = sleep_events.load_sleep_events(log, start_date, end_date, waking_start_time_tz, waking_end_time_tz)
    output_csv_file = input_file_without_ext + ".night_events.csv"
    ha_events.to_csv(output_csv_file, index=False)

    # YASA features
    garbage_collect(log)
    log("Extracting YASA features")
    yasa_feats, channel_feats_dict = yasa_features.extract_yasa_features2(log, channels, mne_filtered)

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
    yasa_copy, json_out = run_yasa.run_yasa_report(log, input_file_without_ext, raw, True)

    # Combine epochs and YASA features
    garbage_collect(log)
    df = yasa_copy.copy()
    df['epoch'] = df['Epoch']
    df.set_index('epoch', inplace=True)
    combined_df = df.join(yasa_feats)

    # Automated waking scoring
    # We're training a model to more accurately predict waking than YASA.  So we have to be judicious in what YASA data we use - while being aware that manually scoring waking is challenging.  So only use data where YASA is supremely confident in wakefulness.
    garbage_collect(log)
    log("Automated waking scoring")
    df_probably_awake = wakings.get_yasa_probably_awake(log, combined_df)

    # Manual waking scoring
    garbage_collect(log)
    log("Manual waking scoring")
    df_definitely_awake = wakings.get_definitely_awake(df_probably_awake, ha_events, waking_start_time_tz, waking_end_time_tz)

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
        microwakings_model = tf.keras.models.load_model('./models/microwakings_1/microwakings_multi1.h5')
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