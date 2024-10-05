import json
import os

import pandas
from mne.io import Raw

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
import argparse
import mne
import yasa
import matplotlib.pyplot as plt
from functools import reduce
import warnings
from sklearn.exceptions import InconsistentVersionWarning

from convert import get_filtered_and_scaled_data
from yasa_helpers import sleep_stability, spindles, slow_waves, channel_comparison

from memory import garbage_collect

logging.getLogger('yasa').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
warnings.filterwarnings("ignore", message="Channel locations not available. Disabling spatial colors.")
warnings.filterwarnings("ignore", message="WARNING - Hypnogram is SHORTER than data")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
mne.set_log_level('ERROR')


def get_sleep_stages(filtered: Raw, channels: list[str], channel_name: str, sfreq: int):
    #channel_data = data[channels.index(channel_name)]


    # Slightly unclear whether to use filtered data or not, with parts of the YASA docs saying it's both not recommended but also optional.
    # In practice it only changes the results a bit.
    sls = yasa.SleepStaging(filtered, eeg_name=channel_name)
    y_pred = sls.predict()
    hypno_pred = yasa.hypno_str_to_int(y_pred) # Convert "W" to 0, "N1" to 1, etc

    confidence = sls.predict_proba().max(1)

    df_pred = pd.DataFrame({'Stage': y_pred, 'Confidence': confidence})

    df_pred['Epoch'] = range(len(df_pred))

    stability = sleep_stability(hypno_pred)

    return (df_pred, stability)


def load_mne_fif_and_run_yasa(log, input_file: str):
    log(f"Reading file {input_file}")
    raw = mne.io.read_raw_fif(input_file, preload=True)
    log(f"Finished reading file {input_file}")
    input_file_without_ext = os.path.splitext(input_file)[0]

    return run_yasa_report(log, input_file_without_ext, raw)


def plot_spectro(input_file_without_ext, data, raw, channels: list[str], sfreq: int, df: pd.DataFrame, channel_name: str):
    # Warmer colors indicate higher spectral power in this specific frequency band at this specific time for this channel. This kind of plot is very useful to quickly identify periods of NREM sleep (high power in frequencies below 5 Hz and spindle-related activity around ~14 Hz) and REM sleep (almost no power in frequencies below 5 Hz).
    hypno_pred = yasa.hypno_str_to_int(df['Stage'])
    hypno_up = yasa.hypno_upsample_to_data(hypno_pred, sf_hypno=1/30, data=raw)
    if channel_name is not None:
        yasa.plot_spectrogram(data[channels.index(channel_name)], sfreq, hypno_up).savefig(input_file_without_ext + f'.spectrogram.{channel_name}.png', dpi=300)
    else:
        yasa.plot_hypnogram(hypno_up, sfreq)


def run_yasa_report(log, input_file_without_ext: str, raw: Raw, skip: bool = False):
    garbage_collect(log)

    channels = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    start_date = raw.info['meas_date']

    raw.plot_psd(average=False).savefig(input_file_without_ext + '.pre_filter_psd_plot.png', dpi=300)

    filtered = get_filtered_and_scaled_data(raw)
    filtered.plot_psd(average=False).savefig(input_file_without_ext + '.post_filter_psd_plot.png', dpi=300)

    all_dfs = []
    json_out = {}
    json_out['Stability'] = {}

    for i, channel in enumerate(channels):
        df_pred_ch, stability_ch = get_sleep_stages(filtered, channels, channel, sfreq);
        json_out['Stability'][channel] = stability_ch
        out = input_file_without_ext + f'.sleep_stages.{channel}.csv'
        df_pred_ch['EpochTime'] = (df_pred_ch['Epoch'] * 30) + start_date.timestamp()
        df_pred_ch['Timestamp'] = pd.to_datetime(df_pred_ch['EpochTime'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/London')
        df_pred_ch.drop('EpochTime', axis=1, inplace=True)
        df_pred_ch.to_csv(out, index=False)
        df_pred_ch.drop('Timestamp', axis=1, inplace=True)
        plot_spectro(input_file_without_ext, filtered.get_data(units='uV'), raw, channels, sfreq, df_pred_ch, channel)
        df_pred_ch.columns = [f"{channel}_{col}" if col not in ['Epoch'] else col for col in df_pred_ch.columns]
        all_dfs.append(df_pred_ch)

    df = reduce(lambda left, right: pd.merge(left, right, on='Epoch', how='outer'), all_dfs)

    # Initialize new columns
    df['Stage'] = None
    df['Confidence'] = 0.0
    df['Source'] = None

    # Iterate through each row to find the channel with the highest confidence
    for index, row in df.iterrows():
        highest_confidence = 0
        selected_channel = None
        selected_stage = None
        for channel in channels:
            confidence_col = f"{channel}_Confidence"
            stage_col = f"{channel}_Stage"
            if row[confidence_col] > highest_confidence:
                highest_confidence = row[confidence_col]
                selected_channel = channel
                selected_stage = row[stage_col]
        df.at[index, 'Stage'] = selected_stage
        df.at[index, 'Confidence'] = highest_confidence
        df.at[index, 'Source'] = selected_channel

    df['EpochTime'] = (df['Epoch'] * 30) + start_date.timestamp()
    df['Timestamp'] = pd.to_datetime(df['EpochTime'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    df.drop('EpochTime', axis=1, inplace=True)
    df['StageInt'] = yasa.hypno_str_to_int(df['Stage']) # Convert "W" to 0, "N1" to 1, etc

    # Reorder df
    cols_to_start = ['Stage', 'Confidence', 'Epoch', 'Timestamp', 'Source']
    remaining_cols = [col for col in df.columns if col not in cols_to_start]
    new_col_order = cols_to_start + remaining_cols
    df = df[new_col_order]

    out = input_file_without_ext + f'.sleep_stages.csv'
    log(f"Writing to {out}")
    df.to_csv(out, index=False)

    if not skip:
        garbage_collect(log)
        log("Processing sleep statistics")
        try:
            json_out['Statistics'] = yasa.sleep_statistics(df['StageInt'], sf_hyp=1/30)
        except Exception as e:
            log("Failed getting statistics: " + str(e))
            pass

        garbage_collect(log)
        log("Processing sleep stability")

        try:
            json_out['Stability']['Aggregated'] = sleep_stability(df['StageInt'])
        except Exception as e:
            log("Failed getting sleep_stability: " + str(e))
            pass

        # Temporarily removing as it is very memory intensive and is OOMing on my Rasperberry Pi 4GB..
        # try:
        #     json_out['Spindles'] = spindles(filtered, input_file_without_ext)
        # except Exception as e:
        #     log("Failed getting spindles: " + str(e))
        #     pass

        garbage_collect(log)

        try:
            json_out['SlowWaves'] = slow_waves(filtered, input_file_without_ext)
        except Exception as e:
            log("Failed getting slow_waves: " + str(e))
            pass

        garbage_collect(log)

        try:
            json_out['ChannelAgreement'] = channel_comparison(df, channels)
        except Exception as e:
            log("Failed getting channel_comparison: " + str(e))
            pass

        try:
            with open(input_file_without_ext + '.sleep.json', 'w') as json_file:
                json.dump(json_out, json_file, indent=4)
        except Exception as e:
            log("Failed writing JSON: " + str(e))
            pass

    log("YASA all done!")

    return df, json_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    args = parser.parse_args()

    input_file = args.input_file
    load_mne_fif_and_run_yasa(lambda: print, input_file)
