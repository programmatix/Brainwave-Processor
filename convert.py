import argparse
import os

import mne
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter

from memory import garbage_collect
import run_yasa

def convert_and_save_brainflow_file(log, input_file: str, output_file: str, channels: list[str]):
    garbage_collect(log)
    board_id = BoardIds.CYTON_BOARD.value

    log(f"Reading Brainflow file {input_file}")
    restored_data = DataFilter.read_file(input_file)
    log("Finished reading Brainflow file")

    garbage_collect(log)

    restored_df = pd.DataFrame(np.transpose(restored_data))

    eeg_channels = BoardShim.get_eeg_channels(board_id)

    log(f"EEG channels: {eeg_channels}")

    selected_columns = [restored_df.columns[0]]
    num_eeg_channels_to_select = min(len(eeg_channels), len(channels))
    selected_eeg_channels = [eeg_channels[i] for i in range(num_eeg_channels_to_select)]
    selected_columns.extend(selected_eeg_channels)
    selected_columns.append(restored_df.columns[-2])
    idx_and_eeg_channels_and_timestamp = restored_df[selected_columns]

    column_names = ['sampleIdx'] + channels + ['timestamp']
    idx_and_eeg_channels_and_timestamp.columns = column_names
    idx_and_eeg_channels_and_timestamp['datetime'] = pd.to_datetime(idx_and_eeg_channels_and_timestamp['timestamp'],unit="s").dt.tz_localize('UTC')

    eeg_channels_only = idx_and_eeg_channels_and_timestamp[channels]


    # Brainflow Cyton data in uV, MNE expects V
    scaled = eeg_channels_only / 1_000_000

    ch_types = ['eeg'] * len(channels)

    initial_timestamp = idx_and_eeg_channels_and_timestamp['datetime'].iloc[0]
    log(f"Initial timestamp: {str(initial_timestamp)} from {idx_and_eeg_channels_and_timestamp['timestamp'].iloc[0]}")

    sfreq = BoardShim.get_sampling_rate(board_id)
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=ch_types)
    info.set_meas_date(initial_timestamp)
    toSave = mne.io.RawArray(np.transpose(scaled), info)

    log(f"Info {info}")

    garbage_collect(log)

    if output_file.endswith(".fif"):
        log(f"Saving to {output_file}")
        toSave.save(output_file, overwrite=True)
        return toSave
    elif output_file.endswith(".edf"):
        log(f"Saving to {output_file}")
        mne.export.export_raw(output_file, toSave, overwrite=True)
        return toSave
    else:
        raise Exception(f"Unknown file type {output_file}")

    garbage_collect(log)

    return toSave


def save_mne_as_downsample_edf(log, mne_filtered, input_file_without_ext):
    resampled = mne_filtered.copy()
    # 100 hz is very similar to 250 hz to naked eye.  50 gets too lossy.
    resampled.resample(100, npad="auto")

    mne.export.export_raw(input_file_without_ext + ".edf", resampled, overwrite=True)


def load_mne_file(log, input_file: str) -> (mne.io.Raw, str, mne.io.Raw):
    log(f"Reading file {input_file}")
    raw = mne.io.read_raw_fif(input_file, preload=True)
    log(f"Finished reading file {input_file}")
    input_file_without_ext = os.path.splitext(input_file)[0]
    mne_filtered = run_yasa.get_filtered_and_scaled_data(raw)
    return raw, input_file_without_ext, mne_filtered


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, help='Input file')
    parser.add_argument('-c', '--channels', type=str, nargs='+', help='Channels')
    args = parser.parse_args()
    convert_and_save_brainflow_file(print, args.input_file, args.channels)
