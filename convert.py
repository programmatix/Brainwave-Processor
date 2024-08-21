import argparse
import os

import mne
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter

def convert_and_save_brainflow_file(log, input_file: str, output_file: str, channels: list[str]):
    board_id = BoardIds.CYTON_BOARD.value

    log(f"Reading file {input_file}")
    restored_data = DataFilter.read_file(input_file)
    log("Finished reading file")
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

    sfreq = BoardShim.get_sampling_rate(board_id)
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=ch_types)
    info.set_meas_date(initial_timestamp)
    toSave = mne.io.RawArray(np.transpose(scaled), info)

    log(f"Info {info}")

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

    return toSave


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, help='Input file')
    parser.add_argument('-c', '--channels', type=str, nargs='+', help='Channels')
    args = parser.parse_args()
    convert_and_save_brainflow_file(print, args.input_file, args.channels)
