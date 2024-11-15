import argparse
import os

import mne
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter

from memory import garbage_collect
import pyedflib


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

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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



def save_buffer_to_edf(buffer, channel_names, sfreq, filename):
    n_channels = buffer.shape[0]
    file = pyedflib.EdfWriter(filename, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)

    channel_info = []
    for ch in channel_names:
        channel_info.append({
            'label': ch,
            'dimension': 'uV',
            'sample_rate': sfreq,
            'physical_min': -100000,
            'physical_max': 100000,
            'digital_min': -32768,
            'digital_max': 32767,
            'transducer': '',
            'prefilter': ''
        })

    file.setSignalHeaders(channel_info)
    file.writeSamples(buffer)
    file.close()



def load_mne_file(log, input_file: str) -> (mne.io.Raw, str, mne.io.Raw):
    log(f"Reading file {input_file}")
    raw = mne.io.read_raw_fif(input_file, preload=True)

    # Fix data bug
    if 'Fpz' in raw.info['ch_names']:
        # channels[channels.index('Fpz')] = 'Fpz-M1'
        raw.rename_channels({'Fpz': 'Fpz-M1'}, verbose=True)

    log(f"Finished reading file {input_file}")
    input_file_without_ext = os.path.splitext(input_file)[0]
    mne_filtered = get_filtered_and_scaled_data(raw)
    return raw, input_file_without_ext, mne_filtered


# MNE is in volts.  Filter it and scale it to uV
def get_filtered_and_scaled_data(raw: mne.io.Raw) -> (mne.io.Raw, mne.io.Raw):
    filtered = raw.copy()

    # AASM recommendation
    filtered.filter(0.3, 35)

    filtered.notch_filter(freqs=[50,100])

    # Bit confused about this, something to do with MNE storing in volts.  But YASA complains it doesn't look uV if I don't do this.
    data = filtered.get_data(units=dict(eeg="uV")) / 1_000_000
    filtered._data = data

    return filtered
