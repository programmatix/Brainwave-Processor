from datetime import datetime, timedelta

import argparse
import os

import mne
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter
from tqdm import tqdm

from memory import garbage_collect
import pyedflib


from sample_rates import real_sample_rate

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


# See https://dynalist.io/d/E7zuJKws_QSGJyv1HMR9-AsC#z=ASBieidwCIeZU6IxZKH-BErM
def remove_out_of_order_samples(df):
    df = df.copy()

    # Identify wraparound behavior
    wrap_point = 255  # Assuming sampleIdx wraps around to 0 after this value

    # Initialize variables for tracking
    valid_samples = []
    last_valid = None

    # Iterate through the DataFrame
    for i, sample in tqdm(enumerate(df['sampleIdx']), desc="Processing rows", total=len(df)):
        if last_valid is None:
            # First sample is always valid
            valid_samples.append(i)
            last_valid = sample
            continue

        # Calculate expected next sample considering wraparound
        expected_next = (last_valid + 1) % (wrap_point + 1)

        if sample == expected_next:
            # Sample is in the correct order
            valid_samples.append(i)
            last_valid = sample
        else:
            # Sequence is broken
            pass

    # Filter the DataFrame to include only valid rows
    df = df.iloc[valid_samples].reset_index(drop=True)
    return df

def gap_fill(df, channels, sfreq):
    # Identify gaps greater than 500 ms
    # Why 500ms?  Because there are very regular gaps of ~430ms.
    gaps = df['datetime'].diff() > pd.Timedelta('500ms')

    print("Gaps: ", gaps.value_counts())

    # Create a list to hold new rows
    new_rows = []

    # Iterate over the DataFrame to find gaps and create new rows
    for i in tqdm(range(1, len(df)), desc="Processing rows"):
        if gaps.iloc[i]:
            start_time = df['datetime'].iloc[i - 1]
            end_time = df['datetime'].iloc[i]
            print(f"Gap {i} start time: ", start_time)
            print(f"Gap {i} end time: ", end_time)
            while start_time < end_time:
                start_time += pd.Timedelta((1 / sfreq), unit='s')
                new_row = {}
                for c in channels:
                    new_row[c] = 0
                new_row['datetime'] = start_time
                new_row['Inserted'] = 1
                new_row['sampleIdx'] = -999
                new_rows.append(new_row)

    # Append new rows to the original DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    df['Inserted'] = 0
    copied = pd.concat([df, new_rows_df], ignore_index=True)

    # Sort the DataFrame by the 'datetime' column
    copied = copied.sort_values(by='datetime').reset_index(drop=True)

    # Fill NaN values with empty values
    copied.fillna('', inplace=True)

    return copied


def convert_and_save_brainflow_file_with_gap_filling(log, input_file: str, output_file: str, channels: list[str], sfreq: float = real_sample_rate):
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


    idx_and_eeg_channels_and_timestamp = remove_out_of_order_samples(idx_and_eeg_channels_and_timestamp)
    idx_and_eeg_channels_and_timestamp = gap_fill(idx_and_eeg_channels_and_timestamp, channels, sfreq)


    eeg_channels_only = idx_and_eeg_channels_and_timestamp[channels]


    # Brainflow Cyton data in uV, MNE expects V
    scaled = eeg_channels_only / 1_000_000

    ch_types = ['eeg'] * len(channels)

    initial_timestamp = idx_and_eeg_channels_and_timestamp['datetime'].iloc[0]
    log(f"Initial timestamp: {str(initial_timestamp)} from {idx_and_eeg_channels_and_timestamp['timestamp'].iloc[0]}")

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

def save_mne_as_edf(log, mne_filtered, input_file_without_ext):
    # Avoid this problem after fixing sample rate:
    #  Signal duration of 31863.802168624s is not exactly divisible by data_record_duration of 0.998339s

    # Make a copy of the data to avoid modifying the original
    raw_copy = mne_filtered.copy()
    
    # Calculate the number of samples needed for exact divisibility
    duration = raw_copy.times[-1]
    record_duration = 1.0  # Use a standard record duration of 1 second
    
    # Calculate how many samples we need for whole seconds
    needed_duration = np.ceil(duration)
    current_samples = raw_copy.n_times
    target_samples = int(needed_duration * raw_copy.info['sfreq'])
    
    # Pad with zeros if needed to get an exact number of seconds
    if target_samples > current_samples:
        pad_samples = target_samples - current_samples
        pad_seconds = pad_samples / raw_copy.info['sfreq']
        log(f"Padding EDF with {pad_samples} samples ({pad_seconds:.4f} seconds)")
        padding = np.zeros((raw_copy.info['nchan'], pad_samples))
        raw_copy._data = np.hstack((raw_copy._data, padding))
    else:
        log("No padding needed for EDF export")
    
    # Export using the fixed data
    log(f"Saving EDF with standardized record duration of {record_duration}s")
    mne.export.export_raw(input_file_without_ext + ".edf", raw_copy, 
                          overwrite=True, data_record_dur=record_duration)


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
    # Note that yasa_features & YASA do this also
    # Clean signal tips: use a USB extension cable (v important!)
    # Changed to 0.5 to 35 2025-01-08
    filtered.filter(0.5, 35, verbose=False)

    # Remove power (probably unnecessary since we already bandstop at 35)
    filtered.notch_filter(freqs=[50, 100], verbose=False)

    start_date = raw.info['meas_date']
    if start_date.date() >= datetime(2024, 11, 20).date() and start_date.date() < datetime(2024, 11, 27).date():
        # Remove the spikes introduced by electric bed!
        # From 27th started only using it for one hour.  Assuming I'll be asleep after it kicks off.
        # If/when I look at the data for settling - I'll need to fix it, for that period it's on.
        filtered.notch_filter(freqs=[16.6, 27.7, 38.8], verbose=False)

    # Bit confused about this, something to do with MNE storing in volts.  But YASA complains it doesn't look uV if I don't do this.
    data = filtered.get_data(units=dict(eeg="uV")) / 1_000_000
    filtered._data = data

    return filtered


import json
import numpy as np
import os
import struct

def save_mne_to_binary_format(mne_filtered, output_base_path):
    data = mne_filtered.get_data(units=dict(eeg="uV"))
    channel_names = mne_filtered.info['ch_names']
    sfreq = mne_filtered.info['sfreq']
    meas_date = mne_filtered.info['meas_date']
    save_recording_to_binary_format(data, channel_names, sfreq, output_base_path, meas_date)


def save_recording_to_binary_format(data, channel_names, sfreq, output_base_path, meas_date=None):
    """
    Save recording data to a binary format with a JSON file for metadata.
    
    Parameters:
    -----------
    data : numpy.ndarray
        EEG data with shape (n_channels, n_samples)
    channel_names : list
        List of channel names
    sfreq : float
        Sampling frequency in Hz
    output_base_path : str
        Base path for output files (without extension)
    meas_date : datetime or None
        Measurement start date/time
    """
    # Create metadata
    metadata = {
        "sampling_frequency": float(sfreq),
        "channels": channel_names,
        "n_channels": len(channel_names),
        "n_samples": data.shape[1],
        "data_format": "int16",
        "byte_order": "little-endian"
    }
    
    # Add timing information
    if meas_date is not None:
        start_time = meas_date.isoformat()
        end_time = (meas_date + timedelta(seconds=data.shape[1]/sfreq)).isoformat()
        metadata["start_time"] = start_time
        metadata["end_time"] = end_time
    
    # Save metadata to JSON
    json_path = f"{output_base_path}.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Prepare binary data file
    bin_path = f"{output_base_path}.bin"
    
    # Convert data to int16 (2 bytes per sample) and clip values
    int16_data = np.clip(data, -32768, 32767).astype(np.int16)
    
    # Write binary data (samples are stored channel by channel)
    with open(bin_path, 'wb') as f:
        int16_data.tofile(f)
    
    print(f"Saved metadata to {json_path}")
    print(f"Saved recording data to {bin_path}")
    print(f"File size: {os.path.getsize(bin_path) / (1024*1024):.2f} MB")
    
    return json_path, bin_path

def read_binary_recording(metadata_path):
    """
    Read recording data from binary format and metadata JSON.
    
    Parameters:
    -----------
    metadata_path : str
        Path to the JSON metadata file
    
    Returns:
    --------
    data : numpy.ndarray
        EEG data with shape (n_channels, n_samples)
    metadata : dict
        Recording metadata
    """
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Derive binary data path
    bin_path = metadata_path.replace('.json', '.bin')
    
    # Read binary data
    data = np.fromfile(bin_path, dtype=np.int16)
    
    # Reshape data based on metadata
    data = data.reshape(metadata["n_channels"], -1)
    
    return data, metadata
