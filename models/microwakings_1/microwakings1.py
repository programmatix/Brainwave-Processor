from typing import Optional

from datetime import timedelta

import mne
import numpy
import pandas as pd
import json
from pandas import DataFrame

# Training time is horrendous, trying to reduce
RESAMPLING_RATE = 10

# Weak areas to improve
# 08-18
# 01:53:21Z (tails off too soon)
# 01:56:36Z (tails off too soon)


class PerFile:
    # What channel came up most often in the microwakings file.  We will only train with this.
    most_frequent_channel: str = None
    microwakings_df: DataFrame = None
    eeg_data: numpy.ndarray = None
    X: numpy.ndarray = None
    y: numpy.ndarray = None

    def __init__(self, marks: Optional[DataFrame], mne_filtered: mne.io.Raw, yasa_df: DataFrame,
                 input_file_without_ext: str):
        self.marks = marks
        self.mne_filtered = mne_filtered
        self.yasa_df = yasa_df
        self.input_file_without_ext = input_file_without_ext
        # "2024-08-22"
        self.day_or_night_of = '-'.join(input_file_without_ext.split('\\')[-2].split('-')[0:3])
        self.start_time = mne_filtered.info['meas_date']
        self.sfreq = mne_filtered.info['sfreq']
        self.end_time = self.start_time + timedelta(seconds=float(mne_filtered.times[-1]))

    def prepare_microwakings(self):
        microwakings = self.marks.copy()
        microwakings['timestamp'] = pd.to_datetime(microwakings['timestamp'], format='ISO8601')
        microwakings['scoredAt'] = pd.to_datetime(microwakings['scoredAt'], format='ISO8601')
        microwakings['type'] = microwakings['type'].astype(str)
        microwakings['channel'] = microwakings['channel'].astype(str)

        # Filter the DataFrame to only include rows with the most frequent channel
        self.most_frequent_channel = microwakings['channel'].value_counts().idxmax()
        microwakings = microwakings[microwakings['channel'] == self.most_frequent_channel]

        # Initialize an empty list to store matched microwakings
        matched_microwakings = []

        # Loop through the DataFrame to find matching MicrowakingStart and MicrowakingEnd
        for i, start_row in microwakings[microwakings['type'] == 'MicrowakingStart'].iterrows():
            start_time = start_row['timestamp']
            for j, end_row in microwakings[microwakings['type'] == 'MicrowakingEnd'].iterrows():
                end_time = end_row['timestamp']
                if start_time <= end_time <= start_time + pd.Timedelta(minutes=2):
                    matched_microwakings.append((start_time, end_time))
                    break  # Assuming one-to-one matching

        microwakings_df = pd.DataFrame(matched_microwakings, columns=['Start', 'End'])
        microwakings_df['Duration'] = microwakings_df['End'] - microwakings_df['Start']

        self.microwakings_df = microwakings_df
        return microwakings_df

    def prepare_model_data(self, resampled_rate: int, remove_wake_epochs: bool):
        resampled = self.mne_filtered.copy()
        resampled.resample(resampled_rate, npad="auto")
        if self.most_frequent_channel is None:
            self.most_frequent_channel = resampled.info['ch_names'][0]  # Get the first EEG channel

        self.eeg_data = resampled.get_data(picks=self.most_frequent_channel, units=dict(eeg="uV"))
        num_samples = self.eeg_data.shape[1]

        self.yasa_df['Timestamp'] = pd.to_datetime(self.yasa_df['Timestamp'], format='ISO8601')
        timestamps = pd.date_range(start=self.start_time, periods=num_samples, freq=pd.Timedelta(seconds=1/resampled_rate))
        samples = pd.DataFrame({'Timestamp': timestamps, 'Microwaking': 0, 'Epoch': None})
        eeg_df = pd.DataFrame(self.eeg_data.T, columns=[f'EEG_{i}' for i in range(self.eeg_data.shape[0])])
        eeg_df['Timestamp'] = pd.date_range(start=self.start_time, periods=self.eeg_data.shape[1], freq=pd.Timedelta(seconds=1/resampled_rate))
        samples = pd.merge(samples, eeg_df, on='Timestamp', how='left')

        for _, row in self.yasa_df.iterrows():
            epoch_start = row['Timestamp']
            epoch_end = epoch_start + pd.Timedelta(seconds=30)
            location_condition = (samples['Timestamp'] >= epoch_start) & (samples['Timestamp'] < epoch_end)
            samples.loc[location_condition, 'Epoch'] = row['Epoch']
            samples.loc[location_condition, 'Epoch'] = row['Epoch']

        if self.microwakings_df is not None:
            for _, row in self.microwakings_df.iterrows():
                samples.loc[(samples['Timestamp'] >= row['Start']) & (samples['Timestamp'] <= row['End']), 'Microwaking'] = 1

        epoch_to_stage = dict(zip(self.yasa_df['Epoch'], self.yasa_df['Stage']))
        samples['Stage'] = samples['Epoch'].map(epoch_to_stage)

        if remove_wake_epochs:
            samples = samples[samples['Stage'] != 'W']

        self.X = samples['EEG_0'].to_numpy()

        if self.microwakings_df is not None:
            self.y = samples['Microwaking'].to_numpy()

        return samples

def load_scoring_file(scoring_file_path: str):
    with open(scoring_file_path, 'r') as file:
        data = json.load(file)
    df = pd.json_normalize(data, 'marks', errors='ignore')
    return df



from models.microwakings_1.microwakings1 import RESAMPLING_RATE
import pandas as pd
import numpy as np
import mne
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import Sequence


class FlexibleDaySequence(Sequence):
    def __init__(self, out_files):
        self.out_files = out_files
        self.total_samples = sum(file.X.shape[0] for file in out_files)

    def __len__(self):
        return 1  # We'll process all data in one batch

    def __getitem__(self, idx):
        X_list = []
        for file in self.out_files:
            X_list.append(file.X.reshape(-1, 1))
        X_out = np.concatenate(X_list, axis=0)
        return X_out


def predict_file(log, model, f: PerFile):
    existing_data = f.eeg_data

    day_sequence_pred = FlexibleDaySequence([f])
    y_pred_proba = model.predict(day_sequence_pred).flatten()
    y_pred_binary = (y_pred_proba > 0.5).astype(int)

    if f.y is not None:
        y_true_flat = f.y

        manual_accuracy = accuracy_score(y_true_flat, y_pred_binary)
        manual_precision = precision_score(y_true_flat, y_pred_binary)
        manual_recall = recall_score(y_true_flat, y_pred_binary)
        manual_auc = roc_auc_score(y_true_flat, y_pred_proba)

        print("Manually calculated metrics:")
        print(f"Accuracy: {manual_accuracy:.4f}")
        print(f"Precision: {manual_precision:.4f}")
        print(f"Recall: {manual_recall:.4f}")
        print(f"AUC: {manual_auc:.4f}")

        cm = confusion_matrix(y_true_flat, y_pred_binary)
        print("\nConfusion Matrix:")
        print(cm)

        # Additional analysis
        print("\nPrediction Statistics:")
        print(f"Min prediction: {y_pred_proba.min():.4f}")
        print(f"Max prediction: {y_pred_proba.max():.4f}")
        print(f"Mean prediction: {y_pred_proba.mean():.4f}")
        print(f"Median prediction: {np.median(y_pred_proba):.4f}")

    high = 400_000_000
    low = 200_000_000
    probabilities_channel = y_pred_proba.reshape(1, -1) * high

    def smooth_predictions(predictions, window_length=11, polyorder=2, moving_avg_window=5):
        # Apply Savitzky-Golay filter
        smoothed = savgol_filter(predictions, window_length, polyorder)
        # Apply moving average
        smoothed = np.convolve(smoothed, np.ones(moving_avg_window)/moving_avg_window, mode='same')
        return smoothed

    smoothed = smooth_predictions(probabilities_channel[0])



    def binary_split_with_values(predictions, threshold=low, high_value=high, low_value=0):
        return np.where(predictions > threshold, high_value, low_value)

    def fill_gaps_with_values(predictions, sample_rate, gap_duration=2, high_value=high, low_value=0):
        gap_samples = gap_duration * sample_rate
        filled_predictions = predictions.copy()

        zero_streak_start = None
        for i in range(len(predictions)):
            if predictions[i] == low_value:
                if zero_streak_start is None:
                    zero_streak_start = i
            else:
                if zero_streak_start is not None:
                    if i - zero_streak_start <= gap_samples:
                        filled_predictions[zero_streak_start:i] = high_value
                    zero_streak_start = None

        # Handle case where the streak goes till the end
        if zero_streak_start is not None and len(predictions) - zero_streak_start <= gap_samples:
            filled_predictions[zero_streak_start:] = high_value

        return filled_predictions

    def remove_short_periods(predictions, sample_rate, period_duration=2, high_value=high, low_value=0):
        period_samples = period_duration * sample_rate
        cleaned_predictions = predictions.copy()

        high_streak_start = None
        for i in range(len(predictions)):
            if predictions[i] == high_value:
                if high_streak_start is None:
                    high_streak_start = i
            else:
                if high_streak_start is not None:
                    if i - high_streak_start < period_samples:
                        cleaned_predictions[high_streak_start:i] = low_value
                    high_streak_start = None

        # Handle case where the streak goes till the end
        if high_streak_start is not None and len(predictions) - high_streak_start < period_samples:
            cleaned_predictions[high_streak_start:] = low_value

        return cleaned_predictions

    def process_predictions(predictions, threshold=low, sample_rate=10, gap_duration=2, period_duration=2, high_value=high, low_value=0):
        binary_predictions = binary_split_with_values(predictions, threshold, high_value, low_value)
        cleaned_initial_predictions = remove_short_periods(binary_predictions, sample_rate, 0.5, high_value, low_value)
        filled_predictions = fill_gaps_with_values(cleaned_initial_predictions, sample_rate, gap_duration, high_value, low_value)
        cleaned_predictions = remove_short_periods(filled_predictions, sample_rate, period_duration, high_value, low_value)
        return cleaned_predictions

    binary_predictions = binary_split_with_values(probabilities_channel[0])
    cleaned_initial_predictions = remove_short_periods(binary_predictions, 10, 0.5, high, 0)
    filled_predictions = fill_gaps_with_values(cleaned_initial_predictions, 10, 2, high, 0)
    cleaned_predictions = remove_short_periods(filled_predictions, 10, 2, high, 0)
    processed_predictions = process_predictions(probabilities_channel[0])

    processed_predictions_0_1 = np.where(processed_predictions > low, 1, 0)

    def find_blocks_of_ones(predictions, timestamps):
        blocks = []
        in_block = False
        start_time = None

        for i, value in enumerate(predictions):
            if value == 1 and not in_block:
                in_block = True
                start_time = timestamps[i]
            elif value == 0 and in_block:
                in_block = False
                end_time = timestamps[i - 1]
                blocks.append((start_time, end_time))

        # Handle case where the last value is 1
        if in_block:
            end_time = timestamps[-1]
            blocks.append((start_time, end_time))

        return pd.DataFrame(blocks, columns=['Start', 'End'])

    num_samples = existing_data.shape[1]
    timestamps = pd.date_range(start=f.start_time, periods=num_samples, freq=pd.Timedelta(seconds=1/RESAMPLING_RATE))
    blocks_df = find_blocks_of_ones(processed_predictions_0_1, timestamps)

    output_csv_file = f.input_file_without_ext + ".microwakings.csv"
    print(f"Saved microwakings to: {output_csv_file}")
    blocks_df.to_csv(output_csv_file, index=False)

    if f.y is not None:
        y_true_flat = f.y
        manual_accuracy = accuracy_score(y_true_flat, processed_predictions_0_1)
        manual_precision = precision_score(y_true_flat, processed_predictions_0_1)
        manual_recall = recall_score(y_true_flat, processed_predictions_0_1)
        cm = confusion_matrix(y_true_flat, processed_predictions_0_1)

        print(f"After processing")
        print(f"Accuracy: {manual_accuracy:.4f}")
        print(f"Precision: {manual_precision:.4f}")
        print(f"Recall: {manual_recall:.4f}")
        print("\nConfusion Matrix:")
        print(cm)

    new_data = np.vstack([f.X, processed_predictions, probabilities_channel, binary_predictions, cleaned_initial_predictions, filled_predictions, cleaned_predictions])

    new_info = mne.create_info(
        ch_names=[f.most_frequent_channel, 'final', 'raw', 'binary', 'cleaned1', 'filled', 'cleaned2'],
        sfreq=10,
        ch_types=['eeg', 'misc', 'misc', 'misc', 'misc', 'misc', 'misc']
    )

    assert new_data.shape[0] == len(new_info['ch_names'])

    new_data_scaled = new_data / 1_000_000

    new_raw = mne.io.RawArray(new_data_scaled, new_info)

    log(f"Exporting to: {f.input_file_without_ext}.with_microwakings_multi1.edf")
    mne.export.export_raw(f.input_file_without_ext + ".with_microwakings_multi1_debug.edf", new_raw, overwrite=True)
