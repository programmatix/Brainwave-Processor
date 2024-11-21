import pandas as pd
import os
import pytz
from datetime import timedelta, datetime

from yasa import sw_detect, spindles_detect
from models.eeg_states.eeg_states import process_row
from tqdm import tqdm

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
import io
import contextlib
from datetime import timedelta
import convert
import scaling
import yasa_features
from memory import garbage_collect
from catboost import CatBoostClassifier
from models.eeg_states.eeg_states_model import predict_only_day_energy_model_pipeline
from models.eeg_states.eeg_states import process_row
from models.eeg_states.eeg_states_model import day_energy_mapping
from tqdm import tqdm

logging.getLogger('yasa').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
warnings.filterwarnings("ignore", message="Channel locations not available. Disabling spatial colors.")
warnings.filterwarnings("ignore", message="WARNING - Hypnogram is SHORTER than data")
mne.set_log_level('ERROR')

force_if_older_than = datetime(2024, 11, 11, 13, 30, 0)

def cached_pipeline(log, input_file: str, stats_df: pd.DataFrame, events: pd.DataFrame):
    input_file_without_ext = os.path.splitext(input_file)[0]
    cached = input_file_without_ext + ".output.csv"

    def regenerate():
        out = pipeline(log, input_file, stats_df, events)
        return out

    if os.path.exists(cached):
        log("Loading cached file " + cached)
        out = pd.read_csv(cached)


        modification_time = os.path.getmtime(cached)
        modification_date = datetime.fromtimestamp(modification_time)

        if modification_date < force_if_older_than:
            log("Cached file " + cached + f" mod date {modification_date} is < {force_if_older_than}, rebuilding")
            return regenerate()
        if not any(col for col in out.columns if col == 'DayEnergyPrediction'):
            log("Cached file " + cached + " is missing DayEnergyPrediction, rebuilding")
            return regenerate()

        return out
    else:
        log(f"No cached file {cached}, rebuilding")
        return regenerate()


def pipeline(log, input_file: str, stats_df: pd.DataFrame, events: pd.DataFrame):
    # Load MNE
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
        log("Loading MNE file " + input_file)
        raw, input_file_without_ext, mne_filtered = convert.load_mne_file(log, input_file)

        channels = raw.info['ch_names']
        sfreq = raw.info['sfreq']
        start_date = raw.info['meas_date']
        duration = timedelta(seconds=float(raw.times[-1]))
        end_date = start_date + duration

        log(f"Start date: {start_date} end {end_date} duration {duration} channels: {channels} sfreq: {sfreq}")

        # Save as EDF
        garbage_collect(log)
        log("Saving as EDF")
        convert.save_mne_as_downsample_edf(log, mne_filtered, input_file_without_ext)

        # YASA features
        garbage_collect(log)
        log("Extracting YASA features")
        yasa_feats, channel_feats_dict = yasa_features.extract_yasa_features2(log, channels, mne_filtered)

        # Timestamps
        df = yasa_feats
        df['EpochTime'] = (df.index * 30) + start_date.timestamp()
        df['TimestampUK'] = pd.to_datetime(df['EpochTime'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/London')
        df.drop('EpochTime', axis=1, inplace=True)

        # Scale
        scale_by_stats = scaling.scale_by_stats(df, stats_df)
        df = df.join(scale_by_stats.add_suffix('_s'))

        # Main channel
        df['Source'] = channels[0]
        df = df.copy()
        scaling.add_main_channel(df)

        # Move columns
        columns = df.columns.tolist()
        columns.insert(0, columns.pop(columns.index('TimestampUK')))
        df = df.reindex(columns=columns)

        # DayEnergy model
        models_and_data = [predict_only_day_energy_model_pipeline('ignored', df, False)]
        model = CatBoostClassifier()
        model.load_model("models/day_data/day-energy-non-realtime_catboost_model.cbm")
        predictions = model.predict_proba(models_and_data[0].X)
        predictions_df = pd.DataFrame(predictions, index=models_and_data[0].X.index)
        predictions_df['DayEnergyPrediction'] = predictions_df[1]
        predictions_df.drop([0, 1], axis=1, inplace=True)
        df = pd.concat([df, predictions_df], axis=1)

        df['DayEnergyManual'] = None

        for i, yasa_row in tqdm(df.iterrows(), total=df.shape[0]):
            epoch_type, matched_night_event = process_row(yasa_row, events)
            if epoch_type is not None:
                df.at[i, 'DayEnergyManual'] = day_energy_mapping[epoch_type]

        # Save
        output_csv_file = input_file_without_ext + ".output.csv"
        df.to_csv(output_csv_file)

        log("All done! " + input_file)


