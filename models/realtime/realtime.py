# Runs models in realtime
import os.path

import logging

import numpy as np
import mne
import pandas as pd

import scaling
from convert import get_filtered_and_scaled_data
import yasa_features
from catboost import CatBoostRegressor
from models.eeg_states.eeg_states_model import predict_only_tired_vs_wired_model_pipeline, \
    predict_only_day_energy_model_pipeline  # TiredVsWired model


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def mne_from_buffer(eeg_buffer: np.array, eeg_ch_names: [str], sfreq: float):
    ch_types = ['eeg']
    info = mne.create_info(ch_names=eeg_ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_buffer, info)
    filtered = get_filtered_and_scaled_data(raw)
    return raw, filtered


def run_models(eeg_buffer: np.array, eeg_ch_names: [str], sfreq: float, stats_df: pd.DataFrame, model_dir: str):
    log = lambda msg: logging.info(msg)

    # Convert to MNE
    raw, mne_filtered = mne_from_buffer(eeg_buffer, eeg_ch_names, sfreq)
    channels = raw.info['ch_names']

    # YASA features
    yasa_feats, channel_feats_dict = yasa_features.extract_yasa_features2(log, channels, mne_filtered)

    # Scale features
    scale_by_stats = scaling.scale_by_stats(yasa_feats, stats_df)
    yasa_df = yasa_feats.join(scale_by_stats.add_suffix('_s'))

    # Models work on 'Main' channel
    yasa_df.rename(columns=lambda x: x.replace(channels[0], 'Main') if x.startswith(channels[0]) else x, inplace=True)

    tired_vs_wired_prediction = run_tired_vs_wired(yasa_df, model_dir)
    day_energy_prediction = run_day_energy(yasa_df, model_dir)

    return {
        "tired_vs_wired": tired_vs_wired_prediction,
        "day_energy": day_energy_prediction,
    }

def run_tired_vs_wired(yasa_df: pd.DataFrame, model_dir: str) -> float:
    models_and_data = predict_only_tired_vs_wired_model_pipeline('ignored', yasa_df, True)
    model = CatBoostRegressor()
    model.load_model(model_dir + os.path.sep + "settling-tired-vs-wired-realtime_catboost_model.cbm")
    predictions = model.predict(models_and_data.X)
    return predictions[0]

def run_day_energy(yasa_df: pd.DataFrame, model_dir: str) -> float:
    models_and_data = predict_only_day_energy_model_pipeline('ignored', yasa_df, True)
    model = CatBoostRegressor()
    model.load_model(model_dir + os.path.sep + "day-energy-realtime_catboost_model.cbm")
    predictions = model.predict(models_and_data.X)
    return predictions[0]