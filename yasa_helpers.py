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
from dataclasses import dataclass

from yasa import sw_detect


# Can get y_pred from
# df['StageInt']
#    sls = yasa.SleepStaging(filtered, eeg_name=channel_name)
#    y_pred = sls.predict()
#    yasa.hypno_str_to_int(y_pred)
# 0 = "W", 1 = "N1", 2 = "N2", 3 = "N3", 4 = "REM"
def sleep_stability(hypno):
    counts, probs = yasa.transition_matrix(hypno)

    sleep_stability_any_stage = np.diag(probs.loc[2:, 2:]).mean().round(3)
    sleep_stability_wake = probs.loc[0, 0].round(3)
    sleep_stability_n1 = probs.loc[1, 1].round(3)
    sleep_stability_n2 = probs.loc[2, 2].round(3)
    sleep_stability_deep = probs.loc[3, 3].round(3)
    sleep_stability_rem = probs.loc[4, 4].round(3)

    return {
        'AnyStage': sleep_stability_any_stage,
        'Wake': sleep_stability_wake,
        'N1': sleep_stability_n1,
        'N2': sleep_stability_n2,
        'De ep': sleep_stability_deep,
        'REM': sleep_stability_rem
    }


def spindles(filtered: Raw, input_file_without_ext: str):
    sfreq = filtered.info['sfreq']
    sp = yasa.spindles_detect(filtered, sfreq)
    summary = sp.summary(grp_chan=True, aggfunc='mean')

    out = {}
    selected_columns = summary.columns[2:]
    mean_dict = {col: summary[col].mean() for col in selected_columns}
    out['Aggregated'] = mean_dict

    for channel in summary.index:
        row = summary.loc[channel]
        selected_columns = summary.columns[2:]
        mean_dict = {col: row[col] for col in selected_columns}
        out[channel] = mean_dict
    axes = sp.plot_average(time_before=0.4, time_after=0.8, center="Peak")
    axes.get_figure().savefig(input_file_without_ext + f'.average_spindle.png', dpi=300)

    return out


def slow_waves(filtered: Raw, input_file_without_ext: str):
    sfreq = filtered.info['sfreq']
    sw = sw_detect(filtered, sfreq)
    summary = sw.summary(grp_chan=True, aggfunc='mean')

    out = {}
    selected_columns = summary.columns[2:]
    mean_dict = {col: summary[col].mean() for col in selected_columns}
    out['Aggregated'] = mean_dict

    for channel in summary.index:
        row = summary.loc[channel]
        selected_columns = summary.columns[2:]
        mean_dict = {col: row[col] for col in selected_columns}
        out[channel] = mean_dict

    axes = sw.plot_average(time_before=0.4, time_after=0.8, center="NegPeak")
    axes.get_figure().savefig(input_file_without_ext + f'.average_slow_wave.png', dpi=300)

    return out


# final_yasa_df is returned from run_yasa_report
def compare_confidence(final_yasa_df, channel1: str, channel2: str, thresholds=[0.05, 0.10, 0.20]):
    rows_where_do_not_agree = final_yasa_df[final_yasa_df[f'{channel1}_Stage'] != final_yasa_df[f'{channel2}_Stage']]
    results = {}
    for threshold in thresholds:
        count = len(rows_where_do_not_agree[rows_where_do_not_agree[f'{channel1}_Confidence'] > rows_where_do_not_agree[f'{channel2}_Confidence'] + threshold])
        results[threshold] = count
    total_stages = len(final_yasa_df)

    output = []
    output.append(f"Number of stages: {total_stages}")
    output.append(f"Stages where {channel1} and {channel2} do not agree: {len(rows_where_do_not_agree)} {len(rows_where_do_not_agree) / total_stages * 100:.2f}%")
    output.append("Of those stages of non-agreement:")
    for threshold, count in results.items():
        output.append(f"{channel1} {threshold*100:.0f}% more confident than {channel2}: {count} rows {count / total_stages * 100:.2f}%")

    return "\n".join(output)

def compare_confidence_with_third_channel(df, channel1, channel2, channel3, thresholds=[0.05, 0.10, 0.20]):
    rows_where_do_not_agree = df[df[f'{channel1}_Stage'] != df[f'{channel2}_Stage']]
    results = {}
    for threshold in thresholds:
        count = len(rows_where_do_not_agree[rows_where_do_not_agree[f'{channel1}_Confidence'] > rows_where_do_not_agree[f'{channel2}_Confidence'] + threshold])
        agree_with_third = len(rows_where_do_not_agree[(rows_where_do_not_agree[f'{channel1}_Confidence'] > rows_where_do_not_agree[f'{channel2}_Confidence'] + threshold) & (rows_where_do_not_agree[f'{channel1}_Stage'] == rows_where_do_not_agree[f'{channel3}_Stage'])])
        results[threshold] = (count, agree_with_third)

    agree_channel1_channel3 = df[(df[f'{channel1}_Stage'] == df[f'{channel3}_Stage']) & (df[f'{channel1}_Stage'] != df[f'{channel2}_Stage'])]
    disagree_channel1_channel3 = len(df[(df[f'{channel1}_Stage'] == df[f'{channel3}_Stage']) & (df[f'{channel1}_Stage'] == df[f'{channel2}_Stage'])])
    agree_breakdown = {}
    for threshold in thresholds:
        count_highest_confidence = len(agree_channel1_channel3[(agree_channel1_channel3[f'{channel1}_Confidence'] > agree_channel1_channel3[f'{channel2}_Confidence'] + threshold) | (agree_channel1_channel3[f'{channel3}_Confidence'] > agree_channel1_channel3[f'{channel2}_Confidence'] + threshold)])
        agree_breakdown[threshold] = count_highest_confidence

    total_stages = len(df)

    output = []
    output.append(f"Number of stages: {total_stages}")
    output.append(f"Stages where {channel1} and {channel2} do not agree: {len(rows_where_do_not_agree)} {len(rows_where_do_not_agree) / total_stages * 100:.2f}%")
    output.append("Of those stages of non-agreement:")
    for threshold, (count, agree_with_third) in results.items():
        output.append(f"{channel1} {threshold*100:.0f}% more confident than {channel2}: {count} rows {count / total_stages * 100:.2f}%")
        output.append(f"  Of these, {agree_with_third} rows agree with {channel3} {agree_with_third / total_stages * 100:.2f}%")
    output.append(f"Stages where {channel1} and {channel3} agree and disagree with {channel2}: {len(agree_channel1_channel3)} rows {len(agree_channel1_channel3) / total_stages * 100:.2f}%")
    for threshold, count_highest_confidence in agree_breakdown.items():
        output.append(f"  Whether either {channel1} or {channel3} {threshold*100:.0f}% more confident than {channel2}: {count_highest_confidence} rows {count_highest_confidence / total_stages * 100:.2f}%")
    output.append(f"Stages where {channel1} and {channel3} agree and agree with {channel2}: {disagree_channel1_channel3} rows {disagree_channel1_channel3 / total_stages * 100:.2f}%")

    return "\n".join(output)


def channel_comparison(df, channels):
    from itertools import permutations

    json_out = {}

    if len(channels) >= 3:
        for perm in permutations(channels, 3):
            json_out['-'.join(perm)] = compare_confidence_with_third_channel(df, perm[0], perm[1], perm[2])
    elif (len(channels) == 2):
        json_out[channels[0] + "-" + channels[1]] = compare_confidence(df, channels[0], channels[1])
        json_out[channels[1] + "-" + channels[0]] = compare_confidence(df, channels[1], channels[0])

    return json_out
