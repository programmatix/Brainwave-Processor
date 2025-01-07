# To handle different channels having different scales, we scale data by all the data we have for a given channel-feature pair.


import pandas as pd
import tqdm

def only_eeg(yasa_df: pd.DataFrame) -> pd.DataFrame:
    # Not calculating for scaled _s cols because - does that make any sense?
    cols = [col for col in yasa_df.columns if "eeg" in col and not col.endswith("_s")]
    eeg = yasa_df[cols]
    return eeg


def stats(yasa_df: pd.DataFrame) -> pd.DataFrame:
    eeg = only_eeg(yasa_df)
    stats_list = []

    stages = yasa_df['Stage'].unique()
    sleep_stages = yasa_df[yasa_df['Stage'] != 'W']
    non_deep_sleep_stages = yasa_df[yasa_df['Stage'].isin(['W', 'N1', 'N2', 'R'])]

    # for idx, col in enumerate(tqdm(eeg.columns, desc="EEG column stats")):
    for col in eeg.columns:
        col_data = eeg[col]
        stats = {
            'Column': col,
            'P10': col_data.quantile(0.10),
            'P90': col_data.quantile(0.90),
            'Min': col_data.min(),
            'Max': col_data.max()
        }

        for stage in stages:
            stage_data = yasa_df[yasa_df['Stage'] == stage]
            stage_col_data = stage_data[col]
            stats.update({
                f'{stage}_P10': stage_col_data.quantile(0.10),
                f'{stage}_P90': stage_col_data.quantile(0.90),
                f'{stage}_Min': stage_col_data.min(),
                f'{stage}_Max': stage_col_data.max()
            })

        for fake_stage, stage_data in [('Sleep', sleep_stages), ('NonDeepSleep', non_deep_sleep_stages)]:
            fake_stage_col_data = stage_data[col]
            stats.update({
                f'{fake_stage}_P10': fake_stage_col_data.quantile(0.10),
                f'{fake_stage}_P90': fake_stage_col_data.quantile(0.90),
                f'{fake_stage}_Min': fake_stage_col_data.min(),
                f'{fake_stage}_Max': fake_stage_col_data.max()
            })

        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    return stats_df

def normalize_series(series, p10, p90):
    return (series - p10) / (p90 - p10)

def scale_by_stats(yasa_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    assert not any(col.startswith("Main") for col in yasa_df), "Do not want to double-scale Main channel"

    eeg = only_eeg(yasa_df)
    stats_dict = stats_df.set_index('Column').to_dict('index')

    def normalize_if_exists(col):
        if col.name in stats_dict:
            return normalize_series(col, stats_dict[col.name]['P10'], stats_dict[col.name]['P90'])
        else:
            return col

    normalized_eeg = eeg.apply(normalize_if_exists)
    return normalized_eeg


def add_main_channel(df):
    columns = df.columns
    channels = df['Source'].unique()
    filtered_channels = [channel for channel in channels if channel.startswith("F")]
    if len(filtered_channels) == 1:
        main_channel = filtered_channels[0]
    elif 'Fpz' in filtered_channels:
        main_channel = 'Fpz'
    else:
        main_channel = filtered_channels[0] if filtered_channels else None

    if main_channel is None:
        return df  # No main channel found, return the original DataFrame

    # Keep both the _s and not versions of the main column
    filtered_columns = [col for col in columns if "eeg" in col and col.startswith(main_channel)]
    new_columns = {col.replace(main_channel, "Main"): df[col] for col in filtered_columns}

    # Concatenate the new columns to the original DataFrame
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df