# To handle different channels having different scales, we scale data by all the data we have for a given channel-feature pair.


import pandas as pd
from tqdm.auto import tqdm

def only_eeg(yasa_df: pd.DataFrame) -> pd.DataFrame:
    # Not calculating for scaled _s cols because - does that make any sense?
    cols = [col for col in yasa_df.columns if "eeg" in col and not col.endswith("_s")]
    eeg = yasa_df[cols]
    return eeg

def quantile_stats(sorted_col_data: pd.Series, percentile: float) -> dict:
    n = len(sorted_col_data)
    index = int(percentile * n)

    if percentile * n == index:        # If it's exactly at an index, take the average of that value and the next
        quantile = (sorted_col_data.iloc[index-1] + sorted_col_data.iloc[index]) / 2
    else:
        quantile = sorted_col_data.iloc[index]
    return quantile

def stats(yasa_df: pd.DataFrame) -> pd.DataFrame:
    eeg = only_eeg(yasa_df)
    stats_list = []

    # Check if 'Stage' column exists
    has_stage = 'Stage' in yasa_df.columns
    
    if has_stage:
        stages = yasa_df['Stage'].unique()
        stage_sleep_masks = {stage: yasa_df['Stage'] == stage for stage in stages}
        sleep_mask = yasa_df['Stage'] != 'W'
        non_deep_sleep_mask = yasa_df['Stage'].isin(['W', 'N1', 'N2', 'R'])
    else:
        stages = []
        stage_sleep_masks = {}
        sleep_mask = None
        non_deep_sleep_mask = None

    for col in tqdm(eeg.columns):
        col_data = eeg[col]
        #print(f"Processing column: {col} length={len(col_data)}")
        sorted_col_data = col_data.sort_values()
        # mean = col_data.mean() 
        # std = col_data.std()
        # z_score = (col_data - mean) / std
        stats = {
            'Column': col,
            'P10': quantile_stats(sorted_col_data, 0.10),
            'P90': quantile_stats(sorted_col_data, 0.90),
            'Min': sorted_col_data.iloc[0],
            'Max': sorted_col_data.iloc[-1],
            # 'Mean': mean,
            # 'Std': std,
            # 'Z-Score-Max': z_score.max(),
            # 'Z-Score-Mean': z_score.mean()
        }

        if has_stage:
            for stage in stages:
                stage_mask = stage_sleep_masks[stage]
                stage_col_data = col_data[stage_mask]
                sorted_stage_col_data = stage_col_data.sort_values()
                # stage_mean = stage_col_data.mean()
                # stage_std = stage_col_data.std()
                # stage_z_score = (stage_col_data - stage_mean) / stage_std
                stats.update({
                    f'{stage}_P10': quantile_stats(sorted_stage_col_data, 0.10),
                    f'{stage}_P90': quantile_stats(sorted_stage_col_data, 0.90),
                    f'{stage}_Min': sorted_stage_col_data.iloc[0],
                    f'{stage}_Max': sorted_stage_col_data.iloc[-1],
                    # f'{stage}_Mean': stage_mean,
                    # f'{stage}_Std': stage_std,
                    # f'{stage}_Z-Score-Max': stage_z_score.max(),
                    # f'{stage}_Z-Score-Mean': stage_z_score.mean()
                })  

            for fake_stage, mask in [('Sleep', sleep_mask), ('NonDeepSleep', non_deep_sleep_mask)]:
                fake_stage_col_data = col_data[mask]
                sorted_fake_stage_col_data = fake_stage_col_data.sort_values()
                # fake_stage_mean = fake_stage_col_data.mean()
                # fake_stage_std = fake_stage_col_data.std()
                # fake_stage_z_score = (fake_stage_col_data - fake_stage_mean) / fake_stage_std
                stats.update({
                    f'{fake_stage}_P10': quantile_stats(sorted_fake_stage_col_data, 0.10),
                    f'{fake_stage}_P90': quantile_stats(sorted_fake_stage_col_data, 0.90),
                    f'{fake_stage}_Min': sorted_fake_stage_col_data.iloc[0],
                    f'{fake_stage}_Max': sorted_fake_stage_col_data.iloc[-1],
                    # f'{fake_stage}_Mean': fake_stage_mean,
                    # f'{fake_stage}_Std': fake_stage_std,
                    # f'{fake_stage}_Z-Score-Max': fake_stage_z_score.max(),
                    # f'{fake_stage}_Z-Score-Mean': fake_stage_z_score.mean()
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