# To handle different channels having different scales, we scale data by all the data we have for a given channel-feature pair.


import pandas as pd


def only_eeg(yasa_df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in yasa_df.columns if "eeg" in col]
    eeg = yasa_df[cols]
    return eeg


def stats(yasa_df: pd.DataFrame) -> pd.DataFrame:
    eeg = only_eeg(yasa_df)

    # List to store the results
    stats_list = []

    # Iterate over each column in the `eeg` DataFrame
    for col in eeg.columns:
        col_data = eeg[col]
        stats = {
            'Column': col,
            'Mean': col_data.mean(),
            'P10': col_data.quantile(0.10),
            'P90': col_data.quantile(0.90),
            'Min': col_data.min(),
            'Max': col_data.max(),
            'StdDev': col_data.std()
        }
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

    filtered_columns = [col for col in columns if "eeg" in col and col.startswith(main_channel) and col.endswith("_s")]
    for col in filtered_columns:
        renamed = col.replace(main_channel, "Main")
        df[renamed] = df[col]