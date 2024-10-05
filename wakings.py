def get_yasa_probably_awake(log, epochs_with_yasa_features):
    df_probably_awake = epochs_with_yasa_features.copy()
    df_probably_awake['YASAProbablyAwake'] = False

    # Mark epochs that are definitely awake based on YASA scoring
    # We only use epochs where YASA is pretty confident and where it marks a few W in a row.
    # Iterate through each row in df_asleep
    for i in range(1, len(df_probably_awake) - 1):
        # Check if the current row and the rows immediately before and after it have the 'Stage' value 'W' and 'Confidence' >= 0.75
        if (df_probably_awake.loc[i - 1, 'Stage'] == 'W' and df_probably_awake.loc[i - 1, 'Confidence'] >= 0.75 and
                df_probably_awake.loc[i, 'Stage'] == 'W' and df_probably_awake.loc[i, 'Confidence'] >= 0.75 and
                df_probably_awake.loc[i + 1, 'Stage'] == 'W' and df_probably_awake.loc[i + 1, 'Confidence'] >= 0.75):
            df_probably_awake.loc[i, 'YASAProbablyAwake'] = True

    return df_probably_awake


def get_definitely_awake(epochs_with_yasa_features, ha_events, waking_start_time_tz = None, waking_end_time_tz = None):
    # Find epochs where confident am awake - based on notes, button presses, etc.
    df = epochs_with_yasa_features.copy()

    # Initialize the new columns
    df['EventTimes'] = None
    df['ManualStage'] = None

    # Iterate through each row in df
    for i in range(len(df) - 1):
        start_time = df.loc[i, 'Timestamp']
        end_time = df.loc[i + 1, 'Timestamp']

        events_in_epoch = ha_events[
            ((ha_events['timestamp_uk'] >= start_time) & (ha_events['timestamp_uk'] < end_time))
        ]

        df.at[i, 'EventTimes'] = events_in_epoch['timestamp_uk'].tolist()

        if (not events_in_epoch.empty
                or (waking_start_time_tz is not None and (start_time >= waking_start_time_tz and end_time <= waking_end_time_tz))):
            df.at[i, 'ManualStage'] = 'W'

    # Handle the last row separately (no next timestamp)
    df.at[len(df) - 1, 'EventTimes'] = []
    df.at[len(df) - 1, 'ManualStage'] = 'W' if not ha_events[
        ha_events['timestamp_uk'] >= df.loc[len(df) - 1, 'Timestamp']
        ].empty else None

    epochs_manually_scored_awake = df.loc[df['ManualStage'] == 'W']
    epochs_awake = df.loc[(df['ManualStage'] == 'W')]

    df['DefinitelyAwake'] = False

    # Set 'DefinitelyAwake' to True for epochs that are in epochs_awake
    df.loc[epochs_awake.index, 'DefinitelyAwake'] = True

    return df