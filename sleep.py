from datetime import timedelta


def probably_asleep(df_combined_awake):
    df_asleep = df_combined_awake.copy()

    # Define the two-minute window
    two_minutes = timedelta(minutes=2)

    # Initialize the 'DefinitelySleep' column
    df_asleep['ProbablySleep'] = True

    # Mark epochs that are definitely awake based on manual scoring
    df_asleep.loc[df_asleep['ProbablyAwake'] == True, 'ProbablySleep'] = False

    # Mark epochs within +/- two minutes of awake events as awake
    for idx, row in df_asleep.iterrows():
        if row['ProbablySleep'] == False:
            start_time = row['Timestamp'] - two_minutes
            end_time = row['Timestamp'] + two_minutes
            df_asleep.loc[(df_asleep['Timestamp'] >= start_time) & (df_asleep['Timestamp'] <= end_time), 'ProbablySleep'] = False

    return df_asleep