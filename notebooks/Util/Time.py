import pandas as pd
import numpy as np

# Convert 'Secs' and 'SSM' columns to hours, e.g. 21.5
def convert_cols_to_hours(df):
    df = df.copy()

    # Convert 'Secs' columns to hours and rename
    for col in df.columns.copy():  # copy to avoid modifying during iteration
        if 'Secs' in col:
            new_col = col.replace('Secs', '')
            df[new_col] = df[col].apply(
                lambda x: x / 3600 if pd.notna(x) else np.nan
            )
            # print(f"{col} -> {new_col}")
            df.drop(columns=[col], inplace=True)
        # else:
        #     day_data_scrubbed.drop(columns=[col], inplace=True)

    # Convert 'SSM' columns (seconds since midnight) to hours
    for col in df.columns.copy():
        if 'SSM' in col:
            new_col = col.replace('SSM', '')
            df[new_col] = df[col].apply(
                lambda x: x / 3600 if pd.notna(x) else np.nan
            )
            # print(f"{col} -> {new_col}")
            df.drop(columns=[col], inplace=True)
        # else:
        #     day_data_scrubbed.drop(columns=[col], inplace=True)

    return df


