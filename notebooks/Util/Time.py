import pandas as pd
import numpy as np

# Convert 'Secs' and 'SSM' columns to hours, e.g. 21.5
def convert_cols_to_hours(df):
    df = df.copy()
    new_data = {}

    # Convert 'Secs' columns to hours and rename
    for col in df.columns.copy():  # copy to avoid modifying during iteration
        if 'Secs' in col:
            new_col = col.replace('Secs', 'Hours')
            new_data[new_col] = df[col] / 3600
            # print(f"{col} -> {new_col}")
        # else:
        #     day_data_scrubbed.drop(columns=[col], inplace=True)

    # Convert 'SSM' columns (seconds since midnight) to hours
    for col in df.columns.copy():
        if 'SSM' in col:
            new_col = col.replace('SSM', 'HSM')
            new_data[new_col] = df[col] / 3600
            # print(f"{col} -> {new_col}")
        # else:
        #     day_data_scrubbed.drop(columns=[col], inplace=True)

    # Join all columns at once
    df = pd.concat([
        df.drop(columns=[col for col in df.columns if 'Secs' in col or 'SSM' in col]),
        pd.DataFrame(new_data, index=df.index)
    ], axis=1)

    return df



def convert_col_name(col_name):
    if 'SSM' in col_name:
        return col_name.replace('SSM', '')
    elif 'Secs' in col_name:
        return col_name.replace('Secs', '')
    else:
        return col_name
