import pandas as pd
from typing import Dict
import notebooks.Util.Time as Time
from importlib import reload
reload(Time)


def prep_day_data_shared(df: pd.DataFrame,
                  convert_time_columns: bool = True,
                  drop_object_columns: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df['dayAndNightOf'])
    df.drop(columns=['dayAndNightOf'], inplace=True)

    if convert_time_columns:
        df = Time.convert_cols_to_hours(df)

    if drop_object_columns:
        obj_cols = [c for c in df.columns if str(df[c].dtype) == 'object']
        cols_before = len(df.columns)
        df.drop(columns=obj_cols, inplace=True)
        cols_after = len(df.columns)
        print(f"Dropped {cols_before - cols_after} object columns: {obj_cols[:5]}")

    return df

def prep_day_data(df: pd.DataFrame,
                  convert_time_columns: bool = True,
                  drop_object_columns: bool = True) -> pd.DataFrame:
    df = prep_day_data_shared(df, convert_time_columns, drop_object_columns)
    return df

def prep_day_data_subset(df: pd.DataFrame,
                  convert_time_columns: bool = True,
                  drop_object_columns: bool = True) -> pd.DataFrame:
    df = prep_day_data_shared(df, convert_time_columns, drop_object_columns)
    safe_list = [c for c in df.columns if 'weather' in c]
    df_subset = df[safe_list]
    return df_subset

def prep_day_data_for_no_missing_values(df: pd.DataFrame,
                  convert_time_columns: bool = True,
                  drop_object_columns: bool = True) -> pd.DataFrame:
    df = prep_day_data_shared(df, convert_time_columns, drop_object_columns)
    return df


def isMlKeyUselessOrDeprecated(mlKey: str) -> bool:
    # These are based on the total hypnogram duration, which completely depends on when I start it and isn't meaningful.
    return (mlKey == 'night:yasaExtended:Statistics:SE'
        or mlKey == 'night:yasaExtended:Statistics:TIB'
        or mlKey.startswith('night:yasaExtended:Statistics:Lat_')

        # I'm removing _s from the YASA data, so it'll take effect eventually on next full rerun.
        or mlKey.startswith('night:') and '_s:' in mlKey
        
        # Just adding a bunch of noise and I'm never using them
        or mlKey.startswith('night:') and 'source:' in mlKey

        # See YasaData for why these are excluded
        or mlKey.startswith('night:') and ':W:' in mlKey
        or mlKey.startswith('night:') and ':All:' in mlKey

        # Replaced with "N3:"
        or mlKey.startswith('night:yasaExtended:Stability:Aggregated:Deep')

        # Surely want to discard
        or mlKey == 'night:yasa:adjusted:debug:discardedMicrowakingsCount')

