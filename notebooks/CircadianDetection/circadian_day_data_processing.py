import pandas as pd
import notebooks.Util.Time as Time
reload(Time)

def clean_day_data(day_data, most_useful):

    columns_to_select = most_useful.copy()  # Create a copy to avoid modifying the original
    columns_to_select.append("dayAndNightOf")

    # Circadian data just gets better here. As I'm doing morning walks.
    df = day_data[day_data['dayAndNightOf'] >= '2024-08-14'][columns_to_select].sort_values(by='dayAndNightOf')

    # Missing because sometimes I didn't go outside.  Setting to 0?
    df["sunExposureCombined:wentOutside"] = df["sunExposureCombined:firstEnteredOutsideSSM"].notna().astype(int)
    df.fillna({"sunExposureCombined:lastOutsideSSM": 0, 
            "sunExposureCombined:firstEnteredOutsideSSM": 0, 
            "sunExposureCombined:firstDurationOutsideSecs": 0, 
            "sunExposureCombined:lightDuringFirstTimeOutside": 0,
            "sunExposureCombined:betweenWakeAndFirstSunSecs": 0}, inplace=True)

    # Similiar - dind't use luminette
    df.fillna({"events:luminetteLevel2:lastSSM": 0, 
            "events:luminetteLevel1:lastSSM": 0, 
            "events:luminetteLevel3:lastSSM": 0, 
            "events:luminetteLevel2:firstSSM": 0,
            "events:luminetteLevel1:firstSSM": 0,
            "events:luminetteLevel3:firstSSM": 0,
            "events:luminette:lastSSM": 0,
            "events:luminette:firstSSM": 0,
            "events:luminette:durationSecs": 0,
            "events:luminette:count": 0,
            "events:luminetteLevel2:count": 0,
            "events:luminetteLevel1:count": 0,
            "events:luminetteLevel3:count": 0,
            }, inplace=True)

    # Same deal
    df.fillna({"events:shower:lastSSM": 0, 
            "events:shower:firstSSM": 0,
            'events:shower:durationSecs': 0}, inplace=True)

    # These can reasonably be interpolated
    df["sunExposureCombined:lightDuringFirstTimeOutside"] = df["sunExposureCombined:lightDuringFirstTimeOutside"].fillna(
        df["sunExposureCombined:lightDuringFirstTimeOutside"].shift().rolling(window=3, min_periods=1).mean()
    )

    # Missing because sometimes I didn't go outside and sometimes I don't have wake time times
    df["sunExposureCombined:betweenWakeAndFirstSunSecs"].fillna(0, inplace=True)

    # Just too often the sensor was flaky
    reluctantly_have_to_drop_as_too_much_missing_data = [c for c in df.columns if 'lux' in c.lower()]
    reluctantly_have_to_drop_as_too_much_missing_data.extend(['night:aggregated:gotIntoBedTimeSSM'])

    day_data_scrubbed = df.copy().drop(columns=reluctantly_have_to_drop_as_too_much_missing_data)

    # Add some features
    day_data_scrubbed['showerEndToLEP'] = day_data_scrubbed['circadian:basic:entries:LEP:datetimeSSM'] - day_data_scrubbed['events:shower:lastSSM']
    day_data_scrubbed['MP1ToLEP'] = day_data_scrubbed['circadian:basic:entries:MP1:datetimeSSM'] -day_data_scrubbed['circadian:basic:entries:LEP:datetimeSSM']
        
    # Add next days
    for c in ['circadian:basic:entries:LEP:datetimeSSM', 
            'circadian:basic:entries:MP1:datetimeSSM', 
            'circadian:basic:entries:LEP:prominence', 
            'circadian:basic:entries:MP1:prominence',
            'circadian:basic:entries:LEP:temp',
            'circadian:basic:entries:MP1:temp'
            ]:
        day_data_scrubbed[f'{c}:nextDay'] = day_data_scrubbed[c].shift(-1)
        day_data_scrubbed[f'{c}:prevDay'] = day_data_scrubbed[c].shift(1)

    missing_info = Data.analyze_missing_values(day_data_scrubbed, day_data_scrubbed.columns)

    # Just drop any rows where < 5% of values are missing.  Saves time hand-treating ti all.
    rows_before = len(day_data_scrubbed)
    low_missing_cols = missing_info[missing_info["Missing %"] < 5.0].index.tolist()
    day_data_scrubbed = day_data_scrubbed.dropna(subset=low_missing_cols)
    rows_after = len(day_data_scrubbed)
    print(f"Dropped {rows_before - rows_after} rows of {rows_before} due to missing a small amount of data")

    import numpy as np


    day_data_scrubbed = Time.convert_cols_to_hours(day_data_scrubbed)


    # Rename cols to be cleaner
    for col in day_data_scrubbed.columns:
        cleaned_col_name = (col.replace('sunExposureCombined:', '')
                            .replace('events:', '')
                            .replace('circadian:basic:entries:', '')
                            .replace('night:aggregated:', ''))
        if cleaned_col_name != col:
            day_data_scrubbed[cleaned_col_name] = day_data_scrubbed[col]
            day_data_scrubbed.drop(columns=[col], inplace=True)


    day_data_scrubbed.index = day_data_scrubbed['dayAndNightOf']
    day_data_scrubbed = day_data_scrubbed.drop(columns=['dayAndNightOf'])



    import circadian_dimensionality_reduction
    reload(circadian_dimensionality_reduction)


    df_fa, fa_model, loadings, results_dict, factor_names = circadian_dimensionality_reduction.apply_factor_analysis_to_circadian(day_data_scrubbed, verbose=False)

    # merge with target
    day_data_scrubbed = pd.merge(day_data_scrubbed, df_fa, left_index=True, right_index=True)


    non_target_cols = [col for col in day_data_scrubbed.columns if not "LEP" in col and not "MP1" in col]
    day_data_features_only = day_data_scrubbed[non_target_cols].copy()

    # Remove any Timestamp type columns
    cols_before = day_data_features_only.columns
    day_data_features_only = day_data_features_only.select_dtypes(exclude=['datetime64', 'datetime64[ns, Europe/London]'])
    cols_after = day_data_features_only.columns
    print(f"Dropped {len(cols_before) - len(cols_after)} columns due to Timestamp type columns ({set(cols_before) - set(cols_after)}) ")    

    # time_cols = ['firstEnteredOutside', 'lastOutside', 'luminette:last', 'shower:last', 'LEP:datetime',  'MP1:datetime', 'wakeTime', 'asleepTime', 'readyToSleepTime']






    reload(Data)
    Data.analyze_missing_values(day_data_features_only, day_data_features_only.columns)



