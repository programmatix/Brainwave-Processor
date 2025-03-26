from typing import List
from influxdb import InfluxDBClient
import dotenv
import os
import pandas as pd
from dataclasses import dataclass

dotenv.load_dotenv()

host = os.getenv('INFLUXDB_HOST')
port = os.getenv('INFLUXDB_PORT')
username = os.getenv('INFLUXDB_USERNAME')
password = os.getenv('INFLUXDB_PASSWORD')
database = os.getenv('INFLUXDB_DATABASE')

def get_data_for_day(day_data, day: str, time1: str = '08:00:00Z', time2: str = '08:00:00Z'):
    day = day
    day_plus_one = (pd.to_datetime(day) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    dd = day_data[day_data['dayAndNightOf'] == day]

    client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database, ssl=True, verify_ssl=False)
    query = f'SELECT mean("temp") AS "Temp" FROM "XL"."autogen"."android_temp" WHERE time >= \'{day}T{time1}\' AND time <= \'{day_plus_one}T{time2}\' AND "temp" > 25 AND "temp" < 50 GROUP BY time(1m) FILL(null)'
    result = client.query(query)
    points = list(result.get_points())
    df = pd.DataFrame(points)

    # Convert time column to datetime64[ns, Europe/London]
    df['time'] = pd.to_datetime(df['time']).dt.tz_convert('Europe/London')
    
    return df, dd


# Function to create annotation chart for time points that exist in dd DataFrame
def create_annotation_charts(dd, time_column, label, color):
    if time_column in dd.columns:
        valid_times = dd[dd[time_column].notna()]
        
        if not valid_times.empty:
            times = pd.to_datetime(valid_times[time_column])
            labels = [f"{label} ({t.strftime('%H:%M')})" for t in times]
            
            annotation_df = pd.DataFrame({
                'time': times,
                'label': labels
            })
            
            rule = alt.Chart(annotation_df).mark_rule(
                color=color,
                strokeWidth=2,
                strokeDash=[4, 2]
            ).encode(
                x='time:T'
            )
            
            text = alt.Chart(annotation_df).mark_text(
                align='left',
                baseline='top',
                dx=5,
                dy=-10,
                angle=270,
                fontSize=12,
                fontWeight='bold'
            ).encode(
                x='time:T',
                text='label',
                color=alt.value(color)
            )
            
            return (rule + text)
    return None


from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, savgol_filter

# May need different versions (or at least params) for different LEP, LT1 etc. But trying to stay generic at first.
def get_peaks_and_valleys(df):
    df = df.copy()

    # Handle small gaps on 2024-08-14
    interpolation_limit_mins = 15

    df['Temp_Interpolated'] = df['Temp'].interpolate(limit=interpolation_limit_mins)
    values = df['Temp_Interpolated'].values
    # Windows length of 50 (rather than 61) for 2024-09-13
    # Reducing window further to 20 for 2025-03-20 - quite common to put the device on fairly late
    df['Temp_Smoothed_Savgol'] = savgol_filter(values, window_length=20, polyorder=2)
    #df['Temp_Smoothed_Gaussian'] = gaussian_filter(values, sigma=9)

    smoothed_temps = df['Temp_Smoothed_Savgol'].values
    temp_gradient = np.gradient(smoothed_temps)
    rise_indices = np.where(temp_gradient > 0.004)[0]
    fall_indices = np.where(temp_gradient < -0.004)[0]

    rise_df = df.iloc[rise_indices].copy()
    rise_df['feature'] = 'Rising Temperature'

    fall_df = df.iloc[fall_indices].copy()
    fall_df['feature'] = 'Falling Temperature'

    # 30m apart for 2024-08-06
    distance_peaks = 30
    # To eliminate some crappy peaks on 2024-08-06
    prominence_peaks = 0.05

    # For valleys I'm particularly interested in when the first big drop ends, so allow more to be discovered
    # 2025-03-24 is tricky with a mid-fall break
    distance_valleys = 10
    prominence_valleys = 0.01

    peaks, _ = find_peaks(df['Temp_Smoothed_Savgol'].values, distance=distance_peaks, prominence=prominence_peaks)
    valleys, _ = find_peaks(-df['Temp_Smoothed_Savgol'].values, distance=distance_valleys, prominence=prominence_valleys)
        #peaks_raw, _ = find_peaks(df['Temp'].values, prominence=prominence, distance=distance)

    # Find nearest actual peaks within ±5 samples of the smoothed ones
    # (Smoothing makes it easier to find the peaks, but we want the real values)
    nearest_peaks = []
    for peak in peaks:
        window_start = max(0, peak - 5)
        window_end = min(len(df), peak + 6)
        window = df['Temp'].iloc[window_start:window_end]
        local_peak_idx = window.idxmax()
        nearest_peaks.append(local_peak_idx)

    nearest_valleys = []
    for valley in valleys:
        window_start = max(0, valley - 5)
        window_end = min(len(df), valley + 6)
        window = df['Temp'].iloc[window_start:window_end]
        local_valley_idx = window.idxmin()
        nearest_valleys.append(local_valley_idx)


    # Smoothed peaks (for debugging)
    peaks_df = df.iloc[peaks].copy()
    peaks_df['feature'] = peaks_df.apply(lambda x: f'PeakSm ({x.time.strftime("%H:%M")}, {x.Temp:.2f}°C)', axis=1)
    
    # The real peaks we will use
    peaks_raw_df = df.iloc[nearest_peaks].copy()
    peaks_raw_df['feature'] = peaks_raw_df.apply(lambda x: f'PeakRl ({x.time.strftime("%H:%M")}, {x.Temp:.2f}°C)', axis=1)

    valleys_df = df.iloc[nearest_valleys].copy()    
    valleys_df['feature'] = valleys_df.apply(lambda x: f'Valley ({x.time.strftime("%H:%M")}, {x.Temp:.2f}°C)', axis=1)

    return df, rise_df, fall_df, peaks_df, peaks_raw_df, valleys_df


import altair as alt
import pandas as pd

def draw_chart(df, dd, merged=None):
    df, rise_df, fall_df, peaks_df, peaks_raw_df, valleys_df = get_peaks_and_valleys(df)

    # Create a selection that can be used to zoom
    brush = alt.selection_interval(encodings=['x'], name='brush')

    # Main temperature chart
    temp_chart = alt.Chart(df).mark_line(
        color='blue',
        opacity=0.7,
        strokeWidth=1.5
    ).encode(
        x=alt.X('time:T', 
                scale=alt.Scale(domain=brush),
                axis=alt.Axis(title='Time', tickCount=10, labelAngle=45, format='%Y-%m-%d %H:%M')),
        y=alt.Y('Temp:Q',
                axis=alt.Axis(title='Temperature (°C)'),
                scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('time:T', format='%Y-%m-%d %H:%M:%S'),
            alt.Tooltip('Temp:Q')
        ]
    )

    savgol = alt.Chart(df).mark_line(
        color='red',
        strokeWidth=2
    ).encode(
        x='time:T',
        y='Temp_Smoothed_Savgol:Q',
        tooltip=[
            alt.Tooltip('time:T', format='%Y-%m-%d %H:%M:%S'),
            alt.Tooltip('Temp_Smoothed_Savgol:Q')
        ]
    )

    # gaussian = alt.Chart(df).mark_line(
    #     color='orange',
    #     strokeWidth=2
    # ).encode(
    #     x='time:T',
    #     y='Temp_Smoothed_Gaussian:Q',
    #     tooltip=['time:T', 'Temp_Smoothed_Gaussian:Q']
    # )

    # 5. Highlight regions of rising temperature
    rising_regions = alt.Chart(rise_df).mark_point(
        color='green',
        size=30,
        opacity=0.5
    ).encode(
        x='time:T',
        y='Temp_Smoothed_Savgol:Q',
        tooltip=[
            alt.Tooltip('time:T', format='%Y-%m-%d %H:%M:%S'),
            alt.Tooltip('Temp_Smoothed_Savgol:Q'),
            alt.Tooltip('feature')
        ]
    )

    falling_regions = alt.Chart(fall_df).mark_point(
        color='red',
        size=30,
        opacity=0.5
    ).encode(
        x='time:T',
        y='Temp_Smoothed_Savgol:Q',
        tooltip=[
            alt.Tooltip('time:T', format='%Y-%m-%d %H:%M:%S'),
            alt.Tooltip('Temp_Smoothed_Savgol:Q'),
            alt.Tooltip('feature')
        ]
    )



    # 6. Mark peaks with stars
    # peaks_markers = alt.Chart(peaks_df).mark_point(
    #     color='purple',
    #     size=120,
    #     shape='star',
    #     filled=True,
    #     opacity=0.7
    # ).encode(
    #     x='time:T',
    #     y='Temp_Smoothed_Savgol:Q',
    #     tooltip=['time:T', 'Temp_Smoothed_Savgol:Q', 'feature']
    # )

    # 7. Add text labels for peaks
    peaks_labels = alt.Chart(peaks_df).mark_text(
        align='center',
        baseline='bottom',
        dy=-25,
        fontSize=10,
        fontWeight='bold'
    ).encode(
        x='time:T',
        y='Temp_Smoothed_Savgol:Q',
        text='feature:N',
        color=alt.value('red')
    )
    peaks_labels_raw = alt.Chart(peaks_raw_df).mark_text(
        align='center',
        baseline='bottom',
        dy=-10,
        fontSize=10,
        fontWeight='bold'
    ).encode(
        x='time:T',
        y='Temp_Smoothed_Savgol:Q',
        text='feature:N',
        color=alt.value('blue')
    )

    valleys_labels = alt.Chart(valleys_df).mark_text(
        align='center',
        baseline='bottom',
        dy=10,
        fontSize=10,
        fontWeight='bold'
    ).encode(   
        x='time:T',
        y='Temp_Smoothed_Savgol:Q',
        text='feature:N',
        color=alt.value('red')
    )



    # Create annotation charts for each time point
    annotations = []

    # Add each type of annotation if the column exists
    if 'gotIntoBedTime' in dd.columns:
        got_into_bed = create_annotation_charts(dd, 'gotIntoBedTime', 'Got Into Bed', 'green')
        if got_into_bed:
            annotations.append(got_into_bed)

    if 'readyToSleepTime' in dd.columns:
        ready_to_sleep = create_annotation_charts(dd, 'readyToSleepTime', 'Ready To Sleep', 'orange')
        if ready_to_sleep:
            annotations.append(ready_to_sleep)

    if 'asleepTime' in dd.columns:
        asleep = create_annotation_charts(dd, 'asleepTime', 'Asleep', 'red')
        if asleep:
            annotations.append(asleep)

    if 'wakeTime' in dd.columns:
        wake = create_annotation_charts(dd, 'wakeTime', 'Wake', 'purple')
        if wake:
            annotations.append(wake)

    # Combine the temperature chart with all annotations
    detail_charts = [temp_chart, savgol, rising_regions, falling_regions, peaks_labels, peaks_labels_raw, valleys_labels]
    detail_charts.extend([a for a in annotations if a is not None])
    detail = alt.layer(*detail_charts).properties(
        width=800,
        height=400,
        title="Temperature Measurements with Sleep Annotations"
    )

    # Overview chart (smaller version at bottom)
    overview = alt.Chart(df).mark_line(
        color='blue',
        opacity=0.7
    ).encode(
        x=alt.X('time:T', 
                axis=alt.Axis(title='', tickCount=10, labelAngle=0)),
        y=alt.Y('Temp:Q', 
                axis=alt.Axis(title='', tickCount=3),
                scale=alt.Scale(zero=False))
    ).properties(
        width=800,
        height=60
    ).add_selection(brush)

    # Combine the charts
    chart = alt.vconcat(detail, overview).configure_view(
        stroke=None
    ).configure_axis(
        grid=True,
        gridColor='#DCDCDC'
    )

    return chart



def do_it_all(day_data, day: str, start_time: str = '08:00:00Z', end_time: str = '08:00:00Z', merged=None):
    df, dd = get_data_for_day(day_data, day, start_time, end_time)
    chart = draw_chart(df, dd, merged)
    return chart


def fetch_raw_data(day_data, all_days):
    days_data = {}
    for day in all_days:
        try:
            df, dd = get_data_for_day(day_data, day)
            days_data[day] = {
                'df': df,
                'dd': dd
            }
        except Exception as e:
            days_data[day] = {
                'error': str(e)
            }
    return days_data


from sleep_events import connect_to_firebase


def process_circadian_reviews(peaks_troughs_series, cr_df):
    # List to hold all processed days
    all_days_data = []
    
    # Process each day (each index in the Series)
    for day_idx in range(len(peaks_troughs_series)):
        day_data = peaks_troughs_series.iloc[day_idx]
        
        # Skip if not a list
        if not isinstance(day_data, list):
            continue
        
        # Initialize a dictionary for this day
        day_flattened = {'day_index': day_idx}
        
        # Add dayAndNightOf from the original DataFrame if it exists
        if 'dayAndNightOf' in cr_df.columns:
            day_flattened['dayAndNightOf'] = cr_df.iloc[day_idx]['dayAndNightOf']
        
        # Process each item in the day's data
        for item in day_data:
            if not isinstance(item, dict) or 'type' not in item:
                continue
                
            type_key = item['type']
            
            # Add status if it exists
            if 'status' in item:
                day_flattened[f"{type_key}_status"] = item['status']
            
            # Add value fields if they exist
            if 'value' in item and isinstance(item['value'], dict):
                for val_key, val in item['value'].items():
                    day_flattened[f"{type_key}_{val_key}"] = val
        
        # Add this day's data to the list
        all_days_data.append(day_flattened)
    
    # Create a DataFrame from all days
    result_df = pd.DataFrame(all_days_data)
    return result_df


def get_circadian_reviews():
    db = connect_to_firebase()
    docs = db.collection('circadianRhythmReviewsExperimental').stream()
    records = [doc.to_dict() for doc in docs]
    cr = pd.DataFrame(records)
    cr['timestampWritten'] = pd.to_datetime(cr['timestampWritten']).dt.tz_convert('Europe/London')


    result_df = process_circadian_reviews(cr['peaksTroughs'], cr)

    return cr, result_df



@dataclass
class DetectionResult:
    dayAndNightOf: str
    type: str # 'LEP' etc.
    status: str
    reason: str
    time: pd.Timestamp = None
    temp: float = None

def extract_lep_data(days_data, verbose=False):
    lep_data_list: List[DetectionResult] = []
    historical_leps: List[pd.Timestamp] = []
    type = 'LEP'
    
    for dayAndNightOf, data in days_data.items():
        if 'error' in data:
            lep_data_list.append(DetectionResult(
                dayAndNightOf=dayAndNightOf,
                type=type,
                status='Error',
                reason=data['error']
            ))
            continue
            
        peaks_raw_df = data['peaks_raw_df']

        if verbose:
            display("Peaks:")
            display(peaks_raw_df)
        
        if 'time' not in peaks_raw_df.columns:
            lep_data_list.append(DetectionResult(
                dayAndNightOf=dayAndNightOf,
                type=type,
                status='Error',
                reason='No time column in peaks data'
            ))
            continue
            
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(peaks_raw_df['time']):
            try:
                peaks_raw_df['time'] = pd.to_datetime(peaks_raw_df['time'])
            except Exception as e:
                lep_data_list.append(DetectionResult(
                    dayAndNightOf=dayAndNightOf,
                    type=type,
                    status='Error',
                    reason=f"Time column conversion error: {e}"
                ))
                continue

        # Get all evening peaks between 8pm and midnight
        evening_peaks = peaks_raw_df[peaks_raw_df['time'].apply(
            lambda x: 20 <= x.hour < 24 if hasattr(x, 'hour') else False
        )]

        if verbose:
            display("Evening Peaks:")
            display(evening_peaks)


        # AFAIK, I'm always asleep after LEP  
        dd = data['dd']
        if 'asleepTime' in dd and dd['asleepTime'].notna().any():
            asleep_time = dd.loc[dd['asleepTime'].notna(), 'asleepTime'].iloc[0]
            evening_peaks = evening_peaks[evening_peaks['time'] <= asleep_time]

            if verbose:
                display("Asleep at:")
                display(asleep_time)
                display("Evening Peaks after asleep:")
                display(evening_peaks)


        if not evening_peaks.empty:
            evening_peaks = evening_peaks.sort_values('time')

        if evening_peaks.empty:
            lep_data_list.append(DetectionResult(
                dayAndNightOf=dayAndNightOf,
                type=type,
                status='Not found',
                reason='No usable peaks found'
            ))
            continue

        # Rest of the code remains the same...
        # Calculate reference time if we have historical data
        reference_time = None
        selection_method = "last_peak"  # default method
        
        if len(historical_leps) >= 3:
            historical_hours = [(t.hour + t.minute/60) for t in historical_leps[-3:]]
            reference_time = sum(historical_hours) / len(historical_hours)
            selection_method = "historical_average"

        # Choose the appropriate peak
        if selection_method == "historical_average":
            peak_hours = evening_peaks['time'].apply(lambda x: x.hour + x.minute/60)
            closest_peak_idx = abs(peak_hours - reference_time).idxmin()
            lep_row = evening_peaks.loc[closest_peak_idx].copy()
            #reasoning = f"Selected peak at {lep_row['time'].strftime('%H:%M')} from {evening_peaks['time'].strftime('%H:%M')} as it's closest to historical average of {int(reference_time)}:{int((reference_time % 1) * 60):02d}"
            reasoning = f"Use {lep_row['time'].strftime('%H:%M')} from {[ep.strftime('%H:%M') for ep in evening_peaks['time']]} as it's closest to historical average of {int(reference_time)}:{int((reference_time % 1) * 60):02d}"
        else:
            lep_row = evening_peaks.iloc[-1].copy()
            reasoning = "Selected last peak of the evening (no historical data available)"

        # Create data entry
        lep_data = DetectionResult(
            dayAndNightOf=dayAndNightOf,
            type=type,
            status='Detected',
            time=lep_row['time'],
            temp=lep_row['Temp'],
            #'LEPp_selection_method': selection_method,
            reason=reasoning,
            #'LEPp_num_peaks_found': len(evening_peaks)
        )
        
        # if 'feature' in lep_row:
        #     lep_data['LEP_feature'] = lep_row['feature']
        
        # Update historical data
        historical_leps.append(lep_row['time'])
        if len(historical_leps) > 3:
            historical_leps.pop(0)
            
        lep_data_list.append(lep_data)
    
    df = pd.DataFrame(lep_data_list)
    df.drop(columns=['type'], inplace=True)
    rename_cols = {col: f"LEP_det_{col}" for col in df.columns if col != 'dayAndNightOf'}
    df = df.rename(columns=rename_cols)
    
    return df

def detect_leps(all_processed, cr_known_lep):
    detected_lep_df = extract_lep_data(all_processed)
    joined_lep_df = pd.merge(cr_known_lep, detected_lep_df, on='dayAndNightOf', how='outer')
    joined_lep_df['LEP_diff'] = (joined_lep_df['LEP_det_time'] - joined_lep_df['LEP_cr_datetime']).abs()
    
    does_not_match_cr = joined_lep_df[joined_lep_df['LEP_diff'] > pd.Timedelta(minutes=5)].sort_values('LEP_diff', ascending=False)

    return joined_lep_df, does_not_match_cr


def process_raw_data(days_data):
    processed_data = {}
    for day, data in days_data.items():
        if 'error' in data:
            processed_data[day] = data
            continue
            
        try:
            df = data['df']
            df, rise_df, fall_df, peaks_df, peaks_raw_df, valleys_df = get_peaks_and_valleys(df)
            dd = data['dd']
            processed_data[day] = {
                'df': df,
                'dd': dd,
                'peaks_raw_df': peaks_raw_df,
                'valleys_df': valleys_df
            }
        except Exception as e:
            processed_data[day] = {
                'error': str(e)
            }
    return processed_data



def cr_known(cr_df, type: str):
    cr_known_lep = cr_df[['dayAndNightOf', f'{type}_datetime', f'{type}_temp', f'{type}_prominence']]
    cr_known_lep = cr_known_lep[~cr_known_lep[f'{type}_datetime'].isna()]
    cr_known_lep[f'{type}_datetime'] = pd.to_datetime(cr_known_lep[f'{type}_datetime']).dt.tz_convert('Europe/London')
    
    rename_cols = {
        f'{type}_datetime': f'{type}_cr_datetime',
        f'{type}_temp': f'{type}_cr_temp',
        f'{type}_prominence': f'{type}_cr_prominence'
    }
    cr_known_lep = cr_known_lep.rename(columns=rename_cols)
    
    return cr_known_lep


def find_first_stable_valley(valleys_df):
    if len(valleys_df) == 0:
        return None
    
    current_valley = valleys_df.iloc[0]
    
    for _, valley in valleys_df.iloc[1:].iterrows():
        if valley['Temp_Smoothed_Savgol'] < (current_valley['Temp_Smoothed_Savgol'] - 0.1):
            current_valley = valley
        else:
            break
    
    return current_valley

def extract_lt1(days_data, merged=None, verbose=False):
    detections: List[DetectionResult] = []
    historical_lt1s: List[pd.Timestamp] = []
    typ = 'LT1'
    
    for dayAndNightOf, data in days_data.items():
        if 'error' in data:
            detections.append(DetectionResult(
                dayAndNightOf=dayAndNightOf,
                type=typ,
                status='Error',
                reason=data['error']
            ))
            continue
            
        valleys_df = data['valleys_df']

        if verbose:
            display("Valleys:")
            display(valleys_df)
        
        if 'time' not in valleys_df.columns:
            detections.append(DetectionResult(
                dayAndNightOf=dayAndNightOf,
                type=typ,
                status='Error',
                reason='No time column in valleys data'
            ))
            continue

        useful_valleys = valleys_df[valleys_df['time'].apply(
            lambda x: 0 <= x.hour < 3 or 23 <= x.hour < 24 if hasattr(x, 'hour') else False
        )]

        # Filter valleys after LEP if available
        lep_time = None
        if merged is not None and 'LEP_merge_datetime' in merged.columns:
            lep_row = merged[merged['dayAndNightOf'] == dayAndNightOf]
            if not lep_row.empty and pd.notna(lep_row['LEP_merge_datetime'].iloc[0]):
                lep_time = lep_row['LEP_merge_datetime'].iloc[0]
                useful_valleys = useful_valleys[useful_valleys['time'] >= lep_time]

        if verbose:
            display("Useful Valleys:")
            display(useful_valleys)

        # Should always come after sleep
        dd = data['dd']
        if 'asleepTime' in dd and dd['asleepTime'].notna().any():
            asleep_time = dd.loc[dd['asleepTime'].notna(), 'asleepTime'].iloc[0]
            useful_valleys = useful_valleys[useful_valleys['time'] >= asleep_time]

            if verbose:
                display("Asleep at:")
                display(asleep_time)
                display("Useful Valleys after asleep:")
                display(useful_valleys)

        if not useful_valleys.empty:    
            useful_valleys = useful_valleys.sort_values('time')

        if useful_valleys.empty:
            reason = 'No relevant valleys found'
            if lep_time is not None:
                reason += f' after LEP at {lep_time.strftime("%H:%M")}'
            detections.append(DetectionResult(
                dayAndNightOf=dayAndNightOf,
                type=typ,
                status='Not found',
                reason=reason
            ))
            continue

        first_stable_valley = find_first_stable_valley(useful_valleys)        
        reasoning = f"Selected first stable valley at {first_stable_valley['time'].strftime('%H:%M')}"
        if lep_time is not None:
            reasoning += f" (after LEP at {lep_time.strftime('%H:%M')})"
        lt1_row = first_stable_valley.copy()

        # Create data entry
        lt1_data = DetectionResult(
            dayAndNightOf=dayAndNightOf,
            type=typ,
            status='Detected',
            time=lt1_row['time'],
            temp=lt1_row['Temp'],
            reason=reasoning,
        )
        
        # Update historical data
        historical_lt1s.append(lt1_row['time'])
        if len(historical_lt1s) > 3:
            historical_lt1s.pop(0)
            
        detections.append(lt1_data)
    
    df = pd.DataFrame(detections)
    df.drop(columns=['type'], inplace=True)
    rename_cols = {col: f"LT1_det_{col}" for col in df.columns if col != 'dayAndNightOf'}
    df = df.rename(columns=rename_cols)
    
    return df


def detect_lt1(all_processed, cr_known_lt1, merged):
    detected_lt1_df = extract_lt1(all_processed, merged)
    joined_lt1_df = pd.merge(cr_known_lt1, detected_lt1_df, on='dayAndNightOf', how='outer')
    joined_lt1_df['LT1_diff'] = (joined_lt1_df['LT1_det_time'] - joined_lt1_df['LT1_cr_datetime']).abs()
    
    does_not_match_cr = joined_lt1_df[joined_lt1_df['LT1_diff'] > pd.Timedelta(minutes=5)].sort_values('LT1_diff', ascending=False)

    return joined_lt1_df, does_not_match_cr


from IPython.display import HTML

def display_stats(df, days_with_cr, prefix):
    def format_timedelta(td):
        total_minutes = int(td.total_seconds() / 60)
        hours = abs(total_minutes) // 60
        minutes = abs(total_minutes) % 60
        sign = '-' if total_minutes < 0 else ''
        return f"{sign}{hours:02d}:{minutes:02d}"

    median_formatted = format_timedelta(df[prefix + '_diff'].median())
    mean_formatted = format_timedelta(df[prefix + '_diff'].mean())
    mismatch_count = len(days_with_cr)
    total_days = len(df[~df[prefix + '_diff'].isna()])
    pct_mismatch = round(mismatch_count / total_days * 100, 1)

    stats_html = f"""
    <div style="background-color: #f5f5f5; padding: 30px; border-radius: 15px; font-family: Arial, sans-serif;">
        <h2 style="color: #2c3e50; margin-top: 0; text-align: center; margin-bottom: 30px;">{prefix} Analysis Summary</h2>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; text-align: center;">
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 18px;">Median Diff vs CR</div>
                <div style="font-size: 48px; font-family: monospace; margin: 10px 0; color: #2c3e50;">{median_formatted}</div>
            </div>
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 18px;">Mean Diff vs CR</div>
                <div style="font-size: 48px; font-family: monospace; margin: 10px 0; color: #2c3e50;">{mean_formatted}</div>
            </div>
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 18px;">Mismatched Days</div>
                <div style="font-size: 48px; font-family: monospace; margin: 10px 0; color: #e74c3c;">{mismatch_count}</div>
            </div>
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 18px;">Total Days CR</div>
                <div style="font-size: 48px; font-family: monospace; margin: 10px 0; color: #2c3e50;">{total_days}</div>
            </div>
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 18px;">Pct Mismatch</div>
                <div style="font-size: 48px; font-family: monospace; margin: 10px 0; color: #2c3e50;">{pct_mismatch}%</div>
            </div>
        </div>
    </div>
    """

    return HTML(stats_html)