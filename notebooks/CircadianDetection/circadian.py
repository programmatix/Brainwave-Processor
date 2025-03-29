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

def get_data_for_day(day_data, day: str, time1: str = '00:00:00Z', time2: str = '12:00:00Z'):
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
    peaks_raw_df['feature'] = peaks_raw_df.apply(lambda x: f'Peak ({x.time.strftime("%H:%M")}, {x.Temp:.2f}°C)', axis=1)

    valleys_df = df.iloc[nearest_valleys].copy()    
    valleys_df['feature'] = valleys_df.apply(lambda x: f'Valley ({x.time.strftime("%H:%M")}, {x.Temp:.2f}°C)', axis=1)

    return df, rise_df, fall_df, peaks_df, peaks_raw_df, valleys_df


import altair as alt
import pandas as pd



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


def create_merged_annotations(merged, day_and_night_of: str):
    merged_for_day = merged[pd.to_datetime(merged['dayAndNightOf']).dt.strftime('%Y-%m-%d') == day_and_night_of]

    if merged_for_day.empty:
        return None

    # Get all unique types from merged columns
    types = set(col.split('_cr_')[0] 
                for col in merged_for_day.columns 
                if col.endswith('_cr_datetime'))
    
    annotation_data = []
    prominence_data = []
    
    for type_name in types:
        cr_datetime = f'{type_name}_cr_datetime'
        cr_temp = f'{type_name}_cr_temp'
        cr_prominence = f'{type_name}_cr_prominence'
        
        det_datetime = f'{type_name}_det_time'
        det_temp = f'{type_name}_det_temp'
        
        for _, row in merged_for_day.iterrows():
            has_cr = pd.notna(row.get(cr_datetime)) and pd.notna(row.get(cr_temp))
            has_det = pd.notna(row.get(det_datetime)) and pd.notna(row.get(det_temp))
            
            if has_cr and has_det:
                if (row[cr_datetime] == row[det_datetime] and 
                    row[cr_temp] == row[det_temp]):
                    annotation_data.append({
                        'x': row[cr_datetime],
                        'y': row[cr_temp],
                        'text': type_name
                    })
                else:
                    annotation_data.append({
                        'x': row[cr_datetime],
                        'y': row[cr_temp],
                        'text': f'{type_name} (CR) {row[cr_datetime].strftime("%H:%M")} {row[cr_temp]:.2f}'
                    })
                    annotation_data.append({
                        'x': row[det_datetime],
                        'y': row[det_temp],
                        'text': f'{type_name} (DET) {row[det_datetime].strftime("%H:%M")} {row[det_temp]:.2f}'
                    })
            elif has_cr:
                annotation_data.append({
                    'x': row[cr_datetime],
                    'y': row[cr_temp],
                    'text': f'{type_name} (CR) {row[cr_datetime].strftime("%H:%M")} {row[cr_temp]:.2f}'
                })
            elif has_det:
                annotation_data.append({
                    'x': row[det_datetime],
                    'y': row[det_temp],
                    'text': f'{type_name} (DET) {row[det_datetime].strftime("%H:%M")} {row[det_temp]:.2f}'
                })
            
            if has_cr and pd.notna(row.get(cr_prominence)):
                prominence_data.append({
                    'x': row[cr_datetime],
                    'y': row[cr_temp],
                    'size': 50 + row[cr_prominence] * 10
                })
    
    if annotation_data:
        annotations = alt.Chart(pd.DataFrame(annotation_data)).mark_text(
                    align='center',
# align='left',
            # dx=10,
            dy=-3
        ).encode(
            x='x:T',
            y='y:Q',
            text='text:N'
        )
        return annotations
        # chart = chart + annotations
    
    return None
    # if prominence_data:
    #     prominence_circles = alt.Chart(pd.DataFrame(prominence_data)).mark_circle(
    #         opacity=0.3
    #     ).encode(
    #         x='x:T',
    #         y='y:Q',
    #         size='size:Q'
    #     )
    #     chart = chart + prominence_circles
    
def draw_chart(df, dd, merged=None, all_processed=None, post_lep_stats=None):
    day_and_night_of: str = pd.to_datetime(dd['dayAndNightOf']).dt.strftime('%Y-%m-%d').iloc[0]
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
    # peaks_labels = alt.Chart(peaks_df).mark_text(
    #     align='center',
    #     baseline='bottom',
    #     dy=-25,
    #     fontSize=10,
    #     fontWeight='bold'
    # ).encode(
    #     x='time:T',
    #     y='Temp_Smoothed_Savgol:Q',
    #     text='feature:N',
    #     color=alt.value('red')
    # )
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

    if merged is not None:
        merged_annotations = create_merged_annotations(merged, day_and_night_of)
        if merged_annotations:
            annotations.append(merged_annotations)


    # Combine the temperature chart with all annotations
    detail_charts = [temp_chart, savgol, rising_regions, falling_regions, peaks_labels_raw, valleys_labels]
    detail_charts.extend([a for a in annotations if a is not None])

    # If we have post_lep_stats data and a valid LEP time in merged data
    if post_lep_stats is not None and merged is not None and not merged.empty:
        day_and_night_of = pd.to_datetime(dd['dayAndNightOf']).dt.strftime('%Y-%m-%d').iloc[0]
        day_lep = merged[merged['dayAndNightOf'] == day_and_night_of]
        
        if not day_lep.empty and pd.notna(day_lep['LEP_cr_datetime'].iloc[0]):
            lep_time = day_lep['LEP_cr_datetime'].iloc[0]
            
            # Create a DataFrame with timestamps for each minute after LEP
            stats_with_time = post_lep_stats.copy()
            stats_with_time['time'] = lep_time + pd.to_timedelta(stats_with_time['minute'], unit='minutes')
            
            # Create mean line
            mean_line = alt.Chart(stats_with_time).mark_line(
                color='pink',
                strokeWidth=2
            ).encode(
                x='time:T',
                y='mean:Q'
            )
            
            # Create std deviation bands
            bands_data = pd.DataFrame({
                'time': pd.concat([stats_with_time['time'], stats_with_time['time'][::-1]]),
                'Temp': pd.concat([stats_with_time['std1_lower'], stats_with_time['std1_upper'][::-1]])
            })
            
            std_band = alt.Chart(bands_data).mark_area(
                color='red',
                opacity=0.2
            ).encode(
                x='time:T',
                y='Temp:Q'
            )

            std1_lower = alt.Chart(stats_with_time).mark_line(
                color='red',
                opacity=0.2
            ).encode(
                x='time:T',
                y='std1_lower:Q'
            )

            std1_upper = alt.Chart(stats_with_time).mark_line(
                color='red',
                opacity=0.2
            ).encode(
                x='time:T',
                y='std1_upper:Q'
            )

            # Add these to your existing charts list
            detail_charts.extend([mean_line, std1_lower, std1_upper])
            # detail_charts.extend([mean_line])

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



def do_it_all(day_data, day: str, start_time: str = '08:00:00Z', end_time: str = '08:00:00Z', merged=None, all_processed=None, post_lep_stats=None):
    df, dd = get_data_for_day(day_data, day, start_time, end_time)
    chart = draw_chart(df, dd, merged, all_processed, post_lep_stats)
    return chart


def fetch_raw_data(day_data, all_days):
    days_data = {}
    for day in all_days:
        try:
            df, dd = get_data_for_day(day_data, day)
            days_data[day] = {
                'df': df,
                'dd': dd,
                'dayAndNightOf': day
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
    print("Getting circadian rhythm reviews")
    docs = db.collection('circadianRhythmReviewsExperimental').stream()
    print("Starting to process circadian rhythm reviews")
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


# Goal is to look for a valley that doesn't have a better valley soon after it, but doesn't necessarily have to be the lowest.
# Basically want the 'end of the run'.
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

def find_first_stable_peak(peaks_df):
    if len(peaks_df) == 0:
        return None
    
    current_peak = peaks_df.iloc[0]
    
    for _, peak in peaks_df.iloc[1:].iterrows():
        if peak['Temp_Smoothed_Savgol'] > (current_peak['Temp_Smoothed_Savgol'] + 0.1):
            current_peak = peak
        # else:
        #     break
    
    return current_peak

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
        if merged is not None and 'LEP_cr_datetime' in merged.columns:
            lep_row = merged[merged['dayAndNightOf'] == dayAndNightOf]
            if not lep_row.empty and pd.notna(lep_row['LEP_cr_datetime'].iloc[0]):
                lep_time = lep_row['LEP_cr_datetime'].iloc[0]
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


def get_date_str(timestamp):
    return timestamp.strftime('%Y-%m-%d')

def extract_mp1(days_data, merged=None, verbose=False):
    detections: List[DetectionResult] = []
    historical_mp1s: List[pd.Timestamp] = []
    typ = 'MP'
    
    for dayAndNightOf, data in days_data.items():
        if 'error' in data:
            detections.append(DetectionResult(
                dayAndNightOf=dayAndNightOf,
                type=typ,
                status='Error',
                reason=data['error']
            ))
            continue
            
        peaks_df = data['peaks_raw_df']

        if verbose:
            display("Peaks:")
            display(peaks_df)
        
        if 'time' not in peaks_df.columns:
            detections.append(DetectionResult(
                dayAndNightOf=dayAndNightOf,
                type=typ,
                status='Error',
                reason='No time column in peaks data'
            ))
            continue

        # print(dayAndNightOf)
        # print(str(dayAndNightOf))

        useful_peaks = peaks_df[
            # Time between 6 and 11
            peaks_df['time'].apply(lambda x: 6 <= x.hour < 11 if hasattr(x, 'hour') else False) &
            # Date matches dayAndNightOf
            (peaks_df['time'].dt.strftime('%Y-%m-%d') == dayAndNightOf)
        ]

        if verbose:
            display("Useful Peaks:")
            display(useful_peaks)

        # # Should always come after wakeup
        # dd = data['dd']
        # if 'wakeupTime' in dd and dd['wakeupTime'].notna().any():
        #     wakeup_time = dd.loc[dd['wakeupTime'].notna(), 'wakeupTime'].iloc[0]
        #     useful_valleys = useful_valleys[useful_valleys['time'] >= asleep_time]

        #     if verbose:
        #         display("Asleep at:")
        #         display(asleep_time)
        #         display("Useful Valleys after asleep:")
        #         display(useful_valleys)

        if not useful_peaks.empty:    
            useful_peaks = useful_peaks.sort_values('time')

        if useful_peaks.empty:
            reason = 'No relevant peaks found'
            detections.append(DetectionResult(
                dayAndNightOf=dayAndNightOf,
                type=typ,
                status='Not found',
                reason=reason
            ))
            continue

        first_stable_peak = find_first_stable_peak(useful_peaks)        
        reasoning = f"Selected first stable peak at {first_stable_peak['time'].strftime('%H:%M')} from {[ep.strftime('%H:%M') for ep in useful_peaks['time']]}"
        lt1_row = first_stable_peak.copy()

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
        historical_mp1s.append(lt1_row['time'])
        if len(historical_mp1s) > 3:
            historical_mp1s.pop(0)
            
        detections.append(lt1_data)
    
    df = pd.DataFrame(detections)
    df.drop(columns=['type'], inplace=True)
    rename_cols = {col: f"MP_det_{col}" for col in df.columns if col != 'dayAndNightOf'}
    df = df.rename(columns=rename_cols)
    
    return df


def detect_mp1(all_processed, cr_known_mp1, merged):
    detected_mp1_df = extract_mp1(all_processed, merged)
    joined_mp1_df = pd.merge(cr_known_mp1, detected_mp1_df, on='dayAndNightOf', how='outer')
    joined_mp1_df['MP_diff'] = (joined_mp1_df['MP_det_time'] - joined_mp1_df['MP_cr_datetime']).abs()
    
    does_not_match_cr = joined_mp1_df[joined_mp1_df['MP_diff'] > pd.Timedelta(minutes=5)].sort_values('MP_diff', ascending=False)

    return joined_mp1_df, does_not_match_cr


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
    ninety_formatted = format_timedelta(df[prefix + '_diff'].quantile(0.9))
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
                <div style="color: #666; font-size: 18px;">90th %ile Diff vs CR</div>
                <div style="font-size: 48px; font-family: monospace; margin: 10px 0; color: #2c3e50;">{ninety_formatted}</div>
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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_post_lep_periods_matplotlib(all_processed, merged_df, minutes=120):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a DataFrame to store all resampled data and sleep times
    all_data = pd.DataFrame()
    sleep_times = []
    
    # First pass: collect all temperature data and sleep times
    for day, data in all_processed.items():
        if 'error' in data:
            continue
            
        day_lep = merged_df[merged_df['dayAndNightOf'] == day]
        if day_lep.empty or pd.isna(day_lep['LEP_cr_datetime'].iloc[0]):
            continue
            
        lep_time = day_lep['LEP_cr_datetime'].iloc[0]
        
        # Get sleep time if available
        if 'dd' in data and 'asleepTime' in data['dd'].columns:
            sleep_time = data['dd']['asleepTime'].iloc[0]
            if pd.notna(sleep_time):
                mins_until_sleep = (sleep_time - lep_time).total_seconds() / 60
                if 0 <= mins_until_sleep <= minutes:
                    sleep_times.append(mins_until_sleep)
        
        df = data['df']
        period_end = lep_time + pd.Timedelta(minutes=minutes)
        period_data = df[(df['time'] >= lep_time) & (df['time'] <= period_end)].copy()
        
        if period_data.empty:
            continue
        
        period_data['minutes_after_lep'] = (period_data['time'] - lep_time).dt.total_seconds() / 60
        
        ax.plot(period_data['minutes_after_lep'], 
                period_data['Temp'],
                alpha=0.3,
                color='gray',
                linewidth=1)
        
        all_data = pd.concat([all_data, period_data[['minutes_after_lep', 'Temp']]])
    
    # Calculate and plot temperature statistics
    if not all_data.empty:
        all_data['minute'] = all_data['minutes_after_lep'].round()
        stats = all_data.groupby('minute')['Temp'].agg(['mean', 'std']).reset_index()
        stats = stats[stats['minute'] <= minutes]
        
        ax.plot(stats['minute'], stats['mean'], 
                color='red', 
                linewidth=2, 
                label='Mean Temp')
        
        ax.fill_between(stats['minute'], 
                       stats['mean'] - stats['std'],
                       stats['mean'] + stats['std'],
                       color='red', alpha=0.2, label='±1σ Temp')
        ax.fill_between(stats['minute'],
                       stats['mean'] - 2*stats['std'],
                       stats['mean'] + 2*stats['std'],
                       color='red', alpha=0.1, label='±2σ Temp')
    
    # Plot sleep time statistics if we have data
    if sleep_times:
        sleep_mean = np.mean(sleep_times)
        sleep_std = np.std(sleep_times)
        
        # Plot mean sleep time
        ax.axvline(x=sleep_mean, color='blue', linestyle='--', label='Mean Sleep Time')
        
        # Plot 1 std dev band
        ax.axvspan(sleep_mean - sleep_std, sleep_mean + sleep_std, 
                  color='blue', alpha=0.2, label='±1σ Sleep Time')
        
        # Plot 2 std dev band
        ax.axvspan(sleep_mean - 2*sleep_std, sleep_mean + 2*sleep_std, 
                  color='blue', alpha=0.1, label='±2σ Sleep Time')
    
    ax.set_xlabel('Minutes after LEP', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Temperature Patterns in the {minutes} Minutes Following LEP', fontsize=14, pad=20)
    
    ax.set_xlim(0, minutes)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    return fig


def calc_avg_mins_after_lep(all_processed, merged, minutes=120):
    if merged is None:
        return None

    # Calculate the average minutes after LEP for each day
    all_data = pd.DataFrame()

    for day, data in all_processed.items():
        if 'error' in data:
            continue
            
        day_lep = merged[merged['dayAndNightOf'] == day]
        if day_lep.empty or pd.isna(day_lep['LEP_cr_datetime'].iloc[0]):
            continue
            
        lep_time = day_lep['LEP_cr_datetime'].iloc[0]
        
        day_df = data['df']
        period_end = lep_time + pd.Timedelta(minutes=minutes)
        period_data = day_df[(day_df['time'] >= lep_time) & (day_df['time'] <= period_end)].copy()
        
        if period_data.empty:
            continue
        
        # Calculate minutes after LEP
        period_data['minutes_after_lep'] = (period_data['time'] - lep_time).dt.total_seconds() / 60
        
        # Add this day's data to the all_data DataFrame
        all_data = pd.concat([all_data, period_data[['minutes_after_lep', 'Temp']]])

    # Calculate statistics for each minute
    if not all_data.empty:
        # Round to nearest minute and group
        all_data['minute'] = all_data['minutes_after_lep'].round()
        stats = all_data.groupby('minute')['Temp'].agg(['mean', 'std']).reset_index()
        stats = stats[stats['minute'] <= minutes]
        
        # Create DataFrame with mean and both standard deviations
        stats_with_bands = pd.DataFrame({
            'minute': stats['minute'],
            'mean': stats['mean'],
            'std1_lower': stats['mean'] - stats['std'],
            'std1_upper': stats['mean'] + stats['std'],
            'std2_lower': stats['mean'] - 2*stats['std'],
            'std2_upper': stats['mean'] + 2*stats['std']
        })        
        return stats_with_bands
    return None


def analyze_post_lep_period(night_data: pd.DataFrame, stats_with_bands: pd.DataFrame, minutes: int = 120) -> dict:
    """
    Analyze how a specific night's temperature pattern compares to historical averages
    in the post-LEP period.
    
    Args:
        night_data: DataFrame with columns ['minutes_after_lep', 'Temp'] for specific night
        stats_with_bands: DataFrame with historical stats (mean, std bands) per minute
        minutes: Analysis period length in minutes
    """
    # Ensure data is properly aligned by minute
    night_data = night_data.copy()
    night_data['minute'] = night_data['minutes_after_lep'].round()
    
    # Merge night data with historical stats
    analysis = pd.merge(
        night_data,
        stats_with_bands,
        on='minute',
        how='inner'
    )
    
    # Calculate deviations from mean
    analysis['deviation'] = analysis['Temp'] - analysis['mean']
    
    metrics = {
        # Overall metrics
        # 'mean_deviation': analysis['deviation'].mean(),
        # 'max_deviation': analysis['deviation'].max(),
        # 'min_deviation': analysis['deviation'].min(),
        
        # Time spent in different bands
        # 'minutes_above_1std': len(analysis[analysis['Temp'] > analysis['std1_upper']]),
        # 'minutes_below_1std': len(analysis[analysis['Temp'] < analysis['std1_lower']]),
        # 'minutes_within_1std': len(analysis[
        #     (analysis['Temp'] >= analysis['std1_lower']) & 
        #     (analysis['Temp'] <= analysis['std1_upper'])
        # ]),

        'minutes_above_mean': len(analysis[analysis['Temp'] > analysis['mean']]),
        'minutes_below_mean': len(analysis[analysis['Temp'] < analysis['mean']]),

        # Longest continuous periods
        # 'longest_above_mean': get_longest_streak(analysis['Temp'] > analysis['mean']),
        # 'longest_below_mean': get_longest_streak(analysis['Temp'] < analysis['mean']),
        
        # Area metrics
        # 'area_above_mean': np.trapz(
        #     analysis[analysis['deviation'] > 0]['deviation']
        # ),
        # 'area_below_mean': abs(np.trapz(
        #     analysis[analysis['deviation'] < 0]['deviation']
        # ))
    }
    
    return metrics

def get_longest_streak(bool_series: pd.Series) -> int:
    """Calculate the longest continuous streak of True values"""
    if not len(bool_series):
        return 0
    streaks = (bool_series != bool_series.shift()).cumsum()
    return bool_series.groupby(streaks).size().max() if any(bool_series) else 0

def compare_night_to_average(all_processed: dict, merged_df: pd.DataFrame, 
                           stats_with_bands: pd.DataFrame, day: str) -> dict:
    """Compare a specific night to historical averages"""
    if day not in all_processed or 'error' in all_processed[day]:
        return None
        
    day_lep = merged_df[merged_df['dayAndNightOf'] == day]
    if day_lep.empty or pd.isna(day_lep['LEP_cr_datetime'].iloc[0]):
        return None
        
    lep_time = day_lep['LEP_cr_datetime'].iloc[0]
    df = all_processed[day]['df']
    
    # Get post-LEP period data
    period_end = lep_time + pd.Timedelta(minutes=120)
    period_data = df[(df['time'] >= lep_time) & (df['time'] <= period_end)].copy()
    
    if period_data.empty:
        return None
        
    # Calculate minutes after LEP
    period_data['minutes_after_lep'] = (
        period_data['time'] - lep_time
    ).dt.total_seconds() / 60
    
    return analyze_post_lep_period(period_data, stats_with_bands)

def plot_pre_mp_periods_matplotlib(all_processed, merged_df, minutes=120):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a DataFrame to store all resampled data and wake times
    all_data = pd.DataFrame()
    wake_times = []
    
    # First pass: collect all temperature data and wake times
    for day, data in all_processed.items():
        if 'error' in data:
            continue
            
        day_mp = merged_df[merged_df['dayAndNightOf'] == day]
        if day_mp.empty or pd.isna(day_mp['MP_cr_datetime'].iloc[0]):
            continue
            
        mp_time = day_mp['MP_cr_datetime'].iloc[0]
        
        # Get wake time from previous day
        prev_day = (pd.to_datetime(day) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        if prev_day in all_processed and 'dd' in all_processed[prev_day]:
            prev_dd = all_processed[prev_day]['dd']
            if 'wakeTime' in prev_dd.columns and not prev_dd['wakeTime'].empty:
                wake_time = prev_dd['wakeTime'].iloc[0]
                if pd.notna(wake_time):
                    mins_before_mp = (mp_time - wake_time).total_seconds() / 60
                    if 0 <= mins_before_mp <= minutes:
                        wake_times.append(mins_before_mp)
        
        df = data['df']
        period_start = mp_time - pd.Timedelta(minutes=minutes)
        period_data = df[(df['time'] >= period_start) & (df['time'] <= mp_time)].copy()
        
        if period_data.empty:
            continue
        
        period_data['minutes_before_mp'] = (mp_time - period_data['time']).dt.total_seconds() / 60
        
        ax.plot(period_data['minutes_before_mp'], 
                period_data['Temp'],
                alpha=0.3,
                color='gray',
                linewidth=1)
        
        all_data = pd.concat([all_data, period_data[['minutes_before_mp', 'Temp']]])
    
    # Calculate and plot temperature statistics
    if not all_data.empty:
        all_data['minute'] = all_data['minutes_before_mp'].round()
        stats = all_data.groupby('minute')['Temp'].agg(['mean', 'std']).reset_index()
        stats = stats[stats['minute'] <= minutes]
        
        ax.plot(stats['minute'], stats['mean'], 
                color='red', 
                linewidth=2, 
                label='Mean Temp')
        
        ax.fill_between(stats['minute'], 
                       stats['mean'] - stats['std'],
                       stats['mean'] + stats['std'],
                       color='red', alpha=0.2, label='±1σ Temp')
        ax.fill_between(stats['minute'],
                       stats['mean'] - 2*stats['std'],
                       stats['mean'] + 2*stats['std'],
                       color='red', alpha=0.1, label='±2σ Temp')
        
        # Set y-axis limits to mean ± 2*std
        y_min = (stats['mean'] - 2*stats['std']).min()
        y_max = (stats['mean'] + 2*stats['std']).max()
        ax.set_ylim(y_min, y_max)
    
    # Plot wake time statistics if we have data
    if wake_times:
        wake_mean = np.mean(wake_times)
        wake_std = np.std(wake_times)
        
        # Plot mean wake time
        ax.axvline(x=wake_mean, color='purple', linestyle='--', label='Mean Wake Time')
        
        # Plot 1 std dev band
        ax.axvspan(wake_mean - wake_std, wake_mean + wake_std, 
                  color='purple', alpha=0.2, label='±1σ Wake Time')
        
        # Plot 2 std dev band
        ax.axvspan(wake_mean - 2*wake_std, wake_mean + 2*wake_std, 
                  color='purple', alpha=0.1, label='±2σ Wake Time')
    
    ax.set_xlabel('Minutes before MP', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Temperature Patterns in the {minutes} Minutes Before MP', fontsize=14, pad=20)
    
    # Reverse x-axis so time flows left to right
    ax.set_xlim(minutes, 0)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    return fig

def prepare_lep_to_mp_data(all_processed, merged_df, highlight_days=None):
    # Create a DataFrame to store all resampled data and sleep/wake times
    all_data = pd.DataFrame()
    sleep_times = []
    wake_times = []
    highlighted_data = []
    
    # First pass: collect all temperature data and sleep/wake times
    for day, data in all_processed.items():
        if 'error' in data:
            continue
            
        # Get LEP time for current day
        day_lep = merged_df[merged_df['dayAndNightOf'] == day]
        if day_lep.empty or pd.isna(day_lep['LEP_cr_datetime'].iloc[0]):
            continue
            
        lep_time = day_lep['LEP_cr_datetime'].iloc[0]
        
        # Get MP time for next day
        next_day = (pd.to_datetime(day) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        next_day_mp = merged_df[merged_df['dayAndNightOf'] == next_day]
        if next_day_mp.empty or pd.isna(next_day_mp['MP_cr_datetime'].iloc[0]):
            continue
            
        mp_time = next_day_mp['MP_cr_datetime'].iloc[0]
        
        # Get sleep and wake times if available
        if 'dd' in data and 'asleepTime' in data['dd'].columns:
            sleep_time = data['dd']['asleepTime'].iloc[0]
            if pd.notna(sleep_time):
                mins_after_lep = (sleep_time - lep_time).total_seconds() / 60
                sleep_times.append((day, mins_after_lep))
        
        if next_day in all_processed and 'dd' in all_processed[day]:
            next_dd = all_processed[day]['dd']
            if 'wakeTime' in next_dd.columns and not next_dd['wakeTime'].empty:
                wake_time = next_dd['wakeTime'].iloc[0]
                if pd.notna(wake_time):
                    mins_after_lep = (wake_time - lep_time).total_seconds() / 60
                    wake_times.append((day, mins_after_lep))
        
        # Get temperature data
        df = data['df']
        period_data = df[(df['time'] >= lep_time) & (df['time'] <= mp_time)].copy()
        
        if period_data.empty:
            continue
        
        period_data['minutes_after_lep'] = (period_data['time'] - lep_time).dt.total_seconds() / 60
        period_data['day'] = day  # Add the day to the period data
        
        # Store data differently based on whether it's a highlighted day
        if highlight_days and day in highlight_days:
            highlighted_data.append({
                'day': day,
                'data': period_data,
                'sleep_time': next((st[1] for st in sleep_times if st[0] == day), None),
                'wake_time': next((wt[1] for wt in wake_times if wt[0] == day), None)
            })
        
        all_data = pd.concat([all_data, period_data[['day', 'minutes_after_lep', 'Temp']]])
    
    # Calculate statistics
    stats = None
    if not all_data.empty:
        all_data['minute'] = all_data['minutes_after_lep'].round()
        stats = all_data.groupby('minute')['Temp'].agg(['mean', 'std']).reset_index()
    
    return {
        'all_data': all_data,
        'stats': stats,
        'highlighted_data': highlighted_data,
        'sleep_times': sleep_times,
        'wake_times': wake_times
    }

def plot_lep_to_mp_period(all_processed, merged_df, highlight_days=None):
    # Prepare the data
    data = prepare_lep_to_mp_data(all_processed, merged_df, highlight_days)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Plot non-highlighted data from all_data
    if not data['all_data'].empty:
        for day, group in data['all_data'].groupby('day'):
            if not highlight_days or day not in highlight_days:
                ax.plot(group['minutes_after_lep'], 
                       group['Temp'],
                       alpha=0.3,
                       color='gray',
                       linewidth=1)
    
    # Plot highlighted data
    for day_data in data['highlighted_data']:
        ax.plot(day_data['data']['minutes_after_lep'], 
               day_data['data']['Temp'],
               alpha=0.8,
               color='blue',
               linewidth=2)
        
        # Plot sleep and wake times
        if day_data['sleep_time'] is not None:
            temp_at_sleep = day_data['data'][
                day_data['data']['minutes_after_lep'].round() == round(day_data['sleep_time'])
            ]['Temp'].iloc[0]
            ax.plot(day_data['sleep_time'], temp_at_sleep,
                   'bo', markersize=10, label='Sleep Time')
        
        if day_data['wake_time'] is not None:
            temp_at_wake = day_data['data'][
                day_data['data']['minutes_after_lep'].round() == round(day_data['wake_time'])
            ]['Temp'].iloc[0]
            ax.plot(day_data['wake_time'], temp_at_wake,
                   'bo', markersize=10, label='Wake Time')
    
    # Plot statistics
    if data['stats'] is not None:
        ax.plot(data['stats']['minute'], data['stats']['mean'], 
                color='red', 
                linewidth=2, 
                label='Mean Temp')
        
        ax.fill_between(data['stats']['minute'], 
                       data['stats']['mean'] - data['stats']['std'],
                       data['stats']['mean'] + data['stats']['std'],
                       color='red', alpha=0.2, label='±1σ Temp')
        ax.fill_between(data['stats']['minute'],
                       data['stats']['mean'] - 2*data['stats']['std'],
                       data['stats']['mean'] + 2*data['stats']['std'],
                       color='red', alpha=0.1, label='±2σ Temp')
    
    # Plot sleep time statistics if we have data
    if data['sleep_times']:
        sleep_times = [st[1] for st in data['sleep_times']]
        sleep_mean = np.mean(sleep_times)
        sleep_std = np.std(sleep_times)
        
        # Plot mean sleep time
        ax.axvline(x=sleep_mean, color='blue', linestyle='--', label='Mean Sleep Time')
        
        # Plot 1 std dev band
        ax.axvspan(sleep_mean - sleep_std, sleep_mean + sleep_std, 
                  color='blue', alpha=0.2, label='±1σ Sleep Time')
        
        # Plot 2 std dev band
        ax.axvspan(sleep_mean - 2*sleep_std, sleep_mean + 2*sleep_std, 
                  color='blue', alpha=0.1, label='±2σ Sleep Time')

    # Plot wake time statistics if we have data
    if data['wake_times']:
        wake_times = [wt[1] for wt in data['wake_times']]
        wake_mean = np.mean(wake_times)
        wake_std = np.std(wake_times)
        
        # Plot mean wake time
        ax.axvline(x=wake_mean, color='purple', linestyle='--', label='Mean Wake Time')
        
        # Plot 1 std dev band
        ax.axvspan(wake_mean - wake_std, wake_mean + wake_std, 
                  color='purple', alpha=0.2, label='±1σ Wake Time')
        
        # Plot 2 std dev band
        ax.axvspan(wake_mean - 2*wake_std, wake_mean + 2*wake_std, 
                  color='purple', alpha=0.1, label='±2σ Wake Time')
    
    ax.set_xlabel('Minutes after LEP', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Temperature Patterns from LEP to MP', fontsize=14, pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    #plt.tight_layout()
    
    return fig

def analyze_sleep_wake_timing_for_day(all_processed, merged_df, data, day, verbose=False):
    if 'error' in data:
        if verbose:
            print(f"Error in data for day {day}")
        return None
        
    # Get LEP time for current day
    day_lep = merged_df[merged_df['dayAndNightOf'] == day]
    if day_lep.empty or pd.isna(day_lep['LEP_cr_datetime'].iloc[0]):
        if verbose:
            print(f"No LEP time found for day {day}")
        return None
        
    lep_time = day_lep['LEP_cr_datetime'].iloc[0]
    
    # Get sleep time
    sleep_time = None
    if 'dd' in data and 'asleepTime' in data['dd'].columns:
        if not data['dd']['asleepTime'].empty and pd.notna(data['dd']['asleepTime'].iloc[0]):
            sleep_time = data['dd']['asleepTime'].iloc[0]
            mins_after_lep = (sleep_time - lep_time).total_seconds() / 60
            
            # Get wake time from next day's data
            #next_day = (pd.to_datetime(day) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            # wake_time = None
            # if next_day in all_processed and 'dd' in all_processed[next_day]:
            #     next_dd = all_processed[next_day]['dd']
            if 'wakeTime' in data['dd'].columns and not data['dd']['wakeTime'].empty:
                wake_time = data['dd']['wakeTime'].iloc[0]
                if pd.notna(wake_time):
                    mins_after_lep_wake = (wake_time - lep_time).total_seconds() / 60
                    
                    return {
                        'date': day,
                        'sleep_mins_after_lep': mins_after_lep,
                        'wake_mins_after_lep': mins_after_lep_wake,
                        'sleep_time': sleep_time.strftime('%H:%M'),
                        'wake_time': wake_time.strftime('%H:%M'),
                        'lep_time': lep_time.strftime('%H:%M')
                    }
                        
    if verbose:
        print(f"No sleep time found for day {day}")
    return None

def analyze_sleep_wake_timing(all_processed, merged_df):
    sleep_data = []
    
    for day, data in all_processed.items():
        sd = analyze_sleep_wake_timing_for_day(all_processed, merged_df, data, day)
        if sd is not None:
            sleep_data.append(sd)
    
    # Create DataFrame and calculate means
    df = pd.DataFrame(sleep_data)
    if not df.empty:
        sleep_mean = df['sleep_mins_after_lep'].mean()
        wake_mean = df['wake_mins_after_lep'].mean()
        
        df['sleep_vs_mean'] = df['sleep_mins_after_lep'] - sleep_mean
        df['wake_vs_mean'] = df['wake_mins_after_lep'] - wake_mean
        
        # Sort by date
        df = df.sort_values('date')
        
        # Round numeric columns to 1 decimal place
        numeric_cols = ['sleep_mins_after_lep', 'wake_mins_after_lep', 'sleep_vs_mean', 'wake_vs_mean']
        df[numeric_cols] = df[numeric_cols].round(1)
    
    return df
