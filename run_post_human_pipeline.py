import os
from catboost import CatBoostClassifier

from datetime import datetime

import run_feature_pipeline
from models.eeg_states.eeg_states import load_and_prepare_eeg_state_events, add_event_type
from models.eeg_states.eeg_states_model import predict_only_model_pipeline
from sleep_events import convert_timestamp_to_uk, convert_timestamps_to_uk
import pandas as pd

force_if_older_than = datetime(2024, 11, 8, 17, 0, 0)

def add_periods(log, dir_name, input_file, stats_df, days_data, yasa_df):
    day_and_night_of_date = datetime.strptime(dir_name, "%Y-%m-%d-%H-%M-%S")
    day_and_night_of = day_and_night_of_date.date().isoformat()
    yasa_df['dayAndNightOf'] = day_and_night_of
    yasa_df['Timestamp'] = yasa_df['Timestamp'].apply(convert_timestamp_to_uk)

    row = days_data[days_data['dayAndNightOf'] == day_and_night_of]

    if row.empty:
        raise Exception(f"Don't have human data yet for {day_and_night_of}")

    if pd.notna(row['asleepTime'].iloc[0]):
        yasa_df['asleepTime'] = row['asleepTime'].iloc[0]

        yasa_df['timeSinceSleep'] = yasa_df['Timestamp'] - yasa_df['asleepTime']
        yasa_df['15MinsBeforeSleep'] = ((yasa_df['timeSinceSleep'] >= pd.Timedelta(minutes=-15)) &
                                        (yasa_df['timeSinceSleep'] <= pd.Timedelta(minutes=0))).astype(int)
        yasa_df['30MinsBeforeSleep'] = ((yasa_df['timeSinceSleep'] >= pd.Timedelta(minutes=-30)) &
                                        (yasa_df['timeSinceSleep'] <= pd.Timedelta(minutes=0))).astype(int)
        yasa_df['60MinsBeforeSleep'] = ((yasa_df['timeSinceSleep'] >= pd.Timedelta(minutes=-60)) &
                                        (yasa_df['timeSinceSleep'] <= pd.Timedelta(minutes=0))).astype(int)
        yasa_df['SleepHour1To2'] = ((yasa_df['timeSinceSleep'] <= pd.Timedelta(hours=2)) & (yasa_df['timeSinceSleep'] >= pd.Timedelta(hours=0))).astype(int)
        yasa_df['SleepHour3To4'] = ((yasa_df['timeSinceSleep'] <= pd.Timedelta(hours=4)) & (yasa_df['timeSinceSleep'] >= pd.Timedelta(hours=2))).astype(int)
        yasa_df['SleepHour5To6'] = ((yasa_df['timeSinceSleep'] <= pd.Timedelta(hours=6)) & (yasa_df['timeSinceSleep'] >= pd.Timedelta(hours=4))).astype(int)
        yasa_df['SleepHour7To8'] = ((yasa_df['timeSinceSleep'] <= pd.Timedelta(hours=8)) & (yasa_df['timeSinceSleep'] >= pd.Timedelta(hours=6))).astype(int)
        yasa_df['minsSinceAsleep'] = yasa_df['timeSinceSleep'].dt.total_seconds() / 60

    if pd.notna(row['readyToSleepTime'].iloc[0]):
        yasa_df['readyToSleepTime'] = row['readyToSleepTime'].iloc[0]
        yasa_df['timeSinceReadyToSleep'] = yasa_df['Timestamp'] - yasa_df['readyToSleepTime']
        yasa_df['15MinsBeforeReadyToSleep'] = ((yasa_df['timeSinceReadyToSleep'] >= pd.Timedelta(minutes=-15)) &
                                               (yasa_df['timeSinceReadyToSleep'] <= pd.Timedelta(minutes=0))).astype(int)
        yasa_df['30MinsBeforeReadyToSleep'] = ((yasa_df['timeSinceReadyToSleep'] >= pd.Timedelta(minutes=-30)) &
                                               (yasa_df['timeSinceReadyToSleep'] <= pd.Timedelta(minutes=0))).astype(int)
        yasa_df['60MinsBeforeReadyToSleep'] = ((yasa_df['timeSinceReadyToSleep'] >= pd.Timedelta(minutes=-60)) &
                                               (yasa_df['timeSinceReadyToSleep'] <= pd.Timedelta(minutes=0))).astype(int)
        yasa_df['DuringReadyToSleep'] = ((yasa_df['Timestamp'] >= yasa_df['readyToSleepTime']) & (yasa_df['Timestamp'] <= yasa_df['asleepTime'])).astype(int)
        yasa_df['minsSinceReadyToSleep'] = yasa_df['timeSinceReadyToSleep'].dt.total_seconds() / 60
        # Drop these for a tidy CSV
        yasa_df.drop(columns=['timeSinceReadyToSleep', 'readyToSleepTime'], inplace=True)
        yasa_df.drop(columns=['asleepTime', 'timeSinceSleep'], inplace=True)

    if pd.notna(row['gotIntoBedTime'].iloc[0]):
        yasa_df['gotIntoBedTime'] = row['gotIntoBedTime'].iloc[0]
        yasa_df['timeSinceGotIntoBed'] = yasa_df['Timestamp'] - yasa_df['gotIntoBedTime']
        yasa_df['minsSinceGotIntoBed'] = yasa_df['timeSinceGotIntoBed'].dt.total_seconds() / 60
        yasa_df.drop(columns=['gotIntoBedTime', 'timeSinceGotIntoBed'], inplace=True)

    if pd.notna(row['LEPTime'].iloc[0]):
        yasa_df['LEPTime'] = row['LEPTime'].iloc[0]
        yasa_df['timeSinceLEP'] = yasa_df['Timestamp'] - yasa_df['LEPTime']
        yasa_df['minsSinceLEP'] = yasa_df['timeSinceLEP'].dt.total_seconds() / 60
        yasa_df.drop(columns=['LEPTime', 'timeSinceLEP'], inplace=True)

    if pd.notna(row['wakeTime'].iloc[0]):
        yasa_df['wakeTime'] = row['wakeTime'].iloc[0]
        yasa_df['timeUntilWake'] = yasa_df['wakeTime'] - yasa_df['Timestamp']
        yasa_df['minsUntilWake'] = yasa_df['timeUntilWake'].dt.total_seconds() / 60
        yasa_df.drop(columns=['wakeTime', 'timeUntilWake'], inplace=True)

    return yasa_df


def post_human_pipeline(log, dir_name, input_file, stats_df, days_data, yasa_df, eeg_state_events):
    convert_timestamps_to_uk(yasa_df, 'Timestamp', 'TimestampUK')
    yasa_df = add_periods(log, dir_name, input_file, stats_df, days_data, yasa_df)
    add_event_type(yasa_df, eeg_state_events)

    # TiredVsWired model
    before_ready_to_sleep = yasa_df[yasa_df['60MinsBeforeReadyToSleep'] == 1]
    models_and_data = [predict_only_model_pipeline('main', before_ready_to_sleep)]
    model = CatBoostClassifier()
    model.load_model("models/eeg_states/main_catboost_model.cbm")
    predictions = model.predict_proba(models_and_data[0].X)
    predictions_df = pd.DataFrame(predictions, index=models_and_data[0].X.index)
    predictions_df['TiredVsWired60MinsBeforeReadyToSleep'] = predictions_df[1]
    predictions_df.drop([0, 1], axis=1, inplace=True)
    yasa_df_with_predictions = pd.concat([yasa_df, predictions_df], axis=1)

    return yasa_df_with_predictions


def cached_post_human_pipeline(log, dir_name: str, input_file: str, stats_df: pd.DataFrame, days_data: pd.DataFrame, yasa_df: pd.DataFrame, eeg_state_events):
    input_file_without_ext = os.path.splitext(input_file)[0]
    cached = input_file_without_ext + ".post_human.csv"

    def regenerate():
        out = post_human_pipeline(log, dir_name, input_file, stats_df, days_data, yasa_df, eeg_state_events)
        output_csv_file = input_file_without_ext + ".post_human.csv"
        log("Saving to: " + output_csv_file)
        out.to_csv(output_csv_file, index=False)
        return out


    if os.path.exists(cached):
        log("Loading cached file " + cached)
        out = pd.read_csv(cached)

        modification_time = os.path.getmtime(cached)
        modification_date = datetime.fromtimestamp(modification_time)

        if modification_date < force_if_older_than:
            log("Cached file " + cached + f" mod date {modification_date} is < {force_if_older_than}, rebuilding")
            return regenerate()
        if not any(col for col in out.columns if col == 'TiredVsWired60MinsBeforeReadyToSleep'):
            log("Cached file " + cached + " is missing TiredVsWired60MinsBeforeReadyToSleep, rebuilding")
            return regenerate()

        out['epoch'] = out['Epoch']
        out.set_index('epoch', inplace=True)
        return out
    else:
        log(f"No cached file {cached}, rebuilding")
        return regenerate()
