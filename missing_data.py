import mne
import numpy as np
import pandas as pd


# When the Cyton cannot connect to its dongle, data is dropped and there are gaps in the times.  EDF cannot
# represent these gaps, so pad them with 0s.
def zero_missing_times(log, raw: mne.io.Raw) -> mne.io.RawArray:

    t = raw.times
    d = raw.get_data(units='uV')

    # Generate expected times based on the sampling frequency
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    expected_times = np.arange(n_samples) / sfreq

    # Create a mask for the missing times
    mask = np.isin(expected_times, t)

    # Create a new array for `d` with the same shape as `d`, initializing it with zeros
    new_d = np.zeros((d.shape[0], len(expected_times)))

    # Copy the original values of `d` into the new array at the positions where `t` is not missing
    new_d[:, mask] = d

    new_d = new_d / 1_000_000

    log(f"Padded missing times from {t.shape} to {expected_times.shape}, {d.shape} to {new_d.shape}")

    new_raw = mne.io.RawArray(new_d, raw.info)
    new_raw.set_meas_date(raw.info['meas_date'])
    return new_raw


def missing_times(raw: mne.io.Raw):
    import numpy as np
    import mne

    # Assuming `raw` is your MNE RawArray object
    sfreq = raw.info['sfreq']  # Sampling frequency
    n_samples = raw.n_times  # Number of samples

    # Generate expected times
    expected_times = np.arange(n_samples) / sfreq

    # Get actual times from the RawArray
    actual_times = raw.times

    # Find missing times
    missing_times = np.setdiff1d(expected_times, actual_times)
    return missing_times


def times_data(raw: mne.io.Raw):
    # Turn actual times in raw into DF
    times = pd.DataFrame(raw.times, columns=['datetime'])
    start = pd.to_datetime(raw.info['meas_date'])
    times['delta'] = pd.to_timedelta(times['datetime'], unit='s')

    #start_timedelta = pd.to_timedelta(start - pd.Timestamp('1970-01-01', tz='UTC'))

    times['pd_datetime'] = start + times['delta']

    # Create expected times
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    expected_times_raw = np.arange(n_samples) / sfreq
    expected_times = pd.DataFrame(expected_times_raw, columns=['datetime'])

    merged_times = expected_times.merge(times, on='datetime', how='left', indicator=True)
    missing_times_df = merged_times[merged_times['_merge'] == 'left_only']


    # Find unique minutes
    times['minute'] = times['pd_datetime'].dt.floor('min')
    unique_minutes = times['minute'].drop_duplicates()
    unique_timestamps = pd.DataFrame(unique_minutes)

    return times, unique_timestamps, expected_times, merged_times, missing_times_df

def times_data_numpy(raw: mne.io.Raw):
    # Turn actual times in raw into NumPy array
    times = raw.times
    start = np.datetime64(raw.info['meas_date'])
    start_timedelta = (start - np.datetime64('1970-01-01T00:00:00Z')).astype('timedelta64[ms]')
    pd_datetimes = np.datetime64('1970-01-01T00:00:00Z') + (times * 1e3).astype('timedelta64[ms]') + start_timedelta

    # Create expected times
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    expected_times = np.arange(n_samples) / sfreq

    # Find missing times
    missing_times = np.setdiff1d(expected_times, times)

    # Find unique minutes
    pd_minutes = (pd_datetimes - pd_datetimes.astype('datetime64[m]')).astype('timedelta64[m]')
    unique_minutes = np.unique(pd_minutes)

    return times, unique_minutes, expected_times, missing_times