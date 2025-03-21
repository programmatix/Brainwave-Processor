from mne.filter import filter_data
freq_broad = (0.4, 30)
from yasa import SleepStaging, sliding_window, bandpower_from_psd_ndarray
import scipy.signal as sp_sig
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simpson
# dt_filt = filter_data(epoch_data_small[0, :], sf, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False)
# # - Extract epochs. Data is now of shape (n_epochs, n_samples).
# times, epochs = sliding_window(dt_filt, sf=sf, window=30)
# dt_filt = filter_data(data_small_raw, sf, l_freq=0.5, h_freq=30, verbose=False, method='fir')
# epoch_filt = dt_filt[epoch_index_small * samples_per_epoch : (epoch_index_small + 1) * samples_per_epoch]
import pandas as pd

bands = [
    (0.4, 1, "sdelta"),
    (1, 4, "fdelta"),
    (4, 8, "theta"),
    (8, 12, "alpha"),
    (12, 16, "sigma"),
    (16, 30, "beta"),
]

def calc_psd(epoch_data, sf, win_sec=5):
    win_sec = 5  # = 2 / freq_broad[0]
    win = int(win_sec * sf)

    # freq_broad = (0.1, 30)
    # dt_filt = filter_data(epoch_data, sf, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False)
    # dt_filt = detrended_data

    kwargs_welch = {
        "window": "hamming",
        "nperseg": win,
        "average": "median"
    }

    freqs, psd = sp_sig.welch([epoch_data], sf, **kwargs_welch)
    psd = psd[0]

    abs_powers_welch = []
    rel_powers_welch = []
    abs_powers_multitaper = []
    rel_powers_multitaper = []

    psd_multitaper, freqs_multitaper = psd_array_multitaper(epoch_data, sf, adaptive=True, normalization='full', verbose=0)

    freq_res_welch = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
    freq_res_multitaper = freqs_multitaper[1] - freqs_multitaper[0]  # = 1 / 4 = 0.25


    # Compute the absolute and relative power for each band
    total_power_welch = simpson(psd, dx=freq_res_welch)
    total_power_multitaper = simpson(psd_multitaper, dx=freq_res_multitaper)

    for low, high, label in bands:
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        band_power_welch = simpson(psd[idx_band], dx=freq_res_welch)
        abs_powers_welch.append(band_power_welch)
        rel_powers_welch.append(band_power_welch / total_power_welch)

    for low, high, label in bands:
        idx_band = np.logical_and(freqs_multitaper >= low, freqs_multitaper <= high)

        band_power_multitaper = simpson(psd_multitaper[idx_band], dx=freq_res_multitaper)
        abs_powers_multitaper.append(band_power_multitaper)
        rel_powers_multitaper.append(band_power_multitaper / total_power_multitaper)

    # Create a DataFrame
    df_powers = pd.DataFrame({
        'Band': [label for _, _, label in bands],
        'Absolute Power Welch (uV^2)': abs_powers_welch,
        'Relative Power Welch': rel_powers_welch,
        'Absolute Power Multitaper (uV^2)': abs_powers_multitaper,
        'Relative Power Multitaper': rel_powers_multitaper
    })

    df_psd_welch = pd.DataFrame({'Frequency (Hz)': freqs, 'Power Spectral Density (uV^2 / Hz)': psd})
    df_psd_multitaper = pd.DataFrame({'Frequency (Hz)': freqs_multitaper, 'Power Spectral Density (uV^2 / Hz)': psd_multitaper})

    return df_psd_welch, df_psd_multitaper, df_powers


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_power_spectral_density(df_psd_welch, df_psd_multitaper):
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightpink', 'lightyellow', 'lightgray']

    freqs_welch = df_psd_welch['Frequency (Hz)']
    psd_welch = df_psd_welch['Power Spectral Density (uV^2 / Hz)']

    freqs_multitaper = df_psd_multitaper['Frequency (Hz)']
    psd_multitaper = df_psd_multitaper['Power Spectral Density (uV^2 / Hz)']

    fig, axes = plt.subplots(1, 2, figsize=(14, 3))

    # Plot Welch's periodogram
    axes[0].plot(freqs_welch, psd_welch, lw=2, color='k')
    for (low, high, label), color in zip(bands, colors):
        idx_band = np.logical_and(freqs_welch >= low, freqs_welch <= high)
        axes[0].fill_between(freqs_welch, psd_welch, where=idx_band, color=color, alpha=0.5, label=label)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power spectral density (uV^2 / Hz)')
    axes[0].set_xlim([0, 35])
    axes[0].set_title("Welch's periodogram")
    sns.despine()

    # Plot multitaper periodogram
    axes[1].plot(freqs_multitaper, psd_multitaper, lw=2, color='k')
    for (low, high, label), color in zip(bands, colors):
        idx_band = np.logical_and(freqs_multitaper >= low, freqs_multitaper <= high)
        axes[1].fill_between(freqs_multitaper, psd_multitaper, where=idx_band, color=color, alpha=0.5, label=label)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power spectral density (uV^2 / Hz)')
    axes[1].set_xlim([0, 35])
    axes[1].set_title("Multitaper periodogram")
    sns.despine()

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import contextlib
import io

def plot_band_powers(df_powers, colors):
    """
    Plots the absolute and relative band powers for Welch and Multitaper methods side by side.

    Parameters:
    df_powers (pd.DataFrame): DataFrame containing the band powers.
    colors (list): List of colors for the bar plots.
    """
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
        fig, ax = plt.subplots(1, 4, figsize=(14, 6))

        sns.barplot(x='Band', y='Absolute Power Welch (uV^2)', data=df_powers, ax=ax[0], palette=colors)
        ax[0].set_title('Abs Band Powers Welch')
        ax[0].set_ylabel('Power (uV^2)')
        ax[0].set_xlabel('Frequency Band')

        sns.barplot(x='Band', y='Relative Power Welch', data=df_powers, ax=ax[1], palette=colors)
        ax[1].set_title('Rel Band Powers Welch')
        ax[1].set_ylabel('Relative Power')
        ax[1].set_xlabel('Frequency Band')

        sns.barplot(x='Band', y='Absolute Power Multitaper (uV^2)', data=df_powers, ax=ax[2], palette=colors)
        ax[2].set_title('Abs Band Powers Multitaper')
        ax[2].set_ylabel('Power (uV^2)')
        ax[2].set_xlabel('Frequency Band')

        sns.barplot(x='Band', y='Relative Power Multitaper', data=df_powers, ax=ax[3], palette=colors)
        ax[3].set_title('Rel Band Powers Multitaper')
        ax[3].set_ylabel('Relative Power')
        ax[3].set_xlabel('Frequency Band')

        plt.tight_layout()
        plt.show()