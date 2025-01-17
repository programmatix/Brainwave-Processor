import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
import mne
from mne.filter import filter_data
import numpy as np

bands = [
    (0.4, 1, "sdelta"),
    (1, 4, "fdelta"),
    (4, 8, "theta"),
    (8, 12, "alpha"),
    (12, 16, "sigma"),
    (16, 30, "beta"),
]

def get_epoch(eeg_data, epoch_idx, samples_per_epoch = 7500):
    if (type(eeg_data) == np.ndarray):
        return eeg_data[epoch_idx * samples_per_epoch : (epoch_idx + 1) * samples_per_epoch]
    if (type(eeg_data) == mne.io.Raw):
        data = eeg_data.get_data(units=dict(eeg="uV"))[0]
        return data[epoch_idx * samples_per_epoch : (epoch_idx + 1) * samples_per_epoch]
    if isinstance(eeg_data, mne.io.edf.edf.RawEDF):
        data = eeg_data.get_data(units=dict(eeg="uV"))[0]
        return data[epoch_idx * samples_per_epoch : (epoch_idx + 1) * samples_per_epoch]
    raise ValueError("Unknown type for eeg_data")

def plot_eeg_epoch(eeg_data, epoch_idx, samples_per_epoch = 7500):
    plot_eeg_data(get_epoch(eeg_data, epoch_idx, samples_per_epoch))

def plot_eeg_data(eeg_data):
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))

    ax.plot(eeg_data, lw=1, color='k', drawstyle='steps')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Voltage')
    ax.set_title('EEG data (0.5 - 35 Hz)')
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_band_power_breakdown(eeg_data_raw, epoch_index, sf=250, samples_per_epoch = 7500):
    # Create a figure with subplots
    fig, axes = plt.subplots(len(bands) + 1, 1, figsize=(12, 4 * len(bands)))

    dt_filt = filter_data(eeg_data_raw, sf, l_freq=0.5, h_freq=30, verbose=False, method='fir')
    epoch_filt = dt_filt[epoch_index * samples_per_epoch : (epoch_index + 1) * samples_per_epoch]

    ax = axes[0]
    ax.plot(epoch_filt, lw=1.5, color='k')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Voltage')
    ax.set_title(f'EEG data (0.5 - 35 Hz)')
    sns.despine()

    # Plot the filtered signals for each band
    for i, (low_freq, high_freq, label) in enumerate(bands):
        dt_filt = filter_data(eeg_data_raw, sf, l_freq=low_freq, h_freq=high_freq, verbose=False, method='fir')
        epoch_filt = dt_filt[epoch_index * samples_per_epoch : (epoch_index + 1) * samples_per_epoch]

        ax = axes[i + 1]
        ax.plot(epoch_filt, lw=1.5, color='k')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Voltage')
        ax.set_title(f'EEG data ({label}: {low_freq}-{high_freq} Hz)')
        sns.despine()

    plt.tight_layout()
    plt.show()


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