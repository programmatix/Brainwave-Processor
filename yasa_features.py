
# BSD 3-Clause License
#
# Copyright (c) 2018, Raphael Vallat
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# This source code is taken from https://github.com/raphaelvallat/yasa, to allow me to extract the same features that
# YASA does from my EEG data.  Full credit to the authors of that code.

import os
import mne
import glob
import joblib
import logging
import numpy as np
import pandas as pd
import antropy as ant
import scipy.signal as sp_sig
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
from mne.filter import filter_data
from sklearn.preprocessing import robust_scale


def extract_yasa_features(data, sfreq):
    # Bandpass filter
    freq_broad = (0.4, 30)
    # FFT & bandpower parameters
    win_sec = 5  # = 2 / freq_broad[0]
    sf = sfreq
    win = int(win_sec * sf)
    kwargs_welch = {
        "window": "hamming",
        "nperseg": win,
        "average": "median"
    }
    bands = [
        (0.4, 1, "sdelta"),
        (1, 4, "fdelta"),
        (4, 8, "theta"),
        (8, 12, "alpha"),
        (12, 16, "sigma"),
        (16, 30, "beta"),
    ]
    features = []
    ch_types = np.array(["eeg"])

    for i, c in enumerate(ch_types):
        # Preprocessing
        # - Filter the data
        dt_filt = filter_data(data[i, :], sf, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False)
        # - Extract epochs. Data is now of shape (n_epochs, n_samples).
        times, epochs = sliding_window(dt_filt, sf=sf, window=30)

        # Calculate standard descriptive statistics
        hmob, hcomp = ant.hjorth_params(epochs, axis=1)

        feat = {
            "std": np.std(epochs, ddof=1, axis=1),
            "iqr": sp_stats.iqr(epochs, rng=(25, 75), axis=1),
            "skew": sp_stats.skew(epochs, axis=1),
            "kurt": sp_stats.kurtosis(epochs, axis=1),
            "nzc": ant.num_zerocross(epochs, axis=1),
            "hmob": hmob,
            "hcomp": hcomp,
        }

        # Calculate spectral power features (for EEG + EOG)
        freqs, psd = sp_sig.welch(epochs, sf, **kwargs_welch)
        if c != "emg":
            bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands, relative=True)
            for j, (_, _, b) in enumerate(bands):
                feat[b] = bp[j]
            bp_abs = bandpower_from_psd_ndarray(psd, freqs, bands=bands, relative=False)
            for j, (_, _, b) in enumerate(bands):
                feat[b + "abs"] = bp_abs[j]

            # Calculate "delta and below" and "delta and above" features
            for j, (_, _, b) in enumerate(bands):
                if b == "sdelta" or b == "beta":
                    continue
                feat[b + "aa"] = sum(bp[k] for k in range(j, len(bands)))
                feat[b + "absaa"] = sum(bp_abs[k] for k in range(j, len(bands)))
                feat[b + "ab"] = sum(bp[k] for k in range(j + 1))
                feat[b + "absab"] = sum(bp_abs[k] for k in range(j + 1))

            # Calculate spectral centroid
            spectral_centroid = np.sum(freqs * psd, axis=1) / np.sum(psd, axis=1)
            feat["spectral_centroid"] = spectral_centroid

        # Add power ratios for EEG
        if c == "eeg":
            delta = feat["sdelta"] + feat["fdelta"]
            feat["dt"] = delta / feat["theta"]
            feat["ds"] = delta / feat["sigma"]
            feat["db"] = delta / feat["beta"]
            feat["at"] = feat["alpha"] / feat["theta"]

        # Add total power
        idx_broad = np.logical_and(freqs >= freq_broad[0], freqs <= freq_broad[1])
        dx = freqs[1] - freqs[0]
        feat["abspow"] = np.trapz(psd[:, idx_broad], dx=dx)

        # Calculate entropy and fractal dimension features
        feat["perm"] = np.apply_along_axis(ant.perm_entropy, axis=1, arr=epochs, normalize=True)
        feat["higuchi"] = np.apply_along_axis(ant.higuchi_fd, axis=1, arr=epochs)
        feat["petrosian"] = ant.petrosian_fd(epochs, axis=1)

        # New features added after YASA

        abs_signal = np.abs(epochs)
        feat["auc"] = np.trapz(abs_signal, axis=1)

        # Possibly overkill to throw everything in Antropy in the mix, but why not
        # We don't normalise new cols because we're going to normalise against multiple days of data later.
        # Cols already in YASA continue to be normalised to not throw off their model.
        # Removing as seems to be broken and already have "perm" above
        #feat["perment"] = ant.perm_entropy(epochs, normalize=False)
        feat["specent"] = ant.spectral_entropy(epochs, sf=sfreq, method='welch', normalize=False, axis=1)
        # Seems to return nothing useful with normalize=False, switching to True 7th Dec
        feat["svdent"] = ant.svd_entropy(epochs, normalize=True)

        # approximate entropy and sample entropy are very expensive to calculate, and sample seems to be a strict improvement
        # feat["apent"] = np.array([ant.app_entropy(epoch) for epoch in epochs])
        # AKA SampEn as used in Automated Detection of Driver Fatigue Based on Entropy and feat Measures, Zhang, 2014
        # Think this is too slow to run.
        # feat["sampent"] = np.array([ant.sample_entropy(epoch) for epoch in epochs])


    # Convert to dataframe
    feat = pd.DataFrame(feat).add_prefix(c + "_")
    features.append(feat)

    # Save features to dataframe
    features = pd.concat(features, axis=1)
    features.index.name = "epoch"

    # Apply centered rolling average (15 epochs = 7 min 30)
    # Triang: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.,
    #          0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
    rollc = features.rolling(window=15, center=True, min_periods=1, win_type="triang").mean()
    rollc[rollc.columns] = robust_scale(rollc, quantile_range=(5, 95))
    rollc = rollc.add_suffix("_c7min_norm")

    # Now look at the past 2 minutes
    rollp = features.rolling(window=4, min_periods=1).mean()
    rollp[rollp.columns] = robust_scale(rollp, quantile_range=(5, 95))
    rollp = rollp.add_suffix("_p2min_norm")

    # Add to current set of features
    features = features.join(rollc).join(rollp)

    #######################################################################
    # TEMPORAL + METADATA FEATURES AND EXPORT
    #######################################################################

    # Add temporal features
    # features["time_hour"] = times / 3600
    # features["time_norm"] = times / times[-1]

    # Add metadata if present
    # if self.metadata is not None:
    #     for c in self.metadata.keys():
    #         features[c] = self.metadata[c]

    # Downcast float64 to float32 (to reduce size of training datasets)
    cols_float = features.select_dtypes(np.float64).columns.tolist()
    features[cols_float] = features[cols_float].astype(np.float32)
    # Make sure that age and sex are encoded as int
    if "age" in features.columns:
        features["age"] = features["age"].astype(int)
    if "male" in features.columns:
        features["male"] = features["male"].astype(int)

    # Sort the column names here (same behavior as lightGBM)
    features.sort_index(axis=1, inplace=True)

    return features


import pandas as pd
from mne.io import Raw
from yasa import SleepStaging, sliding_window, bandpower_from_psd_ndarray


def extract_yasa_features2(log, channels: list[str], mne_filtered: Raw):
    channel_feats_dict = {}
    all_feats_list = []

    for channel in channels:
        # ss = SleepStaging(mne_filtered, channel)
        # feats = ss.get_features()
        # See comment in get_filtered_and_scaled_data for why the 1_000_000
        numpy_data = mne_filtered.get_data(picks=channel, units=dict(eeg="uV")) / 1_000_000
        feats = extract_yasa_features(numpy_data, mne_filtered.info['sfreq'])
        feats.columns = [f"{channel}_{col}" for col in feats.columns]
        channel_feats_dict[channel] = feats
        all_feats_list.append(feats)

    yasa_feats = pd.concat(all_feats_list, axis=1)
    return yasa_feats, channel_feats_dict