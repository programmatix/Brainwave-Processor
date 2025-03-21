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
from mne.time_frequency import psd_array_multitaper
import time
import cProfile
import pstats
from io import StringIO



# minimal when need a quick sanity check (e.g. for artifact adjusting)
def extract_yasa_features(data, sfreq, artifact_regions=None, minimal: bool = False, profile: bool = False):
    if profile:
        pr = cProfile.Profile()
        pr.enable()
        start_time = time.time()
    
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
        if profile:
            filter_start = time.time()
        # The data has already been filtered so this is unnecessary, but the code breaks when I remove it :/
        dt_filt = filter_data(data[i, :], sf, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False)
        if profile:
            print(f"Filtering time: {time.time() - filter_start:.4f}s")
            window_start = time.time()
        # - Extract epochs. Data is now of shape (n_epochs, n_samples).
        times, epochs = sliding_window(dt_filt, sf=sf, window=30)
        if profile:
            print(f"Window extraction time: {time.time() - window_start:.4f}s")
        
        # Handle artifact regions if provided
        removed_percentages = np.zeros(len(epochs))
        if artifact_regions is not None and len(artifact_regions) > 0:
            if profile:
                artifact_start = time.time()
            # Calculate epoch boundaries in samples
            epoch_length_samples = epochs.shape[1]
            epoch_boundaries = [(int(times[i] * sf), int(times[i] * sf) + epoch_length_samples) for i in range(len(times))]
            
            # Create a copy of epochs to modify
            epochs_clean = epochs.copy()
            
            # Calculate percentage of each epoch that overlaps with artifact regions and mask artifacts
            for i, (epoch_start, epoch_end) in enumerate(epoch_boundaries):
                total_removed_samples = 0
                
                for start_idx, end_idx in artifact_regions:
                    # Calculate overlap between artifact region and epoch
                    overlap_start = max(epoch_start, start_idx)
                    overlap_end = min(epoch_end, end_idx)
                    
                    if overlap_start < overlap_end:  # If there's overlap
                        # Convert global sample indices to indices within the epoch
                        local_start = overlap_start - epoch_start
                        local_end = overlap_end - epoch_start
                        
                        # Mask the artifact region with NaN values
                        epochs_clean[i, local_start:local_end] = np.nan
                        
                        total_removed_samples += (overlap_end - overlap_start)
                
                # Calculate percentage of epoch that was removed
                removed_percentages[i] = (total_removed_samples / epoch_length_samples) * 100
                
                # If there are NaN values in this epoch, interpolate them
                # if np.isnan(epochs_clean[i]).any():
                #     # Get indices of non-NaN values
                #     valid_indices = ~np.isnan(epochs_clean[i])
                #     if np.sum(valid_indices) > 0:  # If there are any valid points
                #         # Create interpolation function using valid points
                #         x = np.arange(len(epochs_clean[i]))[valid_indices]
                #         y = epochs_clean[i][valid_indices]
                #         # Use linear interpolation to fill NaN values
                #         epochs_clean[i] = np.interp(
                #             np.arange(len(epochs_clean[i])), 
                #             x, 
                #             y
                #         )
            
            # Use the cleaned epochs for further processing
            epochs = epochs_clean
            if profile:
                print(f"Artifact handling time: {time.time() - artifact_start:.4f}s")

        if profile:
            basic_stats_start = time.time()
        feat = {
            "std": np.std(epochs, ddof=1, axis=1),
            "removed_percentage": removed_percentages,
        }
        if profile:
            print(f"Basic stats calculation time: {time.time() - basic_stats_start:.4f}s")

        if minimal:
            continue

        if profile:
            hjorth_start = time.time()
        # Calculate standard descriptive statistics
        hmob, hcomp = ant.hjorth_params(epochs, axis=1)
        feat["hmob"] = hmob
        feat["hcomp"] = hcomp
        if profile:
            print(f"Hjorth parameters time: {time.time() - hjorth_start:.4f}s")

        if profile:
            stats_start = time.time()
        feat["skew"] = sp_stats.skew(epochs, axis=1)
        feat["kurt"] = sp_stats.kurtosis(epochs, axis=1)
        feat["nzc"] = ant.num_zerocross(epochs, axis=1)
        feat["iqr"] = sp_stats.iqr(epochs, rng=(25, 75), axis=1)
        if profile:
            print(f"Additional stats time: {time.time() - stats_start:.4f}s")

        # Calculate spectral power features (for EEG + EOG)
        if profile:
            psd_start = time.time()
        # freqs, psd = sp_sig.welch(epochs, sf, **kwargs_welch)
        # Multitaper is slightly more accurate than Welch
        psd, freqs = psd_array_multitaper(epochs, sf, adaptive=True, normalization='full', verbose=0)
        if profile:
            print(f"PSD calculation time: {time.time() - psd_start:.4f}s")

        if c != "emg":
            if profile:
                bandpower_start = time.time()
            bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands, relative=True)
            for j, (_, _, b) in enumerate(bands):
                feat[b] = bp[j]
            # Multiply by an arbitrary constant just to make these tiny values much easier to work with
            bp_abs = bandpower_from_psd_ndarray(psd, freqs, bands=bands, relative=False) * 1_000_000_000
            for j, (_, _, b) in enumerate(bands):
                feat[b + "abs"] = bp_abs[j]
            if profile:
                print(f"Bandpower calculation time: {time.time() - bandpower_start:.4f}s")

            if profile:
                ratios_start = time.time()
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
            if profile:
                print(f"Ratio calculations time: {time.time() - ratios_start:.4f}s")

        # Add power ratios for EEG
        if c == "eeg":
            if profile:
                eeg_ratios_start = time.time()
            delta = feat["sdelta"] + feat["fdelta"]
            feat["dt"] = delta / feat["theta"]
            feat["ds"] = delta / feat["sigma"]
            feat["db"] = delta / feat["beta"]
            feat["at"] = feat["alpha"] / feat["theta"]
            if profile:
                print(f"EEG ratio time: {time.time() - eeg_ratios_start:.4f}s")

        # Add total power
        if profile:
            power_start = time.time()
        # This sums the PSD.  "auc" is the area under the eeg line, which is another way of looking at abspow.
        idx_broad = np.logical_and(freqs >= 0.5, freqs <= 35)
        dx = freqs[1] - freqs[0]
        feat["abspow"] = np.trapz(psd[:, idx_broad], dx=dx) * 1_000_000_000
        if profile:
            print(f"Power calculation time: {time.time() - power_start:.4f}s")

        # Calculate entropy and fractal dimension features
        if profile:
            entropy_start = time.time()
        feat["perm"] = np.apply_along_axis(ant.perm_entropy, axis=1, arr=epochs, normalize=True)
        feat["higuchi"] = np.apply_along_axis(ant.higuchi_fd, axis=1, arr=epochs)
        feat["petrosian"] = ant.petrosian_fd(epochs, axis=1)
        if profile:
            print(f"Entropy and fractal dimension time: {time.time() - entropy_start:.4f}s")

        # New features added after YASA
        if profile:
            new_feat_start = time.time()
        abs_signal = np.abs(epochs)
        feat["auc"] = np.trapz(abs_signal, axis=1)

        # Possibly overkill to throw everything in Antropy in the mix, but why not
        # We don't normalise new cols because we're going to normalise against multiple days of data later.
        # Cols already in YASA continue to be normalised to not throw off their model.
        # Removing as seems to be broken and already have "perm" above
        #feat["perment"] = ant.perm_entropy(epochs, normalize=False)
        feat["specent"] = ant.spectral_entropy(epochs, sf=sfreq, method='welch', normalize=False, axis=1)
        # Seems to return nothing useful with normalize=False, switching to True 7th Dec
        # Still doesn't seem very useful and not working well when data has been de-artifacted.
        # feat["svdent"] = ant.svd_entropy(epochs, normalize=True)
        if profile:
            print(f"New features time: {time.time() - new_feat_start:.4f}s")

    # Convert to dataframe
    if profile:
        df_start = time.time()
    feat = pd.DataFrame(feat).add_prefix(c + "_")
    features.append(feat)

    # Save features to dataframe
    features = pd.concat(features, axis=1)
    features.index.name = "epoch"
    if profile:
        print(f"DataFrame creation time: {time.time() - df_start:.4f}s")

    # Apply centered rolling average (15 epochs = 7 min 30)
    if not minimal:
        if profile:
            rolling_start = time.time()
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
        if profile:
            print(f"Rolling average time: {time.time() - rolling_start:.4f}s")

    # If an epoch's removed_percentage is > 50%, set all its features to NaN
    # Do this after the rolling average has been calculated
    if profile:
        nan_start = time.time()
    if np.any(removed_percentages > 50):
        for key in features.keys():
            if key.endswith("_removed_percentage"):
                continue
            if key.endswith("_norm"):
                continue
            # Convert integer arrays to float before assigning NaN
            if np.issubdtype(features[key].dtype, np.integer):
                features[key] = features[key].astype(float)
            features.loc[removed_percentages > 50, key] = np.nan
    if profile:
        print(f"NaN handling time: {time.time() - nan_start:.4f}s")

    # Downcast float64 to float32 (to reduce size of training datasets)
    if profile:
        downcast_start = time.time()
    cols_float = features.select_dtypes(np.float64).columns.tolist()
    features[cols_float] = features[cols_float].astype(np.float32)

    # Sort the column names here (same behavior as lightGBM)
    features.sort_index(axis=1, inplace=True)
    if profile:
        print(f"Downcast and sort time: {time.time() - downcast_start:.4f}s")

    if profile:
        pr.disable()
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.4f}s")
        
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 functions by cumulative time
        print(s.getvalue())

    return features


import pandas as pd
from mne.io import Raw
from yasa import SleepStaging, sliding_window, bandpower_from_psd_ndarray


def extract_yasa_features2(log, channels: list[str], mne_filtered: Raw, artifacts_df: pd.DataFrame, calc_both: bool = True, minimal: bool = False, profile: bool = False):
    if profile:
        start_time = time.time()
        
    channel_feats_dict = {}
    all_feats_list = []

    for i, channel in enumerate(channels):
        if minimal:
            if i >= 1:
                continue
        if profile:
            channel_start = time.time()
            
        numpy_data = mne_filtered.get_data(picks=channel, units=dict(eeg="uV")) / 1_000_000

        if calc_both:
            # Calculate features on raw data
            if profile:
                print(f"\nProcessing raw data for channel {channel} ({i+1}/{len(channels)})")
            feats_raw = extract_yasa_features(numpy_data, mne_filtered.info['sfreq'], None, minimal, profile)
            feats_raw.columns = [f"Raw_{col}" for col in feats_raw.columns]
    
        # Get the same data but consider artifacts for processing
        if profile:
            print(f"\nProcessing artifact-aware data for channel {channel} ({i+1}/{len(channels)})")
        artifacts_for_channel = artifacts_df[artifacts_df['channel'] == channel][['start', 'end']].values
        feats = extract_yasa_features(numpy_data, mne_filtered.info['sfreq'], artifacts_for_channel, minimal, profile)
        feats.columns = [f"{channel}_{col}" for col in feats.columns]
        
        # Combine raw and artifact-aware features
        if calc_both:
            combined_feats = pd.concat([feats, feats_raw], axis=1)
        else:
            combined_feats = feats
        
        channel_feats_dict[channel] = combined_feats
        all_feats_list.append(combined_feats)
        
        if profile:
            print(f"Channel {channel} processing time: {time.time() - channel_start:.4f}s")

    yasa_feats = pd.concat(all_feats_list, axis=1)
    
    if profile:
        print(f"Total extract_yasa_features2 time: {time.time() - start_time:.4f}s")
        
    return yasa_feats, channel_feats_dict