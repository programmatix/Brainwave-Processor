import tempfile
import pyogg
import pytz
from tqdm import tqdm
import io
import re
from datetime import datetime, timedelta
import os
import dotenv
dotenv.load_dotenv()

def load_audio_as_ogg(buffer):
    with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as temp_file:
        temp_filename = temp_file.name
        # Write the in-memory data to the temporary file
        temp_file.write(buffer)
        print("Temp filename: ", temp_filename)

    #ogg = pyogg.OpusFile("C:\\Users\\graha\\AppData\\Local\\Temp\\tmpxxtfwd25.opus")
    ogg = pyogg.OpusFile(temp_filename)

    print("\nRead Ogg Opus file")
    print("Channels:\n  ", ogg.channels)
    print("Frequency (samples per second):\n  ",ogg.frequency)
    print("Buffer Length (bytes):\n  ", len(ogg.buffer))

    #import numpy as np
    #data_array = np.ctypeslib.as_array(ogg.buffer, shape=(ogg.buffer_length,))
    data_array = ogg.as_array()
    
    print(f"Original audio data shape: {data_array.shape}")
    print(f"Original audio data type: {data_array.dtype}")
    print(f"Original audio data range: {np.min(data_array)} to {np.max(data_array)}")
    print(f"Original audio data mean: {np.mean(data_array)}")
    print(f"Original audio data std dev: {np.std(data_array)}")
    print(f"Sample rate: {ogg.frequency} Hz")
    print(f"Original duration: {len(data_array) / ogg.frequency:.2f} seconds")
    
    # Remove the first second of audio
    # data_array = data_array[ogg.frequency:]
    
    # print(f"After trimming first second - new duration: {len(data_array) / ogg.frequency:.2f} seconds")
    
    # Check if the audio data needs to be scaled down (if values are too large)
    # max_abs_val = np.max(np.abs(data_array))
    # if max_abs_val > 100:
    #     print(f"Audio data has unusually large values (max abs: {max_abs_val}), scaling down")
    #     # Scale down to range [-1, 1]
    #     data_array = data_array / max_abs_val
    #     print(f"After scaling - new range: {np.min(data_array)} to {np.max(data_array)}")
    
    return data_array, ogg, temp_filename


import os
import re
from datetime import datetime, timedelta

def get_audio_files_for_day(remote_files, target_date):
    # Convert target_date to date object if it's a string (YYYY-MM-DD)
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    
    # Calculate the next day for comparison
    next_day = target_date + timedelta(days=1)
    
    # Pattern to match recording filenames and extract datetime information
    pattern = re.compile(r'recording_(\d{8})_(\d{6})\.opus')
    
    matching_files = []
    
    for filename in remote_files:
        match = pattern.match(filename)
        if not match:
            continue
            
        date_str, time_str = match.groups()
        file_datetime = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        
        # Check if the file belongs to our target day:
        # 1. Files from target_date after midday (12:00)
        # 2. Files from next_day before midday
        if (file_datetime.date() == target_date and file_datetime.hour >= 12) or \
           (file_datetime.date() == next_day and file_datetime.hour < 12):
            matching_files.append(filename)
    
    return matching_files

def copy_audio_file(sftp, remote_dir: str, file: str):
    remote_file_path = remote_dir + "/" + file
    print(f"Downloading remote:{remote_file_path} to memory")

    # Get the file size
    remote_file_size = sftp.stat(remote_file_path).st_size

    with tqdm(total=remote_file_size, unit='B', unit_scale=True, desc=file, ascii=True) as pbar:
        # Create an in-memory buffer
        memory_buffer = io.BytesIO()
    
        def callback(transferred_so_far, total_to_transfer):
            pbar.update(transferred_so_far - pbar.n)
    
        # Download to memory buffer instead of file
        sftp.getfo(remote_file_path, memory_buffer, callback=callback)
        
        # Reset buffer position to beginning
        memory_buffer.seek(0)
        
        # Store the memory buffer in the dictionary
        return memory_buffer


import librosa
import numpy as np
def is_silence_in_opus(audio_data, sample_rate, threshold_db=-50, min_silence_duration=1.0):
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Get the amplitude envelope
    amplitude_envelope = np.abs(audio_data)
    
    # Convert to dB
    db = librosa.amplitude_to_db(amplitude_envelope, ref=np.max)
    
    # Find regions below threshold
    silence_mask = db < threshold_db
    
    # Find silence intervals
    silence_intervals = []
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silence_mask):
        if is_silent and not in_silence:
            in_silence = True
            silence_start = i
        elif not is_silent and in_silence:
            in_silence = False
            silence_end = i
            duration = (silence_end - silence_start) / sample_rate
            if duration >= min_silence_duration:
                silence_intervals.append((silence_start / sample_rate, silence_end / sample_rate))
    
    # Handle case where file ends during silence
    if in_silence:
        silence_end = len(silence_mask)
        duration = (silence_end - silence_start) / sample_rate
        if duration >= min_silence_duration:
            silence_intervals.append((silence_start / sample_rate, silence_end / sample_rate))
    
    # Calculate percent of silence
    total_silence_duration = sum(end - start for start, end in silence_intervals)
    total_duration = len(audio_data) / sample_rate
    percent_silence = (total_silence_duration / total_duration) * 100
    
    contains_silence = len(silence_intervals) > 0
    
    return contains_silence, silence_intervals, percent_silence


def get_audio(sftp, remote_dir: str, filename: str):
    copied = copy_audio_file(sftp, remote_dir, filename)
    data_array, ogg, temp_filename = load_audio_as_ogg(copied.getvalue())
    display_waveform(data_array, ogg.frequency, use_db=True)
    display_waveform(data_array, ogg.frequency)
    # display_waveform_audacity_style(data_array, ogg.frequency)
    return data_array, ogg


import librosa
import librosa.display
import matplotlib.pyplot as plt

def load_and_display_wav_audio(file_path: str, use_db=False):
    # Load the WAV file from the local path
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # Display the waveform
    display_waveform(audio_data, sample_rate, use_db=use_db)
    
    return audio_data, sample_rate


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def plot_power_spectrum(data_array, sample_rate):
    # Flatten the data_array to 1D
    data_array_flat = np.ravel(data_array)
    # Compute Welch's periodogram
    frequencies, power_spectrum = signal.welch(data_array_flat, fs=sample_rate, nperseg=1024)

    # mask = frequencies < 50
    # frequencies_filtered = frequencies[mask]
    # power_spectrum_filtered = power_spectrum[mask]

    # Plot the periodogram
    plt.figure(figsize=(10, 6))
    plt.semilogy(frequencies, power_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.title('Welch\'s Periodogram')
    plt.grid()
    plt.show()


def filter_audio(data_array, sample_rate):
    data_array_flat = np.ravel(data_array)

    # Design a gentler low-pass filter (order=3, cutoff=100 Hz)
    nyquist = 0.5 * sample_rate
    cutoff = 100  # Lower cutoff for smoother sound
    normalized_cutoff = cutoff / nyquist
    order = 3  # Lower order reduces artifacts
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

    # Filter order (higher = sharper roll-off, but potential artifacts)
    order = 5  
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

    # Apply the filter (zero-phase)
    filtered_data = signal.filtfilt(b, a, data_array_flat)
    return filtered_data
    
    

import simpleaudio as sa # type: ignore

def play_audio(data_array, ogg, duration=None, gain=1.0):
    # Calculate samples for duration if specified
    if duration:
        samples = int(duration * ogg.frequency)
        print(f"Playing {samples} samples")
        data_to_play = data_array[:samples]
    else:
        data_to_play = data_array

    # Print information about the audio data
    print(f"Audio data before gain - min: {np.min(data_to_play)}, max: {np.max(data_to_play)}")
    
    # Normalize the audio data if values are too large
    # max_abs_val = np.max(np.abs(data_to_play))
    # if max_abs_val > 100:
    #     print(f"Audio data has unusually large values (max abs: {max_abs_val}), normalizing")
    #     # Normalize to range [-1, 1] before applying gain
    #     data_to_play = data_to_play / max_abs_val
    
    # Apply gain by multiplying the audio data
    data_to_play = (data_to_play * gain).astype(data_to_play.dtype)
    
    print(f"Audio data after processing - min: {np.min(data_to_play)}, max: {np.max(data_to_play)}")

    # Play the audio
    play_obj = sa.play_buffer(data_to_play,
                            ogg.channels,
                            ogg.bytes_per_sample,
                            ogg.frequency)

    try:
        # Wait until sound has finished playing or interrupted
        play_obj.wait_done()
    except KeyboardInterrupt:
        # Stop playback on interrupt
        play_obj.stop()


import tempfile
import os
import matplotlib.pyplot as plt

def display_waveform(audio_data, sample_rate, figsize=(12, 4), use_db=False):
    """
    Display the waveform for an audio buffer.
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate in Hz
        figsize (tuple): Figure size (width, height) in inches
        use_db (bool): If True, display amplitude in decibels
    """
    # Create time axis
    time = np.arange(0, len(audio_data)) / sample_rate
    
    # Print detailed information about the audio data
    print(f"Audio data shape: {audio_data.shape}")
    print(f"Audio data type: {audio_data.dtype}")
    print(f"Audio data range: {np.min(audio_data)} to {np.max(audio_data)}")
    print(f"Audio data mean: {np.mean(audio_data)}")
    print(f"Audio data std dev: {np.std(audio_data)}")
    print(f"Audio data RMS: {np.sqrt(np.mean(np.square(audio_data)))}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
    
    # Plot waveform
    plt.figure(figsize=figsize)
    
    if use_db:
        # Convert to decibels
        epsilon = 1e-10  # To avoid log(0)
        db_data = 20 * np.log10(np.abs(audio_data) + epsilon)
        
        # Set a reasonable floor for dB values (e.g., -60 dB)
        # db_floor = -60
        # db_data = np.maximum(db_data, db_floor)
        
        plt.plot(time, db_data)
        plt.title('Audio Waveform (dB Scale)')
        plt.ylabel('Amplitude (dB)')
        # plt.ylim(db_floor, 5)  # Set y-axis limits for better visualization
    else:
        # Original amplitude scale
        plt.plot(time, audio_data)
        # Set y-axis limits to reflect actual amplitude
        max_amp = np.max(np.abs(audio_data))
        plt.ylim(-max_amp * 1.1, max_amp * 1.1)
        plt.title('Audio Waveform (Original Scale)')
        plt.ylabel('Amplitude')
    
    plt.xlabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def display_waveform_audacity_style(audio_data, sample_rate, figsize=(12, 4)):
    # Create time axis
    time = np.arange(0, len(audio_data)) / sample_rate
    
    # Calculate RMS in small windows (similar to Audacity's envelope view)
    window_size = int(sample_rate / 100)  # 10ms windows
    num_windows = len(audio_data) // window_size
    rms_values = np.zeros(num_windows)
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window_data = audio_data[start:end]
        rms_values[i] = np.sqrt(np.mean(np.square(window_data)))
    
    # Convert to dB with Audacity-like scale
    epsilon = 1e-10
    db_values = 20 * np.log10(rms_values + epsilon)
    
    # Plot with Audacity-like style
    plt.figure(figsize=figsize)
    plt.plot(np.linspace(0, len(audio_data)/sample_rate, len(rms_values)), db_values)
    plt.ylim(-60, 0)
    plt.title('Audio Waveform (Audacity-like dB Scale)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Also plot a normalized version for better visualization if not in dB mode
    # if not use_db:
    #     plt.figure(figsize=figsize)
        
    #     # Normalize the audio data to range [-1, 1]
    #     if np.max(np.abs(audio_data)) > 0:
    #         normalized_audio = audio_data / np.max(np.abs(audio_data))
    #         plt.plot(time, normalized_audio)
    #         plt.ylim(-1.1, 1.1)
    #         plt.title('Audio Waveform (Normalized to [-1, 1])')
    #         plt.xlabel('Time (seconds)')
    #         plt.ylabel('Normalized Amplitude')
    #         plt.grid(True, alpha=0.3)
    #         plt.tight_layout()
    #         plt.show()
    
    # Also display a spectrogram
    # plt.figure(figsize=figsize)
    
    # # Compute and plot the spectrogram
    # plt.specgram(audio_data, Fs=sample_rate, cmap='viridis')
    # plt.title('Spectrogram')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Frequency (Hz)')
    # plt.colorbar(label='Intensity (dB)')
    # plt.tight_layout()
    # plt.show()



import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import ipywidgets as widgets
import soundfile as sf

def find_audio_events(audio_data, sample_rate, filename,
                     window_size=1024,
                     threshold_multiplier=1.5,
                     min_event_duration=0.1,
                     merge_distance=0.5):
    """
    Find periods of audio that are above background noise level.
    Returns list of dicts with event details
    """
    # Parse recording start time from filename
    # Expected format: recording_YYYYMMDD_HHMMSS.opus
    timestamp_str = filename.split('recording_')[1].split('.')[0]
    recording_start = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    
    # Calculate energy envelope
    window_rms = []
    for i in range(0, len(audio_data), window_size):
        chunk = audio_data[i:i+window_size]
        if len(chunk) == window_size:
            if np.any(np.isnan(chunk)):
                continue
            square = np.square(chunk)
            mean = np.mean(square)
            if mean < 0:
                continue
            rms = np.sqrt(mean)
            window_rms.append(rms)
    
    if not window_rms:
        return []
    
    envelope = np.array(window_rms)
    threshold = threshold_multiplier * np.mean(envelope)
    
    # Find regions above threshold
    above_threshold = envelope > threshold
    changes = np.diff(above_threshold.astype(int))
    start_indices = np.where(changes == 1)[0]
    end_indices = np.where(changes == -1)[0]
    
    if len(start_indices) == 0:
        return []
    
    if above_threshold[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if above_threshold[-1]:
        end_indices = np.append(end_indices, len(above_threshold) - 1)
    
    start_samples = start_indices * window_size
    end_samples = end_indices * window_size
    
    merge_samples = int(merge_distance * sample_rate)
    min_samples = int(min_event_duration * sample_rate)
    
    events = []
    current_start = start_samples[0]
    current_end = end_samples[0]
    
    for i in range(1, len(start_samples)):
        if start_samples[i] - current_end < merge_samples:
            current_end = end_samples[i]
        else:
            if current_end - current_start >= min_samples:
                event_dict = create_event_dict(
                    audio_data[current_start:current_end],
                    current_start,
                    current_end,
                    sample_rate,
                    recording_start
                )
                events.append(event_dict)
            current_start = start_samples[i]
            current_end = end_samples[i]
    
    if current_end - current_start >= min_samples:
        event_dict = create_event_dict(
            audio_data[current_start:current_end],
            current_start,
            current_end,
            sample_rate,
            recording_start
        )
        events.append(event_dict)
    
    return events

def create_event_dict(audio_segment, start_sample, end_sample, sample_rate, recording_start):
    """
    Create a dictionary with all event details
    """
    duration_samples = end_sample - start_sample
    start_time = start_sample / sample_rate
    end_time = end_sample / sample_rate
    duration_time = duration_samples / sample_rate
    
    # Calculate absolute timestamps
    start_timestamp = recording_start + timedelta(seconds=start_time)
    end_timestamp = recording_start + timedelta(seconds=end_time)
    
    return {
        'audio_data': audio_segment,
        'start_sample': int(start_sample),
        'end_sample': int(end_sample),
        'duration_samples': int(duration_samples),
        'start_time': start_time,
        'end_time': end_time,
        'duration_time': duration_time,
        'start_timestamp': start_timestamp,
        'end_timestamp': end_timestamp,
        'sample_rate': sample_rate
    }

def plot_audio_events(audio_data, events):
    """
    Plot original audio and detected events
    """
    if not events:
        return
    
    sample_rate = events[0]['sample_rate']
    
    plt.figure(figsize=(12, 4))
    
    # Plot full audio
    time_axis = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
    plt.plot(time_axis, audio_data)
    plt.title('Full Audio')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (seconds)')
    
    # Mark events on full audio
    for event in events:
        plt.axvspan(event['start_time'], event['end_time'], 
                   color='red', alpha=0.2)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

import json
from datetime import datetime


best_silence_detection_params_tonor = {'window_size': 2048,
 'threshold_multiplier': 1.5,
 'min_event_duration': 0.2,
 'merge_distance': 0.3}

best_silence_detection_params_new_mic = {'window_size': 512,
 'threshold_multiplier': 1.2,
 'min_event_duration': 0.05,
 'merge_distance': 0.3}


def save_events_metadata(events, original_filename, sftp, remote_dir):
    """
    Save events metadata to a JSON file, excluding audio_data
    """
    # Create metadata list without audio_data
    metadata = []
    for event in events:
        # Create a copy of the event dict without audio_data
        event_meta = {k: v for k, v in event.items() if k != 'audio_data'}
        
        # Convert datetime objects to ISO format strings for JSON serialization
        event_meta['start_timestamp'] = event_meta['start_timestamp'].isoformat()
        event_meta['end_timestamp'] = event_meta['end_timestamp'].isoformat()
        
        metadata.append(event_meta)

    top = {
        'audio': metadata
    }
    
    # Create JSON filename from original filename
    json_filename = original_filename.rsplit('.', 1)[0] + '.json'
    remote_path = f"{remote_dir}/{json_filename}"
    
    # Create JSON string
    json_str = json.dumps(top, indent=2)
    
    # Upload using SFTP
    with sftp.file(remote_path, 'w') as f:
        f.write(json_str)
    
    return remote_path



def train_find_audio_events_tonor(sftp, remote_dir):
    tests = {
        "recording_20250404_011856.opus": {
            "expected_events": 0
        },
        "recording_20250404_013557.opus": {
            "expected_events": 0
        },
        "recording_20250404_011125.opus": {
            "expected_events": 0
        },
        "recording_20250403_212804.opus": {
            "min_events": 10  # Now works independently
        },
        "recording_20250403_212734.opus": {
            "min_events": 8
        },
        "recording_20250405_010934.opus": {
            "expected_events": 0
        },
        "recording_20250405_011034.opus": {
            # Snort or something
            "min_events": 1
        },
        'recording_20250404_220046.opus': {
            # Blink sync
            "min_events": 1
        },
        'recording_20250404_220116.opus': {
            # Blink sync
            "min_events": 3 # Arbitrary
        },
        'recording_20250402_022414.opus': {
            # Weird glitch
            "min_events": 1
        },
        'recording_20250402_022445.opus': {
            # Weird glitch
            "min_events": 1
        },
        'recording_20250402_022615': {
            # Something!  Mayeb car at end
            "min_events": 2
        },
        'recording_20250402_022645': {
            # Maybe same car
            "min_events": 1
        },
        'recording_20250402_023415': {
            # Soemthing! and glitches
            "min_events": 2
        },
        'recording_20250402_023445': {
            # Soemthing! and glitches
            "min_events": 2
        },
        
        
        
    }
    talking = [
        'recording_20250401_210915.opus',
        'recording_20250401_210945.opus'
    ]
    for t in talking:
        tests[t] = {
            "min_events": 3 # Arbitrary
        }

    silence = [
        'recording_20250405_011104.opus'
        ]
    for s in silence:
        tests[s] = {
            "expected_events": 0
        }
    
    # Parameter ranges to search
    param_grid = {
        "window_size": [512, 1024, 2048],
        "threshold_multiplier": [1.2, 1.5, 1.8],
        "min_event_duration": [0.05, 0.1, 0.2],
        "merge_distance": [0.3, 0.5, 0.7]
    }
    
    # Cache for audio data
    audio_cache = {}
    
    # Load all audio files once
    for filename in tests.keys():
        if filename not in audio_cache:
            copied = copy_audio_file(sftp, remote_dir, filename)
            data_array, ogg, temp_filename = load_audio_as_ogg(copied.getvalue())
            audio_cache[filename] = {
                "data_array": data_array,
                "sample_rate": ogg.frequency
            }
    
    best_params = None
    best_error = float('inf')
    
    # Try all parameter combinations
    for window_size in param_grid["window_size"]:
        for threshold in param_grid["threshold_multiplier"]:
            for min_duration in param_grid["min_event_duration"]:
                for merge_dist in param_grid["merge_distance"]:
                    total_error = 0
                    
                    # Test current parameters on all test cases
                    for filename, expected in tests.items():
                        cached_audio = audio_cache[filename]
                        
                        events = find_audio_events(
                            cached_audio["data_array"], 
                            cached_audio["sample_rate"], 
                            filename,
                            window_size=window_size,
                            threshold_multiplier=threshold,
                            min_event_duration=min_duration,
                            merge_distance=merge_dist
                        )
                        
                        num_events = len(events)
                        error = 0
                        
                        # Calculate error based on test requirements
                        if "expected_events" in expected:
                            # Exact number of events required
                            error = abs(num_events - expected["expected_events"])
                        
                        if "expected_non_zero_events" in expected and num_events == 0:
                            # Add error if we expect non-zero events but found none
                            error += 1
                            
                        if "min_events" in expected and num_events < expected["min_events"]:
                            # Add error for falling short of minimum events
                            error += expected["min_events"] - num_events
                        
                        total_error += error
                        
                        # Early stopping if this parameter set isn't going to be better
                        if total_error >= best_error:
                            break
                    
                    # Update best parameters if current combination is better
                    if total_error < best_error:
                        best_error = total_error
                        best_params = {
                            "window_size": window_size,
                            "threshold_multiplier": threshold,
                            "min_event_duration": min_duration,
                            "merge_distance": merge_dist
                        }
                        
                        # If we found perfect parameters, stop searching
                        if total_error == 0:
                            print("Found perfect parameters!")
                            print(best_params)
                            print("\nEvents found in each file:")
                            for filename, expected in tests.items():
                                events = find_audio_events(
                                    audio_cache[filename]["data_array"],
                                    audio_cache[filename]["sample_rate"],
                                    filename,
                                    **best_params
                                )
                                print(f"{filename}: {len(events)} events")
                                if "min_events" in expected:
                                    print(f"  (minimum required: {expected['min_events']})")
                            return best_params
    
    print("Best parameters found (but not perfect):")
    print(best_params)
    print(f"Total error: {best_error}")
    
    print("\nEvents found in each file:")
    for filename, expected in tests.items():
        events = find_audio_events(
            audio_cache[filename]["data_array"],
            audio_cache[filename]["sample_rate"],
            filename,
            **best_params
        )
        print(f"{filename}: {len(events)} events")
        if "min_events" in expected:
            print(f"  (minimum required: {expected['min_events']})")
    
    return best_params


def train_find_audio_events_new_mic(sftp, remote_dir):
    tests = {
        "recording_20250513_054004.opus": {
            "min_events": 2
        },
        
    }
    talking = [
    ]
    for t in talking:
        tests[t] = {
            "min_events": 3 # Arbitrary
        }

    very_subtle_noises = [
        "recording_20250513_054835.opus"
    ]
    for v in very_subtle_noises:
        tests[v] = {
            "min_events": 1
        }
    silence = [
        "recording_20250513_054535.opus"
    ]
    for s in silence:
        tests[s] = {
            "expected_events": 0
        }
    
    # Parameter ranges to search
    param_grid = {
        "amplitude_threshold": [160, 170, 180, 190, 200, 210, 220, 230, 240, 250],
        "min_event_duration": [0.05, 0.1, 0.2],
        "merge_distance": [0.3, 0.5, 0.7]
    }
    
    # Cache for audio data
    audio_cache = {}
    
    # Load all audio files once
    for filename in tests.keys():
        if filename not in audio_cache:
            copied = copy_audio_file(sftp, remote_dir, filename)
            data_array, ogg, temp_filename = load_audio_as_ogg(copied.getvalue())
            audio_cache[filename] = {
                "data_array": data_array,
                "sample_rate": ogg.frequency
            }
    
    best_params = None
    best_error = float('inf')

    # Try all parameter combinations
    for amplitude_threshold in param_grid["amplitude_threshold"]:
        for min_duration in param_grid["min_event_duration"]:
            for merge_dist in param_grid["merge_distance"]:
                total_error = 0
                
                # Test current parameters on all test cases
                for filename, expected in tests.items():
                    cached_audio = audio_cache[filename]
                    
                    events = find_audio_events_amplitude(
                        cached_audio["data_array"], 
                        cached_audio["sample_rate"], 
                        filename,
                        amplitude_threshold=amplitude_threshold,
                        min_event_duration=min_duration,
                        merge_distance=merge_dist,
                        visualize=False
                    )
                    
                    num_events = len(events)
                    error = 0
                    
                    # Calculate error based on test requirements
                    if "expected_events" in expected:
                        # Exact number of events required
                        error = abs(num_events - expected["expected_events"])
                    
                    if "expected_non_zero_events" in expected and num_events == 0:
                        # Add error if we expect non-zero events but found none
                        error += 1
                        
                    if "min_events" in expected and num_events < expected["min_events"]:
                        # Add error for falling short of minimum events
                        error += expected["min_events"] - num_events
                    
                    total_error += error
                    
                    # Early stopping if this parameter set isn't going to be better
                    if total_error >= best_error:
                        break
                
                # Update best parameters if current combination is better
                if total_error < best_error:
                    best_error = total_error
                    best_params = {
                        "amplitude_threshold": amplitude_threshold,
                        "min_event_duration": min_duration,
                        "merge_distance": merge_dist
                    }
                    
                    # If we found perfect parameters, stop searching
                    if total_error == 0:
                        print("Found perfect parameters!")
                        print(best_params)
                        print("\nEvents found in each file:")
                        for filename, expected in tests.items():
                            events = find_audio_events_amplitude(
                                audio_cache[filename]["data_array"],
                                audio_cache[filename]["sample_rate"],
                                filename,
                                **best_params,
                                visualize=False
                            )
                            print(f"{filename}: {len(events)} events")
                            if "min_events" in expected:
                                print(f"  (minimum required: {expected['min_events']})")
                        return best_params

    print("Best parameters found (but not perfect):")
    print(best_params)
    print(f"Total error: {best_error}")
    
    print("\nEvents found in each file:")
    for filename, expected in tests.items():
        events = find_audio_events_amplitude(
            audio_cache[filename]["data_array"],
            audio_cache[filename]["sample_rate"],
            filename,
            **best_params,
            visualize=False
        )
        print(f"{filename}: {len(events)} events")
        if "min_events" in expected:
            print(f"  (minimum required: {expected['min_events']})")
    
    return best_params


def play_audio_widget(data_array, ogg, duration=None):
    """
    Create an interactive audio player widget in Jupyter notebook
    """
    if duration:
        samples = int(duration * ogg.frequency)
        data_to_play = data_array[:samples]
    else:
        data_to_play = data_array
    
    # Print information about the audio data
    print(f"Audio data - min: {np.min(data_to_play)}, max: {np.max(data_to_play)}")
    
    # Normalize the audio data if values are too large
    # max_abs_val = np.max(np.abs(data_to_play))
    # if max_abs_val > 100:
    #     print(f"Audio data has unusually large values (max abs: {max_abs_val}), normalizing")
    #     # Normalize to range [-1, 1] before applying gain
    #     data_to_play = data_to_play / max_abs_val
    #     print(f"After normalization - min: {np.min(data_to_play)}, max: {np.max(data_to_play)}")
    
    # Convert to WAV in memory
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, data_to_play, ogg.frequency, 
             format='WAV',
             subtype='PCM_16')
    wav_buffer.seek(0)
    
    # Create audio widget with WAV data
    audio = Audio(data=wav_buffer.read(), 
                 rate=ogg.frequency)
    
    # Create gain slider
    gain_slider = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=5.0,
        step=0.1,
        description='Gain:',
        continuous_update=False
    )
    
    # Create output widget for the audio player
    player_output = widgets.Output()
    
    def update_gain(change):
        with player_output:
            player_output.clear_output()
            # Apply gain and maintain original dtype
            adjusted_data = (data_to_play * change['new']).astype(data_to_play.dtype)
            
            # Convert to WAV
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, adjusted_data, ogg.frequency,
                    format='WAV',
                    subtype='PCM_16')
            wav_buffer.seek(0)
            
            display(Audio(data=wav_buffer.read(), rate=ogg.frequency))
    
    gain_slider.observe(update_gain, names='value')
    
    # Display initial audio player
    with player_output:
        display(audio)
    
    # Stack widgets vertically
    return widgets.VBox([gain_slider, player_output])


def find_audio_events_amplitude(audio_data, sample_rate, filename,
                     amplitude_threshold=200,
                     min_event_duration=0.1,
                     merge_distance=0.5,
                     figsize=(15, 10),
                     visualize=True):
    """
    Find periods of audio that exceed a specific amplitude threshold.
    Returns list of dicts with event details and displays visualizations of the process.
    
    Parameters:
    -----------
    audio_data : array_like
        The audio data array
    sample_rate : int
        Sample rate in Hz
    filename : str
        Original filename (used to parse timestamp)
    amplitude_threshold : float
        Absolute amplitude threshold for event detection
    min_event_duration : float
        Minimum duration of events in seconds
    merge_distance : float
        Maximum gap between events to merge them in seconds
    figsize : tuple
        Figure size for visualization (width, height) in inches
    visualize : bool
        Whether to generate and display visualizations
    
    Returns:
    --------
    list
        List of event dictionaries
    """
    # Parse recording start time from filename
    # Expected format: recording_YYYYMMDD_HHMMSS.opus
    timestamp_str = filename.split('recording_')[1].split('.')[0]
    recording_start = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    
    # Calculate max amplitude in windows
    window_size = 1024  # Fixed window size for amplitude calculation
    window_amplitudes = []
    window_times = []
    
    for i in range(0, len(audio_data), window_size):
        chunk = audio_data[i:i+window_size]
        if len(chunk) > 0:
            max_amp = np.max(np.abs(chunk))
            window_amplitudes.append(max_amp)
            window_times.append(i / sample_rate)
    
    if not window_amplitudes:
        return []
    
    amplitudes = np.array(window_amplitudes)
    
    # Find regions above threshold
    above_threshold = amplitudes > amplitude_threshold
    changes = np.diff(above_threshold.astype(int))
    start_indices = np.where(changes == 1)[0]
    end_indices = np.where(changes == -1)[0]
    
    if len(start_indices) == 0:
        if visualize:
            # Create visualization even if no events found
            plt.figure(figsize=figsize)
            
            # Plot 1: Original Audio Waveform
            plt.subplot(3, 1, 1)
            time_axis = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
            plt.plot(time_axis, audio_data)
            plt.title('Original Audio Waveform')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Max Amplitude per Window
            plt.subplot(3, 1, 2)
            plt.plot(window_times, amplitudes)
            plt.axhline(y=amplitude_threshold, color='r', linestyle='--', label=f'Threshold ({amplitude_threshold})')
            plt.title('Maximum Amplitude per Window')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Silence Detection
            plt.subplot(3, 1, 3)
            plt.plot(window_times, above_threshold.astype(int), drawstyle='steps-post')
            plt.title('Event Detection')
            plt.ylabel('Audio Event (1=Sound, 0=Silence)')
            plt.xlabel('Time (seconds)')
            plt.yticks([0, 1], ['Below Threshold', 'Above Threshold'])
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        return []
    
    if above_threshold[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if above_threshold[-1]:
        end_indices = np.append(end_indices, len(above_threshold) - 1)
    
    start_samples = start_indices * window_size
    end_samples = end_indices * window_size
    
    merge_samples = int(merge_distance * sample_rate)
    min_samples = int(min_event_duration * sample_rate)
    
    events = []
    current_start = start_samples[0]
    current_end = end_samples[0]
    
    # Track merged regions for visualization
    merged_starts = [current_start / sample_rate]
    merged_ends = []
    
    for i in range(1, len(start_samples)):
        if start_samples[i] - current_end < merge_samples:
            # This is a merge
            current_end = end_samples[i]
        else:
            merged_ends.append(current_end / sample_rate)
            if current_end - current_start >= min_samples:
                event_dict = create_event_dict(
                    audio_data[current_start:current_end],
                    current_start,
                    current_end,
                    sample_rate,
                    recording_start
                )
                events.append(event_dict)
            current_start = start_samples[i]
            current_end = end_samples[i]
            merged_starts.append(current_start / sample_rate)
    
    merged_ends.append(current_end / sample_rate)
    
    if current_end - current_start >= min_samples:
        event_dict = create_event_dict(
            audio_data[current_start:current_end],
            current_start,
            current_end,
            sample_rate,
            recording_start
        )
        events.append(event_dict)
    
    if visualize:
        # Create visualization
        plt.figure(figsize=figsize)
        
        # Plot 1: Original Audio Waveform with Events
        plt.subplot(4, 1, 1)
        time_axis = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
        plt.plot(time_axis, audio_data)
        
        # Mark final events
        for event in events:
            plt.axvspan(event['start_time'], event['end_time'], color='green', alpha=0.2)
        
        plt.title('Original Audio Waveform with Detected Events')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Max Amplitude per Window
        plt.subplot(4, 1, 2)
        plt.plot(window_times, amplitudes)
        plt.axhline(y=amplitude_threshold, color='r', linestyle='--', label=f'Threshold ({amplitude_threshold})')
        plt.title('Maximum Amplitude per Window')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Initial Event Detection
        plt.subplot(4, 1, 3)
        plt.plot(window_times, above_threshold.astype(int), drawstyle='steps-post')
        
        # Mark the raw detected events before merging
        for start_idx, end_idx in zip(start_indices, end_indices):
            start_time = start_idx * window_size / sample_rate
            end_time = end_idx * window_size / sample_rate
            plt.axvspan(start_time, end_time, color='blue', alpha=0.2)
        
        plt.title('Initial Event Detection')
        plt.ylabel('Above Threshold')
        plt.yticks([0, 1], ['No', 'Yes'])
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Final Events After Merging and Minimum Duration Filter
        plt.subplot(4, 1, 4)
        plt.plot(window_times, above_threshold.astype(int) * 0, drawstyle='steps-post')  # Empty plot for scale
        
        # Mark the merged events
        for start, end in zip(merged_starts, merged_ends):
            duration = end - start
            if duration >= min_event_duration:
                plt.axvspan(start, end, color='green', alpha=0.4, label='Accepted Event')
            else:
                plt.axvspan(start, end, color='red', alpha=0.2, label='Rejected (Too Short)')
        
        plt.title(f'Final Events (After Merging {merge_distance}s gaps and Min Duration {min_event_duration}s Filter)')
        plt.ylabel('Events')
        plt.xlabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.show()
    
    return events

