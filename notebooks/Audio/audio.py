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
    #import numpy as np
    #data_array = np.ctypeslib.as_array(ogg.buffer, shape=(ogg.buffer_length,))
    data_array = ogg.as_array()
    # Remove the first second of audio
    data_array = data_array[ogg.frequency:]
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
    display_waveform(data_array, ogg.frequency)
    return data_array, ogg


import simpleaudio as sa # type: ignore

def play_audio(data_array, ogg, duration=None, gain=1.0):
    # Calculate samples for duration if specified
    if duration:
        samples = int(duration * ogg.frequency)
        print(f"Playing {samples} samples")
        data_to_play = data_array[:samples]
    else:
        data_to_play = data_array

    # Apply gain by multiplying the audio data
    data_to_play = (data_to_play * gain).astype(data_to_play.dtype)

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

def display_waveform(audio_data, sample_rate, figsize=(12, 4)):
    """
    Display the waveform for an Opus audio buffer.
    
    Args:
        opus_buffer (BytesIO): BytesIO object containing Opus audio data
        figsize (tuple): Figure size (width, height) in inches
    """
    # Convert to wav and get audio data
    #audio_data, sample_rate = opus_to_wav_buffer(opus_buffer)
    
    # Create time axis
    time = np.arange(0, len(audio_data)) / sample_rate
    
    # Plot waveform
    plt.figure(figsize=figsize)
    
    # if len(audio_data.shape) > 1:
    #     # Stereo
    #     plt.plot(time, audio_data[:, 0], label='Left channel', alpha=0.7)
    #     plt.plot(time, audio_data[:, 1], label='Right channel', alpha=0.7)
    #     plt.legend()
    # else:
        # Mono
    plt.plot(time, audio_data)
    
    plt.title('Audio Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Also display a spectrogram
    # plt.figure(figsize=figsize)
    
    # # Convert to mono if stereo
    # if len(audio_data.shape) > 1:
    #     audio_data = np.mean(audio_data, axis=1)
    
    # # Calculate spectrogram
    # D = librosa.amplitude_to_db(
    #     np.abs(librosa.stft(audio_data)), ref=np.max
    # )
    
    # # Plot spectrogram
    # librosa.display.specshow(
    #     D, sr=sample_rate, x_axis='time', y_axis='log'
    # )
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Spectrogram')
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


best_silence_detection_params = {'window_size': 2048,
 'threshold_multiplier': 1.5,
 'min_event_duration': 0.2,
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



def train_find_audio_events(sftp, remote_dir):
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

def play_audio_widget(data_array, ogg, duration=None):
    """
    Create an interactive audio player widget in Jupyter notebook
    """
    if duration:
        samples = int(duration * ogg.frequency)
        data_to_play = data_array[:samples]
    else:
        data_to_play = data_array
    
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

