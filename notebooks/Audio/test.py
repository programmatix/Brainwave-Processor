import pyogg
import librosa

print("Loading opus file")
ogg = pyogg.OpusFile("C:\\Users\\graha\\AppData\\Local\\Temp\\tmpxxtfwd25.opus")

ogg.as_array()
print("Converting to numpy array")
import numpy as np

#data_array = np.ctypeslib.as_array(ogg.buffer, shape=(ogg.buffer_length,))
buffer_copy = np.frombuffer(bytes(ogg.buffer[:ogg.buffer_length]), dtype=np.int16)
print("Buffer copy shape: ", buffer_copy.shape)
# Normalize the audio data to float32 in range [-1.0, 1.0]
data_array = buffer_copy.astype(np.float32) / 32768.0  # Normalize 16-bit audio

print("Data array shape:", data_array.shape)
print("Data array dtype:", data_array.dtype)
print("Data array min/max values:", data_array.min(), data_array.max())
print("Checking for silence")
def is_silence_in_opus(audio_data, sample_rate, threshold_db=-50, min_silence_duration=1.0):
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        print("here")
        audio_data = np.mean(audio_data, axis=1)
    
    print("Audio data shape: ", audio_data.shape)
    
    # Convert to float
    audio_data = audio_data.astype(np.float32)
    print("Audio data type: ", audio_data.dtype)
    
    # Get the amplitude envelope
    amplitude_envelope = np.abs(audio_data)
    print("Amplitude envelope shape: ", amplitude_envelope.shape)
    
    # Convert to dB
    db = librosa.amplitude_to_db(amplitude_envelope, ref=np.max)
    print("DB shape: ", db.shape)
    # Find regions below threshold
    silence_mask = db < threshold_db
    print("Silence mask shape: ", silence_mask.shape)
    
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

try:
    out = is_silence_in_opus(data_array, ogg.frequency)
    print("Contains silence: ", out[0])
    print("Silence intervals: ", out[1])
    print("Percent silence: ", out[2])
except Exception as e:
    import traceback
    print("Error: ", e)
    print(traceback.format_exc())
