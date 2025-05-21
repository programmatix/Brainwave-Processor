import pyogg
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_audio_file(filepath):
    print(f"Analyzing file: {filepath}")
    
    # Load the audio file
    ogg = pyogg.OpusFile(filepath)
    data_array = ogg.as_array()
    
    # Print basic information
    print(f"Audio data shape: {data_array.shape}")
    print(f"Audio data type: {data_array.dtype}")
    print(f"Audio data range: {np.min(data_array)} to {np.max(data_array)}")
    print(f"Audio data mean: {np.mean(data_array)}")
    print(f"Audio data std dev: {np.std(data_array)}")
    print(f"Sample rate: {ogg.frequency} Hz")
    print(f"Duration: {len(data_array) / ogg.frequency:.2f} seconds")
    
    # Create time axis
    time = np.arange(0, len(data_array)) / ogg.frequency
    
    # Plot waveform
    plt.figure(figsize=(12, 8))
    
    # Original waveform
    plt.subplot(3, 1, 1)
    plt.plot(time, data_array)
    plt.title('Original Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.ylim(-100, 100)
    plt.grid(True)
    
    # Normalized waveform
    plt.subplot(3, 1, 2)
    max_abs = np.max(np.abs(data_array))
    if max_abs > 0:
        normalized = data_array / max_abs
        plt.plot(time, normalized)
        plt.title('Normalized Waveform [-1, 1]')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.ylim(-1.1, 1.1)
        plt.grid(True)
    
    # Histogram of values
    plt.subplot(3, 1, 3)
    plt.hist(data_array.flatten(), bins=50)
    plt.title('Histogram of Audio Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{os.path.basename(filepath)}_analysis.png")
    plt.show()
    
    return data_array, ogg

# Example usage
if __name__ == "__main__":
    # Check if a temporary opus file exists in the system temp directory
    import tempfile
    import glob
    
    temp_dir = tempfile.gettempdir()
    opus_files = glob.glob(os.path.join(temp_dir, "*.opus"))
    
    if opus_files:
        print(f"Found {len(opus_files)} opus files in temp directory")
        for i, file in enumerate(opus_files[:3]):  # Analyze up to 3 files
            print(f"\nFile {i+1}: {file}")
            analyze_audio_file(file)
    else:
        print("No opus files found in temp directory")
        print(f"Temp directory: {temp_dir}") 