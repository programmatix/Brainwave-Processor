import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
from scipy import signal
import time

# Global variables
audio_buffer = None
sample_rate = 44100  # Hz
line = None
ax = None
update_count = 0
callback_count = 0
fig = None
line_unfiltered = None
slider_ax = None
quality_factor = 30.0

def find_umc_device():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if 'UMC' in device['name']:
            return i
    return None

def apply_notch_filter(data, sample_rate, notch_freq=50.0, quality_factor=30.0):
    b, a = signal.iirnotch(notch_freq, quality_factor, sample_rate)
    return signal.lfilter(b, a, data)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    
    global audio_buffer, callback_count, quality_factor
    callback_count += 1
    if callback_count % 10 == 0:
        print(f"Audio callback called {callback_count} times. Buffer shape: {audio_buffer.shape}, indata shape: {indata.shape}")
        print(f"Audio data range: {np.min(indata):.4f} to {np.max(indata):.4f}")
    
    # Store unfiltered data for plotting
    unfiltered_data = indata[:, 0]
    
    # Apply notch filter with current quality_factor
    filtered_data = apply_notch_filter(unfiltered_data, sample_rate, quality_factor=quality_factor)
    audio_buffer = np.roll(audio_buffer, -len(filtered_data), axis=0)
    audio_buffer[-len(filtered_data):] = filtered_data
    
    # Update unfiltered plot
    f_unfiltered, Pxx_unfiltered = signal.periodogram(unfiltered_data, fs=sample_rate)
    mask_unfiltered = f_unfiltered <= 200
    line_unfiltered.set_data(f_unfiltered[mask_unfiltered], 10 * np.log10(Pxx_unfiltered[mask_unfiltered] + 1e-10))
    
    # Update filtered plot
    f_filtered, Pxx_filtered = signal.periodogram(filtered_data, fs=sample_rate)
    mask_filtered = f_filtered <= 200
    line.set_data(f_filtered[mask_filtered], 10 * np.log10(Pxx_filtered[mask_filtered] + 1e-10))

def update_plot(frame):
    global audio_buffer, line, ax, sample_rate, update_count, line_unfiltered, quality_factor
    update_count += 1

    print(f"Update plot called {update_count} times")
    
    # Calculate periodogram for unfiltered data (stored in audio_buffer)
    f_unfiltered, Pxx_unfiltered = signal.periodogram(audio_buffer, fs=sample_rate)
    mask_unfiltered = f_unfiltered <= 200
    line_unfiltered.set_data(f_unfiltered[mask_unfiltered], 10 * np.log10(Pxx_unfiltered[mask_unfiltered] + 1e-10))
    
    # Calculate periodogram for filtered data
    f_filtered, Pxx_filtered = signal.periodogram(audio_buffer, fs=sample_rate)
    mask_filtered = f_filtered <= 200
    line.set_data(f_filtered[mask_filtered], 10 * np.log10(Pxx_filtered[mask_filtered] + 1e-10))
    
    # Calculate y-limits with some margin (for both plots)
    min_db = min(np.min(10 * np.log10(Pxx_unfiltered + 1e-10)), np.min(10 * np.log10(Pxx_filtered + 1e-10))) - 10
    max_db = max(np.max(10 * np.log10(Pxx_unfiltered + 1e-10)), np.max(10 * np.log10(Pxx_filtered + 1e-10))) + 10
    ax.set_ylim([min_db, max_db])
    line_unfiltered.axes.set_ylim([min_db, max_db])
    
    if update_count % 5 == 0:
        print(f"Audio buffer stats - min: {np.min(audio_buffer):.4f}, max: {np.max(audio_buffer):.4f}")
        print(f"Frequency range: {np.min(f_unfiltered):.1f} to {np.max(f_unfiltered):.1f} Hz")
        print(f"PSD range (unfiltered): {np.min(Pxx_unfiltered):.6f} to {np.max(Pxx_unfiltered):.6f}")
        print(f"PSD range (filtered): {np.min(Pxx_filtered):.6f} to {np.max(Pxx_filtered):.6f}")
        print(f"Y-axis limits: {min_db:.1f} to {max_db:.1f} dB")
        print("-" * 50)
    
    return line, line_unfiltered

def setup_plot():
    global fig, ax, line, line_unfiltered, slider_ax, quality_factor
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    
    # Subplot for unfiltered signal
    ax_unfiltered = fig.add_subplot(gs[0, 0])
    ax_unfiltered.set_xlim(0, 200)
    ax_unfiltered.set_xlabel('Frequency (Hz)')
    ax_unfiltered.set_ylabel('Power Spectral Density (dB/Hz)')
    ax_unfiltered.set_title('Unfiltered Signal')
    ax_unfiltered.grid(True)
    line_unfiltered, = ax_unfiltered.plot([], [], lw=2, color='blue')
    
    # Subplot for filtered signal
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 200)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (dB/Hz)')
    ax.set_title('Filtered Signal')
    ax.grid(True)
    line, = ax.plot([], [], lw=2, color='red')
    
    # Slider for quality factor
    slider_ax = fig.add_subplot(gs[1, :])
    quality_factor_slider = plt.Slider(
        slider_ax,
        'Quality Factor',
        1.0,
        100.0,
        valinit=30.0,
        valstep=1.0
    )
    
    def update_slider(val):
        global quality_factor
        quality_factor = val
    
    quality_factor_slider.on_changed(update_slider)
    quality_factor = 30.0
    
    print("Plot setup complete")
    return fig

def do_plot():
    # Audio parameters
    global audio_buffer, sample_rate, line, ax, fig
    buffer_duration = 0.5  # Reduced from 2 seconds
    buffer_size = int(sample_rate * buffer_duration)
    audio_buffer = np.zeros(buffer_size)
    
    print(f"Initialized audio buffer with size {buffer_size} ({buffer_duration} seconds at {sample_rate} Hz)")

    # Find UMC device
    device_id = find_umc_device()
    if device_id is None:
        print("No UMC device found. Using default input device.")
        device_id = sd.default.device[0]
    else:
        print(f"Found UMC device at index {device_id}")
    
    # Print device info
    devices = sd.query_devices()
    print(f"Using device: {devices[device_id]['name']}")
    print(f"Device channels: input={devices[device_id]['max_input_channels']}, output={devices[device_id]['max_output_channels']}")
    print(f"Device default samplerate: {devices[device_id]['default_samplerate']}")

    # Setup plot
    fig = setup_plot()

    # Start the audio stream
    stream = sd.InputStream(
        device=device_id,
        channels=1,
        samplerate=sample_rate,
        callback=audio_callback
    )
    print("Audio stream created")

    # Create animation with faster updates
    ani = FuncAnimation(
        fig, 
        update_plot, 
        frames=None,
        blit=True, 
        interval=50,  # Faster refresh (was 100)
        cache_frame_data=False
    )
    print("Animation created with interval=50ms")
    
    # Keep reference to animation to prevent garbage collection
    fig.ani = ani

    # Start streaming
    print("Starting audio stream and showing plot...")
    
    with stream:
        plt.show()  # This will block until plot window is closed
        print("Plot window closed - stopping stream")

if __name__ == "__main__":
    do_plot()