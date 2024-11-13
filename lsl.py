import numpy as np
from pylsl import StreamInlet, resolve_byprop
import threading
import time

class LSLReader:
    def __init__(self, stream_name='brainwave-lsl', buffer_duration=31):
        self.stream_name = stream_name
        self.buffer_duration = buffer_duration
        self.buffer = None
        self.sampling_rate = None
        self.inlet = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        streams = resolve_byprop('name', self.stream_name, timeout=5)
        if len(streams) == 0:
            raise RuntimeError(f"No stream found with name {self.stream_name}")

        self.inlet = StreamInlet(streams[0])
        self.sampling_rate = int(self.inlet.info().nominal_srate())
        nchannels = self.inlet.info().channel_count()
        nsamples = self.sampling_rate * self.buffer_duration
        self.buffer = np.zeros((nchannels, nsamples))
        print("Created LSL reader with buffer shape", self.buffer.shape)

        self.running = True
        self.thread = threading.Thread(target=self.update_buffer)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def update_buffer(self):
        while self.running:
            sample, timestamp = self.inlet.pull_sample()
            with self.lock:
                self.buffer = np.roll(self.buffer, -1, axis=1)
                self.buffer[:, -1] = sample
            time.sleep(1 / self.sampling_rate)

    def get_buffer(self):
        with self.lock:
            return self.buffer.copy()

if __name__ == "__main__":
    reader = LSLReader()
    reader.start()
    try:
        while True:
            time.sleep(1)
            print(reader.get_buffer())
    except KeyboardInterrupt:
        reader.stop()