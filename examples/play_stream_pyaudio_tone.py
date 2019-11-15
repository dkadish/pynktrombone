"""PyAudio Example: Play a wave file (callback version)."""

import pyaudio
import time
import sys
import numpy as np

starting = 0
samplerate = 48000

# instantiate PyAudio (1)
p = pyaudio.PyAudio()


# define callback (2)
def callback(in_data, frame_count, time_info, status):
    global starting

    if status:
        print(status)
    j = starting + np.arange(frame_count, dtype=np.float32) * 2.0 * np.pi * 400.0 / samplerate
    wave = np.sin(j)
    data = np.repeat(wave.reshape(-1, 1), 2, axis=1)
    starting = j[-1]

    print(data.shape)

    return (data, pyaudio.paContinue)

# open stream using callback (3)
stream = p.open(format=pyaudio.paFloat32,
                channels=2,
                rate=samplerate,
                output=True,
                stream_callback=callback)

# start the stream (4)
stream.start_stream()

# wait for stream to finish (5)
while stream.is_active():
    time.sleep(0.1)

# stop stream (6)
stream.stop_stream()
stream.close()

# close PyAudio (7)
p.terminate()