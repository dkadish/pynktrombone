"""PyAudio Example: Play a wave file (callback version)."""

import pyaudio
import time
import numpy as np

from pynkTrombone.voc import Voc

starting = 0
samplerate = 48000

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

vocal = Voc(samplerate)

# define callback (2)
def callback(in_data, frame_count, time_info, status):
    # if status:
    print(status)

    _osc = 0
    _voc = 0
    if vocal.counter == 0:
        _osc = 12 + 16 * (0.5 * (_osc + 1))
        vocal.tongue_shape(_osc, 2.9)

    out = np.array(vocal.compute(), dtype=np.float32)[:frame_count]

    data = np.repeat(out.reshape(-1, 1), 2, axis=1)

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
try:
    while True:
        time.sleep(0.1)
except KeyError:
    pass

# stop stream (6)
stream.stop_stream()
stream.close()

# close PyAudio (7)
p.terminate()