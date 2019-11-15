import numpy as np

import sounddevice as sd

from voc import Voc, CHUNK

sd.default.samplerate = 48000

duration = 5.0  # seconds

vocal = Voc(sd.default.samplerate)

_osc = 0
_voc = 0
if vocal.counter == 0:
    _osc = 12 + 16 * (0.5 * (_osc + 1))
    vocal.tongue_shape(_osc, 2.9)

for i in range(500):
    vocal.compute(randomize=False)