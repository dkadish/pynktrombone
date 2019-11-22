import random
from timeit import timeit

import numpy as np
import sounddevice as sd

from scipy.io import wavfile

from voc import Voc, CHUNK

samplerate = 48000

duration = 2.0  # seconds

def main():

    vocal = Voc(samplerate)
    output = []

    while len(output)*CHUNK < samplerate * duration:
        _osc = 0
        _voc = 0
        if vocal.counter == 0:
            _osc = 12 + 16 * (0.5 * (_osc + 1))
            vocal.tongue_shape(_osc, 2.9)

        out = np.array(vocal.compute(), dtype=np.float32)[:CHUNK]

        output.append(out.reshape(-1))

        # try:
        #     vocal.frequency += random.randint(-5,5)
        # except ValueError as e:
        #     pass
        #
        # print(vocal.frequency)

        # if random.randint(0,100) > 90:
        #     print('DISABLED')
        #     vocal.glottis_disable()
        # else:
        #     print('ENABLED')

    # wavfile.write('test.wav', samplerate, np.concatenate(output))

    sd.play(np.concatenate(output), samplerate=samplerate, blocking=True)

if __name__ == '__main__':
    t = timeit(main, number=1)
    print(t)