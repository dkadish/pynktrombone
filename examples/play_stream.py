import numpy as np

import sounddevice as sd

from voc import Voc, CHUNK

sd.default.samplerate = 48000

duration = 5.0  # seconds

def main():

    vocal = Voc(sd.default.samplerate)

    def process(outdata, frames, time, status):

        if status:
            print(status)

        _osc = 0
        _voc = 0
        if vocal.counter == 0:
            _osc = 12 + 16 * (0.5 * (_osc + 1))
            vocal.tongue_shape(_osc, 2.9)

        out = np.array(vocal.compute(), dtype=np.float32)[:CHUNK]

        print(outdata.shape, out.shape)

        outdata[:] = out.reshape(-1, 1)


    with sd.OutputStream(channels=1, callback=process, blocksize=CHUNK, samplerate=sd.default.samplerate) as ostream:
        print(ostream.cpu_load)
        sd.sleep(int(duration * 1000))
        print(ostream.cpu_load)


if __name__ == '__main__':
    main()