import numpy as np

import sounddevice as sd

from voc import Voc

import hashlib

sd.default.samplerate = 44100

def main():
    vocal = Voc(sd.default.samplerate)

    outdata = np.array([], dtype=np.float32)

    for i in range(100):
        _osc = 0
        _voc = 0
        if vocal.counter == 0:
            _osc = 12 + 16 * (0.5 * (_osc + 1));
            vocal.tongue_shape(_osc, 2.9)

        out = np.array(vocal.compute(randomize=False), dtype=np.float32)

        outdata = np.append(outdata, out.reshape(-1, 1))

    m = hashlib.sha256()
    m.update(outdata)
    print(m.digest())

    # assert m.digest() == b'8i\xa9\t\x0e\xc2f\xc4\x03na\xeb\xcb\xe8\x89\x92\xde\xf2\x8a\xdb\xbcl\xb8,(\x93\xf5\x16\xd9\xb0S\xec'

    outdata = np.repeat(outdata.reshape(-1, 1), 2, axis=1)

    print(outdata, outdata.shape)

    sd.play(outdata, samplerate=sd.default.samplerate, blocking=True)

    print("Done")


if __name__ == '__main__':
    main()
