import numpy as np

import sounddevice as sd

from voc import Voc

import hashlib

sd.default.samplerate = 44100


def remap(value, max_out, min_out, max_in=1.0, min_in=-1.0):
    return (max_out - min_out) * (value + 1.0) / 2.0 + min_out


def main():
    vocal = Voc(sd.default.samplerate)

    outdata = np.array([], dtype=np.float32)

    # minimum = [0.0, 200.0, 0.6, 12.0, 2.0, 0.01]
    # maximum = [1.0, 800.0, 0.9, 30.0, 3.5, 0.04]
    # [touch, frequency, tenseness, tongue_index, tongue_diameter, velum] = activations

    for i in range(200):
        # Tenseness
        t = remap(np.sin(i / 30.0), 0.9, 0.6)
        # vocal.tenseness = t

        # Velum
        v = remap(np.sin(i / 20.0), 0.05, 0.00)
        # vocal.velum = v

        # Tongue
        td = remap(np.sin(i / 5.0), 4.0, 2.0)
        ti = remap(np.sin(i / 50.0), 32.0, 10.0)
        vocal.tongue_shape(ti, td)

        print(t, v, td)
        out = np.array(vocal.compute(randomize=True), dtype=np.float32)

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
