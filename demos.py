from math import sin
from typing import Tuple, List

import numpy as np
import soundfile as sf

from pynkTrombone.voc import voc_demo_d, Mode, Voc


# static int callme( void * outputBuffer, void * inputBuffer, unsigned int numFrames,
#             double streamTime, RtAudioStreamStatus status, void * data )
def callme(output: List[float], numFrames: int, data: voc_demo_d) -> Tuple[List[float]]:
    i: int
    vd: voc_demo_d = data  # Former pointer
    tmp: float = 0.0  # TODO redundant

    for i in range(numFrames):
        if vd.voc.counter == 0:
            print("Counter Zero")
            if vd.mode == Mode.VOC_TONGUE:
                vd.voc.tongue_shape(vd.tongue_pos, vd.tongue_diam)

        tmp = vd.voc.step()
        tmp *= vd.gain
        output[i, 0] = tmp
        output[i, 1] = tmp

    return output


def setup():
    buffer_frames: int = 1024

    vdd: voc_demo_d = voc_demo_d()
    vdd.freq = 160

    return vdd


def tongue_index():

    def ti(voc, x):
        idx = sin(x * 0.05) * 9 + 21
        diam = 2.75
        voc.tongue_shape(idx, diam)
        print(idx, diam)
        return voc

    play_update(ti, 'data/tongue_index_12-30.wav')

def tongue_diameter():

    def td(voc, x):
        idx = sin(0) * 9 + 21
        # diam = sin(x * 0.05) * 1.5 / 2 + 2.75
        diam = sin(x * 0.05) * 3.5 / 2.0 + 3.5/2.0
        voc.tongue_shape(idx, diam)
        print(idx, diam)
        return voc

    play_update(td, 'data/tongue_diameter_0-3.5.wav')

def velum():

    def v(voc, x):
        _v = sin(x * 0.04) * 0.5 + 0.5
        # _v = sin(x * 0.02) * 0.02 + 0.02
        voc.velum = _v
        print(_v)
        return voc

    play_update(v, 'data/velum_0-1.wav')


LIP_START = 39


def lips():
    def t(voc, x):
        _t = sin(x * 0.05) * 3.5 / 2.0 + 3.5 / 2.0
        voc.tract.lips = _t
        print(_t)
        return voc

    play_update(t, 'data/lips_0-3.5.wav')

def lips_open_shut():
    vdd = setup()
    x = 0
    lips = sin(x) * 1.5 / 2.0 + 0.75
    n = len(vdd.voc.tract_diameters[39:])
    vdd.voc.tract_diameters[39:] = [lips for _ in range(n)]
    # with sd.OutputStream(samplerate=vdd.samplerate, channels=2, blocksize=1024, callback=partial(sd_callback, vdd=vdd)):
    #     sd.sleep(int(5 * 1000))

    output = np.empty(shape=(1024, 2))
    out = callme(output, 1024, vdd)
    while out.shape[0] < vdd.sr * 5:
        output = np.empty(shape=(1024, 2))
        out = np.concatenate([out, callme(output, 1024, vdd)])

        x += 0.5
        lips = sin(x) * 1.5 / 2.0 + 0.75
        vdd.voc.tract_diameters[39:] = [lips for _ in range(n)]
        # vdd.self.frequency += 0.5

        print(out.shape, lips)

    sf.write('data/lips_move_0-1.5.wav', out, vdd.sr)

def throat_open_shut():
    def t(voc, x):
        _t = sin(x * 0.05) * 3.5 / 2.0 + 3.5 / 2.0
        voc.tract.trachea = _t
        print(_t)
        return voc

    play_update(t, 'data/trachea_0-3.5.wav')

def throat_and_lips():
    vdd = setup()

    x = y = 0
    throat = sin(x) * 1.5 / 2.0 + 0.75
    n_t = len(vdd.voc.tract_diameters[:7])
    vdd.voc.tract_diameters[:7] = [throat for _ in range(n_t)]

    lips = sin(x) * 1.5 / 2.0 + 0.75
    n_l = len(vdd.voc.tract_diameters[39:])
    vdd.voc.tract_diameters[39:] = [lips for _ in range(n_l)]

    output = np.empty(shape=(1024, 2))
    out = callme(output, 1024, vdd)
    while out.shape[0] < vdd.sr * 5:
        output = np.empty(shape=(1024, 2))
        out = np.concatenate([out, callme(output, 1024, vdd)])

        x += 0.55
        lips = sin(x) * 1.5 / 2.0 + 0.75
        vdd.voc.tract_diameters[39:] = [lips for _ in range(n_l)]

        y += 0.5
        throat = sin(y) * 1.5 / 2.0 + 0.75
        vdd.voc.tract_diameters[:7] = [throat for _ in range(n_t)]
        # vdd.self.frequency += 0.5

        print(out.shape, throat)

    sf.write('data/throat_and_lips.wav', out, vdd.sr)

def frequency():
    sr: float = 44100
    voc = Voc(sr,default_freq=140)
    x = 0
    freq = sin(x) * 60 + 120
    voc.frequency = freq

    out = voc.play_chunk()
    while out.shape[0] < sr * 5:
        out = np.concatenate([out, voc.play_chunk()])

        x += 0.1
        freq = sin(x) * 60 + 120
        voc.frequency = freq

        print(out.shape, freq)

    sf.write('data/frequency_60-180.wav', out, sr)

def tenseness():
    def t(voc, x):
        _t = sin(x * 0.01) * 0.5 + 0.5
        voc.tenseness = _t
        print(_t)
        return voc

    play_update(t, 'data/tenseness_0-1.wav')

def epiglottis():
    def t(voc, x):
        _t = sin(x * 0.05) * 3.5/2.0 + 3.5/2.0
        voc.tract.epiglottis = _t
        print(_t)
        return voc

    play_update(t, 'data/epiglottis_0-3.5.wav')


def nothing():
    def t(voc, x):
        return voc

    play_update(t, 'data/nothing.wav')

def play_update(update_fn, filename):
    sr: float = 44100
    voc = Voc(sr)
    x = 0
    voc = update_fn(voc, x)

    out = voc.play_chunk()
    while out.shape[0] < sr * 5:
        out = np.concatenate([out, voc.play_chunk()])

        x += 1
        voc = update_fn(voc, x)

    sf.write(filename, out, sr)

def main():
    vdd = setup()
    idx, diam, x, y = sin(0) * 2.5 + 23, sin(0) * 1.5 / 2 + 2.75, 0.0, 0.0
    vdd.voc.tongue_shape(idx, diam)
    # with sd.OutputStream(samplerate=vdd.samplerate, channels=2, blocksize=1024, callback=partial(sd_callback, vdd=vdd)):
    #     sd.sleep(int(5 * 1000))

    output = np.empty(shape=(1024, 2))
    out = callme(output, 1024, vdd)
    while out.shape[0] < vdd.sr * 5:
        output = np.empty(shape=(1024, 2))
        out = np.concatenate([out, callme(output, 1024, vdd)])

        x += 0  # .1
        y += 0.1
        idx, diam = sin(x) * 2.5 + 23, sin(y) * 1.5 / 2 + 2.75
        vdd.voc.tongue_shape(idx, diam)
        # vdd.self.frequency += 0.5

        print(out.shape, idx, diam)

    sf.write('data/stereo_file.wav', out, vdd.sr)


if __name__ == '__main__':
    # throat_and_lips()
    # tongue_index()
    # tongue_diameter()
    # lips_open_shut()
    # throat_open_shut()
    # frequency()
    # tenseness()
    # velum()
    # epiglottis()
    # lips()
    nothing()