from typing import Tuple, List

from math import sin

from pynkTrombone.two.voc import voc_demo_d, sp_voc_set_tongue_shape, sp_voc_compute, Mode

import numpy as np
import soundfile as sf

# static int callme( void * outputBuffer, void * inputBuffer, unsigned int numFrames,
#             double streamTime, RtAudioStreamStatus status, void * data )
def callme(output: List[float], numFrames: int, data: voc_demo_d) -> Tuple[List[float]]:
    i: int
    vd: voc_demo_d = data  # Former pointer
    tmp: float = 0.0  # TODO redundant

    for i in range(numFrames):
        if vd.voc.counter == 0:
            if vd.mode == Mode.VOC_TONGUE:
                vd.voc = sp_voc_set_tongue_shape(vd.voc,
                                                 vd.tongue_pos, vd.tongue_diam)

        vd.sp, vd.voc, tmp = sp_voc_compute(vd.sp, vd.voc, tmp)
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

    vdd = setup()
    idx, diam, x, y = sin(0)*2.5+23, sin(0)*1.5/2+2.75, 0.0, 0.0
    vdd.voc = sp_voc_set_tongue_shape(vdd.voc, idx, diam)

    output = np.empty(shape=(1024, 2))
    out = callme(output, 1024, vdd)
    while out.shape[0] < vdd.sp.sr * 5:
        output = np.empty(shape=(1024, 2))
        out = np.concatenate([out, callme(output, 1024, vdd)])

        x += 0.1
        y += 0#.1
        idx, diam = sin(x)*2.5+23, sin(y)*1.5/2+2.75
        vdd.voc = sp_voc_set_tongue_shape(vdd.voc, idx, diam)

        print(out.shape, idx, diam)

    sf.write('tongue_index_20.5-25.5.wav', out, vdd.sp.sr)

def tongue_diameter():

    vdd = setup()
    idx, diam, x, y = sin(0)*2.5+23, sin(0)*1.5/2+2.75, 0.0, 0.0
    vdd.voc = sp_voc_set_tongue_shape(vdd.voc, idx, diam)

    output = np.empty(shape=(1024, 2))
    out = callme(output, 1024, vdd)
    while out.shape[0] < vdd.sp.sr * 5:
        output = np.empty(shape=(1024, 2))
        out = np.concatenate([out, callme(output, 1024, vdd)])

        x += 0#.1
        y += 0.1
        idx, diam = sin(x)*2.5+23, sin(y)*1.5/2+2.75
        vdd.voc = sp_voc_set_tongue_shape(vdd.voc, idx, diam)

        print(out.shape, idx, diam)

    sf.write('tongue_diameter_2-3.5.wav', out, vdd.sp.sr)

LIP_START = 39

def lips_open_shut():
    vdd = setup()
    x = 0
    lips = sin(x) * 1.5/2.0 + 0.75
    n = len(vdd.voc.tract_diameters[39:])
    vdd.voc.tract_diameters[39:] = [lips for _ in range(n)]
    # with sd.OutputStream(samplerate=vdd.sp.sr, channels=2, blocksize=1024, callback=partial(sd_callback, vdd=vdd)):
    #     sd.sleep(int(5 * 1000))

    output = np.empty(shape=(1024, 2))
    out = callme(output, 1024, vdd)
    while out.shape[0] < vdd.sp.sr * 5:
        output = np.empty(shape=(1024, 2))
        out = np.concatenate([out, callme(output, 1024, vdd)])

        x += 0.1
        lips = sin(x) * 1.5 / 2.0 + 0.75
        vdd.voc.tract_diameters[39:] = [lips for _ in range(n)]
        # vdd.voc.frequency += 0.5

        print(out.shape, lips)

    sf.write('stereo_file.wav', out, vdd.sp.sr)

def main():
    vdd = setup()
    idx, diam, x, y = sin(0)*2.5+23, sin(0)*1.5/2+2.75, 0.0, 0.0
    vdd.voc = sp_voc_set_tongue_shape(vdd.voc, idx, diam)
    # with sd.OutputStream(samplerate=vdd.sp.sr, channels=2, blocksize=1024, callback=partial(sd_callback, vdd=vdd)):
    #     sd.sleep(int(5 * 1000))

    output = np.empty(shape=(1024, 2))
    out = callme(output, 1024, vdd)
    while out.shape[0] < vdd.sp.sr * 5:
        output = np.empty(shape=(1024, 2))
        out = np.concatenate([out, callme(output, 1024, vdd)])

        x += 0#.1
        y += 0.1
        idx, diam = sin(x)*2.5+23, sin(y)*1.5/2+2.75
        vdd.voc = sp_voc_set_tongue_shape(vdd.voc, idx, diam)
        # vdd.voc.frequency += 0.5

        print(out.shape, idx, diam)

    sf.write('stereo_file.wav', out, vdd.sp.sr)

if __name__ == '__main__':
    # tongue_index()
    # tongue_diameter()
    lips_open_shut()