import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.DEBUG)
from pynkTrombone.voc import Voc
import numpy as np
from math import sin
import soundfile as sf

CHUNK = 512
LENGTH = 1 * 44100
STEP = int(LENGTH/CHUNK)
#v = Voc()
#buf = v.compute()
#print('start')
#for i in range(STEP):
#    buf = v.compute()
#print("end")
#from NewPynkTrombone.glottis import Glottis

def tongue_index():

    def ti(voc, x):
        idx = sin(x * 0.05) * 9 + 21
        diam = 2.75
        voc.tongue_shape(idx, diam)
        print(idx, diam)
        return voc

    play_update(ti, 'tongue_index_12-30.wav')
def play_update(update_fn, filename):
    sr: float = 44100
    voc = Voc(sr,256)
    x = 0
    voc = update_fn(voc, x)

    out = voc.play_chunk()
    while out.shape[0] < sr * 5:
        out = np.concatenate([out, voc.play_chunk()])

        x += 1
        voc = update_fn(voc, x)

    sf.write(filename, out, sr)
#g = Glottis(44100)

if __name__ == "__main__":
    tongue_index()