from timeit import timeit

import numpy as np

from pynkTrombone.voc import Voc

n = int(round(48000 / 512)) # The number of times needed to produce a second of sound.

t = []
v = Voc(48000.0)
for i in range(5):
    t.append(timeit(lambda: v.compute(randomize=False), number=n))

print(np.average(t), np.std(t))

t = []
v = Voc(48000.0)
for i in range(5):
    t.append(timeit(lambda: v.compute(randomize=False, use_np=True), number=n))

print(np.average(t), np.std(t))