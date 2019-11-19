from timeit import timeit

import numpy as np

from voc import Voc, CHUNK

t = []
v = Voc(48000.0)
for i in range(5):
    t.append(timeit(lambda: v.compute(randomize=False), number=500))

print(np.average(t), np.std(t))

t = []
v = Voc(48000.0)
for i in range(5):
    t.append(timeit(lambda: v.compute(randomize=False, use_np=True), number=500))

print(np.average(t), np.std(t))