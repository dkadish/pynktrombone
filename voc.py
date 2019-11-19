import math
from typing import List

import numpy as np

from glottis import Glottis, GlottisNP
from tools import sp_data
from tract import Tract

CHUNK = 512

class Voc:

    def __init__(self, sr: float):
        self.glot = Glottis(sr)  # FIXME rename to self.glottis
        self.tr = Tract(sr)  # FIXME rename to self.tract
        self._counter = 0

        self.glotnp = GlottisNP(sr)

    def compute(self, randomize: bool = True, use_np=False) -> List[float]:  # C Version returns an int and sets a referenced float *out. This returns *out instead.
        #TODO What if I just compute the next and store in a buffer?
        if use_np:
            self.glotnp.update(self.tr.block_time)
        else:
            self.glot.update(self.tr.block_time)

        self.tr.reshape()
        self.tr.calculate_reflections()

        if use_np:
            buf = self._compute_np(randomize)
        else:
            buf = self._compute(randomize)

        #################################################
        #
        # vocal_output = 0
        # lambda1 = np.arange(0,1,1/CHUNK)
        # lambda2 = np.arange((0+0.5) / CHUNK, (CHUNK+0.5) / CHUNK, 1 / CHUNK)
        # glot = self.glot.compute(lambda1, randomize)
        #
        # self.tr.compute(glot, lambda1)
        # vocal_output += self.tr.lip_output + self.tr.nose_output
        #
        # self.tr.compute(glot, lambda2)
        # vocal_output += self.tr.lip_output + self.tr.nose_output
        # buf = vocal_output * 0.125
        #
        #################################################

        # self.tr.compute(glot, lambda1)
        # vocal_output += self.tr.lip_output + self.tr.nose_output
        #
        # self.tr.compute(glot, lambda2)
        # vocal_output += self.tr.lip_output + self.tr.nose_output
        # buf.append(vocal_output * 0.125)

        return buf

    def _compute(self, randomize):
        buf = []

        for i in range(CHUNK):
            vocal_output = 0
            lambda1 = i / float(CHUNK)
            lambda2 = (i + 0.5) / float(CHUNK)
            glot = self.glot.compute(randomize)
            # glotnp = self.glotnp.compute(randomize)

            # print(i, glot, glotnp)

            self.tr.compute(glot, lambda1)
            vocal_output += self.tr.lip_output + self.tr.nose_output

            self.tr.compute(glot, lambda2)
            vocal_output += self.tr.lip_output + self.tr.nose_output
            buf.append(vocal_output * 0.125)

        return buf

    def _compute_np(self, randomize):
        buf = []

        vocal_output = np.zeros(shape=(CHUNK,))
        lambda1 = np.arange(CHUNK, dtype=float) / float(CHUNK)
        lambda2 = (np.arange(CHUNK, dtype=float) + 0.5) / float(CHUNK)
        glot = self.glotnp.compute_np(randomize)

        for i in range(CHUNK):
            self.tr.compute(glot[i], lambda1[i])
            vocal_output[i] += self.tr.lip_output + self.tr.nose_output

            self.tr.compute(glot[i], lambda2[i])
            vocal_output[i] += self.tr.lip_output + self.tr.nose_output
            buf.append(vocal_output[i] * 0.125)

        return buf

    def tract_compute(self, sp: sp_data, zin) -> float:
        if self._counter == 0:
            self.tr.reshape()
            self.tr.calculate_reflections()

        vocal_output = 0
        lambda1 = self._counter / float(CHUNK)
        lambda2 = (self._counter + 0.5) / float(CHUNK)

        self.tr.compute(zin, lambda1)
        vocal_output += self.tr.lip_output + self.tr.nose_output
        self.tr.compute(zin, lambda2)
        vocal_output += self.tr.lip_output + self.tr.nose_output

        out = vocal_output * 0.125
        self._counter = (self._counter + 1) % CHUNK
        return out

    # Unnecessary in Python
    def create(self):
        pass

    # Unnecessary in Python
    def destroy(self):
        pass

    # Getters
    @property
    def counter(self):
        return self._counter

    @property
    def current_tract_diameters(self):
        return self.tr.diameter

    @property
    def frequency(self):
        return self.glot.freq

    @property
    def nose_diameters(self):
        return self.tr.nose_diameter

    @property
    def nose_size(self):
        return self.tr.nose_length

    @property
    def tenseness(self):
        return self.glot.tenseness

    @property
    def tract_diameters(self):
        return self.tr.target_diameter

    @property
    def tract_size(self):
        return self.tr.n

    @property
    def velum(self):
        return self.tr.velum_target

    # Setters
    def diameters(self, blade_start: int,
                  lip_start: int,
                  tip_start: int,
                  tongue_index: float,
                  tongue_diameter: float,
                  diameters: List[float]) -> List[float]:  # Was a pointer

        grid_offset = 1.7
        fixed_tongue_diameter = 2 + (tongue_diameter - 2) / 1.5
        tongue_amplitude = (1.5 - fixed_tongue_diameter + grid_offset)

        for i in range(blade_start, lip_start):
            t = 1.1 * math.pi * (tongue_index - i) / (tip_start - blade_start)
            curve = tongue_amplitude * math.cos(t)
            if i == lip_start - 1:
                curve *= 0.8
            if i == blade_start or i == lip_start - 2:
                curve *= 0.94
            diameters[i] = 1.5 - curve

        return diameters

    @frequency.setter
    def frequency(self, f):
        self.glot.freq = f

    def glottis_enable(self):
        pass

    @tenseness.setter
    def tenseness(self, t):
        self.glot.tenseness = t

    def tongue_shape(self, tongue_index: float, tongue_diameter: float) -> None:
        diameters = self.tract_diameters
        self.diameters(10, 39, 32, tongue_index, tongue_diameter, diameters)

    @velum.setter
    def velum(self, v):
        self.tr.velum_target = v
