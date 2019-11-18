import math
from typing import List

import numpy as np

from glottis import Glottis
from tools import sp_data
from tract import Tract

CHUNK = 512

class Voc:

    def __init__(self, sr: float):
        self.glot = Glottis(sr)  # FIXME rename to self.glottis
        self.tr = Tract(sr)  # FIXME rename to self.tract
        self._counter = 0

    def compute(self, randomize: bool = True) -> List[float]:  # C Version returns an int and sets a referenced float *out. This returns *out instead.

        self.glot.update(self.tr.block_time)
        self.tr.reshape()
        self.tr.calculate_reflections()

        buf = []

        for i in range(CHUNK):
            vocal_output = 0
            lambda1 = i / float(CHUNK)
            lambda2 = (i + 0.5) / float(CHUNK)
            glot = self.glot.compute(lambda1, randomize)

            self.tr.compute(glot, lambda1)
            vocal_output += self.tr.lip_output + self.tr.nose_output

            self.tr.compute(glot, lambda2)
            vocal_output += self.tr.lip_output + self.tr.nose_output
            buf.append(vocal_output * 0.125)

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
    def frequency_ptr(self):  # FIXME rename from frequency_ptr to frequency
        return self.glot.freq

    @property
    def nose_diameters(self):
        return self.tr.nose_diameter

    @property
    def nose_size(self):
        return self.tr.nose_length

    @property
    def tenseness_ptr(self):  # FIXME rename from tenseness_ptr to tenseness
        return self.glot.tenseness

    @property
    def tract_diameters(self):
        return self.tr.target_diameter

    @property
    def tract_size(self):
        return self.tr.n

    @property
    def velum_ptr(self):  # FIXME rename from velum_ptr to velum
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

    def frequency(self):
        pass

    def glottis_enable(self):
        pass

    def tenseness(self):
        pass

    def tongue_shape(self, tongue_index: float, tongue_diameter: float) -> None:
        diameters = self.tract_diameters
        self.diameters(10, 39, 32, tongue_index, tongue_diameter, diameters)

    def velum(self):
        pass
