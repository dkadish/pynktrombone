import math
from typing import List

import numpy as np

from glottis import Glottis
from tools import sp_data
from tract import Tract

CHUNK = 512

class Voc:

    def __init__(self, sr: float):
        self.glottis = Glottis(sr)  # FIXME rename to self.glottis
        self.tract = Tract(sr)  # FIXME rename to self.tract
        self._counter = 0

    def compute(self, randomize: bool = True) -> List[float]:  # C Version returns an int and sets a referenced float *out. This returns *out instead.
        #TODO What if I just compute the next and store in a buffer?
        self.glottis.update(self.tract.block_time)
        self.tract.reshape()
        self.tract.calculate_reflections()
        buf = self._compute(randomize)

        return buf

    def _compute(self, randomize):
        vocal_output = np.zeros(shape=(CHUNK,), dtype=np.float32)
        lambda1 = np.arange(CHUNK, dtype=np.float32) / float(CHUNK)
        lambda2 = (np.arange(CHUNK, dtype=np.float32) + 0.5) / float(CHUNK)
        glot = self.glottis.compute(randomize)

        for i in range(CHUNK):
            self.tract.compute(glot[i], lambda1[i])
            vocal_output[i] += self.tract.lip_output + self.tract.nose_output

            self.tract.compute(glot[i], lambda2[i])
            vocal_output[i] += self.tract.lip_output + self.tract.nose_output

        return vocal_output * 0.125

    def tract_compute(self, sp: sp_data, zin) -> float:
        if self._counter == 0:
            self.tract.reshape()
            self.tract.calculate_reflections()

        vocal_output = 0
        lambda1 = self._counter / float(CHUNK)
        lambda2 = (self._counter + 0.5) / float(CHUNK)

        self.tract.compute(zin, lambda1)
        vocal_output += self.tract.lip_output + self.tract.nose_output
        self.tract.compute(zin, lambda2)
        vocal_output += self.tract.lip_output + self.tract.nose_output

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
        return self.tract.diameter

    @property
    def frequency(self):
        return self.glottis.freq

    @property
    def nose_diameters(self):
        return self.tract.nose_diameter

    @property
    def nose_size(self):
        return self.tract.nose_length

    @property
    def tenseness(self):
        return self.glottis.tenseness

    @property
    def tract_diameters(self):
        return self.tract.target_diameter

    @property
    def tract_size(self):
        return self.tract.n

    @property
    def velum(self):
        return self.tract.velum_target

    # Setters
    #TODO Currently, trachea, epiglottis, lips are fixed. They don't have to be.
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
        if f < 100:
            raise ValueError('Frequency must be above 100 Hz.')
        self.glottis.freq = f

    def glottis_enable(self):
        self.glottis.enable = True

    def glottis_disable(self):
        self.glottis.enable = False

    @tenseness.setter
    def tenseness(self, t):
        '''

        :param t: Must be in the range of [0,1]. Good values between [0.6, 0.9]
        :return:
        '''
        self.glottis.tenseness = t

    def tongue_shape(self, tongue_index: float, tongue_diameter: float) -> None:
        '''

        :param tongue_index: Where on the diameter index curve (VOC p25) the tongue is pointed. Should be on [blade_start,tip_start], which by default is [10, 32]
        :param tongue_diameter: Should be between [2.0, 3.5].
        :return: None
        '''
        diameters = self.tract_diameters
        self.diameters(10, 39, 32, tongue_index, tongue_diameter, diameters)

    @velum.setter
    def velum(self, v):
        '''

        :param v: Defaults to 0.01. Nasally sounds at 0.04. Try between [0.0, 0.05]
        :return:
        '''
        self.tract.velum_target = v
