import random as rnd
from enum import Enum
from math import cos
from typing import List

import numpy as np

from pynkTrombone.two.glottis import Glottis
from pynkTrombone.two.tract import Tract

rnd.seed(a=42)

M_PI = 3.14159265358979323846

EPSILON = 1.0e-38

MAX_TRANSIENTS = 4


class Voc:
    # int sp_voc_init(sp_data *sp, Voc *self)
    def __init__(self, sr: float = 44100):
        self.glottis: Glottis = Glottis(sr)
        self.tract: Tract = Tract(sr)
        self.buf: np.ndarray = zeros(512)  # len = 512
        self._counter: int = 0

    @property
    def frequency(self) -> float:
        return self.glottis.freq

    @frequency.setter
    def frequency(self, f: float):
        self.glottis.freq = f

    # void sp_voc_set_frequency(Voc *self, SPFLOAT freq)
    # {
    #     self->self.freq = freq
    # }
    #
    #
    # SPFLOAT * sp_voc_get_frequency_ptr(Voc *self)
    # {
    #     return &self->self.freq
    # }

    @property
    def tract_diameters(self) -> np.ndarray:
        return self.tract.target_diameter

    # SPFLOAT* sp_voc_get_tract_diameters(Voc *self)
    # {
    #     return self->self.target_diameter
    # }

    @property
    def current_tract_diameters(self) -> np.ndarray:
        return self.tract.diameter

    # SPFLOAT* sp_voc_get_current_tract_diameters(Voc *self)
    # {
    #     return self->self.diameter
    # }

    @property
    def tract_size(self) -> int:
        return self.tract.n

    # int sp_voc_get_tract_size(Voc *self)
    # {
    #     return self->self.n
    # }

    @property
    def nose_diameters(self) -> float:
        return self.tract.nose_diameter

    # SPFLOAT* sp_voc_get_nose_diameters(Voc *self)
    # {
    #     return self->self.nose_diameter
    # }

    @property
    def nose_size(self) -> int:
        return self.tract.nose_length

    # int sp_voc_get_nose_size(Voc *self)
    # {
    #     return self->self.nose_length
    # }

    @property
    def counter(self) -> int:
        return self._counter

    @counter.setter
    def counter(self, i):
        self._counter = i

    # int sp_voc_get_counter(Voc *self)
    # {
    #     return self->counter
    # }

    @property
    def tenseness(self) -> float:
        return self.glottis.tenseness

    @tenseness.setter
    def tenseness(self, t: float):
        self.glottis.tenseness = t

    # SPFLOAT * sp_voc_get_tenseness_ptr(Voc *self)
    # {
    #     return &self->self.tenseness
    # }

    # void sp_voc_set_tenseness(Voc *self, SPFLOAT tenseness)
    # {
    #     self->self.tenseness = tenseness
    # }

    @property
    def velum(self) -> float:
        return self.tract.velum_target

    @velum.setter
    def velum(self, t: float):
        self.tract.velum_target = t

    # void sp_voc_set_velum(Voc *self, SPFLOAT velum)
    # {
    #     self->self.velum_target = velum
    # }
    #
    #
    # SPFLOAT *sp_voc_get_velum_ptr(Voc *self)
    # {
    #     return &self->self.velum_target
    # }

    # int sp_voc_compute(sp_data *sp, Voc *self, SPFLOAT *out)
    def compute(self) -> float:

        if self.counter == 0:
            self.tract.reshape()
            self.tract.calculate_reflections()
            for i in range(512):
                vocal_output = 0
                lmbd1 = float(i) / 512.0
                lmbd2 = float(i + 0.5) / 512.0
                glot = self.glottis.compute(lmbd1)
                # sp, self.self, self = glottis_compute(sp, self.self, lmbd1)

                self.tract.compute(glot, lmbd1)
                # sp, self.self = tract_compute(sp, self.self, glot, lmbd1)
                vocal_output += self.tract.lip_output + self.tract.nose_output

                self.tract.compute(glot, lmbd2)
                # sp, self.self = tract_compute(sp, self.self, glot, lmbd2)
                vocal_output += self.tract.lip_output + self.tract.nose_output
                self.buf[i] = vocal_output * 0.125

        out = self.buf[self.counter]
        self.counter = (self.counter + 1) % 512
        return out

    # void sp_voc_set_diameters(Voc *self,
    #     int blade_start,
    #     int lip_start,
    #     int tip_start,
    #     SPFLOAT tongue_index,
    #     SPFLOAT tongue_diameter,
    #     SPFLOAT *diameters)
    def set_diameters(self,
                      blade_start: int,
                      lip_start: int,
                      tip_start: int,
                      tongue_index: float,
                      tongue_diameter: float) -> None:
        # FIXME: NB Odd, self is not used here That appears to be the case in the original code...

        i: int
        t: float
        fixed_tongue_diameter: float
        curve: float
        grid_offset: int = 0

        # for(i = blade_start i < lip_start i++) {
        for i in range(blade_start, lip_start):
            t = 1.1 * M_PI * float(tongue_index - i) / (tip_start - blade_start)
            fixed_tongue_diameter = 2 + (tongue_diameter - 2) / 1.5
            curve = (1.5 - fixed_tongue_diameter + grid_offset) * cos(t)
            if i == blade_start - 2 or i == lip_start - 1: curve *= 0.8
            if i == blade_start or i == lip_start - 2: curve *= 0.94
            self.tract_diameters[i] = 1.5 - curve

    # void sp_voc_set_tongue_shape(Voc *self,
    #     SPFLOAT tongue_index,
    #     SPFLOAT tongue_diameter)
    def tongue_shape(self,
                     tongue_index: float,
                     tongue_diameter: float) -> None:
        # diameters: List[float]
        # diameters = self.tract_diameters
        # self, diameters = sp_voc_set_diameters(self, 10, 39, 32,
        #                                       tongue_index, tongue_diameter, diameters)
        self.set_diameters(10, 39, 32, tongue_index, tongue_diameter)

    def glottis_enable(self):
        self.glottis.enable = True

    def glottis_disable(self):
        self.glottis.enable = False

    def set_glottis_parameters(self, enable=True, frequency=140):
        # if enable:
        #     self.glottis_enable()
        # else:
        #     self.glottis_disable()

        self.glottis.freq = frequency

    def set_tract_parameters(self, trachea=0.6, epiglottis=1.1, velum=0.01, tongue_index=20, tongue_diameter=2.0, lips=1.5):
        self.tract.trachea = trachea
        self.tract.epiglottis = epiglottis
        self.velum = velum
        self.tongue_shape(tongue_index, tongue_diameter)
        self.tract.lips = lips

    def play_chunk(self) -> np.ndarray:
        """Play until the next control time.

        :return:
        """
        out = [self.compute()]
        while self.counter != 0:
            out.append(self.compute())

        return np.array(out)

# static SPFLOAT move_towards(SPFLOAT current, SPFLOAT target,
#         SPFLOAT amt_up, SPFLOAT amt_down)
def move_towards(current: float, target: float, amt_up: float, amt_down: float) -> float:
    tmp: float
    if current < target:
        tmp = current + amt_up
        return min(tmp, target)

    else:
        tmp = current - amt_down
        return max(tmp, target)

    return 0.0


def zeros(n):
    # return [0.0 for _ in range(n)]
    return np.zeros(shape=(n,))


class Mode(Enum):
    VOC_NONE = 0
    VOC_TONGUE = 1


class voc_demo_d():
    def __init__(self):
        self.sr: float = 44100
        self.voc: Voc = Voc(self.sr)  # Former pointer
        self.tract: float = self.voc.tract_diameters  # Former pointer
        self.freq: float = self.voc.frequency  # Former pointer
        self.velum: float = self.voc.velum  # Former pointer
        self.tenseness: float = self.voc.tenseness  # Former pointer
        self.tract_size: int = self.voc.tract_size
        self.gain: float = 1
        self.mode: int = Mode.VOC_NONE
        self.tongue_pos: float
        self.tongue_diam: float
