import random as rnd
from enum import Enum
from math import cos
from typing import *
import numpy as np
import numba
from numba.experimental import jitclass
from numba import int32, float64, int64

from .glottis import Glottis
from .tract import Tract,zeros
from .consts import M_PI

rnd.seed(a=42)

spec = [
    ("glottis",Glottis.class_type.instance_type),
    ("tract", Tract.class_type.instance_type),
    ("buf",float64[:]),
    ("sr",float64),
    ("CHUNK",int32),
    ("vocal_output_scaler",float64),
    ("__counter",int64),
]
@jitclass(spec)
class Voc:
    __doc__ = """Human vocal organ model
    Generate voice by controlling Glottis and Tract.
    GlottisとTractを制御することによって声を生成します。
    
    {0}

    {1}
    """.format(Glottis.__doc__,Tract.__doc__)

    def __init__(
        self, samplerate:float = 44100, CHUNK:int = 512, vocal_output_scaler:float = 0.125,

        default_freq:float=400, default_tenseness:float = 0.6,

        n:int = 44, nose_length:int = 28,
        nose_start:int = 17, tip_start:int = 32, blade_start:int =12,
        epiglottis_start:int = 6, lip_start:int = 39,
    ) -> None:
        self.glottis: Glottis = Glottis(samplerate,default_freq,default_tenseness)
        self.tract: Tract = Tract(
            samplerate,n, nose_length,
            nose_start,tip_start, blade_start,
            epiglottis_start, lip_start,
        )
        self.buf: np.ndarray = zeros(CHUNK)
        self.CHUNK:int = CHUNK
        self.sr:float = samplerate
        self.vocal_output_scaler:float = vocal_output_scaler
        self.__counter = 0

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

    def step(self) -> np.ndarray:
        """
        Generate a wave for 1 CHUNK. 
        1CHUNKの分の波形を生成します。
        """
        self.tract.reshape()
        self.tract.calculate_reflections()
        for i in range(self.CHUNK):
            vocal_output = 0
            lmbd1 = float(i) / float(self.CHUNK) 
            lmbd2 = float(i + 0.5) / float(self.CHUNK) 
            glot = self.glottis.compute(lmbd1)
            # sp, self.self, self = glottis_compute(sp, self.self, lmbd1)

            self.tract.compute(glot, lmbd1)
            # sp, self.self = tract_compute(sp, self.self, glot, lmbd1)
            vocal_output += self.tract.lip_output + self.tract.nose_output

            self.tract.compute(glot, lmbd2)
            # sp, self.self = tract_compute(sp, self.self, glot, lmbd2)
            vocal_output += self.tract.lip_output + self.tract.nose_output
            self.buf[i] = vocal_output * self.vocal_output_scaler
        
        return self.buf

    def compute(self) -> float:

        if self.counter == 0:
            self.step()
        out = self.buf[self.counter]
        self.counter = (self.counter + 1) % self.CHUNK
        return out

    def set_diameters(self,
                      blade_start: int,
                      lip_start: int,
                      tip_start: int,
                      tongue_index: float,
                      tongue_diameter: float) -> None:

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

    def tongue_shape(self,
                     tongue_index: float,
                     tongue_diameter: float) -> None:
        # diameters: List[float]
        # diameters = self.tract_diameters
        # self, diameters = sp_voc_set_diameters(self, 10, 39, 32,
        #                                       tongue_index, tongue_diameter, diameters)
        self.set_diameters(
            self.tract.blade_start, self.tract.lip_start, self.tract.tip_start,
            tongue_index, tongue_diameter
        )

    def set_tract_parameters(self, trachea=0.6, epiglottis=1.1, velum=0.01, tongue_index=20, tongue_diameter=2.0, lips=1.5):
        self.tract.trachea = trachea
        self.tract.epiglottis = epiglottis
        self.velum = velum
        self.tongue_shape(tongue_index, tongue_diameter)
        self.tract.lips = lips


    @property
    def counter(self) -> int:
        return self.__counter

    @counter.setter
    def counter(self, i):
        self.__counter = i

    def play_chunk(self):
        return self.step()

class Mode(Enum):
    VOC_NONE = 0
    VOC_TONGUE = 1

class voc_demo_d():
    def __init__(self):
        self.sr: float = 44100
        self.chunk = 512
        self.voc: Voc = Voc(self.sr,self.chunk)  # Former pointer
        self.tract: float = self.voc.tract_diameters  # Former pointer
        self.freq: float = self.voc.frequency  # Former pointer
        self.velum: float = self.voc.velum  # Former pointer
        self.tenseness: float = self.voc.tenseness  # Former pointer
        self.tract_size: int = self.voc.tract_size
        self.gain: float = 1
        self.mode: int = Mode.VOC_NONE
        self.tongue_pos: float
        self.tongue_diam: float

