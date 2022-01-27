import numpy as np
import numba 
from numba import int32, int64, float32, float64
from numba.experimental import jitclass
from collections import OrderedDict

from .transient import TransientPool
from .consts import BASE_N

@numba.njit()
def move_towards(current: float, target: float, amt_up: float, amt_down: float) -> float:
    tmp: float
    if current < target:
        tmp = current + amt_up
        return min(tmp, target)

    else:
        tmp = current - amt_down
        return max(tmp, target)

@numba.jit()
def zeros(n: int) -> np.ndarray:
    return np.zeros((n,),dtype=np.float64)

spec = [
    ("sr",float64),
    ("n",int32),
    ("__blade_start",int32),
    ("__lip_start",int32),
    ("__epiglottis_start",int32),

    ("diameter",float64[:]),
    ("rest_diameter",float64[:]),
    ("target_diameter",float64[:]),
    ("new_diameter",float64[:]),
    ("R",float64[:]),
    ("L",float64[:]),
    ("reflection",float64[:]),
    ("new_reflection",float64[:]),
    ("junction_outL",float64[:]),
    ("junction_outR",float64[:]),
    ("A",float64[:]),

    ("nose_length", int32),
    ("nose_start", int32),
    ("tip_start", int32),
    ("noseL",float64[:]),
    ("noseR",float64[:]),
    ("nose_junc_outL",float64[:]),
    ("nose_junc_outR",float64[:]),
    ("nose_reflection",float64[:]),
    ("nose_diameter",float64[:]),
    ("noseA",float64[:]),

    ("reflection_left",float64),
    ("reflection_right",float64),
    ("reflection_nose",float64),

    ("new_reflection_left",float64),
    ("new_reflection_right",float64),
    ("new_reflection_nose",float64),

    ("velum_target",float64),

    ("glottal_reflection",float64),
    ("lip_reflection",float64),
    ("last_obstruction",int32),
    ("fade",float64),
    ("movement_speed",float64),
    ("lip_output",float64),
    ("nose_output",float64),
    ("block_time",float64),

    ("tpool",TransientPool.class_type.instance_type),
    ("T",float64)
]


@jitclass(spec)
class Tract:
    """ Human tract model
    
    """
    def __init__(
        self, samplerate: float, n:int = 44, nose_length:int = 28,
        nose_start:int = 17, tip_start:int = 32, blade_start=10,
        epiglottis_start:int = 6, lip_start:int = 39,
        ) -> None:
        self.sr:float = samplerate
        self.n = n
        self.__blade_start = blade_start
        self.__lip_start = lip_start
        self.__epiglottis_start = epiglottis_start

        n_z = zeros(n)
        np1_z = zeros(n+1)

        self.diameter: np.ndarray = n_z.copy()  
        self.rest_diameter: np.ndarray = n_z.copy()  
        self.target_diameter: np.ndarray = n_z.copy()  
        self.new_diameter: np.ndarray = n_z.copy()  
        self.R: np.ndarray = n_z.copy()  
        self.L: np.ndarray = n_z.copy()  
        self.reflection: np.ndarray = np1_z.copy()  
        self.new_reflection: np.ndarray = np1_z.copy()  
        self.junction_outL: np.ndarray = np1_z.copy()  
        self.junction_outR: np.ndarray = np1_z.copy()  
        self.A: np.ndarray = n_z.copy()  

        self.nose_length = nose_length
        self.nose_start = nose_start
        self.tip_start = tip_start

        noze_z = zeros(nose_length)
        nozep1_z = zeros(nose_length+1)
        self.noseL: np.ndarray = noze_z.copy()  
        self.noseR: np.ndarray = noze_z.copy()  
        self.nose_junc_outL: np.ndarray = nozep1_z.copy()  
        self.nose_junc_outR: np.ndarray = nozep1_z.copy()  
        # FIXME: I don't think self.nose_reflection[0] ever gets touched.
        self.nose_reflection: np.ndarray = nozep1_z.copy()  
        self.nose_diameter: np.ndarray = noze_z.copy()  
        self.noseA: np.ndarray = noze_z.copy()  

        self.reflection_left: float = 0.0
        self.reflection_right: float = 0.0
        self.reflection_nose: float = 0.0

        self.new_reflection_left: float = 0.0
        self.new_reflection_right: float = 0.0
        self.new_reflection_nose: float = 0.0

        self.velum_target: float = 0.01

        self.glottal_reflection: float = 0.75
        self.lip_reflection: float = -0.85
        self.last_obstruction: int = -1
        self.fade: float
        self.movement_speed: float = 15
        self.lip_output: float = 0
        self.nose_output: float = 0
        self.block_time: float = 512.0 / samplerate

        self.tpool: TransientPool = TransientPool()
        self.T: float = 1.0 / samplerate

        self.calculate_diameters()
        self.calculate_nose_diameter()
    
    def calculate_diameters(self):
        # TODO Pythonify
        for i in range(self.n):
            diameter = 0
            if i < 7 * float(self.n) / BASE_N - 0.5: # BASE_N is 44
                diameter = 0.6
            elif i < 12 * float(self.n) / BASE_N: 
                diameter = 1.1
            else:
                diameter = 1.5

            self.diameter[i] = self.rest_diameter[i] = self.target_diameter[i] = self.new_diameter[i] = diameter
    
    def calculate_nose_diameter(self):
        # TODO Pythonify
        for i in range(self.nose_length):
            d = 2 * (float(i) / self.nose_length)
            if d < 1:
                diameter = 0.4 + 1.6 * d
            else:
                diameter = 0.5 + 1.5 * (2 - d)

            diameter = min(diameter, 1.9)
            self.nose_diameter[i] = diameter