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

@numba.njit()
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
        self.calculate_reflections()
        self.calculate_nose_reflections()
        self.nose_diameter[0] = self.velum_target

    def calculate_diameters(self):
        # TODO Pythonify
        for i in range(self.n):
            diameter = 0
            # calculate diameters until epigottis_start
            if i < (1+self.epiglottis_start) * float(self.n) / BASE_N - 0.5: # BASE_N is 44
                diameter = 0.6
            # calculate diameters from epigottis_start to blade_start
            elif i < self.blade_start * float(self.n) / BASE_N: 
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

    def calculate_reflections(self):
        # TODO refactor rename i
        i: int
        _sum: float

        self.A[:] = self.diameter **2 # /* Calculate area from diameter squared*/

        for i in range(1, self.n):
            self.reflection[i] = self.new_reflection[i]
            if self.A[i] == 0:
                self.new_reflection[i] = 0.999  # /* to prevent bad behavior if 0 */
            else:
                self.new_reflection[i] = (self.A[i - 1] - self.A[i]) / (self.A[i - 1] + self.A[i])

        self.reflection_left = self.new_reflection_left
        self.reflection_right = self.new_reflection_right
        self.reflection_nose = self.new_reflection_nose

        _sum = self.A[self.nose_start] + self.A[self.nose_start + 1] + self.noseA[0]
        self.new_reflection_left = float(2 * self.A[self.nose_start] - _sum) / _sum
        self.new_reflection_right = float(2 * self.A[self.nose_start + 1] - _sum) / _sum
        self.new_reflection_nose = float(2 * self.noseA[0] - _sum) / _sum

    def calculate_nose_reflections(self):
        for i in range(self.nose_length):
            self.noseA[i] = self.nose_diameter[i] * self.nose_diameter[i]

        for i in range(1, self.nose_length):
            self.nose_reflection[i] = (self.noseA[i - 1] - self.noseA[i]) / (self.noseA[i - 1] + self.noseA[i])

    def compute(self, _in: float, lmbd: float) -> None:

        pool = self.tpool
        transients = pool.get_valid_transients()
        for n in transients:
            amp = n.strength * pow(2, -1.0 * n.exponent * n.time_alive)
            self.L[n.position] += amp * 0.5
            self.R[n.position] += amp * 0.5
            n.time_alive += self.T * 0.5
            if n.time_alive > n.lifetime:
                pool.remove(n.id)

        # TODO: junction_outR[0] doesn't get used until _calculate_lip_output. And it is the only place that _in is used.
        #       Perhaps, it could be moved to later and then the first part of the calculation could be parallelized...
        self.junction_outR[0] = self.L[0] * self.glottal_reflection + _in
        self.junction_outL[self.n] = self.R[self.n - 1] * self.lip_reflection

        self._calculate_junctions(lmbd)

        i = self.nose_start
        r = self.new_reflection_left * (1 - lmbd) + self.reflection_left * lmbd
        self.junction_outL[i] = r * self.R[i - 1] + (1 + r) * (self.noseL[0] + self.L[i])
        r = self.new_reflection_right * (1 - lmbd) + self.reflection_right * lmbd
        self.junction_outR[i] = r * self.L[i] + (1 + r) * (self.R[i - 1] + self.noseL[0])
        r = self.new_reflection_nose * (1 - lmbd) + self.reflection_nose * lmbd
        self.nose_junc_outR[0] = r * self.noseL[0] + (1 + r) * (self.L[i] + self.R[i - 1])

        self._calculate_lip_output()

        self.nose_junc_outL[self.nose_length] = self.noseR[self.nose_length - 1] * self.lip_reflection

        self._calculate_nose_junc_out()

        self._calculate_nose()
        self.nose_output = self.noseR[self.nose_length - 1]

    def _calculate_nose(self):
        n = self.nose_length

        # nr = zeros(n)
        # nl = zeros(n)
        # for i in range(self.nose_length):
        #     nr[i] = self.nose_junc_outR[i]
        #     nl[i] = self.nose_junc_outL[i + 1]

        nr_n = self.nose_junc_outR[:n]
        nl_n = self.nose_junc_outL[1:n + 1]

        # np.testing.assert_equal(nr, nr_n)
        # np.testing.assert_equal(nl, nl_n)

        self.noseR[:n] = nr_n
        self.noseL[:n] = nl_n

    def _calculate_nose_junc_out(self):
        n = self.nose_length

        # njoR = zeros(n-1)
        # njoL = zeros(n-1)
        # for i in range(1, self.nose_length):
        #     w = self.nose_reflection[i] * (self.noseR[i - 1] + self.noseL[i])
        #     njoR[i-1] = self.noseR[i - 1] - w
        #     njoL[i-1] = self.noseL[i] + w

        w_n = self.nose_reflection[1:n] * (self.noseR[:n - 1] + self.noseL[1:n])
        njoR_n = self.noseR[:n - 1] - w_n
        njoL_n = self.noseL[1:n] + w_n

        # np.testing.assert_equal(njoR, njoR_n)
        # np.testing.assert_equal(njoL, njoL_n)

        self.nose_junc_outR[1:n] = njoR_n
        self.nose_junc_outL[1:n] = njoL_n

    def _calculate_lip_output(self):
        # r = zeros(self.n)
        # l = zeros(self.n)
        # for i in range(self.n):
        #     r[i] = self.junction_outR[i] * 0.999
        #     l[i] = self.junction_outL[i + 1] * 0.999

        r_n = self.junction_outR[:self.n] * 0.999
        l_n = self.junction_outL[1:self.n + 1] * 0.999

        # np.testing.assert_equal(r, r_n)
        # np.testing.assert_equal(l, l_n)

        self.R[:self.n] = r_n
        self.L[:self.n] = l_n
        self.lip_output = self.R[self.n - 1]

    def _calculate_junctions(self, lmbd):

        # j_outR = zeros(self.n - 1)
        # j_outL = zeros(self.n - 1)
        # for i in range(1, self.n):
        #     r = self.reflection[i] * (1 - lmbd) + self.new_reflection[i] * lmbd
        #     w = r * (self.R[i - 1] + self.L[i])
        #     j_outR[i - 1] = self.R[i - 1] - w
        #     j_outL[i - 1] = self.L[i] + w

        r = self.reflection[1:self.n] * (1 - lmbd) + self.new_reflection[1:self.n] * lmbd
        w = r * (self.R[:self.n - 1] + self.L[1:self.n])
        j_outR_n = self.R[:self.n - 1] - w
        j_outL_n = self.L[1:self.n] + w

        # np.testing.assert_equal(j_outR, j_outR_n)
        # np.testing.assert_equal(j_outL, j_outL_n)

        self.junction_outR[1:self.n] = j_outR_n
        self.junction_outL[1:self.n] = j_outL_n

    def reshape(self) -> None:
        current_obstruction: int = -1
        amount = self.block_time * self.movement_speed

        for i in range(self.n):
            slow_return = 0
            diameter = self.diameter[i]
            target_diameter = self.target_diameter[i]

            if diameter < 0.001: current_obstruction = i

            if i < self.nose_start:
                slow_return = 0.6
            elif i >= self.tip_start:
                slow_return = 1.0
            else:
                slow_return = 0.6 + 0.4 * (i - self.nose_start) / (self.tip_start - self.nose_start)

            self.diameter[i] = move_towards(diameter, target_diameter,
                                                slow_return * amount, 2 * amount)

        if self.last_obstruction > -1 and current_obstruction == -1 and self.noseA[0] < 0.05:
            self.tpool.append(self.last_obstruction)
        self.last_obstruction = current_obstruction

        self.nose_diameter[0] = move_towards(self.nose_diameter[0], self.velum_target,
                                                 amount * 0.25, amount * 0.1)
        self.noseA[0] = self.nose_diameter[0] * self.nose_diameter[0]

    @property
    def lip_start(self):
        return self.__lip_start

    @property
    def blade_start(self):
        '''The end of the epiglottis and the beginning of the tongue

        :return:
        '''
        return self.__blade_start

    @property
    def epiglottis_start(self):
        '''The end of the trachea and the beginning of the epiglottis

        :return:
        '''
        return self.__epiglottis_start

    @property
    def lips(self):
        return np.mean(self.target_diameter[self.lip_start:])

    @lips.setter
    def lips(self, d):
        self.target_diameter[self.lip_start:] = d

    @property
    def epiglottis(self):
        return np.mean(self.target_diameter[self.epiglottis_start:self.blade_start])

    @epiglottis.setter
    def epiglottis(self, d):
        self.target_diameter[self.epiglottis_start:self.blade_start] = d

    @property
    def trachea(self):
        return np.mean(self.target_diameter[:self.epiglottis_start])

    @trachea.setter
    def trachea(self, d):
        self.target_diameter[:self.epiglottis_start] = d
