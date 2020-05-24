import random as rnd
from enum import Enum
from itertools import count
from math import exp, sin, log, sqrt, cos
from random import random
from typing import List

import numpy as np

rnd.seed(a=42)

M_PI = 3.14159265358979323846

EPSILON = 1.0e-38

MAX_TRANSIENTS = 4


class Glottis:

    def __init__(self, sr: float):
        self.freq: float = 140  # 140Hz frequency by default
        self.tenseness: float = 0.6  # value between 0 and 1
        self.Rd: float
        self.waveform_length: float
        self.time_in_waveform: float = 0
        self.alpha: float
        self.E0: float
        self.epsilon: float
        self.shift: float
        self.delta: float
        self.Te: float
        self.omega: float
        self.T: float = 1.0 / sr  # big T

        self.setup_waveform(0)

    # static void glottis_init(Glottis *self, SPFLOAT sr)
    # CHANGE: self is not a pointer, is returned from fn
    # def glottis_init(self: Glottis, sr: float) -> Glottis:

    # static void glottis_setup_waveform(Glottis *self, SPFLOAT lmbd)
    # CHANGE: self is not a pointer, is returned from fn
    def setup_waveform(self, lmbd: float):
        Rd: float
        Ra: float
        Rk: float
        Rg: float

        Ta: float
        Tp: float
        Te: float

        epsilon: float
        shift: float
        delta: float
        rhs_integral: float

        lower_integral: float
        upper_integral: float

        omega: float
        s: float
        y: float
        z: float

        alpha: float
        E0: float

        self.Rd = 3 * (1 - self.tenseness)
        self.waveform_length = 1.0 / self.freq

        Rd = self.Rd
        if (Rd < 0.5): Rd = 0.5
        if (Rd > 2.7): Rd = 2.7

        Ra = -0.01 + 0.048 * Rd
        Rk = 0.224 + 0.118 * Rd
        Rg = (Rk / 4) * (0.5 + 1.2 * Rk) / (0.11 * Rd - Ra * (0.5 + 1.2 * Rk))

        Ta = Ra
        Tp = float(1.0 / (2 * Rg))
        Te = Tp + Tp * Rk

        epsilon = float(1.0 / Ta)
        shift = exp(-epsilon * (1 - Te))
        delta = 1 - shift

        rhs_integral = float((1.0 / epsilon) * (shift - 1) + (1 - Te) * shift)
        rhs_integral = rhs_integral / delta
        lower_integral = - (Te - Tp) / 2 + rhs_integral
        upper_integral = -lower_integral

        omega = M_PI / Tp
        s = sin(omega * Te)

        y = -M_PI * s * upper_integral / (Tp * 2)
        z = log(y)
        alpha = z / (Tp / 2 - Te)
        E0 = -1 / (s * exp(alpha * Te))

        self.alpha = alpha
        self.E0 = E0
        self.epsilon = epsilon
        self.shift = shift
        self.delta = delta
        self.Te = Te
        self.omega = omega

    # static SPFLOAT glottis_compute(sp_data *sp, Glottis *self, SPFLOAT lmbd)
    # CHANGE: sp is not a pointer, is returned from fn
    # CHANGE: self is not a pointer, is returned from fn
    def compute(self, lmbd: float) -> float:
        intensity: float = 1.0

        self.time_in_waveform += self.T

        if self.time_in_waveform > self.waveform_length:
            self.time_in_waveform -= self.waveform_length
            self.setup_waveform(lmbd)

        t = (self.time_in_waveform / self.waveform_length)

        if t > self.Te:
            out = (-exp(-self.epsilon * (t - self.Te)) + self.shift) / self.delta
        else:
            out = self.E0 * exp(self.alpha * t) * sin(self.omega * t)

        noise = 2.0 * random() - 1

        aspiration = intensity * (1 - sqrt(self.tenseness)) * 0.3 * noise

        aspiration *= 0.2

        out += aspiration

        return out


class Transient:
    ids = count(0)

    def __init__(self):
        self.position: int = 0
        self.time_alive: float = 0
        self.lifetime: float
        self.strength: float = 0
        self.exponent: float = 0
        self.is_free: str = 1
        self.id: int = next(self.ids)
        self.next: Transient  # PTR

    @classmethod
    def reset_count(self):
        self.ids = count(0)


class TransientPool:
    def __init__(self):
        Transient.reset_count()

        self.pool: List[Transient] = []  # Should be limited to MAX_TRANSIENTS
        self.root: Transient = None  # PTR
        self.size: int = 0
        self.next_free: int = 0

        for i in range(MAX_TRANSIENTS):
            self.pool.append(Transient())

    # static int append_transient(TransientPool *self, int position)
    # CHANGE: self is not a pointer, is returned from fn
    def append(self, position: int) -> None:
        i: int
        free_id: int
        # Transient *t
        t: Transient

        free_id = self.next_free
        if self.size == MAX_TRANSIENTS:
            return

        if free_id == -1:
            for i in range(MAX_TRANSIENTS):
                if self.pool[i].is_free:
                    free_id = i
                    break

        if free_id == -1:
            return

        t = self.pool[free_id]
        t.next = self.root
        self.root = t
        self.size += 1
        t.is_free = 0
        t.time_alive = 0
        t.lifetime = 0.2
        t.strength = 0.3
        t.exponent = 200
        t.position = position
        self.next_free = -1

    # static void remove_transient(TransientPool *self, unsigned int id)
    # CHANGE: self is not a pointer, is returned from fn
    def remove(self, id: int) -> None:
        i: int
        n: Transient

        self.next_free = id
        n = self.root
        if id == n.id:
            self.root = n.next
            self.size -= 1
            return

        for i in range(self.size):
            if n.next.id == id:
                self.size -= 1
                n.next.is_free = 1
                n.next = n.next.next
                break
            n = n.next


class Tract:
    def __init__(self, sr: float):
        self.n: int = 44

        self.diameter: List[float] = zeros(self.n)  # len = 44
        self.rest_diameter: List[float] = zeros(self.n)  # len = 44
        self.target_diameter: List[float] = zeros(self.n)  # len = 44
        self.new_diameter: List[float] = zeros(self.n)  # len = 44
        self.R: List[float] = zeros(self.n)  # len = 44
        self.L: List[float] = zeros(self.n)  # len = 44
        self.reflection: List[float] = zeros(self.n + 1)  # len = 44
        self.new_reflection: List[float] = zeros(self.n + 1)  # len = 44
        self.junction_outL: List[float] = zeros(self.n + 1)  # len = 44
        self.junction_outR: List[float] = zeros(self.n + 1)  # len = 44
        self.A: List[float] = zeros(self.n)  # len = 44

        self.nose_length: int = 28

        self.nose_start: int = 17

        self.tip_start: int = 32
        self.noseL: List[float] = zeros(self.nose_length)  # len = 28
        self.noseR: List[float] = zeros(self.nose_length)  # len = 28
        self.nose_junc_outL: List[float] = zeros(self.nose_length + 1)  # len = 29
        self.nose_junc_outR: List[float] = zeros(self.nose_length + 1)  # len = 29
        # FIXME: I don't think self.nose_reflection[0] ever gets touched.
        self.nose_reflection: List[float] = zeros(self.nose_length + 1)  # len = 29
        self.nose_diameter: List[float] = zeros(self.nose_length)  # len = 28
        self.noseA: List[float] = zeros(self.nose_length)  # len = 28

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
        self.block_time: float = 512.0 / sr

        self.tpool: TransientPool = TransientPool()
        self.T: float = 1.0 / sr

        # TODO Pythonify
        for i in range(self.n):
            diameter = 0
            if i < 7 * float(self.n) / 44 - 0.5:
                diameter = 0.6
            elif i < 12 * float(self.n) / 44:
                diameter = 1.1
            else:
                diameter = 1.5

            self.diameter[i] = self.rest_diameter[i] = self.target_diameter[i] = self.new_diameter[i] = diameter

        # TODO Pythonify
        for i in range(self.nose_length):
            d = 2 * (float(i) / self.nose_length)
            if d < 1:
                diameter = 0.4 + 1.6 * d
            else:
                diameter = 0.5 + 1.5 * (2 - d)

            diameter = min(diameter, 1.9)
            self.nose_diameter[i] = diameter

        # TODO Pythonify. This *SHOULD* work right now, but it's dumb
        self.calculate_reflections()
        self.calculate_nose_reflections()
        self.nose_diameter[0] = self.velum_target

    # static void tract_init(sp_data *sp, Tract *self)
    # CHANGE: sp is not a pointer, is returned from fn. | sp is not changed. no longer returned.
    # CHANGE: self is not a pointer, is returned from fn
    # def tract_init(self: Tract, sr: float) -> Tract:

    # static void tract_calculate_reflections(Tract *self)
    # CHANGE: self is not a pointer, is returned from fn
    def calculate_reflections(self):
        # TODO refactor rename i
        i: int
        _sum: float

        for i in range(self.n):
            self.A[i] = self.diameter[i] * self.diameter[i]
            # /* Calculate area from diameter squared*/

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

    # static void tract_calculate_nose_reflections(Tract *self)
    # CHANGE: self is not a pointer, is returned from fn
    def calculate_nose_reflections(self):
        # TODO refactor rename i
        i: int

        for i in range(self.nose_length):
            self.noseA[i] = self.nose_diameter[i] * self.nose_diameter[i]

        for i in range(1, self.nose_length):
            self.nose_reflection[i] = (self.noseA[i - 1] - self.noseA[i]) / (self.noseA[i - 1] + self.noseA[i])

    # static void tract_compute(sp_data *sp, Tract *self,
    #     SPFLOAT  in,
    #     SPFLOAT  lmbd)
    def compute(self, _in: float, lmbd: float) -> None:

        pool = self.tpool  # Python treats this as a reference, so this should be fine.
        current_size = pool.size
        n = pool.root
        for i in range(current_size):
            amp = n.strength * pow(2, -1.0 * n.exponent * n.time_alive)
            self.L[n.position] += amp * 0.5
            self.R[n.position] += amp * 0.5
            n.time_alive += self.T * 0.5
            if n.time_alive > n.lifetime:
                pool.remove(n.id)
            n = n.next

        #TODO: junction_outR[0] doesn't get used until _calculate_lip_output. And it is the only place that _in is used.
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

    # static void tract_reshape(Tract *self)
    # CHANGE: self is not a pointer, is returned from fn
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


class Voc:
    # int sp_voc_init(sp_data *sp, Voc *self)
    def __init__(self, sr: float = 44100):
        self.glot: Glottis = Glottis(sr)
        self.tr: Tract = Tract(sr)
        self.buf: List[float] = zeros(512)  # len = 512
        self._counter: int = 0

    @property
    def frequency(self) -> float:
        return self.glot.freq

    @frequency.setter
    def frequency(self, f: float):
        self.glot.freq = f

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
    def tract_diameters(self) -> float:
        return self.tr.target_diameter

    # SPFLOAT* sp_voc_get_tract_diameters(Voc *self)
    # {
    #     return self->self.target_diameter
    # }

    @property
    def current_tract_diameters(self) -> float:
        return self.tr.diameter

    # SPFLOAT* sp_voc_get_current_tract_diameters(Voc *self)
    # {
    #     return self->self.diameter
    # }

    @property
    def tract_size(self) -> int:
        return self.tr.n

    # int sp_voc_get_tract_size(Voc *self)
    # {
    #     return self->self.n
    # }

    @property
    def nose_diameters(self) -> float:
        return self.tr.nose_diameter

    # SPFLOAT* sp_voc_get_nose_diameters(Voc *self)
    # {
    #     return self->self.nose_diameter
    # }

    @property
    def nose_size(self) -> int:
        return self.tr.nose_length

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
        return self.glot.tenseness

    @tenseness.setter
    def tenseness(self, t: float):
        self.glot.tenseness = t

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
        return self.tr.velum_target

    @velum.setter
    def velum(self, t: float):
        self.tr.velum_target = t

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
    def compute(self, out: float) -> float:

        if self.counter == 0:
            self.tr.reshape()
            self.tr.calculate_reflections()
            for i in range(512):
                vocal_output = 0
                lmbd1 = float(i) / 512.0
                lmbd2 = float(i + 0.5) / 512.0
                glot = self.glot.compute(lmbd1)
                # sp, self.self, self = glottis_compute(sp, self.self, lmbd1)

                self.tr.compute(glot, lmbd1)
                # sp, self.self = tract_compute(sp, self.self, glot, lmbd1)
                vocal_output += self.tr.lip_output + self.tr.nose_output

                self.tr.compute(glot, lmbd2)
                # sp, self.self = tract_compute(sp, self.self, glot, lmbd2)
                vocal_output += self.tr.lip_output + self.tr.nose_output
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
    def set_tongue_shape(self,
                         tongue_index: float,
                         tongue_diameter: float) -> None:
        # diameters: List[float]
        # diameters = self.tract_diameters
        # self, diameters = sp_voc_set_diameters(self, 10, 39, 32,
        #                                       tongue_index, tongue_diameter, diameters)
        self.set_diameters(10, 39, 32, tongue_index, tongue_diameter)


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
