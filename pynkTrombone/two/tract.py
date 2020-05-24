import numpy as np

from . import voc
from .transient import TransientPool


class Tract:
    def __init__(self, samplerate: float):
        self.n: int = 44

        self.diameter: np.ndarray = voc.zeros(self.n)  # len = 44
        self.rest_diameter: np.ndarray = voc.zeros(self.n)  # len = 44
        self.target_diameter: np.ndarray = voc.zeros(self.n)  # len = 44
        self.new_diameter: np.ndarray = voc.zeros(self.n)  # len = 44
        self.R: np.ndarray = voc.zeros(self.n)  # len = 44
        self.L: np.ndarray = voc.zeros(self.n)  # len = 44
        self.reflection: np.ndarray = voc.zeros(self.n + 1)  # len = 44
        self.new_reflection: np.ndarray = voc.zeros(self.n + 1)  # len = 44
        self.junction_outL: np.ndarray = voc.zeros(self.n + 1)  # len = 44
        self.junction_outR: np.ndarray = voc.zeros(self.n + 1)  # len = 44
        self.A: np.ndarray = voc.zeros(self.n)  # len = 44

        self.nose_length: int = 28

        self.nose_start: int = 17

        self.tip_start: int = 32
        self.noseL: np.ndarray = voc.zeros(self.nose_length)  # len = 28
        self.noseR: np.ndarray = voc.zeros(self.nose_length)  # len = 28
        self.nose_junc_outL: np.ndarray = voc.zeros(self.nose_length + 1)  # len = 29
        self.nose_junc_outR: np.ndarray = voc.zeros(self.nose_length + 1)  # len = 29
        # FIXME: I don't think self.nose_reflection[0] ever gets touched.
        self.nose_reflection: np.ndarray = voc.zeros(self.nose_length + 1)  # len = 29
        self.nose_diameter: np.ndarray = voc.zeros(self.nose_length)  # len = 28
        self.noseA: np.ndarray = voc.zeros(self.nose_length)  # len = 28

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
    # def tract_init(self: Tract, samplerate: float) -> Tract:

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

            self.diameter[i] = voc.move_towards(diameter, target_diameter,
                                                slow_return * amount, 2 * amount)

        if self.last_obstruction > -1 and current_obstruction == -1 and self.noseA[0] < 0.05:
            self.tpool.append(self.last_obstruction)
        self.last_obstruction = current_obstruction

        self.nose_diameter[0] = voc.move_towards(self.nose_diameter[0], self.velum_target,
                                                 amount * 0.25, amount * 0.1)
        self.noseA[0] = self.nose_diameter[0] * self.nose_diameter[0]
