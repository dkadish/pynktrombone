import numpy as np
np.seterr(divide='warn', over='warn', invalid='warn')

from tools import move_towards
from transient import MAX_TRANSIENTS, TransientPool, Transient

zeros = lambda n: np.zeros(n, dtype=np.float32)

N = 44
NOSE_LENGTH = 28
NOSE_START = 17

CHUNK = 512

class Tract:

    def __init__(self, samplerate):
        # FIXME unused
        # self.fade

        # Setup arrays
        self._diameter = zeros(N)
        # self.rest_diameter = zeros(N)
        self._target_diameter = zeros(N)
        self.R = zeros(N)
        self.L = zeros(N)
        self.reflection = zeros(N+1)
        self.new_reflection = zeros(N+1)
        self.junction_outL = zeros(N+1)
        self.junction_outR = zeros(N+1)
        self.A = zeros(N)

        self.noseL = zeros(NOSE_LENGTH)
        self.noseR = zeros(NOSE_LENGTH)
        self.nose_junc_outL = zeros(NOSE_LENGTH+1)
        self.nose_junc_outR = zeros(NOSE_LENGTH+1)
        self.nose_reflection = zeros(NOSE_LENGTH+1)
        self.nose_diameter = zeros(NOSE_LENGTH)
        self.noseA = zeros(NOSE_LENGTH)

        # Default parameters
        self.reflection_left = 0.0
        self.reflection_right = 0.0
        self.reflection_nose = 0.0
        self.new_reflection_left = 0.0
        self.new_reflection_right = 0.0
        self.new_reflection_nose = 0.0
        self.velum_target = 0.01
        self.glottal_reflection = 0.75
        self.lip_reflection = -0.85
        self.last_obstruction = -1
        self.movement_speed = 10 #15
        self.lip_output = 0
        self.nose_output = 0
        self.tip_start = 32

        # FIXME Can be redone with array operations.
        # Set up the vocal tract diameters
        for i in range(N):
            if i < (7.0 * N / 44.0 - 0.5):
                diameter = 0.6
            elif i < (12.0 * N / 44.0):
                diameter = 1.1
            else:
                diameter = 1.5

            # self._diameter[i] = self.rest_diameter[i] = self._target_diameter[i] = diameter
            self._diameter[i] = self._target_diameter[i] = diameter

        # Set up the nasal passage parameters
        for i in range(NOSE_LENGTH):
            d = 2.0 * (i / NOSE_LENGTH)
            if d < 1:
                diameter = 0.4 + 1.6 * d
            else:
                diameter = 0.5 + 1.5 * (2 - d)

            diameter = min(diameter, 1.9)
            self.nose_diameter[i] = diameter

        self.calculate_reflections()
        self.calculate_nose_reflections()

        self.nose_diameter[0] = self.velum_target

        self.block_time = 512.0 / samplerate
        self.T = 1.0 / samplerate

        self.tpool = TransientPool()
        self.tpool.size = 0
        self.tpool.next_free = 0
        for i in range(MAX_TRANSIENTS):
            tr = Transient(position=0, id=i, next=None)
            self.tpool.pool.append(tr)
            self.tpool.pool[i].is_free = 1
            self.tpool.pool[i].time_alive = 0
            self.tpool.pool[i].strength = 0
            self.tpool.pool[i].exponent = 0

    def _calc_scattering_junctions(self, excitation, xfade_coeff):
        # Calculate Scattering Junctions
        self.junction_outR[0] = self.L[0] * self.glottal_reflection + excitation
        self.junction_outL[N] = self.R[N - 1] * self.lip_reflection

        r = self.reflection[1:N] * (1 - xfade_coeff) + self.new_reflection[1:N] * xfade_coeff
        w = r * (self.R[:-1] + self.L[1:])
        self.junction_outR[1:N] = self.R[:-1] - w
        self.junction_outL[1:N] = self.L[1:] + w

    def _calc_scattering_for_nose(self, xfade_coeff):
        # Calculate Scattering for Nose
        i = NOSE_START
        r = self.new_reflection_left * (1.0 - xfade_coeff) + self.reflection_left * xfade_coeff
        self.junction_outL[i] = r * self.R[i - 1] + (1.0 + r) * (self.noseL[0] + self.L[i])
        r = self.new_reflection_right * (1.0 - xfade_coeff) + self.reflection_right * xfade_coeff
        self.junction_outR[i] = r * self.L[i] + (1.0 + r) * (self.R[i - 1] + self.noseL[0])
        r = self.new_reflection_nose * (1.0 - xfade_coeff) + self.reflection_nose * xfade_coeff
        # try:
        self.nose_junc_outR[0] = r * self.noseL[0] + (1.0 + r) * (self.L[i] + self.R[i - 1])
        # except FloatingPointError as e:
        #     print('tract.py:120', e, r, self.noseL[0], (1.0 + r), self.L[i], self.R[i - 1])

    def _update_delay_lines_and_lip(self):
        # Update Left/Right delay lines and set lip output
        self.R[:N] = self.junction_outR[:N] * 0.999
        self.L[:N] = self.junction_outL[1:N + 1] * 0.999

        self.lip_output = self.R[N - 1]

    def _calc_nose_scattering_jcts(self):
        # Calculate Nose Scattering Junctions
        self.nose_junc_outL[NOSE_LENGTH] = self.noseR[NOSE_LENGTH - 1] * self.lip_reflection

        w = self.nose_reflection[1:NOSE_LENGTH] * (self.noseR[:NOSE_LENGTH - 1] + self.noseL[1:NOSE_LENGTH])
        self.nose_junc_outR[1:NOSE_LENGTH] = self.noseR[:NOSE_LENGTH - 1] - w
        self.nose_junc_outL[1:NOSE_LENGTH] = self.noseL[1:NOSE_LENGTH] + w

    def _update_nose_delay_and_output(self):
        # Update Nose Left/Right delay lines and set nose output
        self.noseR[:NOSE_LENGTH] = self.nose_junc_outR[:NOSE_LENGTH]
        self.noseL[:NOSE_LENGTH] = self.nose_junc_outL[1:NOSE_LENGTH + 1]

        self.nose_output = self.noseR[NOSE_LENGTH - 1]

    def compute(self, excitation: float, xfade_coeff: float) -> None:  # in > excitation, lambda > xfade_coeff
        # Process Transients
        pool = self.tpool
        current_size = pool.size
        n = pool.root
        for i in range(current_size):
            amp = n.strength * pow(2.0, -1.0 * n.exponent * n.time_alive)
            self.L[n.position] += amp * 0.5
            self.R[n.position] += amp * 0.5
            n.time_alive += self.T * 0.5
            if n.time_alive > n.lifetime:
                pool.remove(n.id)
            n = n.next

        self._calc_scattering_junctions(excitation, xfade_coeff)
        self._calc_scattering_for_nose(xfade_coeff)
        self._update_delay_lines_and_lip()
        self._calc_nose_scattering_jcts()
        self._update_nose_delay_and_output()

    def calculate_nose_reflections(self):
        # noseA = self.noseA.copy()
        # noseReflection = self.nose_reflection.copy()
        #
        # for i in range(NOSE_LENGTH):
        #     noseA[i] = self.nose_diameter[i] * self.nose_diameter[i]
        #
        # for i in range(1, NOSE_LENGTH):
        #     noseReflection[i] = (noseA[i-1] - noseA[i])/(noseA[i-1] + noseA[i])

        self.noseA[:NOSE_LENGTH] = np.power(self.nose_diameter[:NOSE_LENGTH], 2)

        self.nose_reflection[1:NOSE_LENGTH] = (self.noseA[:NOSE_LENGTH-1] - self.noseA[1:NOSE_LENGTH]) / (self.noseA[:NOSE_LENGTH-1] + self.noseA[1:NOSE_LENGTH])

        # np.testing.assert_array_equal(noseA, self.noseA)
        # np.testing.assert_array_equal(noseReflection, self.nose_reflection)

    def calculate_reflections(self):
        # self.A[i] = self.diameter[i] * self.diameter[i]
        self.A = np.power(self._diameter, 2)

        self.reflection[1:N] = self.new_reflection[1:N]
        self.new_reflection[1:N][self.A[1:N] == 0] = 0.999
        non_zero = (self.A[:N-1] - self.A[1:N])/(self.A[:N-1] + self.A[1:N])
        self.new_reflection[1:N][self.A[1:N] != 0] = non_zero[self.A[1:N] != 0]

        self.reflection_left = self.new_reflection_left
        self.reflection_right = self.new_reflection_right
        self.reflection_nose = self.new_reflection_nose

        total = self.A[NOSE_START] + self.A[NOSE_START + 1] + self.noseA[0]
        self.new_reflection_left = (2.0 * self.A[NOSE_START] - total) / total
        self.new_reflection_right = (2.0 * self.A[NOSE_START + 1] - total) / total
        self.new_reflection_nose = (2.0 * self.noseA[0] - total) / total

    def reshape(self):
        current_obstruction = -1
        amount = self.block_time * self.movement_speed
        # print('Before: ', self.diameter, self.diameter - self.target_diameter)
        for i in range(N):
            diameter = self._diameter[i]
            target_diameter = self._target_diameter[i]

            if diameter < 0.001: current_obstruction = i

            if i < NOSE_START:
                slow_return = 0.6
            elif i >= self.tip_start:
                slow_return = 1.0
            else:
                slow_return = 0.6 + 0.4 * (i - NOSE_START) / (self.tip_start - NOSE_START)

            self._diameter[i] = move_towards(diameter, target_diameter, slow_return * amount, 2 * amount)

        # print('After: ', self.diameter)
        if self.last_obstruction > -1 and current_obstruction == -1 and self.noseA[0] < 0.05:
            self.tpool.append(self.last_obstruction)

        self.last_obstruction = current_obstruction

        self.nose_diameter[0] = move_towards(self.nose_diameter[0], self.velum_target,
                                             amount * 0.25, amount * 0.1)
        self.noseA[0] = np.power(self.nose_diameter[0], 2)

    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self, d):
        self._diameter = d

    @property
    def target_diameter(self):
        return self._target_diameter

    @target_diameter.setter
    def target_diameter(self, d):
        self._target_diameter = d

    @property
    def lip_start(self):
        return 39

    @property
    def blade_start(self):
        '''The end of the epiglottis and the beginning of the tongue

        :return:
        '''
        return 12 #10

    @property
    def epiglottis_start(self):
        '''The end of the trachea and the beginning of the epiglottis

        :return:
        '''
        return 6

    @property
    def lips(self):
        return np.average(self._target_diameter[self.lip_start:])

    @lips.setter
    def lips(self, d):
        self._target_diameter[self.lip_start:] = d

    @property
    def epiglottis(self):
        return np.average(self._target_diameter[self.epiglottis_start:self.lip_start])

    @epiglottis.setter
    def epiglottis(self, d):
        self._target_diameter[self.epiglottis_start:self.lip_start] = d

    @property
    def trachea(self):
        return np.average(self._target_diameter[:self.epiglottis_start])

    @trachea.setter
    def trachea(self, d):
        self._target_diameter[:self.epiglottis_start] = d